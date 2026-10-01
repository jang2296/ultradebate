"""Extract MLP training features from ultradebate operational logs.

Output: CSV with columns matching FEATURE_ORDER (see allocator.py).

Sliding-window approach: for each event in claude.jsonl + codex.jsonl, take the
moment of the event as "now" and compute lookback aggregates over the prior
1h/6h/24h. Label is set later (by label_dataset.py); this script outputs only
features + raw context for labeling.

Usage:
    python scripts/extract_features.py \
        --inbox-dir /tmp/ultradebate_v2/inbox \
        --jobs-dir /tmp/ultradebate_v2/jobs \
        --out artifacts/features_real.csv \
        [--min-history-hours 1]

The script is deterministic - same input -> same output (no MLP here).
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

FEATURE_ORDER = [
    "claude_5h_used", "codex_5h_used",
    "claude_week_used", "codex_week_used",
    "claude_1h_avg", "codex_1h_avg",
    "claude_6h_avg", "codex_6h_avg",
    "claude_24h_avg", "codex_24h_avg",
    "hour_sin", "hour_cos",
    "dow_sin", "dow_cos",
    "queue_len",
    "recent_task_type_bucket",
    "est_job_minutes",
    "recent_ack_miss_rate",
    "last_response_latency_s",
]

TASK_TYPE_BUCKETS = {
    "priority_send": 1, "ping": 2, "verdict": 3,
    "job_started": 4, "task_request": 5,
}


def parse_ts(ts: str) -> float:
    if not ts:
        return 0.0
    try:
        s = ts.replace("Z", "+00:00")
        if len(s) >= 5 and s[-5] in "+-" and s[-3] != ":":
            s = s[:-2] + ":" + s[-2:]
        return datetime.fromisoformat(s).timestamp()
    except Exception:
        return 0.0


def iter_jsonl(p: Path) -> Iterator[dict]:
    if not p.exists():
        return iter([])
    out = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return iter(out)


@dataclass
class Snapshot:
    ts: float
    event_id: str
    sender: str
    target: str
    msg_type: str
    features: dict


def estimate_usage(events: list[dict], *, now: float, window_seconds: float) -> dict[str, float]:
    cutoff = now - window_seconds
    cl, cx = 0, 0
    for e in events:
        et = e.get("_ts_epoch", 0)
        if et < cutoff or et > now:
            continue
        sender = e.get("sender") or ""
        if sender == "claude":
            cl += 1
        elif sender == "codex":
            cx += 1
    cap = max(5, int(window_seconds / 300))
    return {
        "claude": min(1.0, cl / cap),
        "codex": min(1.0, cx / cap),
    }


def compute_features(now: float, all_events: list[dict],
                     dow: int | None = None, hour: int | None = None) -> dict[str, float]:
    fivh = estimate_usage(all_events, now=now, window_seconds=5 * 3600)
    week = estimate_usage(all_events, now=now, window_seconds=7 * 86400)
    h1 = estimate_usage(all_events, now=now, window_seconds=3600)
    h6 = estimate_usage(all_events, now=now, window_seconds=6 * 3600)
    h24 = estimate_usage(all_events, now=now, window_seconds=24 * 3600)

    dt = datetime.fromtimestamp(now, tz=timezone.utc).astimezone()
    h = hour if hour is not None else dt.hour
    d = dow if dow is not None else dt.weekday()
    hour_sin, hour_cos = math.sin(2 * math.pi * h / 24), math.cos(2 * math.pi * h / 24)
    dow_sin, dow_cos = math.sin(2 * math.pi * d / 7), math.cos(2 * math.pi * d / 7)

    qcut = now - 60
    queue_len = sum(1 for e in all_events if qcut <= e.get("_ts_epoch", 0) <= now)

    recent_bucket = 0
    if all_events:
        last_e = max(all_events, key=lambda e: e.get("_ts_epoch", 0))
        recent_bucket = TASK_TYPE_BUCKETS.get(last_e.get("type") or "", 0)

    est_job_minutes = 0.0
    for e in reversed(all_events):
        if e.get("type") == "job_started" and now - e.get("_ts_epoch", 0) <= 3600:
            est_job_minutes = float(e.get("payload", {}).get("est_minutes") or 0)
            break

    cut = now - 1800
    sends, acks = 0, 0
    for e in all_events:
        if e.get("_ts_epoch", 0) < cut:
            continue
        t = e.get("type") or ""
        if "send" in t or "priority" in t:
            sends += 1
        payload_str = json.dumps(e.get("payload", {})) if e.get("payload") else ""
        if "ack" in t or "received_id" in payload_str:
            acks += 1
    miss = max(0.0, (sends - acks) / max(1, sends))
    miss = min(1.0, miss)

    last_lat = 0.0
    if all_events:
        last_e = max(all_events, key=lambda e: e.get("_ts_epoch", 0))
        last_lat = max(0.0, now - last_e.get("_ts_epoch", 0))

    return {
        "claude_5h_used": fivh["claude"], "codex_5h_used": fivh["codex"],
        "claude_week_used": week["claude"], "codex_week_used": week["codex"],
        "claude_1h_avg": h1["claude"], "codex_1h_avg": h1["codex"],
        "claude_6h_avg": h6["claude"], "codex_6h_avg": h6["codex"],
        "claude_24h_avg": h24["claude"], "codex_24h_avg": h24["codex"],
        "hour_sin": hour_sin, "hour_cos": hour_cos,
        "dow_sin": dow_sin, "dow_cos": dow_cos,
        "queue_len": float(queue_len),
        "recent_task_type_bucket": float(recent_bucket),
        "est_job_minutes": est_job_minutes,
        "recent_ack_miss_rate": miss,
        "last_response_latency_s": last_lat,
    }


def extract_from_inbox(inbox_dir: Path, jobs_dir: Path,
                       min_history_hours: float = 0.5) -> list[Snapshot]:
    events = []
    for fname in ["claude.jsonl", "codex.jsonl"]:
        for ev in iter_jsonl(inbox_dir / fname):
            ts_epoch = parse_ts(ev.get("ts") or "")
            if ts_epoch == 0.0:
                continue
            ev["_ts_epoch"] = ts_epoch
            events.append(ev)
    events.sort(key=lambda e: e["_ts_epoch"])
    if not events:
        return []
    earliest = events[0]["_ts_epoch"]

    snaps: list[Snapshot] = []
    for ev in events:
        now = ev["_ts_epoch"]
        if now - earliest < min_history_hours * 3600:
            continue
        prior = [e for e in events if e["_ts_epoch"] < now]
        feats = compute_features(now, prior)
        snaps.append(Snapshot(
            ts=now, event_id=str(ev.get("id", ""))[:16],
            sender=ev.get("sender") or "", target=ev.get("target") or "",
            msg_type=ev.get("type") or "", features=feats,
        ))
    return snaps


def write_csv(snaps: list[Snapshot], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["ts", "event_id", "sender", "target", "msg_type"] + FEATURE_ORDER
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for s in snaps:
            w.writerow([s.ts, s.event_id, s.sender, s.target, s.msg_type]
                       + [s.features.get(k, 0.0) for k in FEATURE_ORDER])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inbox-dir", default="/tmp/ultradebate_v2/inbox")
    ap.add_argument("--jobs-dir", default="/tmp/ultradebate_v2/jobs")
    ap.add_argument("--out", default="artifacts/features_real.csv")
    ap.add_argument("--min-history-hours", type=float, default=0.5)
    a = ap.parse_args()
    snaps = extract_from_inbox(Path(a.inbox_dir), Path(a.jobs_dir),
                               min_history_hours=a.min_history_hours)
    write_csv(snaps, Path(a.out))
    print(f"OK Wrote {len(snaps)} snapshots -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
