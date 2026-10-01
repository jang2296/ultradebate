"""Build a labeled TRAINING dataset from REAL usage snapshots (collect_usage.py).

This is the bridge from "real state collected" -> "model trained on real state".
It reads the usage_history.jsonl that collect_usage.py accumulates, pairs each
claude snapshot with the nearest-in-time codex snapshot, and emits rows in the
SAME schema as features_augmented.csv (so train_allocator_mlp.py consumes them
directly, and the msg_type tag 'real_live' makes the train script count them as
the 'real' subset).

HONEST SCOPE (Codex plan review 7.5/10)
  * FEATURES become real: claude usage from the transcript proxy, codex usage
    from the authoritative /status probe.
  * LABELS remain the heuristic allocation policy (label_real_features.label_row)
    applied to real state — this is "real-state-distribution POLICY DISTILLATION",
    NOT observed-optimal outcomes. Do not over-claim.
  * codex has no 1h/6h/24h windows in /status (only 5h + weekly). Those slots are
    filled with codex_5h_used as the best available proxy (flagged in --help).
  * operational slots not captured by the quota collector (queue_len,
    recent_task_type_bucket, est_job_minutes, recent_ack_miss_rate,
    last_response_latency_s) default to 0.0.
  * Rows with NO usage signal (both 5h usages ~0) are SKIPPED — they are exactly
    the degenerate rows that poisoned the legacy real set.
  * reset-time is emitted as EXTRA trailing columns (not in FEATURE_ORDER yet —
    staged deferral) so it is available for a future feature-schema expansion.

USAGE
  python scripts/build_real_dataset.py \
      --history artifacts/usage_history.jsonl \
      --out artifacts/features_real_live.csv --max-gap 1800
  # then (once >=~50 rows accumulate) combine + retrain:
  #   cat artifacts/features_augmented.csv <(tail -n +2 features_real_live.csv) > combined.csv
  #   python scripts/train_allocator_mlp.py --data combined.csv ...
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
from extract_features import FEATURE_ORDER  # noqa: E402
from label_real_features import label_row  # noqa: E402

EXTRA_COLS = [
    "claude_refill_5h_s", "claude_refill_week_s", "codex_reset_5h_s", "codex_reset_week_s",
    "pair_gap_s", "codex_observed_ts", "codex_stale_age_s",  # pairing freshness (Codex review M-4)
]
EPS = 1e-3  # "no usage signal" threshold


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _epoch(ts: str) -> float:
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).timestamp()
    except (ValueError, TypeError):
        return 0.0


def _nearest(snaps: list[tuple[float, dict]], t: float, max_gap: float) -> tuple[dict | None, float, float]:
    """Return (snapshot, observed_epoch, gap_secs) for the closest snapshot within
    max_gap, else (None, 0.0, inf). Gap is recorded so a model never treats a
    30-min-stale codex point as fresh (Codex review M-4)."""
    best, best_et, bestd = None, 0.0, max_gap
    for et, s in snaps:
        d = abs(et - t)
        if d <= bestd:
            best, best_et, bestd = s, et, d
    return best, best_et, (bestd if best is not None else float("inf"))


def _codex_used(cx: dict, key: str) -> float | None:
    if key in cx and cx[key] is not None:
        try:
            return max(0.0, min(1.0, 1.0 - float(cx[key]) / 100.0))
        except (ValueError, TypeError):
            return None
    return None


def build_rows(claude_snaps, codex_snaps, *, max_gap: float) -> tuple[list[dict], dict]:
    cx_idx = sorted((_epoch(s.get("ts", "")), s) for s in codex_snaps if _epoch(s.get("ts", "")))
    rows: list[dict] = []
    stats = {"claude_total": 0, "paired": 0, "skipped_no_codex": 0, "skipped_no_signal": 0}
    for cl in claude_snaps:
        t = _epoch(cl.get("ts", ""))
        if not t:
            continue
        stats["claude_total"] += 1
        cx, cx_et, gap = _nearest(cx_idx, t, max_gap)
        if cx is None:
            stats["skipped_no_codex"] += 1
            continue
        used = cl.get("used") or {}
        c5 = float(used.get("5h", 1.0 - cl.get("proxy_est_5h_pct_left", 100) / 100.0))
        x5 = _codex_used(cx, "5h_pct_left")
        if x5 is None:
            stats["skipped_no_codex"] += 1
            continue
        if max(c5, x5) < EPS:
            stats["skipped_no_signal"] += 1
            continue
        cw = float(used.get("week", 1.0 - cl.get("proxy_est_weekly_pct_left", 100) / 100.0))
        xw = _codex_used(cx, "weekly_pct_left")
        xw = xw if xw is not None else x5
        dt = datetime.fromtimestamp(t, tz=timezone.utc).astimezone()
        h, dow = dt.hour, dt.weekday()
        feats = {
            "claude_5h_used": c5, "codex_5h_used": x5,
            "claude_week_used": cw, "codex_week_used": xw,
            "claude_1h_avg": float(used.get("1h", c5)), "codex_1h_avg": x5,   # codex finer windows
            "claude_6h_avg": float(used.get("6h", c5)), "codex_6h_avg": x5,   # unavailable from /status
            "claude_24h_avg": float(used.get("24h", c5)), "codex_24h_avg": x5,  # -> 5h proxy
            "hour_sin": math.sin(2 * math.pi * h / 24), "hour_cos": math.cos(2 * math.pi * h / 24),
            "dow_sin": math.sin(2 * math.pi * dow / 7), "dow_cos": math.cos(2 * math.pi * dow / 7),
            "queue_len": 0.0, "recent_task_type_bucket": 0.0, "est_job_minutes": 0.0,
            "recent_ack_miss_rate": 0.0, "last_response_latency_s": 0.0,
        }
        clr = cl.get("secs_to_reset") or cl.get("secs_to_refill_proxy") or {}
        cxr = cx.get("secs_to_reset") or {}
        extra = {
            "claude_refill_5h_s": clr.get("5h", ""), "claude_refill_week_s": clr.get("week", ""),
            "codex_reset_5h_s": cxr.get("5h", ""), "codex_reset_week_s": cxr.get("week", ""),
            "pair_gap_s": round(gap, 1),
            "codex_observed_ts": round(cx_et, 1),
            "codex_stale_age_s": round(max(0.0, t - cx_et), 1),
        }
        rows.append({"ts": t, "label": label_row(feats), "feats": feats, "extra": extra})
        stats["paired"] += 1
    return rows, stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--history", default=str(_SCRIPTS.parent / "artifacts" / "usage_history.jsonl"),
                    help="persistent snapshot log from collect_usage.py")
    ap.add_argument("--also-live", default="/tmp/ultradebate_v2/quota_log.jsonl",
                    help="extra snapshot log to merge (live); '' to skip")
    ap.add_argument("--out", default=str(_SCRIPTS.parent / "artifacts" / "features_real_live.csv"))
    ap.add_argument("--max-gap", type=float, default=1800.0,
                    help="max secs between a claude row and its paired codex probe")
    ap.add_argument("--dedup-codex-observed-ts", action="store_true",
                    help="keep one row per distinct codex_observed_ts (min pair_gap) — avoids "
                         "overweighting long codex-idle periods in training (Codex review M-3)")
    a = ap.parse_args()

    snaps = _load(Path(a.history))
    if a.also_live:
        seen = {(s.get("agent"), s.get("ts")) for s in snaps}
        merged = skipped_legacy = 0
        for s in _load(Path(a.also_live)):
            if (s.get("agent"), s.get("ts")) in seen:
                continue
            if "pct_left_source" not in s:  # legacy pre-collect_usage row (Codex review M-6)
                skipped_legacy += 1
                continue
            snaps.append(s)
            merged += 1
        if skipped_legacy:
            print(f"  --also-live merged {merged}, skipped {skipped_legacy} legacy rows (no pct_left_source)")
    claude_snaps = [s for s in snaps if s.get("agent") == "claude"]
    codex_snaps = [s for s in snaps if s.get("agent") == "codex"]
    print(f"loaded {len(snaps)} snapshots: claude={len(claude_snaps)} codex={len(codex_snaps)}")

    rows, stats = build_rows(claude_snaps, codex_snaps, max_gap=a.max_gap)
    if a.dedup_codex_observed_ts and rows:
        best: dict = {}
        for r in rows:
            k = r["extra"].get("codex_observed_ts")
            if k not in best or r["extra"]["pair_gap_s"] < best[k]["extra"]["pair_gap_s"]:
                best[k] = r
        before = len(rows)
        rows = sorted(best.values(), key=lambda r: r["ts"])
        print(f"  --dedup-codex-observed-ts: {before} -> {len(rows)} rows (one per distinct codex observation)")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["ts", "event_id", "sender", "target", "msg_type"] + FEATURE_ORDER + ["label"] + EXTRA_COLS
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow(
                [r["ts"], "", "", "", "real_live"]
                + [r["feats"][k] for k in FEATURE_ORDER]
                + [r["label"]]
                + [r["extra"][k] for k in EXTRA_COLS]
            )
    from collections import Counter
    bal = dict(Counter(r["label"] for r in rows))
    print(f"OK wrote {len(rows)} real_live rows -> {out}")
    print(f"   stats={stats}  label_balance={bal}")
    if len(rows) < 50:
        print(f"   NOTE: only {len(rows)} rows — need >=50 to train. Keep the collector "
              f"daemon running to accumulate more, then combine with features_augmented.csv.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
