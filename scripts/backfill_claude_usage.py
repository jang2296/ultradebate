"""Backfill historical claude usage rows with reset-anchored FIXED windows.

WHY (Codex plan review 7.8/10, 2026-06-07): only the 103 claude snapshots taken
after the 2026-06-06 16:54 KST calibration carry correct fixed-window usage.
The ~726 earlier rows were computed with a ROLLING 7-day window + wrong caps
(weekly inflated up to 27x vs /status ground truth). Transcript token events
are still fully available back past the previous weekly reset (verified:
19,083 events since 2026-05-30 06:00 UTC, no gaps), so every historical row
can be recomputed exactly the way the live collector now does it.

VALIDATION GATE (mandatory, per review): the 103 already-correct rows
(reset_modeled=True) are recomputed and compared against their stored values.
If any |recomputed - stored| used-fraction exceeds --tolerance, the backfill
ABORTS — older rows must not be trusted when the gate fails (fallback: train
on the 103 calibrated rows + disclosed synthetic gap-fill).

Codex rows pass through untouched (they are authoritative rollout reads).

Usage:
    python scripts/backfill_claude_usage.py \
        --history artifacts/usage_history.jsonl \
        --out artifacts/usage_history_backfilled.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from bisect import bisect_right
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect_usage import (  # noqa: E402
    DEFAULT_CAP_5H,
    DEFAULT_CAP_WEEK,
    WEEK_SECONDS,
    _fixed_window,
    _load_reset_anchors,
    compute_snapshot,
    iter_usage_events,
    load_caps,
)


def _epoch(ts: str) -> float | None:
    try:
        # trailing-Z UTC suffix → +00:00 (Codex review L5: defensive normalize)
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except (ValueError, TypeError, AttributeError):
        return None


def recompute_claude_row(
    row: dict,
    events: list[tuple[float, float]],
    epochs: list[float],
    anchors: dict,
    cap_5h: float,
    cap_week: float,
) -> dict | None:
    """Return a corrected copy of a claude history row, or None if ts unparsable.

    Mirrors the live claude_snapshot() path exactly, except events are the
    preloaded full list cut at the row's timestamp (events after t did not
    exist when the original snapshot was taken).
    """
    t = _epoch(row.get("ts", ""))
    if t is None:
        return None
    ev_t = events[: bisect_right(epochs, t)]

    snap = compute_snapshot(ev_t, now=t, cap_5h=cap_5h, cap_week=cap_week)
    u5, t5, r5 = _fixed_window(ev_t, t, anchors["reset_5h_epoch"],
                               anchors["reset_5h_period"], cap_5h)
    uw, tw, rw = _fixed_window(ev_t, t, anchors["reset_week_epoch"],
                               anchors["reset_week_period"], cap_week)

    out = dict(row)  # preserve unknown fields
    out["used"] = dict(snap["used"])
    out["used"]["5h"], out["used"]["week"] = round(u5, 4), round(uw, 4)
    out["tokens"] = dict(snap["tokens"])
    out["tokens"]["5h"], out["tokens"]["week"] = int(t5), int(tw)
    out["caps"] = {"5h": cap_5h, "week": cap_week}
    out["5h_pct_left"] = int(round(100 * (1 - u5)))
    out["weekly_pct_left"] = int(round(100 * (1 - uw)))
    out["proxy_est_5h_pct_left"] = out["5h_pct_left"]
    out["proxy_est_weekly_pct_left"] = out["weekly_pct_left"]
    out["event_count_window"] = snap.get("event_count_window", len(ev_t))
    out["reset_kind"] = "fixed_window_anchored"
    out["reset_modeled"] = True
    out["secs_to_reset"] = {"5h": round(r5, 1), "week": round(rw, 1)}
    out.pop("secs_to_refill_proxy", None)  # rolling-proxy field is obsolete here
    out["backfilled"] = True
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", default="artifacts/usage_history.jsonl")
    ap.add_argument("--out", default="artifacts/usage_history_backfilled.jsonl")
    ap.add_argument("--projects-dir", default=str(Path.home() / ".claude" / "projects"))
    ap.add_argument("--tolerance", type=float, default=0.02,
                    help="max |recomputed - stored| used-fraction on the validation rows")
    a = ap.parse_args()

    anchors = _load_reset_anchors()
    if not anchors:
        print("ERROR no reset anchors in claude_caps.json — calibrate first")
        return 1
    cap_5h, cap_week = load_caps(DEFAULT_CAP_5H, DEFAULT_CAP_WEEK)

    rows: list[dict] = []
    with Path(a.history).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    claude_rows = [r for r in rows if r.get("agent") == "claude"]
    ts_list = [t for r in claude_rows if (t := _epoch(r.get("ts", ""))) is not None]
    if not ts_list:
        print("ERROR no claude rows with parsable ts")
        return 1

    # one scan covers everything: rolling 24h needs t-24h, fixed week needs the
    # anchored window start; oldest_ts - WEEK is a superset of both
    since = min(ts_list) - WEEK_SECONDS
    events = sorted(iter_usage_events(Path(a.projects_dir), since_epoch=since))
    epochs = [e[0] for e in events]
    print(f"loaded {len(rows)} history rows ({len(claude_rows)} claude), "
          f"{len(events)} token events since "
          f"{datetime.fromtimestamp(since, tz=timezone.utc).isoformat()}")

    # ---- VALIDATION GATE: recomputed must round-trip the 103 calibrated rows ----
    gate_rows = [r for r in claude_rows if r.get("reset_modeled")]
    worst = 0.0
    worst_row = None
    for r in gate_rows:
        rc = recompute_claude_row(r, events, epochs, anchors, cap_5h, cap_week)
        if rc is None:
            continue
        for w in ("5h", "week"):
            diff = abs(rc["used"][w] - float((r.get("used") or {}).get(w, 0.0)))
            if diff > worst:
                worst, worst_row = diff, (r.get("ts"), w, (r.get("used") or {}).get(w), rc["used"][w])
    print(f"validation gate: {len(gate_rows)} calibrated rows, worst |Δused| = {worst:.4f} "
          f"(tolerance {a.tolerance})")
    if worst > a.tolerance:
        print(f"GATE FAILED at {worst_row} — older backfilled rows must NOT be trusted.")
        print("Fallback per review: train on the calibrated rows + disclosed synthetic gap-fill.")
        return 2

    # ---- full backfill ----
    n_backfilled = 0
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            if r.get("agent") == "claude":
                rc = recompute_claude_row(r, events, epochs, anchors, cap_5h, cap_week)
                if rc is not None:
                    r = rc
                    n_backfilled += 1
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"OK wrote {len(rows)} rows -> {out_path} (claude backfilled: {n_backfilled})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
