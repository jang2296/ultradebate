"""Continuous Claude usage collector — transcript token proxy (no manual /status).

WHY THIS EXISTS
  quota_probe.probe_claude_quota() returns 'not_directly_probable': Claude Code's
  /status renders in the host UI and cannot be captured from inside the agent
  loop, so the MLP's claude_5h_used / claude_week_used had to be entered by hand.
  That made "며칠 동안 지속 수집" impractical and left the real training set with
  claude usage == 0 (see research_ultradebate_mlp_20260603.md §3).

  BUT Claude Code writes every assistant turn — with exact token counts — to
  ~/.claude/projects/<proj>/<session>.jsonl (message.usage.{input,output,
  cache_*}_tokens + a top-level ISO `timestamp`). The rate limit is ACCOUNT-WIDE,
  so summing those token counts across ALL projects over rolling windows gives a
  continuously-collectable, fully-automatic proxy for "how much has Claude been
  working" — exactly the signal the balancer needs (who has more headroom / who
  is idle).

WHAT IT EMITS
  Appends snapshots to the same quota_log.jsonl that quota_probe / predict_
  allocation_live already read, shaped so snapshot_to_features picks them up:
    {agent:'claude', source:'transcript_proxy', '5h_pct_left':N, 'weekly_pct_left':N,
     used:{'5h':f,'week':f,'1h':f,'6h':f,'24h':f}, tokens:{...}, ts:...}

  pct_left = 100*(1 - tokens_in_window / cap).  ⚠ The caps are ESTIMATES — the
  exact Anthropic token budget per 5h/week is not published. Calibrate by running
  /status once, comparing the real %left to this proxy, and adjusting --cap-5h /
  --cap-week (or env ULTRADEBATE_CLAUDE_CAP_5H / _WEEK). The *relative* trend
  (claude vs codex, rising vs falling) is robust to cap error; only the absolute
  % depends on it.

USAGE
  One-shot (cron-friendly):
    python scripts/collect_claude_usage.py --once
  Continuous "alarm" (append a snapshot every 5 min):
    nohup python scripts/collect_claude_usage.py --watch --interval 300 \
        > /tmp/ultradebate_v2/claude_usage_collector.log 2>&1 &

  Pair with the codex /status probe to keep BOTH sides fresh, then call
  ultradebate_v2_predict_allocation_live for a live recommendation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# --- billable-token weighting (proxy for limit consumption, not exact cost) ----
# Anthropic exposes input_tokens (uncached), cache_creation_input_tokens,
# cache_read_input_tokens (cheap), output_tokens. These weights approximate how
# much each counts toward the rolling limit; tune if you have better numbers.
W_OUTPUT = float(os.environ.get("ULTRADEBATE_CLAUDE_W_OUTPUT", "1.0"))
W_INPUT = float(os.environ.get("ULTRADEBATE_CLAUDE_W_INPUT", "1.0"))
W_CACHE_CREATE = float(os.environ.get("ULTRADEBATE_CLAUDE_W_CACHE_CREATE", "1.0"))
W_CACHE_READ = float(os.environ.get("ULTRADEBATE_CLAUDE_W_CACHE_READ", "0.1"))

# Defaults calibrated against observed heavy usage on this account (≈59M billable
# tokens / 5h peak, ≈1.5B / week). RECALIBRATE for your plan: run /status once,
# read the real %left, and set ULTRADEBATE_CLAUDE_CAP_5H / _WEEK so the proxy's
# %left matches. Too-low caps pin the value at 0% (no gradient → no MLP signal).
DEFAULT_CAP_5H = float(os.environ.get("ULTRADEBATE_CLAUDE_CAP_5H", "120000000"))
DEFAULT_CAP_WEEK = float(os.environ.get("ULTRADEBATE_CLAUDE_CAP_WEEK", "3000000000"))

DEFAULT_PROJECTS_DIR = Path(
    os.environ.get("ULTRADEBATE_CLAUDE_PROJECTS", os.path.expanduser("~/.claude/projects"))
)
QUOTA_LOG = Path(
    os.environ.get("ULTRADEBATE_QUOTA_LOG", "/tmp/ultradebate_v2/quota_log.jsonl")
)

WEEK_SECONDS = 7 * 86400


def _parse_ts(ts: str) -> float:
    """ISO-8601 (with trailing Z) -> epoch seconds. 0.0 on failure."""
    if not ts:
        return 0.0
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except (ValueError, TypeError):
        return 0.0


def _billable(usage: dict) -> float:
    """Weighted token count from a message.usage dict."""
    return (
        W_OUTPUT * float(usage.get("output_tokens") or 0)
        + W_INPUT * float(usage.get("input_tokens") or 0)
        + W_CACHE_CREATE * float(usage.get("cache_creation_input_tokens") or 0)
        + W_CACHE_READ * float(usage.get("cache_read_input_tokens") or 0)
    )


def iter_usage_events(
    projects_dir: Path, *, since_epoch: float
) -> list[tuple[float, float]]:
    """Scan every transcript, return [(epoch, billable_tokens), ...] >= since_epoch.

    Account-wide: the rate limit pools across all projects, so we read them all.
    Files untouched since ``since_epoch`` are skipped via mtime (cheap prefilter).
    """
    events: list[tuple[float, float]] = []
    if not projects_dir.exists():
        return events
    for jsonl in projects_dir.glob("*/*.jsonl"):
        try:
            if jsonl.stat().st_mtime < since_epoch - 3600:
                continue  # whole file predates the window (1h slack for clock skew)
        except OSError:
            continue
        try:
            with jsonl.open(encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line or '"usage"' not in line:
                        continue
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    msg = d.get("message") or {}
                    usage = msg.get("usage")
                    if not usage:
                        continue
                    if d.get("type") != "assistant" and msg.get("role") != "assistant":
                        continue
                    et = _parse_ts(d.get("timestamp") or "")
                    if et < since_epoch:
                        continue
                    events.append((et, _billable(usage)))
        except OSError:
            continue
    events.sort(key=lambda e: e[0])
    return events


def _window_tokens(events: list[tuple[float, float]], now: float, seconds: float) -> float:
    cutoff = now - seconds
    return sum(tok for et, tok in events if cutoff <= et <= now)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def compute_snapshot(
    events: list[tuple[float, float]],
    *,
    now: float,
    cap_5h: float,
    cap_week: float,
) -> dict:
    """Build a quota_log snapshot from token events using rolling windows."""
    tok = {
        "5h": _window_tokens(events, now, 5 * 3600),
        "week": _window_tokens(events, now, WEEK_SECONDS),
        "1h": _window_tokens(events, now, 3600),
        "6h": _window_tokens(events, now, 6 * 3600),
        "24h": _window_tokens(events, now, 24 * 3600),
    }
    # 1h/6h/24h are normalized against time-scaled fractions of the 5h cap so
    # they live on the same 0..1 "used" axis as 5h_used.
    cap_per_hour = cap_5h / 5.0
    used = {
        "5h": _clamp(tok["5h"] / cap_5h, 0.0, 1.0) if cap_5h > 0 else 0.0,
        "week": _clamp(tok["week"] / cap_week, 0.0, 1.0) if cap_week > 0 else 0.0,
        "1h": _clamp(tok["1h"] / (cap_per_hour * 1), 0.0, 1.0) if cap_per_hour > 0 else 0.0,
        "6h": _clamp(tok["6h"] / (cap_per_hour * 6), 0.0, 1.0) if cap_per_hour > 0 else 0.0,
        "24h": _clamp(tok["24h"] / (cap_per_hour * 24), 0.0, 1.0) if cap_per_hour > 0 else 0.0,
    }
    return {
        "ok": True,
        "agent": "claude",
        "source": "transcript_proxy",
        "ts": datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
        "5h_pct_left": int(round(100 * (1 - used["5h"]))),
        "weekly_pct_left": int(round(100 * (1 - used["week"]))),
        "used": {k: round(v, 4) for k, v in used.items()},
        "tokens": {k: int(v) for k, v in tok.items()},
        "caps": {"5h": int(cap_5h), "week": int(cap_week)},
        "event_count_window": sum(1 for et, _ in events if et >= now - WEEK_SECONDS),
    }


def record(snapshot: dict, *, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(snapshot, ensure_ascii=False) + "\n")


def run_once(args) -> dict:
    now = time.time()
    events = iter_usage_events(Path(args.projects_dir), since_epoch=now - WEEK_SECONDS)
    snap = compute_snapshot(events, now=now, cap_5h=args.cap_5h, cap_week=args.cap_week)
    if not args.dry_run:
        record(snap, log_path=Path(args.out))
    return snap


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--projects-dir", default=str(DEFAULT_PROJECTS_DIR))
    ap.add_argument("--out", default=str(QUOTA_LOG))
    ap.add_argument("--cap-5h", type=float, default=DEFAULT_CAP_5H, dest="cap_5h")
    ap.add_argument("--cap-week", type=float, default=DEFAULT_CAP_WEEK, dest="cap_week")
    ap.add_argument("--watch", action="store_true", help="loop forever, append every --interval s")
    ap.add_argument("--interval", type=float, default=300.0)
    ap.add_argument("--once", action="store_true", help="single snapshot then exit (default)")
    ap.add_argument("--dry-run", action="store_true", help="compute + print, do not append")
    a = ap.parse_args()

    if a.watch:
        print(f"[collect_claude_usage] watch start interval={a.interval}s out={a.out}", flush=True)
        while True:
            try:
                snap = run_once(a)
                print(
                    f"[collect_claude_usage] {snap['ts']} 5h_left={snap['5h_pct_left']}% "
                    f"week_left={snap['weekly_pct_left']}% tok5h={snap['tokens']['5h']} "
                    f"events={snap['event_count_window']}",
                    flush=True,
                )
            except Exception as e:  # never let the collector die silently
                print(f"[collect_claude_usage] ERROR {e!r}", file=sys.stderr, flush=True)
            time.sleep(max(5.0, float(a.interval)))

    snap = run_once(a)
    print(json.dumps(snap, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
