"""Unified REAL-usage collector — claude + codex remaining usage + time-to-reset.

WHY (research_ultradebate_mlp_20260603 §3, cross-verified with Codex 2026-06-03)
  The shipped allocator MLP trains on ~92% AR(1)-synthetic rows; the 134 "real"
  rows had every usage field == 0 (the old extract_features path derived usage
  from inbox message *counts*, not real quota, and zeroed out). So the balancer
  never saw real Claude/Codex headroom. This collector fixes the *data source*:
  it writes genuine, account-wide usage snapshots so the model can later be
  retrained on real state.

  Account-wide insight (user, 2026-06-03): Claude/Codex rate limits pool across
  ALL sessions, so collecting from THIS mcp folder serves every session — one
  daemon, shared log, usable by the MCP from any session.

  HONEST FRAMING (Codex plan review 7.5/10): retraining on these snapshots is
  "real-state-distribution POLICY DISTILLATION" — the *features* (usage state)
  become real, but the *labels* are still the heuristic allocation policy, NOT
  observed optimal outcomes. Do not claim "trained on real optimal decisions"
  until outcome labels exist.

WHAT EACH TICK WRITES (to the live quota_log predict_allocation_live reads, AND a
persistent history file that survives a /tmp wipe)
  claude (source='transcript_proxy', proxy_est=True, reset_authoritative=False):
    rolling-window token sums -> 5h_pct_left (kept for back-compat) +
    proxy_est_*_pct_left + pct_left_source='proxy_est' + used{5h,week,1h,6h,24h}
    + secs_to_refill_proxy{5h,week} (rolling-window refill PROXY — see below).
  codex (source='rollout_passive' [DEFAULT], proxy_est=False, reset_authoritative=True):
    authoritative 5h/weekly pct_left + secs_to_reset{5h,week} read PASSIVELY from
    ~/.codex/sessions/**/rollout-*.jsonl rate_limits (used_percent + resets_at) —
    NON-intrusive, account-wide, as fresh as codex's last turn. observed_age_s
    records staleness; a window past its resets_at is treated as reset (idle).
    The legacy intrusive '/status' tmux probe (source='status_probe') remains via
    --codex-source status.

RESET-TIME SEMANTICS (Codex flagged this as the weakest dim — kept DISTINCT)
  Codex uses FIXED windows -> exact reset clock (rollout rate_limits.resets_at, or
  /status) -> stored as ``secs_to_reset`` (authoritative).
  Claude's transcript proxy is a ROLLING window with no single reset clock; the
  honest "time until usable amount refills" is when the window fully clears if
  activity stops = (newest_token_ts + window) - now. Stored under a DISTINCT key
  ``secs_to_refill_proxy`` so it is never merged with codex's authoritative clock
  into one feature column by accident. ``reset_kind`` + ``reset_authoritative``
  make the distinction explicit per snapshot.

CAPS / CALIBRATION
  Claude caps (5h/week token budgets) are ESTIMATES — Anthropic does not publish
  them. Run:  python collect_usage.py --calibrate-from-status "<paste /status 5h+
  weekly lines>"  to back out caps from a real %left + measured token sum; each
  call appends a calibration POINT and the persisted cap is the MEDIAN across
  points (robust to the formula's instability near 0%/100% and to coarse integer
  %). Persisted to artifacts/claude_caps.json (auto-loaded). The *relative* trend
  is robust to cap error; only the absolute % depends on it — hence proxy_est_*.

USAGE
  one-shot:   python scripts/collect_usage.py --once
  daemon:     nohup python scripts/collect_usage.py --watch --interval 300 \
                  --codex-interval 1800 --codex-pane 5:1.1 \
                  > /tmp/ultradebate_v2/usage_collector.log 2>&1 &
  calibrate:  python scripts/collect_usage.py --calibrate-from-status "5h limit: [..] 37% left (resets 02:00)
              Weekly limit: [..] 48% left (resets 09:00 on 7 Jun)"
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
# src/ultradebate on path so `import quota_probe` works standalone (its only
# relative import — .text_safety — is lazy + try/except, so it degrades cleanly).
sys.path.insert(0, str(_SCRIPTS.parent / "src" / "ultradebate"))

from collect_claude_usage import (  # noqa: E402
    DEFAULT_CAP_5H,
    DEFAULT_CAP_WEEK,
    DEFAULT_PROJECTS_DIR,
    WEEK_SECONDS,
    W_CACHE_CREATE,
    W_CACHE_READ,
    W_INPUT,
    W_OUTPUT,
    compute_snapshot,
    iter_usage_events,
)

CALIB_FORMULA_VERSION = "cap=window_tokens/(1-pct_left/100); v1 2026-06-03"

QUOTA_LOG = Path(
    os.environ.get("ULTRADEBATE_QUOTA_LOG", "/tmp/ultradebate_v2/quota_log.jsonl")
)
# Persistent training history (survives a /tmp wipe — Codex plan review).
HISTORY_LOG = Path(
    os.environ.get(
        "ULTRADEBATE_USAGE_HISTORY",
        str(_SCRIPTS.parent / "artifacts" / "usage_history.jsonl"),
    )
)
CAPS_CONFIG = Path(
    os.environ.get(
        "ULTRADEBATE_CLAUDE_CAPS_CONFIG",
        str(_SCRIPTS.parent / "artifacts" / "claude_caps.json"),
    )
)
DEFAULT_CODEX_PANE = os.environ.get("ULTRADEBATE_CODEX_PANE", "5:1.1")
# Codex writes authoritative rate_limits (used_percent + resets_at) into its
# rollout JSONL on every token_count event — so codex usage can be read PASSIVELY
# (no /status injection), account-wide, as often as we like. Discovered 2026-06-04.
CODEX_SESSIONS_DIR = Path(
    os.environ.get("ULTRADEBATE_CODEX_SESSIONS", os.path.expanduser("~/.codex/sessions"))
)

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


# --------------------------------------------------------------------------- #
# caps
# --------------------------------------------------------------------------- #
def load_caps(default_5h: float, default_week: float) -> tuple[float, float]:
    """artifacts/claude_caps.json overrides the env/default caps when present."""
    if CAPS_CONFIG.exists():
        try:
            d = json.loads(CAPS_CONFIG.read_text(encoding="utf-8"))
            return (
                float(d.get("cap_5h", default_5h)),
                float(d.get("cap_week", default_week)),
            )
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    return default_5h, default_week


def calibrate_from_status(status_text: str, *, projects_dir: Path) -> dict:
    """Back out caps from a real /status %left + the measured token sums.

    used = 1 - pct_left/100 = tokens_window / cap  ->  cap = tokens / used.
    Appends a calibration POINT; the persisted cap is the MEDIAN over points
    (robust to instability near 0%/100% and to coarse integer %). Stores raw
    inputs + formula version + weights for auditability (Codex plan review).
    """
    import quota_probe  # type: ignore  # local import — optional dep

    parsed = quota_probe.parse_status_output(status_text)
    if "5h_pct_left" not in parsed and "weekly_pct_left" not in parsed:
        return {"ok": False, "error": "no_status_lines_parsed", "input_tail": status_text[-200:]}

    now = time.time()
    events = iter_usage_events(projects_dir, since_epoch=now - WEEK_SECONDS)
    tok_5h = sum(t for et, t in events if now - 5 * 3600 <= et <= now)
    tok_week = sum(t for et, t in events if now - WEEK_SECONDS <= et <= now)

    point: dict = {
        "ts": datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
        "measured_tokens": {"5h": int(tok_5h), "week": int(tok_week)},
        "formula_version": CALIB_FORMULA_VERSION,
        "weights": {"output": W_OUTPUT, "input": W_INPUT,
                    "cache_create": W_CACHE_CREATE, "cache_read": W_CACHE_READ},
    }
    unstable: list[str] = []
    if "5h_pct_left" in parsed:
        p = parsed["5h_pct_left"]
        point["5h_pct_left_real"] = p
        if p >= 100 or p <= 0:
            unstable.append("5h")
        else:
            point["cap_5h"] = round(tok_5h / (1.0 - p / 100.0))
    if "weekly_pct_left" in parsed:
        p = parsed["weekly_pct_left"]
        point["weekly_pct_left_real"] = p
        if p >= 100 or p <= 0:
            unstable.append("week")
        else:
            point["cap_week"] = round(tok_week / (1.0 - p / 100.0))
    if unstable:
        point["unstable_windows"] = unstable  # pct at 0/100 -> no usable gradient

    if "cap_5h" not in point and "cap_week" not in point:
        return {"ok": False, "error": "pct_left_at_bound_no_signal", "point": point}

    CAPS_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    cfg: dict = {}
    if CAPS_CONFIG.exists():
        try:
            cfg = json.loads(CAPS_CONFIG.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            cfg = {}
    points: list[dict] = list(cfg.get("calibration_points") or [])
    points.append(point)
    cfg["calibration_points"] = points[-20:]  # keep last 20
    caps_5h = [pt["cap_5h"] for pt in cfg["calibration_points"] if "cap_5h" in pt]
    caps_wk = [pt["cap_week"] for pt in cfg["calibration_points"] if "cap_week" in pt]
    if caps_5h:
        cfg["cap_5h"] = round(statistics.median(caps_5h))
    if caps_wk:
        cfg["cap_week"] = round(statistics.median(caps_wk))
    cfg["formula_version"] = CALIB_FORMULA_VERSION
    cfg["updated_at"] = point["ts"]
    CAPS_CONFIG.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return {
        "ok": True,
        "point": point,
        "cap_5h": cfg.get("cap_5h"),
        "cap_week": cfg.get("cap_week"),
        "n_points": len(cfg["calibration_points"]),
        "config_path": str(CAPS_CONFIG),
        "note": "cap = median over calibration points" + (
            f"; unstable (pct at bound) skipped: {unstable}" if unstable else ""
        ),
    }


# --------------------------------------------------------------------------- #
# claude
# --------------------------------------------------------------------------- #
def _claude_refill_secs(events: list[tuple[float, float]], now: float, window: float) -> float:
    """Rolling-window REFILL proxy: secs until the window fully clears assuming no
    new activity = (newest_in_window_ts + window) - now, clamped >= 0.

    A rolling limit has no single reset clock. The honest "time until usable amount
    refills" is when the most recent token ages out (the whole window empties if
    you stop now). Under continuous use this pins near `window`; it shrinks as the
    pane goes idle. The oldest-token variant was rejected — it sits at ~0 under
    load (no signal). PROXY only — distinct from codex's authoritative CLI clock.
    """
    cutoff = now - window
    in_win = [et for et, _ in events if cutoff <= et <= now]
    if not in_win:
        return 0.0
    return max(0.0, (max(in_win) + window) - now)


def _load_reset_anchors() -> dict | None:
    """Real claude reset schedule (from a /status calibration) — enables FIXED-window
    modeling instead of the rolling proxy. Anthropic's 5h/weekly limits are FIXED
    windows (reset at a clock time), NOT rolling; without an anchor the proxy slides
    and never resets sharply — it MISSED the real weekly reset (user-observed
    2026-06-06: rolling 7-day summed 1361M while the real window since reset was 15.4M)."""
    if CAPS_CONFIG.exists():
        try:
            d = json.loads(CAPS_CONFIG.read_text(encoding="utf-8"))
            if "reset_5h_epoch" in d and "reset_week_epoch" in d:
                return d
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    return None


def _fixed_window(events: list[tuple[float, float]], now: float, reset_epoch: float,
                  period: float, cap: float) -> tuple[float, float, float]:
    """Tokens since the last reset BOUNDARY (fixed window anchored to reset_epoch),
    NOT a rolling window. Returns (used_fraction, tokens, secs_to_next_reset)."""
    start = reset_epoch + math.floor((now - reset_epoch) / period) * period
    nxt = start + period
    while nxt <= now:
        nxt += period
    toks = sum(t for e, t in events if start <= e <= now)
    used = min(1.0, toks / cap) if cap > 0 else 0.0
    return used, toks, max(0.0, nxt - now)


def claude_snapshot(projects_dir: Path, cap_5h: float, cap_week: float, *, now: float | None = None) -> dict:
    now = now if now is not None else time.time()
    events = iter_usage_events(projects_dir, since_epoch=now - WEEK_SECONDS)
    snap = compute_snapshot(events, now=now, cap_5h=cap_5h, cap_week=cap_week)

    anchors = _load_reset_anchors()
    if anchors:
        # FIXED-window anchored to the real /status reset schedule -> captures resets
        u5, t5, r5 = _fixed_window(events, now, anchors["reset_5h_epoch"],
                                   anchors["reset_5h_period"], cap_5h)
        uw, tw, rw = _fixed_window(events, now, anchors["reset_week_epoch"],
                                   anchors["reset_week_period"], cap_week)
        snap["used"]["5h"], snap["used"]["week"] = round(u5, 4), round(uw, 4)
        snap["tokens"]["5h"], snap["tokens"]["week"] = int(t5), int(tw)
        snap["5h_pct_left"] = int(round(100 * (1 - u5)))
        snap["weekly_pct_left"] = int(round(100 * (1 - uw)))
        snap["reset_kind"] = "fixed_window_anchored"
        snap["reset_modeled"] = True
        snap["secs_to_reset"] = {"5h": round(r5, 1), "week": round(rw, 1)}
    else:
        # no anchor yet -> rolling-window refill PROXY (slides, never resets sharply)
        snap["reset_kind"] = "rolling_window_refill_proxy"
        snap["secs_to_refill_proxy"] = {
            "5h": round(_claude_refill_secs(events, now, 5 * 3600), 1),
            "week": round(_claude_refill_secs(events, now, WEEK_SECONDS), 1),
        }
    # provenance: still a token-based ESTIMATE (cap-dependent) even when reset-modeled
    snap["proxy_est"] = True
    snap["pct_left_source"] = "proxy_est"
    snap["proxy_est_5h_pct_left"] = snap["5h_pct_left"]      # 5h_pct_left kept for back-compat
    snap["proxy_est_weekly_pct_left"] = snap["weekly_pct_left"]
    snap["reset_authoritative"] = False
    return snap


# --------------------------------------------------------------------------- #
# codex
# --------------------------------------------------------------------------- #
def parse_codex_reset(reset_str: str, now_dt: datetime) -> float | None:
    """'00:49 on 24 May' | '12:34 on 27 May 2026' | '02:00' -> epoch seconds.

    Times are interpreted in the local timezone (codex CLI prints local). Missing
    year -> current year, rolled forward a year if that lands >2 days in the past.
    Bare 'HH:MM' (5h reset) -> next occurrence today/tomorrow.
    """
    if not reset_str:
        return None
    s = reset_str.strip().lower()
    m = re.match(r"(\d{1,2}):(\d{2})(?:\s+on\s+(\d{1,2})\s+([a-z]{3})[a-z]*(?:\s+(\d{4}))?)?", s)
    if not m:
        return None
    hh, mm = int(m.group(1)), int(m.group(2))
    if not (0 <= hh <= 23 and 0 <= mm <= 59):
        return None
    local_now = now_dt.astimezone()
    if m.group(3) and m.group(4):  # explicit "on DD Mon [YYYY]"
        day = int(m.group(3))
        mon = _MONTHS.get(m.group(4)[:3])
        if not mon:
            return None
        year = int(m.group(5)) if m.group(5) else local_now.year
        try:
            cand = local_now.replace(
                year=year, month=mon, day=day, hour=hh, minute=mm, second=0, microsecond=0
            )
        except ValueError:
            return None
        if not m.group(5) and cand < local_now - timedelta(days=2):
            try:
                cand = cand.replace(year=year + 1)
            except ValueError:
                return None
    else:  # bare HH:MM -> next occurrence
        cand = local_now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if cand <= local_now:
            cand += timedelta(days=1)
    return cand.timestamp()


# Mirror of server.py _PANE_BUSY_MARKERS — MUST agree (Codex code review High-2:
# a lone "esc to interrupt" check missed "Working"/"Pursuing goal"/"thinking" and
# could fire /status into an actively-working Codex pane, e.g. "Pursuing goal (9h)").
_PANE_BUSY_MARKERS = ("esc to interrupt", "working", "pursuing goal", "thinking")


def _pane_is_idle(pane: str) -> bool:
    """True when the codex pane shows NO busy marker in its LIVE STATUS REGION.

    CRITICAL (2026-06-04): scan ONLY the last few non-empty lines, not the whole
    pane. Matching the markers against the FULL capture caused false positives —
    our own conversation prose contains the words "working"/"thinking"/"esc to
    interrupt"/"pursuing goal" (we literally discuss the markers), which pinned
    the pane as forever-busy and stalled codex collection ~9h while it was idle.
    Codex's real busy indicators ("• Working (Xs • esc to interrupt)",
    "Pursuing goal (Xh)") live on the bottom status line(s), so the tail suffices
    and excludes scrollback prose.
    """
    try:
        proc = subprocess.run(
            ["tmux", "capture-pane", "-p", "-t", pane],
            capture_output=True, timeout=5.0, check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False
    if proc.returncode != 0:
        return False
    lines = [ln for ln in proc.stdout.decode("utf-8", errors="replace").splitlines() if ln.strip()]
    tail = "\n".join(lines[-4:]).lower()  # live status region only (spinner/status bar)
    return not any(m in tail for m in _PANE_BUSY_MARKERS)


def codex_snapshot(pane: str, *, now: float | None = None, submit_key: str = "C-m") -> dict | None:
    """Probe codex /status (idle-guarded, intrusive) -> snapshot with secs_to_reset.

    Returns None when the pane is busy/absent/unparseable (caller keeps the last
    known codex snapshot rather than disturbing a working pane).
    """
    now = now if now is not None else time.time()
    if not _pane_is_idle(pane):
        return None
    try:
        import quota_probe  # type: ignore
    except ImportError:
        return None
    res = quota_probe.probe_codex_quota(
        pane, send_status_cmd=True, wait_after_send=4.0, submit_key=submit_key
    )
    if not res.get("ok"):
        return None
    res["source"] = "status_probe"
    res["proxy_est"] = False  # codex /status reports authoritative quota
    res["pct_left_source"] = "status_probe"
    res["reset_authoritative"] = True
    res["reset_kind"] = "cli_clock"
    # never persist raw pane text into the history (Codex plan review: hygiene)
    for k in ("raw_capture_tail", "note"):
        res.pop(k, None)
    now_dt = datetime.fromtimestamp(now, tz=timezone.utc)
    sr: dict = {}
    if res.get("5h_resets"):
        v = parse_codex_reset(res["5h_resets"], now_dt)
        if v is not None:
            sr["5h"] = round(max(0.0, v - now), 1)
    if res.get("weekly_resets"):
        v = parse_codex_reset(res["weekly_resets"], now_dt)
        if v is not None:
            sr["week"] = round(max(0.0, v - now), 1)
    if sr:
        res["secs_to_reset"] = sr  # authoritative CLI clock (distinct from claude proxy)
    return res


def _codex_next_reset(resets_at, window_minutes, now: float) -> float | None:
    """Seconds until the NEXT reset. If the recorded resets_at already passed
    (codex idle since), roll forward by whole window_minutes to the next boundary."""
    if not resets_at:
        return None
    try:
        nr = float(resets_at)
    except (TypeError, ValueError):
        return None
    win = float(window_minutes or 0) * 60.0
    if win > 0:
        while nr < now:
            nr += win
    return max(0.0, nr - now)


def codex_passive_snapshot(sessions_dir: Path, *, now: float | None = None,
                           max_age_days: float = 3.0, max_files: int = 8,
                           tail_bytes: int = 262144) -> dict | None:
    """Read codex's AUTHORITATIVE rate_limits PASSIVELY from rollout JSONL — no
    /status injection, account-wide, callable as often as claude (Codex logs
    rate_limits{primary,secondary}{used_percent,window_minutes,resets_at} on every
    token_count event).

    Reflects codex state at its LAST turn; ``observed_age_s`` records staleness, and
    a window whose resets_at already passed is treated as reset (pct_left=100) since
    no usage has occurred while idle. Returns None if no recent event is found.
    """
    now = now if now is not None else time.time()
    try:
        files = glob.glob(os.path.join(str(sessions_dir), "*", "*", "*", "rollout-*.jsonl"))
    except OSError:
        files = []
    cutoff = now - max_age_days * 86400
    recent: list[tuple[float, str]] = []
    for f in files:
        try:
            m = os.path.getmtime(f)
        except OSError:
            continue
        if m >= cutoff:
            recent.append((m, f))
    recent.sort(reverse=True)              # newest-modified first (the active session)
    recent = recent[: max(1, max_files)]   # bound the scan (Codex review M-2: scan cost)

    latest: tuple[float, dict] | None = None
    for _m, f in recent:
        # tail-read only the last tail_bytes — the newest token_count/rate_limits
        # event sits near EOF, so we avoid reading multi-MB rollout files in full.
        try:
            sz = os.path.getsize(f)
            with open(f, "rb") as fh:
                if sz > tail_bytes:
                    fh.seek(sz - tail_bytes)
                    fh.readline()  # discard partial first line
                chunk = fh.read().decode("utf-8", errors="replace")
        except OSError:
            continue
        for line in chunk.splitlines():
            if '"rate_limits"' not in line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            rl = (d.get("payload") or {}).get("rate_limits")
            if not rl:
                continue
            try:
                et = datetime.fromisoformat(
                    str(d.get("timestamp", "")).replace("Z", "+00:00")
                ).timestamp()
            except (ValueError, TypeError):
                continue
            if latest is None or et > latest[0]:
                latest = (et, rl)
    if latest is None:
        return None

    et, rl = latest
    p = rl.get("primary") or {}
    s = rl.get("secondary") or {}

    inferred: list[str] = []

    def _left(win: dict, name: str) -> int | None:
        up = win.get("used_percent")
        if up is None:
            return None
        ra = win.get("resets_at")
        if ra:
            try:
                if now > float(ra):
                    inferred.append(name)  # reset elapsed since last turn -> INFERRED, not observed
                    return 100
            except (TypeError, ValueError):
                pass
        return max(0, min(100, int(round(100.0 - float(up)))))

    p5, sw = _left(p, "5h"), _left(s, "week")
    snap: dict = {
        "ok": True,
        "agent": "codex",
        "source": "rollout_passive",
        "proxy_est": False,  # authoritative server-reported %, not an estimate
        "pct_left_source": "rollout_passive",
        "ts": datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
        "reset_authoritative": True,
        "reset_kind": "cli_clock",
        "observed_ts": datetime.fromtimestamp(et, tz=timezone.utc).isoformat(),
        "observed_age_s": round(max(0.0, now - et), 1),
        "plan_type": p.get("plan_type") or rl.get("plan_type"),
    }
    if p5 is not None:
        snap["5h_pct_left"] = p5
    if sw is not None:
        snap["weekly_pct_left"] = sw
    if inferred:
        # these windows' pct_left is INFERRED from an elapsed reset (idle assumption),
        # NOT a fresh observation — flagged so consumers don't over-trust headroom
        # during an active post-reset turn before the next token_count lands (Codex review M-1).
        snap["pct_left_inferred_after_reset"] = inferred
    sr: dict = {}
    r5 = _codex_next_reset(p.get("resets_at"), p.get("window_minutes"), now)
    rw = _codex_next_reset(s.get("resets_at"), s.get("window_minutes"), now)
    if r5 is not None:
        sr["5h"] = round(r5, 1)
    if rw is not None:
        sr["week"] = round(rw, 1)
    if sr:
        snap["secs_to_reset"] = sr
    if "5h_pct_left" not in snap and "weekly_pct_left" not in snap:
        return None
    return snap


# --------------------------------------------------------------------------- #
# record / loop
# --------------------------------------------------------------------------- #
def record(snapshot: dict, *, log_paths: list[Path]) -> None:
    """Append one snapshot to every log path (live + persistent history)."""
    line = json.dumps(snapshot, ensure_ascii=False) + "\n"
    for log_path in log_paths:
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(line)
        except OSError as e:
            print(f"[collect_usage] record WARN {log_path}: {e!r}", file=sys.stderr, flush=True)


def _log_paths(args) -> list[Path]:
    paths = [Path(args.out)]
    if args.history_out:
        hp = Path(args.history_out)
        if all(hp != p for p in paths):
            paths.append(hp)
    return paths


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    for ln in path.read_text(encoding="utf-8", errors="replace").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            continue
    return out


def _last_codex_probe_epoch(paths: list[Path]) -> float:
    """Most recent codex snapshot ts across logs — makes the 30-min throttle
    survive restarts / a second daemon / --once (Codex code review Medium-5:
    in-memory-only throttling let two probes land 87s apart)."""
    latest = 0.0
    for p in paths:
        for s in _read_jsonl(p):
            if s.get("agent") == "codex":
                try:
                    e = datetime.fromisoformat(
                        str(s.get("ts", "")).replace("Z", "+00:00")
                    ).timestamp()
                except (ValueError, TypeError):
                    continue
                latest = max(latest, e)
    return latest


def run_once(args, *, last_codex_t: float) -> tuple[dict, dict | None, float]:
    now = time.time()
    paths = _log_paths(args)
    cl = claude_snapshot(Path(args.projects_dir), args.cap_5h, args.cap_week, now=now)
    if not args.dry_run:
        record(cl, log_paths=paths)

    cx: dict | None = None
    if (not args.codex) or args.codex_source == "off":
        return cl, None, last_codex_t

    if args.codex_source == "rollout":
        # passive, non-intrusive -> read every tick (same 5-min cadence as claude)
        cx = codex_passive_snapshot(Path(args.codex_sessions), now=now)
        if cx is not None and not args.dry_run:
            record(cx, log_paths=paths)
    elif args.codex_source == "status" and (now - last_codex_t) >= args.codex_interval:
        # intrusive /status tmux probe (throttled + idle-guarded)
        cx = codex_snapshot(args.codex_pane, now=now, submit_key=args.codex_submit_key)
        if cx is not None:
            if not args.dry_run:
                record(cx, log_paths=paths)
            last_codex_t = now
        else:
            # busy/failed: retry in ~5 min instead of waiting a full interval
            last_codex_t = now - args.codex_interval + min(300.0, args.codex_interval)
    return cl, cx, last_codex_t


def _fmt(cl: dict, cx: dict | None) -> str:
    cr5 = (cl.get("secs_to_reset") or cl.get("secs_to_refill_proxy") or {}).get("5h", 0)
    kind = "reset" if cl.get("reset_modeled") else "refill"
    c = (
        f"claude {cl['proxy_est_5h_pct_left']}%/5h {cl['weekly_pct_left']}%/wk "
        f"{kind}5h={cr5 / 3600:.1f}h ev={cl['event_count_window']}"
    )
    if cx is None:
        return c + " | codex (none)"
    sr = cx.get("secs_to_reset", {})
    r5 = f"{sr['5h'] / 3600:.1f}h" if "5h" in sr else "?"
    age = f" age={cx['observed_age_s'] / 60:.0f}m" if cx.get("source") == "rollout_passive" else ""
    return c + (
        f" | codex {cx.get('5h_pct_left', '?')}%/5h "
        f"{cx.get('weekly_pct_left', '?')}%/wk reset5h={r5}{age}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--projects-dir", default=str(DEFAULT_PROJECTS_DIR))
    ap.add_argument("--out", default=str(QUOTA_LOG), help="live log the MCP server reads")
    ap.add_argument("--history-out", default=str(HISTORY_LOG),
                    help="persistent append-only training history (survives /tmp wipe); '' to disable")
    ap.add_argument("--cap-5h", type=float, default=None, dest="cap_5h")
    ap.add_argument("--cap-week", type=float, default=None, dest="cap_week")
    ap.add_argument("--watch", action="store_true", help="loop forever, append every --interval s")
    ap.add_argument("--interval", type=float, default=300.0, help="claude snapshot cadence (s)")
    ap.add_argument("--codex-interval", type=float, default=1800.0,
                    help="min seconds between (intrusive) codex /status probes")
    ap.add_argument("--codex-pane", default=DEFAULT_CODEX_PANE)
    ap.add_argument("--codex-submit-key", default="C-m", help="tmux submit key for /status (user mandate: C-m)")
    ap.add_argument("--codex-source", choices=["rollout", "status", "off"], default="rollout",
                    help="rollout=PASSIVE read of ~/.codex rate_limits (DEFAULT — non-intrusive, "
                         "authoritative, 5-min capable like claude); status=intrusive /status tmux "
                         "probe (throttled, legacy); off=skip codex")
    ap.add_argument("--codex-sessions", default=str(CODEX_SESSIONS_DIR),
                    help="codex rollout sessions dir for --codex-source rollout")
    ap.add_argument("--no-codex", dest="codex", action="store_false", help="claude-only (no codex)")
    ap.add_argument("--once", action="store_true", help="single snapshot then exit (default)")
    ap.add_argument("--dry-run", action="store_true", help="compute + print, do not append")
    ap.add_argument("--calibrate-from-status", default=None,
                    help="paste /status 5h+weekly lines to back out + persist caps, then exit")
    a = ap.parse_args()

    # caps: explicit flag > calibrated config > env/default
    cfg_5h, cfg_week = load_caps(DEFAULT_CAP_5H, DEFAULT_CAP_WEEK)
    if a.cap_5h is None:
        a.cap_5h = cfg_5h
    if a.cap_week is None:
        a.cap_week = cfg_week

    if a.calibrate_from_status is not None:
        res = calibrate_from_status(a.calibrate_from_status, projects_dir=Path(a.projects_dir))
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return 0 if res.get("ok") else 1

    if a.watch:
        print(
            f"[collect_usage] watch start interval={a.interval}s codex_source={a.codex_source} "
            f"(status-throttle={a.codex_interval}s) caps(5h={int(a.cap_5h)},week={int(a.cap_week)}) "
            f"out={a.out} history={a.history_out}",
            flush=True,
        )
        last_codex_t = _last_codex_probe_epoch(_log_paths(a))
        if last_codex_t and a.codex_source == "status":
            print(f"[collect_usage] resuming codex /status throttle from last probe "
                  f"{int(max(0, time.time() - last_codex_t))}s ago", flush=True)
        while True:
            try:
                cl, cx, last_codex_t = run_once(a, last_codex_t=last_codex_t)
                print(f"[collect_usage] {cl['ts']} {_fmt(cl, cx)}", flush=True)
            except Exception as e:  # never let the collector die silently
                print(f"[collect_usage] ERROR {e!r}", file=sys.stderr, flush=True)
            time.sleep(max(5.0, float(a.interval)))

    cl, cx, _ = run_once(a, last_codex_t=_last_codex_probe_epoch(_log_paths(a)))
    print(json.dumps({"claude": cl, "codex": cx}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
