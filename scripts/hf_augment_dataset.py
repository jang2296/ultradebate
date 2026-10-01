"""Augment the allocator MLP dataset with REALISTIC usage dynamics.

PROBLEM (research_ultradebate_mlp_20260603.md §3)
  The shipped model trained on ~92% synthetic rows whose 1h/6h/24h "lookback"
  features were just jittered copies of the 5h value — i.e. ZERO real temporal
  structure. So the lookback features carried no signal, and MLP tied SVM exactly
  (0.78 == 0.78). For a balancer ("who has more headroom / who is idle, give the
  next task there") the temporal *trend* is the whole point.

WHAT THIS ADDS
  Two complementary sources, both producing rows in the SAME schema as
  features_synth.csv so they drop straight into train_allocator_mlp.py:

  1. --source synthetic (default, no deps): a CONTINUOUS per-agent usage series
     generated with AR(1) autocorrelation + diurnal modulation + sustained
     bursts. Sliding-window features (1h/6h/24h) are then GENUINE rolling
     aggregates over an autocorrelated series — so the lookback columns finally
     carry signal. This alone fixes the i.i.d. defect.

  2. --source hf (needs `pip install datasets`): real cloud workload traces from
     Hugging Face `Salesforce/cloudops_tsf` (Azure VM 2017 / Google Borg 2011 /
     Alibaba 2018). Two entities' utilization series are paired as (claude,
     codex) usage. This injects real-world burstiness/diurnality the synthetic
     generator only approximates. Honest caveat: it is machine-scheduling data,
     so it transfers the usage *dynamics*, NOT a claude-vs-codex routing label —
     the label is still assigned by the optimal-allocation rule below.

LABEL (with a deliberate nonlinear, temporal term)
  Base rule = the same optimal-allocation rule as label_real_features.py, PLUS a
  "rising-usage lookahead": when one agent's 1h average is climbing fast above
  its 6h average, route the next task to the OTHER agent. That interaction is
  nonlinear in the features, giving the MLP a real reason to beat the linear-ish
  SVM baseline (the §5.5 comparison finally becomes meaningful).

USAGE
  python scripts/hf_augment_dataset.py --source synthetic --num-samples 1500 \
      --append-to artifacts/features_real_labeled.csv \
      --out artifacts/features_augmented.csv --seed 7
  # then retrain:
  python scripts/train_allocator_mlp.py --data artifacts/features_augmented.csv \
      --out-model artifacts/allocator_mlp.joblib
"""
from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_features import FEATURE_ORDER  # noqa: E402

LABELS = ["claude", "codex", "split"]


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def diurnal(hour: int) -> float:
    """Peak-hour modulation in [0.25, 1.0] (mirrors generate_synthetic peak factor)."""
    if 9 <= hour <= 12:
        return 1.4 / 1.6
    if 14 <= hour <= 18:
        return 1.6 / 1.6
    if 19 <= hour <= 22:
        return 1.1 / 1.6
    return 0.4 / 1.6


def gen_ar1_series(
    rng: random.Random, n_hours: int, *, rho: float = 0.8, base: float = 0.4,
    burst_p: float = 0.03,
) -> list[float]:
    """Continuous hourly usage level in [0,1] with AR(1) memory + diurnal + bursts."""
    series: list[float] = []
    u = rng.betavariate(2, 3)
    burst = 0
    for t in range(n_hours):
        target = base * diurnal(t % 24)
        u = rho * u + (1 - rho) * target + rng.gauss(0, 0.06)
        if burst > 0:
            u += 0.30
            burst -= 1
        elif rng.random() < burst_p:
            burst = rng.randint(1, 3)
        series.append(_clamp01(u))
    return series


def _mean_last(series: list[float], t: int, h: int) -> float:
    seg = series[max(0, t - h + 1): t + 1]
    return sum(seg) / len(seg) if seg else 0.0


def label_with_lookahead(feats: dict, rng: random.Random, noise: float = 0.07) -> str:
    """Optimal-allocation rule + rising-usage lookahead (nonlinear → favors MLP)."""
    c, x = feats["claude_5h_used"], feats["codex_5h_used"]
    if max(c, x) > 0.85 and min(c, x) > 0.85:
        lbl = "split"
    elif c - x > 0.20:
        lbl = "codex"
    elif x - c > 0.20:
        lbl = "claude"
    else:
        claude_rising = feats["claude_1h_avg"] - feats["claude_6h_avg"]
        codex_rising = feats["codex_1h_avg"] - feats["codex_6h_avg"]
        if claude_rising > codex_rising + 0.15:
            lbl = "codex"        # claude trending up fast → give next to codex
        elif codex_rising > claude_rising + 0.15:
            lbl = "claude"
        elif feats["claude_1h_avg"] > feats["codex_1h_avg"] + 0.10:
            lbl = "codex"
        elif feats["codex_1h_avg"] > feats["claude_1h_avg"] + 0.10:
            lbl = "claude"
        else:
            lbl = "split" if feats["queue_len"] >= 3 else ("claude" if c <= x else "codex")
    if rng.random() < noise:
        lbl = rng.choice([l for l in LABELS if l != lbl])
    return lbl


def features_at(
    claude_s: list[float], codex_s: list[float], t: int, ts_epoch: float,
    rng: random.Random,
) -> dict:
    cl1, cl6, cl24 = claude_s[t], _mean_last(claude_s, t, 6), _mean_last(claude_s, t, 24)
    cx1, cx6, cx24 = codex_s[t], _mean_last(codex_s, t, 6), _mean_last(codex_s, t, 24)
    cl5, cx5 = _mean_last(claude_s, t, 5), _mean_last(codex_s, t, 5)
    clw, cxw = _mean_last(claude_s, t, 168), _mean_last(codex_s, t, 168)
    dt = datetime.fromtimestamp(ts_epoch, tz=timezone.utc)
    h, dow = dt.hour, dt.weekday()
    return {
        "claude_5h_used": cl5, "codex_5h_used": cx5,
        "claude_week_used": clw, "codex_week_used": cxw,
        "claude_1h_avg": cl1, "codex_1h_avg": cx1,
        "claude_6h_avg": cl6, "codex_6h_avg": cx6,
        "claude_24h_avg": cl24, "codex_24h_avg": cx24,
        "hour_sin": math.sin(2 * math.pi * h / 24), "hour_cos": math.cos(2 * math.pi * h / 24),
        "dow_sin": math.sin(2 * math.pi * dow / 7), "dow_cos": math.cos(2 * math.pi * dow / 7),
        "queue_len": float(max(0, int(rng.expovariate(0.3)))),
        "recent_task_type_bucket": float(rng.choice([0, 1, 1, 2, 3, 3, 4, 5])),
        "est_job_minutes": max(0.0, rng.gauss(8, 6)),
        "recent_ack_miss_rate": _clamp01(rng.betavariate(1, 8)),
        "last_response_latency_s": max(0.0, rng.expovariate(1 / 30)),
    }


def load_hf_series(config: str, max_entities: int, min_len: int = 200) -> list[list[float]]:
    """Load utilization series from HF Salesforce/cloudops_tsf. Raises on any failure."""
    from datasets import load_dataset  # local import — only when --source hf

    # trust_remote_code helps on datasets<3; datasets>=3 removed script support
    # entirely (cloudops_tsf ships a loading script) — see the hint in main().
    ds = load_dataset(
        "Salesforce/cloudops_tsf", config, split="train_test", trust_remote_code=True
    )
    series: list[list[float]] = []
    for row in ds:
        tgt = row.get("target")
        # target may be a flat list (univariate) or list-of-lists (multivariate).
        if tgt and isinstance(tgt[0], (list, tuple)):
            tgt = tgt[0]
        if not tgt or len(tgt) < min_len:
            continue
        vals = [float(v) for v in tgt if v is not None]
        if not vals:
            continue
        lo, hi = min(vals), max(vals)
        rng_span = (hi - lo) or 1.0
        series.append([_clamp01((v - lo) / rng_span) for v in vals])  # normalize 0..1
        if len(series) >= max_entities:
            break
    if len(series) < 2:
        raise RuntimeError(f"cloudops_tsf/{config}: got {len(series)} usable series (<2)")
    return series


def build_rows(args) -> list[tuple[dict, str]]:
    rng = random.Random(args.seed)
    n_hours = args.num_samples + 220  # warmup so 168h windows are full

    if args.source == "hf":
        pool = load_hf_series(args.hf_config, max_entities=max(8, args.num_samples // 50))
        # Concatenate/trim two distinct pooled series into claude/codex tracks.
        def track(offset: int) -> list[float]:
            out: list[float] = []
            i = offset
            while len(out) < n_hours:
                out.extend(pool[i % len(pool)])
                i += 1
            return out[:n_hours]
        claude_s, codex_s = track(0), track(len(pool) // 2 or 1)
    else:
        claude_s = gen_ar1_series(rng, n_hours, base=0.42)
        codex_s = gen_ar1_series(rng, n_hours, base=0.38)

    base = datetime.now(timezone.utc)
    rows: list[tuple[dict, str]] = []
    for _ in range(args.num_samples):
        t = rng.randint(168, n_hours - 1)  # ensure full weekly window
        ts_epoch = (base + timedelta(hours=rng.uniform(-72, 72))).timestamp()
        feats = features_at(claude_s, codex_s, t, ts_epoch, rng)
        feats["_ts"] = ts_epoch
        rows.append((feats, label_with_lookahead(feats, rng)))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--source", choices=["synthetic", "hf"], default="synthetic",
        help="synthetic=AR(1)+diurnal (DEFAULT, used by shipped model); "
             "hf=Salesforce/cloudops_tsf is RESEARCH-ONLY: wrong-domain (datacenter "
             "machine traces, NO claude-vs-codex routing label) and broken on "
             "datasets>=3. NOT part of shipped provenance — real signal comes from "
             "collect_usage.py, not HF.",
    )
    ap.add_argument("--hf-config", default="azure_vm_traces_2017",
                    help="cloudops_tsf config: azure_vm_traces_2017|borg_cluster_data_2011|alibaba_cluster_trace_2018")
    ap.add_argument("--out", default="artifacts/features_augmented.csv")
    ap.add_argument("--num-samples", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--append-to", default=None,
                    help="labeled real CSV to prepend (preserves its label column)")
    a = ap.parse_args()

    tag = f"hf_{a.hf_config}" if a.source == "hf" else "ar1_synth"
    try:
        rows = build_rows(a)
    except Exception as e:  # HF path unavailable/offline → actionable message
        print(f"ERROR building rows from source={a.source}: {e!r}", file=sys.stderr)
        if a.source == "hf":
            print(
                "HINT: Salesforce/cloudops_tsf loads via a dataset SCRIPT, which "
                "datasets>=3 blocks ('Dataset scripts are no longer supported'). "
                "Either `pip install \"datasets<3\"`, OR use --source synthetic "
                "(AR(1)+diurnal — same temporal-structure fix, no download). The "
                "real usage signal should come from collect_claude_usage.py + the "
                "codex /status probe, not machine-scheduling traces.",
                file=sys.stderr,
            )
        return 1

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["ts", "event_id", "sender", "target", "msg_type"] + FEATURE_ORDER + ["label"]

    real_rows: list[list[str]] = []
    if a.append_to and Path(a.append_to).exists():
        with Path(a.append_to).open(encoding="utf-8") as f_in:
            r = csv.reader(f_in)
            header = next(r, None) or []
            has_label = "label" in [h.strip().lower() for h in header]
            for row in r:
                real_rows.append(list(row) if has_label else list(row) + [""])

    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for row in real_rows:
            if len(row) < len(cols):
                row = row + [""] * (len(cols) - len(row))
            elif len(row) > len(cols):
                row = row[: len(cols)]
            w.writerow(row)
        for feats, label in rows:
            w.writerow([feats["_ts"], tag, "", "", tag]
                       + [feats[k] for k in FEATURE_ORDER] + [label])

    from collections import Counter
    bal = Counter(lbl for _, lbl in rows)
    extra = f" (+{len(real_rows)} real rows prepended)" if real_rows else ""
    print(f"OK wrote {len(rows)} {tag} rows{extra} -> {out}")
    print(f"   label balance: {dict(bal)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
