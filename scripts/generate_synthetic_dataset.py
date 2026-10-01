"""Synthetic dataset generator for allocator MLP training.

Why synthetic: real op logs have only ~135 events (insufficient for ML - see
design doc §5.2). We augment with simulated days drawn from plausible usage
patterns. The slide MUST disclose this - never claim "real production data".

Strategy (deterministic given seed):
  * Simulate N hours of dual-agent operation.
  * For each hour: draw claude/codex token usage from a peak-hour-modulated
    distribution. Inject correlated bursts (one agent runs long jobs -> that
    agent's usage spikes for 1-2h).
  * Compute the 19 features at sampled timestamps.
  * Assign label by the deterministic optimal-allocation rule:
      if claude_5h_used > 0.7 and codex_5h_used < 0.5 -> "codex"
      if codex_5h_used > 0.7 and claude_5h_used < 0.5 -> "claude"
      if max(c,x) > 0.85 and min(c,x) > 0.85 -> "split"
      else -> softer rule based on 1h_avg + queue_len
  * Add label noise (5-10%) so MLP can't trivially recover the rule via SVM ->
    motivates non-linearity (§5.5 SVM baseline comparison).

Usage:
    python scripts/generate_synthetic_dataset.py \
        --out artifacts/features_synth.csv \
        --num-samples 1500 --seed 42

Combine with real:
    python scripts/generate_synthetic_dataset.py --append-to artifacts/features_real.csv \
        --out artifacts/features_combined.csv --num-samples 1500
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


def peak_hour_factor(h: int) -> float:
    if 9 <= h <= 12:
        return 1.4
    if 14 <= h <= 18:
        return 1.6
    if 19 <= h <= 22:
        return 1.1
    return 0.4


def simulate_sample(rng: random.Random, base_time: datetime) -> tuple[dict, str]:
    h = rng.choice(range(24))
    dow = rng.choice(range(7))
    pf = peak_hour_factor(h) * (0.7 if dow >= 5 else 1.0)

    claude_5h = max(0.0, min(1.0, rng.betavariate(2, 3) * pf + rng.gauss(0, 0.08)))
    codex_5h = max(0.0, min(1.0, rng.betavariate(2, 3) * pf + rng.gauss(0, 0.08)))

    def jitter(x: float) -> float:
        return max(0.0, min(1.0, x + rng.gauss(0, 0.10)))

    feats = {
        "claude_5h_used": claude_5h, "codex_5h_used": codex_5h,
        "claude_week_used": min(1.0, claude_5h * 1.4 + rng.gauss(0, 0.05)),
        "codex_week_used": min(1.0, codex_5h * 1.4 + rng.gauss(0, 0.05)),
        "claude_1h_avg": jitter(claude_5h),
        "codex_1h_avg": jitter(codex_5h),
        "claude_6h_avg": jitter(claude_5h),
        "codex_6h_avg": jitter(codex_5h),
        "claude_24h_avg": jitter(claude_5h * 0.9),
        "codex_24h_avg": jitter(codex_5h * 0.9),
        "hour_sin": math.sin(2 * math.pi * h / 24),
        "hour_cos": math.cos(2 * math.pi * h / 24),
        "dow_sin": math.sin(2 * math.pi * dow / 7),
        "dow_cos": math.cos(2 * math.pi * dow / 7),
        "queue_len": float(max(0, int(rng.expovariate(0.3)))),
        "recent_task_type_bucket": float(rng.choice([0, 1, 1, 2, 3, 3, 4, 5])),
        "est_job_minutes": max(0.0, rng.gauss(8, 6)),
        "recent_ack_miss_rate": max(0.0, min(1.0, rng.betavariate(1, 8))),
        "last_response_latency_s": max(0.0, rng.expovariate(1 / 30)),
    }

    c, x = feats["claude_5h_used"], feats["codex_5h_used"]
    if max(c, x) > 0.85 and min(c, x) > 0.85:
        label = "split"
    elif c - x > 0.20:
        label = "codex"
    elif x - c > 0.20:
        label = "claude"
    else:
        if feats["claude_1h_avg"] > feats["codex_1h_avg"] + 0.10:
            label = "codex"
        elif feats["codex_1h_avg"] > feats["claude_1h_avg"] + 0.10:
            label = "claude"
        else:
            label = "split" if feats["queue_len"] >= 3 else rng.choice(["claude", "codex"])

    if rng.random() < 0.07:
        label = rng.choice([l for l in LABELS if l != label])

    ts = (base_time + timedelta(hours=rng.uniform(-72, 72))).timestamp()
    feats["_ts"] = ts
    feats["_label"] = label
    return feats, label


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Combine real labeled features + synthetic samples into a training CSV. "
            "Example: python scripts/generate_synthetic_dataset.py "
            "--out artifacts/features_combined.csv "
            "--append-to artifacts/features_real_labeled.csv --num-samples 1500"
        )
    )
    ap.add_argument("--out", default="artifacts/features_synth.csv")
    ap.add_argument("--num-samples", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--append-to", default=None,
        help=(
            "Real-feature CSV to prepend before synth rows. If the CSV already "
            "has a 'label' column (e.g. features_real_labeled.csv), its labels "
            "are preserved verbatim. If it lacks 'label', an empty label is "
            "appended so column counts match."
        ),
    )
    a = ap.parse_args()

    rng = random.Random(a.seed)
    base = datetime.now(timezone.utc)
    rows = [simulate_sample(rng, base) for _ in range(a.num_samples)]

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["ts", "event_id", "sender", "target", "msg_type"] + FEATURE_ORDER + ["label"]

    # v2.6 fix (audit gap #3): real CSV가 이미 label 컬럼이 있으면 빈 값을
    # extra append하지 않는다. 이전 코드는 무조건 `row + [""]`로 25→26 컬럼
    # 짜리 malformed row를 생성했음.
    real_rows: list[list[str]] = []
    real_has_label = False
    if a.append_to and Path(a.append_to).exists():
        with Path(a.append_to).open(encoding="utf-8") as f_in:
            r = csv.reader(f_in)
            real_header = next(r, None) or []
            real_has_label = "label" in [h.strip().lower() for h in real_header]
            for row in r:
                if real_has_label:
                    # 이미 label 컬럼 있음 — 그대로 사용 (label 위치가 맨 끝이라고 가정)
                    real_rows.append(list(row))
                else:
                    # label 없음 — 합본 형식에 맞추기 위해 빈 라벨 append
                    real_rows.append(list(row) + [""])

    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for row in real_rows:
            # 안전 가드: real header에 label이 있었더라도 실제 row의 컬럼 수가
            # 합본 cols 길이와 다르면 패딩/잘라내기 (방어적).
            if len(row) < len(cols):
                row = row + [""] * (len(cols) - len(row))
            elif len(row) > len(cols):
                row = row[: len(cols)]
            w.writerow(row)
        for feats, label in rows:
            w.writerow([feats["_ts"], "synth", "", "", "synthetic"]
                       + [feats[k] for k in FEATURE_ORDER] + [label])

    label_state = "labeled" if real_has_label else "unlabeled"
    extra = (
        f" (+{len(real_rows)} {label_state} real rows prepended)"
        if real_rows else ""
    )
    print(f"OK Wrote {len(rows)} synthetic samples{extra} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
