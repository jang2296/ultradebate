"""Apply heuristic post-hoc labels to real feature CSV.

Same rule as the synthetic generator, applied to real rows. If you have
ground-truth labels (manual 200-300 entries), prefer those - see the label
mapping section in the README. This script just bootstraps a labeled set
from heuristics so the MLP can train on a unified CSV.

Usage:
    python scripts/label_real_features.py \
        --in artifacts/features_real.csv \
        --out artifacts/features_real_labeled.csv
"""
from __future__ import annotations
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_features import FEATURE_ORDER  # noqa: E402


def label_row(feats: dict[str, float]) -> str:
    c = feats.get("claude_5h_used", 0.0)
    x = feats.get("codex_5h_used", 0.0)
    if max(c, x) > 0.85 and min(c, x) > 0.85:
        return "split"
    if c - x > 0.20:
        return "codex"
    if x - c > 0.20:
        return "claude"
    if feats.get("claude_1h_avg", 0) > feats.get("codex_1h_avg", 0) + 0.10:
        return "codex"
    if feats.get("codex_1h_avg", 0) > feats.get("claude_1h_avg", 0) + 0.10:
        return "claude"
    if feats.get("queue_len", 0) >= 3:
        return "split"
    # Near-equal/near-zero usage carries no routing signal. The old
    # `claude if c<=x` made every zero-usage real row "claude" — a degenerate
    # constant label that poisoned the real training set (Codex audit 2026-06-03:
    # all 134 real rows were label=claude). Treat a true tie as 'split'; rows with
    # no usage signal at all should be EXCLUDED upstream (see build_real_dataset.py).
    if abs(c - x) < 1e-6:
        return "split"
    return "claude" if c < x else "codex"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="artifacts/features_real.csv")
    ap.add_argument("--out", default="artifacts/features_real_labeled.csv")
    a = ap.parse_args()

    src = Path(a.inp)
    if not src.exists():
        print(f"ERROR input not found: {src}")
        return 1

    counter: Counter[str] = Counter()
    with src.open(encoding="utf-8") as f_in:
        rows = list(csv.reader(f_in))
    if not rows:
        print("ERROR empty input")
        return 1

    header = rows[0]
    out_header = header + ["label"]
    out_rows: list[list[str]] = [out_header]

    feat_idx = {name: header.index(name) for name in FEATURE_ORDER if name in header}
    for row in rows[1:]:
        feats = {name: float(row[i]) for name, i in feat_idx.items()}
        lbl = label_row(feats)
        counter[lbl] += 1
        out_rows.append(row + [lbl])

    dst = Path(a.out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", newline="", encoding="utf-8") as f_out:
        csv.writer(f_out).writerows(out_rows)

    total = sum(counter.values())
    balance = ", ".join(f"{k}={v} ({v / max(1, total):.1%})" for k, v in counter.most_common())
    print(f"OK Labeled {total} real rows -> {dst}")
    print(f"   class balance: {balance}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
