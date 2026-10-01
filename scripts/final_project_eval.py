"""Final course-project evaluation — honest MLP-vs-SVM numbers for the Jun 9 talk.

Protocol (Codex plan review 7.8/10 PASS, 2026-06-07):
  PRIMARY  — TEMPORAL split. Real rows sorted by ts; first (1 - holdout) train,
             last `holdout` fraction is a 100%-REAL chronological holdout.
             5-min snapshots are strongly autocorrelated, so a random split
             leaks near-duplicate neighbors into the test set; the headline
             number must be the temporal one.
  GAP-FILL — synthetic rows are added to the TRAIN side only, and only for what
             real data lacks (split-class rows + codex-high usage quadrants).
             Tagged msg_type=ar1_synth; the fraction is reported everywhere.
  SECONDARY— course-pattern replication: stratified random
             train_test_split(train_size=0.6, random_state=42) on the same
             hybrid set (교재 프로그램 3-5) + 5-fold cross_val_score (프로그램
             3-6), with the real/synth test-subset breakdown disclosed.

Reuses the lecture-faithful pipelines from train_allocator_mlp.py (DRY):
StandardScaler→MLPClassifier(128,64,32, relu, adam, early_stopping) vs SVC(rbf).

Usage:
    python scripts/final_project_eval.py \
        --real artifacts/features_real_backfilled.csv \
        --synth artifacts/features_synth.csv \
        --out-dir artifacts/final_project
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_features import FEATURE_ORDER  # noqa: E402
from train_allocator_mlp import (  # noqa: E402
    LABELS,
    plot_confusion,
    train_mlp,
    train_svm,
)

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split  # noqa: E402
from sklearn.preprocessing import LabelEncoder  # noqa: E402


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if (r.get("label") or "") in LABELS]


def to_xy(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    X = np.array([[float(r[c]) for c in FEATURE_ORDER] for r in rows])
    y = np.array([r["label"] for r in rows])
    return X, y


def quadrant(r: dict) -> tuple[bool, bool]:
    return (float(r["claude_5h_used"]) > 0.5, float(r["codex_5h_used"]) > 0.5)


def pick_gap_fill(real_train: list[dict], synth: list[dict], *, per_gap: int,
                  seed: int) -> tuple[list[dict], dict]:
    """Minimum targeted synth: split-class rows + quadrants the real TRAIN lacks."""
    rng = np.random.default_rng(seed)
    have_quadrants = {quadrant(r) for r in real_train}
    have_labels = {r["label"] for r in real_train}
    chosen: list[dict] = []
    note: dict = {"per_gap": per_gap, "gaps": []}

    def sample(pool: list[dict], n: int) -> list[dict]:
        if len(pool) <= n:
            return list(pool)
        idx = rng.choice(len(pool), size=n, replace=False)
        return [pool[i] for i in idx]

    for lbl in LABELS:
        if lbl not in have_labels:
            pool = [r for r in synth if r["label"] == lbl]
            got = sample(pool, per_gap)
            chosen.extend(got)
            note["gaps"].append({"kind": "label", "value": lbl, "added": len(got)})
    chosen_ids = {id(r) for r in chosen}
    for q in [(False, True), (True, True), (True, False), (False, False)]:
        if q not in have_quadrants:
            pool = [r for r in synth if quadrant(r) == q and id(r) not in chosen_ids]
            got = sample(pool, per_gap)
            chosen.extend(got)
            chosen_ids.update(id(r) for r in got)
            note["gaps"].append({"kind": "quadrant", "value": list(q), "added": len(got)})
    return chosen, note


# --------------------------------------------------------------------------- #
# evaluation helpers
# --------------------------------------------------------------------------- #
def eval_model(model, enc: LabelEncoder, X, y_str) -> dict:
    yp = enc.inverse_transform(model.predict(X))
    cm = confusion_matrix(y_str, yp, labels=LABELS)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_str, yp, labels=LABELS, zero_division=0
    )
    return {
        "n": int(len(y_str)),
        "accuracy": float(accuracy_score(y_str, yp)),
        "confusion_matrix": cm.tolist(),
        "per_class": {
            lbl: {"precision": float(p), "recall": float(r),
                  "f1": float(f), "support": int(s)}
            for lbl, p, r, f, s in zip(LABELS, prec, rec, f1, support)
        },
    }


def subset_acc(model, enc, X, y_str, src, grp: str) -> dict:
    mask = src == grp
    n = int(mask.sum())
    if not n:
        return {"n": 0, "acc": None}
    yp = enc.inverse_transform(model.predict(X[mask]))
    return {"n": n, "acc": float(accuracy_score(y_str[mask], yp))}


def plot_accuracy_comparison(results: dict, out: Path) -> None:
    """Grouped bars: MLP vs SVM x {temporal real holdout (multi-seed), random, CV}."""
    groups = ["Temporal\n(real holdout)", "Random 60/40\n(hybrid)", "5-fold CV\n(hybrid)"]
    tg = results["temporal"]["variants"]["real_only"]  # headline variant
    mlp_vals = [tg["mlp"]["mean"],
                results["random"]["mlp"]["accuracy"],
                results["cv"]["mlp"]["mean"]]
    svm_vals = [tg["svm"]["mean"],
                results["random"]["svm"]["accuracy"],
                results["cv"]["svm"]["mean"]]
    mlp_err = [tg["mlp"]["std"], 0, results["cv"]["mlp"]["std"]]
    svm_err = [tg["svm"]["std"], 0, results["cv"]["svm"]["std"]]
    x = np.arange(len(groups))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.2))
    b1 = ax.bar(x - w / 2, mlp_vals, w, yerr=mlp_err, capsize=4,
                label="MLP", color="#1f77b4")
    b2 = ax.bar(x + w / 2, svm_vals, w, yerr=svm_err, capsize=4,
                label="SVM", color="#ff7f0e")
    for bars, vals in ((b1, mlp_vals), (b2, svm_vals)):
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}",
                    ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Accuracy")
    ax.set_title("MLP vs SVM — allocation accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default="artifacts/features_real_backfilled.csv")
    ap.add_argument("--synth", default="artifacts/features_augmented.csv",
                    help="pool filtered to msg_type=ar1_synth — the shipped model's "
                         "synthetic provenance (features_synth.csv is the older "
                         "generator tagged 'synthetic' and is NOT src-recognized)")
    ap.add_argument("--out-dir", default="artifacts/final_project")
    ap.add_argument("--report", default="claudedocs/final_project_report.md")
    ap.add_argument("--holdout", type=float, default=0.3,
                    help="temporal holdout fraction of the real timeline (0.25-0.4)")
    ap.add_argument("--per-gap", type=int, default=75,
                    help="synth rows added per missing label/quadrant gap")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-seeds", type=int, default=5,
                    help="temporal protocol repeats (seed..seed+n-1) for mean±std")
    a = ap.parse_args()

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    real = load_csv(Path(a.real))
    synth = [r for r in load_csv(Path(a.synth))
             if (r.get("msg_type") or "").startswith(("ar1", "hf_"))]
    if not synth:
        print(f"ERROR no ar1_synth rows in {a.synth}")
        return 1
    real.sort(key=lambda r: float(r["ts"]))
    enc = LabelEncoder().fit(LABELS)

    # ---------------- PRIMARY: temporal split ----------------
    # Single-seed temporal numbers proved UNSTABLE (sweep 2026-06-07: MLP
    # 0.57-0.88, SVM 0.41-0.72 across holdout/per-gap choices), so the headline
    # is a MULTI-SEED mean±std at fixed protocol, with a real-only-train
    # ablation to show what the synth gap-fill costs/buys on the real holdout.
    n_hold = int(round(len(real) * a.holdout))
    real_train, real_hold = real[:-n_hold], real[-n_hold:]
    Xho, yho = to_xy(real_hold)
    seeds = [a.seed + i for i in range(a.n_seeds)]

    def temporal_run(seed: int, with_fill: bool) -> dict:
        if with_fill:
            fill, note = pick_gap_fill(real_train, synth, per_gap=a.per_gap, seed=seed)
        else:
            fill, note = [], {"per_gap": 0, "gaps": []}
        train_rows = real_train + fill
        Xtr, ytr = to_xy(train_rows)
        ytr_int = enc.transform(ytr)
        out = {"seed": seed, "n_fill": len(fill),
               "synth_fraction_of_train": round(len(fill) / len(train_rows), 4),
               "train_label_balance": dict(Counter(ytr)), "fill_note": note}
        for name, factory in (("mlp", train_mlp), ("svm", train_svm)):
            out[name] = eval_model(factory(Xtr, ytr_int, seed=seed), enc, Xho, yho)
        return out

    variants: dict[str, dict] = {}
    for vname, with_fill in (("gapfill", True), ("real_only", False)):
        runs = [temporal_run(s, with_fill) for s in seeds]
        agg = {}
        for name in ("mlp", "svm"):
            accs = np.array([r[name]["accuracy"] for r in runs])
            agg[name] = {"mean": float(accs.mean()), "std": float(accs.std()),
                         "per_seed": [float(x) for x in accs]}
        variants[vname] = {"agg": agg, "seed0_detail": runs[0]}

    # HEADLINE = real_only: the ablation showed synth gap-fill HURTS and
    # destabilizes the real holdout (MLP 0.69±0.16 vs 0.87±0.004 real-only) —
    # synth coverage of never-observed regions (split/codex-high) is a
    # deployment-coverage tradeoff, reported via the gapfill variant + CV.
    detail = variants["real_only"]["seed0_detail"]
    gf_detail = variants["gapfill"]["seed0_detail"]
    # ZeroR sanity floor — the holdout is codex-heavy, so "always predict the
    # majority class" is the number every model must beat to claim any skill.
    majority_baseline = max(Counter(yho).values()) / len(yho)
    temporal = {
        "protocol": (
            f"chronological: first {len(real_train)} real rows -> train; "
            f"last {n_hold} real rows -> 100%-real holdout; "
            f"{a.n_seeds} seeds {seeds}; headline variant = real_only train, "
            f"'gapfill' variant adds targeted synth (train-side only)"
        ),
        "holdout_label_balance": dict(Counter(yho)),
        "majority_baseline": round(float(majority_baseline), 4),
        "variants": {v: variants[v]["agg"] for v in variants},
        "seed0_train_label_balance": detail["train_label_balance"],
        "gapfill_seed0_synth_fill": gf_detail["fill_note"],
        "synth_fraction_of_train": gf_detail["synth_fraction_of_train"],
        "mlp": detail["mlp"],   # real_only seed0 detail (CM, per-class) — headline
        "svm": detail["svm"],
        "gapfill_seed0": {"mlp": gf_detail["mlp"], "svm": gf_detail["svm"]},
    }
    plot_confusion(temporal["mlp"]["confusion_matrix"], out_dir / "cm_mlp_temporal.png",
                   title="MLP — temporal real holdout (real-only train)")
    plot_confusion(temporal["svm"]["confusion_matrix"], out_dir / "cm_svm_temporal.png",
                   title="SVM — temporal real holdout (real-only train)")

    # ---------------- SECONDARY: course-pattern random split ----------------
    # deterministic seed0 gap-fill (same sample as the gapfill variant's seed0 run)
    fill, _ = pick_gap_fill(real_train, synth, per_gap=a.per_gap, seed=a.seed)
    hybrid = real + fill  # same disclosed gap-fill, now in the shuffled pool
    Xh, yh = to_xy(hybrid)
    src = np.array(["synth" if (r.get("msg_type") or "").startswith(("ar1", "hf_"))
                    else "real" for r in hybrid])
    yh_int = enc.transform(yh)
    Xtr2, Xte2, ytr2, yte2, src_tr, src_te = train_test_split(
        Xh, yh_int, src, train_size=0.6, random_state=a.seed, stratify=yh_int
    )
    yte2_str = enc.inverse_transform(yte2)
    mlp_r = train_mlp(Xtr2, ytr2, seed=a.seed)
    svm_r = train_svm(Xtr2, ytr2, seed=a.seed)
    random_split = {
        "protocol": "stratified train_test_split(train_size=0.6, random_state=42) on "
                    "hybrid (course 프로그램 3-5 pattern; leakage-prone — secondary only)",
        "mlp": eval_model(mlp_r, enc, Xte2, yte2_str),
        "svm": eval_model(svm_r, enc, Xte2, yte2_str),
        "mlp_subset": {g: subset_acc(mlp_r, enc, Xte2, yte2_str, src_te, g)
                       for g in ("real", "synth")},
        "svm_subset": {g: subset_acc(svm_r, enc, Xte2, yte2_str, src_te, g)
                       for g in ("real", "synth")},
    }
    plot_confusion(random_split["mlp"]["confusion_matrix"], out_dir / "cm_mlp_random.png",
                   title="MLP — random 60/40 (hybrid)")

    # ---------------- 5-fold CV (course 프로그램 3-6 pattern) ----------------
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=a.seed)
    cv = {}
    # tiny balanced prototype fit (a few rows of EVERY class) just to obtain the
    # lecture-faithful pipeline config from train_*; clone() strips the fit
    proto_idx = np.concatenate([np.where(yh_int == c)[0][:4] for c in np.unique(yh_int)])
    for name, factory in (("mlp", train_mlp), ("svm", train_svm)):
        from sklearn.base import clone
        proto = factory(Xh[proto_idx], yh_int[proto_idx], seed=a.seed)
        scores = cross_val_score(clone(proto), Xh, yh_int, cv=skf, scoring="accuracy")
        cv[name] = {"scores": [float(s) for s in scores],
                    "mean": float(scores.mean()), "std": float(scores.std())}

    results = {"temporal": temporal, "random": random_split, "cv": cv,
               "data": {
                   "real_rows": len(real), "synth_pool": len(synth),
                   "real_label_balance": dict(Counter(r["label"] for r in real)),
                   "holdout_fraction": a.holdout, "per_gap": a.per_gap,
                   "seed": a.seed,
                   "real_csv": str(a.real), "synth_csv": str(a.synth),
               }}
    plot_accuracy_comparison(results, out_dir / "accuracy_comparison.png")
    (out_dir / "metrics.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---------------- report ----------------
    rep = Path(a.report)
    rep.parent.mkdir(parents=True, exist_ok=True)

    def cm_md(cm: list[list[int]]) -> str:
        head = "| true\\pred | " + " | ".join(LABELS) + " |\n|---|" + "---|" * len(LABELS)
        body = "\n".join(
            f"| **{LABELS[i]}** | " + " | ".join(str(v) for v in row) + " |"
            for i, row in enumerate(cm)
        )
        return head + "\n" + body

    lines = [
        "# Final Project — MLP vs SVM 평가 결과",
        "",
        f"- real rows: {len(real)} (backfilled fixed-window), synth pool: {len(synth)}",
        f"- real label balance: {results['data']['real_label_balance']}",
        f"- temporal protocol: {temporal['protocol']}",
        f"- synth fraction of temporal train: {temporal['synth_fraction_of_train']:.1%}",
        "",
        "## 정확률 비교표",
        "",
        "| 모델 | 시간 분할 실데이터 홀드아웃, real-only 학습 (5시드 mean±std) **[헤드라인]** "
        "| 〃 +합성 갭필 변형 | 무작위 60/40 (하이브리드) | 5-fold CV (하이브리드) |",
        "|---|---|---|---|---|",
        f"| MLP | **{temporal['variants']['real_only']['mlp']['mean']:.3f} ± "
        f"{temporal['variants']['real_only']['mlp']['std']:.3f}** "
        f"| {temporal['variants']['gapfill']['mlp']['mean']:.3f} ± "
        f"{temporal['variants']['gapfill']['mlp']['std']:.3f} "
        f"| {random_split['mlp']['accuracy']:.4f} "
        f"(real {random_split['mlp_subset']['real']['acc']:.4f} / synth {random_split['mlp_subset']['synth']['acc']:.4f}) "
        f"| {cv['mlp']['mean']:.4f} ± {cv['mlp']['std']:.4f} |",
        f"| SVM | **{temporal['variants']['real_only']['svm']['mean']:.3f} ± "
        f"{temporal['variants']['real_only']['svm']['std']:.3f}** "
        f"| {temporal['variants']['gapfill']['svm']['mean']:.3f} ± "
        f"{temporal['variants']['gapfill']['svm']['std']:.3f} "
        f"| {random_split['svm']['accuracy']:.4f} "
        f"(real {random_split['svm_subset']['real']['acc']:.4f} / synth {random_split['svm_subset']['synth']['acc']:.4f}) "
        f"| {cv['svm']['mean']:.4f} ± {cv['svm']['std']:.4f} |",
        f"| (베이스라인) 다수 클래스 | {temporal['majority_baseline']:.3f} | — | — | — |",
        "",
        "## 혼동행렬 — 시간 분할 실데이터 홀드아웃",
        "",
        f"holdout 라벨 분포: {temporal['holdout_label_balance']} "
        "(⚠️ split 클래스는 실데이터에 없음 — split 일반화는 보조 수치로만 평가)",
        "",
        "### MLP",
        cm_md(temporal["mlp"]["confusion_matrix"]),
        "",
        "### SVM",
        cm_md(temporal["svm"]["confusion_matrix"]),
        "",
        "## 클래스별 정밀도/재현율 (MLP, temporal)",
        "",
        "| class | precision | recall | f1 | support |",
        "|---|---|---|---|---|",
    ]
    for lbl, d in temporal["mlp"]["per_class"].items():
        lines.append(f"| {lbl} | {d['precision']:.3f} | {d['recall']:.3f} "
                     f"| {d['f1']:.3f} | {d['support']} |")
    lines += [
        "",
        "## 정직성 공개 (발표 멘트에 반영)",
        "- 헤드라인 수치는 **시간 기반 분할 + 실데이터만 학습**: 5분 간격 스냅샷의 자기상관 때문에 "
        "무작위 분할은 이웃 행 누수로 정확률이 부풀 수 있음 (무작위 60/40·CV는 수업 패턴 재현용 보조 — "
        "실제로 0.90~0.94로 더 높게 나오는 것 자체가 누수의 증거).",
        "- **합성 갭필 ablation**: 실데이터가 못 본 영역(split 클래스·codex 고사용 사분면)을 합성으로 "
        f"채우면(train의 {temporal['synth_fraction_of_train']:.1%}) 실측 홀드아웃 정확률이 떨어지고 "
        "분산이 커짐 (MLP 0.87→0.69±0.16). 미관측 상황 대응력과 실측 구간 성능 사이의 트레이드오프이며, "
        "홀드아웃은 항상 100% 실데이터.",
        f"- 다수 클래스(ZeroR) 베이스라인 {temporal['majority_baseline']:.3f} 대비 MLP "
        f"+{temporal['variants']['real_only']['mlp']['mean'] - temporal['majority_baseline']:.3f}: "
        f"홀드아웃이 codex 편중이라 정확률만으로는 과대평가 — claude recall"
        f"({temporal['mlp']['per_class']['claude']['recall']:.2f})과 precision"
        f"({temporal['mlp']['per_class']['claude']['precision']:.2f})의 비대칭"
        "(보수적 claude 라우팅)을 함께 봐야 함.",
        "- 시간 홀드아웃에는 split 클래스가 없음 — split 일반화는 CV 보조 수치로만 평가 가능.",
        "- claude 사용량은 보정된 토큰 추정치(fixed-window backfill, 검증 게이트 통과 "
        "worst |Δused|=0.0017), codex 사용량은 rollout 실측.",
    ]
    rep.write_text("\n".join(lines) + "\n", encoding="utf-8")

    tg, tr = temporal["variants"]["gapfill"], temporal["variants"]["real_only"]
    print(json.dumps({
        "temporal_mlp_gapfill": f"{tg['mlp']['mean']:.4f}±{tg['mlp']['std']:.4f}",
        "temporal_svm_gapfill": f"{tg['svm']['mean']:.4f}±{tg['svm']['std']:.4f}",
        "temporal_mlp_realonly": f"{tr['mlp']['mean']:.4f}±{tr['mlp']['std']:.4f}",
        "temporal_svm_realonly": f"{tr['svm']['mean']:.4f}±{tr['svm']['std']:.4f}",
        "random_mlp": random_split["mlp"]["accuracy"],
        "random_svm": random_split["svm"]["accuracy"],
        "cv_mlp": f"{cv['mlp']['mean']:.4f}±{cv['mlp']['std']:.4f}",
        "cv_svm": f"{cv['svm']['mean']:.4f}±{cv['svm']['std']:.4f}",
        "synth_fraction_of_train": temporal["synth_fraction_of_train"],
        "out": str(out_dir),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
