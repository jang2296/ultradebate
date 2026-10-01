"""Train allocator MLPClassifier — final project core artifact.

Strategy (design doc §5.5):
  Primary model: sklearn MLPClassifier(hidden=[50,50], early_stopping=True)
  Baseline:      sklearn SVC (matches lecture §4.6.2 sklearn handwriting example)
  Comparison:    accuracy + per-class F1 + confusion matrix + bar chart

Strict mapping to lecture materials:
  - 3rd week (MLP video lecture): MLPClassifier, StandardScaler, make_pipeline,
    early_stopping, train_test_split (random_state=42)
  - 5th week §4.6.2: SVC baseline for MLP-vs-SVM comparison

Usage:
    python scripts/train_allocator_mlp.py \\
        --data artifacts/features_synth.csv \\
        --out-model artifacts/allocator_mlp.joblib \\
        --out-plots artifacts/plots \\
        --seed 42
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path


def _sha256_of_file(path: Path) -> str:
    """Stream sha256 for the model file. Recorded in meta for SECURITY_REVIEW_V26 F1."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

# extract_features.py를 sibling import
sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_features import FEATURE_ORDER  # noqa: E402

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless 환경 대응
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score,
)
import joblib

LABELS = ["claude", "codex", "split"]


def load_data(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CSV → (X, y, src). src[i]='synth' for ar1_synth/hf_* tagged rows else 'real'
    — drives the honest real-vs-synth test-subset breakdown (Codex audit 2026-06-03)."""
    X_rows: list[list[float]] = []
    y_rows: list[str] = []
    src_rows: list[str] = []
    with path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            label = (row.get("label") or "").strip()
            if not label or label not in LABELS:
                continue
            try:
                x = [float(row[c]) for c in FEATURE_ORDER]
            except (KeyError, ValueError):
                continue
            X_rows.append(x)
            y_rows.append(label)
            tag = (row.get("msg_type") or "").strip()
            src_rows.append("synth" if (tag == "ar1_synth" or tag.startswith("hf_")) else "real")
    return np.array(X_rows), np.array(y_rows), np.array(src_rows)


def train_mlp(X_train, y_train, *, seed: int,
              hidden_layers=(128, 64, 32), alpha: float = 1e-2,
              lr_init: float = 3e-3):
    """MLPClassifier — lecture stack (3주차) with CV-tuned capacity (2026-06-03).

    Same pipeline as the lecture example (StandardScaler → MLPClassifier, ReLU,
    adam, early_stopping). The architecture was upgraded from the design-doc
    starting point [50,50] to (128,64,32) after a GridSearchCV (cv=3,
    scoring=f1_macro) on the augmented temporal dataset (AR(1)+diurnal usage +
    rising-usage lookahead labels). On that data the tuned MLP edges the rbf-SVM
    baseline on accuracy (0.814 vs 0.799). Only capacity/regularization changed;
    the method stays faithful to the lecture material.
    """
    pipe = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=tuple(hidden_layers),
            activation="relu",
            solver="adam",
            alpha=alpha,
            learning_rate_init=lr_init,
            max_iter=1500,
            early_stopping=True,
            validation_fraction=0.1,  # sklearn default; CV-selected config uses it
            n_iter_no_change=25,
            random_state=seed,
        ),
    )
    pipe.fit(X_train, y_train)
    return pipe


def train_svm(X_train, y_train, *, seed: int):
    """SVC baseline — 5주차 §4.6.2 sklearn 손글씨 예제 매핑."""
    return make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", random_state=seed),
    ).fit(X_train, y_train)


def evaluate(name: str, model, X_test, y_test_str, encoder: LabelEncoder) -> dict:
    """model이 int를 반환하므로 encoder로 string label 복원 후 평가."""
    yp_int = model.predict(X_test)
    yp = encoder.inverse_transform(yp_int)
    acc = accuracy_score(y_test_str, yp)
    cm = confusion_matrix(y_test_str, yp, labels=LABELS)
    f1_per = f1_score(y_test_str, yp, labels=LABELS, average=None, zero_division=0)
    report = classification_report(y_test_str, yp, labels=LABELS, zero_division=0)
    return {
        "name": name,
        "accuracy": float(acc),
        "confusion_matrix": cm.tolist(),
        "f1_per_class": dict(zip(LABELS, [float(x) for x in f1_per])),
        "report_text": report,
    }


def subset_breakdown(model, encoder: LabelEncoder, X_test, y_test_str, src_test) -> dict:
    """Honest real-vs-synth test accuracy. Codex audit 2026-06-03: zero-usage real
    rows are trivially-correct freebies that INFLATE the headline accuracy — this
    surfaces the synth-only number (the meaningful one)."""
    yp = encoder.inverse_transform(model.predict(X_test))
    out = {"overall": {"n": int(len(y_test_str)), "acc": float(accuracy_score(y_test_str, yp))}}
    for grp in ("real", "synth"):
        mask = src_test == grp
        n = int(mask.sum())
        out[grp] = {"n": n, "acc": float(accuracy_score(y_test_str[mask], yp[mask])) if n else None}
    return out


def plot_confusion(cm, out: Path, *, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    arr = np.array(cm)
    im = ax.imshow(arr, cmap="Blues")
    ax.set_xticks(range(len(LABELS)))
    ax.set_yticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS)
    ax.set_yticklabels(LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    threshold = arr.max() / 2 if arr.max() > 0 else 1
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(
                j, i, int(arr[i, j]),
                ha="center", va="center",
                color="white" if arr[i, j] > threshold else "black",
            )
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_comparison(mlp_acc: float, svm_acc: float, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(
        ["MLP", "SVM"], [mlp_acc, svm_acc],
        color=["#1f77b4", "#ff7f0e"],
    )
    for b, v in zip(bars, [mlp_acc, svm_acc]):
        ax.text(
            b.get_x() + b.get_width() / 2, v + 0.01,
            f"{v:.3f}", ha="center", fontsize=11,
        )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Test Accuracy")
    ax.set_title("MLP vs SVM (allocator)")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_loss_curve(mlp_pipe, out: Path) -> None:
    # pipeline 안의 MLPClassifier를 named_steps로 접근
    mlp = mlp_pipe.named_steps["mlpclassifier"]
    if not hasattr(mlp, "loss_curve_"):
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(mlp.loss_curve_, label="train loss", color="#1f77b4")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss", color="#1f77b4")
    if hasattr(mlp, "validation_scores_") and mlp.validation_scores_ is not None:
        ax2 = ax.twinx()
        ax2.plot(mlp.validation_scores_, color="orange", label="val acc")
        ax2.set_ylabel("val accuracy", color="orange")
    ax.set_title("MLP training curve")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def write_report(
    out: Path,
    *,
    mlp_result: dict,
    svm_result: dict,
    mlp_sub: dict,
    svm_sub: dict,
    n_train: int,
    n_test: int,
    classes: list[str],
) -> None:
    def _row(name: str, sub: dict) -> str:
        r, s, o = sub["real"], sub["synth"], sub["overall"]
        ra = f"{r['acc']:.4f}" if r["acc"] is not None else "n/a"
        sa = f"{s['acc']:.4f}" if s["acc"] is not None else "n/a"
        return f"| {name} | {o['acc']:.4f} | {ra} (n={r['n']}) | {sa} (n={s['n']}) |"

    txt = f"""# Training Report — Allocator MLP

## Dataset
- Train: {n_train}  Test: {n_test}
- Classes: {classes}

## MLPClassifier (hidden=[50,50], early_stopping)
- Accuracy: **{mlp_result['accuracy']:.4f}**
- F1 per class: {mlp_result['f1_per_class']}

```
{mlp_result['report_text']}
```

## SVC baseline (rbf kernel)
- Accuracy: **{svm_result['accuracy']:.4f}**
- F1 per class: {svm_result['f1_per_class']}

```
{svm_result['report_text']}
```

## Comparison
| Model | Accuracy |
|-------|----------|
| MLP   | {mlp_result['accuracy']:.4f} |
| SVM   | {svm_result['accuracy']:.4f} |
| Delta | {mlp_result['accuracy']-svm_result['accuracy']:+.4f} |

## ⚠️ Honest subset breakdown (real vs synth test rows)
Zero-usage legacy "real" rows are trivially correct for BOTH models and inflate the
headline. The **synth-subset** column is the meaningful comparison.

| Model | Overall | Real-subset | Synth-subset |
|-------|---------|-------------|--------------|
{_row("MLP", mlp_sub)}
{_row("SVM", svm_sub)}

> Disclosure: synthetic-bootstrap / real-state policy-distillation model, NOT a
> production-validated balancer. Real usage is now collected via collect_usage.py
> for a future retrain on real state.
"""
    out.write_text(txt)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="artifacts/features_synth.csv")
    ap.add_argument("--out-model", default="artifacts/allocator_mlp.joblib")
    ap.add_argument("--out-meta", default="artifacts/allocator_meta.json")
    ap.add_argument("--out-plots", default="artifacts/plots")
    ap.add_argument("--out-report", default="artifacts/training_report.md")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-size", type=float, default=0.25)
    ap.add_argument("--hidden", default="128,64,32",
                    help="comma-separated MLP hidden layer sizes (CV-tuned default)")
    ap.add_argument("--alpha", type=float, default=1e-2, help="MLP L2 regularization")
    ap.add_argument("--lr", type=float, default=3e-3, help="MLP learning_rate_init")
    a = ap.parse_args()

    data_path = Path(a.data)
    if not data_path.exists():
        print(f"ERROR: {data_path} not found. Run generate_synthetic_dataset.py first.")
        return 1

    X, y, src = load_data(data_path)
    if len(X) < 50:
        print(f"ERROR: only {len(X)} labeled samples - need >=50")
        return 1
    from collections import Counter as _C
    print(f"Loaded {len(X)} samples, {X.shape[1]} features, classes={sorted(set(y.tolist()))}, "
          f"src={dict(_C(src.tolist()))}")

    # 단일 클래스만 존재하면 stratify 불가
    unique_classes = sorted(set(y.tolist()))
    stratify_arg = y if len(unique_classes) > 1 else None
    X_train, X_test, y_train, y_test, src_train, src_test = train_test_split(
        X, y, src, test_size=a.test_size, random_state=a.seed, stratify=stratify_arg,
    )

    # sklearn 1.6 MLPClassifier(early_stopping=True)는 string 라벨에서 _score 시
    # np.isnan(y_pred)를 호출하여 TypeError 발생. LabelEncoder로 int 변환 우회.
    encoder = LabelEncoder()
    encoder.fit(LABELS)  # 클래스 순서 고정
    y_train_enc = encoder.transform(y_train)

    print("Training MLP...")
    _hidden = tuple(int(x) for x in str(a.hidden).split(",") if x.strip())
    mlp = train_mlp(X_train, y_train_enc, seed=a.seed,
                    hidden_layers=_hidden, alpha=a.alpha, lr_init=a.lr)
    print("Training SVM baseline...")
    svm = train_svm(X_train, y_train_enc, seed=a.seed)

    mlp_r = evaluate("MLP", mlp, X_test, y_test, encoder)
    svm_r = evaluate("SVM", svm, X_test, y_test, encoder)
    mlp_sub = subset_breakdown(mlp, encoder, X_test, y_test, src_test)
    svm_sub = subset_breakdown(svm, encoder, X_test, y_test, src_test)
    print(f"\nMLP acc={mlp_r['accuracy']:.4f}  SVM acc={svm_r['accuracy']:.4f}")
    print(f"  [honest subset] MLP real={mlp_sub['real']['acc']} synth={mlp_sub['synth']['acc']} "
          f"| SVM real={svm_sub['real']['acc']} synth={svm_sub['synth']['acc']}")

    # 산출물 저장 — pipeline과 encoder를 dict로 묶어서 저장
    # 추론 시: bundle = joblib.load(...); enc = bundle["encoder"]; pipe = bundle["pipeline"]
    #         pred_str = enc.inverse_transform(pipe.predict(X))
    out_model = Path(a.out_model)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": mlp, "encoder": encoder, "feature_order": FEATURE_ORDER, "classes": LABELS}, out_model)

    plots_dir = Path(a.out_plots)
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion(
        mlp_r["confusion_matrix"], plots_dir / "confusion_matrix.png",
        title=f"MLP Confusion Matrix (acc={mlp_r['accuracy']:.3f})",
    )
    plot_confusion(
        svm_r["confusion_matrix"], plots_dir / "confusion_matrix_svm.png",
        title=f"SVM Confusion Matrix (acc={svm_r['accuracy']:.3f})",
    )
    plot_comparison(mlp_r["accuracy"], svm_r["accuracy"], plots_dir / "mlp_vs_svm_accuracy.png")
    plot_loss_curve(mlp, plots_dir / "training_loss.png")

    # SECURITY_REVIEW_V26 F1: record sha256 so allocator.py can verify before
    # joblib.load. Hash is computed AFTER joblib.dump (above) so the file is final.
    model_sha = _sha256_of_file(out_model)
    meta = {
        "feature_order": FEATURE_ORDER,
        "classes": LABELS,
        "metrics": {"mlp": mlp_r, "svm": svm_r},
        "subset_metrics": {"mlp": mlp_sub, "svm": svm_sub},
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "train_src": dict(_C(src_train.tolist())),
        "test_src": dict(_C(src_test.tolist())),
        "seed": int(a.seed),
        "model_sha256": model_sha,
        "disclosure": (
            "Trained predominantly on AR(1)-synthetic usage rows; legacy 'real' rows "
            "(inbox-count path) had zero usage signal. Headline accuracy is inflated "
            "by trivially-correct zero-usage real rows — see subset_metrics for the "
            "honest synth-only number. Synthetic-bootstrap / real-state policy-"
            "distillation model, NOT a production-validated balancer. Real usage now "
            "collected via scripts/collect_usage.py for a future retrain."
        ),
    }
    Path(a.out_meta).write_text(json.dumps(meta, indent=2))
    write_report(
        Path(a.out_report),
        mlp_result=mlp_r, svm_result=svm_r,
        mlp_sub=mlp_sub, svm_sub=svm_sub,
        n_train=len(X_train), n_test=len(X_test),
        classes=LABELS,
    )
    print(f"\nSaved model  -> {out_model}")
    print(f"Saved plots  -> {plots_dir}")
    print(f"Saved meta   -> {a.out_meta}")
    print(f"Saved report -> {a.out_report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
