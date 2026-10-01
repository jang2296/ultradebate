"""Tests for v2.6 allocator — inference-only contract.

Pins:
  * No-model returns graceful error (does not raise)
  * Loaded-model returns valid prediction + probability dict
  * Missing features default to 0.0 + report list
  * Confidence is in [0, 1]
  * Round-2 audit (2026-05-22): default-strict sha256 manifest verification
      - Manifest absent + no legacy env → refuse load (round-2 default)
      - Manifest absent + ULTRADEBATE_ALLOCATOR_LEGACY=1 → load with warning
      - Manifest matches sha256 → load (round-2 path-of-least-friction)
"""
from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path

import pytest


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _reload_allocator(monkeypatch, model_path: Path, meta_path: Path):
    """Reload the allocator module so it re-reads env vars at import time."""
    monkeypatch.setenv("ULTRADEBATE_ALLOCATOR_MODEL", str(model_path))
    monkeypatch.setenv("ULTRADEBATE_ALLOCATOR_META", str(meta_path))
    # Round-2: clear legacy env unless test explicitly sets it.
    monkeypatch.delenv("ULTRADEBATE_ALLOCATOR_LEGACY", raising=False)
    monkeypatch.delenv("ULTRADEBATE_ALLOCATOR_STRICT", raising=False)
    sys.modules.pop("ultradebate.allocator", None)
    return importlib.import_module("ultradebate.allocator")


def _import_feature_order():
    """Import FEATURE_ORDER from scripts/extract_features.py."""
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    sys.modules.pop("extract_features", None)
    ef = importlib.import_module("extract_features")
    return ef.FEATURE_ORDER


def _train_tiny_pipeline(
    model_path: Path,
    meta_path: Path,
    feature_order: list[str],
    *,
    write_sha256: bool = True,
):
    """Train a deterministic tiny MLP, dump to disk + meta JSON.

    ``write_sha256=True`` (round-2 default) matches the real trainer which
    fills ``model_sha256``. Pass False to test the missing-manifest path.
    """
    import joblib
    import numpy as np
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(7)
    X = rng.random((60, len(feature_order)))
    y = ["claude"] * 20 + ["codex"] * 20 + ["split"] * 20
    pipe = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(16,), max_iter=200, random_state=0),
    )
    pipe.fit(X, y)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, model_path)
    meta: dict = {
        "feature_order": feature_order,
        "classes": ["claude", "codex", "split"],
        "metrics": {"test_accuracy": 0.99},
    }
    if write_sha256:
        meta["model_sha256"] = _sha256(model_path)
    meta_path.write_text(json.dumps(meta))


def test_no_model_graceful(tmp_path, monkeypatch):
    """When model file is absent, predict + info return ok=False, never raise."""
    allocator = _reload_allocator(
        monkeypatch, tmp_path / "nope.joblib", tmp_path / "nope.json"
    )
    info = allocator.model_info()
    assert info["ok"] is False
    assert info.get("error") == "no_model"

    pred = allocator.predict({"claude_5h_used": 0.5})
    assert pred["ok"] is False


def test_loaded_model_returns_valid_prediction(tmp_path, monkeypatch):
    """Train a tiny model on synthetic data, load via env, predict."""
    feature_order = _import_feature_order()
    mp = tmp_path / "alloc.joblib"
    mep = tmp_path / "alloc_meta.json"
    _train_tiny_pipeline(mp, mep, feature_order)

    allocator = _reload_allocator(monkeypatch, mp, mep)

    info = allocator.model_info()
    assert info["ok"] is True
    assert info["model_loaded"] is True
    assert info["feature_order"] == feature_order

    feats = {k: 0.5 for k in feature_order}
    pred = allocator.predict(feats)
    assert pred["ok"] is True
    assert pred["target"] in {"claude", "codex", "split"}
    assert 0.0 <= pred["confidence"] <= 1.0
    assert set(pred["probabilities"].keys()) == {"claude", "codex", "split"}
    # Probabilities sum to ~1
    total = sum(pred["probabilities"].values())
    assert 0.95 <= total <= 1.05


def test_missing_features_default_to_zero(tmp_path, monkeypatch):
    """If caller omits some features, they default to 0.0 and are reported."""
    feature_order = _import_feature_order()
    mp = tmp_path / "m.joblib"
    mep = tmp_path / "me.json"
    _train_tiny_pipeline(mp, mep, feature_order)

    allocator = _reload_allocator(monkeypatch, mp, mep)

    # Supply only 3 of N features
    feats = {"claude_5h_used": 0.8, "codex_5h_used": 0.2, "queue_len": 1}
    pred = allocator.predict(feats)
    assert pred["ok"] is True
    # (len(feature_order) - 3) features must be reported as defaulted
    assert (
        len(pred["missing_features_defaulted_to_zero"])
        == len(feature_order) - 3
    )
    # The 3 supplied features are NOT in the missing list
    missing = set(pred["missing_features_defaulted_to_zero"])
    assert "claude_5h_used" not in missing
    assert "codex_5h_used" not in missing
    assert "queue_len" not in missing


def test_predict_with_explicit_feature_order(tmp_path, monkeypatch):
    """expected_feature_order param overrides meta-derived order."""
    feature_order = _import_feature_order()
    mp = tmp_path / "m2.joblib"
    mep = tmp_path / "me2.json"
    _train_tiny_pipeline(mp, mep, feature_order)

    allocator = _reload_allocator(monkeypatch, mp, mep)
    feats = {k: 0.1 for k in feature_order}
    pred = allocator.predict(feats, expected_feature_order=feature_order)
    assert pred["ok"] is True
    assert pred["target"] in {"claude", "codex", "split"}


# --- Round-2 audit (2026-05-22): default-strict sha256 verification ----------

def test_round2_strict_default_blocks_no_manifest(tmp_path, monkeypatch):
    """No model_sha256 + no legacy env → refuse load (round-2 default).

    Pins the structural claim "joblib.load gated by sha256 manifest" to be
    unconditionally true in default mode.
    """
    feature_order = _import_feature_order()
    mp = tmp_path / "no_sha.joblib"
    mep = tmp_path / "no_sha_meta.json"
    _train_tiny_pipeline(mp, mep, feature_order, write_sha256=False)

    allocator = _reload_allocator(monkeypatch, mp, mep)
    info = allocator.model_info()
    assert info["ok"] is False
    assert info.get("error") == "sha256_required"
    pred = allocator.predict({k: 0.5 for k in feature_order})
    assert pred["ok"] is False


def test_round2_legacy_env_allows_no_manifest(tmp_path, monkeypatch, capsys):
    """ULTRADEBATE_ALLOCATOR_LEGACY=1 + no sha256 → load with stderr WARNING.

    The opt-out is preserved for emergency rollback but must emit a strong
    warning so the operator cannot miss it.
    """
    feature_order = _import_feature_order()
    mp = tmp_path / "legacy.joblib"
    mep = tmp_path / "legacy_meta.json"
    _train_tiny_pipeline(mp, mep, feature_order, write_sha256=False)

    monkeypatch.setenv("ULTRADEBATE_ALLOCATOR_LEGACY", "1")
    allocator = _reload_allocator(monkeypatch, mp, mep)
    # Re-set LEGACY env (reload helper clears it).
    monkeypatch.setenv("ULTRADEBATE_ALLOCATOR_LEGACY", "1")
    info = allocator.model_info()
    assert info["ok"] is True
    captured = capsys.readouterr()
    assert "ULTRADEBATE_ALLOCATOR_LEGACY=1" in captured.err
    assert "without sha256" in captured.err.lower()


def test_round2_manifest_match_allows_load(tmp_path, monkeypatch):
    """Matching sha256 manifest → load (default-strict success path).

    This is the shipped artifact's actual code path — pin it explicitly so
    the trainer-writes-hash contract never silently breaks.
    """
    feature_order = _import_feature_order()
    mp = tmp_path / "good.joblib"
    mep = tmp_path / "good_meta.json"
    _train_tiny_pipeline(mp, mep, feature_order, write_sha256=True)

    # Sanity: meta must actually contain the hash and it must match the file.
    meta = json.loads(mep.read_text())
    assert "model_sha256" in meta
    assert meta["model_sha256"] == _sha256(mp)

    allocator = _reload_allocator(monkeypatch, mp, mep)
    info = allocator.model_info()
    assert info["ok"] is True
    assert info["model_loaded"] is True


def test_round2_sha256_mismatch_refuses_load(tmp_path, monkeypatch):
    """Manifest sha256 != file sha256 → refuse load (tamper detection)."""
    feature_order = _import_feature_order()
    mp = tmp_path / "tamp.joblib"
    mep = tmp_path / "tamp_meta.json"
    _train_tiny_pipeline(mp, mep, feature_order, write_sha256=True)

    # Tamper with the manifest's recorded hash so it no longer matches the file.
    meta = json.loads(mep.read_text())
    meta["model_sha256"] = "0" * 64  # obviously wrong
    mep.write_text(json.dumps(meta))

    allocator = _reload_allocator(monkeypatch, mp, mep)
    info = allocator.model_info()
    assert info["ok"] is False
    assert info.get("error") == "sha256_mismatch"
