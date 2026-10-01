"""Inference-only adapter for the trained allocator MLP.

Design contract (doc 6.1, 6.2, 6.4):
  * Loads model from disk lazily on first call. Hot-reload if file mtime changes.
  * No training here -- training happens offline via scripts/train_allocator_mlp.py.
  * Returns a *recommendation* -- never executes dispatch. The caller (a
    deterministic dispatch tool) decides whether to honor or override.

The model file is a sklearn Pipeline (StandardScaler + MLPClassifier).
Inference cost: <5ms per call. Never blocks the agent loop.

SECURITY (SECURITY_REVIEW_V26.md F1, v2.6 round-2 audit):
  joblib.load == pickle.load. A tampered .joblib file = RCE on load.

  Policy (round 2, default-strict — flipped from round 1's opt-in strict):
    * If allocator_meta.json contains "model_sha256", it is verified before
      joblib.load. Mismatch -> hard fail (no load attempted).
    * If the manifest is missing/has no hash, load is REFUSED by default.
      The operator must either re-run the trainer (which writes the hash) or
      explicitly opt out via ``ULTRADEBATE_ALLOCATOR_LEGACY=1`` — in which
      case load proceeds with a stderr WARNING. ``ULTRADEBATE_ALLOCATOR_STRICT=1``
      is honored as a no-op alias for compatibility with round-1 deployments.

  Why default-strict: round-1 codex audit noted that "the legacy non-strict
  path still allows joblib.load without a manifest, so the original
  structural claim 'joblib.load gated by sha256 manifest verification' is
  still only conditionally true." Default-strict makes the structural claim
  unconditional; the legacy opt-out exists for emergency rollback only.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = REPO_ROOT / "artifacts" / "allocator_mlp.joblib"
DEFAULT_META_PATH = REPO_ROOT / "artifacts" / "allocator_meta.json"

MODEL_PATH = Path(os.environ.get("ULTRADEBATE_ALLOCATOR_MODEL", DEFAULT_MODEL_PATH))
META_PATH = Path(os.environ.get("ULTRADEBATE_ALLOCATOR_META", DEFAULT_META_PATH))

_LOCK = threading.Lock()
_CACHE: dict[str, Any] = {
    "model": None,
    "meta": None,
    "model_mtime": 0.0,
    "meta_mtime": 0.0,
}


def _iso(epoch: float) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    """Stream the file through sha256; bounded memory regardless of file size."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_model_integrity(
    model_path: Path, meta_path: Path
) -> tuple[bool, dict[str, Any] | None]:
    """Pre-load integrity check. Returns (ok_to_load, error_status_dict_or_None).

    Behavior (SECURITY_REVIEW_V26.md F1, round-2 audit — *default-strict*):
      * If meta contains "model_sha256": hash must match. Mismatch -> refuse load.
      * If meta is missing/has no hash: refuse load by default.
        Opt-out is ``ULTRADEBATE_ALLOCATOR_LEGACY=1`` — allows load with a
        visible stderr WARNING. ``ULTRADEBATE_ALLOCATOR_STRICT=1`` is honored
        as a no-op alias (round-1 deployments may still set it).

    The shipped artifact already includes ``model_sha256`` in
    ``allocator_meta.json``, so default-strict does not break the in-repo
    production path. Operators who train their own model must either re-run
    the official trainer (which writes the hash) or set the legacy env var.
    """
    import sys

    # Legacy opt-out. STRICT=1 is honored as a no-op alias (round-1 compat).
    legacy_ok = os.environ.get("ULTRADEBATE_ALLOCATOR_LEGACY") == "1"
    expected_sha: str | None = None
    meta_existed = meta_path.exists()
    if meta_existed:
        try:
            meta_preview = json.loads(meta_path.read_text(encoding="utf-8"))
            expected_sha = meta_preview.get("model_sha256")
        except (OSError, json.JSONDecodeError):
            expected_sha = None
    if expected_sha:
        try:
            actual_sha = _sha256_file(model_path)
        except OSError as exc:
            return False, {
                "ok": False,
                "error": "sha256_compute_failed",
                "detail": repr(exc)[:200],
            }
        if actual_sha != expected_sha:
            return False, {
                "ok": False,
                "error": "sha256_mismatch",
                "expected_prefix": expected_sha[:16],
                "actual_prefix": actual_sha[:16],
                "note": (
                    "Model file hash does not match allocator_meta.json. "
                    "Re-train with scripts/train_allocator_mlp.py or set "
                    "ULTRADEBATE_ALLOCATOR_LEGACY=1 to load anyway "
                    "(NOT RECOMMENDED — see SECURITY_REVIEW_V26.md F1)."
                ),
            }
        return True, None
    # No expected hash recorded. Round-2 policy: refuse by default.
    if not legacy_ok:
        return False, {
            "ok": False,
            "error": "sha256_required",
            "note": (
                "allocator_meta.json has no model_sha256 field. Round-2 "
                "policy is default-strict (joblib pickle RCE guard). Either "
                "re-train via scripts/train_allocator_mlp.py (writes hash) "
                "or set ULTRADEBATE_ALLOCATOR_LEGACY=1 to bypass for "
                "emergency rollback (NOT RECOMMENDED for production)."
            ),
        }
    # legacy opt-out path: load는 허용하되 운영자가 모를 수 없도록 강한 경고.
    if meta_existed:
        reason = "allocator_meta.json present but no model_sha256 field"
    else:
        reason = "allocator_meta.json missing"
    print(
        f"WARNING: ULTRADEBATE_ALLOCATOR_LEGACY=1 active — loading model "
        f"without sha256 verification ({reason}). This bypasses the F1 "
        "joblib pickle RCE guard. Re-train to populate model_sha256 in "
        "allocator_meta.json and remove this env var ASAP "
        "(see SECURITY_REVIEW_V26.md F1, round-2 audit).",
        file=sys.stderr,
    )
    return True, None


def _maybe_reload() -> tuple[Any, dict | None, dict[str, Any]]:
    """Return (model, meta, status_dict). On absent model, status['ok']=False."""
    status: dict[str, Any] = {
        "ok": True,
        "model_path": str(MODEL_PATH),
        "meta_path": str(META_PATH),
    }
    if not MODEL_PATH.exists():
        return None, None, {
            "ok": False,
            "error": "no_model",
            "note": f"Run: python scripts/train_allocator_mlp.py to create {MODEL_PATH}",
            "model_path": str(MODEL_PATH),
        }

    mmtime = MODEL_PATH.stat().st_mtime
    meta_mtime = META_PATH.stat().st_mtime if META_PATH.exists() else 0.0

    with _LOCK:
        if (
            _CACHE["model"] is None
            or _CACHE["model_mtime"] != mmtime
            or _CACHE["meta_mtime"] != meta_mtime
        ):
            # SECURITY (F1): verify sha256 manifest BEFORE joblib.load.
            ok_to_load, integrity_err = _verify_model_integrity(MODEL_PATH, META_PATH)
            if not ok_to_load:
                # Do NOT cache; force re-check on next call (operator may fix meta).
                return None, None, integrity_err or {
                    "ok": False, "error": "integrity_check_failed",
                }
            try:
                import joblib
            except ImportError:
                return None, None, {
                    "ok": False,
                    "error": "joblib_not_installed",
                    "note": "pip install joblib (or install [mlp] extra)",
                }
            try:
                _CACHE["model"] = joblib.load(MODEL_PATH)
                _CACHE["model_mtime"] = mmtime
                if META_PATH.exists():
                    _CACHE["meta"] = json.loads(META_PATH.read_text())
                    _CACHE["meta_mtime"] = meta_mtime
                else:
                    _CACHE["meta"] = None
                    _CACHE["meta_mtime"] = 0.0
            except Exception as e:
                return None, None, {
                    "ok": False,
                    "error": "load_failed",
                    "detail": repr(e)[:300],
                }
        return _CACHE["model"], _CACHE["meta"], status


def model_info() -> dict[str, Any]:
    """Diagnostic: model loaded? feature_order? classes? mtime?"""
    model, meta, st = _maybe_reload()
    if not st["ok"]:
        return st
    out: dict[str, Any] = {
        "ok": True,
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH),
        "model_mtime_iso": _iso(MODEL_PATH.stat().st_mtime),
    }
    if meta:
        out["classes"] = meta.get("classes", [])
        out["feature_order"] = meta.get("feature_order", [])
        out["metrics"] = meta.get("metrics", {})
    return out


def _unwrap_model(raw: Any) -> tuple[Any, Any, list[str] | None]:
    """Accept either form the trainer might persist:
      (a) bare sklearn Pipeline/estimator (legacy / simple path).
      (b) bundle dict {"pipeline": Pipeline, "encoder": LabelEncoder|None,
                       "feature_order": [...], "classes": [...]}.
    Returns (estimator, label_decoder_or_None, bundled_feature_order_or_None).

    Why a bundle exists: sklearn 1.6 MLPClassifier(early_stopping=True) with
    string labels triggers `np.isnan(y_pred)` on `<U6` dtype and crashes.
    `train_allocator_mlp.py` works around it via LabelEncoder + int labels,
    then ships the encoder so predict() can map ints back to strings.
    Inference must accept both encodings.
    """
    if isinstance(raw, dict) and "pipeline" in raw:
        return raw.get("pipeline"), raw.get("encoder"), raw.get("feature_order")
    return raw, None, None


def predict(
    features: dict[str, float],
    *,
    expected_feature_order: list[str] | None = None,
) -> dict[str, Any]:
    """Return recommended target + confidence + per-class probabilities.

    features: {feature_name: value, ...}. Unknown features ignored.
    expected_feature_order: if given, used; else taken from bundle/meta. If
                            all three sources are missing, returns error
                            (we cannot guess column order).
    Missing required features default to 0.0 with a warning entry.
    """
    model, meta, st = _maybe_reload()
    if not st["ok"]:
        return st

    estimator, encoder, bundle_feat_order = _unwrap_model(model)
    if estimator is None:
        return {
            "ok": False,
            "error": "no_estimator",
            "note": "Model bundle has no 'pipeline' key and is not an estimator",
        }

    feat_order = (
        expected_feature_order
        or bundle_feat_order
        or (meta.get("feature_order") if meta else None)
    )
    if not feat_order:
        return {
            "ok": False,
            "error": "no_feature_order",
            "note": "Bundle and meta both missing feature_order and none provided",
        }

    missing = [k for k in feat_order if k not in features]
    import numpy as np

    x = np.array([[float(features.get(k, 0.0)) for k in feat_order]])

    try:
        raw_pred = estimator.predict(x)[0]
        proba_arr = estimator.predict_proba(x)[0]
        raw_classes = list(estimator.classes_)
        if encoder is not None:
            # estimator was trained on int labels; decode back to strings.
            decoded_classes = list(encoder.inverse_transform(raw_classes))
            pred_label = encoder.inverse_transform([raw_pred])[0]
        else:
            decoded_classes = raw_classes
            pred_label = raw_pred
        proba = {str(c): float(p) for c, p in zip(decoded_classes, proba_arr)}
        conf = float(max(proba.values()))
    except Exception as e:
        return {"ok": False, "error": "predict_failed", "detail": repr(e)[:300]}

    return {
        "ok": True,
        "target": str(pred_label),
        "confidence": round(conf, 4),
        "probabilities": {k: round(v, 4) for k, v in proba.items()},
        "missing_features_defaulted_to_zero": missing,
    }
