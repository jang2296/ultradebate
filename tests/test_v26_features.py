"""Tests for feature extraction — FEATURE_ORDER stability + determinism.

If FEATURE_ORDER ever changes shape, allocator.predict() breaks because the
trained model expects a fixed column order. These tests pin the schema.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="module", autouse=True)
def _scripts_on_path():
    """Add scripts/ to sys.path so we can import extract_features as a module."""
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    sys.modules.pop("extract_features", None)
    yield


def test_feature_order_has_19():
    """The trained MLP expects exactly 19 features in this order."""
    from extract_features import FEATURE_ORDER

    assert len(FEATURE_ORDER) == 19
    # Verify expected names are present
    expected = [
        "claude_5h_used",
        "codex_5h_used",
        "hour_sin",
        "hour_cos",
        "queue_len",
        "recent_ack_miss_rate",
        "last_response_latency_s",
    ]
    for need in expected:
        assert need in FEATURE_ORDER, f"FEATURE_ORDER missing: {need}"


def test_feature_order_no_duplicates():
    """FEATURE_ORDER must not contain duplicates (would silently break order)."""
    from extract_features import FEATURE_ORDER

    assert len(FEATURE_ORDER) == len(set(FEATURE_ORDER))


def test_compute_features_deterministic():
    """Same input → same output. No timestamp.now() leakage, no randomness."""
    from extract_features import compute_features

    events = [{"_ts_epoch": 1700000000.0, "sender": "claude", "type": "ping"}]
    v1 = compute_features(1700001000.0, events, hour=10, dow=1)
    v2 = compute_features(1700001000.0, events, hour=10, dow=1)
    assert v1 == v2


def test_compute_features_keys_match_FEATURE_ORDER():
    """Every key in FEATURE_ORDER must appear as a dict key in the output."""
    from extract_features import FEATURE_ORDER, compute_features

    v = compute_features(1700000000.0, [], hour=12, dow=3)
    assert set(v.keys()) == set(FEATURE_ORDER)


def test_compute_features_empty_events_no_crash():
    """compute_features([]) must succeed and return numeric values."""
    from extract_features import FEATURE_ORDER, compute_features

    v = compute_features(1700000000.0, [], hour=0, dow=0)
    for k in FEATURE_ORDER:
        assert isinstance(v[k], (int, float))


def test_compute_features_values_in_expected_ranges():
    """Bounded features must respect their bounds."""
    from extract_features import compute_features

    events = [
        {"_ts_epoch": 1700000000.0, "sender": "claude", "type": "priority_send"},
        {"_ts_epoch": 1700000100.0, "sender": "codex", "type": "ping"},
    ]
    v = compute_features(1700000200.0, events, hour=14, dow=2)
    # Usage fractions are clipped to [0, 1]
    for k in ("claude_5h_used", "codex_5h_used", "claude_1h_avg", "codex_1h_avg"):
        assert 0.0 <= v[k] <= 1.0, f"{k}={v[k]} out of [0,1]"
    # sin/cos bounded by [-1, 1]
    for k in ("hour_sin", "hour_cos", "dow_sin", "dow_cos"):
        assert -1.0 <= v[k] <= 1.0
    # miss rate bounded
    assert 0.0 <= v["recent_ack_miss_rate"] <= 1.0


def test_compute_features_hour_dow_overrides_used():
    """Explicit hour/dow params override datetime extraction."""
    from extract_features import compute_features
    import math

    # hour=0 → hour_sin == 0, hour_cos == 1
    v = compute_features(1700000000.0, [], hour=0, dow=0)
    assert abs(v["hour_sin"] - 0.0) < 1e-9
    assert abs(v["hour_cos"] - 1.0) < 1e-9
    assert abs(v["dow_sin"] - 0.0) < 1e-9
    assert abs(v["dow_cos"] - 1.0) < 1e-9

    # hour=6 → sin(2π*6/24) == sin(π/2) == 1
    v6 = compute_features(1700000000.0, [], hour=6, dow=0)
    assert abs(v6["hour_sin"] - 1.0) < 1e-9
    assert abs(v6["hour_cos"] - 0.0) < 1e-9


def test_feature_order_matches_allocator_expectation():
    """FEATURE_ORDER must remain compatible between extractor and inference.

    This is a structural test: it ensures the offline-trained model and the
    online inference adapter agree on column layout.
    """
    from extract_features import FEATURE_ORDER, compute_features

    v = compute_features(1700000000.0, [], hour=10, dow=1)
    # Build feature vector in FEATURE_ORDER (what allocator.predict does)
    vec = [v[k] for k in FEATURE_ORDER]
    assert len(vec) == len(FEATURE_ORDER) == 19
    # All numeric
    for x in vec:
        assert isinstance(x, (int, float))
