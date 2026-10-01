"""Tests for quota_probe — parser + integration with allocator.

Covers:
  * parse_status_output: realistic codex output, no-match, unicode bars,
    multi-section (main + Spark) first-occurrence semantics.
  * snapshot_to_features: %left → *_used conversion + missing/invalid data.
  * probe_claude_quota: structured 'not_directly_probable' explanation.
  * record_snapshot / read_recent_snapshots: persistent JSONL log roundtrip
    and agent filter.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class TestParseStatusOutput:
    def test_codex_realistic_output(self):
        from ultradebate.quota_probe import parse_status_output

        text = """
        Model:                       gpt-5.5
        Directory:                   /tmp
        5h limit:                    [█████████████████░░░] 85% left (resets 00:43)
        Weekly limit:                [████████████████░░░░] 79% left (resets 12:34 on 27 May)
        """
        out = parse_status_output(text)
        assert out["5h_pct_left"] == 85
        assert out["weekly_pct_left"] == 79
        assert "00:43" in out.get("5h_resets", "")
        assert "27 May" in out.get("weekly_resets", "")

    def test_no_match_returns_empty(self):
        from ultradebate.quota_probe import parse_status_output

        assert parse_status_output("nothing relevant here") == {}

    def test_empty_input_returns_empty(self):
        from ultradebate.quota_probe import parse_status_output

        assert parse_status_output("") == {}
        assert parse_status_output(None) == {}  # type: ignore[arg-type]

    def test_handles_unicode_bars(self):
        from ultradebate.quota_probe import parse_status_output

        text = "5h limit: [##########----------] 50% left"
        out = parse_status_output(text)
        assert out.get("5h_pct_left") == 50

    def test_multi_section_takes_first(self):
        """Codex shows two sets (main model + Spark) — we take the first."""
        from ultradebate.quota_probe import parse_status_output

        text = """
        5h limit: [▓▓▓▓░░░░] 50% left
        Weekly limit: [▓▓░░] 30% left
        GPT-5.3-Codex-Spark limit:
        5h limit: [▓▓▓▓▓▓▓▓] 100% left
        Weekly limit: [▓▓▓▓▓▓▓▓] 100% left
        """
        out = parse_status_output(text)
        assert out["5h_pct_left"] == 50
        assert out["weekly_pct_left"] == 30

    def test_percent_clamped_to_range(self):
        from ultradebate.quota_probe import parse_status_output

        # Pathologically over-range value should clamp at 100.
        text = "5h limit: [#] 999% left"
        out = parse_status_output(text)
        assert out["5h_pct_left"] == 100


class TestSnapshotToFeatures:
    def test_basic_conversion(self):
        from ultradebate.quota_probe import snapshot_to_features

        snaps = {
            "claude": {"ok": True, "5h_pct_left": 80, "weekly_pct_left": 60},
            "codex": {"ok": True, "5h_pct_left": 30, "weekly_pct_left": 15},
        }
        feats = snapshot_to_features(snaps)
        assert abs(feats["claude_5h_used"] - 0.20) < 1e-9
        assert abs(feats["claude_week_used"] - 0.40) < 1e-9
        assert abs(feats["codex_5h_used"] - 0.70) < 1e-9
        assert abs(feats["codex_week_used"] - 0.85) < 1e-9

    def test_missing_data_defaults_zero(self):
        from ultradebate.quota_probe import snapshot_to_features

        feats = snapshot_to_features({})
        assert all(v == 0.0 for v in feats.values())

    def test_ok_false_ignored(self):
        from ultradebate.quota_probe import snapshot_to_features

        snaps = {
            "claude": {"ok": False, "5h_pct_left": 99, "weekly_pct_left": 99},
            "codex": {"ok": True, "5h_pct_left": 50, "weekly_pct_left": 25},
        }
        feats = snapshot_to_features(snaps)
        assert feats["claude_5h_used"] == 0.0
        assert feats["claude_week_used"] == 0.0
        assert abs(feats["codex_5h_used"] - 0.50) < 1e-9
        assert abs(feats["codex_week_used"] - 0.75) < 1e-9

    def test_invalid_input_returns_zeros(self):
        from ultradebate.quota_probe import snapshot_to_features

        feats = snapshot_to_features("not a dict")  # type: ignore[arg-type]
        assert all(v == 0.0 for v in feats.values())


class TestClaudeProbeNotDirectlyAvailable:
    def test_returns_structured_explanation(self):
        from ultradebate.quota_probe import probe_claude_quota

        r = probe_claude_quota()
        assert r["ok"] is False
        assert r["agent"] == "claude"
        assert r.get("error") == "not_directly_probable"
        assert "manual_recipe" in r
        assert "step1" in r["manual_recipe"]
        assert "step2" in r["manual_recipe"]
        assert "step3" in r["manual_recipe"]


class TestRecordSnapshot:
    def test_record_and_read_back(self, tmp_path, monkeypatch):
        import ultradebate.quota_probe as qp

        monkeypatch.setattr(qp, "QUOTA_LOG", tmp_path / "qlog.jsonl")
        r = qp.record_snapshot({"agent": "codex", "5h_pct_left": 50, "weekly_pct_left": 25})
        assert r["ok"] is True
        snaps = qp.read_recent_snapshots()
        assert len(snaps) == 1
        assert snaps[0]["5h_pct_left"] == 50
        assert "ts" in snaps[0]

    def test_agent_filter(self, tmp_path, monkeypatch):
        import ultradebate.quota_probe as qp

        monkeypatch.setattr(qp, "QUOTA_LOG", tmp_path / "qlog.jsonl")
        qp.record_snapshot({"agent": "codex", "5h_pct_left": 50})
        qp.record_snapshot({"agent": "claude", "5h_pct_left": 70})
        qp.record_snapshot({"agent": "codex", "5h_pct_left": 40})
        assert len(qp.read_recent_snapshots(agent="claude")) == 1
        assert len(qp.read_recent_snapshots(agent="codex")) == 2
        assert len(qp.read_recent_snapshots()) == 3

    def test_max_n_returns_newest(self, tmp_path, monkeypatch):
        import ultradebate.quota_probe as qp

        monkeypatch.setattr(qp, "QUOTA_LOG", tmp_path / "qlog.jsonl")
        for pct in (10, 20, 30, 40, 50):
            qp.record_snapshot({"agent": "codex", "5h_pct_left": pct})
        snaps = qp.read_recent_snapshots(max_n=2)
        assert len(snaps) == 2
        # Newest last — pct 40 then 50 should be the tail.
        assert snaps[-1]["5h_pct_left"] == 50
        assert snaps[-2]["5h_pct_left"] == 40

    def test_missing_log_returns_empty(self, tmp_path, monkeypatch):
        import ultradebate.quota_probe as qp

        monkeypatch.setattr(qp, "QUOTA_LOG", tmp_path / "nope.jsonl")
        assert qp.read_recent_snapshots() == []

    def test_corrupt_lines_are_skipped(self, tmp_path, monkeypatch):
        import ultradebate.quota_probe as qp

        log = tmp_path / "qlog.jsonl"
        log.write_text(
            '{"agent": "codex", "5h_pct_left": 10}\n'
            "not-valid-json\n"
            '{"agent": "codex", "5h_pct_left": 20}\n',
            encoding="utf-8",
        )
        monkeypatch.setattr(qp, "QUOTA_LOG", log)
        snaps = qp.read_recent_snapshots()
        assert len(snaps) == 2
        assert snaps[0]["5h_pct_left"] == 10
        assert snaps[1]["5h_pct_left"] == 20
