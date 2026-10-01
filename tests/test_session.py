"""Session lifecycle (init / save / load)."""
from __future__ import annotations

from ultradebate.session import init_session, load_meta, save_meta


def test_init_creates_dirs(tmp_path):
    meta = init_session(tmp_path, "topic", "strict", 3, 100_000)
    assert meta.session_id
    sdir = tmp_path / meta.session_id
    assert (sdir / "meta.json").exists()
    assert (sdir / "mailbox" / "inbox").is_dir()
    assert (sdir / "mailbox" / "processed").is_dir()


def test_save_load_roundtrip(tmp_path):
    meta = init_session(tmp_path, "topic", "strict", 3, 100_000)
    sdir = tmp_path / meta.session_id
    meta.current_round = 2
    meta.tokens_used = 500
    meta.consensus_history = [0.5, 0.7]
    save_meta(sdir, meta)
    loaded = load_meta(sdir)
    assert loaded.current_round == 2
    assert loaded.tokens_used == 500
    assert loaded.consensus_history == [0.5, 0.7]


def test_unique_session_ids(tmp_path):
    ids = {init_session(tmp_path, "t", "strict", 1, 100).session_id for _ in range(20)}
    assert len(ids) == 20  # all unique
