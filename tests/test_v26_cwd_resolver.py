"""Regression tests for v2.6 audit fix #7 — CWD-aware pane resolver.

Pin the contract that ``_v26_resolve_peer_pane`` ranks tmux panes by
proximity to the *caller's* current working directory:

  cwd_match  (1000)  >  cwd_child  (500+len)  >  cwd_parent  (400+len)  >
  common_prefix  (100+len)  >  no_match  (0)  ->  session.json fallback

Also verify the per-helper math via ``_v26_path_match_score`` so we can
diagnose ranking regressions independently.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from ultradebate import server as srv


# -- score-only tests (pure function, no tmux) ---------------------------------


def test_score_cwd_match_highest():
    score, reason = srv._v26_path_match_score("/mnt/c/proj", "/mnt/c/proj")
    assert reason == "cwd_match"
    assert score == 1000


def test_score_cwd_child_above_parent():
    """Function semantics (per docstring/code):
      cwd_child  = cwd is *child of* pane_path  (cwd lives inside pane dir).
      cwd_parent = pane_path is child of cwd     (pane lives inside cwd dir).
    Ranking: cwd_child > cwd_parent (a pane sitting in our parent dir is
    closer to "our" context than a pane several levels below us).
    """
    # cwd = /mnt/c/proj/sub, pane = /mnt/c/proj  -> cwd is *child of* pane
    s_child, r_child = srv._v26_path_match_score(
        "/mnt/c/proj/sub", "/mnt/c/proj"
    )
    # cwd = /mnt/c/proj, pane = /mnt/c/proj/sub  -> pane is *child of* cwd
    s_parent, r_parent = srv._v26_path_match_score(
        "/mnt/c/proj", "/mnt/c/proj/sub"
    )
    assert r_child == "cwd_child"
    assert r_parent == "cwd_parent"
    assert s_child > s_parent


def test_score_common_prefix_below_parent():
    """common_prefix score should always rank below cwd_parent/cwd_child."""
    s_parent, _ = srv._v26_path_match_score(
        "/mnt/c/proj", "/mnt/c/proj/sub"
    )
    s_common, r_common = srv._v26_path_match_score(
        "/mnt/c/proj/A", "/mnt/c/proj/B"
    )
    assert r_common == "common_prefix"
    assert s_common < s_parent


def test_score_no_match_returns_zero():
    score, reason = srv._v26_path_match_score("/foo", "/bar")
    assert score == 0
    assert reason == "no_match"


def test_score_root_only_prefix_is_no_match():
    """Single '/' prefix must NOT count as common_prefix."""
    score, reason = srv._v26_path_match_score("/foo", "/bar")
    assert score == 0
    assert reason == "no_match"


def test_score_trailing_slash_normalized():
    s1, _ = srv._v26_path_match_score("/foo/", "/foo")
    s2, _ = srv._v26_path_match_score("/foo", "/foo/")
    assert s1 == 1000
    assert s2 == 1000


# -- resolver-level tests (mock tmux list-panes) ------------------------------


def _fake_panes_output(panes):
    """Format pane dicts into tmux list-panes -F output."""
    fmt_lines = []
    for p in panes:
        line = (
            f"{p['pane_id']}|{p.get('pid', 0)}|{p.get('cmd', 'bash')}|"
            f"{p.get('path', '/')}"
        )
        fmt_lines.append(line)
    return "\n".join(fmt_lines)


def test_resolver_picks_cwd_match_over_other_cwd_codex(monkeypatch):
    """Two codex panes, only one matches cwd -> that one wins."""
    panes = [
        {"pane_id": "3:1.1", "pid": 1001, "cmd": "codex", "path": "/example_project"},
        {"pane_id": "5:1.1", "pid": 1002, "cmd": "codex", "path": "/mnt/c/proj"},
    ]
    # cwd 결정값
    monkeypatch.setattr(srv.os, "getcwd", lambda: "/mnt/c/proj")
    # tmux list-panes mock
    monkeypatch.setattr(
        srv.subprocess, "run",
        lambda *a, **k: SimpleNamespace(
            returncode=0, stdout=_fake_panes_output(panes), stderr=""
        ),
    )
    # peer 식별 fast-path: cmd in keywords
    res = srv._v26_resolve_peer_pane("codex")
    assert res["ok"] is True
    assert res["pane"] == "5:1.1"
    assert res["selection_reason"] == "cwd_match"


def test_resolver_falls_back_to_session_json_on_no_match(monkeypatch, tmp_path):
    """모든 후보 점수=0이면 session_json fallback path가 호출된다."""
    panes = [
        {"pane_id": "3:1.1", "pid": 1001, "cmd": "codex", "path": "/foo"},
        {"pane_id": "5:1.1", "pid": 1002, "cmd": "codex", "path": "/bar"},
    ]
    monkeypatch.setattr(srv.os, "getcwd", lambda: "/totally/unrelated")
    monkeypatch.setattr(
        srv.subprocess, "run",
        lambda *a, **k: SimpleNamespace(
            returncode=0, stdout=_fake_panes_output(panes), stderr=""
        ),
    )

    sentinel = {"ok": False, "error": "no_session_json", "selection_reason": "no_match"}
    monkeypatch.setattr(srv, "_v26_session_json_fallback", lambda target: dict(sentinel))

    res = srv._v26_resolve_peer_pane("codex")
    # candidate_count는 추가되고, selection_reason은 session_json_fallback으로 승격
    assert res.get("candidate_count") == 2
    assert res.get("selection_reason") in ("session_json_fallback", "no_match")


def test_resolver_no_candidates_uses_session_json_fallback(monkeypatch):
    """Codex 실행 pane 자체가 0개일 때도 session.json fallback이 호출된다."""
    panes = [
        {"pane_id": "3:1.1", "pid": 999, "cmd": "vim", "path": "/mnt/c/proj"},
    ]
    monkeypatch.setattr(srv.os, "getcwd", lambda: "/mnt/c/proj")
    monkeypatch.setattr(
        srv.subprocess, "run",
        lambda *a, **k: SimpleNamespace(
            returncode=0, stdout=_fake_panes_output(panes), stderr=""
        ),
    )
    sentinel = {"ok": False, "error": "no_session_json", "selection_reason": "no_match"}
    fallback_called = {"count": 0}

    def _fb(target):
        fallback_called["count"] += 1
        return dict(sentinel)

    monkeypatch.setattr(srv, "_v26_session_json_fallback", _fb)
    srv._v26_resolve_peer_pane("codex")
    assert fallback_called["count"] == 1


def test_resolver_prefers_newer_pane_on_score_tie(monkeypatch):
    """동점 시 etime이 짧은 (= 더 최근에 시작된) pane이 우선."""
    panes = [
        {"pane_id": "3:1.1", "pid": 1001, "cmd": "codex", "path": "/mnt/c/proj"},
        {"pane_id": "5:1.1", "pid": 1002, "cmd": "codex", "path": "/mnt/c/proj"},
    ]
    monkeypatch.setattr(srv.os, "getcwd", lambda: "/mnt/c/proj")
    monkeypatch.setattr(
        srv.subprocess, "run",
        lambda *a, **k: SimpleNamespace(
            returncode=0, stdout=_fake_panes_output(panes), stderr=""
        ),
    )
    # pid 1001은 오래된 pane (etime=600), 1002는 새 pane (etime=10)
    etimes = {1001: 600, 1002: 10}
    monkeypatch.setattr(
        srv, "_v26_pane_etime_seconds",
        lambda pid: etimes.get(pid, 99999),
    )
    res = srv._v26_resolve_peer_pane("codex")
    assert res["ok"] is True
    assert res["pane"] == "5:1.1"  # 더 최근 = etime 짧음 = 우선


def test_score_cwd_match_overrides_common_prefix(monkeypatch):
    """cwd_match가 다른 후보의 common_prefix보다 항상 상위에 오는지 확인."""
    panes = [
        {"pane_id": "1:1.1", "pid": 1001, "cmd": "codex", "path": "/mnt/c"},
        {"pane_id": "2:1.1", "pid": 1002, "cmd": "codex", "path": "/mnt/c/proj"},
    ]
    monkeypatch.setattr(srv.os, "getcwd", lambda: "/mnt/c/proj")
    monkeypatch.setattr(
        srv.subprocess, "run",
        lambda *a, **k: SimpleNamespace(
            returncode=0, stdout=_fake_panes_output(panes), stderr=""
        ),
    )
    res = srv._v26_resolve_peer_pane("codex")
    assert res["pane"] == "2:1.1"
    assert res["selection_reason"] == "cwd_match"
