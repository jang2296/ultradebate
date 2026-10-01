"""Smoke test for presentation/README.md reproduction commands (round-2 audit).

Round-2 codex audit (2026-05-22) called out that no test executed the README
reproduction commands, so a typo in any one of them could rot silently. This
test extracts fenced ``bash`` code blocks from presentation/README.md and
verifies the *invocable* parts in a smoke-test fashion:

  * Every ``python3`` or ``$RTX_PYTHON scripts/<name>.py`` referenced in the README must accept
    ``--help`` without crashing (returncode 0). This catches script renames,
    missing flags, and import errors without paying the cost of a full
    end-to-end run.
  * The README itself must not reference scripts that do not exist.

We intentionally do NOT run the full train pipeline here — it takes 10s+ and
is already covered by ``tests/test_v26_features.py`` and downstream contracts.
The goal is "structural reproducibility": the documented commands point at
real scripts whose CLIs match what the README claims.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
PRES_README = REPO / "presentation" / "README.md"
ROOT_README = REPO / "README.md"


def _extract_bash_blocks(md_path: Path) -> list[str]:
    """Return the textual content of every ```bash fenced block."""
    text = md_path.read_text(encoding="utf-8")
    return re.findall(r"```bash\n(.*?)```", text, flags=re.DOTALL)


def _extract_script_invocations(blocks: list[str]) -> set[Path]:
    """Find every ``python3 scripts/<name>.py`` reference inside the blocks."""
    found: set[Path] = set()
    pat = re.compile(r"(?:python3?|[\"']?\$RTX_PYTHON[\"']?)\s+(scripts/[\w_]+\.py)")
    for blk in blocks:
        for m in pat.finditer(blk):
            found.add(REPO / m.group(1))
    return found


@pytest.fixture(scope="module")
def presentation_scripts() -> set[Path]:
    blocks = _extract_bash_blocks(PRES_README)
    return _extract_script_invocations(blocks)


def test_presentation_readme_references_real_scripts(presentation_scripts):
    """Every ``scripts/*.py`` in presentation/README.md must exist on disk."""
    assert presentation_scripts, (
        "Expected at least one python3 scripts/*.py command in "
        f"{PRES_README}; the round-2 audit pin needs it for staleness "
        "detection."
    )
    missing = [p for p in presentation_scripts if not p.exists()]
    assert not missing, (
        f"presentation/README.md references missing scripts: {missing}. "
        "Either restore the script or update the README."
    )


@pytest.mark.parametrize("idx", range(8))
def test_referenced_scripts_accept_help(idx, presentation_scripts):
    """Each referenced script must accept ``--help`` (returncode 0).

    Parameterized over a fixed range so each script gets its own test name in
    the report (better failure isolation than a single loop test).
    """
    scripts = sorted(presentation_scripts)
    if idx >= len(scripts):
        pytest.skip(f"only {len(scripts)} scripts referenced in README")
    script = scripts[idx]
    proc = subprocess.run(
        # Use the interpreter running this test (the RTX dev environment),
        # rather than the bare WSL system Python, so declared optional MLP
        # dependencies are resolved from the same environment as the suite.
        [sys.executable, str(script), "--help"],
        capture_output=True, text=True, timeout=20, check=False,
    )
    assert proc.returncode == 0, (
        f"{script.name} --help failed (rc={proc.returncode})\n"
        f"stderr: {proc.stderr[:400]}"
    )
    # --help output must mention the script name OR a recognizable usage hint.
    haystack = (proc.stdout + proc.stderr).lower()
    assert "usage" in haystack or script.stem.lower() in haystack, (
        f"{script.name} --help produced no usage info; likely the script "
        f"prints nothing. stdout[:200]={proc.stdout[:200]!r}"
    )


def test_root_readme_documents_legacy_env_var():
    """Round-2 default-strict: README must mention ULTRADEBATE_ALLOCATOR_LEGACY."""
    text = ROOT_README.read_text(encoding="utf-8")
    assert "ULTRADEBATE_ALLOCATOR_LEGACY" in text, (
        "Round-2 default-strict requires README to document the legacy "
        "opt-out env var so operators can recover from a missing manifest."
    )
    # Either the new "default strict" wording or the round-1 hint, but the
    # legacy env var must be discoverable.
    assert "sha256" in text.lower()
