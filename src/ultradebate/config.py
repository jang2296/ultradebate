"""Runtime configuration sourced from environment variables.

Environment variables (all optional):
    ULTRADEBATE_BASE              base directory for sessions (default ~/.ultradebate)
    ULTRADEBATE_MAX_ROUNDS        default round cap when caller omits it (default 5)
    ULTRADEBATE_TOKEN_BUDGET      session token budget across active agents (default 300_000)
    ULTRADEBATE_PER_ROUND_LIMIT   per-agent per-round soft cap (default 8_000)
    ULTRADEBATE_MODE              strict | lenient (default lenient)
    ULTRADEBATE_CONVERGENCE_DELTA delta-consensus threshold (default 0.05)
    ULTRADEBATE_MOCK              "1" to force MockWorker (no real CLI calls)
    ULTRADEBATE_MAX_MODE          "1" to force every CLI to its highest reasoning mode

    ULTRADEBATE_DEDICATED_REVIEWER  rotate | off  (default rotate)

    ULTRADEBATE_CLAUDE_BIN / _ARGS         override for claude CLI
    ULTRADEBATE_CODEX_BIN  / _ARGS         override for codex CLI
    ULTRADEBATE_GEMINI_BIN / _ARGS         override for gemini CLI

    ULTRADEBATE_CLAUDE_THINKING_TOKENS     default 31999 (max for claude)
    ULTRADEBATE_CODEX_REASONING            xhigh|high|medium (default xhigh)
    ULTRADEBATE_GEMINI_THINKING            MAX|HIGH|MEDIUM (default MAX)

    ULTRADEBATE_DEFAULT_ROSTER             "claude=1,codex=1,gemini=1,grok=0"
"""
from __future__ import annotations

import os
import shlex
from dataclasses import dataclass, field
from pathlib import Path


def _expand(p: str | os.PathLike) -> Path:
    return Path(os.path.expanduser(os.path.expandvars(str(p)))).resolve()


def _envflag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _parse_roster(raw: str) -> dict[str, int]:
    """Parse "claude=1,codex=1,gemini=1,grok=0" → {claude:1,...}. Tolerates whitespace."""
    out: dict[str, int] = {}
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk or "=" not in chunk:
            continue
        name, val = chunk.split("=", 1)
        name = name.strip().lower()
        try:
            out[name] = 1 if int(val.strip()) == 1 else 0
        except ValueError:
            continue
    return out


@dataclass(frozen=True)
class CLISpec:
    binary: str
    args: tuple[str, ...] = ()

    @classmethod
    def from_env(cls, prefix: str, default_bin: str, default_args: str) -> "CLISpec":
        binary = os.environ.get(f"ULTRADEBATE_{prefix}_BIN", default_bin)
        raw_args = os.environ.get(f"ULTRADEBATE_{prefix}_ARGS", default_args)
        return cls(binary=binary, args=tuple(shlex.split(raw_args)))


@dataclass(frozen=True)
class MaxModePolicy:
    """Forces every CLI to its highest reasoning mode + max-perf model (R15).

    These knobs are read at every Worker.invoke() call. When `enabled=False`,
    workers fall back to their v0.1 defaults (no extra args, no env injection).
    """
    enabled: bool = True

    # Claude — `ultrathink` keyword + MAX_THINKING_TOKENS env + (optional) --model
    # Default empty: rely on the CLI's account-tier default. Override via
    # ULTRADEBATE_CLAUDE_MODEL when a specific pin is required (e.g.
    # "claude-opus-4-7"). Empty value skips --model injection.
    claude_model: str = ""
    claude_thinking_tokens: int = 31999
    claude_inject_ultrathink: bool = True

    # Codex — TOML reasoning effort override + (optional) --model. ChatGPT-account
    # Codex builds reject custom models — leave empty by default.
    codex_model: str = ""
    codex_reasoning_effort: str = "xhigh"

    # Gemini — (optional) --model + (optional, version-gated) --thinking-level.
    # Default empty so older CLI builds without "gemini-3.1-pro" don't error.
    gemini_model: str = ""
    gemini_thinking_level: str = "MAX"

    # Grok (placeholder for future GrokWorker)
    grok_model: str = ""
    grok_reasoning: str = "max"


@dataclass(frozen=True)
class Config:
    base_dir: Path = field(default_factory=lambda: _expand("~/.ultradebate"))
    max_rounds: int = 5
    token_budget: int = 300_000
    per_round_limit: int = 8_000
    mode: str = "lenient"
    convergence_delta: float = 0.05
    mock: bool = False

    claude_cli: CLISpec = field(default_factory=lambda: CLISpec("claude", ("-p",)))
    codex_cli: CLISpec = field(
        default_factory=lambda: CLISpec(
            "codex",
            ("exec", "--skip-git-repo-check", "--sandbox", "workspace-write"),
        )
    )
    gemini_cli: CLISpec = field(default_factory=lambda: CLISpec("gemini", ()))

    max_policy: MaxModePolicy = field(default_factory=MaxModePolicy)
    default_roster: dict[str, int] = field(
        default_factory=lambda: {"claude": 1, "codex": 1, "gemini": 1, "grok": 0}
    )
    dedicated_reviewer: str = "rotate"  # rotate | off

    # R15-B — design §9.2: token estimate + abort threshold so a misconfigured
    # roster × rounds combo cannot silently burn budget.
    abort_cost_usd: float = 20.00
    warn_cost_usd: float = 5.00

    @classmethod
    def from_env(cls) -> "Config":
        roster_raw = os.environ.get(
            "ULTRADEBATE_DEFAULT_ROSTER", "claude=1,codex=1,gemini=1,grok=0"
        )
        roster = _parse_roster(roster_raw)
        if not roster:
            roster = {"claude": 1, "codex": 1, "gemini": 1, "grok": 0}

        max_mode = _envflag("ULTRADEBATE_MAX_MODE", "1")
        policy = MaxModePolicy(
            enabled=max_mode,
            # Empty defaults: rely on CLI account/tier defaults unless user pins.
            claude_model=os.environ.get("ULTRADEBATE_CLAUDE_MODEL", ""),
            claude_thinking_tokens=int(
                os.environ.get("ULTRADEBATE_CLAUDE_THINKING_TOKENS", "31999")
            ),
            codex_model=os.environ.get("ULTRADEBATE_CODEX_MODEL", ""),
            codex_reasoning_effort=os.environ.get(
                "ULTRADEBATE_CODEX_REASONING", "xhigh"
            ),
            gemini_model=os.environ.get("ULTRADEBATE_GEMINI_MODEL", ""),
            gemini_thinking_level=os.environ.get(
                "ULTRADEBATE_GEMINI_THINKING", "MAX"
            ),
            grok_model=os.environ.get("ULTRADEBATE_GROK_MODEL", ""),
            grok_reasoning=os.environ.get("ULTRADEBATE_GROK_REASONING", "max"),
        )

        return cls(
            base_dir=_expand(os.environ.get("ULTRADEBATE_BASE", "~/.ultradebate")),
            max_rounds=int(os.environ.get("ULTRADEBATE_MAX_ROUNDS", "5")),
            token_budget=int(os.environ.get("ULTRADEBATE_TOKEN_BUDGET", "300000")),
            per_round_limit=int(os.environ.get("ULTRADEBATE_PER_ROUND_LIMIT", "8000")),
            mode=os.environ.get("ULTRADEBATE_MODE", "lenient"),
            convergence_delta=float(
                os.environ.get("ULTRADEBATE_CONVERGENCE_DELTA", "0.05")
            ),
            mock=_envflag("ULTRADEBATE_MOCK", "0"),
            claude_cli=CLISpec.from_env("CLAUDE", "claude", "-p"),
            codex_cli=CLISpec.from_env(
                "CODEX",
                "codex",
                "exec --skip-git-repo-check --sandbox workspace-write",
            ),
            gemini_cli=CLISpec.from_env("GEMINI", "gemini", ""),
            max_policy=policy,
            default_roster=roster,
            dedicated_reviewer=os.environ.get(
                "ULTRADEBATE_DEDICATED_REVIEWER", "rotate"
            ).strip().lower(),
            abort_cost_usd=float(
                os.environ.get("ULTRADEBATE_ABORT_COST_USD", "20.00")
            ),
            warn_cost_usd=float(
                os.environ.get("ULTRADEBATE_WARN_COST_USD", "5.00")
            ),
        )
