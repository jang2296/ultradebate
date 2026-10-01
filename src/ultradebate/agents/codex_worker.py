"""OpenAI Codex CLI worker.

Why pty? `codex exec` (issue #19945, 2026) silently exits with empty stdout when
its stdio is detached from a controlling TTY. MCP servers spawned over stdio
*are* TTY-detached. The pty wrapper presents a real TTY to the child process and
restores normal output behavior.

Default args (Codex 0.130+): `exec --skip-git-repo-check --sandbox workspace-write`.
Older `--full-auto` is deprecated; the modern equivalent is the explicit sandbox
flag. Override via `ULTRADEBATE_CODEX_ARGS` if your build differs.

Prompt delivery: the prompt is passed as a positional CLI argument
(`codex exec [PROMPT]`) rather than via stdin. PTY-piped stdin proved unreliable
because `codex exec` interprets it as interactive input without a clear EOF signal.
Argv has plenty of headroom (Linux ARG_MAX ≥ 128KB) for our 5-10KB prompts.

Max-mode policy (R15): when MaxModePolicy.enabled and the reasoning-effort knob
is not already set in user args, append `-c model_reasoning_effort='<level>'`.
The flag form `--reasoning-effort` was removed in 0.130+; only the TOML override
works on current builds.
"""
from __future__ import annotations

import asyncio
import json

from ..config import CLISpec, MaxModePolicy
from ..parser import parse_response
from ._subprocess import estimate_tokens, find_executable, run_pty
from .base import AgentResponse


class CodexWorker:
    name = "codex"

    def __init__(
        self,
        spec: CLISpec | None = None,
        max_policy: MaxModePolicy | None = None,
    ):
        self.spec = spec or CLISpec(
            "codex",
            ("exec", "--skip-git-repo-check", "--sandbox", "workspace-write"),
        )
        self.max_policy = max_policy or MaxModePolicy()

    async def is_available(self) -> bool:
        return find_executable(self.spec.binary) is not None

    def _final_args(self) -> list[str]:
        """Inject reasoning_effort and --model when max-mode is on.

        Codex 0.130+ exposes reasoning effort via TOML config override (`-c key=value`)
        rather than a dedicated flag. We detect either form so users on older
        builds can still override via ULTRADEBATE_CODEX_ARGS.

        TOML safety: we encode the value via json.dumps so any quote/backslash/
        unicode in the level string is escaped to valid TOML basic-string syntax
        (TOML basic strings share JSON's escape rules for the relevant subset).
        """
        args = list(self.spec.args)
        if self.max_policy.enabled:
            already_reasoning = any(
                "model_reasoning_effort" in a or a.startswith("--reasoning-effort")
                for a in args
            )
            if not already_reasoning:
                level_toml = json.dumps(self.max_policy.codex_reasoning_effort)
                args.extend(["-c", f"model_reasoning_effort={level_toml}"])
            already_model = any(
                a in {"--model", "-m"} or a.startswith("--model=") for a in args
            )
            if self.max_policy.codex_model and not already_model:
                args.extend(["--model", self.max_policy.codex_model])
        return args

    async def invoke(self, prompt: str, timeout: int = 180) -> AgentResponse:
        # Pass prompt as positional CLI arg (not stdin) — see module docstring.
        cmd = [self.spec.binary, *self._final_args(), prompt]
        # Double-bind reasoning effort: env var alongside TOML override. Some
        # Codex builds honor the env path, others honor `-c`; design doc §2.2
        # accepts both. Provides redundancy without conflict.
        env: dict[str, str] = {}
        if self.max_policy.enabled and self.max_policy.codex_reasoning_effort:
            env["CODEX_REASONING_EFFORT"] = self.max_policy.codex_reasoning_effort
        try:
            rc, output, latency = await run_pty(cmd, "", env=env, timeout=timeout)
        except FileNotFoundError:
            return AgentResponse(
                agent=self.name,
                error=(
                    f"executable not found: {self.spec.binary!r} — Codex CLI not installed yet. "
                    "Run `scripts/install_codex.sh` once OpenAI publishes the package."
                ),
            )
        except asyncio.TimeoutError:
            return AgentResponse(agent=self.name, error=f"codex timed out after {timeout}s")

        if not output.strip():
            return AgentResponse(
                agent=self.name,
                error=(
                    f"codex returned empty output (rc={rc}). "
                    "Possible causes: TTY-detach bug not mitigated, network failure, "
                    "or auth missing. Verify OPENAI_API_KEY and `codex --version`."
                ),
                latency_ms=latency,
            )

        parsed = parse_response(output)
        return AgentResponse(
            agent=self.name,
            text=parsed.analysis,
            parsed=parsed,
            tokens_in=estimate_tokens(prompt),
            tokens_out=estimate_tokens(output),
            latency_ms=latency,
            raw=output,
        )

    async def shutdown(self) -> None:
        return None
