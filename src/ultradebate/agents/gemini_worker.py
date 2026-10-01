"""Google Gemini CLI worker — pipes prompt via stdin (auto-headless on TTY-detach).

Max-mode policy (R15): the `--thinking-level` flag exists only on newer Gemini
CLI builds (≥ 1.x). On 0.29.x it triggers a yargs error. Therefore by default we
DO NOT auto-inject the flag — instead we rely on the prompt-level "ultrathink"
keyword baked into ROUND_PROMPT_TEMPLATE (which Gemini honors via its built-in
"think hard"/"think deeply" trigger pattern). Users on a newer CLI can opt in by
setting `ULTRADEBATE_GEMINI_ARGS="--thinking-level MAX"` — `_final_args()` will
detect the user-supplied flag and not double-inject.
"""
from __future__ import annotations

import asyncio

from ..config import CLISpec, MaxModePolicy
from ..parser import parse_response
from ._subprocess import estimate_tokens, find_executable, run_pipe
from .base import AgentResponse


class GeminiWorker:
    name = "gemini"

    def __init__(
        self,
        spec: CLISpec | None = None,
        max_policy: MaxModePolicy | None = None,
    ):
        # Default args = (); piping prompt via stdin auto-triggers headless mode
        # in Gemini CLI 0.29.x (TTY-detached detection). The `-p` flag in this
        # version requires a positional value, which doesn't fit our stdin pattern.
        self.spec = spec or CLISpec("gemini", ())
        self.max_policy = max_policy or MaxModePolicy()

    async def is_available(self) -> bool:
        return find_executable(self.spec.binary) is not None

    def _final_args(self) -> list[str]:
        """Inject --model when max-mode is on. The --thinking-level flag is
        not auto-injected (unstable across Gemini CLI versions); max-mode for
        Gemini relies on the prompt-level "ultrathink" keyword.
        """
        args = list(self.spec.args)
        if self.max_policy.enabled and self.max_policy.gemini_model and not any(
            a in {"--model", "-m"} or a.startswith("--model=") for a in args
        ):
            args.extend(["--model", self.max_policy.gemini_model])
        return args

    async def invoke(self, prompt: str, timeout: int = 120) -> AgentResponse:
        cmd = [self.spec.binary, *self._final_args()]
        try:
            rc, stdout, stderr, latency = await run_pipe(cmd, prompt, timeout=timeout)
        except FileNotFoundError:
            return AgentResponse(
                agent=self.name,
                error=f"executable not found: {self.spec.binary!r} — install Gemini CLI",
            )
        except asyncio.TimeoutError:
            return AgentResponse(agent=self.name, error=f"gemini timed out after {timeout}s")

        if rc != 0 and not stdout.strip():
            return AgentResponse(
                agent=self.name,
                error=f"gemini exited rc={rc}: {stderr[:300] or '(no stderr)'}",
                raw=stderr,
                latency_ms=latency,
            )

        parsed = parse_response(stdout)
        return AgentResponse(
            agent=self.name,
            text=parsed.analysis,
            parsed=parsed,
            tokens_in=estimate_tokens(prompt),
            tokens_out=estimate_tokens(stdout),
            latency_ms=latency,
            raw=stdout,
        )

    async def shutdown(self) -> None:
        return None
