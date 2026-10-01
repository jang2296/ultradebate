"""Claude Code CLI worker — uses `claude -p` headless (Anthropic's official auto-mode).

Max-mode policy (R15): when MaxModePolicy.enabled, prepend the literal `ultrathink`
keyword to the prompt (which Claude Code interprets as a 31,999 token thinking
budget signal) and set MAX_THINKING_TOKENS env var as a defensive double-binding.
"""
from __future__ import annotations

import asyncio

from ..config import CLISpec, MaxModePolicy
from ..parser import ParsedResponse, parse_response
from ._subprocess import estimate_tokens, find_executable, run_pipe
from .base import AgentResponse


class ClaudeWorker:
    name = "claude"

    def __init__(
        self,
        spec: CLISpec | None = None,
        max_policy: MaxModePolicy | None = None,
    ):
        self.spec = spec or CLISpec("claude", ("-p",))
        self.max_policy = max_policy or MaxModePolicy()

    async def is_available(self) -> bool:
        return find_executable(self.spec.binary) is not None

    def _prepare(self, prompt: str) -> tuple[str, dict]:
        """Apply max-mode injection. Returns (final_prompt, env_overrides)."""
        env: dict[str, str] = {}
        final = prompt
        if self.max_policy.enabled:
            if self.max_policy.claude_inject_ultrathink and "ultrathink" not in prompt.lower():
                final = "ultrathink\n\n" + prompt
            env["MAX_THINKING_TOKENS"] = str(self.max_policy.claude_thinking_tokens)
        return final, env

    def _final_args(self) -> list[str]:
        """Inject --model when max-mode is on and user hasn't already set it."""
        args = list(self.spec.args)
        if self.max_policy.enabled and self.max_policy.claude_model and not any(
            a in {"--model", "-m"} or a.startswith("--model=") for a in args
        ):
            args.extend(["--model", self.max_policy.claude_model])
        return args

    async def invoke(self, prompt: str, timeout: int = 120) -> AgentResponse:
        final_prompt, env = self._prepare(prompt)
        cmd = [self.spec.binary, *self._final_args()]
        try:
            rc, stdout, stderr, latency = await run_pipe(
                cmd, final_prompt, env=env, timeout=timeout
            )
        except FileNotFoundError:
            return AgentResponse(
                agent=self.name,
                error=f"executable not found: {self.spec.binary!r} — install Claude Code CLI",
            )
        except asyncio.TimeoutError:
            return AgentResponse(agent=self.name, error=f"claude timed out after {timeout}s")

        if rc != 0 and not stdout.strip():
            return AgentResponse(
                agent=self.name,
                error=f"claude exited rc={rc}: {stderr[:300] or '(no stderr)'}",
                raw=stderr,
                latency_ms=latency,
            )

        parsed: ParsedResponse = parse_response(stdout)
        return AgentResponse(
            agent=self.name,
            text=parsed.analysis,
            parsed=parsed,
            tokens_in=estimate_tokens(final_prompt),
            tokens_out=estimate_tokens(stdout),
            latency_ms=latency,
            raw=stdout,
        )

    async def shutdown(self) -> None:
        return None
