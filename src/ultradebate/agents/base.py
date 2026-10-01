"""Worker contract — kept stable across Pattern A (per-round spawn) and Pattern B (long-lived pty)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ..parser import ParsedResponse


@dataclass
class AgentResponse:
    """Per-agent output from a single Worker.invoke() call.

    Note: this dataclass intentionally does NOT carry session-level metadata
    such as `cost_estimate`. R15-B (cost-pre-announcement) lives at the
    session boundary — see `server.py::_create_session` which returns
    `cost_estimate` in the session-create response dict (per design §9.2),
    NOT here.
    """
    agent: str
    text: str = ""
    parsed: ParsedResponse = field(default_factory=ParsedResponse)
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: int = 0
    raw: str = ""
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


class AgentWorker(Protocol):
    """Async worker that takes a fully-rendered prompt and returns a parsed response."""

    name: str

    async def is_available(self) -> bool:
        """True iff the underlying CLI is on PATH (or this worker is a mock)."""
        ...

    async def invoke(self, prompt: str, timeout: int = 120) -> AgentResponse:
        """Run one round. Must never raise — embed errors in AgentResponse.error."""
        ...

    async def shutdown(self) -> None:
        """Release any held resources (no-op for stateless Pattern A workers)."""
        ...
