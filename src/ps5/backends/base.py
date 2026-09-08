"""Backend abstraction.

A backend turns (system prompt, user message, tool schemas, generation config)
into a `GenerationResult`. It is the ONLY layer permitted to know which precision
is being served -- everything downstream is precision-blind, which is what makes
the comparison controlled.

Backends must not:
  * modify the system prompt,
  * modify the tool schemas,
  * vary generation parameters,
  * retry on the basis of response *content*.

Transport-level retries are permitted and are recorded per case.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

__all__ = ["ToolCall", "GenerationResult", "Backend", "BackendError", "get_backend"]


class BackendError(RuntimeError):
    """A transport or configuration failure. Not a model quality signal."""


@dataclass
class ToolCall:
    """A tool call as emitted by the model, before any validation.

    `name` and `arguments` are recorded exactly as received. `parse_error` is set
    when the model emitted something call-shaped that could not be decoded --
    that is a genuine result (a malformed call), not an infrastructure problem,
    so it is carried forward to the scorer rather than raised.
    """

    name: Optional[str]
    arguments: Optional[Dict[str, Any]]
    raw_arguments: Optional[str] = None
    parse_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GenerationResult:
    text: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    finish_reason: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    latency_ms: Optional[float] = None
    attempts: int = 1
    ok: bool = True
    error: Optional[str] = None
    error_kind: Optional[str] = None  # "timeout" | "connection" | "http" | "empty" | "other"
    raw_response: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        return d


class Backend(abc.ABC):
    """Interface every serving backend implements."""

    #: Set True only on backends that fabricate output. Propagates into metadata
    #: and blocks findings-report generation unless explicitly overridden.
    synthetic: bool = False

    def __init__(self, name: str, model_tag: str, options: Optional[Dict[str, Any]] = None):
        self.name = name
        self.model_tag = model_tag
        self.options = options or {}

    @abc.abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_message: str,
        tools: Optional[List[Dict[str, Any]]],
        generation: Any,
        case: Optional[Dict[str, Any]] = None,
    ) -> GenerationResult:
        """Produce one completion. Must not raise on model misbehaviour.

        `case` carries the evaluation case's metadata, including its expected
        answer. **Real serving backends MUST ignore it entirely** -- conditioning
        generation on the expected answer would be cheating the experiment. It
        exists solely so the deterministic MockBackend can synthesise
        pipeline-validation output, and `tests/test_backends.py` asserts that the
        real backends' request payloads are unaffected by it.
        """

    def describe(self) -> Dict[str, Any]:
        """Runtime facts about the served model, resolved at run time.

        Subclasses override to report the digest/revision actually loaded, which
        is what pins the experiment to a specific set of weights.
        """
        return {"backend": self.name, "model_tag": self.model_tag, "synthetic": self.synthetic}

    def health_check(self) -> None:
        """Raise BackendError if the model cannot be served. Called before a run.

        Failing here is strongly preferred to discovering mid-run that a tag was
        never pulled, because a partially completed arm invites the temptation to
        compare unequal sample sizes.
        """
        return None

    def unload(self) -> None:
        """Release the model's memory after an arm finishes. Best-effort, never raises.

        Arms run sequentially, so without this the previous arm's weights can
        still be resident when the next one loads. On a memory-constrained
        machine that pushes the later arms into swap, which changes their
        latency and can cause timeouts -- an uncontrolled variable introduced by
        the hardware rather than by the quantization being tested.
        """
        return None


def get_backend(name: str, model_tag: str, options: Optional[Dict[str, Any]] = None) -> Backend:
    """Factory. Imports lazily so an unused backend's deps are not required."""
    options = options or {}
    if name == "ollama":
        from .ollama import OllamaBackend
        return OllamaBackend(model_tag, options)
    if name == "mock":
        from .mock import MockBackend
        return MockBackend(model_tag, options)
    raise BackendError(
        f"Unknown backend '{name}'. Known backends: ollama, mock."
    )
