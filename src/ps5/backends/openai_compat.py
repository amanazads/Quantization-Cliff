"""OpenAI-compatible backend, targeting vLLM.

This is the path that makes all four precisions -- including FP8 -- runnable on
one serving stack, which is the strongest available form of experimental control
for PS-5. Launch commands per precision are in README.md.

Deliberately implemented with plain `requests` against /v1/chat/completions
rather than the `openai` SDK: one fewer dependency whose version could differ
between arms, and the raw envelope is easier to store verbatim.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

import requests

from .base import Backend, BackendError, GenerationResult, ToolCall

__all__ = ["OpenAICompatBackend"]

DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"


class OpenAICompatBackend(Backend):
    synthetic = False

    def __init__(
        self,
        model_tag: str,
        options: Optional[Dict[str, Any]] = None,
        name: str = "vllm",
    ):
        super().__init__(name, model_tag, options)
        self.base_url = (self.options.get("base_url") or DEFAULT_BASE_URL).rstrip("/")
        self.api_key = self.options.get("api_key", "EMPTY")
        self._resolved: Optional[Dict[str, Any]] = None

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    # -- lifecycle ---------------------------------------------------------- #

    def health_check(self) -> None:
        try:
            resp = requests.get(f"{self.base_url}/models", headers=self._headers(), timeout=20)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise BackendError(
                f"Cannot reach an OpenAI-compatible server at {self.base_url}: {exc}\n"
                "Start vLLM first; see README.md for the per-precision launch command."
            ) from exc

        served = [m.get("id") for m in resp.json().get("data", [])]
        if self.model_tag not in served:
            raise BackendError(
                f"Model '{self.model_tag}' is not served at {self.base_url}.\n"
                f"Served models: {served}\n\n"
                "Aborting rather than falling back: serving a different checkpoint "
                "under this precision label would invalidate the experiment."
            )
        self._resolved = {"served_models": served}

    def describe(self) -> Dict[str, Any]:
        # Same thinking-status fields as the Ollama backend, so the aggregator's
        # cross-arm check reads one shape regardless of serving stack. On vLLM the
        # flag is a chat-template argument: an unsupported one is ignored rather
        # than rejected, so there is nothing to negotiate and "sent" is simply
        # what was configured. That it cannot be confirmed is stated, not glossed.
        requested = bool(self.options.get("disable_thinking", True))
        return {
            "backend": self.name,
            "model_tag": self.model_tag,
            "base_url": self.base_url,
            "synthetic": False,
            "resolved": self._resolved or {},
            "thinking_disable_requested": requested,
            "thinking_disable_sent": requested,
            "thinking_unsupported_by_model": False,
            "thinking_status": (
                "enable_thinking:false passed via chat_template_kwargs "
                "(silently ignored if the template has no such flag)"
                if requested else "not requested"
            ),
        }

    # -- generation --------------------------------------------------------- #

    def generate(
        self,
        system_prompt: str,
        user_message: str,
        tools: Optional[List[Dict[str, Any]]],
        generation: Any,
        case: Optional[Dict[str, Any]] = None,
    ) -> GenerationResult:
        # `case` is accepted for interface symmetry and DELIBERATELY UNUSED.
        # Conditioning a real backend on the expected answer would cheat the
        # experiment; tests/test_backends.py enforces that it changes nothing.
        del case
        payload: Dict[str, Any] = {
            "model": self.model_tag,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": generation.temperature,
            "top_p": generation.top_p,
            "max_tokens": generation.max_tokens,
            "seed": generation.seed,
            "stream": False,
        }
        if generation.stop:
            payload["stop"] = list(generation.stop)
        if tools:
            payload["tools"] = tools
            # "auto" rather than "required": whether the model decides to call a
            # tool at all is part of what PS-3 measures. Forcing a call would
            # make missed_call_rate and spurious_call_rate unmeasurable.
            payload["tool_choice"] = "auto"
        # vLLM-specific knob, sent only when explicitly configured.
        if "top_k" in self.options or generation.top_k is not None:
            payload.setdefault("extra_body", {})["top_k"] = generation.top_k

        # THINKING MODE OFF -- see the note in backends/ollama.py. On vLLM this is
        # a chat-template argument rather than a sampling parameter, so it is
        # passed through chat_template_kwargs. Sent unconditionally so it cannot
        # differ between arms.
        if self.options.get("disable_thinking", True):
            payload.setdefault("chat_template_kwargs", {})["enable_thinking"] = False

        last_error: Optional[str] = None
        last_kind: Optional[str] = None

        for attempt in range(1, generation.transport_retries + 2):
            started = time.perf_counter()
            try:
                resp = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self._headers(),
                    json=payload,
                    timeout=generation.request_timeout_s,
                )
                latency_ms = (time.perf_counter() - started) * 1000.0
                resp.raise_for_status()
                return self._parse(resp.json(), latency_ms, attempt)

            except requests.Timeout as exc:
                last_error, last_kind = str(exc), "timeout"
            except requests.ConnectionError as exc:
                last_error, last_kind = str(exc), "connection"
            except requests.HTTPError as exc:
                detail = ""
                try:
                    detail = f" | body: {resp.text[:500]}"
                except Exception:
                    pass
                last_error, last_kind = f"{exc}{detail}", "http"
            except ValueError as exc:
                last_error, last_kind = f"invalid JSON envelope: {exc}", "other"

            if attempt <= generation.transport_retries:
                time.sleep(generation.transport_retry_backoff_s * attempt)

        return GenerationResult(
            ok=False, error=last_error, error_kind=last_kind,
            attempts=generation.transport_retries + 1,
        )

    @staticmethod
    def _parse(body: Dict[str, Any], latency_ms: float, attempt: int) -> GenerationResult:
        choices = body.get("choices") or []
        if not choices:
            return GenerationResult(
                ok=False, error="no choices in response", error_kind="empty",
                latency_ms=latency_ms, attempts=attempt, raw_response=body,
            )

        choice = choices[0]
        message = choice.get("message", {}) or {}
        text = message.get("content") or ""

        calls: List[ToolCall] = []
        for raw_call in message.get("tool_calls", []) or []:
            fn = (raw_call or {}).get("function", {}) or {}
            name = fn.get("name")
            raw_args = fn.get("arguments")
            if isinstance(raw_args, str):
                try:
                    calls.append(
                        ToolCall(name=name, arguments=json.loads(raw_args), raw_arguments=raw_args)
                    )
                except json.JSONDecodeError as exc:
                    calls.append(
                        ToolCall(name=name, arguments=None, raw_arguments=raw_args,
                                 parse_error=f"arguments not valid JSON: {exc}")
                    )
            elif isinstance(raw_args, dict):
                calls.append(
                    ToolCall(name=name, arguments=raw_args,
                             raw_arguments=json.dumps(raw_args, ensure_ascii=False))
                )
            else:
                calls.append(
                    ToolCall(name=name, arguments=None, raw_arguments=repr(raw_args),
                             parse_error=f"arguments had unexpected type {type(raw_args).__name__}")
                )

        if not text.strip() and not calls:
            return GenerationResult(
                text=text, ok=False, error="empty completion", error_kind="empty",
                latency_ms=latency_ms, attempts=attempt, raw_response=body,
            )

        usage = body.get("usage", {}) or {}
        return GenerationResult(
            text=text,
            tool_calls=calls,
            finish_reason=choice.get("finish_reason"),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            latency_ms=latency_ms,
            attempts=attempt,
            ok=True,
            raw_response=body,
        )
