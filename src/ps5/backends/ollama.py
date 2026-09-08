"""Ollama backend (llama.cpp / GGUF).

Serves Q4 and Q8 genuinely. Serves the reference arm as F16, which is NOT BF16 --
see DEV-BF16-OLLAMA-1 in configs/experiments/bf16.yaml. Cannot serve FP8 at all;
the config marks that arm unavailable and the runner refuses it.

Uses the /api/chat endpoint with native `tools`, so tool calls come back as
structured objects rather than needing to be scraped out of prose. When the model
emits a call in prose anyway -- common at low precision, and itself a finding --
the fallback extractor in scoring/toolcall_parse.py handles it.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

import requests

from .base import Backend, BackendError, GenerationResult, ToolCall

__all__ = ["OllamaBackend"]

DEFAULT_HOST = "http://127.0.0.1:11434"


class OllamaBackend(Backend):
    synthetic = False

    def __init__(self, model_tag: str, options: Optional[Dict[str, Any]] = None):
        super().__init__("ollama", model_tag, options)
        self.host = (self.options.get("host") or DEFAULT_HOST).rstrip("/")
        self._resolved: Optional[Dict[str, Any]] = None

    # -- lifecycle ---------------------------------------------------------- #

    def health_check(self) -> None:
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=15)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise BackendError(
                f"Cannot reach Ollama at {self.host}: {exc}\n"
                "Start it with `ollama serve`, or set backend option `host`."
            ) from exc

        tags = {m.get("name") for m in resp.json().get("models", [])}
        if self.model_tag not in tags:
            raise BackendError(
                f"Model tag '{self.model_tag}' is not present on this Ollama host.\n"
                f"Available: {sorted(t for t in tags if t)}\n\n"
                f"Pull it first:  ollama pull {self.model_tag}\n\n"
                "The run is aborted rather than silently falling back to a "
                "different tag, because serving a different quantization under "
                "this label would invalidate the experiment."
            )
        self._resolve_model_identity()

    def _resolve_model_identity(self) -> None:
        """Pin the exact weights in use: digest, parameter count, real quant type.

        This is what lets a reader verify that the Q4 arm really was Q4, rather
        than trusting the filename.
        """
        try:
            resp = requests.post(
                f"{self.host}/api/show", json={"model": self.model_tag}, timeout=30
            )
            resp.raise_for_status()
            body = resp.json()
        except (requests.RequestException, ValueError) as exc:
            self._resolved = {"resolve_error": str(exc)}
            return

        details = body.get("details", {}) or {}
        info = body.get("model_info", {}) or {}
        self._resolved = {
            "digest": body.get("digest"),
            "parameter_size": details.get("parameter_size"),
            "quantization_level": details.get("quantization_level"),
            "family": details.get("family"),
            "format": details.get("format"),
            "architecture": info.get("general.architecture"),
            "file_type": info.get("general.file_type"),
            "context_length": info.get(
                f"{info.get('general.architecture', 'unknown')}.context_length"
            ),
        }

    def describe(self) -> Dict[str, Any]:
        if self._resolved is None:
            self._resolve_model_identity()
        return {
            "backend": self.name,
            "model_tag": self.model_tag,
            "host": self.host,
            "synthetic": False,
            "resolved": self._resolved or {},
        }

    def unload(self) -> None:
        """Ask Ollama to evict this model immediately (`keep_alive: 0`).

        Ollama otherwise keeps a model resident for several minutes after the
        last request. With arms running back to back on a machine with limited
        unified memory, that can leave the previous arm's weights in memory while
        the next arm loads, pushing it into swap. Swapping changes latency and
        can trigger timeouts, which would show up as a difference between
        precisions that quantization did not cause.

        Best-effort: a failure here costs memory, not correctness, so it is
        swallowed rather than allowed to fail a completed run.
        """
        try:
            requests.post(
                f"{self.host}/api/generate",
                json={"model": self.model_tag, "keep_alive": 0},
                timeout=30,
            )
        except requests.RequestException:
            pass

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
            "stream": False,
            "options": {
                "temperature": generation.temperature,
                "top_p": generation.top_p,
                "top_k": generation.top_k,
                "seed": generation.seed,
                "num_predict": generation.max_tokens,
                # Fixed so that the KV/context configuration is identical across
                # arms; a differing context window would be an uncontrolled variable.
                "num_ctx": int(self.options.get("num_ctx", 4096)),
            },
        }
        if generation.stop:
            payload["options"]["stop"] = list(generation.stop)
        if tools:
            payload["tools"] = tools

        # THINKING MODE OFF -- required by the specification's environment setup:
        # "Thinking mode must be disabled for every Track 1 run." The Qwen3.x line
        # ships with it on and it generates roughly 2-5x more tokens, which would
        # inflate token counts, distort latency, and interact with max_tokens
        # differently at different precisions. Sent unconditionally so it cannot
        # be on for one arm and off for another; harmless on non-thinking models.
        if self.options.get("disable_thinking", True):
            payload["think"] = False

        last_error: Optional[str] = None
        last_kind: Optional[str] = None

        for attempt in range(1, generation.transport_retries + 2):
            started = time.perf_counter()
            try:
                resp = requests.post(
                    f"{self.host}/api/chat",
                    json=payload,
                    timeout=generation.request_timeout_s,
                )
                latency_ms = (time.perf_counter() - started) * 1000.0
                resp.raise_for_status()
                body = resp.json()
                return self._parse(body, latency_ms, attempt)

            except requests.Timeout as exc:
                last_error, last_kind = str(exc), "timeout"
            except requests.ConnectionError as exc:
                last_error, last_kind = str(exc), "connection"
            except requests.HTTPError as exc:
                last_error, last_kind = str(exc), "http"
            except ValueError as exc:  # undecodable JSON envelope
                last_error, last_kind = f"invalid JSON envelope: {exc}", "other"

            if attempt <= generation.transport_retries:
                time.sleep(generation.transport_retry_backoff_s * attempt)

        return GenerationResult(
            ok=False,
            error=last_error,
            error_kind=last_kind,
            attempts=generation.transport_retries + 1,
        )

    @staticmethod
    def _parse(body: Dict[str, Any], latency_ms: float, attempt: int) -> GenerationResult:
        message = body.get("message", {}) or {}
        text = message.get("content", "") or ""

        calls: List[ToolCall] = []
        for raw_call in message.get("tool_calls", []) or []:
            fn = (raw_call or {}).get("function", {}) or {}
            name = fn.get("name")
            args = fn.get("arguments")

            # Ollama usually returns arguments already decoded; some builds and
            # some models return a JSON string. Both are handled, and an
            # undecodable string is preserved as a malformed call rather than
            # discarded -- discarding it would understate malformed_argument_rate.
            if isinstance(args, str):
                try:
                    parsed = json.loads(args)
                    calls.append(ToolCall(name=name, arguments=parsed, raw_arguments=args))
                except json.JSONDecodeError as exc:
                    calls.append(
                        ToolCall(name=name, arguments=None, raw_arguments=args,
                                 parse_error=f"arguments not valid JSON: {exc}")
                    )
            elif isinstance(args, dict):
                calls.append(
                    ToolCall(name=name, arguments=args, raw_arguments=json.dumps(args, ensure_ascii=False))
                )
            else:
                calls.append(
                    ToolCall(name=name, arguments=None, raw_arguments=repr(args),
                             parse_error=f"arguments had unexpected type {type(args).__name__}")
                )

        # An empty completion with no call is a transport-level non-result, not a
        # model decision to stay silent, so it is marked not-ok and excluded from
        # scoring denominators via generation_failure_rate.
        if not text.strip() and not calls:
            return GenerationResult(
                text=text, tool_calls=[], ok=False, error="empty completion",
                error_kind="empty", latency_ms=latency_ms, attempts=attempt,
                raw_response=body,
            )

        return GenerationResult(
            text=text,
            tool_calls=calls,
            finish_reason=body.get("done_reason"),
            prompt_tokens=body.get("prompt_eval_count"),
            completion_tokens=body.get("eval_count"),
            latency_ms=latency_ms,
            attempts=attempt,
            ok=True,
            raw_response=body,
        )
