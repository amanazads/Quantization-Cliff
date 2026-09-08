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
        #: Filled by health_check from /api/tags, the only endpoint that has them.
        self._tag_digest: Optional[str] = None
        self._tag_size_bytes: Optional[int] = None
        #: Whether `think: false` was sent and accepted. Ollama rejects the
        #: parameter outright on models with no thinking mode, so this is
        #: negotiated once on the first request and then held for the whole arm.
        #: Recorded in metadata so a reader can see which it was.
        self._send_think: bool = bool(self.options.get("disable_thinking", True))
        self._think_rejected: bool = False

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

        models = resp.json().get("models", []) or []
        # /api/tags is the ONLY endpoint that returns the manifest digest;
        # /api/show does not, so a run relying on /api/show alone recorded
        # `digest: null` while the README claimed the weights were pinned by it.
        # Captured here, where it is available, and merged in below.
        self._tag_digest = next(
            (m.get("digest") for m in models if m.get("name") == self.model_tag), None
        )
        self._tag_size_bytes = next(
            (m.get("size") for m in models if m.get("name") == self.model_tag), None
        )

        tags = {m.get("name") for m in models}
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
        than trusting the filename. `quantization_level` is the load-bearing
        field -- it comes from the GGUF header, so it reports what was actually
        loaded rather than what the tag was called. The digest pins the exact
        manifest on top of that, and is fetched from /api/tags because /api/show
        does not return one.
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
            # /api/show first (some builds do return it), then the digest
            # /api/tags gave us during health_check.
            "digest": body.get("digest") or getattr(self, "_tag_digest", None),
            "size_bytes": getattr(self, "_tag_size_bytes", None),
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
            # The specification requires thinking mode off for every run, so
            # whether that was actually applied is recorded, not assumed.
            "thinking_disable_requested": bool(self.options.get("disable_thinking", True)),
            "thinking_disable_sent": self._send_think,
            "thinking_unsupported_by_model": self._think_rejected,
            "thinking_status": (
                "model has no thinking mode; nothing to disable"
                if self._think_rejected else
                "think:false sent and accepted" if self._send_think else
                "not requested"
            ),
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
        # differently at different precisions.
        #
        # Ollama returns 400 for `think` on a model that has no thinking mode, so
        # the parameter cannot be sent unconditionally: doing so would fail every
        # single case and destroy the whole arm. It is negotiated once -- sent on
        # the first request, and if the server rejects it for that reason, dropped
        # for the rest of the arm and recorded in metadata. Dropping it is safe
        # precisely because such a model has no thinking mode to leave on, so the
        # control the specification asks for still holds.
        if self._send_think:
            payload["think"] = False

        last_error: Optional[str] = None
        last_kind: Optional[str] = None

        attempt = 0
        max_attempts = generation.transport_retries + 1
        # Renegotiating `think` is not a transport retry and must not consume the
        # retry budget, so it is granted one extra attempt of its own.
        think_retry_granted = False

        while attempt < max_attempts:
            attempt += 1
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
                # The status line alone ("400 Client Error: Bad Request") says
                # nothing about the cause. Ollama puts the reason in the body, so
                # it is surfaced -- both to diagnose the think case below and so
                # that any other 400 reaches the operator legibly instead of as an
                # anonymous failure repeated across every case in the arm.
                body_text = ""
                try:
                    body_text = (exc.response.text or "")[:500] if exc.response is not None else ""
                except Exception:  # pragma: no cover - defensive
                    body_text = ""
                last_error = f"{exc}{(': ' + body_text) if body_text else ''}"
                last_kind = "http"

                if (
                    not think_retry_granted
                    and self._send_think
                    and "think" in payload
                    and self._is_think_unsupported(exc, body_text)
                ):
                    # Negotiated once per backend instance, so the remaining cases
                    # in this arm go out without the parameter and never pay this
                    # round trip again.
                    self._send_think = False
                    self._think_rejected = True
                    payload.pop("think", None)
                    think_retry_granted = True
                    max_attempts += 1
                    continue  # immediate retry, no backoff: nothing is overloaded
            except ValueError as exc:  # undecodable JSON envelope
                last_error, last_kind = f"invalid JSON envelope: {exc}", "other"

            if attempt < max_attempts:
                time.sleep(generation.transport_retry_backoff_s * attempt)

        return GenerationResult(
            ok=False,
            error=last_error,
            error_kind=last_kind,
            attempts=attempt,
        )

    # -- `think` negotiation ------------------------------------------------- #

    #: Substrings that identify a rejection of the `think` parameter itself, as
    #: opposed to any other 400. Ollama's wording has changed across releases
    #: ("does not support thinking", "thinking is not supported by this model"),
    #: so matching is on the concept, not one exact string.
    _THINK_REJECTION_MARKERS = (
        "does not support thinking",
        "thinking is not supported",
        "not support thinking",
        "thinking not supported",
        '"think"',
        "'think'",
        " think ",
    )

    @classmethod
    def _is_think_unsupported(cls, exc: requests.HTTPError, body_text: str) -> bool:
        """Is this 400 specifically a rejection of `think`, or something else?

        Narrow on purpose. Retrying without `think` after an unrelated 400 would
        silently drop the thinking control and leave the arm running under a
        different configuration from its siblings -- an uncontrolled variable
        introduced by an error handler. So the retry fires only on a 4xx whose
        body actually mentions thinking; anything else fails loudly.
        """
        status = getattr(exc.response, "status_code", None)
        if status is None or not (400 <= int(status) < 500):
            return False
        haystack = f" {body_text.lower()} "
        return any(marker in haystack for marker in cls._THINK_REJECTION_MARKERS)

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
