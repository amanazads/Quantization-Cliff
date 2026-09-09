"""Backend contract tests.

Three source comments in `backends/` promise this file enforces things; it does.

The properties under test are experimental-integrity properties, not merely
functional ones:

  * a real backend must not condition generation on the expected answer;
  * the thinking-mode control must be honestly recorded, never silently dropped;
  * a rejected `think` parameter must not be allowed to destroy an entire arm;
  * an unrelated HTTP 400 must NOT be quietly "fixed" by changing the request,
    because that would leave one arm running under a different configuration
    from its siblings.

No network. A stub Session-level `requests.post` stands in for the server, so
these run in CI on a machine with no Ollama installed.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Dict, List, Optional

import pytest
import requests

from ps5.backends.base import get_backend
from ps5.backends.ollama import OllamaBackend
from ps5.config import GenerationConfig

from conftest import PTP_CASE


# A generation config with retries and backoff that cost no wall-clock time.
GEN = GenerationConfig(transport_retries=2, transport_retry_backoff_s=0.0)

SYSTEM = "You are a collections agent for Arthik Finance."
USER = "I can pay 5000 on the 12th."

TOOLS = [{
    "type": "function",
    "function": {"name": "capture_ptp", "description": "x", "parameters": {"type": "object"}},
}]


# --------------------------------------------------------------------------- #
# fake transport
# --------------------------------------------------------------------------- #

class FakeResponse:
    def __init__(self, status_code: int, body: Any, text: Optional[str] = None):
        self.status_code = status_code
        self._body = body
        self.text = text if text is not None else json.dumps(body)

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            err = requests.HTTPError(
                f"{self.status_code} Client Error: Bad Request for url: http://fake/api/chat"
            )
            err.response = self
            raise err


#: What `/api/show` returns. Housekeeping, not part of any assertion -- it is
#: answered generically so that `describe()` can be called inside a test without
#: perturbing the scripted sequence of generation requests.
#: Deliberately carries NO "digest" key -- that matches what Ollama's /api/show
#: actually returns, and is the whole reason the digest has to come from
#: /api/tags. A fixture that invented one would have hidden the bug.
SHOW_BODY = {
    "details": {"parameter_size": "4.0B", "quantization_level": "Q8_0",
                "family": "qwen3", "format": "gguf"},
    "model_info": {"general.architecture": "qwen3", "qwen3.context_length": 32768},
}


class Recorder:
    """Stands in for `requests.post`.

    Only generation requests (`/api/chat`, `/v1/chat/completions`) are scripted
    and recorded; `/api/show` and the `/api/generate` unload poke are answered
    generically, since they are housekeeping rather than the behaviour under test.
    """

    def __init__(self, responses: List[Any]):
        #: One entry per expected generation call. An Exception entry is raised.
        self._responses = list(responses)
        self.payloads: List[Dict[str, Any]] = []
        self.urls: List[str] = []

    def __call__(self, url, json=None, **kwargs):  # noqa: A002 - mirrors requests' signature
        if url.endswith("/api/show"):
            return FakeResponse(200, SHOW_BODY)
        if url.endswith("/api/generate"):  # unload
            return FakeResponse(200, {})

        self.urls.append(url)
        # Deep-copied: the backend reuses and mutates one payload dict across
        # attempts, so holding a reference would rewrite history and hide exactly
        # the before/after difference these tests exist to check. A copy is also
        # what the wire really carries.
        self.payloads.append(copy.deepcopy(json))
        if not self._responses:
            raise AssertionError(f"unexpected extra request #{len(self.payloads)} to {url}")
        nxt = self._responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    @property
    def calls(self) -> int:
        return len(self.payloads)


def chat_ok(text: str = "Noted, I will record that.") -> FakeResponse:
    return FakeResponse(200, {
        "message": {"role": "assistant", "content": text},
        "done_reason": "stop",
        "prompt_eval_count": 100,
        "eval_count": 20,
    })


#: Real wording observed from Ollama when `think` is sent to a model that has no
#: thinking mode. Matched on the concept rather than the exact string, because
#: the phrasing has changed between releases.
THINK_REJECTED = FakeResponse(
    400,
    {"error": "\"qwen2.5:1.5b-instruct-q8_0\" does not support thinking"},
    text='{"error":"\\"qwen2.5:1.5b-instruct-q8_0\\" does not support thinking"}',
)

#: A 400 that has nothing to do with thinking. Must NOT trigger renegotiation.
UNRELATED_400 = FakeResponse(
    400,
    {"error": "invalid options: num_ctx must be a positive integer"},
    text='{"error":"invalid options: num_ctx must be a positive integer"}',
)


def ollama_post(monkeypatch, recorder: Recorder) -> Recorder:
    import ps5.backends.ollama as mod
    monkeypatch.setattr(mod.requests, "post", recorder)
    return recorder


# --------------------------------------------------------------------------- #
# the expected answer must not reach the wire
# --------------------------------------------------------------------------- #

def test_ollama_ignores_the_case_metadata_entirely(monkeypatch):
    """A real backend conditioned on the expected answer would cheat PS-5.

    The check is on the bytes actually sent, not on the returned text: a backend
    could use `case` to pick a retry strategy or a stop sequence and still return
    plausible prose. Identical payloads is the property that matters.
    """
    rec_without = ollama_post(monkeypatch, Recorder([chat_ok()]))
    b1 = OllamaBackend("qwen2.5:1.5b-instruct-q8_0")
    b1.generate(SYSTEM, USER, TOOLS, GEN, case=None)

    rec_with = ollama_post(monkeypatch, Recorder([chat_ok()]))
    b2 = OllamaBackend("qwen2.5:1.5b-instruct-q8_0")
    b2.generate(SYSTEM, USER, TOOLS, GEN, case=PTP_CASE)

    assert rec_without.payloads == rec_with.payloads, (
        "the Ollama request payload changed when the expected answer was supplied; "
        "a real backend must be blind to it"
    )
    # And specifically: no fragment of the expected answer appears anywhere.
    wire = json.dumps(rec_with.payloads[0])
    assert "capture_ptp" in wire, "sanity: the tool schema itself is legitimately present"
    assert "2026-09-12" not in wire
    assert "promised_date" not in wire


def test_the_prompt_reaches_the_wire_unmodified(monkeypatch):
    """Backends must not touch the baseline prompt -- §6.4 is frozen."""
    rec = ollama_post(monkeypatch, Recorder([chat_ok()]))
    OllamaBackend("qwen2.5:1.5b-instruct-q8_0").generate(SYSTEM, USER, TOOLS, GEN)
    messages = rec.payloads[0]["messages"]
    assert messages[0] == {"role": "system", "content": SYSTEM}
    assert messages[1] == {"role": "user", "content": USER}


# --------------------------------------------------------------------------- #
# thinking-mode negotiation
# --------------------------------------------------------------------------- #

def test_think_false_is_sent_by_default(monkeypatch):
    rec = ollama_post(monkeypatch, Recorder([chat_ok()]))
    b = OllamaBackend("qwen2.5:1.5b-instruct-q8_0")
    res = b.generate(SYSTEM, USER, TOOLS, GEN)
    assert res.ok
    assert rec.payloads[0]["think"] is False
    assert b.describe()["thinking_disable_sent"] is True
    assert b.describe()["thinking_unsupported_by_model"] is False


def test_a_rejected_think_parameter_does_not_destroy_the_arm(monkeypatch):
    """The regression this test exists for.

    Against a server that rejects `think`, every case in the arm previously
    failed with an opaque 400 -- 200/200 lost, and the operator was told only
    "400 Client Error: Bad Request". One arm silently missing is worse than a
    crash, because the remaining three still produce a comparison table.
    """
    rec = ollama_post(monkeypatch, Recorder([THINK_REJECTED, chat_ok()]))
    b = OllamaBackend("qwen2.5:1.5b-instruct-q8_0")
    res = b.generate(SYSTEM, USER, TOOLS, GEN)

    assert res.ok, "the request should have succeeded on the retry without `think`"
    assert rec.calls == 2
    assert rec.payloads[0]["think"] is False
    assert "think" not in rec.payloads[1]
    # Everything else about the request is unchanged -- only `think` was dropped.
    assert {k: v for k, v in rec.payloads[0].items() if k != "think"} == rec.payloads[1]


def test_the_think_decision_is_negotiated_once_and_held_for_the_arm(monkeypatch):
    """200 cases must not each pay a failed round trip to rediscover this."""
    rec = ollama_post(monkeypatch, Recorder([THINK_REJECTED, chat_ok(), chat_ok(), chat_ok()]))
    b = OllamaBackend("qwen2.5:1.5b-instruct-q8_0")
    for _ in range(3):
        assert b.generate(SYSTEM, USER, TOOLS, GEN).ok

    assert rec.calls == 4, "only the first case should have paid the rejection"
    assert all("think" not in p for p in rec.payloads[1:])


def test_a_rejected_think_parameter_is_recorded_not_hidden(monkeypatch):
    """The specification requires thinking off; metadata must say what happened."""
    ollama_post(monkeypatch, Recorder([THINK_REJECTED, chat_ok()]))
    b = OllamaBackend("qwen2.5:1.5b-instruct-q8_0")
    b.generate(SYSTEM, USER, TOOLS, GEN)

    d = b.describe()
    assert d["thinking_disable_requested"] is True
    assert d["thinking_disable_sent"] is False
    assert d["thinking_unsupported_by_model"] is True
    assert "no thinking mode" in d["thinking_status"]


def test_renegotiation_does_not_consume_the_transport_retry_budget(monkeypatch):
    """Dropping `think` is a configuration fix, not a flaky-network retry.

    If it ate a retry, a case that hit the rejection *and* a real timeout would
    fail where an identical case later in the arm would have survived -- making
    the failure rate depend on position in the run.
    """
    rec = ollama_post(monkeypatch, Recorder([
        THINK_REJECTED,                      # negotiation, free
        requests.Timeout("timed out"),       # transport retry 1
        requests.Timeout("timed out"),       # transport retry 2
        chat_ok(),                           # final attempt succeeds
    ]))
    b = OllamaBackend("qwen2.5:1.5b-instruct-q8_0")
    res = b.generate(SYSTEM, USER, TOOLS, GEN)
    assert res.ok, "two timeouts is exactly the configured retry budget"
    assert rec.calls == 4


def test_an_unrelated_400_is_not_papered_over(monkeypatch):
    """Retrying without `think` after any 400 would change the arm's configuration.

    An error handler must not introduce an uncontrolled variable. A 400 that says
    nothing about thinking fails loudly, with `think` still in place.
    """
    rec = ollama_post(monkeypatch, Recorder([UNRELATED_400, UNRELATED_400, UNRELATED_400]))
    b = OllamaBackend("qwen2.5:1.5b-instruct-q8_0")
    res = b.generate(SYSTEM, USER, TOOLS, GEN)

    assert not res.ok
    assert rec.calls == 3, "three transport attempts, no bonus renegotiation attempt"
    assert all(p["think"] is False for p in rec.payloads)
    assert b._send_think is True
    assert b.describe()["thinking_unsupported_by_model"] is False


def test_the_http_error_body_reaches_the_operator(monkeypatch):
    """`400 Client Error: Bad Request` alone is not a diagnosable message."""
    ollama_post(monkeypatch, Recorder([UNRELATED_400] * 3))
    res = OllamaBackend("qwen2.5:1.5b-instruct-q8_0").generate(SYSTEM, USER, TOOLS, GEN)
    assert res.error_kind == "http"
    assert "num_ctx must be a positive integer" in res.error


def test_thinking_can_be_disabled_by_configuration(monkeypatch):
    rec = ollama_post(monkeypatch, Recorder([chat_ok()]))
    b = OllamaBackend("qwen2.5:1.5b-instruct-q8_0", {"disable_thinking": False})
    b.generate(SYSTEM, USER, TOOLS, GEN)
    assert "think" not in rec.payloads[0]
    assert b.describe()["thinking_status"] == "not requested"


# --------------------------------------------------------------------------- #
# generation-parameter fidelity
# --------------------------------------------------------------------------- #

def test_decoding_parameters_are_passed_through_verbatim(monkeypatch):
    """A quiet difference in decoding between arms would invalidate everything."""
    rec = ollama_post(monkeypatch, Recorder([chat_ok()]))
    OllamaBackend("qwen2.5:1.5b-instruct-q8_0").generate(SYSTEM, USER, TOOLS, GEN)
    opts = rec.payloads[0]["options"]
    assert opts["temperature"] == GEN.temperature == 0.0
    assert opts["top_p"] == GEN.top_p
    assert opts["top_k"] == GEN.top_k
    assert opts["seed"] == GEN.seed
    assert opts["num_predict"] == GEN.max_tokens
    assert opts["num_ctx"] == 4096, "context window fixed across arms"
    assert rec.payloads[0]["stream"] is False


def test_context_window_is_identical_across_precision_arms(monkeypatch):
    """num_ctx is an uncontrolled variable if it is allowed to drift per arm."""
    seen = set()
    for tag in ("qwen2.5:1.5b-instruct-q4_K_M", "qwen2.5:1.5b-instruct-q8_0", "qwen2.5:1.5b-instruct-fp16"):
        rec = ollama_post(monkeypatch, Recorder([chat_ok()]))
        OllamaBackend(tag).generate(SYSTEM, USER, TOOLS, GEN)
        seen.add(rec.payloads[0]["options"]["num_ctx"])
    assert len(seen) == 1


# --------------------------------------------------------------------------- #
# response parsing
# --------------------------------------------------------------------------- #

def test_tool_call_arguments_as_a_json_string_are_decoded(monkeypatch):
    ollama_post(monkeypatch, Recorder([FakeResponse(200, {
        "message": {"content": "", "tool_calls": [
            {"function": {"name": "capture_ptp",
                          "arguments": '{"promised_amount": 5000}'}}]},
    })]))
    res = OllamaBackend("m").generate(SYSTEM, USER, TOOLS, GEN)
    assert res.ok
    assert res.tool_calls[0].arguments == {"promised_amount": 5000}
    assert res.tool_calls[0].parse_error is None


def test_undecodable_arguments_are_preserved_as_malformed_not_dropped(monkeypatch):
    """Discarding them would understate malformed_argument_rate -- a headline metric.

    At low precision this is exactly the failure mode PS-5 is looking for, so it
    must survive into the record rather than vanishing into an exception handler.
    """
    ollama_post(monkeypatch, Recorder([FakeResponse(200, {
        "message": {"content": "", "tool_calls": [
            {"function": {"name": "capture_ptp", "arguments": '{"promised_amount": 500'}}]},
    })]))
    res = OllamaBackend("m").generate(SYSTEM, USER, TOOLS, GEN)
    assert res.ok, "a malformed call is a model result, not a transport failure"
    assert len(res.tool_calls) == 1
    assert res.tool_calls[0].arguments is None
    assert res.tool_calls[0].raw_arguments == '{"promised_amount": 500'
    assert "not valid JSON" in res.tool_calls[0].parse_error


def test_an_empty_completion_is_a_non_result_not_a_silent_pass(monkeypatch):
    ollama_post(monkeypatch, Recorder([FakeResponse(200, {"message": {"content": "   "}})]))
    res = OllamaBackend("m").generate(SYSTEM, USER, TOOLS, GEN)
    assert not res.ok
    assert res.error_kind == "empty"


def test_transport_failures_are_returned_not_raised(monkeypatch):
    """The runner must record a failure per case, never abort the arm mid-way."""
    rec = ollama_post(monkeypatch, Recorder([requests.ConnectionError("refused")] * 3))
    res = OllamaBackend("m").generate(SYSTEM, USER, TOOLS, GEN)
    assert not res.ok
    assert res.error_kind == "connection"
    assert res.attempts == rec.calls == 3


# --------------------------------------------------------------------------- #
# factory
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# weights provenance
# --------------------------------------------------------------------------- #

def test_the_weights_digest_is_captured_from_the_tags_endpoint(monkeypatch):
    """The README claims the exact weights are pinned into metadata. They were not.

    Only /api/tags returns the manifest digest; /api/show does not. Resolving
    identity from /api/show alone recorded `digest: null` on every real run
    while the documentation said otherwise -- an asserted control rather than a
    verified one, which is the failure mode this repository exists to avoid.
    """
    import ps5.backends.ollama as mod

    digest = "sha256:" + "cd" * 32
    tags = {"models": [{"name": "qwen2.5:1.5b-instruct-q8_0",
                        "digest": digest, "size": 1646315}]}

    monkeypatch.setattr(mod.requests, "get", lambda *a, **k: FakeResponse(200, tags))
    monkeypatch.setattr(mod.requests, "post", Recorder([]))

    b = OllamaBackend("qwen2.5:1.5b-instruct-q8_0")
    b.health_check()

    resolved = b.describe()["resolved"]
    assert resolved["digest"] == digest, "the digest must reach metadata.json"
    assert resolved["quantization_level"] == "Q8_0", (
        "the GGUF header's own quantization level is the load-bearing check: it "
        "reports what was loaded, not what the tag was named")


def test_a_missing_tag_is_refused_before_anything_runs(monkeypatch):
    import ps5.backends.ollama as mod
    from ps5.backends.base import BackendError

    monkeypatch.setattr(mod.requests, "get", lambda *a, **k: FakeResponse(
        200, {"models": [{"name": "some:other-tag", "digest": "sha256:00"}]}))

    with pytest.raises(BackendError) as exc:
        OllamaBackend("qwen2.5:1.5b-instruct-q8_0").health_check()
    assert "not present on this Ollama host" in str(exc.value)
    assert "invalidate the experiment" in str(exc.value)


def test_only_the_mock_backend_is_marked_synthetic():
    """`synthetic` gates findings-report generation, so it must not be wrong."""
    assert get_backend("ollama", "m").synthetic is False
    assert get_backend("mock", "m").synthetic is True
