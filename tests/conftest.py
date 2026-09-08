import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import pytest  # noqa: E402

from ps5.backends.base import GenerationResult, ToolCall  # noqa: E402
from ps5.scoring.ps1_guardrails import load_rules  # noqa: E402
from ps5.scoring.ps3_toolcalls import load_tool_schemas  # noqa: E402


@pytest.fixture(scope="session")
def repo() -> Path:
    return REPO


@pytest.fixture(scope="session")
def schemas():
    _tools, by_name = load_tool_schemas(str(REPO / "schemas" / "tools.json"))
    return by_name


@pytest.fixture(scope="session")
def tools():
    tools, _ = load_tool_schemas(str(REPO / "schemas" / "tools.json"))
    return tools


@pytest.fixture(scope="session")
def rules():
    return load_rules(str(REPO / "configs" / "guardrail_rules.json"))


def ok(text: str = "", calls=None) -> GenerationResult:
    return GenerationResult(text=text, tool_calls=list(calls or []), ok=True,
                            finish_reason="stop", latency_ms=1.0)


def call(name, arguments, raw=None, parse_error=None) -> ToolCall:
    return ToolCall(name=name, arguments=arguments, raw_arguments=raw, parse_error=parse_error)


def failed(kind="timeout") -> GenerationResult:
    return GenerationResult(ok=False, error="synthetic failure", error_kind=kind)


#: Argument names follow the FROZEN Section 6.3 schemas exactly.
CTX = {"LENDER": "Arthik Finance", "NAME": "Ramesh Patil", "DPD": 30,
       "PRODUCT": "a personal loan", "AMOUNT": "INR 18,400"}

PTP_CASE = {
    "case_id": "T-001", "suite": "ps3", "language": "en", "context": CTX,
    "expected_tool": "capture_ptp",
    "expected_arguments": {"promised_amount": 5000, "promised_date": "2026-09-12"},
}

NO_CALL_CASE = {
    "case_id": "T-002", "suite": "ps3", "language": "en", "context": CTX,
    "expected_tool": None,
    "expected_arguments": {},
}
