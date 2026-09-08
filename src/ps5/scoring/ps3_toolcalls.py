"""PS-3 scorer: structured output and tool-calling validity.

Implements docs/METRICS.md section 2 exactly. Pure and deterministic: a function
of (case, generation result, frozen schemas) only. It never sees the precision,
which is what guarantees it cannot treat one arm differently from another.

Re-scoring raw JSONL with a newer scorer version is cheap and is the supported
way to correct a scoring bug -- re-generation is not required.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

from ..backends.base import GenerationResult, ToolCall
from .toolcall_parse import extract_tool_calls_from_text

__all__ = [
    "SCORER_VERSION",
    "PS3Outcome",
    "PS3Score",
    "score_ps3_case",
    "load_tool_schemas",
    "normalise_value",
]

SCORER_VERSION = "ps3-scorer/2.0.0"

#: Free-text arguments are stored but NOT scored -- judging them needs a judge
#: model, and a judge would reintroduce the confound the design removes.
FREE_TEXT_FIELDS = frozenset({"borrower_statement", "notes"})


class PS3Outcome:
    GENERATION_FAILURE = "generation_failure"
    MALFORMED = "malformed"
    SPURIOUS_CALL = "spurious_call"
    MISSED_CALL = "missed_call"
    WRONG_TOOL = "wrong_tool"
    WRONG_ARGUMENTS = "wrong_arguments"
    PASS = "pass"

    ALL = (
        GENERATION_FAILURE, MALFORMED, SPURIOUS_CALL, MISSED_CALL,
        WRONG_TOOL, WRONG_ARGUMENTS, PASS,
    )


@dataclass
class PS3Score:
    case_id: str
    outcome: str
    scorable: bool
    expected_tool: Optional[str]
    actual_tool: Optional[str]
    expected_arguments: Dict[str, Any] = field(default_factory=dict)
    actual_arguments: Optional[Dict[str, Any]] = None
    n_calls: int = 0
    via_fallback: bool = False
    schema_errors: List[str] = field(default_factory=list)
    argument_matches: int = 0
    argument_total: int = 0
    argument_mismatches: List[Dict[str, Any]] = field(default_factory=list)
    scorer_version: str = SCORER_VERSION
    notes: List[str] = field(default_factory=list)

    @property
    def is_pass(self) -> bool:
        return self.outcome == PS3Outcome.PASS

    @property
    def correct_tool(self) -> bool:
        return self.outcome in (PS3Outcome.PASS, PS3Outcome.WRONG_ARGUMENTS)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["is_pass"] = self.is_pass
        d["correct_tool"] = self.correct_tool
        return d


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #


def load_tool_schemas(path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Return (tools-as-sent-to-the-model, {tool_name: parameter schema})."""
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    tools = doc["tools"]
    by_name = {
        t["function"]["name"]: t["function"].get("parameters", {}) for t in tools
    }
    return tools, by_name


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #

_NUM_CLEAN_RE = re.compile(r"[,\s ]|(?:rs\.?|inr|₹)", re.IGNORECASE)
_ISO_DATE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")


def _to_number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = _NUM_CLEAN_RE.sub("", value.strip())
        if not cleaned:
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _to_date(value: Any) -> Optional[str]:
    """Normalise to ISO YYYY-MM-DD, or None if it is not a date.

    Relative expressions ("kal", "tomorrow") are NOT resolved: each PS-3 case
    fixes an explicit reference date in its prompt, so the correct absolute date
    is unambiguous and resolving relatives here would let the scorer forgive an
    error the model actually made.
    """
    if isinstance(value, (date, datetime)):
        return value.strftime("%Y-%m-%d")
    if not isinstance(value, str):
        return None
    text = value.strip()
    if _ISO_DATE_RE.match(text):
        try:
            datetime.strptime(text, "%Y-%m-%d")
            return text
        except ValueError:
            return None
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d", "%d %b %Y", "%d %B %Y"):
        try:
            return datetime.strptime(text, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def normalise_value(expected: Any, actual: Any) -> Tuple[Any, Any, str]:
    """Normalise a pair for comparison, dispatching on the EXPECTED value's type.

    Dispatching on the expected side (not the actual side) keeps the comparison
    rule fixed per field regardless of what the model produced, so a model cannot
    change how it is judged by changing its output type.
    """
    exp_num = _to_number(expected)
    if exp_num is not None:
        act_num = _to_number(actual)
        return exp_num, act_num, "number"

    exp_date = _to_date(expected)
    if exp_date is not None:
        return exp_date, _to_date(actual), "date"

    exp_s = str(expected).strip().casefold() if expected is not None else None
    act_s = str(actual).strip().casefold() if actual is not None else None
    return exp_s, act_s, "string"


def _values_match(expected: Any, actual: Any) -> Tuple[bool, str]:
    exp, act, kind = normalise_value(expected, actual)
    if act is None:
        return False, kind
    if kind == "number":
        return bool(math.isclose(exp, act, rel_tol=0.0, abs_tol=1e-9)), kind
    return bool(exp == act), kind


# --------------------------------------------------------------------------- #
# Schema validation
# --------------------------------------------------------------------------- #


def _validate_against_schema(name: str, args: Any, schema: Dict[str, Any]) -> List[str]:
    """Structural validation. Returns a list of human-readable errors.

    Hand-rolled rather than delegating to `jsonschema` so that the exact set of
    conditions counted as `malformed` is visible in this file and cannot shift
    with a library upgrade -- a silent change there would move a headline metric.
    """
    errors: List[str] = []
    if not isinstance(args, dict):
        return [f"arguments for '{name}' are {type(args).__name__}, expected object"]

    props: Dict[str, Any] = schema.get("properties", {}) or {}
    required: List[str] = schema.get("required", []) or []
    allow_extra = schema.get("additionalProperties", True)

    for key in required:
        if key not in args:
            errors.append(f"missing required argument '{key}'")

    if allow_extra is False:
        for key in args:
            if key not in props:
                errors.append(f"unexpected argument '{key}' (additionalProperties is false)")

    for key, value in args.items():
        spec = props.get(key)
        if not spec:
            continue
        expected_type = spec.get("type")
        if expected_type == "number":
            if _to_number(value) is None:
                errors.append(f"argument '{key}' is not numeric: {value!r}")
        elif expected_type == "integer":
            num = _to_number(value)
            if num is None or num != int(num):
                errors.append(f"argument '{key}' is not an integer: {value!r}")
        elif expected_type == "string":
            if not isinstance(value, str):
                errors.append(f"argument '{key}' is not a string: {value!r}")
        elif expected_type == "boolean":
            if not isinstance(value, bool):
                errors.append(f"argument '{key}' is not a boolean: {value!r}")

        enum = spec.get("enum")
        if enum is not None and value is not None:
            allowed = {str(e).casefold() for e in enum}
            if str(value).strip().casefold() not in allowed:
                errors.append(f"argument '{key}' value {value!r} not in enum {enum}")
    return errors


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


def _resolve_calls(
    result: GenerationResult, known_tools: set
) -> Tuple[List[ToolCall], bool]:
    """Prefer the structured channel; fall back to prose extraction only if empty."""
    if result.tool_calls:
        return list(result.tool_calls), False
    recovered = extract_tool_calls_from_text(result.text or "", known_tools)
    return recovered, bool(recovered)


def score_ps3_case(
    case: Dict[str, Any],
    result: GenerationResult,
    schemas_by_name: Dict[str, Dict[str, Any]],
) -> PS3Score:
    """Score a single PS-3 case. See docs/METRICS.md section 2.1 for the order."""
    case_id = case.get("case_id", "unknown")
    expected_tool: Optional[str] = case.get("expected_tool")
    expected_args: Dict[str, Any] = dict(case.get("expected_arguments") or {})
    known_tools = set(schemas_by_name)

    # 1. transport failure -> not scorable
    if not result.ok:
        return PS3Score(
            case_id=case_id, outcome=PS3Outcome.GENERATION_FAILURE, scorable=False,
            expected_tool=expected_tool, actual_tool=None,
            expected_arguments=expected_args,
            notes=[f"generation failed: {result.error_kind}: {result.error}"],
        )

    calls, via_fallback = _resolve_calls(result, known_tools)

    # 2. malformed -- any attempted call that cannot be made into a valid one
    schema_errors: List[str] = []
    for call in calls:
        if call.parse_error:
            schema_errors.append(f"{call.name or '<unnamed>'}: {call.parse_error}")
            continue
        if not call.name or call.name not in schemas_by_name:
            schema_errors.append(f"unknown tool name {call.name!r} (not in frozen schema)")
            continue
        schema_errors.extend(
            f"{call.name}: {e}"
            for e in _validate_against_schema(call.name, call.arguments, schemas_by_name[call.name])
        )

    if schema_errors:
        first = calls[0] if calls else None
        return PS3Score(
            case_id=case_id, outcome=PS3Outcome.MALFORMED, scorable=True,
            expected_tool=expected_tool,
            actual_tool=first.name if first else None,
            expected_arguments=expected_args,
            actual_arguments=first.arguments if first else None,
            n_calls=len(calls), via_fallback=via_fallback, schema_errors=schema_errors,
        )

    # 3/4. no tool expected
    if expected_tool is None:
        if calls:
            return PS3Score(
                case_id=case_id, outcome=PS3Outcome.SPURIOUS_CALL, scorable=True,
                expected_tool=None, actual_tool=calls[0].name,
                expected_arguments=expected_args, actual_arguments=calls[0].arguments,
                n_calls=len(calls), via_fallback=via_fallback,
                notes=["a tool was called although no call was correct for this case"],
            )
        return PS3Score(
            case_id=case_id, outcome=PS3Outcome.PASS, scorable=True,
            expected_tool=None, actual_tool=None, expected_arguments=expected_args,
            n_calls=0, via_fallback=False,
            notes=["correctly abstained from calling a tool"],
        )

    # 5. a tool was expected
    if not calls:
        return PS3Score(
            case_id=case_id, outcome=PS3Outcome.MISSED_CALL, scorable=True,
            expected_tool=expected_tool, actual_tool=None,
            expected_arguments=expected_args, n_calls=0, via_fallback=False,
        )

    # Extra well-formed calls beyond the one expected are a spurious action.
    # Stricter than "at least one was right", and deliberately so: an unwanted
    # send_payment_link alongside a correct capture_ptp is a production incident.
    if len(calls) > 1:
        return PS3Score(
            case_id=case_id, outcome=PS3Outcome.SPURIOUS_CALL, scorable=True,
            expected_tool=expected_tool, actual_tool=calls[0].name,
            expected_arguments=expected_args, actual_arguments=calls[0].arguments,
            n_calls=len(calls), via_fallback=via_fallback,
            notes=[f"{len(calls)} calls emitted where exactly 1 was expected: "
                   f"{[c.name for c in calls]}"],
        )

    call = calls[0]
    if call.name != expected_tool:
        return PS3Score(
            case_id=case_id, outcome=PS3Outcome.WRONG_TOOL, scorable=True,
            expected_tool=expected_tool, actual_tool=call.name,
            expected_arguments=expected_args, actual_arguments=call.arguments,
            n_calls=1, via_fallback=via_fallback,
        )

    # 6/7. right tool -- now compare arguments field by field
    actual_args = call.arguments or {}
    matches = 0
    total = 0
    mismatches: List[Dict[str, Any]] = []

    for key, exp_value in expected_args.items():
        if key in FREE_TEXT_FIELDS:
            continue
        total += 1
        act_value = actual_args.get(key, None)
        ok, kind = _values_match(exp_value, act_value)
        if ok:
            matches += 1
        else:
            mismatches.append({
                "field": key, "kind": kind,
                "expected": exp_value,
                "actual": act_value,
                "reason": "missing" if key not in actual_args else "value_mismatch",
            })

    outcome = PS3Outcome.PASS if not mismatches else PS3Outcome.WRONG_ARGUMENTS
    return PS3Score(
        case_id=case_id, outcome=outcome, scorable=True,
        expected_tool=expected_tool, actual_tool=call.name,
        expected_arguments=expected_args, actual_arguments=actual_args,
        n_calls=1, via_fallback=via_fallback,
        argument_matches=matches, argument_total=total, argument_mismatches=mismatches,
    )
