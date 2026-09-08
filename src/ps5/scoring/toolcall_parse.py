"""Fallback extraction of tool calls emitted as prose rather than as structured calls.

Low-precision models frequently stop using the native tool-calling channel and
start printing a JSON blob into the message body instead. That is a real,
interesting degradation -- but if the harness only ever looked at the structured
channel it would be recorded as `missed_call`, conflating "did not decide to act"
with "acted but could not format it". Those are different failures with different
production fixes, so they are distinguished.

Every fallback extraction is flagged `via_fallback: true` in the raw record, and
`fallback_extraction_rate` is reported alongside the headline metrics. The
fallback is applied IDENTICALLY at every precision, so it cannot advantage one arm.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from ..backends.base import ToolCall

__all__ = ["extract_tool_calls_from_text", "find_json_objects"]

# ```json { ... } ``` or ``` { ... } ```
_FENCE_RE = re.compile(r"```(?:json|tool_code|python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
# <tool_call> { ... } </tool_call>  (Qwen/Hermes style)
_TAG_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE)
# name(arg=value, ...) -- python-ish call syntax some models fall back to
_PYCALL_RE = re.compile(r"\b([a-z_][a-z0-9_]*)\s*\(([^()]{0,600})\)", re.IGNORECASE)


def find_json_objects(text: str) -> List[str]:
    """Return balanced top-level {...} substrings, brace-matched, strings respected."""
    out: List[str] = []
    depth = 0
    start = -1
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    out.append(text[start : i + 1])
                    start = -1
    return out


def _as_call(obj: Any, known_tools: set) -> Optional[Tuple[str, Any, str]]:
    """Interpret a decoded object as (name, arguments, shape) if it looks like a call."""
    if not isinstance(obj, dict):
        return None

    # {"name": ..., "arguments"|"parameters"|"args": {...}}
    name = obj.get("name") or obj.get("tool") or obj.get("tool_name") or obj.get("function")
    if isinstance(name, dict):  # {"function": {"name":..., "arguments":...}}
        inner = name
        nm = inner.get("name")
        if isinstance(nm, str):
            return (nm, inner.get("arguments", inner.get("parameters")), "nested_function")
        return None
    if isinstance(name, str):
        for key in ("arguments", "parameters", "args", "input"):
            if key in obj:
                return (name, obj[key], f"name+{key}")
        return (name, {}, "name_only")

    # {"capture_ptp": {...}} -- single key that happens to be a known tool
    if len(obj) == 1:
        only_key = next(iter(obj))
        if only_key in known_tools:
            return (only_key, obj[only_key], "tool_keyed")
    return None


def _parse_pycall_args(blob: str) -> Optional[Dict[str, Any]]:
    """Parse `amount=5000, promise_date="2026-09-10"` into a dict. Conservative."""
    args: Dict[str, Any] = {}
    for part in re.split(r",(?=(?:[^\"']*[\"'][^\"']*[\"'])*[^\"']*$)", blob):
        part = part.strip()
        if not part or "=" not in part:
            continue
        key, _, value = part.partition("=")
        key = key.strip()
        value = value.strip().strip("\"'")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            return None
        if re.fullmatch(r"-?\d+", value):
            args[key] = int(value)
        elif re.fullmatch(r"-?\d*\.\d+", value):
            args[key] = float(value)
        elif value.lower() in {"true", "false"}:
            args[key] = value.lower() == "true"
        else:
            args[key] = value
    return args or None


def extract_tool_calls_from_text(text: str, known_tools: set) -> List[ToolCall]:
    """Best-effort recovery of calls printed into the message body.

    Only returns calls whose name is a KNOWN tool. This is deliberate: without
    that filter the extractor would invent calls out of ordinary prose containing
    braces, inflating spurious_call_rate at exactly the precisions most likely to
    ramble -- which would look like a finding and would be an artefact.
    """
    if not text or not text.strip():
        return []

    candidates: List[str] = []
    for regex in (_TAG_RE, _FENCE_RE):
        candidates.extend(m.strip() for m in regex.findall(text))
    candidates.extend(find_json_objects(text))

    found: List[ToolCall] = []
    seen: set = set()

    for blob in candidates:
        blob = blob.strip()
        if not blob:
            continue
        try:
            decoded = json.loads(blob)
        except json.JSONDecodeError:
            continue

        items = decoded if isinstance(decoded, list) else [decoded]
        for item in items:
            parsed = _as_call(item, known_tools)
            if not parsed:
                continue
            name, args, _shape = parsed
            if name not in known_tools:
                continue
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    found.append(ToolCall(name=name, arguments=None, raw_arguments=args,
                                          parse_error="fallback: arguments not valid JSON"))
                    continue
            if args is None:
                args = {}
            if not isinstance(args, dict):
                found.append(ToolCall(name=name, arguments=None, raw_arguments=repr(args),
                                      parse_error=f"fallback: arguments were {type(args).__name__}, not an object"))
                continue
            key = (name, json.dumps(args, sort_keys=True, default=str))
            if key in seen:
                continue
            seen.add(key)
            found.append(ToolCall(name=name, arguments=args,
                                  raw_arguments=json.dumps(args, ensure_ascii=False)))

    if found:
        return found

    # Last resort: python-style call syntax.
    for name, blob in _PYCALL_RE.findall(text):
        if name not in known_tools:
            continue
        args = _parse_pycall_args(blob)
        if args is None:
            continue
        key = (name, json.dumps(args, sort_keys=True, default=str))
        if key in seen:
            continue
        seen.add(key)
        found.append(ToolCall(name=name, arguments=args,
                              raw_arguments=json.dumps(args, ensure_ascii=False)))
    return found
