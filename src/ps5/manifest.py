"""The evaluation manifest: the guarantee that every precision saw identical inputs.

PS-5's requirement that all arms consume the same cases is only meaningful if it
is *enforced*, not merely intended. The manifest is the enforcement point: it
pins every case ID, the per-case content hash, and the whole-file hash of each
suite. Each run records the manifest hash it consumed, and the aggregator refuses
to compare runs whose manifest hashes differ.

A per-case content hash, not just an ID list, is used deliberately: without it,
editing the text of case PS1-017 while keeping its ID would silently change the
experiment while every ID-level check still passed.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .hashing import sha256_file, sha256_json

__all__ = [
    "load_suite",
    "build_manifest",
    "verify_manifest",
    "ManifestError",
    "case_content_hash",
]


class ManifestError(RuntimeError):
    """Raised when the suites on disk do not match the manifest."""


#: Fields that define what the model is asked and what counts as correct.
#: Presentation-only fields (comments, provenance notes) are excluded so that
#: fixing a typo in a comment does not invalidate an entire experiment.
_PS1_MATERIAL = ("case_id", "suite", "language", "target_violation", "prompt",
                 "expected_behaviour", "context")
_PS3_MATERIAL = ("case_id", "suite", "language", "prompt", "expected_tool",
                 "expected_arguments", "context")


def case_content_hash(case: Dict[str, Any]) -> str:
    suite = case.get("suite")
    fields = _PS1_MATERIAL if suite == "ps1" else _PS3_MATERIAL
    return sha256_json({k: case.get(k) for k in fields})


def load_suite(path: str | Path) -> List[Dict[str, Any]]:
    """Load a JSONL suite, with line numbers in any error message."""
    p = Path(path)
    if not p.exists():
        raise ManifestError(f"Suite file not found: {p}")

    cases: List[Dict[str, Any]] = []
    seen: set = set()
    for lineno, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            case = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ManifestError(f"{p}:{lineno}: invalid JSON: {exc}") from exc
        cid = case.get("case_id")
        if not cid:
            raise ManifestError(f"{p}:{lineno}: case is missing `case_id`")
        if cid in seen:
            raise ManifestError(
                f"{p}:{lineno}: duplicate case_id {cid!r}. Duplicate IDs would "
                "double-weight a case in the metrics."
            )
        seen.add(cid)
        cases.append(case)

    if not cases:
        raise ManifestError(f"Suite file {p} contains no cases")
    return cases


def build_manifest(
    suite_paths: Dict[str, str | Path],
    version: str = "1.0.0",
    notes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build the manifest from the suites on disk."""
    suites: Dict[str, Any] = {}
    for suite_id, path in sorted(suite_paths.items()):
        p = Path(path)
        cases = load_suite(p)
        entries = [
            {
                "case_id": c["case_id"],
                "language": c.get("language"),
                "content_hash": case_content_hash(c),
                **({"target_violation": c.get("target_violation")} if suite_id == "ps1" else {}),
                **({"expected_tool": c.get("expected_tool")} if suite_id == "ps3" else {}),
            }
            for c in cases
        ]
        by_language: Dict[str, int] = {}
        for c in cases:
            by_language[c.get("language", "unknown")] = by_language.get(c.get("language", "unknown"), 0) + 1

        suites[suite_id] = {
            "path": str(Path(path).as_posix()),
            "file_hash": sha256_file(p),
            "n_cases": len(cases),
            "counts_by_language": dict(sorted(by_language.items())),
            "cases": entries,
        }

    manifest: Dict[str, Any] = {
        "manifest_version": version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Pins the exact evaluation cases consumed by EVERY precision arm. "
            "All four arms must consume this manifest unchanged; the aggregator "
            "refuses to compare runs whose manifest hashes differ."
        ),
        "data_policy": (
            "All cases are SYNTHETIC and were authored for this repository. No real "
            "borrower data is present. Names, amounts, dates and account identifiers "
            "are invented."
        ),
        "notes": notes or [],
        "suites": suites,
        "total_cases": sum(s["n_cases"] for s in suites.values()),
    }
    # Self-hash over content only, so that regenerating the manifest without
    # changing any case does not invalidate previous runs via the timestamp.
    manifest["content_hash"] = sha256_json(
        {sid: {"file_hash": s["file_hash"], "cases": s["cases"]} for sid, s in suites.items()}
    )
    return manifest


def verify_manifest(
    manifest: Dict[str, Any],
    suite_paths: Dict[str, str | Path],
    strict: bool = True,
) -> List[str]:
    """Check the suites on disk against the manifest. Returns a list of problems."""
    problems: List[str] = []

    for suite_id, spec in (manifest.get("suites") or {}).items():
        path = suite_paths.get(suite_id)
        if path is None:
            problems.append(f"suite '{suite_id}' is in the manifest but no path was supplied")
            continue

        try:
            cases = load_suite(path)
        except ManifestError as exc:
            problems.append(str(exc))
            continue

        on_disk = {c["case_id"]: case_content_hash(c) for c in cases}
        expected = {e["case_id"]: e["content_hash"] for e in spec.get("cases", [])}

        for missing in sorted(set(expected) - set(on_disk)):
            problems.append(f"{suite_id}: case {missing} is in the manifest but missing on disk")
        for extra in sorted(set(on_disk) - set(expected)):
            problems.append(f"{suite_id}: case {extra} is on disk but not in the manifest")
        for cid in sorted(set(expected) & set(on_disk)):
            if expected[cid] != on_disk[cid]:
                problems.append(
                    f"{suite_id}: case {cid} CONTENT CHANGED since the manifest was built "
                    "(same ID, different material fields). Any comparison spanning this "
                    "change is invalid."
                )

        file_hash = sha256_file(path)
        if file_hash != spec.get("file_hash"):
            msg = (
                f"{suite_id}: file hash differs from the manifest "
                f"({file_hash} vs {spec.get('file_hash')})"
            )
            # Non-material edits (comments, key order, formatting) change the file
            # hash without changing any case, so this is only fatal in strict mode.
            problems.append(msg if strict else msg + " [non-strict: content hashes matched]")

    if strict and problems:
        raise ManifestError(
            "Evaluation manifest verification FAILED:\n  - " + "\n  - ".join(problems)
        )
    return problems
