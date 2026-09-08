"""Canonical hashing helpers.

Every artifact that defines the identity of an experiment -- the prompt, the tool
schemas, the evaluation manifest, the generation config -- is hashed with these
functions. Comparability between two runs is decided by comparing these hashes,
so the hashing must be canonical: insensitive to key order and whitespace in
structured data, and exact for text.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

__all__ = [
    "sha256_bytes",
    "sha256_text",
    "sha256_file",
    "sha256_json",
    "short",
]

_HASH_PREFIX = "sha256:"


def sha256_bytes(data: bytes) -> str:
    return _HASH_PREFIX + hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    """Hash text with normalised line endings.

    Line endings are normalised so that a checkout on Windows does not change the
    prompt hash and thereby invalidate a comparison for a reason that has nothing
    to do with the experiment.
    """
    normalised = text.replace("\r\n", "\n").replace("\r", "\n")
    return sha256_bytes(normalised.encode("utf-8"))


def sha256_file(path: str | Path) -> str:
    p = Path(path)
    raw = p.read_bytes()
    # Text-like files get newline normalisation; anything else is hashed raw.
    if p.suffix.lower() in {".md", ".txt", ".json", ".jsonl", ".yaml", ".yml", ".py"}:
        return sha256_text(raw.decode("utf-8"))
    return sha256_bytes(raw)


def sha256_json(obj: Any) -> str:
    """Hash a JSON-serialisable object canonically.

    Keys are sorted and separators are tight, so two structurally identical
    objects hash identically regardless of how they were constructed.
    """
    canonical = json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    )
    return sha256_text(canonical)


def short(digest: str, length: int = 12) -> str:
    """Human-readable abbreviation of a hash, for filenames and log lines."""
    body = digest.split(":", 1)[-1]
    return body[:length]
