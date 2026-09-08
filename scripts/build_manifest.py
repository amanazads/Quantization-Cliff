#!/usr/bin/env python3
"""Build data/evaluation_manifest.json from the suites on disk.

The manifest is the guarantee that every precision consumed identical inputs.
Rebuilding it after a suite edit INVALIDATES every prior run, so this script
refuses to overwrite a manifest whose content hash would change unless --force is
given, and says exactly what would break.

  python scripts/build_manifest.py
  python scripts/build_manifest.py --check     # verify, do not write
  python scripts/build_manifest.py --force     # accept an intentional suite change
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ps5.manifest import build_manifest, verify_manifest  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO / "data" / "evaluation_manifest.json"
SUITES = {
    "ps1": REPO / "data" / "ps1_guardrail_suite.jsonl",
    "ps3": REPO / "data" / "ps3_toolcall_suite.jsonl",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--version", default=None)
    args = parser.parse_args()

    relative = {sid: p.relative_to(REPO).as_posix() for sid, p in SUITES.items()}
    existing = None
    if MANIFEST_PATH.exists():
        existing = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    version = args.version or (existing or {}).get("manifest_version", "1.0.0")
    fresh = build_manifest(
        {sid: REPO / rel for sid, rel in relative.items()},
        version=version,
        notes=[
            "Suites authored for this repository from the challenge brief; they are "
            "NOT the official Predixion PS-1/PS-3 suites.",
            "All content is synthetic. No real borrower data is present.",
        ],
    )
    # Store repo-relative paths so the manifest is machine-independent.
    for sid, rel in relative.items():
        fresh["suites"][sid]["path"] = rel

    if args.check:
        if existing is None:
            print("No manifest on disk.", file=sys.stderr)
            return 1
        problems = verify_manifest(existing, SUITES, strict=False)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        if existing.get("content_hash") != fresh["content_hash"]:
            print("MANIFEST DRIFT: the suites on disk no longer match the manifest.",
                  file=sys.stderr)
            return 1
        print(f"ok: manifest matches the suites ({existing['total_cases']} cases, "
              f"content_hash={existing['content_hash'][:19]}...)")
        return 0

    if existing and existing.get("content_hash") != fresh["content_hash"] and not args.force:
        print(
            "REFUSING TO OVERWRITE.\n\n"
            f"  existing content_hash: {existing.get('content_hash')}\n"
            f"  new content_hash:      {fresh['content_hash']}\n\n"
            "The evaluation cases have changed. Rebuilding the manifest makes every\n"
            "existing result in results/ incomparable with anything produced after it,\n"
            "because the arms would no longer have seen the same inputs.\n\n"
            "If the change is intentional: bump --version, pass --force, delete or\n"
            "archive results/, and re-run ALL FOUR precisions. Never re-run a subset.",
            file=sys.stderr,
        )
        return 2

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(fresh, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote {MANIFEST_PATH.relative_to(REPO)}")
    print(f"  total cases  : {fresh['total_cases']}")
    for sid, spec in fresh["suites"].items():
        print(f"  {sid:>4}         : {spec['n_cases']} cases  {spec['counts_by_language']}")
    print(f"  content_hash : {fresh['content_hash']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
