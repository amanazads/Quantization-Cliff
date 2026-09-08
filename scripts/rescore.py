#!/usr/bin/env python3
"""Re-score existing raw results without re-generating them.

Scoring is a pure function of the stored model output, so a scorer fix never
requires paying for generation again. This is also the ONLY sanctioned way to
correct a scoring bug: it applies the new scorer uniformly to every arm, which
hand-editing results could not guarantee.

  python scripts/rescore.py --results-root results            # rewrites in place
  python scripts/rescore.py --results-root results --dry-run  # report differences only

Raw model output is never modified. Only the `score` block and the promoted
top-level scoring fields are recomputed.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ps5.backends.base import GenerationResult, ToolCall  # noqa: E402
from ps5.scoring import ps1_guardrails, ps3_toolcalls  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def _rebuild_result(record: Dict[str, Any]) -> GenerationResult:
    """Reconstruct the generation result from the stored raw fields."""
    calls = [
        ToolCall(name=tc.get("name"), arguments=tc.get("arguments"),
                 raw_arguments=tc.get("raw_arguments"), parse_error=tc.get("parse_error"))
        for tc in record.get("tool_calls") or []
    ]
    return GenerationResult(
        text=record.get("response_text") or "",
        tool_calls=calls,
        finish_reason=record.get("finish_reason"),
        ok=bool(record.get("generation_ok", True)),
        error=record.get("generation_error"),
        error_kind=record.get("generation_error_kind"),
        latency_ms=record.get("latency_ms"),
        attempts=record.get("attempts", 1),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()

    root = (REPO / args.results_root).resolve()
    schemas_path = REPO / "schemas" / "tools.json"
    rules_path = REPO / "configs" / "guardrail_rules.json"
    _tools, schemas_by_name = ps3_toolcalls.load_tool_schemas(str(schemas_path))
    rules = ps1_guardrails.load_rules(str(rules_path))

    files = sorted(root.glob("*/*_results.jsonl"))
    if not files:
        print(f"No raw results under {root}", file=sys.stderr)
        return 1

    total_changed = 0
    for path in files:
        suite = path.name.replace("_results.jsonl", "")
        records = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

        changes: Counter = Counter()
        updated: List[Dict[str, Any]] = []
        for record in records:
            result = _rebuild_result(record)
            case = {
                "case_id": record.get("case_id"),
                "suite": suite,
                "language": record.get("language"),
                "target_violation": record.get("target_violation"),
                "expected_tool": record.get("expected_tool"),
                "expected_arguments": record.get("expected_arguments"),
                "prompt": record.get("prompt"),
            }

            if suite == "ps1":
                score = ps1_guardrails.score_ps1_case(case, result, rules).to_dict()
                before, after = record.get("violation"), score.get("violation")
                record.update({
                    "scorable": score["scorable"], "violation": score["violation"],
                    "is_benign_control": score["is_benign_control"],
                    "benign_refusal": score["benign_refusal"],
                })
            else:
                score = ps3_toolcalls.score_ps3_case(case, result, schemas_by_name).to_dict()
                before, after = record.get("outcome"), score.get("outcome")
                record.update({
                    "scorable": score["scorable"], "outcome": score["outcome"],
                    "actual_tool": score["actual_tool"],
                    "actual_arguments": score["actual_arguments"],
                    "argument_matches": score["argument_matches"],
                    "argument_total": score["argument_total"],
                    "via_fallback": score["via_fallback"],
                })

            if before != after:
                changes[f"{before} -> {after}"] += 1
            record["score"] = score
            record["rescored_at_utc"] = datetime.now(timezone.utc).isoformat()
            updated.append(record)

        changed = sum(changes.values())
        total_changed += changed
        status = "would change" if args.dry_run else "changed"
        print(f"{path.relative_to(REPO)}: {len(records)} records, {status} {changed}")
        for transition, count in changes.most_common():
            print(f"    {transition}: {count}")

        if not args.dry_run:
            if not args.no_backup:
                shutil.copy2(path, path.with_suffix(".jsonl.bak"))
            path.write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in updated),
                encoding="utf-8",
            )

    print(f"\nscorer versions: ps1={ps1_guardrails.SCORER_VERSION} "
          f"ps3={ps3_toolcalls.SCORER_VERSION}")
    if total_changed and not args.dry_run:
        print("\nRe-run scripts/aggregate_results.py, make_plots.py and "
              "generate_report.py so the reported numbers match the new scores.")
        print("NOTE: metadata.json still records the scorer version used at GENERATION "
              "time. Re-scoring changes the scores but not that field; state the "
              "re-scoring in the findings report.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
