#!/usr/bin/env python3
"""Aggregate raw per-case results into metrics, degradation curves and a cliff verdict.

  python scripts/aggregate_results.py
  python scripts/aggregate_results.py --results-root results_mock
  python scripts/aggregate_results.py --precisions f16 q8 q4
  python scripts/aggregate_results.py --allow-deviation   # proceed despite divergence

Outputs, all machine-readable and all derived from raw JSONL:
  <results-root>/aggregate/aggregate.json    full metrics + intervals + cliff analysis
  <results-root>/aggregate/summary.csv       one tidy row per (suite, metric, precision)
  <results-root>/aggregate/summary.md        human-readable summary

No number in any report is ever typed by hand; every figure and table is rendered
from aggregate.json.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ps5.aggregate import ComparabilityError, aggregate, discover_runs, write_outputs  # noqa: E402
from ps5.cliff import load_criterion  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-root", default="results-qwen2.5-1.5b")
    parser.add_argument("--precisions", nargs="+", default=None,
                        help="restrict to these precision arms")
    parser.add_argument("--criterion", default="configs/cliff_criterion.yaml")
    parser.add_argument("--out", default=None, help="output dir (default <results-root>/aggregate)")
    parser.add_argument("--allow-deviation", action="store_true",
                        help="proceed even if control fields diverge across arms. The "
                             "divergence is recorded in the aggregate and reproduced in "
                             "the findings report; it does not become valid.")
    args = parser.parse_args()

    results_root = (REPO / args.results_root).resolve()
    out_dir = Path(args.out).resolve() if args.out else results_root / "aggregate"
    criterion = load_criterion(str(REPO / args.criterion))

    runs = discover_runs(results_root, args.precisions)
    if not runs and args.results_root == "results":
        fallback = REPO / "results-qwen2.5-1.5b"
        fallback_runs = discover_runs(fallback, args.precisions)
        if fallback_runs:
            print(f"[info] '{results_root.relative_to(REPO)}' has no runs; defaulting to '{fallback.relative_to(REPO)}'", file=sys.stderr)
            results_root = fallback
            out_dir = Path(args.out).resolve() if args.out else results_root / "aggregate"
            runs = fallback_runs

    if not runs:
        print(f"No runs found under {results_root}.\n"
              "Run at least one arm first, e.g.:\n"
              "  python -m ps5.run --precision f16 --backend mock --suite ps1 ps3",
              file=sys.stderr)
        return 1

    print(f"arms found: {', '.join(sorted(runs))}")
    missing = [p for p in criterion.precision_order if p not in runs]
    if missing:
        print(f"arms MISSING: {', '.join(missing)}  "
              "(these are gaps in coverage, not null results)")

    try:
        agg = aggregate(runs, criterion, allow_deviation=args.allow_deviation)
    except ComparabilityError as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return 2

    written = write_outputs(agg, criterion, out_dir)
    for kind, path in written.items():
        print(f"wrote {kind:>8}: {path.relative_to(REPO)}")

    if agg.get("synthetic"):
        print("\n  *** THIS AGGREGATE IS SYNTHETIC. The numbers are a pipeline-validation")
        print("  *** fixture from the mock backend, not a measurement of any model.\n")

    mvp = agg["degradation"]["minimum_viable_precision"]
    print(f"\nminimum viable precision: {mvp.get('precision')}")
    for suite, analyses in agg["degradation"]["analyses"].items():
        for analysis in analyses:
            print(f"  [{suite}] {analysis['metric']:<28} pattern={analysis['pattern']:<10} "
                  f"cliff_at={analysis['cliff_precision'] or '-'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
