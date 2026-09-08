#!/usr/bin/env python3
"""Render every PS-5 figure from aggregate.json.

  python scripts/make_plots.py
  python scripts/make_plots.py --results-root results_mock

Figures are derived only from aggregate.json, so they cannot disagree with the
tables. Rate axes are fixed to 0-100% and every point carries its Wilson 95%
interval; see src/ps5/plots.py for why.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ps5.plots import render_all  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--out", default="reports/figures")
    args = parser.parse_args()

    aggregate_path = REPO / args.results_root / "aggregate" / "aggregate.json"
    if not aggregate_path.exists():
        print(f"{aggregate_path} not found. Run scripts/aggregate_results.py first.",
              file=sys.stderr)
        return 1

    out_dir = REPO / args.out
    written = render_all(aggregate_path, out_dir)
    for path in written:
        print(f"wrote {path.relative_to(REPO)}")
    if not written:
        print("No figures produced -- the aggregate contains no metrics.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
