#!/usr/bin/env python3
"""Render reports/FINDINGS.md from aggregate.json.

  python scripts/generate_report.py
  python scripts/generate_report.py --results-root results_mock --allow-synthetic

Refuses to write a findings report from mock-backend data unless
--allow-synthetic is given, in which case the output is stamped as a validation
artefact on its first page.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ps5.cliff import load_criterion  # noqa: E402
from ps5.report import SyntheticReportRefused, render_findings  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--figures", default="reports/figures")
    parser.add_argument("--out", default="reports/FINDINGS.md")
    parser.add_argument("--criterion", default="configs/cliff_criterion.yaml")
    parser.add_argument("--allow-synthetic", action="store_true")
    args = parser.parse_args()

    aggregate_path = REPO / args.results_root / "aggregate" / "aggregate.json"
    figures_dir = REPO / args.figures
    out_path = REPO / args.out

    if not aggregate_path.exists() and args.results_root == "results":
        fallback_agg = REPO / "results-qwen2.5-1.5b" / "aggregate" / "aggregate.json"
        if fallback_agg.exists():
            print(f"[info] '{aggregate_path.relative_to(REPO)}' not found; defaulting to '{fallback_agg.relative_to(REPO)}'", file=sys.stderr)
            aggregate_path = fallback_agg
            if args.figures == "reports/figures":
                figures_dir = REPO / "reports" / "figures-qwen2.5-1.5b"
            if args.out == "reports/FINDINGS.md":
                out_path = REPO / "reports" / "FINDINGS-qwen2.5-1.5b.md"

    if not aggregate_path.exists():
        print(f"{aggregate_path} not found. Run scripts/aggregate_results.py first.",
              file=sys.stderr)
        return 1

    # Belt and braces: even with --allow-synthetic, fabricated output must not be
    # written to the path a reader will take for the real findings report.
    if args.allow_synthetic and "MOCK" not in out_path.name.upper():
        print(
            f"REFUSING to write synthetic output to '{args.out}'.\n\n"
            "A validation artefact must not occupy the path a reader will take for "
            "the real findings report. Write it somewhere clearly named instead:\n\n"
            f"  --out reports/FINDINGS_MOCK.md --figures reports/figures_mock\n",
            file=sys.stderr,
        )
        return 3
    try:
        relative = Path(figures_dir).relative_to(out_path.parent.relative_to(REPO)).as_posix()
    except ValueError:
        relative = figures_dir.as_posix()

    criterion = load_criterion(str(REPO / args.criterion))
    full_path = out_path.with_name(out_path.stem + "_FULL" + out_path.suffix)

    try:
        # The specification caps the submitted findings document at four pages, so
        # the default output is the concise one and the complete set of breakdown
        # tables goes to a companion appendix. Both render from the same
        # aggregate, so they cannot disagree with each other.
        concise_md = render_findings(
            aggregate_path, criterion, figures_dir=figures_dir,
            allow_synthetic=args.allow_synthetic, figures_relative=relative,
            concise=True, repo_root=REPO,
        )
        full_md = render_findings(
            aggregate_path, criterion, figures_dir=figures_dir,
            allow_synthetic=args.allow_synthetic, figures_relative=relative,
            concise=False, repo_root=REPO,
        )
    except SyntheticReportRefused as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return 2

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(concise_md, encoding="utf-8")
    full_path.write_text(full_md, encoding="utf-8")
    if not args.allow_synthetic and out_path.name != "FINDINGS.md":
        # Keep reports/FINDINGS.md in sync with the primary real report
        (out_path.parent / "FINDINGS.md").write_text(concise_md, encoding="utf-8")
        (out_path.parent / "FINDINGS_FULL.md").write_text(full_md, encoding="utf-8")

    lines = len(concise_md.splitlines())
    print(f"wrote {out_path.relative_to(REPO)} ({lines} lines) -- the submission document")
    print(f"wrote {full_path.relative_to(REPO)} ({len(full_md.splitlines())} lines) -- appendix")
    # ~50 rendered lines per page is a rough but useful guard against drifting
    # past the four-page cap without noticing.
    if lines > 200:
        print(f"\n  NOTE: the concise document is {lines} lines, which may exceed the")
        print("  specification's four-page cap once rendered. Check before submitting.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
