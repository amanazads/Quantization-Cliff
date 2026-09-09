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


def _pdf_page_count(path: Path) -> Optional[int]:
    if not path.exists():
        return None
    try:
        import subprocess
        out = subprocess.check_output(
            ["mdls", "-name", "kMDItemNumberOfPages", "-raw", str(path)],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        if out.isdigit():
            return int(out)
    except Exception:
        pass
    try:
        import re
        content = path.read_bytes()
        pages = re.findall(rb"/Type\s*/Page(?=[^s]|$)", content)
        if pages:
            return len(pages)
    except Exception:
        pass
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-root", default="results-qwen2.5-1.5b")
    parser.add_argument("--figures", default="reports/figures")
    parser.add_argument("--out", default="reports/FINDINGS.md")
    parser.add_argument("--criterion", default="configs/cliff_criterion.yaml")
    parser.add_argument("--allow-synthetic", action="store_true")
    args = parser.parse_args()

    aggregate_path = REPO / args.results_root / "aggregate" / "aggregate.json"
    figures_dir = REPO / args.figures
    out_path = REPO / args.out

    if not aggregate_path.exists():
        print(f"No aggregate found at {aggregate_path}.\n"
              "Run aggregate_results.py first.", file=sys.stderr)
        return 1

    relative = args.figures
    if Path(relative).is_absolute():
        relative = figures_dir.as_posix()

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

    criterion = load_criterion(str(REPO / args.criterion))

    try:
        concise_md = render_findings(
            aggregate_path, criterion, figures_dir=figures_dir,
            allow_synthetic=args.allow_synthetic, figures_relative=relative,
            concise=True, repo_root=REPO,
        )
    except SyntheticReportRefused as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return 2

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(concise_md, encoding="utf-8")
    if not args.allow_synthetic and out_path.name != "FINDINGS.md":
        (out_path.parent / "FINDINGS.md").write_text(concise_md, encoding="utf-8")

    lines = len(concise_md.splitlines())
    print(f"wrote {out_path.relative_to(REPO)} ({lines} lines) -- the submission document")

    pdf_path = out_path.with_suffix(".pdf")
    pdf_pages = _pdf_page_count(pdf_path)
    if pdf_pages is not None:
        if pdf_pages <= 4:
            print(f"verified PDF: {pdf_path.relative_to(REPO)} is {pdf_pages} pages (within the 4-page cap).")
        else:
            print(f"\n  WARNING: {pdf_path.relative_to(REPO)} is {pdf_pages} pages, exceeding the 4-page cap!")
    elif lines > 300:
        print(f"\n  NOTE: the concise document is {lines} lines. Check rendered page count before submitting.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
