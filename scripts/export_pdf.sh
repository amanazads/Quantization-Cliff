#!/usr/bin/env bash
# Render a findings document to PDF and ENFORCE the specification's page cap.
#
#   bash scripts/export_pdf.sh reports/FINDINGS-qwen2.5-1.5b.md
#   bash scripts/export_pdf.sh reports/FINDINGS-qwen2.5-1.5b.md 4
#
#   $1 markdown file   $2 max pages (default 4, the specification's cap)
#
# The specification caps the submitted findings document at four pages. A
# markdown file has no page count until something renders it, so the rendering
# is part of the deliverable rather than the judge's problem: this script pins
# the typography (scripts/report_pdf.css) and FAILS if the result is over the
# cap. A limit that is checked is a limit; one that is remembered is not.
#
# Requires pandoc and wkhtmltopdf:
#   macOS:  brew install pandoc && brew install --cask wkhtmltopdf
#   Debian: apt-get install pandoc wkhtmltopdf

set -uo pipefail

SRC="${1:-reports/FINDINGS.md}"
MAX_PAGES="${2:-4}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

if [ ! -f "$SRC" ]; then
  echo "ERROR: no such file: $SRC"
  echo "Generate one first:  bash scripts/run_all.sh ollama qwen2.5-1.5b"
  exit 1
fi

for TOOL in pandoc wkhtmltopdf; do
  if ! command -v "$TOOL" >/dev/null 2>&1; then
    echo "ERROR: $TOOL is not installed, so the page count cannot be checked."
    echo "  macOS:  brew install pandoc && brew install --cask wkhtmltopdf"
    echo "  Debian: sudo apt-get install pandoc wkhtmltopdf"
    echo
    echo "The markdown at $SRC is still valid; only the PDF export is blocked."
    exit 1
  fi
done

OUT="${SRC%.md}.pdf"
CSS="scripts/report_pdf.css"

# Rendered from the directory holding the markdown so the relative figure paths
# in the document resolve.
( cd "$(dirname "$SRC")" && pandoc "$(basename "$SRC")" -o "$(basename "$OUT")" \
    --pdf-engine=wkhtmltopdf --css="$REPO/$CSS" \
    -V margin-top=0 -V margin-bottom=0 -V margin-left=0 -V margin-right=0 \
    >/dev/null 2>&1 )

if [ ! -f "$OUT" ]; then
  echo "ERROR: pandoc produced no output for $SRC."
  exit 1
fi

PAGES=""
if command -v pdfinfo >/dev/null 2>&1; then
  PAGES="$(pdfinfo "$OUT" | awk '/^Pages:/ {print $2}')"
fi

echo "wrote $OUT"
if [ -z "$PAGES" ]; then
  echo
  echo "WARNING: pdfinfo is not installed, so the page count was NOT verified."
  echo "  macOS: brew install poppler   Debian: apt-get install poppler-utils"
  echo "  Open $OUT and count the pages before submitting; the cap is $MAX_PAGES."
  exit 0
fi

echo "pages: $PAGES  (specification cap: $MAX_PAGES)"
if [ "$PAGES" -gt "$MAX_PAGES" ]; then
  echo
  echo "OVER THE CAP by $((PAGES - MAX_PAGES)) page(s). This document is not submittable as is."
  echo
  echo "The concise renderer already drops the per-category, per-language and"
  echo "failure-mode breakdowns to the _FULL appendix. To cut further, edit the"
  echo "\`concise\` branches in src/ps5/report.py -- do NOT hand-edit $SRC, which"
  echo "is regenerated and would silently lose the change."
  exit 1
fi

echo "Within the cap."
