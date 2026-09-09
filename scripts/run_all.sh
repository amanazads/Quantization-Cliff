#!/usr/bin/env bash
# Run every precision arm, then aggregate, plot and report.
#
#   bash scripts/run_all.sh ollama                  # Qwen2.5-1.5B on Ollama
#   bash scripts/run_all.sh ollama qwen2.5-1.5b     # same (explicit config set)
#   bash scripts/run_all.sh mock                    # pipeline check, FABRICATED
#
#   $1 backend (default: ollama)   $2 config set (default: qwen2.5-1.5b)   $3 results root override
#
# Arms run SEQUENTIALLY in fixed order (f16, fp8, q8, q4). A failing arm does not
# stop the others; the aggregator reports missing arms as GAPS IN COVERAGE,
# never as null results.

set -uo pipefail

BACKEND="${1:-ollama}"
CONFIG_SET="${2:-qwen2.5-1.5b}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

if [ "$CONFIG_SET" = "default" ] || [ "$CONFIG_SET" = "qwen2.5-1.5b" ]; then
  CONFIG_DIR="configs/experiments-qwen2.5-1.5b"
  SUFFIX="-qwen2.5-1.5b"
else
  CONFIG_DIR="configs/experiments-$CONFIG_SET"
  SUFFIX="-$CONFIG_SET"
fi

if [ ! -d "$CONFIG_DIR" ]; then
  echo "ERROR: no config set named '$CONFIG_SET' (looked for $CONFIG_DIR)."
  echo "Available sets:"
  for D in configs/experiments-*/; do
    [ -d "$D" ] && echo "  $(basename "$D" | sed 's/^experiments-//')"
  done
  exit 1
fi

if [ "$BACKEND" = "mock" ]; then
  RESULTS_ROOT="${3:-results_mock$SUFFIX}"
  EXTRA="--allow-synthetic"
else
  RESULTS_ROOT="${3:-results-qwen2.5-1.5b}"
  EXTRA=""
fi
FIGURES="reports/figures"
REPORT="reports/FINDINGS.md"

# Resolve a Python interpreter (prefer repository venv)
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
    PY="$VIRTUAL_ENV/bin/python"
  elif [ -x "$REPO/.venv/bin/python" ]; then
    PY="$REPO/.venv/bin/python"
  elif [ -x "$REPO/venv/bin/python" ]; then
    PY="$REPO/venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PY="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PY="$(command -v python)"
  else
    echo "ERROR: no Python interpreter found. Set PYTHON=/path/to/python, or create .venv."
    exit 1
  fi
fi

echo "backend      : $BACKEND"
echo "config set   : $CONFIG_SET  ($CONFIG_DIR)"
echo "results root : $RESULTS_ROOT"
echo "python       : $PY"
echo

# --------------------------------------------------------------------------- #
# Verify the frozen inputs.
# --------------------------------------------------------------------------- #
echo "== verifying frozen inputs =="
if ! "$PY" scripts/build_suites.py --check; then
  echo "ABORT: the suite check did not pass."
  exit 1
fi
if ! "$PY" scripts/build_manifest.py --check; then
  echo "ABORT: the manifest check did not pass."
  exit 1
fi
echo

# --------------------------------------------------------------------------- #
# Refuse to mix fabricated and measured arms in one results root.
# --------------------------------------------------------------------------- #
CLASH="$("$PY" - "$RESULTS_ROOT" "$BACKEND" <<'PY'
import json, sys
from pathlib import Path
root, backend = Path(sys.argv[1]), sys.argv[2]
incoming_synthetic = backend == "mock"
for meta in sorted(root.glob("*/metadata.json")):
    try:
        existing = bool(json.loads(meta.read_text())["backend"]["synthetic"])
    except Exception:
        continue
    if existing != incoming_synthetic:
        print(f"{meta.parent.name}:{'synthetic' if existing else 'real'}")
PY
)"
if [ -n "$CLASH" ]; then
  echo "ABORT: '$RESULTS_ROOT' already holds arms of the other kind:"
  for ARM in $CLASH; do echo "    ${ARM%%:*}  (${ARM##*:})"; done
  echo
  exit 1
fi

# --------------------------------------------------------------------------- #
# Preflight check
# --------------------------------------------------------------------------- #
echo "== preflight =="
"$PY" scripts/preflight.py --backend "$BACKEND" --config-set "$CONFIG_SET"
PREFLIGHT=$?
echo

if [ $PREFLIGHT -eq 2 ]; then
  echo "ABORT: the backend is unreachable. Nothing was run."
  exit 1
fi

FAILED=()
RAN=()
for PRECISION in f16 fp8 q8 q4; do
  echo "== $PRECISION =="
  if "$PY" -m ps5.run --precision "$PRECISION" --backend "$BACKEND" \
       --config-set "$CONFIG_SET" --suite ps1 ps3 \
       --results-root "$RESULTS_ROOT" --no-summary; then
    echo "   ok"
    RAN+=("$PRECISION")
  else
    echo "   NOT RUN (exit $?) -- recorded as a coverage gap, not a null result"
    FAILED+=("$PRECISION")
  fi
  echo
done

if [ ${#RAN[@]} -eq 0 ]; then
  echo "No arm completed, so there is nothing to aggregate."
  echo "Arms not run: ${FAILED[*]}"
  exit 1
fi

echo "== aggregate =="
"$PY" scripts/aggregate_results.py --results-root "$RESULTS_ROOT" || exit 1
echo
echo "== plots -> $FIGURES =="
"$PY" scripts/make_plots.py --results-root "$RESULTS_ROOT" --out "$FIGURES"
echo
echo "== findings report -> $REPORT =="
"$PY" scripts/generate_report.py --results-root "$RESULTS_ROOT" \
  --figures "$FIGURES" --out "$REPORT" $EXTRA

echo
echo "arms completed : ${RAN[*]}"
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "arms NOT run   : ${FAILED[*]}"
  echo
  echo "Those are gaps in precision coverage. The cliff can only be located among"
  echo "the arms that actually ran, and the findings report says so explicitly."
fi
