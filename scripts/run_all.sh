#!/usr/bin/env bash
# Run every precision arm, then aggregate, plot and report.
#
#   bash scripts/run_all.sh ollama                  # default set: Qwen3.5-4B
#   bash scripts/run_all.sh ollama qwen2.5-1.5b     # the set that fits in 8 GB
#   bash scripts/run_all.sh vllm                    # all four precisions
#   bash scripts/run_all.sh mock                    # pipeline check, FABRICATED
#
#   $1 backend   $2 config set (default: "default")   $3 results root override
#
# Each config set is a different MODEL, so each gets its own results root
# (results, results-qwen2.5-1.5b, ...). The aggregator refuses to compare across
# model families anyway, but a shared directory would still let one run overwrite
# the other's raw JSONL -- so collision is prevented, not merely detected.
#
# Mock runs default to `results_mock*` for the same reason, and mixing fabricated
# with measured arms in one root is refused before anything runs.
#
# Use `bash scripts/run_all.sh ...` rather than `./scripts/run_all.sh ...` unless
# you have run `chmod +x scripts/*.sh`; the execute bit does not survive every
# way this repository might reach your machine.
#
# Arms run SEQUENTIALLY in a fixed order. A failing arm does not stop the others;
# the aggregator reports missing arms as GAPS IN COVERAGE, never as null results.

set -uo pipefail

BACKEND="${1:-ollama}"
CONFIG_SET="${2:-default}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

if [ "$CONFIG_SET" = "default" ]; then
  CONFIG_DIR="configs/experiments"
  SUFFIX=""
else
  CONFIG_DIR="configs/experiments-$CONFIG_SET"
  SUFFIX="-$CONFIG_SET"
fi
if [ ! -d "$CONFIG_DIR" ]; then
  echo "ERROR: no config set named '$CONFIG_SET' (looked for $CONFIG_DIR)."
  echo "Available sets:"
  echo "  default"
  for D in configs/experiments-*/; do
    [ -d "$D" ] && echo "  $(basename "$D" | sed 's/^experiments-//')"
  done
  exit 1
fi

# A mock run defaults to its OWN results root. The raw JSONL is the primary
# evidence of the experiment and each real arm costs a long time to produce, so a
# pipeline check run afterwards must not be able to overwrite it. An explicit
# third argument still wins -- this is a safe default, not a restriction.
if [ "$BACKEND" = "mock" ]; then
  RESULTS_ROOT="${3:-results_mock$SUFFIX}"
else
  RESULTS_ROOT="${3:-results$SUFFIX}"
fi

# --------------------------------------------------------------------------- #
# Resolve a Python interpreter.
#
# `python` does not exist on a stock macOS, and only exists inside an ACTIVATED
# virtualenv. Hard-coding it meant this script worked in the shell where the venv
# was active and failed in the next one -- so resolve it explicitly, and prefer
# the repo's own .venv so activation is not required at all.
# --------------------------------------------------------------------------- #
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
    PY="$VIRTUAL_ENV/bin/python"
  elif [ -x "$REPO/.venv/bin/python" ]; then
    PY="$REPO/.venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PY="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PY="$(command -v python)"
  else
    echo "ERROR: no Python interpreter found."
    echo "  Create one with:  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
    echo "  Or point at yours: PYTHON=/path/to/python bash scripts/run_all.sh $BACKEND"
    exit 1
  fi
fi

export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"

echo "backend      : $BACKEND"
echo "config set   : $CONFIG_SET  ($CONFIG_DIR)"
echo "results root : $RESULTS_ROOT"
echo "python       : $PY  ($("$PY" -V 2>&1))"
echo

# Fail with a useful message, not an ImportError traceback three steps later.
if ! "$PY" - <<'PY' 2>/dev/null
import sys
missing = []
for mod, pkg in (("yaml", "PyYAML"), ("requests", "requests"), ("matplotlib", "matplotlib")):
    try:
        __import__(mod)
    except ImportError:
        missing.append(pkg)
if missing:
    sys.stderr.write("MISSING: " + ", ".join(missing) + "\n")
    sys.exit(1)
PY
then
  echo "ERROR: required packages are missing for $PY"
  echo "  Install them with:  $PY -m pip install -r requirements.txt"
  exit 1
fi

# --------------------------------------------------------------------------- #
# Verify the frozen inputs.
#
# Each check is reported on its own, with an accurate reason. An earlier version
# collapsed both checks into one `||` and printed "suites drifted" for ANY
# failure -- including a missing interpreter, which is a wrong diagnosis and
# exactly the kind of misleading error this repository should not produce.
# --------------------------------------------------------------------------- #
echo "== verifying frozen inputs =="
if ! "$PY" scripts/build_suites.py --check; then
  echo
  echo "ABORT: the suite check did not pass. See the error above."
  echo "  Exit 1 from that script means the JSONL on disk no longer matches"
  echo "  build_suites.py. Any other error is a problem running the check itself."
  exit 1
fi
if ! "$PY" scripts/build_manifest.py --check; then
  echo
  echo "ABORT: the manifest check did not pass. See the error above."
  echo "  If the suites changed intentionally, rebuild the manifest with --force"
  echo "  and re-run ALL FOUR precisions -- a partial re-run is not comparable."
  exit 1
fi
echo

# --------------------------------------------------------------------------- #
# Refuse to mix fabricated and measured arms in one results root.
#
# The aggregator warns about synthetic arms, but by then the raw JSONL of a real
# arm may already have been overwritten -- and that file is the evidence, costly
# to reproduce. So the check happens BEFORE anything runs, and refuses.
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
  if [ "$BACKEND" = "mock" ]; then
    echo "  Those are MEASURED results. A pipeline check must not overwrite them."
    echo "  Run:  bash scripts/run_all.sh mock results_mock"
  else
    echo "  Those are FABRICATED mock arms. Aggregating them alongside real ones"
    echo "  would put invented numbers in the findings report."
    echo "  Remove them first:  rm -rf $RESULTS_ROOT"
  fi
  exit 1
fi

# --------------------------------------------------------------------------- #
# Preflight: report every arm's readiness at once, before running anything.
# --------------------------------------------------------------------------- #
echo "== preflight =="
"$PY" scripts/preflight.py --backend "$BACKEND" --config-set "$CONFIG_SET"
PREFLIGHT=$?
echo
if [ $PREFLIGHT -eq 2 ]; then
  echo "ABORT: the backend is unreachable. Nothing was run."
  exit 1
fi
if [ $PREFLIGHT -eq 1 ]; then
  echo "Some models are missing (see above). Continuing anyway: the arms that CAN"
  echo "run will run, and the missing ones are recorded as NOT RUN. Pull them and"
  echo "re-run this script to fill the gaps."
  echo
fi

FAILED=()
RAN=()
for PRECISION in bf16 fp8 q8 q4; do
  echo "== $PRECISION =="
  if "$PY" -m ps5.run --precision "$PRECISION" --backend "$BACKEND" \
       --config-set "$CONFIG_SET" --suite ps1 ps3 \
       --results-root "$RESULTS_ROOT" --quiet; then
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

# A validation run must NEVER overwrite the real findings report or its figures,
# and neither must a different MODEL's run: two config sets measure two different
# models, so each gets its own report and figure directory. Everything a run
# writes is keyed by (backend kind, config set), which is what makes it safe to
# run the small set and the full set on the same checkout.
if [ "$BACKEND" = "mock" ]; then
  FIGURES="reports/figures_mock$SUFFIX"
  REPORT="reports/FINDINGS_MOCK$SUFFIX.md"
  EXTRA="--allow-synthetic"
else
  FIGURES="reports/figures$SUFFIX"
  REPORT="reports/FINDINGS$SUFFIX.md"
  EXTRA=""
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
  exit 1
fi
echo "All four arms completed."
