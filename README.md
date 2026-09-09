# Quantization Cliff

A reproducible empirical evaluation of post-training quantization effects on open-weight debt collections agents, developed for the Predixion AI Challenge (Track 2, Problem Statement 5).

---

## Problem

Problem Statement 5 (PS-5) investigates whether post-training quantization creates an abrupt degradation cliff on safety guardrail adherence (PS-1) or structured tool-calling correctness (PS-3), or whether performance degrades smoothly across bit-widths.

Crucially, safety and tool-calling validity are evaluated and reported independently. Combining them into a single score would mask distinct failure modes where an agent remains safe but structurally broken, or structurally valid but legally non-compliant.

---

## Experiment

- **Model**: `Qwen2.5-1.5B-Instruct`
- **Backend**: Ollama (Metal acceleration on macOS)
- **Evaluated Arms**:
  1. **F16 Reference**: `qwen2.5:1.5b-instruct-fp16` (3.1 GB)
  2. **Q8_0**: `qwen2.5:1.5b-instruct-q8_0` (1.6 GB)
  3. **Q4_K_M**: `qwen2.5:1.5b-instruct-q4_K_M` (986 MB)
- **FP8 Status**: **NOT RUN**. A runnable FP8/MXFP8 GGUF artifact is unavailable for this model under Ollama. It is reported as an explicit precision coverage gap rather than simulated or treated as a null result.

---

## Why 1.5B

The original challenge turn-loop candidate was Qwen3.5-4B. However, its unquantized BF16 reference arm is approximately 9.3 GB, exceeding the 8 GB unified memory limit of an Apple M1 development machine.

To preserve scientific rigor, every degradation delta must be anchored against a runnable full-precision reference executed on the same machine. Downscaling to `Qwen2.5-1.5B-Instruct` fits all tested arms comfortably in unified memory, providing genuine empirical measurements with no out-of-memory crashes or swap-induced latency confounds.

---

## What Was Held Constant

All non-quantization variables were strictly controlled and hash-verified across arms:
- **System prompt**: Pinned to v2 (`prompts/collections_agent_v2.md`, hash verified)
- **Tool schemas**: Published challenge signatures frozen (`schemas/tools.json`)
- **Evaluation manifest**: Exact 392 cases (192 PS-1 + 200 PS-3) presented in identical order
- **Decoding parameters**: Greedy decoding (`temperature=0.0`, `top_k=1`, `top_p=1.0`, `max_tokens=512`)
- **Seed**: Fixed seed (`20260907`)
- **Thinking mode**: Disabled uniformly (`think: false`)
- **Scorer & rules**: Deterministic regex/keyword guardrail rules (`guardrail_rules.json`)
- **Hardware & environment**: Same Apple M1 machine (hardware fingerprint verified)
- **Execution order**: Sequential execution with backend memory unloads between arms

---

## Results

Aggregated results across 392 evaluation cases per arm:

| Headline Metric | Direction | Threshold | F16 (Ref) | Q8_0 | Q4_K_M | Detected Cliff? |
|---|---|---|---|---|---|---|
| **PS-1 Violation Rate** | Lower is better | 2.0 pp | 6.9% | 7.5% (+0.6) | 5.0% (-1.9) | **None** (underpowered) |
| **PS-1 Benign Refusal Rate** | Lower is better | 5.0 pp | 0.0% | 0.0% (+0.0) | 0.0% (+0.0) | **None** (underpowered) |
| **PS-3 Task Success Rate** | Higher is better | 5.0 pp | 35.5% | 35.5% (-0.0) | 32.8% (+2.7) | **None** (underpowered) |
| **PS-3 Output Validity** | Higher is better | 5.0 pp | 100.0% | 100.0% (-0.0) | 99.5% (+0.5) | **None** (adequately powered) |

*Notes: Parentheses indicate degradation vs F16 in percentage points (positive is worse). Values carry Wilson 95% confidence intervals in the full report.*

---

## Key Finding

> *"Within the tested Qwen2.5-1.5B F16/Q8/Q4 range and this sample size/hardware setup, Q4 is the lowest tested precision without a detected cliff on the headline metrics."*

- **Structured output validity** remained stable across precisions down to Q4 (100% → 100% → 99.5%) with adequate statistical power (MDD 3.9% $\le$ 5.0% threshold).
- **No cliff detected** on safety violation rate or task success rate under the pre-registered criterion (threshold met AND difference interval excludes zero).
- **Important caveat**: This does **not** imply Q4 is universally production-ready. Absence of evidence is not evidence of absence on metrics with wide confidence intervals.

---

## Limitations

1. **Reference dtype**: F16 is used as the local reference because the Qwen2.5-1.5B Ollama artifact available for this setup is F16 rather than BF16 (`DEV-F16-OLLAMA-REF`).
2. **Missing FP8 rung**: No FP8 GGUF artifact is available for Ollama on this model, leaving a gap between F16 and Q8.
3. **Model size**: 1.5B parameters is smaller than the target 4B model; smaller models may exhibit different sensitivity profiles.
4. **Statistical power**: 3 of 4 null results (PS-1 violation, benign refusal, PS-3 task success) are underpowered; real effects smaller than the minimum detectable difference (10.2%–21.2%) were invisible rather than absent.
5. **Tool-calling floor**: Correct tool selection was low at reference (6.8%), leaving limited headroom to detect degradation.
6. **Scorer validation**: Automated PS-1 scoring was validated on an $n=80$ human-labelled subset with Cohen's $\kappa = 0.471$ (moderate agreement, 77.5% raw agreement). Rule bias is held constant across arms, making relative deltas more reliable than absolute rates.

---

## Run

To reproduce the experiment from scratch:

```bash
# 1. Environment setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$PWD/src:$PYTHONPATH"

# 2. Verify frozen test suites and run unit tests (239 tests)
python3 scripts/build_suites.py --check
python3 scripts/build_manifest.py --check
pytest -q

# 3. Preflight check for Ollama models
python3 scripts/preflight.py --backend ollama --config-set qwen2.5-1.5b

# 4. Pull models (if not cached)
ollama pull qwen2.5:1.5b-instruct-fp16
ollama pull qwen2.5:1.5b-instruct-q8_0
ollama pull qwen2.5:1.5b-instruct-q4_K_M

# 5. Run full evaluation pipeline
bash scripts/run_all.sh ollama qwen2.5-1.5b

# 6. Aggregate results, generate plots, and render report
python3 scripts/aggregate_results.py
python3 scripts/make_plots.py
python3 scripts/generate_report.py
```

---

## Output

- **Terminal**: Live preflight checks, per-arm execution logs, MDD power tables, and cliff detection summaries.
- `results-qwen2.5-1.5b/`: Raw evidence per arm (`f16/`, `q8/`, `q4/`) containing `metadata.json`, `ps1_results.jsonl`, and `ps3_results.jsonl`, plus `fp8/NOT_RUN.json`.
- `results-qwen2.5-1.5b/aggregate/`: `aggregate.json` (full data & statistical intervals), `summary.csv`, and `summary.md`.
- `reports/FINDINGS.md`: The primary submission findings document (~227 lines, $\le$ 4 pages).
- `reports/FINDINGS.pdf`: Rendered submission PDF report.
- `reports/figures/`: High-resolution figures (`01` through `07`).
- `reports/validation/`: Human-annotated validation data (`agreement.json`, `agreement.md`, `ps1_validation_labelled.csv`).

---

## Project Structure

```text
configs/
  base.yaml                      # Shared controls inherited by all precisions
  cliff_criterion.yaml           # Pre-registered cliff decision criterion
  guardrail_rules.json           # PS-1 conduct rule patterns
  experiments-qwen2.5-1.5b/      # Per-arm configs (f16, fp8, q8, q4)
data/
  evaluation_manifest.json       # Frozen manifest of 392 cases
  ps1_guardrail_suite.jsonl      # 192 safety cases
  ps3_toolcall_suite.jsonl       # 200 tool-calling cases
docs/
  METRICS.md                     # Statistical formulas & pre-registered thresholds
prompts/
  collections_agent_v2.md        # Frozen system prompt
reports/
  FINDINGS.md                    # Primary submission findings document
  FINDINGS.pdf                   # Formatted PDF deliverable
  figures/                       # Generated charts
  validation/                    # Scorer vs human validation data
results-qwen2.5-1.5b/            # Raw JSONL and aggregate outputs
schemas/
  tools.json                     # Challenge-defined tool schemas
scripts/                         # Workflow and verification scripts
src/ps5/                         # Core execution, scoring, and analysis engine
tests/                           # Unit and integration test suite (239 tests)
```

---

## License

This project is licensed under the Apache License 2.0 - see [LICENSE](LICENSE) and [NOTICE](NOTICE) for details.
