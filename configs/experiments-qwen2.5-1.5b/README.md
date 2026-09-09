# Config set: `qwen2.5-1.5b`

The primary submitted experiment set, evaluating Qwen2.5-1.5B-Instruct locally.

## Why this set exists

The intended set targets Qwen3.5-4B, which the specification names as its
"turn-loop candidate, smallest viable". Its reference arm is `qwen3.5:4b-bf16`
at **9.3 GB**, which does not fit on an 8 GB machine — and the reference arm is
the one every degradation figure is measured against, so without it there is no
experiment at all, only three unanchored numbers.

This set drops to **Qwen2.5-1.5B-Instruct**, whose largest arm is 3.1 GB. It
runs on 8 GB of unified memory today, with no download beyond what a developer
working on this repository is likely to already have.

That is a real trade, not a free one. What it costs is stated in each config's
`deviations` and summarised here.

## Arms

| Precision | Tag | Size | Status |
|---|---|---|---|
| F16 (reference) | `qwen2.5:1.5b-instruct-fp16` | 3.1 GB | **F16, not BF16** — material deviation |
| FP8 | — | — | **NOT RUN** — no FP8 GGUF exists for this model |
| Q8_0 | `qwen2.5:1.5b-instruct-q8_0` | 1.6 GB | genuine |
| Q4_K_M | `qwen2.5:1.5b-instruct-q4_K_M` | 986 MB | genuine |

## What this set cannot tell you

1. **The reference is IEEE F16, not bfloat16.** Same 16 bits, different split:
   F16 has 5 exponent bits and 10 mantissa; BF16 has 8 and 7. Qwen2.5 was
   trained in bfloat16, so the F16 file is a *converted* copy of the weights,
   not the training dtype. For a 1.5B model the conversion is very unlikely to
   lose anything measurable — F16's dynamic range comfortably covers these
   weights — but "very unlikely" is not "verified", and it is the reference
   every other arm is subtracted from. Recorded as `DEV-F16-OLLAMA-REF`.

2. **The FP8 rung is missing entirely.** Ollama publishes no FP8 or MXFP8 GGUF
   for Qwen2.5-1.5B. This is reported as a gap in precision coverage, never as
   "FP8 showed no degradation". A three-point curve (F16 → Q8 → Q4) cannot
   distinguish a cliff located between FP8 and Q8 from one between Q8 and Q4.

3. **A 1.5B model is not a 4B model.** Smaller models are generally *more*
   fragile under quantization, so a cliff found here is not evidence that the
   same cliff exists at 4B or 9B, and a null here is weaker evidence of safety
   at larger sizes than it looks. The direction of the bias is at least known
   and stated: results here are more likely to over-state degradation than to
   hide it.

None of this is a reason not to run it. It is a reason to write down what was
run, which the findings report does automatically from these configs.

## Running it

```bash
python3 scripts/preflight.py --backend ollama --config-set qwen2.5-1.5b
bash scripts/run_all.sh ollama qwen2.5-1.5b
```

Results land in `results-qwen2.5-1.5b/`, never in `results/`, so the two models
can never be aggregated into one comparison. The aggregator would refuse that
anyway — differing `model.family` is a hard failure — but a shared directory
would still let one run overwrite the other's raw JSONL.
