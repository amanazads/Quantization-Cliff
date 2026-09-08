# PS-5: The Quantization Cliff

_Generated from `aggregate.json` at 2026-09-08T19:06:24.891040+00:00. Metric spec `2.0.0-spec-6.4`. Every figure and table in this document is rendered from raw results; none is typed by hand._

---

## 1. Objective

Determine how quantization affects an open-weight collections agent along three separately-reported axes — guardrail adherence (PS-1), structured output and tool-calling validity (PS-3), and language-specific behaviour — and locate the precision at which degradation becomes meaningful, using a criterion fixed before any result was observed.

Safety and structured output are **never combined into a single score**. The central question PS-5 asks is whether they degrade differently, and an average would destroy exactly that signal.

---

## 2. Experimental setup

| | BF16 | Q8 | Q4 |
|---|---|---|---|
| model | `qwen2.5-1.5b-instruct` | `qwen2.5-1.5b-instruct` | `qwen2.5-1.5b-instruct` |
| model tag | `qwen2.5:1.5b-instruct-fp16` | `qwen2.5:1.5b-instruct-q8_0` | `qwen2.5:1.5b-instruct-q4_K_M` |
| quantization format | GGUF F16 (IEEE half) -- SUBSTITUTED FOR BF16 | GGUF Q8_0 | GGUF Q4_K_M |
| resolved quant level | F16 | Q8_0 | Q4_K_M |
| weights digest | `—` | `—` | `—` |
| backend | `ollama` | `ollama` | `ollama` |
| cases run | ps1=192, ps3=200 | ps1=192, ps3=200 | ps1=192, ps3=200 |
| repeats | 1 | 1 | 1 |

**Hardware and software** (identical across arms; the fingerprint is verified by the aggregator, not asserted): `macOS-26.5-arm64-arm-64bit-Mach-O` · `Apple M1` · 8.0 GB RAM · accelerator `Apple M1` (`metal`) · CUDA `n/a` · Python `3.14.7` · code `6d03ce3e47bf` **(working tree dirty — the recorded commit does not fully describe the code that ran)** · fingerprint `sha256:7e8a83e234c4…`

**Decoding** (identical across arms, hash-verified): greedy — temperature 0.0, top_p 1.0, top_k 1, seed 20260907, max_tokens 512; serial execution; thinking mode disabled as the specification requires.

### Arms not run

These required precisions were **not executed**. They are gaps in coverage, and must not be read as null results:

- **FP8** — refused, not skipped. Ollama publishes no FP8 or MXFP8 GGUF for Qwen2.5-1.5B-Instruct, and llama.cpp has no FP8 tensor type to convert one into.

---

## 3. Experimental controls

### Held constant

Held constant and hash-verified: system prompt; tool schemas; evaluation manifest; decoding parameters; scorer versions; case order; concurrency; context window.

The aggregator **verifies** these rather than trusting them: it compares `manifest_hash`, `system_prompt_hash`, `tool_schema_hash`, `generation_config_hash`, `metric_spec_version`, `guardrail_rules_hash`, `hardware_fingerprint`, `thinking_mode` across arms and refuses to produce a comparison when any of them diverges.

### Verification result

✅ All control fields matched across every arm in this comparison.


### Could NOT be held constant

- **[material] `DEV-BF16-OLLAMA-F16`** (bf16): THIS ARM IS F16, NOT BF16. Qwen2.5 was trained in bfloat16, so this file is a format conversion of the training weights rather than the training weights themselves. F16 trades exponent range for mantissa precision.
  - _Impact:_ Any weight whose magnitude falls outside F16's narrower dynamic range is flushed or clipped on conversion. At 1.5B parameters that is very unlikely to be measurable, but this is the arm every other arm is subtracted from, so an unmeasured assumption sits underneath every degradation figure in this set. Report the reference as "F16", never as "BF16", and do not pool it with a genuine BF16 arm from another set.
  - _Remediation:_ Use configs/experiments/ (Qwen3.5-4B), whose reference arm is a genuine bf16 GGUF. It needs more than 8 GB of unified memory.

- Wall-clock time and machine thermal state differed between arms; runs were sequential.
- Kernel selection inside the inference engine differs per quantization format by design. That is inherent to the treatment rather than a failure of control, but it does mean 'precision' here means 'precision as served by this stack', not an isolated numerical-format change.

---

## 3b. Scorer validation against human labels

> **NOT YET MEASURED.** The specification requires the automated scorer to be validated against human labels on a subset, with the agreement reported. That has not been done for this run, so every absolute violation rate below rests on an unvalidated scorer and must be read as provisional.

> To close this gap:
> ```bash
> python3 scripts/validation_subset.py export --results-root results-qwen2.5-1.5b --n 80   # blind, stratified
> # a human labels reports/validation/ps1_validation_labelled.csv
> python3 scripts/validation_subset.py score
> ```

---

## 4. PS-1 results — guardrail adherence

| metric | BF16 | Q8 | Q4 |
|---|---|---|---|
| `violation_rate` | 6.9% <sub>[3.9%, 11.9%] n=160</sub> | 7.5% <sub>[4.3%, 12.7%] n=160</sub> | 5.0% <sub>[2.6%, 9.6%] n=160</sub> |
| `compliance_rate` | 93.1% <sub>[88.1%, 96.1%] n=160</sub> | 92.5% <sub>[87.3%, 95.7%] n=160</sub> | 95.0% <sub>[90.4%, 97.4%] n=160</sub> |
| `benign_refusal_rate` | 0.0% <sub>[0.0%, 10.7%] n=32</sub> | 0.0% <sub>[0.0%, 10.7%] n=32</sub> | 0.0% <sub>[0.0%, 10.7%] n=32</sub> |
| `generation_failure_rate` | 0.0% <sub>[0.0%, 2.0%] n=192</sub> | 0.0% <sub>[0.0%, 2.0%] n=192</sub> | 0.0% <sub>[0.0%, 2.0%] n=192</sub> |

<sub>Values are point estimates with Wilson 95% intervals and the denominator. ⚠ marks a cell below the pre-registered small-sample threshold; those are directional and no significance is claimed.</sub>

### English vs Indic

| precision | English | Indic | delta | 95% CI on the delta | significant |
|---|---|---|---|---|---|
| BF16 | 7.5% <sub>[2.6%, 19.9%] n=40</sub> | 6.7% <sub>[3.4%, 12.6%] n=120</sub> | -0.8% | [-13.6%, 6.9%] | no |
| Q8 | 7.5% <sub>[2.6%, 19.9%] n=40</sub> | 7.5% <sub>[4.0%, 13.6%] n=120</sub> | 0.0% | [-12.9%, 7.9%] | no |
| Q4 | 10.0% <sub>[4.0%, 23.1%] n=40</sub> | 3.3% <sub>[1.3%, 8.3%] n=120</sub> | -6.7% | [-19.9%, 1.1%] | no |

<sub>Positive delta means Indic-language safety is worse than English.</sub>

<sub>Per-category (V1–V8) and per-language breakdowns are in `FINDINGS_FULL.md`, rendered from the same aggregate.</sub>

---

## 5. PS-3 results — structured output and tool calling

| metric | BF16 | Q8 | Q4 |
|---|---|---|---|
| `task_success_rate` | 35.5% <sub>[29.2%, 42.4%] n=197</sub> | 35.5% <sub>[29.2%, 42.4%] n=197</sub> | 32.8% <sub>[26.7%, 39.6%] n=198</sub> |
| `correct_tool_rate` | 6.8% <sub>[3.6%, 12.4%] n=133</sub> | 6.8% <sub>[3.6%, 12.4%] n=133</sub> | 2.2% <sub>[0.8%, 6.4%] n=134</sub> |
| `argument_accuracy` | 88.9% <sub>[67.2%, 96.9%] n=18</sub> ⚠ | 88.9% <sub>[67.2%, 96.9%] n=18</sub> ⚠ | 66.7% <sub>[30.0%, 90.3%] n=6</sub> ⚠ |
| `structured_output_validity` | 100.0% <sub>[98.1%, 100.0%] n=197</sub> | 100.0% <sub>[98.1%, 100.0%] n=197</sub> | 99.5% <sub>[97.2%, 99.9%] n=198</sub> |
| `malformed_argument_rate` | 0.0% <sub>[0.0%, 1.9%] n=197</sub> | 0.0% <sub>[0.0%, 1.9%] n=197</sub> | 0.5% <sub>[0.1%, 2.8%] n=198</sub> |
| `missed_call_rate` | 91.0% <sub>[84.9%, 94.8%] n=133</sub> | 91.0% <sub>[84.9%, 94.8%] n=133</sub> | 96.3% <sub>[91.6%, 98.4%] n=134</sub> |
| `generation_failure_rate` | 1.5% <sub>[0.5%, 4.3%] n=200</sub> | 1.5% <sub>[0.5%, 4.3%] n=200</sub> | 1.0% <sub>[0.3%, 3.6%] n=200</sub> |

<sub>Values are point estimates with Wilson 95% intervals and the denominator. ⚠ marks a cell below the pre-registered small-sample threshold; those are directional and no significance is claimed.</sub>

`correct_tool_rate` counts a case as correct when the tool identity is right, **regardless of the argument values**. `task_success_rate` requires the arguments to be right too. The gap between them is the argument-corruption rate, and it is the single most operationally dangerous failure in this suite: a `capture_ptp` for ₹50,000 instead of ₹5,000 is a well-formed, correctly-routed, wrong commitment.

### English vs Indic (task success)

| precision | English | Indic | delta | 95% CI on the delta | significant |
|---|---|---|---|---|---|
| BF16 | 38.3% <sub>[25.8%, 52.6%] n=47</sub> | 34.7% <sub>[27.5%, 42.6%] n=150</sub> | -3.6% | [-19.6%, 11.2%] | no |
| Q8 | 38.3% <sub>[25.8%, 52.6%] n=47</sub> | 34.7% <sub>[27.5%, 42.6%] n=150</sub> | -3.6% | [-19.6%, 11.2%] | no |
| Q4 | 35.4% <sub>[23.4%, 49.6%] n=48</sub> | 32.0% <sub>[25.1%, 39.8%] n=150</sub> | -3.4% | [-19.2%, 10.9%] | no |

<sub>NEGATIVE delta means Indic tool-calling is worse than English.</sub>

---

## 6. Quantization degradation

**Guardrail adherence and over-refusal vs precision**

![Guardrail adherence and over-refusal vs precision](figures-qwen2.5-1.5b/01_guardrail_adherence_vs_precision.png)

**Structured-output metrics vs precision**

![Structured-output metrics vs precision](figures-qwen2.5-1.5b/02_structured_output_vs_precision.png)

**Degradation vs the reference precision**

![Degradation vs the reference precision](figures-qwen2.5-1.5b/07_degradation_vs_reference.png)

---

## 7. The quantization cliff

### Methodology

Fixed in `configs/cliff_criterion.yaml` and `docs/METRICS.md` **before any model was run**. A precision is *past the cliff* on a metric only when **both** hold: degradation vs the reference meets the pre-registered threshold for that metric, **and** the Newcombe 95% interval for the difference excludes zero. Requiring both stops a large-but-noisy difference at small `n` being called a cliff, and equally stops a statistically clean but operationally trivial one.

**The thresholds are an experimental convention of this repository, not defined by the challenge specification.** Safety uses a tighter threshold than structured output, because a conduct breach is a regulatory event whereas a malformed tool call is a retry. Justification is in `docs/METRICS.md` §4.3.

### Verdict per headline metric

| suite · metric | Q8 | Q4 | pattern | cliff at |
|---|---|---|---|---|
| PS1 · `violation_rate` | +0.6 | -1.9 | `none` | — |
| PS1 · `benign_refusal_rate` | +0.0 | +0.0 | `none` | — |
| PS3 · `task_success_rate` | -0.0 | +2.7 | `none` | — |
| PS3 · `structured_output_validity` | -0.0 | +0.5 | `none` | — |

<sub>Cells are degradation vs the reference in percentage points; positive is always worse. **✱** = past the cliff (threshold met AND the difference interval excludes zero). **·** = threshold met but the interval still includes zero, so it is NOT called a cliff. Per-metric intervals and sample sizes are in `FINDINGS_FULL.md`.</sub>

- **No cliff detected** on: PS1 `violation_rate`, PS1 `benign_refusal_rate`, PS3 `task_success_rate`, PS3 `structured_output_validity`.

**How much each null is worth.** A null means something only where the minimum detectable difference (MDD) at the achieved sample size is no larger than the effect the criterion was pre-registered to look for.

| suite · metric | threshold | MDD | is this null informative? |
|---|---|---|---|
| PS1 · `violation_rate` | 2.0% | 10.2% | **No** — ~5× too few cases to see a 2.0% effect |
| PS1 · `benign_refusal_rate` | 5.0% | 21.2% | **No** — ~4× too few cases to see a 5.0% effect |
| PS3 · `task_success_rate` | 5.0% | 14.0% | **No** — ~3× too few cases to see a 5.0% effect |
| PS3 · `structured_output_validity` | 5.0% | 3.9% | **Yes** — a cliff of the pre-registered size would have been visible |

<sub>1 of 4 null results were adequately powered. For the rest, a real effect smaller than the MDD was invisible to this experiment rather than absent — they are not evidence that quantization did no harm.</sub>

---

## 8. Production recommendation

### Minimum viable precision: **Q4**

q4 is the lowest-fidelity precision that cleared every headline metric in both suites under the pre-registered criterion. This is a statement about THIS suite, THIS model and THIS hardware at THIS sample size. It is not a claim that the precision is universally production-safe.

### What this recommendation is, and is not

**Supported:** the minimum precision for *this* suite, model, hardware and sample size — that is exactly what was measured. **Not supported:** a universally safe production precision, and any claim that the recommended precision is production-*safe*. Absence of a detected cliff is not evidence of safety, especially where the minimum detectable difference exceeds the effect that would matter operationally — see the power table in §7.

Confidence: **moderate for the direction of the effect, low for its exact magnitude.** Sample sizes are fixed and modest, the PS-1 scorer's agreement with human judgement is unmeasured, and the suites are single-turn.

---

## 9. Limitations

Stated in advance in `docs/METRICS.md` §7 rather than discovered afterwards.

- **Incomplete precision coverage: FP8 was not run.** The cliff can only be located among the arms that were actually executed; a cliff could lie at an untested precision.
- **The PS-1 scorer is rule-based and its validity is unmeasured.** It has not been checked against human labels. It will miss paraphrased violations and may fire on quoted or negated text. Its error is constant across arms, so it biases absolute violation rates more than it biases the between-precision comparison — but the absolute numbers should not be quoted as a safety rate in isolation. This is the single biggest methodological weakness here.
- **One target category is scored per case.** Cross-category violations are not detected.
- **Free-text tool arguments are not scored**, so `argument_accuracy` covers structured fields only.
- **Per-category and per-language cells are small** and are directional only.
- **Single-turn only.** Multi-turn drift, where quantization damage plausibly compounds, is not measured at all.
- _4 further limitations are listed in `FINDINGS_FULL.md` and `docs/METRICS.md` §7._

---

## 10. Reproduction

```bash
pip install -r requirements.txt && export PYTHONPATH=$PWD/src
python3 scripts/build_suites.py --check && python3 scripts/build_manifest.py --check
bash scripts/run_all.sh ollama qwen2.5-1.5b
```

Reproduction is exact only when the six control hashes in §2 match; all are recorded in every run's `metadata.json`. Runbook in README §4, full breakdowns in `FINDINGS_FULL.md`.

