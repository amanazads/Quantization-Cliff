# PS-5: The Quantization Cliff

_Generated from `aggregate.json` at 2026-09-08T18:47:15.897023+00:00. Metric spec `2.0.0-spec-6.4`. Every figure and table in this document is rendered from raw results; none is typed by hand._

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

**Hardware and software** (identical across arms; the fingerprint is verified by the aggregator, not asserted):

- OS: `macOS-26.5-arm64-arm-64bit-Mach-O`
- CPU: `Apple M1`
- RAM: `8.0 GB`
- Accelerator: `Apple M1` (`metal`)
- CUDA: `n/a` · compute capability: `n/a`
- Python: `3.14.7`
- Code revision: `6d03ce3e47bf23663437199cfd71144ac8ee4d90` **(working tree was dirty — the recorded commit does not fully describe the code that ran)**
- Hardware fingerprint: `sha256:7e8a83e234c408251ee4d09539a0f18df83f3117dd8f2b27ca76f0c958bd5eb8`

**Decoding parameters** (identical across arms; hash-verified):

```json
{
  "temperature": 0.0,
  "top_p": 1.0,
  "top_k": 1,
  "seed": 20260907,
  "max_tokens": 512,
  "stop": [],
  "transport_retries": 2,
  "transport_retry_backoff_s": 2.0,
  "request_timeout_s": 180
}
```

### Arms not run

These required precisions were **not executed**. They are gaps in coverage, and must not be read as null results:

- **FP8** — not run in this comparison set.

---

## 3. Experimental controls

### Held constant

- system prompt
- tool schemas
- evaluation manifest
- decoding parameters
- scorer versions
- case order
- concurrency
- context window

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
> python3 scripts/validation_subset.py export --n 80   # blind, stratified
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

### By violation category

| category | BF16 | Q8 | Q4 |
|---|---|---|---|
| V1 | 0.0% <sub>[0.0%, 16.1%] n=20</sub> ⚠ | 0.0% <sub>[0.0%, 16.1%] n=20</sub> ⚠ | 0.0% <sub>[0.0%, 16.1%] n=20</sub> ⚠ |
| V2 | 30.0% <sub>[14.5%, 51.9%] n=20</sub> ⚠ | 30.0% <sub>[14.5%, 51.9%] n=20</sub> ⚠ | 20.0% <sub>[8.1%, 41.6%] n=20</sub> ⚠ |
| V3 | 0.0% <sub>[0.0%, 16.1%] n=20</sub> ⚠ | 0.0% <sub>[0.0%, 16.1%] n=20</sub> ⚠ | 0.0% <sub>[0.0%, 16.1%] n=20</sub> ⚠ |
| V4 | 5.0% <sub>[0.9%, 23.6%] n=20</sub> ⚠ | 10.0% <sub>[2.8%, 30.1%] n=20</sub> ⚠ | 0.0% <sub>[0.0%, 16.1%] n=20</sub> ⚠ |
| V5 | 0.0% <sub>[0.0%, 16.1%] n=20</sub> ⚠ | 0.0% <sub>[0.0%, 16.1%] n=20</sub> ⚠ | 0.0% <sub>[0.0%, 16.1%] n=20</sub> ⚠ |
| V6 | 5.0% <sub>[0.9%, 23.6%] n=20</sub> ⚠ | 5.0% <sub>[0.9%, 23.6%] n=20</sub> ⚠ | 5.0% <sub>[0.9%, 23.6%] n=20</sub> ⚠ |
| V7 | 5.0% <sub>[0.9%, 23.6%] n=20</sub> ⚠ | 5.0% <sub>[0.9%, 23.6%] n=20</sub> ⚠ | 5.0% <sub>[0.9%, 23.6%] n=20</sub> ⚠ |
| V8 | 10.0% <sub>[2.8%, 30.1%] n=20</sub> ⚠ | 10.0% <sub>[2.8%, 30.1%] n=20</sub> ⚠ | 10.0% <sub>[2.8%, 30.1%] n=20</sub> ⚠ |

### By language

| language | BF16 | Q8 | Q4 |
|---|---|---|---|
| en | 7.5% <sub>[2.6%, 19.9%] n=40</sub> | 7.5% <sub>[2.6%, 19.9%] n=40</sub> | 10.0% <sub>[4.0%, 23.1%] n=40</sub> |
| hi | 12.5% <sub>[5.5%, 26.1%] n=40</sub> | 12.5% <sub>[5.5%, 26.1%] n=40</sub> | 5.0% <sub>[1.4%, 16.5%] n=40</sub> |
| hinglish | 5.0% <sub>[1.4%, 16.5%] n=40</sub> | 7.5% <sub>[2.6%, 19.9%] n=40</sub> | 2.5% <sub>[0.4%, 12.9%] n=40</sub> |
| mr | 2.5% <sub>[0.4%, 12.9%] n=40</sub> | 2.5% <sub>[0.4%, 12.9%] n=40</sub> | 2.5% <sub>[0.4%, 12.9%] n=40</sub> |

---

## 5. PS-3 results — structured output and tool calling

| metric | BF16 | Q8 | Q4 |
|---|---|---|---|
| `task_success_rate` | 35.5% <sub>[29.2%, 42.4%] n=197</sub> | 35.5% <sub>[29.2%, 42.4%] n=197</sub> | 32.8% <sub>[26.7%, 39.6%] n=198</sub> |
| `correct_tool_rate` | 6.8% <sub>[3.6%, 12.4%] n=133</sub> | 6.8% <sub>[3.6%, 12.4%] n=133</sub> | 2.2% <sub>[0.8%, 6.4%] n=134</sub> |
| `argument_accuracy` | 88.9% <sub>[67.2%, 96.9%] n=18</sub> ⚠ | 88.9% <sub>[67.2%, 96.9%] n=18</sub> ⚠ | 66.7% <sub>[30.0%, 90.3%] n=6</sub> ⚠ |
| `structured_output_validity` | 100.0% <sub>[98.1%, 100.0%] n=197</sub> | 100.0% <sub>[98.1%, 100.0%] n=197</sub> | 99.5% <sub>[97.2%, 99.9%] n=198</sub> |
| `malformed_argument_rate` | 0.0% <sub>[0.0%, 1.9%] n=197</sub> | 0.0% <sub>[0.0%, 1.9%] n=197</sub> | 0.5% <sub>[0.1%, 2.8%] n=198</sub> |
| `wrong_tool_rate` | 1.5% <sub>[0.5%, 4.4%] n=197</sub> | 1.5% <sub>[0.5%, 4.4%] n=197</sub> | 0.5% <sub>[0.1%, 2.8%] n=198</sub> |
| `wrong_argument_rate` | 1.0% <sub>[0.3%, 3.6%] n=197</sub> | 1.0% <sub>[0.3%, 3.6%] n=197</sub> | 1.0% <sub>[0.3%, 3.6%] n=198</sub> |
| `spurious_call_rate` | 0.5% <sub>[0.1%, 2.8%] n=197</sub> | 0.5% <sub>[0.1%, 2.8%] n=197</sub> | 0.0% <sub>[0.0%, 1.9%] n=198</sub> |
| `missed_call_rate` | 91.0% <sub>[84.9%, 94.8%] n=133</sub> | 91.0% <sub>[84.9%, 94.8%] n=133</sub> | 96.3% <sub>[91.6%, 98.4%] n=134</sub> |
| `fallback_extraction_rate` | 0.0% <sub>[0.0%, 1.9%] n=197</sub> | 0.0% <sub>[0.0%, 1.9%] n=197</sub> | 0.0% <sub>[0.0%, 1.9%] n=198</sub> |
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

### By language

| language | BF16 | Q8 | Q4 |
|---|---|---|---|
| en | 38.3% <sub>[25.8%, 52.6%] n=47</sub> | 38.3% <sub>[25.8%, 52.6%] n=47</sub> | 35.4% <sub>[23.4%, 49.6%] n=48</sub> |
| hi | 38.0% <sub>[25.9%, 51.8%] n=50</sub> | 38.0% <sub>[25.9%, 51.8%] n=50</sub> | 32.0% <sub>[20.8%, 45.8%] n=50</sub> |
| hinglish | 34.0% <sub>[22.4%, 47.8%] n=50</sub> | 34.0% <sub>[22.4%, 47.8%] n=50</sub> | 32.0% <sub>[20.8%, 45.8%] n=50</sub> |
| mr | 32.0% <sub>[20.8%, 45.8%] n=50</sub> | 32.0% <sub>[20.8%, 45.8%] n=50</sub> | 32.0% <sub>[20.8%, 45.8%] n=50</sub> |

<sub>Task success rate by language.</sub>

---

## 6. Quantization degradation

**Guardrail adherence and over-refusal vs precision**

![Guardrail adherence and over-refusal vs precision](figures-qwen2.5-1.5b/01_guardrail_adherence_vs_precision.png)

**Structured-output metrics vs precision**

![Structured-output metrics vs precision](figures-qwen2.5-1.5b/02_structured_output_vs_precision.png)

**PS-3 failure modes by precision**

![PS-3 failure modes by precision](figures-qwen2.5-1.5b/03_ps3_failure_modes.png)

**Per-language breakdown**

![Per-language breakdown](figures-qwen2.5-1.5b/04_language_breakdown.png)

**English vs Indic**

![English vs Indic](figures-qwen2.5-1.5b/05_english_vs_indic.png)

**PS-1 violation rate by category**

![PS-1 violation rate by category](figures-qwen2.5-1.5b/06_ps1_category_heatmap.png)

**Degradation vs the reference precision**

![Degradation vs the reference precision](figures-qwen2.5-1.5b/07_degradation_vs_reference.png)

---

## 7. The quantization cliff

### Methodology

Fixed in `configs/cliff_criterion.yaml` and `docs/METRICS.md` **before any model was run**. A precision is *past the cliff* on a metric only when **both** conditions hold:

1. **Practical** — degradation vs the reference meets the pre-registered threshold for that metric.
2. **Statistical** — the Newcombe 95% interval for the difference excludes zero.

Requiring both is what stops a large-but-noisy difference at small `n` being reported as a cliff, and equally stops a statistically clean but operationally trivial difference being reported as one.

**The thresholds are an experimental convention of this repository, not defined by the challenge specification.** Safety uses a tighter threshold than structured output, because a conduct breach is a regulatory event whereas a malformed tool call is a retry. Justification is in `docs/METRICS.md` §4.3.

### Verdict per headline metric

#### PS1

**`violation_rate`** — threshold 2.0%, pattern **`none`**

No quantization cliff was detected for `violation_rate` within the tested precision range, at a threshold of 2.0% and 95% confidence, with n=160 at the reference precision. The minimum difference detectable at this sample size was 10.2%; a real effect smaller than that would not have been visible to this experiment.

| precision | value | degradation vs ref | 95% CI on the difference | meets threshold | CI excludes 0 | past cliff |
|---|---|---|---|---|---|---|
| Q8 | 7.5% | 0.6% | [-5.3%, 6.6%] | no | no | no |
| Q4 | 5.0% | -1.9% | [-7.5%, 3.6%] | no | no | no |


**`benign_refusal_rate`** — threshold 5.0%, pattern **`none`**

No quantization cliff was detected for `benign_refusal_rate` within the tested precision range, at a threshold of 5.0% and 95% confidence, with n=32 at the reference precision. The minimum difference detectable at this sample size was 21.2%; a real effect smaller than that would not have been visible to this experiment.

| precision | value | degradation vs ref | 95% CI on the difference | meets threshold | CI excludes 0 | past cliff |
|---|---|---|---|---|---|---|
| Q8 | 0.0% | 0.0% | [-10.7%, 10.7%] | no | no | no |
| Q4 | 0.0% | 0.0% | [-10.7%, 10.7%] | no | no | no |


#### PS3

**`task_success_rate`** — threshold 5.0%, pattern **`none`**

No quantization cliff was detected for `task_success_rate` within the tested precision range, at a threshold of 5.0% and 95% confidence, with n=197 at the reference precision. The minimum difference detectable at this sample size was 14.0%; a real effect smaller than that would not have been visible to this experiment.

| precision | value | degradation vs ref | 95% CI on the difference | meets threshold | CI excludes 0 | past cliff |
|---|---|---|---|---|---|---|
| Q8 | 35.5% | -0.0% | [-9.4%, 9.4%] | no | no | no |
| Q4 | 32.8% | 2.7% | [-12.0%, 6.6%] | no | no | no |


**`structured_output_validity`** — threshold 5.0%, pattern **`none`**

No quantization cliff was detected for `structured_output_validity` within the tested precision range, at a threshold of 5.0% and 95% confidence, with n=197 at the reference precision. The minimum difference detectable at this sample size was 3.9%; a real effect smaller than that would not have been visible to this experiment.

| precision | value | degradation vs ref | 95% CI on the difference | meets threshold | CI excludes 0 | past cliff |
|---|---|---|---|---|---|---|
| Q8 | 100.0% | -0.0% | [-1.9%, 1.9%] | no | no | no |
| Q4 | 99.5% | 0.5% | [-2.8%, 1.5%] | no | no | no |


---

## 8. Production recommendation

### Minimum viable precision: **Q4**

q4 is the lowest-fidelity precision that cleared every headline metric in both suites under the pre-registered criterion. This is a statement about THIS suite, THIS model and THIS hardware at THIS sample size. It is not a claim that the precision is universally production-safe.

### What this recommendation is, and is not

| claim | supported by this experiment? |
|---|---|
| Minimum precision supported by **this** suite, model, hardware and sample size | **Yes** — that is exactly what was measured. |
| Universally safe production precision for collections | **No.** This experiment cannot support that claim and does not make it. |
| Evidence that the recommended precision is production-**safe** | **No.** Absence of a detected cliff is not evidence of safety, particularly where the minimum detectable difference is larger than the effect that would matter operationally. |

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
- **The FP8-above-Q8 fidelity ordering is an assumption**, not a measurement.
- **Determinism is best-effort.** Greedy decoding with a fixed seed is requested, but llama.cpp/vLLM do not guarantee bit-identical output across differing batch or thread configurations.
- **The suites were authored for this repository** from the challenge brief. They are not the official PS-1/PS-3 suites, and results are not comparable with runs on the official ones.
- **One model family at one size, chosen to fit the hardware.** Quantization sensitivity varies sharply with model size, and smaller models are generally LESS robust to it. Two consequences, in opposite directions: a cliff found here is plausibly pessimistic for a larger deployment model, while a NULL here is close to uninformative about one. Check the reference arm's absolute value before reading any null as reassuring -- if the BF16 baseline is already weak on a metric, that metric had little room to degrade and the null reflects a floor effect rather than robustness. See `docs/METRICS.md` section 7.8.

---

## 10. Reproduction

```bash
# 1. environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. verify the frozen inputs are unmodified
python scripts/build_suites.py --check
python scripts/build_manifest.py --check

# 3. run each arm (see README for backend setup)
python -m ps5.run --precision bf16 --backend ollama --suite ps1 ps3
python -m ps5.run --precision fp8 --backend ollama --suite ps1 ps3   # NOT RUN in this comparison set
python -m ps5.run --precision q8 --backend ollama --suite ps1 ps3
python -m ps5.run --precision q4 --backend ollama --suite ps1 ps3

# 4. aggregate, plot, report
python scripts/aggregate_results.py
python scripts/make_plots.py
python scripts/generate_report.py
```

Reproduction is only exact when these match the values in §2: `manifest_hash`, `system_prompt_hash`, `tool_schema_hash`, `generation_config_hash`, `guardrail_rules_hash`, `hardware_fingerprint`. They are recorded in every run's `metadata.json`.

