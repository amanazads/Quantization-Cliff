# PS-5: The Quantization Cliff

> # ⚠ SYNTHETIC VALIDATION ARTEFACT — NOT A RESULT
>
> Every number in this document was produced by the deterministic **mock backend**, which fabricates output from a hand-written failure profile. It exists to demonstrate that the pipeline computes, aggregates and reports correctly. It is **not a measurement of any model**, and no statement in it may be cited as a PS-5 finding.

_Generated from `aggregate.json` at 2026-09-08T13:13:42.298630+00:00. Metric spec `1.0.0`. Every figure and table in this document is rendered from raw results; none is typed by hand._

---

## 1. Objective

Determine how quantization affects an open-weight collections agent along three separately-reported axes — guardrail adherence (PS-1), structured output and tool-calling validity (PS-3), and language-specific behaviour — and locate the precision at which degradation becomes meaningful, using a criterion fixed before any result was observed.

Safety and structured output are **never combined into a single score**. The central question PS-5 asks is whether they degrade differently, and an average would destroy exactly that signal.

---

## 2. Experimental setup

| | BF16 | FP8 | Q8 | Q4 |
|---|---|---|---|---|
| model | `qwen3.5-4b-base` | `qwen3.5-4b-base` | `qwen3.5-4b-base` | `qwen3.5-4b-base` |
| model tag | `mock:bf16` | `mock:fp8` | `mock:q8` | `mock:q4` |
| quantization format | SYNTHETIC -- not a real quantization | SYNTHETIC -- not a real quantization | SYNTHETIC -- not a real quantization | SYNTHETIC -- not a real quantization |
| resolved quant level | — | — | — | — |
| weights digest | `—` | `—` | `—` | `—` |
| backend | `mock` | `mock` | `mock` | `mock` |
| cases run | ps1=192, ps3=200 | ps1=192, ps3=200 | ps1=192, ps3=200 | ps1=192, ps3=200 |
| repeats | 1 | 1 | 1 | 1 |

**Hardware and software** (identical across arms; the fingerprint is verified by the aggregator, not asserted):

- OS: `Linux-6.8.0-136-generic-aarch64-with-glibc2.35`
- CPU: `None`
- RAM: `3.82 GB`
- Accelerator: `None` (`None`)
- CUDA: `n/a` · compute capability: `n/a`
- Python: `3.10.12`
- Code revision: `18f42c3f9d3cf4c153d0b1b00697869ec32c568c` **(working tree was dirty — the recorded commit does not fully describe the code that ran)**
- Hardware fingerprint: `sha256:83402de9a7e0792ff1425d2bc580b2e5b5adc67b80056b4222713c730d0b0dfd`

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

- ⚠ Arms ['bf16', 'fp8', 'q4', 'q8'] were produced by the SYNTHETIC mock backend. Their numbers are FABRICATED pipeline-validation fixtures and are not measurements of any model.

### Could NOT be held constant

- **[blocking] `DEV-BF16-MOCK-1`** (bf16): Output came from the deterministic mock backend. These numbers are a pipeline-validation fixture, not a measurement of any model, and must never be reported as a PS-5 result.
- **[blocking] `DEV-FP8-MOCK-1`** (fp8): Output came from the deterministic mock backend. These numbers are a pipeline-validation fixture, not a measurement of any model, and must never be reported as a PS-5 result.
- **[blocking] `DEV-Q4-MOCK-1`** (q4): Output came from the deterministic mock backend. These numbers are a pipeline-validation fixture, not a measurement of any model, and must never be reported as a PS-5 result.
- **[blocking] `DEV-Q8-MOCK-1`** (q8): Output came from the deterministic mock backend. These numbers are a pipeline-validation fixture, not a measurement of any model, and must never be reported as a PS-5 result.

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

| metric | BF16 | FP8 | Q8 | Q4 |
|---|---|---|---|---|
| `violation_rate` | 3.1% <sub>[1.3%, 7.1%] n=160</sub> | 1.9% <sub>[0.6%, 5.4%] n=160</sub> | 6.2% <sub>[3.4%, 11.1%] n=160</sub> | 12.3% <sub>[8.0%, 18.4%] n=155</sub> |
| `compliance_rate` | 96.9% <sub>[92.9%, 98.7%] n=160</sub> | 98.1% <sub>[94.6%, 99.4%] n=160</sub> | 93.8% <sub>[88.9%, 96.6%] n=160</sub> | 87.7% <sub>[81.6%, 92.0%] n=155</sub> |
| `benign_refusal_rate` | 3.1% <sub>[0.6%, 15.7%] n=32</sub> | 3.1% <sub>[0.6%, 15.7%] n=32</sub> | 3.1% <sub>[0.6%, 15.7%] n=32</sub> | 12.5% <sub>[5.0%, 28.1%] n=32</sub> |
| `generation_failure_rate` | 0.0% <sub>[0.0%, 2.0%] n=192</sub> | 0.0% <sub>[0.0%, 2.0%] n=192</sub> | 0.0% <sub>[0.0%, 2.0%] n=192</sub> | 2.6% <sub>[1.1%, 6.0%] n=192</sub> |

<sub>Values are point estimates with Wilson 95% intervals and the denominator. ⚠ marks a cell below the pre-registered small-sample threshold; those are directional and no significance is claimed.</sub>

### English vs Indic

| precision | English | Indic | delta | 95% CI on the delta | significant |
|---|---|---|---|---|---|
| BF16 | 5.0% <sub>[1.4%, 16.5%] n=40</sub> | 2.5% <sub>[0.9%, 7.1%] n=120</sub> | -2.5% | [-14.1%, 3.3%] | no |
| FP8 | 7.5% <sub>[2.6%, 19.9%] n=40</sub> | 0.0% <sub>[0.0%, 3.1%] n=120</sub> | -7.5% | [-19.9%, -1.7%] | yes |
| Q8 | 5.0% <sub>[1.4%, 16.5%] n=40</sub> | 6.7% <sub>[3.4%, 12.6%] n=120</sub> | 1.7% | [-10.3%, 8.6%] | no |
| Q4 | 13.2% <sub>[5.8%, 27.3%] n=38</sub> | 12.0% <sub>[7.3%, 19.1%] n=117</sub> | -1.2% | [-16.1%, 9.1%] | no |

<sub>Positive delta means Indic-language safety is worse than English.</sub>

### By violation category

| category | BF16 | FP8 | Q8 | Q4 |
|---|---|---|---|---|
| V1 | 5.0% <sub>[0.9%, 23.6%] n=20</sub> ⚠ | 5.0% <sub>[0.9%, 23.6%] n=20</sub> ⚠ | 0.0% <sub>[0.0%, 16.1%] n=20</sub> ⚠ | 16.7% <sub>[5.8%, 39.2%] n=18</sub> ⚠ |
| V2 | 0.0% <sub>[0.0%, 16.1%] n=20</sub> ⚠ | 5.0% <sub>[0.9%, 23.6%] n=20</sub> ⚠ | 10.0% <sub>[2.8%, 30.1%] n=20</sub> ⚠ | 15.0% <sub>[5.2%, 36.0%] n=20</sub> ⚠ |
| V3 | 5.0% <sub>[0.9%, 23.6%] n=20</sub> ⚠ | 0.0% <sub>[0.0%, 16.1%] n=20</sub> ⚠ | 20.0% <sub>[8.1%, 41.6%] n=20</sub> ⚠ | 15.0% <sub>[5.2%, 36.0%] n=20</sub> ⚠ |
| V4 | 10.0% <sub>[2.8%, 30.1%] n=20</sub> ⚠ | 0.0% <sub>[0.0%, 16.1%] n=20</sub> ⚠ | 0.0% <sub>[0.0%, 16.1%] n=20</sub> ⚠ | 15.0% <sub>[5.2%, 36.0%] n=20</sub> ⚠ |
| V5 | 0.0% <sub>[0.0%, 16.1%] n=20</sub> ⚠ | 0.0% <sub>[0.0%, 16.1%] n=20</sub> ⚠ | 5.0% <sub>[0.9%, 23.6%] n=20</sub> ⚠ | 10.5% <sub>[2.9%, 31.4%] n=19</sub> ⚠ |
| V6 | 0.0% <sub>[0.0%, 16.1%] n=20</sub> ⚠ | 0.0% <sub>[0.0%, 16.1%] n=20</sub> ⚠ | 0.0% <sub>[0.0%, 16.1%] n=20</sub> ⚠ | 10.0% <sub>[2.8%, 30.1%] n=20</sub> ⚠ |
| V7 | 5.0% <sub>[0.9%, 23.6%] n=20</sub> ⚠ | 5.0% <sub>[0.9%, 23.6%] n=20</sub> ⚠ | 5.0% <sub>[0.9%, 23.6%] n=20</sub> ⚠ | 0.0% <sub>[0.0%, 16.8%] n=19</sub> ⚠ |
| V8 | 0.0% <sub>[0.0%, 16.1%] n=20</sub> ⚠ | 0.0% <sub>[0.0%, 16.1%] n=20</sub> ⚠ | 10.0% <sub>[2.8%, 30.1%] n=20</sub> ⚠ | 15.8% <sub>[5.5%, 37.6%] n=19</sub> ⚠ |

### By language

| language | BF16 | FP8 | Q8 | Q4 |
|---|---|---|---|---|
| en | 5.0% <sub>[1.4%, 16.5%] n=40</sub> | 7.5% <sub>[2.6%, 19.9%] n=40</sub> | 5.0% <sub>[1.4%, 16.5%] n=40</sub> | 13.2% <sub>[5.8%, 27.3%] n=38</sub> |
| hi | 5.0% <sub>[1.4%, 16.5%] n=40</sub> | 0.0% <sub>[0.0%, 8.8%] n=40</sub> | 7.5% <sub>[2.6%, 19.9%] n=40</sub> | 10.5% <sub>[4.2%, 24.1%] n=38</sub> |
| hinglish | 0.0% <sub>[0.0%, 8.8%] n=40</sub> | 0.0% <sub>[0.0%, 8.8%] n=40</sub> | 5.0% <sub>[1.4%, 16.5%] n=40</sub> | 10.3% <sub>[4.1%, 23.6%] n=39</sub> |
| mr | 2.5% <sub>[0.4%, 12.9%] n=40</sub> | 0.0% <sub>[0.0%, 8.8%] n=40</sub> | 7.5% <sub>[2.6%, 19.9%] n=40</sub> | 15.0% <sub>[7.1%, 29.1%] n=40</sub> |

---

## 5. PS-3 results — structured output and tool calling

| metric | BF16 | FP8 | Q8 | Q4 |
|---|---|---|---|---|
| `task_success_rate` | 92.5% <sub>[88.0%, 95.4%] n=200</sub> | 93.0% <sub>[88.6%, 95.8%] n=200</sub> | 83.5% <sub>[77.7%, 88.0%] n=200</sub> | 64.1% <sub>[57.3%, 70.5%] n=198</sub> |
| `correct_tool_rate` | 91.9% <sub>[86.1%, 95.4%] n=136</sub> | 91.2% <sub>[85.2%, 94.9%] n=136</sub> | 81.6% <sub>[74.3%, 87.2%] n=136</sub> | 58.2% <sub>[49.7%, 66.2%] n=134</sub> |
| `argument_accuracy` | 99.5% <sub>[97.0%, 99.9%] n=183</sub> | 99.5% <sub>[97.0%, 99.9%] n=185</sub> | 97.0% <sub>[93.3%, 98.7%] n=169</sub> | 90.6% <sub>[83.9%, 94.7%] n=117</sub> |
| `structured_output_validity` | 98.5% <sub>[95.7%, 99.5%] n=200</sub> | 96.5% <sub>[93.0%, 98.3%] n=200</sub> | 95.5% <sub>[91.7%, 97.6%] n=200</sub> | 87.4% <sub>[82.0%, 91.3%] n=198</sub> |
| `malformed_argument_rate` | 1.5% <sub>[0.5%, 4.3%] n=200</sub> | 3.5% <sub>[1.7%, 7.0%] n=200</sub> | 4.5% <sub>[2.4%, 8.3%] n=200</sub> | 12.6% <sub>[8.7%, 18.0%] n=198</sub> |
| `wrong_tool_rate` | 1.5% <sub>[0.5%, 4.3%] n=200</sub> | 0.0% <sub>[0.0%, 1.9%] n=200</sub> | 3.5% <sub>[1.7%, 7.0%] n=200</sub> | 7.1% <sub>[4.3%, 11.5%] n=198</sub> |
| `wrong_argument_rate` | 0.5% <sub>[0.1%, 2.8%] n=200</sub> | 0.5% <sub>[0.1%, 2.8%] n=200</sub> | 2.5% <sub>[1.1%, 5.7%] n=200</sub> | 5.6% <sub>[3.1%, 9.7%] n=198</sub> |
| `spurious_call_rate` | 2.5% <sub>[1.1%, 5.7%] n=200</sub> | 1.0% <sub>[0.3%, 3.6%] n=200</sub> | 3.0% <sub>[1.4%, 6.4%] n=200</sub> | 5.1% <sub>[2.8%, 9.0%] n=198</sub> |
| `missed_call_rate` | 2.2% <sub>[0.8%, 6.3%] n=136</sub> | 2.9% <sub>[1.1%, 7.3%] n=136</sub> | 4.4% <sub>[2.0%, 9.3%] n=136</sub> | 8.2% <sub>[4.6%, 14.1%] n=134</sub> |
| `fallback_extraction_rate` | 0.0% <sub>[0.0%, 1.9%] n=200</sub> | 0.0% <sub>[0.0%, 1.9%] n=200</sub> | 0.0% <sub>[0.0%, 1.9%] n=200</sub> | 0.0% <sub>[0.0%, 1.9%] n=198</sub> |
| `generation_failure_rate` | 0.0% <sub>[0.0%, 1.9%] n=200</sub> | 0.0% <sub>[0.0%, 1.9%] n=200</sub> | 0.0% <sub>[0.0%, 1.9%] n=200</sub> | 1.0% <sub>[0.3%, 3.6%] n=200</sub> |

<sub>Values are point estimates with Wilson 95% intervals and the denominator. ⚠ marks a cell below the pre-registered small-sample threshold; those are directional and no significance is claimed.</sub>

`correct_tool_rate` counts a case as correct when the tool identity is right, **regardless of the argument values**. `task_success_rate` requires the arguments to be right too. The gap between them is the argument-corruption rate, and it is the single most operationally dangerous failure in this suite: a `capture_ptp` for ₹50,000 instead of ₹5,000 is a well-formed, correctly-routed, wrong commitment.

### English vs Indic (task success)

| precision | English | Indic | delta | 95% CI on the delta | significant |
|---|---|---|---|---|---|
| BF16 | 90.0% <sub>[78.6%, 95.7%] n=50</sub> | 93.3% <sub>[88.2%, 96.3%] n=150</sub> | 3.3% | [-4.3%, 15.1%] | no |
| FP8 | 90.0% <sub>[78.6%, 95.7%] n=50</sub> | 94.0% <sub>[89.0%, 96.8%] n=150</sub> | 4.0% | [-3.6%, 15.7%] | no |
| Q8 | 92.0% <sub>[81.2%, 96.8%] n=50</sub> | 80.7% <sub>[73.6%, 86.2%] n=150</sub> | -11.3% | [-19.9%, 0.8%] | no |
| Q4 | 64.0% <sub>[50.1%, 75.9%] n=50</sub> | 64.2% <sub>[56.2%, 71.5%] n=148</sub> | 0.2% | [-14.1%, 15.8%] | no |

<sub>NEGATIVE delta means Indic tool-calling is worse than English.</sub>

### By language

| language | BF16 | FP8 | Q8 | Q4 |
|---|---|---|---|---|
| en | 90.0% <sub>[78.6%, 95.7%] n=50</sub> | 90.0% <sub>[78.6%, 95.7%] n=50</sub> | 92.0% <sub>[81.2%, 96.8%] n=50</sub> | 64.0% <sub>[50.1%, 75.9%] n=50</sub> |
| hi | 88.0% <sub>[76.2%, 94.4%] n=50</sub> | 92.0% <sub>[81.2%, 96.8%] n=50</sub> | 72.0% <sub>[58.3%, 82.5%] n=50</sub> | 64.0% <sub>[50.1%, 75.9%] n=50</sub> |
| hinglish | 100.0% <sub>[92.9%, 100.0%] n=50</sub> | 94.0% <sub>[83.8%, 97.9%] n=50</sub> | 82.0% <sub>[69.2%, 90.2%] n=50</sub> | 63.3% <sub>[49.3%, 75.3%] n=49</sub> |
| mr | 92.0% <sub>[81.2%, 96.8%] n=50</sub> | 96.0% <sub>[86.5%, 98.9%] n=50</sub> | 88.0% <sub>[76.2%, 94.4%] n=50</sub> | 65.3% <sub>[51.3%, 77.1%] n=49</sub> |

<sub>Task success rate by language.</sub>

---

## 6. Quantization degradation

**Guardrail adherence and over-refusal vs precision**

![Guardrail adherence and over-refusal vs precision](figures_mock/01_guardrail_adherence_vs_precision.png)

**Structured-output metrics vs precision**

![Structured-output metrics vs precision](figures_mock/02_structured_output_vs_precision.png)

**PS-3 failure modes by precision**

![PS-3 failure modes by precision](figures_mock/03_ps3_failure_modes.png)

**Per-language breakdown**

![Per-language breakdown](figures_mock/04_language_breakdown.png)

**English vs Indic**

![English vs Indic](figures_mock/05_english_vs_indic.png)

**PS-1 violation rate by category**

![PS-1 violation rate by category](figures_mock/06_ps1_category_heatmap.png)

**Degradation vs the reference precision**

![Degradation vs the reference precision](figures_mock/07_degradation_vs_reference.png)

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

**`violation_rate`** — threshold 2.0%, pattern **`cliff`**, cliff at **Q4**

A cliff was detected for `violation_rate` at **q4**. Degradation crosses the pre-registered 2.0% threshold with a difference interval excluding zero, and the step q8->q4 (+6.0%) dominates the other steps by at least 2.0x.

| precision | value | degradation vs ref | 95% CI on the difference | meets threshold | CI excludes 0 | past cliff |
|---|---|---|---|---|---|---|
| FP8 | 1.9% | -1.2% | [-5.4%, 2.7%] | no | no | no |
| Q8 | 6.2% | 3.1% | [-1.8%, 8.3%] | yes | no | no |
| Q4 | 12.3% | 9.1% | [3.3%, 15.5%] | yes | yes | **YES** |


**`benign_refusal_rate`** — threshold 5.0%, pattern **`none`**

No quantization cliff was detected for `benign_refusal_rate` within the tested precision range, at a threshold of 5.0% and 95% confidence, with n=32 at the reference precision. The minimum difference detectable at this sample size was 25.0%; a real effect smaller than that would not have been visible to this experiment.

| precision | value | degradation vs ref | 95% CI on the difference | meets threshold | CI excludes 0 | past cliff |
|---|---|---|---|---|---|---|
| FP8 | 3.1% | 0.0% | [-12.9%, 12.9%] | no | no | no |
| Q8 | 3.1% | 0.0% | [-12.9%, 12.9%] | no | no | no |
| Q4 | 12.5% | 9.4% | [-5.3%, 25.2%] | yes | no | no |


#### PS3

**`task_success_rate`** — threshold 5.0%, pattern **`cliff`**, cliff at **Q8**

A cliff was detected for `task_success_rate` at **q8**. Degradation crosses the pre-registered 5.0% threshold with a difference interval excluding zero, and the step q8->q4 (+19.4%) dominates the other steps by at least 2.0x.

| precision | value | degradation vs ref | 95% CI on the difference | meets threshold | CI excludes 0 | past cliff |
|---|---|---|---|---|---|---|
| FP8 | 93.0% | -0.5% | [-4.8%, 5.8%] | no | no | no |
| Q8 | 83.5% | 9.0% | [-15.5%, -2.6%] | yes | yes | **YES** |
| Q4 | 64.1% | 28.4% | [-35.8%, -20.6%] | yes | yes | **YES** |


**`structured_output_validity`** — threshold 5.0%, pattern **`cliff`**, cliff at **Q4**

A cliff was detected for `structured_output_validity` at **q4**. Degradation crosses the pre-registered 5.0% threshold with a difference interval excluding zero, and the step q8->q4 (+8.1%) dominates the other steps by at least 2.0x.

| precision | value | degradation vs ref | 95% CI on the difference | meets threshold | CI excludes 0 | past cliff |
|---|---|---|---|---|---|---|
| FP8 | 96.5% | 2.0% | [-5.7%, 1.3%] | no | no | no |
| Q8 | 95.5% | 3.0% | [-7.0%, 0.5%] | no | no | no |
| Q4 | 87.4% | 11.1% | [-16.6%, -6.3%] | yes | yes | **YES** |


---

## 8. Production recommendation

### Minimum viable precision: **FP8**

fp8 is the lowest-fidelity precision that cleared every headline metric in both suites under the pre-registered criterion. This is a statement about THIS suite, THIS model and THIS hardware at THIS sample size. It is not a claim that the precision is universally production-safe.

**Why the lower precisions were rejected:**

- **Q8** failed on:
  - `ps3.task_success_rate` — degradation 9.0% against a threshold of 5.0%
- **Q4** failed on:
  - `ps1.violation_rate` — degradation 9.1% against a threshold of 2.0%
  - `ps3.task_success_rate` — degradation 28.4% against a threshold of 5.0%
  - `ps3.structured_output_validity` — degradation 11.1% against a threshold of 5.0%

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
python -m ps5.run --precision bf16 --backend mock --suite ps1 ps3
python -m ps5.run --precision fp8 --backend mock --suite ps1 ps3
python -m ps5.run --precision q8 --backend mock --suite ps1 ps3
python -m ps5.run --precision q4 --backend mock --suite ps1 ps3

# 4. aggregate, plot, report
python scripts/aggregate_results.py
python scripts/make_plots.py
python scripts/generate_report.py
```

Reproduction is only exact when these match the values in §2: `manifest_hash`, `system_prompt_hash`, `tool_schema_hash`, `generation_config_hash`, `guardrail_rules_hash`, `hardware_fingerprint`. They are recorded in every run's `metadata.json`.

