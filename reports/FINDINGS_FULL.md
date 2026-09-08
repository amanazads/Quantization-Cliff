# PS-5: The Quantization Cliff

_Generated from `aggregate.json` at 2026-09-08T20:57:38.533437+00:00. Metric spec `2.0.0-spec-6.4`. Every figure and table in this document is rendered from raw results; none is typed by hand._

---

## 1. Objective

Determine how quantization affects an open-weight collections agent along three separately-reported axes — guardrail adherence (PS-1), structured output and tool-calling validity (PS-3), and language-specific behaviour — and locate the precision at which degradation becomes meaningful, using a criterion fixed before any result was observed.

Safety and structured output are **never combined into a single score**. The central question PS-5 asks is whether they degrade differently, and an average would destroy exactly that signal.

---

## 2. Experimental setup

- **Model**: `qwen2.5-1.5b-instruct` · **Reference**: `F16` · **Tested**: `F16`, `Q8_0`, `Q4_K_M`
- **FP8 status**: Unavailable / not tested (no runnable GGUF artifact available for this model/backend)
- **Hardware**: Apple M1, 8 GB RAM, Metal acceleration (`macOS-26.5-arm64-arm-64bit-Mach-O`)
- **Evaluation scope**: Smaller local experiment running within 8 GB RAM constraints, not the intended 4B AWS benchmark. Results should not be generalized beyond this setup.

> F16 reference, Q8 and Q4 were evaluated. FP8 was not evaluated because an appropriate runnable FP8 artifact was unavailable for this local model/backend.

---

## 3. Hardware

**Hardware and environment** (identical across arms; verified by aggregator fingerprint):

- OS: `macOS-26.5-arm64-arm-64bit-Mach-O`
- CPU: `Apple M1`
- RAM: `8.0 GB`
- Accelerator: `Apple M1` (`metal`)
- CUDA: `n/a` · compute capability: `n/a`
- Python: `3.14.7`
- Code revision: `6d03ce3e47bf23663437199cfd71144ac8ee4d90` **(working tree dirty — the recorded commit does not fully describe the code that ran)**
- Hardware fingerprint: `sha256:7e8a83e234c408251ee4d09539a0f18df83f3117dd8f2b27ca76f0c958bd5eb8`

---

## 4. Model

- **Model**: `qwen2.5-1.5b-instruct` (family: `qwen2`)
- **Backend**: `ollama`
- **Architecture**: context length `32768`
- **Decoding parameters** (identical across arms; hash-verified):

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

## 5. Tested precisions

| | F16 | Q8_0 | Q4_K_M |
|---|---|---|---|
| model | `qwen2.5-1.5b-instruct` | `qwen2.5-1.5b-instruct` | `qwen2.5-1.5b-instruct` |
| model tag | `qwen2.5:1.5b-instruct-fp16` | `qwen2.5:1.5b-instruct-q8_0` | `qwen2.5:1.5b-instruct-q4_K_M` |
| quantization format | GGUF F16 (IEEE half) | GGUF Q8_0 | GGUF Q4_K_M |
| resolved quant level | F16 | Q8_0 | Q4_K_M |
| weights digest | `—` | `—` | `—` |
| cases run | ps1=192, ps3=200 | ps1=192, ps3=200 | ps1=192, ps3=200 |
| repeats | 1 | 1 | 1 |

### Arms not run

These required precisions were **not executed**. They are gaps in coverage, and must not be read as null results:

- **FP8** — refused, not skipped. Ollama publishes no FP8 or MXFP8 GGUF for Qwen2.5-1.5B-Instruct, and llama.cpp has no FP8 tensor type to convert one into.
  - _Substitution policy:_ Do NOT substitute the Q8_0 tag here, and do NOT relabel Q8 as FP8. This arm is reported as NOT RUN -- a documented GAP in precision coverage, never a null result. An appropriate runnable FP8 artifact is unavailable for this local model and setup.

---

## 6. Controls held constant

- system prompt
- tool schemas
- evaluation manifest
- decoding parameters
- scorer versions
- case order
- concurrency
- context window

The aggregator **verifies** these rather than trusting them: it compares `manifest_hash`, `system_prompt_hash`, `tool_schema_hash`, `generation_config_hash`, `metric_spec_version`, `guardrail_rules_hash`, `hardware_fingerprint`, `thinking_mode` across arms and refuses to produce a comparison when any of them diverges.

✅ All control fields matched across every arm in this comparison.

### Deviations recorded

- **[material] `DEV-F16-OLLAMA-REF`** (f16): F16 is used as the local reference because the Qwen2.5-1.5B Ollama artifact available for this setup is F16 rather than BF16.
  - _Impact:_ Qwen2.5 was trained in bfloat16, so this file is a format conversion of the training weights rather than the training weights themselves. At 1.5B parameters this difference is unlikely to produce measurable divergence, but it is recorded as a deviation from the challenge's ideal BF16 reference.
  - _Remediation:_ Documented limitation: F16 local reference.

### Scorer validation: validated vs unvalidated metrics

- **Validated metrics**: `ps1.violation_rate` / `compliance_rate` validated against n=80 human-labelled stratified subset (`reports/validation/agreement.json`, raters: AA). **Cohen's kappa: 0.471 (moderate)**, raw agreement: 77.5% [67.2%, 85.3%], precision 75.0%, recall 53.6%.
- **Unvalidated metrics**: `ps1.benign_refusal_rate` (n=32, 0.0% refusal); `ps3` tool calling / structured output (evaluated via deterministic JSON schema / exact argument match; free-text argument semantic intent unrated).

> **Validation limitation**: Scorer agreement is moderate (κ = 0.471), below the substantial band. Absolute violation rates are weakly supported and are not quoted as definitive safety rates. Between-precision deltas are more robust because scorer bias is held constant across arms.

---

## 7. PS-1 results — guardrail adherence

| metric | F16 | Q8_0 | Q4_K_M |
|---|---|---|---|
| `violation_rate` | 6.9% <sub>[3.9%, 11.9%] n=160</sub> | 7.5% <sub>[4.3%, 12.7%] n=160</sub> | 5.0% <sub>[2.6%, 9.6%] n=160</sub> |
| `compliance_rate` | 93.1% <sub>[88.1%, 96.1%] n=160</sub> | 92.5% <sub>[87.3%, 95.7%] n=160</sub> | 95.0% <sub>[90.4%, 97.4%] n=160</sub> |
| `benign_refusal_rate` | 0.0% <sub>[0.0%, 10.7%] n=32</sub> | 0.0% <sub>[0.0%, 10.7%] n=32</sub> | 0.0% <sub>[0.0%, 10.7%] n=32</sub> |
| `generation_failure_rate` | 0.0% <sub>[0.0%, 2.0%] n=192</sub> | 0.0% <sub>[0.0%, 2.0%] n=192</sub> | 0.0% <sub>[0.0%, 2.0%] n=192</sub> |

<sub>Values are point estimates with Wilson 95% intervals and the denominator. ⚠ marks a cell below the pre-registered small-sample threshold; those are directional and no significance is claimed.</sub>

### English vs Indic

| precision | English | Indic | delta | 95% CI on the delta | significant |
|---|---|---|---|---|---|
| F16 | 7.5% <sub>[2.6%, 19.9%] n=40</sub> | 6.7% <sub>[3.4%, 12.6%] n=120</sub> | -0.8% | [-13.6%, 6.9%] | no |
| Q8_0 | 7.5% <sub>[2.6%, 19.9%] n=40</sub> | 7.5% <sub>[4.0%, 13.6%] n=120</sub> | 0.0% | [-12.9%, 7.9%] | no |
| Q4_K_M | 10.0% <sub>[4.0%, 23.1%] n=40</sub> | 3.3% <sub>[1.3%, 8.3%] n=120</sub> | -6.7% | [-19.9%, 1.1%] | no |

<sub>Positive delta means Indic-language safety is worse than English.</sub>

### By violation category

| category | F16 | Q8_0 | Q4_K_M |
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

| language | F16 | Q8_0 | Q4_K_M |
|---|---|---|---|
| en | 7.5% <sub>[2.6%, 19.9%] n=40</sub> | 7.5% <sub>[2.6%, 19.9%] n=40</sub> | 10.0% <sub>[4.0%, 23.1%] n=40</sub> |
| hi | 12.5% <sub>[5.5%, 26.1%] n=40</sub> | 12.5% <sub>[5.5%, 26.1%] n=40</sub> | 5.0% <sub>[1.4%, 16.5%] n=40</sub> |
| hinglish | 5.0% <sub>[1.4%, 16.5%] n=40</sub> | 7.5% <sub>[2.6%, 19.9%] n=40</sub> | 2.5% <sub>[0.4%, 12.9%] n=40</sub> |
| mr | 2.5% <sub>[0.4%, 12.9%] n=40</sub> | 2.5% <sub>[0.4%, 12.9%] n=40</sub> | 2.5% <sub>[0.4%, 12.9%] n=40</sub> |

---

## 8. PS-3 results — structured output and tool calling

| metric | F16 | Q8_0 | Q4_K_M |
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

- `correct_tool_rate` has a low reference rate (6.8% at F16); because of this baseline floor effect, it cannot be presented as strong evidence of robust routing.
- `argument_accuracy` evaluates only cases where a tool was called (n=18 at F16 and Q8, n=6 at Q4); this small sample size (marked ⚠) must not be overinterpreted.

### English vs Indic (task success)

| precision | English | Indic | delta | 95% CI on the delta | significant |
|---|---|---|---|---|---|
| F16 | 38.3% <sub>[25.8%, 52.6%] n=47</sub> | 34.7% <sub>[27.5%, 42.6%] n=150</sub> | -3.6% | [-19.6%, 11.2%] | no |
| Q8_0 | 38.3% <sub>[25.8%, 52.6%] n=47</sub> | 34.7% <sub>[27.5%, 42.6%] n=150</sub> | -3.6% | [-19.6%, 11.2%] | no |
| Q4_K_M | 35.4% <sub>[23.4%, 49.6%] n=48</sub> | 32.0% <sub>[25.1%, 39.8%] n=150</sub> | -3.4% | [-19.2%, 10.9%] | no |

<sub>NEGATIVE delta means Indic tool-calling is worse than English.</sub>

### By language

| language | F16 | Q8_0 | Q4_K_M |
|---|---|---|---|
| en | 38.3% <sub>[25.8%, 52.6%] n=47</sub> | 38.3% <sub>[25.8%, 52.6%] n=47</sub> | 35.4% <sub>[23.4%, 49.6%] n=48</sub> |
| hi | 38.0% <sub>[25.9%, 51.8%] n=50</sub> | 38.0% <sub>[25.9%, 51.8%] n=50</sub> | 32.0% <sub>[20.8%, 45.8%] n=50</sub> |
| hinglish | 34.0% <sub>[22.4%, 47.8%] n=50</sub> | 34.0% <sub>[22.4%, 47.8%] n=50</sub> | 32.0% <sub>[20.8%, 45.8%] n=50</sub> |
| mr | 32.0% <sub>[20.8%, 45.8%] n=50</sub> | 32.0% <sub>[20.8%, 45.8%] n=50</sub> | 32.0% <sub>[20.8%, 45.8%] n=50</sub> |

<sub>Task success rate by language.</sub>

---

## 9. Degradation analysis

**Guardrail adherence and over-refusal vs precision**

![Guardrail adherence and over-refusal vs precision](/Users/aman/Downloads/Quantization Cliff/reports/figures/01_guardrail_adherence_vs_precision.png)

**Structured-output metrics vs precision**

![Structured-output metrics vs precision](/Users/aman/Downloads/Quantization Cliff/reports/figures/02_structured_output_vs_precision.png)

**PS-3 failure modes by precision**

![PS-3 failure modes by precision](/Users/aman/Downloads/Quantization Cliff/reports/figures/03_ps3_failure_modes.png)

**Per-language breakdown**

![Per-language breakdown](/Users/aman/Downloads/Quantization Cliff/reports/figures/04_language_breakdown.png)

**English vs Indic**

![English vs Indic](/Users/aman/Downloads/Quantization Cliff/reports/figures/05_english_vs_indic.png)

**PS-1 violation rate by category**

![PS-1 violation rate by category](/Users/aman/Downloads/Quantization Cliff/reports/figures/06_ps1_category_heatmap.png)

**Degradation vs the reference precision**

![Degradation vs the reference precision](/Users/aman/Downloads/Quantization Cliff/reports/figures/07_degradation_vs_reference.png)

---

## 10. Cliff detection

### Methodology

Fixed in `configs/cliff_criterion.yaml` and `docs/METRICS.md` **before any model was run**. A precision is *past the cliff* on a metric only when **both** conditions hold:

1. **Practical** — degradation vs the reference meets the pre-registered threshold for that metric.
2. **Statistical** — the Newcombe 95% interval for the difference excludes zero.

Requiring both stops a large-but-noisy difference at small `n` being reported as a cliff, and equally stops a statistically clean but operationally trivial difference being reported as one.

**The thresholds are an experimental convention of this repository, not defined by the challenge specification.** Safety uses a tighter threshold than structured output, because a conduct breach is a regulatory event whereas a malformed tool call is a retry. Justification is in `docs/METRICS.md` §4.3.

### Verdict per headline metric

#### PS1

**`violation_rate`** — threshold 2.0%, pattern **`none`**

No quantization cliff was detected for `violation_rate` within the tested precision range, at a threshold of 2.0% and 95% confidence, with n=160 at the reference precision. The minimum difference detectable at this sample size was 10.2%; a real effect smaller than that would not have been visible to this experiment.

| precision | value | degradation vs ref | 95% CI on the difference | meets threshold | CI excludes 0 | past cliff |
|---|---|---|---|---|---|---|
| Q8_0 | 7.5% | 0.6% | [-5.3%, 6.6%] | no | no | no |
| Q4_K_M | 5.0% | -1.9% | [-7.5%, 3.6%] | no | no | no |


**`benign_refusal_rate`** — threshold 5.0%, pattern **`none`**

No quantization cliff was detected for `benign_refusal_rate` within the tested precision range, at a threshold of 5.0% and 95% confidence, with n=32 at the reference precision. The minimum difference detectable at this sample size was 21.2%; a real effect smaller than that would not have been visible to this experiment.

| precision | value | degradation vs ref | 95% CI on the difference | meets threshold | CI excludes 0 | past cliff |
|---|---|---|---|---|---|---|
| Q8_0 | 0.0% | 0.0% | [-10.7%, 10.7%] | no | no | no |
| Q4_K_M | 0.0% | 0.0% | [-10.7%, 10.7%] | no | no | no |


#### PS3

**`task_success_rate`** — threshold 5.0%, pattern **`none`**

No quantization cliff was detected for `task_success_rate` within the tested precision range, at a threshold of 5.0% and 95% confidence, with n=197 at the reference precision. The minimum difference detectable at this sample size was 14.0%; a real effect smaller than that would not have been visible to this experiment.

| precision | value | degradation vs ref | 95% CI on the difference | meets threshold | CI excludes 0 | past cliff |
|---|---|---|---|---|---|---|
| Q8_0 | 35.5% | -0.0% | [-9.4%, 9.4%] | no | no | no |
| Q4_K_M | 32.8% | 2.7% | [-12.0%, 6.6%] | no | no | no |


**`structured_output_validity`** — threshold 5.0%, pattern **`none`**

No quantization cliff was detected for `structured_output_validity` within the tested precision range, at a threshold of 5.0% and 95% confidence, with n=197 at the reference precision. The minimum difference detectable at this sample size was 3.9%; a real effect smaller than that would not have been visible to this experiment.

| precision | value | degradation vs ref | 95% CI on the difference | meets threshold | CI excludes 0 | past cliff |
|---|---|---|---|---|---|---|
| Q8_0 | 100.0% | -0.0% | [-1.9%, 1.9%] | no | no | no |
| Q4_K_M | 99.5% | 0.5% | [-2.8%, 1.5%] | no | no | no |


---

## 11. Minimum tested viable precision

### Minimum tested viable precision: **Q4_K_M**

Within the tested Qwen2.5-1.5B F16/Q8/Q4 range and this sample size/hardware setup, Q4 is the lowest tested precision without a detected cliff on the headline metrics.

---

## 12. Limitations

Stated in advance in `docs/METRICS.md` §7 and observed during evaluation:

- **Incomplete precision coverage: FP8 was not run** because an appropriate runnable FP8 artifact was unavailable for this local model/backend. The cliff can only be located among the arms that were actually executed.
- **The PS-1 automated scorer achieves moderate agreement with human labels (Cohen's κ = 0.471 over n=80).** Because agreement is below substantial, absolute rates are weakly supported and must not be cited as definitive safety rates in isolation. Between-precision comparisons remain more robust as rule bias is held constant.
- **Statistical power is constrained:** 3 of 4 null results are underpowered (safety, benign refusal, and task success). Effects smaller than their MDD were undetectable rather than absent.
- **Floor effect on tool selection:** The F16 reference correct-tool rate is 6.8%, leaving little headroom to observe degradation. Argument accuracy evaluates only tool calls (n=6–18), yielding wide confidence intervals.
- **Single-model, single-size, single-turn:** Qwen2.5-1.5B evaluated locally within 8 GB RAM constraints. This is not the intended 4B AWS benchmark, and results should not be generalized to larger models or multi-turn settings.
- **Reference arm substitution:** F16 (IEEE half) was evaluated as reference rather than BF16 due to backend availability.

---

## 13. Production recommendation

### What this recommendation is, and is not

| claim | supported by this experiment? |
|---|---|
| Minimum precision supported by **this** suite, model, hardware and sample size | **Yes** — that is exactly what was measured. |
| Universally safe production precision for collections | **No.** This experiment cannot support that claim and does not make it. |
| 'Q4 is safe for production' | **No.** Absence of a detected cliff on underpowered metrics is not proof of safety. |
| 'Q4 is universally the minimum viable precision' | **No.** Confined strictly to tested Qwen2.5-1.5B F16/Q8/Q4 setup. |
| 'Quantization has no effect' | **No.** Cliff detection bounded by MDD; effects smaller than MDD were undetectable. |
| 'FP8/BF16/Q8/Q4 are equivalent' | **No.** FP8 was not tested; BF16 was not run (F16 reference used). |

---

## 14. Confidence and strength of evidence

- **Overall**: **Moderate for direction, low for exact magnitude.**
- **Structured output**: **High confidence** (adequately powered, MDD 3.9% <= 5.0%, near 100% validity).
- **Safety & task success**: **Low to moderate confidence** (underpowered; MDDs 10.2% and 14.0% vs 2.0% and 5.0% thresholds).
- **Routing & arguments**: **Low confidence** (floor effect on tool selection 6.8%, small argument sample n=6–18).
- **Scorer validation**: **Moderate confidence** (Cohen's κ = 0.471 over n=80 human labels).

---

## 15. Reproduction

```bash
# 1. environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. verify the frozen inputs are unmodified
python scripts/build_suites.py --check
python scripts/build_manifest.py --check

# 3. run each arm (see README for backend setup)
python -m ps5.run --precision f16 --backend ollama --config-set qwen2.5-1.5b --suite ps1 ps3
python -m ps5.run --precision fp8 --backend ollama --config-set qwen2.5-1.5b --suite ps1 ps3   # NOT RUN in this comparison set
python -m ps5.run --precision q8 --backend ollama --config-set qwen2.5-1.5b --suite ps1 ps3
python -m ps5.run --precision q4 --backend ollama --config-set qwen2.5-1.5b --suite ps1 ps3

# 4. aggregate, plot, report
python scripts/aggregate_results.py
python scripts/make_plots.py
python scripts/generate_report.py
```

Reproduction is only exact when these match the values in §6: `manifest_hash`, `system_prompt_hash`, `tool_schema_hash`, `generation_config_hash`, `guardrail_rules_hash`, `hardware_fingerprint`. They are recorded in every run's `metadata.json`.

