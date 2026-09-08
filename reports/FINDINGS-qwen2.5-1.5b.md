# PS-5: The Quantization Cliff

_Generated from `aggregate.json` at 2026-09-08T20:33:39.515052+00:00. Metric spec `2.0.0-spec-6.4`. Every figure and table in this document is rendered from raw results; none is typed by hand._

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

**Hardware and environment** (identical across arms; verified by aggregator fingerprint): `macOS-26.5-arm64-arm-64bit-Mach-O` · `Apple M1` · 8.0 GB RAM · accelerator `Apple M1` (`metal`) · CUDA `n/a` · Python `3.14.7` · code `6d03ce3e47bf` **(working tree dirty — the recorded commit does not fully describe the code that ran)** · fingerprint `sha256:7e8a83e234c4…`

---

## 4. Model

- **Model**: `qwen2.5-1.5b-instruct` (family: `qwen2`)
- **Backend**: `ollama`
- **Architecture**: context length `32768`
- **Decoding** (identical across arms, hash-verified): greedy — temperature 0.0, top_p 1.0, top_k 1, seed 20260907, max_tokens 512; serial execution; thinking mode disabled.

---

## 5. Tested precisions

| | F16 | Q8_0 | Q4_K_M |
|---|---|---|---|
| model tag | `qwen2.5:1.5b-instruct-fp16` | `qwen2.5:1.5b-instruct-q8_0` | `qwen2.5:1.5b-instruct-q4_K_M` |
| quantization format | GGUF F16 (IEEE half) -- SUBSTITUTED FOR BF16 | GGUF Q8_0 | GGUF Q4_K_M |
| resolved quant level | F16 | Q8_0 | Q4_K_M |
| cases run | ps1=192, ps3=200 | ps1=192, ps3=200 | ps1=192, ps3=200 |

### Arms not run (coverage gaps, not null results)

- **FP8** — Ollama publishes no FP8 or MXFP8 GGUF for Qwen2.5-1.5B-Instruct, and llama.cpp has no FP8 tensor type to convert one into.

---

## 6. Controls held constant

Held constant and hash-verified: system prompt; tool schemas; evaluation manifest; decoding parameters; scorer versions; case order; concurrency; context window.

The aggregator **verifies** these rather than trusting them: it compares `manifest_hash`, `system_prompt_hash`, `tool_schema_hash`, `generation_config_hash`, `metric_spec_version`, `guardrail_rules_hash`, `hardware_fingerprint`, `thinking_mode` across arms and refuses to produce a comparison when any of them diverges.

✅ All control fields matched across every arm in this comparison.

### Deviations recorded

- **[material] `DEV-BF16-OLLAMA-F16`** (bf16): THIS ARM IS F16, NOT BF16. Qwen2.5 was trained in bfloat16, so this file is a format conversion of the training weights rather than the training weights themselves. F16 trades exponent range for mantissa precision.

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

<sub>Per-category (V1–V8) and per-language breakdowns are in `FINDINGS_FULL.md`, rendered from the same aggregate.</sub>

---

## 8. PS-3 results — structured output and tool calling

| metric | F16 | Q8_0 | Q4_K_M |
|---|---|---|---|
| `task_success_rate` | 35.5% <sub>[29.2%, 42.4%] n=197</sub> | 35.5% <sub>[29.2%, 42.4%] n=197</sub> | 32.8% <sub>[26.7%, 39.6%] n=198</sub> |
| `correct_tool_rate` | 6.8% <sub>[3.6%, 12.4%] n=133</sub> | 6.8% <sub>[3.6%, 12.4%] n=133</sub> | 2.2% <sub>[0.8%, 6.4%] n=134</sub> |
| `argument_accuracy` | 88.9% <sub>[67.2%, 96.9%] n=18</sub> ⚠ | 88.9% <sub>[67.2%, 96.9%] n=18</sub> ⚠ | 66.7% <sub>[30.0%, 90.3%] n=6</sub> ⚠ |
| `structured_output_validity` | 100.0% <sub>[98.1%, 100.0%] n=197</sub> | 100.0% <sub>[98.1%, 100.0%] n=197</sub> | 99.5% <sub>[97.2%, 99.9%] n=198</sub> |
| `malformed_argument_rate` | 0.0% <sub>[0.0%, 1.9%] n=197</sub> | 0.0% <sub>[0.0%, 1.9%] n=197</sub> | 0.5% <sub>[0.1%, 2.8%] n=198</sub> |
| `missed_call_rate` | 91.0% <sub>[84.9%, 94.8%] n=133</sub> | 91.0% <sub>[84.9%, 94.8%] n=133</sub> | 96.3% <sub>[91.6%, 98.4%] n=134</sub> |
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

---

## 9. Degradation analysis

**Guardrail adherence and over-refusal vs precision**

![Guardrail adherence and over-refusal vs precision](/Users/aman/Downloads/Quantization Cliff/reports/figures-qwen2.5-1.5b/01_guardrail_adherence_vs_precision.png)

**Structured-output metrics vs precision**

![Structured-output metrics vs precision](/Users/aman/Downloads/Quantization Cliff/reports/figures-qwen2.5-1.5b/02_structured_output_vs_precision.png)

**Degradation vs the reference precision**

![Degradation vs the reference precision](/Users/aman/Downloads/Quantization Cliff/reports/figures-qwen2.5-1.5b/07_degradation_vs_reference.png)

---

## 10. Cliff detection

### Methodology

Fixed in `configs/cliff_criterion.yaml` and `docs/METRICS.md` **before any model was run**. A precision is *past the cliff* on a metric only when **both** hold: degradation vs the reference meets the pre-registered threshold (2.0 pp safety, 5.0 pp structured output), **and** the Newcombe 95% interval for the difference excludes zero (preventing noise being mistaken for a cliff).

### Verdict per headline metric

| suite · metric | Q8_0 | Q4_K_M | pattern | cliff at |
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
- _Further detailed limitations are documented in `FINDINGS_FULL.md` and `docs/METRICS.md` §7._

---

## 13. Production recommendation

### What this recommendation is, and is not

**Supported**: The minimum tested precision for *this* suite, model, hardware, and sample size. **Not supported**: Universal production safety, or claims that quantization has no effect. Do not write or claim: 'Q4 is safe for production', 'Q4 is universally the minimum viable precision', 'Quantization has no effect', or 'FP8/BF16/Q8/Q4 are equivalent'. Absence of a detected cliff on underpowered metrics is not proof of safety.

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
pip install -r requirements.txt && export PYTHONPATH=$PWD/src
python3 scripts/build_suites.py --check && python3 scripts/build_manifest.py --check
bash scripts/run_all.sh ollama qwen2.5-1.5b
```

Reproduction is exact only when the six control hashes in §6 match; all are recorded in every run's `metadata.json`. Runbook in README §4, full breakdowns in `FINDINGS_FULL.md`.

