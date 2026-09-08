# PS-5 aggregate summary

- generated: `2026-09-08T19:06:24.891040+00:00`
- metric spec: `2.0.0-spec-6.4` (recorded by every arm and verified equal across them)
- cliff criterion: `1.0.0`
- reference precision: `bf16`
- arms present: `bf16`, `q4`, `q8`
- **arms NOT run: `fp8`** (see Limitations -- these are gaps, not null results)

## Comparability

All control fields matched across arms: `manifest_hash`, `system_prompt_hash`, `tool_schema_hash`, `generation_config_hash`, `metric_spec_version`, `guardrail_rules_hash`, `hardware_fingerprint`, `thinking_mode`.

## PS1 metrics

| metric | bf16 | q8 | q4 |
|---|---|---|---|
| `violation_rate` | 6.9%<br><sub>3.9%–11.9%, n=160</sub> | 7.5%<br><sub>4.3%–12.7%, n=160</sub> | 5.0%<br><sub>2.6%–9.6%, n=160</sub> |
| `compliance_rate` | 93.1%<br><sub>88.1%–96.1%, n=160</sub> | 92.5%<br><sub>87.3%–95.7%, n=160</sub> | 95.0%<br><sub>90.4%–97.4%, n=160</sub> |
| `benign_refusal_rate` | 0.0%<br><sub>0.0%–10.7%, n=32</sub> | 0.0%<br><sub>0.0%–10.7%, n=32</sub> | 0.0%<br><sub>0.0%–10.7%, n=32</sub> |
| `generation_failure_rate` | 0.0%<br><sub>0.0%–2.0%, n=192</sub> | 0.0%<br><sub>0.0%–2.0%, n=192</sub> | 0.0%<br><sub>0.0%–2.0%, n=192</sub> |

## PS3 metrics

| metric | bf16 | q8 | q4 |
|---|---|---|---|
| `task_success_rate` | 35.5%<br><sub>29.2%–42.4%, n=197</sub> | 35.5%<br><sub>29.2%–42.4%, n=197</sub> | 32.8%<br><sub>26.7%–39.6%, n=198</sub> |
| `structured_output_validity` | 100.0%<br><sub>98.1%–100.0%, n=197</sub> | 100.0%<br><sub>98.1%–100.0%, n=197</sub> | 99.5%<br><sub>97.2%–99.9%, n=198</sub> |
| `correct_tool_rate` | 6.8%<br><sub>3.6%–12.4%, n=133</sub> | 6.8%<br><sub>3.6%–12.4%, n=133</sub> | 2.2%<br><sub>0.8%–6.4%, n=134</sub> |
| `argument_accuracy` | 88.9%<br><sub>67.2%–96.9%, n=18</sub><br><sub>⚠ small n</sub> | 88.9%<br><sub>67.2%–96.9%, n=18</sub><br><sub>⚠ small n</sub> | 66.7%<br><sub>30.0%–90.3%, n=6</sub><br><sub>⚠ small n</sub> |
| `malformed_argument_rate` | 0.0%<br><sub>0.0%–1.9%, n=197</sub> | 0.0%<br><sub>0.0%–1.9%, n=197</sub> | 0.5%<br><sub>0.1%–2.8%, n=198</sub> |
| `spurious_call_rate` | 0.5%<br><sub>0.1%–2.8%, n=197</sub> | 0.5%<br><sub>0.1%–2.8%, n=197</sub> | 0.0%<br><sub>0.0%–1.9%, n=198</sub> |
| `missed_call_rate` | 91.0%<br><sub>84.9%–94.8%, n=133</sub> | 91.0%<br><sub>84.9%–94.8%, n=133</sub> | 96.3%<br><sub>91.6%–98.4%, n=134</sub> |
| `wrong_tool_rate` | 1.5%<br><sub>0.5%–4.4%, n=197</sub> | 1.5%<br><sub>0.5%–4.4%, n=197</sub> | 0.5%<br><sub>0.1%–2.8%, n=198</sub> |
| `generation_failure_rate` | 1.5%<br><sub>0.5%–4.3%, n=200</sub> | 1.5%<br><sub>0.5%–4.3%, n=200</sub> | 1.0%<br><sub>0.3%–3.6%, n=200</sub> |

## Degradation and cliff detection

Thresholds are an experimental convention of this repository, pre-registered before any result was observed.

### PS1

**`violation_rate`** (threshold 2.0%, pattern: `none`)

No quantization cliff was detected for `violation_rate` within the tested precision range, at a threshold of 2.0% and 95% confidence, with n=160 at the reference precision. The minimum difference detectable at this sample size was 10.2%; a real effect smaller than that would not have been visible to this experiment.

**`benign_refusal_rate`** (threshold 5.0%, pattern: `none`)

No quantization cliff was detected for `benign_refusal_rate` within the tested precision range, at a threshold of 5.0% and 95% confidence, with n=32 at the reference precision. The minimum difference detectable at this sample size was 21.2%; a real effect smaller than that would not have been visible to this experiment.

### PS3

**`task_success_rate`** (threshold 5.0%, pattern: `none`)

No quantization cliff was detected for `task_success_rate` within the tested precision range, at a threshold of 5.0% and 95% confidence, with n=197 at the reference precision. The minimum difference detectable at this sample size was 14.0%; a real effect smaller than that would not have been visible to this experiment.

**`structured_output_validity`** (threshold 5.0%, pattern: `none`)

No quantization cliff was detected for `structured_output_validity` within the tested precision range, at a threshold of 5.0% and 95% confidence, with n=197 at the reference precision. The minimum difference detectable at this sample size was 3.9%; a real effect smaller than that would not have been visible to this experiment.

## Minimum viable precision

**q4**

q4 is the lowest-fidelity precision that cleared every headline metric in both suites under the pre-registered criterion. This is a statement about THIS suite, THIS model and THIS hardware at THIS sample size. It is not a claim that the precision is universally production-safe.

_'Minimum precision supported by this experiment' and 'universally safe production precision' are different claims. This experiment can only support the first._

## Recorded deviations

- **[material] DEV-BF16-OLLAMA-F16** (bf16): THIS ARM IS F16, NOT BF16. Qwen2.5 was trained in bfloat16, so this file is a format conversion of the training weights rather than the training weights themselves. F16 trades exponent range for mantissa precision.

