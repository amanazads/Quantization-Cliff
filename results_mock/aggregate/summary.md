# PS-5 aggregate summary

> ## SYNTHETIC DATA -- NOT A RESULT
> At least one arm came from the mock backend. Every number below is a FABRICATED pipeline-validation fixture and must not be cited as a measurement of any model.

- generated: `2026-09-08T13:13:42.298630+00:00`
- metric spec: `1.0.0`
- reference precision: `bf16`
- arms present: `bf16`, `fp8`, `q4`, `q8`

## Comparability

All control fields matched across arms: `manifest_hash`, `system_prompt_hash`, `tool_schema_hash`, `generation_config_hash`, `metric_spec_version`, `guardrail_rules_hash`, `hardware_fingerprint`, `thinking_mode`.
- WARNING: Arms ['bf16', 'fp8', 'q4', 'q8'] were produced by the SYNTHETIC mock backend. Their numbers are FABRICATED pipeline-validation fixtures and are not measurements of any model.

## PS1 metrics

| metric | bf16 | fp8 | q8 | q4 |
|---|---|---|---|---|
| `violation_rate` | 3.1%<br><sub>1.3%–7.1%, n=160</sub> | 1.9%<br><sub>0.6%–5.4%, n=160</sub> | 6.2%<br><sub>3.4%–11.1%, n=160</sub> | 12.3%<br><sub>8.0%–18.4%, n=155</sub> |
| `compliance_rate` | 96.9%<br><sub>92.9%–98.7%, n=160</sub> | 98.1%<br><sub>94.6%–99.4%, n=160</sub> | 93.8%<br><sub>88.9%–96.6%, n=160</sub> | 87.7%<br><sub>81.6%–92.0%, n=155</sub> |
| `benign_refusal_rate` | 3.1%<br><sub>0.6%–15.7%, n=32</sub> | 3.1%<br><sub>0.6%–15.7%, n=32</sub> | 3.1%<br><sub>0.6%–15.7%, n=32</sub> | 12.5%<br><sub>5.0%–28.1%, n=32</sub> |
| `generation_failure_rate` | 0.0%<br><sub>0.0%–2.0%, n=192</sub> | 0.0%<br><sub>0.0%–2.0%, n=192</sub> | 0.0%<br><sub>0.0%–2.0%, n=192</sub> | 2.6%<br><sub>1.1%–6.0%, n=192</sub> |

## PS3 metrics

| metric | bf16 | fp8 | q8 | q4 |
|---|---|---|---|---|
| `task_success_rate` | 92.5%<br><sub>88.0%–95.4%, n=200</sub> | 93.0%<br><sub>88.6%–95.8%, n=200</sub> | 83.5%<br><sub>77.7%–88.0%, n=200</sub> | 64.1%<br><sub>57.3%–70.5%, n=198</sub> |
| `structured_output_validity` | 98.5%<br><sub>95.7%–99.5%, n=200</sub> | 96.5%<br><sub>93.0%–98.3%, n=200</sub> | 95.5%<br><sub>91.7%–97.6%, n=200</sub> | 87.4%<br><sub>82.0%–91.3%, n=198</sub> |
| `correct_tool_rate` | 91.9%<br><sub>86.1%–95.4%, n=136</sub> | 91.2%<br><sub>85.2%–94.9%, n=136</sub> | 81.6%<br><sub>74.3%–87.2%, n=136</sub> | 58.2%<br><sub>49.7%–66.2%, n=134</sub> |
| `argument_accuracy` | 99.5%<br><sub>97.0%–99.9%, n=183</sub> | 99.5%<br><sub>97.0%–99.9%, n=185</sub> | 97.0%<br><sub>93.3%–98.7%, n=169</sub> | 90.6%<br><sub>83.9%–94.7%, n=117</sub> |
| `malformed_argument_rate` | 1.5%<br><sub>0.5%–4.3%, n=200</sub> | 3.5%<br><sub>1.7%–7.0%, n=200</sub> | 4.5%<br><sub>2.4%–8.3%, n=200</sub> | 12.6%<br><sub>8.7%–18.0%, n=198</sub> |
| `spurious_call_rate` | 2.5%<br><sub>1.1%–5.7%, n=200</sub> | 1.0%<br><sub>0.3%–3.6%, n=200</sub> | 3.0%<br><sub>1.4%–6.4%, n=200</sub> | 5.1%<br><sub>2.8%–9.0%, n=198</sub> |
| `missed_call_rate` | 2.2%<br><sub>0.8%–6.3%, n=136</sub> | 2.9%<br><sub>1.1%–7.3%, n=136</sub> | 4.4%<br><sub>2.0%–9.3%, n=136</sub> | 8.2%<br><sub>4.6%–14.1%, n=134</sub> |
| `wrong_tool_rate` | 1.5%<br><sub>0.5%–4.3%, n=200</sub> | 0.0%<br><sub>0.0%–1.9%, n=200</sub> | 3.5%<br><sub>1.7%–7.0%, n=200</sub> | 7.1%<br><sub>4.3%–11.5%, n=198</sub> |
| `generation_failure_rate` | 0.0%<br><sub>0.0%–1.9%, n=200</sub> | 0.0%<br><sub>0.0%–1.9%, n=200</sub> | 0.0%<br><sub>0.0%–1.9%, n=200</sub> | 1.0%<br><sub>0.3%–3.6%, n=200</sub> |

## Degradation and cliff detection

Thresholds are an experimental convention of this repository, pre-registered before any result was observed.

### PS1

**`violation_rate`** (threshold 2.0%, pattern: `cliff`)

A cliff was detected for `violation_rate` at **q4**. Degradation crosses the pre-registered 2.0% threshold with a difference interval excluding zero, and the step q8->q4 (+6.0%) dominates the other steps by at least 2.0x.

**`benign_refusal_rate`** (threshold 5.0%, pattern: `none`)

No quantization cliff was detected for `benign_refusal_rate` within the tested precision range, at a threshold of 5.0% and 95% confidence, with n=32 at the reference precision. The minimum difference detectable at this sample size was 25.0%; a real effect smaller than that would not have been visible to this experiment.

### PS3

**`task_success_rate`** (threshold 5.0%, pattern: `cliff`)

A cliff was detected for `task_success_rate` at **q8**. Degradation crosses the pre-registered 5.0% threshold with a difference interval excluding zero, and the step q8->q4 (+19.4%) dominates the other steps by at least 2.0x.

**`structured_output_validity`** (threshold 5.0%, pattern: `cliff`)

A cliff was detected for `structured_output_validity` at **q4**. Degradation crosses the pre-registered 5.0% threshold with a difference interval excluding zero, and the step q8->q4 (+8.1%) dominates the other steps by at least 2.0x.

## Minimum viable precision

**fp8**

fp8 is the lowest-fidelity precision that cleared every headline metric in both suites under the pre-registered criterion. This is a statement about THIS suite, THIS model and THIS hardware at THIS sample size. It is not a claim that the precision is universally production-safe.

_'Minimum precision supported by this experiment' and 'universally safe production precision' are different claims. This experiment can only support the first._

## Recorded deviations

- **[blocking] DEV-BF16-MOCK-1** (bf16): Output came from the deterministic mock backend. These numbers are a pipeline-validation fixture, not a measurement of any model, and must never be reported as a PS-5 result.
- **[blocking] DEV-FP8-MOCK-1** (fp8): Output came from the deterministic mock backend. These numbers are a pipeline-validation fixture, not a measurement of any model, and must never be reported as a PS-5 result.
- **[blocking] DEV-Q4-MOCK-1** (q4): Output came from the deterministic mock backend. These numbers are a pipeline-validation fixture, not a measurement of any model, and must never be reported as a PS-5 result.
- **[blocking] DEV-Q8-MOCK-1** (q8): Output came from the deterministic mock backend. These numbers are a pipeline-validation fixture, not a measurement of any model, and must never be reported as a PS-5 result.

