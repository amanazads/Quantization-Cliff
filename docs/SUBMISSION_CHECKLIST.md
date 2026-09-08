# PS-5 submission checklist

Maps every requirement in Section 7 ("Submission and judging") and the PS-5
problem statement to the artifact that satisfies it, and states plainly what is
still outstanding. Verified against the challenge specification v1.0 (28 Aug 2026).

## Section 7 — what to submit

| # | Requirement | Status | Artifact |
|---|---|---|---|
| 1 | Public Git repository with all code, test suites and results | ⬜ **you must publish it** | this repository; `git init && git push` |
| 2 | README reproducing the run from scratch, with exact model versions, quantization level and hardware | ✅ | `README.md` §4; model tags pinned per arm, weights digest resolved at run time into `metadata.json` |
| 3 | Findings document, **maximum four pages** | ✅ generated | `reports/FINDINGS.md` (concise) + `reports/FINDINGS_FULL.md` (appendix) |
| 4 | Raw results as structured data, not only charts | ✅ | `results/<precision>/ps{1,3}_results.jsonl` — full model output retained per case |
| 5 | Stated limitations section ("this is not a formality; we weight it") | ✅ | `docs/METRICS.md` §7, declared before results; reproduced into the report |

## PS-5 — what to build

| Requirement | Status | Where |
|---|---|---|
| PS-1 and PS-3 re-run at Q4, Q8, FP8, BF16 on identical hardware, all else fixed | ✅ implemented, ⬜ **not yet executed** | `scripts/run_all.sh`; hardware fingerprint hash-verified across arms |
| Degradation curves per metric, cliff **located** not described | ✅ | `src/ps5/cliff.py`, figures 01–07 |
| Guardrail adherence and structured-output validity treated **separately** | ✅ | no composite score exists anywhere; a test asserts this |
| A stated minimum viable precision | ✅ computed | `minimum_viable_precision`, worst-metric-governs |
| Explicit statement of what was held constant and what wasn't | ✅ | report §3, from recorded metadata rather than prose |
| A recommendation with the confidence you actually have in it | ✅ | report §8, separating "supported by this experiment" from "universally safe" |
| Willingness to report a null result | ✅ | null path reports the minimum detectable difference; tested |

## Judging criteria

| Criterion | Weight | How this repository addresses it |
|---|---|---|
| Methodological rigour | 35% | Metrics and cliff thresholds pre-registered in `docs/METRICS.md` / `cliff_criterion.yaml` **before any run**; controls hash-verified, not asserted; the aggregator refuses an invalid comparison |
| Reproducibility | 25% | Deterministic suites and manifest with drift checks; greedy decoding with fixed seed; environment and weights digest captured per run; 239 tests; every report number rendered from raw JSONL |
| Insight | 20% | ⬜ **depends on the actual run** — the framework separates safety from structured output and by language so a differential result can surface |
| Intellectual honesty | 15% | Deviations recorded rather than smoothed; unavailable arms refused rather than substituted; synthetic output blocked from the findings report; scorer-validation gap stated in the report itself |
| Craft | 5% | ✅ |

## Outstanding before submission

1. **Run the arms.** Two paths, and the choice is methodological rather than a convenience:
   - `bash scripts/run_all.sh ollama` — the intended experiment on **Qwen3.5-4B**, the model the specification names: four rungs, genuine bfloat16 reference. Needs **more than 8 GB of RAM** for the 9.3 GB reference arm, or the Track 2 GPU. This is the run that answers PS-5 as asked.
   - `bash scripts/run_all.sh ollama qwen2.5-1.5b` — fits in 8 GB and produces a real three-rung measurement on **Qwen2.5-1.5B**, with an **F16 reference** and **no FP8 arm**. Fully documented and defensible, but a smaller claim: it cannot locate a cliff between FP8 and Q8, and its reference is a converted dtype rather than the training one. Submit it as what it is, never as a 4B result.

   Either way the report states the model, the deviations and the missing arms from recorded metadata rather than from prose.
2. **Validate the scorer against human labels.** `scripts/validation_subset.py export → label → score`. The specification requires it and judges it under "judge quality, measured by agreement with human raters rather than asserted". Currently unmeasured, and the report says so.
3. **Publish the repository** and confirm Apache 2.0 licensing per the challenge terms.
4. **Check the rendered findings document is within four pages** — the generator warns above 200 lines.

## Conduct and data policy

- **AI assistance disclosed** in `README.md` §1, as Section 8 requires where it materially shaped the methodology.
- **Synthetic data only.** No real borrower data anywhere; every case is invented and marked `synthetic: true`.
- **No third-party hosted API.** Every backend is local or self-hosted, so no challenge data leaves the machine. No hosted baseline is declared, because PS-5 compares precisions of one open model rather than models against a baseline.
- **Attacks are for evaluation only**, as Section 8 requires.
