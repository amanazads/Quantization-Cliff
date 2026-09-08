# PS-5: The Quantization Cliff — Findings

> ## ⏳ PLACEHOLDER — THE Qwen3.5-4B EXPERIMENT HAS NOT BEEN RUN
>
> This file is a scaffold containing **no results**. The `default` config set
> needs a 9.3 GB BF16 reference arm and has not been executed.
>
> **A different, completed experiment lives in
> [`FINDINGS-qwen2.5-1.5b.md`](FINDINGS-qwen2.5-1.5b.md)** — Qwen2.5-1.5B at
> F16/Q8/Q4, three rungs, real numbers. It answers a smaller question than the
> one below; read its §3 and README §6 before quoting anything from it.
>
> This file is regenerated in full by:
>
> ```bash
> bash scripts/run_all.sh ollama      # or: bash scripts/run_all.sh vllm
> ```
>
> which renders it from `results/aggregate/aggregate.json` together with a
> companion `FINDINGS_FULL.md` appendix. **Do not type numbers into this file** —
> it is overwritten, and a hand-edited figure that survives one regeneration and
> not the next is how a report starts disagreeing with its own data.
>
> A fully populated example built from the deterministic mock backend is in
> `reports/FINDINGS_MOCK.md`. Every number there is fabricated and watermarked;
> it demonstrates the pipeline, not the model. `run_all.sh mock` writes to
> `results_mock/` and the `_MOCK` report paths, cannot overwrite this file, and
> refuses to start at all if asked to write into a results root that already
> holds measured arms.

---

## 1. Objective

Locate the precision at which quantization actually breaks instruction-following and structured output, using the PS-1 and PS-3 suites as the measuring instrument, with the cliff *located* rather than described.

Guardrail adherence and structured-output validity are reported **separately and never averaged**. Whether they degrade at different rates is the question PS-5 asks.

## 2. Experimental setup

- **Model:** `qwen3.5-4b` — the specification's "turn-loop candidate, smallest viable". Single family and size across every arm.
- **Precisions:** Q4 (`q4_K_M`), Q8 (`q8_0`), FP8 (`mxfp8`), BF16 — all genuine formats, no substitutions.
- **Serving stack:** _[pending — Ollama or vLLM; recorded automatically]_
- **Hardware:** _[pending — captured automatically into `metadata.json`]_
- **Decoding:** greedy (`temperature=0.0`, `top_p=1.0`, `top_k=1`), `seed=20260907`, `max_tokens=512`, serial, **thinking mode disabled** as the specification requires.
- **Suites:** PS-1 192 cases (160 adversarial + 32 benign controls) · PS-3 200 cases · identical across arms, pinned by `evaluation_manifest.json`.

## 3. Experimental controls

Held constant and **hash-verified** across arms: baseline prompt template, tool schemas, evaluation manifest and case order, decoding parameters, scorer versions, guardrail rules, concurrency, context window, hardware fingerprint. `scripts/aggregate_results.py` refuses to produce a comparison if any diverges.

**Deviations that cannot be eliminated:**

- `DEV-FP8-OLLAMA-MXFP8` (**material**) — the Ollama FP8 arm is MXFP8 (block-scaled), not per-tensor E4M3. Genuinely 8-bit float, different scaling granularity; not poolable with a vLLM FP8 arm.
- `DEV-BF16-OLLAMA-METAL` (**minor**) — Metal's BF16 support is partial and may upcast some operations. Verify the compute path in the server log.

**Verification result:** _[pending]_

## 3b. Scorer validation against human labels

**Status: _[pending]_.** The specification requires the scorer to be validated against human labels with the agreement reported, and judges it. Until `scripts/validation_subset.py score` has been run, every absolute violation rate below rests on an unvalidated scorer and is provisional.

## 4. PS-1 results — guardrail adherence

_[pending]_ — violation rate, benign over-refusal, per-category (V1–V8), per-language, and the English-vs-Indic delta with a Newcombe interval.

## 5. PS-3 results — structured output

_[pending]_ — task success, correct-tool rate, argument accuracy, structured-output validity, and the malformed / wrong-tool / wrong-argument / spurious / missed breakdown. The English-vs-Hinglish delta is the specification's headline number.

## 6. Quantization degradation

_[pending]_ — figures rendered into `reports/figures/`.

## 7. The quantization cliff

Criterion fixed in `configs/cliff_criterion.yaml` **before any run**: past the cliff requires degradation ≥ the pre-registered threshold (2.0 pp safety, 5.0 pp structured output) **and** a Newcombe 95% interval on the difference that excludes zero.

**Verdict:** _[pending]_

A null result is valid and will be reported as one, with the minimum detectable difference at the achieved sample size.

## 8. Production recommendation

**Minimum viable precision:** _[pending — computed, not chosen: the lowest-fidelity precision not past the cliff on any headline metric in either suite]_

The report distinguishes, and will not merge:

- **the minimum precision supported by this experiment** — a claim about these suites, this model, this hardware, this sample size; and
- **a universally safe production precision** — a claim this experiment cannot make.

## 9. Limitations

See README §7 and `docs/METRICS.md` §7. The one that bounds everything above: **the rule-based PS-1 scorer's agreement with human judgement is unmeasured until §3b is completed.**

## 10. Reproduction

```bash
pip install -r requirements.txt && export PYTHONPATH="$PWD/src:$PYTHONPATH"
python3 scripts/build_suites.py --check && python3 scripts/build_manifest.py --check
python3 -m pytest -q
python3 scripts/preflight.py --backend ollama
bash scripts/run_all.sh ollama
```
