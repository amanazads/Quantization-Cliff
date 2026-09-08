# Reviewer guide

Ten minutes, no GPU, no model download. This verifies the claims in the findings
document rather than asking you to take them on trust.

---

## The one-paragraph summary

PS-5 asks where quantization breaks a collections agent. This repository runs the
PS-1 (guardrail) and PS-3 (tool-calling) suites at multiple precisions with every
other variable held constant **and hash-verified**, then applies a cliff criterion
that was fixed in `configs/cliff_criterion.yaml` before any model was run.

One experiment is complete: **Qwen2.5-1.5B at F16 → Q8_0 → Q4_K_M**, 392 cases per arm.
**No cliff was detected on any headline metric** — and the report says plainly
that only *one* of the four metrics had the statistical power for that null to
mean anything. The intended experiment (Qwen3.5-4B, four rungs, genuine bfloat16 reference)
needs more than 8 GB of RAM and cannot run locally on an 8 GB M1 machine; Qwen2.5-1.5B is
the practical local experiment and is labelled as such everywhere.

---

## Read these three, in order

| # | File | Why |
|---|---|---|
| 1 | **`reports/FINDINGS-qwen2.5-1.5b.pdf`** | The submission document. 3 pages. §7 is the part that matters: the power table saying which nulls are worth anything. |
| 2 | **`docs/METRICS.md`** | Every metric and the cliff rule, frozen before any run. §8 is a changelog of what was wrong in v1 and why. |
| 3 | **`README.md` §6** | What the completed run actually showed, including the floor effect that limits it. |

`reports/FINDINGS-qwen2.5-1.5b_FULL.md` is the appendix — per-category, per-language
and failure-mode breakdowns, rendered from the same aggregate so it cannot disagree.

---

## Verify it yourself in ten minutes

```bash
git clone <this repo> && cd Quantization-Cliff
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

**1 — The frozen inputs are unmodified.** The tool schemas and baseline prompt are
transcribed from the specification and must not drift; the suites are generated
deterministically and the manifest is hashed.

```bash
python3 scripts/build_suites.py --check      # suites match their generator
python3 scripts/build_manifest.py --check    # manifest matches the suites
python3 -m pytest -q                         # 241 tests, ~5 s
```

**2 — The report's numbers come from the raw data, not from prose.** Delete the
report, regenerate it, and diff:

```bash
cp reports/FINDINGS-qwen2.5-1.5b.md /tmp/before.md
python3 scripts/aggregate_results.py  --results-root results-qwen2.5-1.5b
python3 scripts/generate_report.py    --results-root results-qwen2.5-1.5b \
        --figures reports/figures-qwen2.5-1.5b --out reports/FINDINGS-qwen2.5-1.5b.md
diff /tmp/before.md reports/FINDINGS-qwen2.5-1.5b.md   # only the timestamp differs
```

**3 — The controls are enforced, not asserted.** Corrupt one control hash and the
aggregator refuses to produce a comparison at all:

```bash
python3 - <<'PY'
import json, pathlib
p = pathlib.Path("results-qwen2.5-1.5b/q4/metadata.json")
m = json.loads(p.read_text()); m["system_prompt_hash"] = "sha256:TAMPERED"
p.write_text(json.dumps(m, indent=2))
PY
python3 scripts/aggregate_results.py --results-root results-qwen2.5-1.5b
# -> COMPARABILITY CHECK FAILED, naming system_prompt_hash. Exit code 1.
git checkout results-qwen2.5-1.5b/q4/metadata.json
```

**4 — An unavailable precision is refused, never substituted.** The FP8 arm has no
GGUF for this model. It is not quietly filled with Q8_0:

```bash
python3 -m ps5.run --precision fp8 --backend ollama --config-set qwen2.5-1.5b --suite ps1
# -> ARM NOT RUNNABLE, with the substitution policy printed. Exit code 3.
cat results-qwen2.5-1.5b/fp8/NOT_RUN.json
```

**5 — The whole pipeline runs with no model at all**, on fabricated data that is
watermarked and blocked from the real report:

```bash
bash scripts/run_all.sh mock          # ~30 s, writes only to results_mock/
```

---

## Where each judging criterion is answered

| Criterion | Where to look |
|---|---|
| **Methodological rigour** (35%) | `docs/METRICS.md` and `configs/cliff_criterion.yaml` — thresholds fixed before any run, and the code cannot lower them. `src/ps5/aggregate.py::check_comparability` refuses an invalid comparison. Verify with step 3 above. |
| **Reproducibility** (25%) | Deterministic suites with drift checks; greedy decoding at a fixed seed; environment, GGUF quantization level and control hashes captured per run into `metadata.json`; 241 tests; every report number rendered from raw JSONL. Verify with steps 1–2. |
| **Insight** (20%) | `reports/FINDINGS-qwen2.5-1.5b.pdf` §7 — the power table. The interesting finding here is *which* of four nulls survives scrutiny, and why the other three do not. |
| **Intellectual honesty** (15%) | The deviation `DEV-BF16-OLLAMA-F16` on the reference arm, printed in the report rather than buried; FP8 reported as a refused gap with its reason; the scorer-validation gap stated in the report's own §3b; `docs/METRICS.md` §8 listing nine things v1 got wrong. |
| **Craft** (5%) | `scripts/run_all.sh` refuses to mix fabricated and measured arms before running; `scripts/export_pdf.sh` fails the build if the findings document exceeds the four-page cap. |

---

## Known gaps, stated up front

1. **The Qwen3.5-4B experiment has not been run.** Its BF16 reference is 9.3 GB and
   the available machine has 8 GB of unified memory. Without the reference arm no
   degradation can be computed, so the smaller model was used instead — and
   labelled as such everywhere rather than presented as the 4B result.
2. **The scorer's agreement with human labels is moderate (κ = 0.471 over n=80).**
   Validation was completed on an n=80 stratified subset (`reports/validation/agreement.json`).
   Because agreement is below substantial (0.61), absolute violation rates are treated as
   weakly supported and interpreted directionally. Between-precision comparisons remain robust
   because scorer bias is held constant across arms.
3. **The evaluation suites are authored here, not official.** No official suite is
   published, so absolute numbers are not comparable across teams. The
   between-precision comparison is unaffected: every arm consumes the identical
   hashed manifest.
4. **Single-turn only.** Multi-turn register drift, where quantization damage
   plausibly compounds, is not measured.

Full list: `docs/METRICS.md` §7, written before results existed.
