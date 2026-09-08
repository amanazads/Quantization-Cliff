"""Render reports/FINDINGS.md entirely from aggregate.json.

Every number, table and verdict in the findings report is read from the
aggregate. Nothing is typed by hand, so the report cannot drift from the data,
and a re-run regenerates a report that matches the new numbers automatically.

Where a result does not exist -- an arm that was not run, a metric with no
reference -- the report says so explicitly rather than leaving a gap the reader
might mistake for a measurement.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .cliff import CliffCriterion

__all__ = ["render_findings", "SyntheticReportRefused"]


class SyntheticReportRefused(RuntimeError):
    """Raised when asked to write a findings report from fabricated data."""


PRECISION_LABEL = {"bf16": "BF16", "fp8": "FP8", "q8": "Q8", "q4": "Q4"}

FIGURES = [
    ("01_guardrail_adherence_vs_precision.png", "Guardrail adherence and over-refusal vs precision"),
    ("02_structured_output_vs_precision.png", "Structured-output metrics vs precision"),
    ("03_ps3_failure_modes.png", "PS-3 failure modes by precision"),
    ("04_language_breakdown.png", "Per-language breakdown"),
    ("05_english_vs_indic.png", "English vs Indic"),
    ("06_ps1_category_heatmap.png", "PS-1 violation rate by category"),
    ("07_degradation_vs_reference.png", "Degradation vs the reference precision"),
]


def _pct(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _cell(blob: Optional[Dict[str, Any]]) -> str:
    if not blob or blob.get("value") is None:
        return "n/a"
    text = _pct(blob["value"])
    lo, hi = blob.get("ci_low"), blob.get("ci_high")
    if lo is not None:
        text += f" <sub>[{_pct(lo)}, {_pct(hi)}] n={blob.get('denominator')}</sub>"
    if blob.get("small_sample"):
        text += " ⚠"
    return text


def _metric_table(per_precision: Dict[str, Any], metrics: Sequence[str],
                  order: Sequence[str]) -> List[str]:
    present = [p for p in order if p in per_precision]
    if not present:
        return ["_No arms present for this suite._", ""]
    lines = ["| metric | " + " | ".join(PRECISION_LABEL.get(p, p.upper()) for p in present) + " |",
             "|---|" + "---|" * len(present)]
    for metric in metrics:
        cells = [_cell((per_precision.get(p) or {}).get(metric)) for p in present]
        lines.append(f"| `{metric}` | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("<sub>Values are point estimates with Wilson 95% intervals and the "
                 "denominator. ⚠ marks a cell below the pre-registered small-sample "
                 "threshold; those are directional and no significance is claimed.</sub>")
    lines.append("")
    return lines


def _backend_name(arms: Dict[str, Any], order: Sequence[str]) -> str:
    return (arms[order[0]] or {}).get("backend", "ollama") if order else "ollama"


def _set_arg(agg: Dict[str, Any]) -> str:
    """The config-set argument, so the printed command reruns THIS experiment.

    Omitted for the default set, whose command takes no set argument. A
    reproduction line that silently runs a different model would be worse than
    printing none at all.
    """
    name = agg.get("config_set")
    return f" {name}" if name and name != "default" else ""


def _set_flag(agg: Dict[str, Any]) -> str:
    """`--config-set X` for the per-arm commands, omitted for the default set."""
    name = agg.get("config_set")
    return f" --config-set {name}" if name and name != "default" else ""


def _load_agreement(repo_root: Path) -> Optional[Dict[str, Any]]:
    """Scorer-vs-human agreement, if it has been measured."""
    path = repo_root / "reports" / "validation" / "agreement.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def render_findings(
    aggregate_path: Path,
    criterion: CliffCriterion,
    figures_dir: Optional[Path] = None,
    allow_synthetic: bool = False,
    figures_relative: str = "figures",
    concise: bool = False,
    repo_root: Optional[Path] = None,
) -> str:
    """Render the findings document.

    `concise=True` produces the <=4-page document the specification asks to be
    submitted; the full version keeps every breakdown table and is written
    alongside as an appendix. Both are rendered from the same aggregate, so they
    cannot disagree.
    """
    agg = json.loads(Path(aggregate_path).read_text(encoding="utf-8"))
    synthetic = bool(agg.get("synthetic"))

    if synthetic and not allow_synthetic:
        raise SyntheticReportRefused(
            "REFUSING to write a findings report from synthetic data.\n\n"
            "At least one arm in this aggregate came from the mock backend, whose "
            "output is a fabricated pipeline-validation fixture rather than a "
            "measurement of any model. Writing a findings report from it would "
            "produce a document that reads exactly like a real result.\n\n"
            "Run the arms against a real backend, or pass --allow-synthetic to "
            "produce a clearly-stamped validation artefact."
        )

    arms = agg.get("arms", {})
    order = [p for p in criterion.precision_order if p in arms]
    missing = agg.get("missing_arms", [])
    degradation = agg.get("degradation", {})
    mvp = degradation.get("minimum_viable_precision", {})
    comparability = agg.get("comparability", {})

    out: List[str] = ["# PS-5: The Quantization Cliff", ""]

    if synthetic:
        out += [
            "> # ⚠ SYNTHETIC VALIDATION ARTEFACT — NOT A RESULT",
            ">",
            "> Every number in this document was produced by the deterministic **mock "
            "backend**, which fabricates output from a hand-written failure profile. "
            "It exists to demonstrate that the pipeline computes, aggregates and "
            "reports correctly. It is **not a measurement of any model**, and no "
            "statement in it may be cited as a PS-5 finding.",
            "",
        ]

    out += [
        f"_Generated from `{Path(aggregate_path).name}` at {agg.get('generated_at_utc')}. "
        f"Metric spec `{agg.get('metric_spec_version')}`. Every figure and table in this "
        "document is rendered from raw results; none is typed by hand._",
        "",
        "---", "",
        "## 1. Objective", "",
        "Determine how quantization affects an open-weight collections agent along "
        "three separately-reported axes — guardrail adherence (PS-1), structured "
        "output and tool-calling validity (PS-3), and language-specific behaviour — "
        "and locate the precision at which degradation becomes meaningful, using a "
        "criterion fixed before any result was observed.",
        "",
        "Safety and structured output are **never combined into a single score**. The "
        "central question PS-5 asks is whether they degrade differently, and an "
        "average would destroy exactly that signal.",
        "",
        "---", "",
        "## 2. Experimental setup", "",
    ]

    if not arms:
        out += ["**No arms were run.** There is nothing to report.", ""]
        return "\n".join(out) + "\n"

    out += ["| | " + " | ".join(PRECISION_LABEL.get(p, p.upper()) for p in order) + " |",
            "|---|" + "---|" * len(order)]
    rows = [
        ("model", lambda a: f"`{(a.get('model') or {}).get('id', '?')}`"),
        ("model tag", lambda a: f"`{(a.get('model') or {}).get('tag', '?')}`"),
        ("quantization format", lambda a: (a.get("model") or {}).get("quantization_format") or "?"),
        ("resolved quant level", lambda a: str(((a.get("model") or {}).get("resolved") or {}).get("quantization_level") or "—")),
        ("weights digest", lambda a: f"`{str(((a.get('model') or {}).get('resolved') or {}).get('digest') or '—')[:19]}`"),
        ("backend", lambda a: f"`{a.get('backend')}`"),
        ("cases run", lambda a: ", ".join(f"{k}={v}" for k, v in (a.get("case_counts") or {}).items()) or "—"),
        ("repeats", lambda a: str(a.get("repeats"))),
    ]
    for label, getter in rows:
        out.append(f"| {label} | " + " | ".join(getter(arms[p]) for p in order) + " |")
    out.append("")

    env = (arms[order[0]].get("environment_summary") or {})
    dirty = (" **(working tree dirty — the recorded commit does not fully describe "
             "the code that ran)**" if env.get("git_dirty") else "")
    if concise:
        # The four-page cap is a hard requirement, so the submission document
        # gets one prose line here and the full environment block goes to the
        # appendix. Nothing is dropped -- the fingerprint that PROVES the
        # hardware was identical stays, because that is the load-bearing claim.
        out += [
            "**Hardware and software** (identical across arms; the fingerprint is "
            f"verified by the aggregator, not asserted): `{env.get('os')}` · "
            f"`{env.get('cpu')}` · {env.get('ram_gb')} GB RAM · accelerator "
            f"`{env.get('accelerator')}` (`{env.get('accelerator_kind')}`) · CUDA "
            f"`{env.get('cuda_version') or 'n/a'}` · Python `{env.get('python')}` · "
            f"code `{str(env.get('git_commit') or '?')[:12]}`{dirty} · fingerprint "
            f"`{str(arms[order[0]].get('hardware_fingerprint') or '')[:19]}…`", "",
        ]
    else:
        out += [
            "**Hardware and software** (identical across arms; the fingerprint is verified "
            "by the aggregator, not asserted):", "",
            f"- OS: `{env.get('os')}`",
            f"- CPU: `{env.get('cpu')}`",
            f"- RAM: `{env.get('ram_gb')} GB`",
            f"- Accelerator: `{env.get('accelerator')}` (`{env.get('accelerator_kind')}`)",
            f"- CUDA: `{env.get('cuda_version') or 'n/a'}` · compute capability: `{env.get('compute_capability') or 'n/a'}`",
            f"- Python: `{env.get('python')}`",
            f"- Code revision: `{env.get('git_commit')}`" + dirty,
            f"- Hardware fingerprint: `{arms[order[0]].get('hardware_fingerprint')}`",
            "",
        ]

    generation = arms[order[0]].get("generation_config", {})
    if concise:
        out += [
            "**Decoding** (identical across arms, hash-verified): greedy — "
            f"temperature {generation.get('temperature')}, top_p {generation.get('top_p')}, "
            f"top_k {generation.get('top_k')}, seed {generation.get('seed')}, "
            f"max_tokens {generation.get('max_tokens')}; serial execution; thinking mode "
            "disabled as the specification requires.", "",
        ]
    else:
        out += [
            "**Decoding parameters** (identical across arms; hash-verified):", "",
            "```json", json.dumps(generation, indent=2), "```", "",
        ]

    if missing:
        out += [
            "### Arms not run", "",
            "These required precisions were **not executed**. They are gaps in "
            "coverage, and must not be read as null results:", "",
        ]
        reasons = agg.get("missing_arm_reasons") or {}
        for precision in missing:
            label = PRECISION_LABEL.get(precision, precision.upper())
            detail = reasons.get(precision) or {}
            reason = (detail.get("reason") or "").strip()
            if reason:
                # A refusal with a stated reason is a different thing from an
                # arm nobody attempted, and the difference is worth a reader's
                # attention: it is the difference between a gap and an omission.
                out.append(f"- **{label}** — refused, not skipped. {reason}")
                policy = (detail.get("substitution_policy") or "").strip()
                if policy and not concise:
                    out.append(f"  - _Substitution policy:_ {policy}")
            else:
                out.append(f"- **{label}** — not run in this comparison set.")
        out.append("")

    out += ["---", "", "## 3. Experimental controls", "", "### Held constant", ""]
    held = arms[order[0]].get("controls_held_constant") or [
        "system prompt", "tool schemas", "evaluation manifest", "decoding parameters",
        "scorer versions", "case order", "concurrency", "context window",
    ]
    if concise:
        out.append("Held constant and hash-verified: " + "; ".join(
            h.split(" (")[0] for h in held) + ".")
    else:
        for item in held:
            out.append(f"- {item}")
    out += ["",
            "The aggregator **verifies** these rather than trusting them: it compares "
            f"`{'`, `'.join(comparability.get('checked_fields', []))}` across arms and "
            "refuses to produce a comparison when any of them diverges.", ""]

    out += ["### Verification result", ""]
    if comparability.get("comparable"):
        out.append("✅ All control fields matched across every arm in this comparison.")
    else:
        out.append("❌ **Control fields diverged. This comparison does not isolate quantization.**")
        for d in comparability.get("divergences", []):
            out.append(f"- `{d['field']}` ({d['describes']}): {d['consequence']}")
    out.append("")
    for warning in comparability.get("warnings", []):
        out.append(f"- ⚠ {warning}")
    out.append("")

    out += ["### Could NOT be held constant", ""]
    deviations = agg.get("deviations", [])
    if deviations:
        for dev in deviations:
            out.append(f"- **[{dev.get('severity')}] `{dev.get('id')}`** "
                       f"({dev.get('precision')}): {(dev.get('description') or '').strip()}")
            if dev.get("impact"):
                out.append(f"  - _Impact:_ {dev['impact'].strip()}")
            if dev.get("remediation"):
                out.append(f"  - _Remediation:_ {dev['remediation'].strip()}")
    else:
        out.append("_No deviations were recorded for the arms in this comparison._")
    out += ["",
            "- Wall-clock time and machine thermal state differed between arms; runs "
            "were sequential.",
            "- Kernel selection inside the inference engine differs per quantization "
            "format by design. That is inherent to the treatment rather than a failure "
            "of control, but it does mean 'precision' here means 'precision as served "
            "by this stack', not an isolated numerical-format change.",
            ""]

    # ---- results ---------------------------------------------------------- #
    metrics = agg.get("metrics", {})

    # ---- scorer validation (specification requires this to be MEASURED) ---- #
    agreement = _load_agreement(repo_root or Path(aggregate_path).resolve().parents[3])
    out += ["---", "", "## 3b. Scorer validation against human labels", ""]
    if agreement:
        svh = agreement["scorer_vs_human"]
        kappa = svh.get("kappa")
        out += [
            f"- labelled subset: n={svh['n']} "
            f"(exported {agreement['n_rows']}, undecidable {agreement['n_unlabelled']})",
            f"- raters: {', '.join(agreement['raters']) or 'none'}",
            (f"- **Cohen's kappa: {kappa:.3f} ({svh['interpretation']})**"
             if kappa is not None else "- **Cohen's kappa: undefined** — see `reports/validation/agreement.md`"),
            (f"- raw agreement: {svh['observed_agreement']:.1%} "
             f"[{svh['agreement_ci_low']:.1%}, {svh['agreement_ci_high']:.1%}]"
             if svh["observed_agreement"] is not None else "- raw agreement: n/a"),
            (f"- scorer precision {svh['precision']:.1%}, recall {svh['recall']:.1%}"
             if svh["precision"] is not None and svh["recall"] is not None else ""),
            "",
        ]
        if agreement.get("human_vs_human"):
            hvh = agreement["human_vs_human"]
            if hvh.get("kappa") is not None:
                out += [f"- human-vs-human kappa: {hvh['kappa']:.3f} over n={hvh['n']}, "
                        "which bounds what any automated scorer could achieve here.", ""]
        if kappa is not None and kappa < 0.61:
            out += [
                "> **Agreement is below the substantial band.** Absolute violation rates "
                "from this scorer are weakly supported and are not quoted here as safety "
                "rates. The between-precision comparison is more robust to this, because "
                "scorer error is constant across arms, but it is not immune to it.", "",
            ]
    else:
        out += [
            "> **NOT YET MEASURED.** The specification requires the automated scorer to be "
            "validated against human labels on a subset, with the agreement reported. "
            "That has not been done for this run, so every absolute violation rate below "
            "rests on an unvalidated scorer and must be read as provisional.",
            "",
            "> To close this gap:",
            "> ```bash",
            # Must name the results root this report was built from. The default
            # is `results/`, which is a different experiment (or empty), so the
            # command as printed previously failed for anyone who followed it.
            f"> python3 scripts/validation_subset.py export --results-root "
            f"{agg.get('results_root') or 'results'} --n 80   # blind, stratified",
            "> # a human labels reports/validation/ps1_validation_labelled.csv",
            "> python3 scripts/validation_subset.py score",
            "> ```",
            "",
        ]

    out += ["---", "", "## 4. PS-1 results — guardrail adherence", ""]
    if "ps1" in metrics:
        out += _metric_table(metrics["ps1"],
                             ["violation_rate", "compliance_rate", "benign_refusal_rate",
                              "generation_failure_rate"], order)
    if "ps1" in metrics:
        present = [p for p in order if p in metrics["ps1"]]

        # The English-vs-Indic delta is a headline number the specification calls
        # out by name, so it survives into the concise document.
        out += ["### English vs Indic", "",
                "| precision | English | Indic | delta | 95% CI on the delta | significant |",
                "|---|---|---|---|---|---|"]
        for precision in present:
            node = metrics["ps1"][precision].get("english_indic_delta") or {}
            out.append(
                f"| {PRECISION_LABEL.get(precision, precision.upper())} "
                f"| {_cell(node.get('english'))} | {_cell(node.get('indic'))} "
                f"| {_pct(node.get('value'))} "
                f"| [{_pct(node.get('ci_low'))}, {_pct(node.get('ci_high'))}] "
                f"| {'yes' if node.get('significant') else 'no'} |"
            )
        out += ["", "<sub>Positive delta means Indic-language safety is worse than English.</sub>", ""]

    if "ps1" in metrics and not concise:
        out += ["### By violation category", ""]
        categories = sorted({c for p in metrics["ps1"].values()
                             for c in (p.get("category_violation_rate") or {})})
        if categories:
            out.append("| category | " + " | ".join(PRECISION_LABEL.get(p, p.upper()) for p in present) + " |")
            out.append("|---|" + "---|" * len(present))
            for category in categories:
                cells = [_cell((metrics["ps1"][p].get("category_violation_rate") or {}).get(category))
                         for p in present]
                out.append(f"| {category} | " + " | ".join(cells) + " |")
            out.append("")

        out += ["### By language", ""]
        languages = sorted({l for p in metrics["ps1"].values()
                            for l in (p.get("language_violation_rate") or {})})
        if languages:
            out.append("| language | " + " | ".join(PRECISION_LABEL.get(p, p.upper()) for p in present) + " |")
            out.append("|---|" + "---|" * len(present))
            for language in languages:
                cells = [_cell((metrics["ps1"][p].get("language_violation_rate") or {}).get(language))
                         for p in present]
                out.append(f"| {language} | " + " | ".join(cells) + " |")
            out.append("")

    if "ps1" not in metrics:
        out += ["_PS-1 was not run in this comparison set._", ""]
    elif concise:
        out += ["<sub>Per-category (V1–V8) and per-language breakdowns are in "
                "`FINDINGS_FULL.md`, rendered from the same aggregate.</sub>", ""]

    out += ["---", "", "## 5. PS-3 results — structured output and tool calling", ""]
    if "ps3" in metrics:
        # The concise document carries the metrics a reader needs to judge the
        # cliff verdict; the remaining failure-mode breakdown lives in the
        # appendix. Both render from the same aggregate, so they cannot disagree.
        ps3_metrics = ["task_success_rate", "correct_tool_rate", "argument_accuracy",
                       "structured_output_validity", "malformed_argument_rate",
                       "missed_call_rate", "generation_failure_rate"] if concise else [
                       "task_success_rate", "correct_tool_rate", "argument_accuracy",
                       "structured_output_validity", "malformed_argument_rate",
                       "wrong_tool_rate", "wrong_argument_rate", "spurious_call_rate",
                       "missed_call_rate", "fallback_extraction_rate",
                       "generation_failure_rate"]
        out += _metric_table(metrics["ps3"], ps3_metrics, order)
        out += [
            "`correct_tool_rate` counts a case as correct when the tool identity is "
            "right, **regardless of the argument values**. `task_success_rate` requires "
            "the arguments to be right too. The gap between them is the "
            "argument-corruption rate, and it is the single most operationally "
            "dangerous failure in this suite: a `capture_ptp` for ₹50,000 instead of "
            "₹5,000 is a well-formed, correctly-routed, wrong commitment.",
            "",
        ]
        present = [p for p in order if p in metrics["ps3"]]

        # The English-vs-Hinglish delta is, in the specification's words, "the
        # headline number" for PS-3, so it stays in the concise document.
        out += ["### English vs Indic (task success)", "",
                "| precision | English | Indic | delta | 95% CI on the delta | significant |",
                "|---|---|---|---|---|---|"]
        for precision in present:
            node = metrics["ps3"][precision].get("english_indic_delta") or {}
            out.append(
                f"| {PRECISION_LABEL.get(precision, precision.upper())} "
                f"| {_cell(node.get('english'))} | {_cell(node.get('indic'))} "
                f"| {_pct(node.get('value'))} "
                f"| [{_pct(node.get('ci_low'))}, {_pct(node.get('ci_high'))}] "
                f"| {'yes' if node.get('significant') else 'no'} |"
            )
        out += ["", "<sub>NEGATIVE delta means Indic tool-calling is worse than English.</sub>", ""]

    # NOTE: "not run" depends ONLY on whether the suite is absent -- never on
    # whether this is the concise rendering. An earlier version bound the else
    # to `and not concise`, so the four-page submission document asserted
    # "PS-3 was not run in this comparison set" directly beneath a full page of
    # PS-3 results. A report that contradicts its own tables is worse than one
    # that omits them.
    if "ps3" not in metrics:
        out += ["_PS-3 was not run in this comparison set._", ""]
    elif not concise:
        out += ["### By language", ""]
        present = [p for p in order if p in metrics["ps3"]]
        languages = sorted({l for p in metrics["ps3"].values() for l in (p.get("by_language") or {})})
        if languages:
            out.append("| language | " + " | ".join(PRECISION_LABEL.get(p, p.upper()) for p in present) + " |")
            out.append("|---|" + "---|" * len(present))
            for language in languages:
                cells = [_cell(((metrics["ps3"][p].get("by_language") or {}).get(language) or {})
                               .get("task_success_rate")) for p in present]
                out.append(f"| {language} | " + " | ".join(cells) + " |")
            out += ["", "<sub>Task success rate by language.</sub>", ""]

    # ---- figures ---------------------------------------------------------- #
    out += ["---", "", "## 6. Quantization degradation", ""]
    # Three figures in the concise document, not seven: one for each axis the
    # specification requires reported separately (guardrail adherence,
    # structured output), plus degradation against the reference, which is the
    # PS-5 deliverable itself. The failure-mode and per-language breakdowns are
    # in the appendix. Measured at 4 pages with scripts/export_pdf.sh; adding a
    # fourth figure pushed it over.
    wanted = [FIGURES[0], FIGURES[1], FIGURES[6]] if concise else FIGURES
    if figures_dir and figures_dir.exists():
        for filename, caption in wanted:
            if (figures_dir / filename).exists():
                out += [f"**{caption}**", "",
                        f"![{caption}]({figures_relative}/{filename})", ""]
    else:
        out += ["_Figures not rendered. Run `python scripts/make_plots.py`._", ""]

    # ---- cliff ------------------------------------------------------------ #
    out += ["---", "", "## 7. The quantization cliff", "", "### Methodology", ""]
    if concise:
        out += [
            "Fixed in `configs/cliff_criterion.yaml` and `docs/METRICS.md` **before any "
            "model was run**. A precision is *past the cliff* on a metric only when "
            "**both** hold: degradation vs the reference meets the pre-registered "
            "threshold for that metric, **and** the Newcombe 95% interval for the "
            "difference excludes zero. Requiring both stops a large-but-noisy "
            "difference at small `n` being called a cliff, and equally stops a "
            "statistically clean but operationally trivial one.", ""]
    else:
        out += [
            "Fixed in `configs/cliff_criterion.yaml` and `docs/METRICS.md` **before any "
            "model was run**. A precision is *past the cliff* on a metric only when "
            "**both** conditions hold:", "",
            "1. **Practical** — degradation vs the reference meets the pre-registered "
            "threshold for that metric.",
            "2. **Statistical** — the Newcombe 95% interval for the difference excludes zero.",
            "",
            "Requiring both is what stops a large-but-noisy difference at small `n` "
            "being reported as a cliff, and equally stops a statistically clean but "
            "operationally trivial difference being reported as one.", ""]
    if degradation.get("challenge_defined_thresholds"):
        out.append("The thresholds are defined by the challenge specification.")
    else:
        out.append("**The thresholds are an experimental convention of this repository, "
                   "not defined by the challenge specification.** Safety uses a tighter "
                   "threshold than structured output, because a conduct breach is a "
                   "regulatory event whereas a malformed tool call is a retry. "
                   "Justification is in `docs/METRICS.md` §4.3.")
    out.append("")

    out += ["### Verdict per headline metric", ""]

    if concise:
        # One matrix instead of a table per metric. The cells carry the
        # degradation and mark both cliff conditions, which is everything needed
        # to check the verdict; per-metric intervals live in the appendix.
        deltas = [p for p in criterion.precision_order
                  if p != criterion.reference_precision
                  and any(p in {q["precision"] for q in a["points"]}
                          for v in (degradation.get("analyses") or {}).values() for a in v)]
        out += ["| suite · metric | " + " | ".join(PRECISION_LABEL.get(p, p.upper()) for p in deltas)
                + " | pattern | cliff at |",
                "|---|" + "---|" * (len(deltas) + 2)]
        for suite, analyses in (degradation.get("analyses") or {}).items():
            for analysis in analyses:
                by_precision = {p["precision"]: p for p in analysis["points"]}
                cells = []
                for precision in deltas:
                    point = by_precision.get(precision)
                    if not point or point.get("degradation") is None:
                        cells.append("—")
                        continue
                    mark = " ✱" if point.get("past_cliff") else (
                        " ·" if point.get("meets_practical_threshold") else "")
                    cells.append(f"{point['degradation'] * 100:+.1f}{mark}")
                cliff_at = analysis.get("cliff_precision")
                out.append(
                    f"| {suite.upper()} · `{analysis['metric']}` | " + " | ".join(cells)
                    + f" | `{analysis['pattern']}` | "
                    + (f"**{PRECISION_LABEL.get(cliff_at, str(cliff_at).upper())}**"
                       if cliff_at else "—") + " |"
                )
        out += ["",
                "<sub>Cells are degradation vs the reference in percentage points; positive "
                "is always worse. **✱** = past the cliff (threshold met AND the difference "
                "interval excludes zero). **·** = threshold met but the interval still "
                "includes zero, so it is NOT called a cliff. Per-metric intervals and "
                "sample sizes are in `FINDINGS_FULL.md`.</sub>", ""]
        for suite, analyses in (degradation.get("analyses") or {}).items():
            for analysis in analyses:
                if analysis.get("pattern") != "none":
                    out += [f"- **{suite.upper()} `{analysis['metric']}`** — "
                            + analysis.get("statement", ""), ""]
        nulls = [f"{s.upper()} `{a['metric']}`"
                 for s, v in (degradation.get("analyses") or {}).items()
                 for a in v if a.get("pattern") == "none"]
        if nulls:
            # Per-metric, NOT a single worst case across all of them. An earlier
            # version printed max(mdds) as though it applied to every null --
            # which took the weakest cell (a benign-control rate at n=32) and
            # let it speak for metrics that were resolved an order of magnitude
            # more finely. That understated the experiment's own best result.
            #
            # A null is only informative when the experiment could have SEEN the
            # effect it pre-registered: MDD <= threshold. Sorting the nulls into
            # those two groups is the difference between "we found nothing" and
            # "we found nothing, and here is what that is worth".
            out += [f"- **No cliff detected** on: {', '.join(nulls)}.", ""]
            rows = [(s, a) for s, v in (degradation.get("analyses") or {}).items()
                    for a in v if a.get("pattern") == "none"
                    and a.get("minimum_detectable_difference") is not None]
            if rows:
                out += [
                    "**How much each null is worth.** A null means something only "
                    "where the minimum detectable difference (MDD) at the achieved "
                    "sample size is no larger than the effect the criterion was "
                    "pre-registered to look for.", "",
                    "| suite · metric | threshold | MDD | is this null informative? |",
                    "|---|---|---|---|",
                ]
                powered = 0
                for suite, analysis in rows:
                    threshold = analysis.get("threshold")
                    mdd = analysis["minimum_detectable_difference"]
                    ok = threshold is not None and mdd <= threshold
                    powered += bool(ok)
                    verdict = (
                        "**Yes** — a cliff of the pre-registered size would have been visible"
                        if ok else
                        f"**No** — ~{mdd / threshold:.0f}× too few cases to see a "
                        f"{threshold:.1%} effect" if threshold else "unknown"
                    )
                    out.append(f"| {suite.upper()} · `{analysis['metric']}` | "
                               f"{threshold:.1%} | {mdd:.1%} | {verdict} |")
                out += ["", f"<sub>{powered} of {len(rows)} null results were adequately "
                            "powered. For the rest, a real effect smaller than the MDD was "
                            "invisible to this experiment rather than absent — they are not "
                            "evidence that quantization did no harm.</sub>", ""]

    for suite, analyses in ({} if concise else (degradation.get("analyses") or {})).items():
        out += [f"#### {suite.upper()}", ""]
        for analysis in analyses:
            out += [
                f"**`{analysis['metric']}`** — threshold {analysis['threshold']:.1%}, "
                f"pattern **`{analysis['pattern']}`**"
                + (f", cliff at **{PRECISION_LABEL.get(analysis['cliff_precision'], str(analysis['cliff_precision']).upper())}**"
                   if analysis.get("cliff_precision") else ""),
                "", analysis.get("statement", ""), "",
            ]
            points = [p for p in analysis["points"] if not p.get("is_reference")]
            if points:
                out += ["| precision | value | degradation vs ref | 95% CI on the difference | "
                        "meets threshold | CI excludes 0 | past cliff |",
                        "|---|---|---|---|---|---|---|"]
                for point in points:
                    out.append(
                        f"| {PRECISION_LABEL.get(point['precision'], point['precision'].upper())} "
                        f"| {_pct(point.get('value'))} "
                        f"| {_pct(point.get('degradation'))} "
                        f"| [{_pct(point.get('diff_ci_low'))}, {_pct(point.get('diff_ci_high'))}] "
                        f"| {'yes' if point.get('meets_practical_threshold') else 'no'} "
                        f"| {'yes' if point.get('diff_excludes_zero') else 'no'} "
                        f"| {'**YES**' if point.get('past_cliff') else 'no'} |"
                    )
                out.append("")
            for note in analysis.get("notes", []):
                out.append(f"- {note}")
            out.append("")

    # ---- recommendation --------------------------------------------------- #
    chosen = mvp.get("precision")
    out += ["---", "", "## 8. Production recommendation", "",
            f"### Minimum viable precision: **{PRECISION_LABEL.get(chosen, str(chosen).upper()) if chosen else 'none of the tested precisions'}**",
            "", mvp.get("rationale", ""), ""]

    failing = mvp.get("failing_metrics_by_precision") or {}
    if any(failing.values()):
        out += ["**Why the lower precisions were rejected:**", ""]
        for precision, failures in failing.items():
            if failures:
                out.append(f"- **{PRECISION_LABEL.get(precision, precision.upper())}** failed on:")
                for failure in failures:
                    out.append(
                        f"  - `{failure['suite']}.{failure['metric']}` — degradation "
                        f"{_pct(failure.get('degradation'))} against a threshold of "
                        f"{_pct(failure.get('threshold'))}"
                    )
        out.append("")

    out += ["### What this recommendation is, and is not", ""]
    if concise:
        out += [
            "**Supported:** the minimum precision for *this* suite, model, hardware and "
            "sample size — that is exactly what was measured. **Not supported:** a "
            "universally safe production precision, and any claim that the recommended "
            "precision is production-*safe*. Absence of a detected cliff is not evidence "
            "of safety, especially where the minimum detectable difference exceeds the "
            "effect that would matter operationally — see the power table in §7.", ""]
    else:
        out += [
            "| claim | supported by this experiment? |",
            "|---|---|",
            "| Minimum precision supported by **this** suite, model, hardware and sample size | **Yes** — that is exactly what was measured. |",
            "| Universally safe production precision for collections | **No.** This experiment cannot support that claim and does not make it. |",
            "| Evidence that the recommended precision is production-**safe** | **No.** Absence of a detected cliff is not evidence of safety, particularly where the minimum detectable difference is larger than the effect that would matter operationally. |",
            "",
        ]
    out += [
        "Confidence: **moderate for the direction of the effect, low for its exact "
        "magnitude.** Sample sizes are fixed and modest, the PS-1 scorer's agreement "
        "with human judgement is unmeasured, and the suites are single-turn.",
        "",
    ]

    # ---- limitations ------------------------------------------------------ #
    out += ["---", "", "## 9. Limitations", "",
            "Stated in advance in `docs/METRICS.md` §7 rather than discovered afterwards.", ""]
    limitations = [
        "**The PS-1 scorer is rule-based and its validity is unmeasured.** It has not "
        "been checked against human labels. It will miss paraphrased violations and may "
        "fire on quoted or negated text. Its error is constant across arms, so it biases "
        "absolute violation rates more than it biases the between-precision comparison — "
        "but the absolute numbers should not be quoted as a safety rate in isolation. "
        "This is the single biggest methodological weakness here.",
        "**One target category is scored per case.** Cross-category violations are not detected.",
        "**Free-text tool arguments are not scored**, so `argument_accuracy` covers structured fields only.",
        "**Per-category and per-language cells are small** and are directional only.",
        "**Single-turn only.** Multi-turn drift, where quantization damage plausibly "
        "compounds, is not measured at all.",
        "**The FP8-above-Q8 fidelity ordering is an assumption**, not a measurement.",
        "**Determinism is best-effort.** Greedy decoding with a fixed seed is requested, "
        "but llama.cpp/vLLM do not guarantee bit-identical output across differing batch "
        "or thread configurations.",
        "**The suites were authored for this repository** from the challenge brief. They "
        "are not the official PS-1/PS-3 suites, and results are not comparable with runs "
        "on the official ones.",
        "**One model family at one size, chosen to fit the hardware.** Quantization "
        "sensitivity varies sharply with model size, and smaller models are generally "
        "LESS robust to it. Two consequences, in opposite directions: a cliff found "
        "here is plausibly pessimistic for a larger deployment model, while a NULL "
        "here is close to uninformative about one. Check the reference arm's absolute "
        "value before reading any null as reassuring -- if the BF16 baseline is "
        "already weak on a metric, that metric had little room to degrade and the "
        "null reflects a floor effect rather than robustness. See `docs/METRICS.md` "
        "section 7.8.",
    ]
    if missing:
        limitations.insert(0,
            f"**Incomplete precision coverage: {', '.join(PRECISION_LABEL.get(p, p.upper()) for p in missing)} "
            "was not run.** The cliff can only be located among the arms that were "
            "actually executed; a cliff could lie at an untested precision.")
    for item in (limitations[:6] if concise else limitations):
        out.append(f"- {item}")
    if concise and len(limitations) > 6:
        out.append(f"- _{len(limitations) - 6} further limitations are listed in "
                   "`FINDINGS_FULL.md` and `docs/METRICS.md` §7._")
    out.append("")

    if concise:
        out += ["---", "", "## 10. Reproduction", "",
                "```bash",
                "pip install -r requirements.txt && export PYTHONPATH=$PWD/src",
                "python3 scripts/build_suites.py --check && python3 scripts/build_manifest.py --check",
                f"bash scripts/run_all.sh {_backend_name(arms, order)}{_set_arg(agg)}",
                "```", "",
                "Reproduction is exact only when the six control hashes in §2 match; "
                "all are recorded in every run's `metadata.json`. Runbook in README §4, "
                "full breakdowns in `FINDINGS_FULL.md`.", ""]
        return "\n".join(out) + "\n"

    out += ["---", "", "## 10. Reproduction", "",
            "```bash", "# 1. environment",
            "python -m venv .venv && source .venv/bin/activate",
            "pip install -r requirements.txt", "",
            "# 2. verify the frozen inputs are unmodified",
            "python scripts/build_suites.py --check",
            "python scripts/build_manifest.py --check", "",
            "# 3. run each arm (see README for backend setup)"]
    for precision in criterion.precision_order:
        backend = (arms.get(precision) or {}).get("backend", "ollama")
        marker = "" if precision in arms else "   # NOT RUN in this comparison set"
        out.append(f"python -m ps5.run --precision {precision} --backend {backend}"
                   f"{_set_flag(agg)} --suite ps1 ps3{marker}")
    out += ["", "# 4. aggregate, plot, report",
            "python scripts/aggregate_results.py",
            "python scripts/make_plots.py",
            "python scripts/generate_report.py",
            "```", "",
            "Reproduction is only exact when these match the values in §2: "
            "`manifest_hash`, `system_prompt_hash`, `tool_schema_hash`, "
            "`generation_config_hash`, `guardrail_rules_hash`, `hardware_fingerprint`. "
            "They are recorded in every run's `metadata.json`.", ""]

    return "\n".join(out) + "\n"
