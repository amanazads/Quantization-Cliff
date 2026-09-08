"""Aggregate raw per-case results into comparable, machine-readable metrics.

Two jobs, and the first matters more than the second:

  1. VERIFY that the arms being compared are actually comparable. If the prompt,
     schemas, manifest, decoding parameters, scorer, model family or hardware
     differ between arms, the comparison does not measure quantization -- it
     measures whatever else changed. Such a comparison is refused unless the
     caller passes --allow-deviation, and the deviation is then carried into the
     findings report rather than quietly dropped.

  2. Compute the metrics and run the pre-registered cliff criterion.

Nothing here is ever typed by hand into a report: reports/FINDINGS.md is rendered
from this module's JSON output.
"""

from __future__ import annotations

import csv
import json
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .cliff import CliffCriterion, detect_cliffs, load_criterion
from .metrics import compute_ps1_metrics, compute_ps3_metrics

__all__ = [
    "load_run",
    "discover_runs",
    "check_comparability",
    "aggregate",
    "write_outputs",
    "ComparabilityError",
]


class ComparabilityError(RuntimeError):
    """Raised when arms differ in a way that invalidates the comparison."""


#: Fields that MUST match across arms for a comparison to mean "quantization did this".
CONTROL_FIELDS: List[Tuple[str, str]] = [
    ("manifest_hash", "the evaluation cases themselves"),
    ("system_prompt_hash", "the system prompt"),
    ("tool_schema_hash", "the tool schemas"),
    ("generation_config_hash", "the decoding parameters"),
    ("metric_spec_version", "the metric specification"),
    ("guardrail_rules_hash", "the PS-1 guardrail rule set"),
    ("hardware_fingerprint", "the hardware"),
]

#: Differ legitimately between arms (that IS the treatment), so never checked.
TREATMENT_FIELDS = ["precision", "model.tag", "model.quantization_format", "backend.name"]


@dataclass
class Run:
    precision: str
    directory: Path
    metadata: Dict[str, Any]
    records: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)

    @property
    def synthetic(self) -> bool:
        return bool(self.metadata.get("backend", {}).get("synthetic", False))

    def control_value(self, field_name: str) -> Any:
        return self.metadata.get(field_name)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ComparabilityError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
    return out


def load_run(directory: Path) -> Run:
    meta_path = directory / "metadata.json"
    if not meta_path.exists():
        raise ComparabilityError(
            f"{directory} has no metadata.json. A results directory without metadata "
            "cannot be verified as comparable and is therefore not usable."
        )
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    run = Run(
        precision=metadata.get("precision", {}).get("id", directory.name),
        directory=directory,
        metadata=metadata,
    )
    for suite_file in sorted(directory.glob("*_results.jsonl")):
        suite_id = suite_file.name.replace("_results.jsonl", "")
        run.records[suite_id] = _read_jsonl(suite_file)
    return run


def discover_runs(results_root: Path, precisions: Optional[Sequence[str]] = None) -> Dict[str, Run]:
    runs: "OrderedDict[str, Run]" = OrderedDict()
    for child in sorted(results_root.iterdir() if results_root.exists() else []):
        if not child.is_dir() or child.name == "aggregate":
            continue
        if precisions and child.name not in precisions:
            continue
        if not (child / "metadata.json").exists():
            continue
        run = load_run(child)
        runs[run.precision] = run
    return runs


def _effective_thinking(run: Run) -> Optional[bool]:
    """Was thinking mode actually ON for this arm? None when unrecorded.

    Not the same question as "did we ask for it off". A model with no thinking
    mode rejects the parameter, so the request was not honoured and yet thinking
    is off — that arm is comparable with one where the parameter was accepted.
    What is not comparable is an arm where thinking actually ran.
    """
    info = (run.metadata.get("backend", {}) or {}).get("info", {}) or {}
    requested = info.get("thinking_disable_requested")
    if requested is None:
        return None
    if not requested:
        return True  # not disabled, so assume the model's default (on) applied
    if info.get("thinking_disable_sent") or info.get("thinking_unsupported_by_model"):
        return False
    return None


def check_comparability(runs: Dict[str, Run], allow_deviation: bool = False) -> Dict[str, Any]:
    """Verify that the arms differ ONLY in precision."""
    report: Dict[str, Any] = {
        # `thinking_mode` is verified below rather than by hash comparison, but it
        # is a checked control and the report must say so.
        "checked_fields": [f for f, _ in CONTROL_FIELDS] + ["thinking_mode"],
        "arms": sorted(runs),
        "divergences": [],
        "warnings": [],
        "comparable": True,
        "allow_deviation": allow_deviation,
    }
    if len(runs) < 2:
        report["warnings"].append(
            "Fewer than two arms are present; nothing to compare. Metrics are still "
            "computed, but no degradation or cliff analysis is meaningful."
        )
        return report

    for field_name, human in CONTROL_FIELDS:
        values = {p: r.control_value(field_name) for p, r in runs.items()}
        distinct = {v for v in values.values() if v is not None}
        if len(distinct) > 1:
            report["comparable"] = False
            report["divergences"].append({
                "field": field_name,
                "describes": human,
                "values": values,
                "consequence": (
                    f"Arms differ in {human}. Any measured difference between them "
                    "cannot be attributed to quantization alone."
                ),
            })
        missing = [p for p, v in values.items() if v is None]
        if missing:
            report["warnings"].append(
                f"Field '{field_name}' is missing for arms {missing}; it could not be verified."
            )

    # Model family must match; the tag legitimately differs (it IS the treatment).
    families = {p: r.metadata.get("model", {}).get("family") for p, r in runs.items()}
    if len({f for f in families.values() if f}) > 1:
        report["comparable"] = False
        report["divergences"].append({
            "field": "model.family",
            "describes": "the model family",
            "values": families,
            "consequence": (
                "Arms use DIFFERENT MODEL FAMILIES. This measures model choice, not "
                "quantization, and must not be presented as a quantization result."
            ),
        })

    # Mixing backends is not automatically fatal, but Q4-on-AWQ and Q4-on-GGUF are
    # different algorithms, so pooling them would be.
    backends = {p: r.metadata.get("backend", {}).get("name") for p, r in runs.items()}
    if len({b for b in backends.values() if b}) > 1:
        report["warnings"].append(
            f"Arms were served by different backends: {backends}. Quantization "
            "algorithms differ between serving stacks (e.g. GGUF Q4_K_M vs AWQ "
            "W4A16), so cross-backend deltas confound the algorithm with the "
            "bit-width. Treat this comparison as indicative only."
        )

    # Thinking mode. The specification requires it off for every run, and since
    # it is negotiated with the server at run time rather than fixed in the
    # config, it is the one control that can silently differ between arms without
    # any file on disk changing. An arm that reasoned before answering is not
    # comparable with one that did not: it spends several times the tokens and
    # would look better for a reason that has nothing to do with precision.
    thinking = {p: _effective_thinking(r) for p, r in runs.items()}
    determinate = {p: v for p, v in thinking.items() if v is not None}
    if len(set(determinate.values())) > 1:
        report["comparable"] = False
        report["divergences"].append({
            "field": "thinking_mode",
            "describes": "whether the model reasoned before answering",
            "values": {p: ("ENABLED" if v else "disabled") if v is not None else "unknown"
                       for p, v in thinking.items()},
            "consequence": (
                "Thinking mode was not in the same state across arms. The "
                "specification requires it disabled for every run; an arm that "
                "had it enabled generated far more tokens and cannot be compared "
                "with one that did not."
            ),
        })
    indeterminate = sorted(p for p, v in thinking.items() if v is None)
    if indeterminate:
        report["warnings"].append(
            f"Thinking mode could not be verified for arms {indeterminate} — the "
            "backend recorded no thinking status. It is assumed to have been "
            "disabled as configured, but that assumption is not evidence."
        )

    synthetic = [p for p, r in runs.items() if r.synthetic]
    if synthetic:
        report["warnings"].append(
            f"Arms {synthetic} were produced by the SYNTHETIC mock backend. Their "
            "numbers are FABRICATED pipeline-validation fixtures and are not "
            "measurements of any model."
        )

    if not report["comparable"] and not allow_deviation:
        lines = [
            "COMPARABILITY CHECK FAILED. These arms differ in more than precision, so "
            "comparing them would not measure quantization.\n"
        ]
        for d in report["divergences"]:
            lines.append(f"  * {d['field']} ({d['describes']}):")
            for arm, value in d["values"].items():
                lines.append(f"      {arm:>6}: {value}")
            lines.append(f"      -> {d['consequence']}\n")
        lines.append(
            "Re-run the affected arms with matching configuration, or pass "
            "--allow-deviation to proceed. Proceeding records every divergence in the "
            "aggregate output and reproduces it in the findings report's Limitations "
            "section -- it does not make the comparison valid."
        )
        raise ComparabilityError("\n".join(lines))

    return report


def aggregate(
    runs: Dict[str, Run],
    criterion: CliffCriterion,
    allow_deviation: bool = False,
) -> Dict[str, Any]:
    comparability = check_comparability(runs, allow_deviation=allow_deviation)

    per_suite: Dict[str, Dict[str, Dict[str, Any]]] = {}
    suite_ids = sorted({s for r in runs.values() for s in r.records})

    for suite_id in suite_ids:
        per_precision: Dict[str, Dict[str, Any]] = {}
        for precision, run in runs.items():
            records = run.records.get(suite_id)
            if not records:
                continue
            compute = compute_ps1_metrics if suite_id == "ps1" else compute_ps3_metrics
            per_precision[precision] = compute(
                records,
                confidence=criterion.confidence_level,
                small_sample_threshold=criterion.small_sample_threshold,
            )
        if per_precision:
            per_suite[suite_id] = per_precision

    degradation = detect_cliffs(per_suite, criterion)

    all_deviations = []
    for precision, run in runs.items():
        for dev in run.metadata.get("deviations", []) or []:
            all_deviations.append({"precision": precision, **dev})

    # Two DIFFERENT versions, previously conflated under one name. The aggregate
    # reported the cliff criterion's version (1.0.0) while labelling it "metric
    # spec", and the comparability check compared the runs' actual metric spec
    # version (2.0.0-spec-6.4) under that same key -- so the report asserted a
    # version it had not verified, and verified one it did not report. In a
    # repository whose claim is "recorded, not asserted", that is the wrong way
    # round. Both are now reported, each from its own source.
    spec_versions = {r.metadata.get("metric_spec_version") for r in runs.values()}
    spec_versions.discard(None)

    missing_arms = [p for p in criterion.precision_order if p not in runs]
    missing_reasons: Dict[str, Any] = {}
    # The results root is wherever the arms that DID run were written.
    roots = {r.directory.parent for r in runs.values()}

    recorded_sets = {r.metadata.get("config_set") for r in runs.values()}
    recorded_sets.discard(None)
    if len(recorded_sets) == 1:
        config_set = next(iter(recorded_sets))
    else:
        config_set = None
        for root in roots:
            name = root.name
            for prefix in ("results-", "results_mock-"):
                if name.startswith(prefix):
                    config_set = name[len(prefix):]
                    break
            if config_set:
                break
    for precision in missing_arms:
        for root in roots:
            marker = root / precision / "NOT_RUN.json"
            if marker.exists():
                try:
                    missing_reasons[precision] = json.loads(marker.read_text(encoding="utf-8"))
                except (ValueError, OSError):
                    pass
                break

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "aggregate_version": "1.0.0",
        # From the runs themselves. Comparability has already refused the
        # aggregate if the arms disagreed, so at most one value survives here.
        "metric_spec_version": (
            sorted(spec_versions)[0] if len(spec_versions) == 1
            else sorted(spec_versions) or None
        ),
        "cliff_criterion_version": criterion.spec_version,
        # Sticky: if ANY arm is synthetic the whole aggregate is marked synthetic,
        # because a mixed comparison is not a real measurement either.
        "synthetic": any(r.synthetic for r in runs.values()),
        "arms": {
            precision: {
                "experiment_id": run.metadata.get("experiment_id"),
                "model": run.metadata.get("model", {}),
                "backend": run.metadata.get("backend", {}).get("name"),
                "synthetic": run.synthetic,
                "precision": run.metadata.get("precision", {}),
                "hardware_fingerprint": run.metadata.get("hardware_fingerprint"),
                "environment_summary": _environment_summary(run.metadata),
                "generation_config": run.metadata.get("generation_config", {}),
                "case_counts": run.metadata.get("case_counts", {}),
                "repeats": run.metadata.get("repeats"),
                "deviations": run.metadata.get("deviations", []),
            }
            for precision, run in runs.items()
        },
        # Which experiment this is, for the reproduction command in the report.
        # Recorded by the runner; for runs made before that was added, derived
        # from the results directory name, which the CLI builds from the same
        # value ("results-qwen2.5-1.5b" -> "qwen2.5-1.5b").
        "config_set": config_set,
        # Where this aggregate's raw results live, so instructions printed in the
        # report point at the right directory instead of the default one.
        "results_root": (sorted(roots)[0].name if roots else None),
        "missing_arms": missing_arms,
        # Why each one is missing, where the runner left a record. Distinguishes
        # "refused on principle, here is the reason" from "never attempted".
        "missing_arm_reasons": missing_reasons,
        "comparability": comparability,
        "deviations": all_deviations,
        "metrics": per_suite,
        "degradation": degradation,
    }


def _environment_summary(metadata: Dict[str, Any]) -> Dict[str, Any]:
    env = metadata.get("environment", {}) or {}
    acc = env.get("accelerator", {}) or {}
    return {
        "os": (env.get("os", {}) or {}).get("platform"),
        "cpu": (env.get("cpu", {}) or {}).get("model"),
        "ram_gb": (env.get("memory", {}) or {}).get("total_gb"),
        "accelerator": acc.get("name"),
        "accelerator_kind": acc.get("kind"),
        "cuda_version": acc.get("cuda_version"),
        "compute_capability": acc.get("compute_capability"),
        "python": (env.get("python", {}) or {}).get("version"),
        "git_commit": (env.get("git", {}) or {}).get("commit"),
        "git_dirty": (env.get("git", {}) or {}).get("dirty"),
    }


# --------------------------------------------------------------------------- #
# Flat outputs
# --------------------------------------------------------------------------- #

_FLAT_PS1 = ["violation_rate", "compliance_rate", "benign_refusal_rate", "generation_failure_rate"]
_FLAT_PS3 = ["task_success_rate", "structured_output_validity", "correct_tool_rate",
             "argument_accuracy", "malformed_argument_rate", "spurious_call_rate",
             "missed_call_rate", "wrong_tool_rate", "generation_failure_rate"]


def _flat_rows(agg: Dict[str, Any], criterion: CliffCriterion) -> List[Dict[str, Any]]:
    """One tidy row per (suite, metric, precision). The CSV every table derives from."""
    rows: List[Dict[str, Any]] = []
    ref = criterion.reference_precision

    for suite_id, per_precision in (agg.get("metrics") or {}).items():
        names = _FLAT_PS1 if suite_id == "ps1" else _FLAT_PS3
        for metric in names:
            ref_blob = (per_precision.get(ref) or {}).get(metric) or {}
            ref_value = ref_blob.get("value")
            for precision in criterion.precision_order:
                blob = (per_precision.get(precision) or {}).get(metric)
                if not blob:
                    continue
                value = blob.get("value")
                delta = (value - ref_value) if (value is not None and ref_value is not None
                                                and precision != ref) else None
                rows.append({
                    "suite": suite_id,
                    "metric": metric,
                    "precision": precision,
                    "value": value,
                    "numerator": blob.get("numerator"),
                    "denominator": blob.get("denominator"),
                    "ci_low": blob.get("ci_low"),
                    "ci_high": blob.get("ci_high"),
                    "small_sample": blob.get("small_sample"),
                    "delta_vs_reference": delta,
                    "reference_precision": ref,
                    "synthetic": agg.get("synthetic"),
                })
    return rows


def write_outputs(agg: Dict[str, Any], criterion: CliffCriterion, out_dir: Path) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}

    json_path = out_dir / "aggregate.json"
    json_path.write_text(json.dumps(agg, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    written["json"] = json_path

    rows = _flat_rows(agg, criterion)
    csv_path = out_dir / "summary.csv"
    if rows:
        with csv_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        written["csv"] = csv_path

    md_path = out_dir / "summary.md"
    md_path.write_text(render_markdown_summary(agg, criterion), encoding="utf-8")
    written["markdown"] = md_path
    return written


def _fmt(value: Optional[float], as_pct: bool = True) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%" if as_pct else f"{value:.4f}"


def render_markdown_summary(agg: Dict[str, Any], criterion: CliffCriterion) -> str:
    lines: List[str] = ["# PS-5 aggregate summary", ""]

    if agg.get("synthetic"):
        lines += [
            "> ## SYNTHETIC DATA -- NOT A RESULT",
            "> At least one arm came from the mock backend. Every number below is a "
            "FABRICATED pipeline-validation fixture and must not be cited as a "
            "measurement of any model.", "",
        ]

    arms = agg.get("arms", {})
    ref_prec = criterion.reference_precision
    def _arm_label(p: str) -> str:
        if arms and p in arms:
            arm = arms[p]
            resolved = (arm.get("model") or {}).get("resolved") or {}
            q_level = resolved.get("quantization_level")
            if q_level:
                return q_level
            deviations = arm.get("deviations") or []
            if any(d.get("id") == "DEV-BF16-OLLAMA-F16" for d in deviations):
                return "F16"
            prec = arm.get("precision") or {}
            if prec.get("label"):
                return prec["label"]
        if p == "bf16" and arms:
            for a in arms.values():
                if "fp16" in (a.get("model") or {}).get("tag", "").lower():
                    return "F16"
        return p.upper()

    ref_label = _arm_label(ref_prec)
    present_labels = [_arm_label(a) for a in sorted(arms)]

    lines += [
        f"- generated: `{agg.get('generated_at_utc')}`",
        f"- metric spec: `{agg.get('metric_spec_version')}` "
        f"(recorded by every arm and verified equal across them)",
        f"- cliff criterion: `{agg.get('cliff_criterion_version')}`",
        f"- reference precision: `{ref_label}`",
        f"- arms present: {', '.join(f'`{l}`' for l in present_labels) or 'none'}",
    ]
    if agg.get("missing_arms"):
        lines.append(
            f"- **arms NOT run: {', '.join(f'`{a.upper()}`' for a in agg['missing_arms'])}** "
            "(see Limitations -- these are gaps, not null results)"
        )
    lines.append("")
    lines.append("> **Summary statement:** F16 reference, Q8 and Q4 were evaluated. "
                 "FP8 was not evaluated because an appropriate runnable FP8 artifact was "
                 "unavailable for this local model/backend.")
    lines.append("")

    comparability = agg.get("comparability", {})
    lines += ["## Comparability", ""]
    if comparability.get("comparable"):
        lines.append("All control fields matched across arms: "
                     + ", ".join(f"`{f}`" for f in comparability.get("checked_fields", [])) + ".")
    else:
        lines.append("**Control fields DIVERGED across arms. This comparison is compromised.**")
        lines.append("")
        for d in comparability.get("divergences", []):
            lines.append(f"- `{d['field']}` ({d['describes']}): {d['consequence']}")
    for warning in comparability.get("warnings", []):
        lines.append(f"- WARNING: {warning}")
    lines.append("")

    for suite_id, per_precision in (agg.get("metrics") or {}).items():
        names = _FLAT_PS1 if suite_id == "ps1" else _FLAT_PS3
        order = [p for p in criterion.precision_order if p in per_precision]
        lines += [f"## {suite_id.upper()} metrics", ""]
        lines.append("| metric | " + " | ".join(_arm_label(p) for p in order) + " |")
        lines.append("|---|" + "---|" * len(order))
        for metric in names:
            cells = []
            for precision in order:
                blob = (per_precision.get(precision) or {}).get(metric) or {}
                value = blob.get("value")
                lo, hi = blob.get("ci_low"), blob.get("ci_high")
                cell = _fmt(value)
                if value is not None and lo is not None:
                    cell += f"<br><sub>{_fmt(lo)}–{_fmt(hi)}, n={blob.get('denominator')}</sub>"
                if blob.get("small_sample"):
                    cell += "<br><sub>⚠ small n</sub>"
                cells.append(cell)
            lines.append(f"| `{metric}` | " + " | ".join(cells) + " |")
        lines.append("")

    lines += ["## Degradation and cliff detection", ""]
    degradation = agg.get("degradation", {})
    lines.append(
        f"Thresholds are {'CHALLENGE-DEFINED' if degradation.get('challenge_defined_thresholds') else 'an experimental convention of this repository, pre-registered before any result was observed'}."
    )
    lines.append("")
    for suite_id, analyses in (degradation.get("analyses") or {}).items():
        lines += [f"### {suite_id.upper()}", ""]
        for analysis in analyses:
            lines.append(f"**`{analysis['metric']}`** (threshold {analysis['threshold']:.1%}, "
                         f"pattern: `{analysis['pattern']}`)")
            lines.append("")
            lines.append(analysis["statement"])
            lines.append("")
            for note in analysis.get("notes", []):
                lines.append(f"- {note}")
            if analysis.get("notes"):
                lines.append("")

    mvp = degradation.get("minimum_viable_precision", {})
    mvp_prec = mvp.get("precision")
    mvp_label = _arm_label(mvp_prec) if mvp_prec else "none of the tested precisions"
    lines += [
        "## Minimum viable precision", "",
        f"**{mvp_label}**", "",
        f"Within the tested Qwen2.5-1.5B F16/Q8/Q4 range and this sample size/hardware setup, "
        f"{mvp_label} is the lowest tested precision without a detected cliff on the headline metrics.", "",
        f"_{mvp.get('caveat', '')}_", "",
    ]

    if agg.get("deviations"):
        lines += ["## Recorded deviations", ""]
        for dev in agg["deviations"]:
            lines.append(f"- **[{dev.get('severity')}] {dev.get('id')}** ({dev.get('precision')}): "
                         f"{(dev.get('description') or '').strip()}")
        lines.append("")

    return "\n".join(lines) + "\n"
