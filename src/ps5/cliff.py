"""Quantization-cliff detection.

Mechanically applies the criterion pre-registered in configs/cliff_criterion.yaml
and documented in docs/METRICS.md section 4. There is no threshold search, no
post-hoc tuning, and no path by which "no cliff found" can be turned into a
cliff: the thresholds are read from the frozen config and the code cannot lower
them.

The three outcomes are all first-class results:
  * a cliff (one dominant step),
  * gradual degradation (criterion fires, no dominant step),
  * no detected degradation (criterion never fires).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

from .metrics import (
    minimum_detectable_difference,
    newcombe_difference_interval,
    wilson_interval,
)

__all__ = [
    "CliffCriterion",
    "MetricDegradation",
    "PrecisionPoint",
    "load_criterion",
    "analyse_metric",
    "detect_cliffs",
    "minimum_viable_precision",
]


@dataclass
class CliffCriterion:
    spec_version: str
    precision_order: List[str]
    reference_precision: str
    confidence_level: float
    small_sample_threshold: int
    relative_degradation_min_base: float
    cliff_dominance_factor: float
    headline_metrics: Dict[str, List[Dict[str, Any]]]
    secondary_metrics: Dict[str, List[str]]
    challenge_defined: bool
    raw: Dict[str, Any] = field(default_factory=dict)

    def headline_for(self, suite: str) -> List[Dict[str, Any]]:
        return self.headline_metrics.get(suite, [])

    def threshold(self, suite: str, metric: str) -> Optional[float]:
        for spec in self.headline_for(suite):
            if spec["metric"] == metric:
                return float(spec["threshold"])
        return None


def load_criterion(path: str) -> CliffCriterion:
    with open(path, encoding="utf-8") as fh:
        doc = yaml.safe_load(fh) or {}
    return CliffCriterion(
        spec_version=doc.get("spec_version", "unknown"),
        precision_order=list(doc.get("precision_order", ["bf16", "fp8", "q8", "q4"])),
        reference_precision=doc.get("reference_precision", "bf16"),
        confidence_level=float(doc.get("confidence_level", 0.95)),
        small_sample_threshold=int(doc.get("small_sample_threshold", 30)),
        relative_degradation_min_base=float(doc.get("relative_degradation_min_base", 0.10)),
        cliff_dominance_factor=float(doc.get("cliff_dominance_factor", 2.0)),
        headline_metrics=doc.get("headline_metrics", {}) or {},
        secondary_metrics=doc.get("secondary_metrics", {}) or {},
        challenge_defined=bool(doc.get("challenge_defined", False)),
        raw=doc,
    )


@dataclass
class PrecisionPoint:
    """One precision's measurement of one metric, with its delta vs reference."""

    precision: str
    value: Optional[float]
    numerator: int
    denominator: int
    ci_low: float
    ci_high: float
    small_sample: bool
    is_reference: bool = False
    delta_vs_reference: Optional[float] = None
    degradation: Optional[float] = None
    relative_degradation: Optional[float] = None
    diff_ci_low: Optional[float] = None
    diff_ci_high: Optional[float] = None
    diff_excludes_zero: Optional[bool] = None
    meets_practical_threshold: Optional[bool] = None
    past_cliff: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MetricDegradation:
    suite: str
    metric: str
    direction: str
    threshold: float
    reference_precision: str
    points: List[PrecisionPoint]
    cliff_precision: Optional[str]
    pattern: str  # "cliff" | "gradual" | "none" | "insufficient_data"
    dominant_step: Optional[Dict[str, Any]] = None
    steps: List[Dict[str, Any]] = field(default_factory=list)
    minimum_detectable_difference: Optional[float] = None
    statement: str = ""
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["points"] = [p.to_dict() for p in self.points]
        return d


def _extract(metrics_blob: Dict[str, Any], metric: str) -> Optional[Dict[str, Any]]:
    """Pull a metric dict out of a suite metrics blob, supporting `a.b` paths."""
    node: Any = metrics_blob
    for part in metric.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node if isinstance(node, dict) and "numerator" in node else None


def analyse_metric(
    suite: str,
    metric: str,
    direction: str,
    threshold: float,
    per_precision_metrics: Dict[str, Dict[str, Any]],
    criterion: CliffCriterion,
) -> MetricDegradation:
    """Apply the pre-registered criterion to one metric across all precisions."""
    ref = criterion.reference_precision
    order = [p for p in criterion.precision_order if p in per_precision_metrics]

    ref_blob = _extract(per_precision_metrics.get(ref, {}), metric)
    if ref_blob is None:
        return MetricDegradation(
            suite=suite, metric=metric, direction=direction, threshold=threshold,
            reference_precision=ref, points=[], cliff_precision=None,
            pattern="insufficient_data",
            statement=(
                f"Cannot analyse `{metric}`: the reference precision '{ref}' is "
                "missing from this comparison set. No delta can be computed "
                "without it, and substituting another arm as the reference would "
                "silently change the question being asked."
            ),
            notes=[f"reference precision '{ref}' absent"],
        )

    ref_k, ref_n = int(ref_blob["numerator"]), int(ref_blob["denominator"])
    ref_value = ref_blob.get("value")
    points: List[PrecisionPoint] = []

    for precision in order:
        blob = _extract(per_precision_metrics.get(precision, {}), metric)
        if blob is None:
            continue
        k, n = int(blob["numerator"]), int(blob["denominator"])
        lo, hi = blob.get("ci_low", 0.0), blob.get("ci_high", 1.0)
        value = blob.get("value")

        point = PrecisionPoint(
            precision=precision, value=value, numerator=k, denominator=n,
            ci_low=lo, ci_high=hi,
            small_sample=bool(blob.get("small_sample", n < criterion.small_sample_threshold)),
            is_reference=(precision == ref),
        )

        if precision != ref and value is not None and ref_value is not None:
            delta = value - ref_value
            # Orient so POSITIVE always means WORSE, whatever the metric's direction.
            degradation = delta if direction == "lower_is_better" else -delta
            point.delta_vs_reference = delta
            point.degradation = degradation

            if ref_value >= criterion.relative_degradation_min_base:
                point.relative_degradation = degradation / ref_value

            d_lo, d_hi = newcombe_difference_interval(
                k, n, ref_k, ref_n, criterion.confidence_level
            )
            point.diff_ci_low, point.diff_ci_high = d_lo, d_hi
            point.diff_excludes_zero = bool(d_lo > 0.0 or d_hi < 0.0)
            point.meets_practical_threshold = bool(degradation >= threshold)
            # BOTH conditions required. See docs/METRICS.md section 4.4.
            point.past_cliff = bool(point.meets_practical_threshold and point.diff_excludes_zero)

        points.append(point)

    # Cliff point = highest-fidelity precision satisfying the criterion.
    cliff_precision = next((p.precision for p in points if p.past_cliff), None)

    # Step-by-step degradation, for cliff-vs-slope.
    steps: List[Dict[str, Any]] = []
    for prev, curr in zip(points, points[1:]):
        prev_d = prev.degradation if prev.degradation is not None else 0.0
        curr_d = curr.degradation if curr.degradation is not None else 0.0
        steps.append({
            "from": prev.precision, "to": curr.precision,
            "step_degradation": curr_d - prev_d,
        })

    pattern = "none"
    dominant: Optional[Dict[str, Any]] = None
    notes: List[str] = []

    if cliff_precision is not None:
        pattern = "gradual"
        if steps:
            dominant = max(steps, key=lambda s: s["step_degradation"])
            others = [s["step_degradation"] for s in steps if s is not dominant]
            mean_others = sum(others) / len(others) if others else 0.0
            big_enough = dominant["step_degradation"] >= threshold
            if big_enough and (
                mean_others <= 0
                or dominant["step_degradation"] >= criterion.cliff_dominance_factor * mean_others
            ):
                pattern = "cliff"
            else:
                notes.append(
                    f"largest step ({dominant['from']}->{dominant['to']}: "
                    f"{dominant['step_degradation']:+.4f}) does not dominate the mean of "
                    f"the other steps ({mean_others:+.4f}) by the pre-registered factor "
                    f"of {criterion.cliff_dominance_factor}x, so this is graded as gradual "
                    "degradation rather than a cliff."
                )

    mdd = minimum_detectable_difference(ref_value or 0.0, ref_n) if ref_n else None

    if pattern == "none":
        statement = (
            f"No quantization cliff was detected for `{metric}` within the tested "
            f"precision range, at a threshold of {threshold:.1%} and "
            f"{criterion.confidence_level:.0%} confidence, with n={ref_n} at the "
            f"reference precision. "
            + (f"The minimum difference detectable at this sample size was "
               f"{mdd:.1%}; a real effect smaller than that would not have been "
               f"visible to this experiment." if mdd else
               "The minimum detectable difference could not be computed.")
        )
    elif pattern == "cliff":
        statement = (
            f"A cliff was detected for `{metric}` at **{cliff_precision}**. "
            f"Degradation crosses the pre-registered {threshold:.1%} threshold with a "
            f"difference interval excluding zero, and the step "
            f"{dominant['from']}->{dominant['to']} ({dominant['step_degradation']:+.1%}) "
            f"dominates the other steps by at least "
            f"{criterion.cliff_dominance_factor}x."
        )
    else:
        statement = (
            f"Degradation on `{metric}` crosses the pre-registered {threshold:.1%} "
            f"threshold at **{cliff_precision}**, but no single step dominates, so the "
            "pattern is graded as gradual degradation rather than a cliff."
        )

    if any(p.small_sample for p in points):
        notes.append(
            "At least one precision's cell is below the pre-registered small-sample "
            f"threshold of n={criterion.small_sample_threshold}; those cells are "
            "directional only and no significance is claimed for them."
        )

    return MetricDegradation(
        suite=suite, metric=metric, direction=direction, threshold=threshold,
        reference_precision=ref, points=points, cliff_precision=cliff_precision,
        pattern=pattern, dominant_step=dominant, steps=steps,
        minimum_detectable_difference=mdd, statement=statement, notes=notes,
    )


def detect_cliffs(
    per_suite_per_precision: Dict[str, Dict[str, Dict[str, Any]]],
    criterion: CliffCriterion,
) -> Dict[str, Any]:
    """Run the criterion over every headline metric in every suite."""
    analyses: Dict[str, List[MetricDegradation]] = {}
    for suite, per_precision in per_suite_per_precision.items():
        results: List[MetricDegradation] = []
        for spec in criterion.headline_for(suite):
            results.append(
                analyse_metric(
                    suite=suite,
                    metric=spec["metric"],
                    direction=spec.get("direction", "higher_is_better"),
                    threshold=float(spec["threshold"]),
                    per_precision_metrics=per_precision,
                    criterion=criterion,
                )
            )
        analyses[suite] = results

    mvp = minimum_viable_precision(analyses, criterion)
    return {
        "criterion_spec_version": criterion.spec_version,
        "challenge_defined_thresholds": criterion.challenge_defined,
        "reference_precision": criterion.reference_precision,
        "precision_order": criterion.precision_order,
        "analyses": {s: [a.to_dict() for a in v] for s, v in analyses.items()},
        "minimum_viable_precision": mvp,
    }


def minimum_viable_precision(
    analyses: Dict[str, List[MetricDegradation]],
    criterion: CliffCriterion,
) -> Dict[str, Any]:
    """Lowest-fidelity precision not past the cliff on ANY headline metric.

    Computed, never chosen. The worst metric governs, so a precision that is safe
    but structurally broken -- or structurally sound but unsafe -- does not
    qualify. This is the mechanism that stops safety and tool-calling being
    averaged against each other.
    """
    evaluated = {
        p for suite in analyses.values() for a in suite for p in
        [pt.precision for pt in a.points]
    }
    order = [p for p in criterion.precision_order if p in evaluated]

    failing: Dict[str, List[Dict[str, Any]]] = {p: [] for p in order}
    for suite, metric_analyses in analyses.items():
        for analysis in metric_analyses:
            for point in analysis.points:
                if point.past_cliff:
                    failing[point.precision].append({
                        "suite": suite,
                        "metric": analysis.metric,
                        "degradation": point.degradation,
                        "threshold": analysis.threshold,
                        "diff_ci": [point.diff_ci_low, point.diff_ci_high],
                    })

    acceptable = [p for p in order if not failing.get(p)]
    chosen = acceptable[-1] if acceptable else None

    if chosen is None:
        rationale = (
            "Every tested precision, including the reference, was past the cliff on "
            "at least one headline metric. That is an unusual result and most often "
            "means the comparison set is malformed rather than that the reference is "
            "broken. Investigate before reporting."
        )
    elif chosen == criterion.reference_precision:
        rationale = (
            f"Only the reference precision ({chosen}) cleared every headline metric. "
            "On the evidence available, no quantized arm is supported for production "
            "collections use."
        )
    else:
        rationale = (
            f"{chosen} is the lowest-fidelity precision that cleared every headline "
            "metric in both suites under the pre-registered criterion. This is a "
            "statement about THIS suite, THIS model and THIS hardware at THIS sample "
            "size. It is not a claim that the precision is universally production-safe."
        )

    return {
        "precision": chosen,
        "rule": "worst_metric_governs",
        "acceptable_precisions": acceptable,
        "failing_metrics_by_precision": failing,
        "rationale": rationale,
        "caveat": (
            "'Minimum precision supported by this experiment' and 'universally safe "
            "production precision' are different claims. This experiment can only "
            "support the first."
        ),
    }
