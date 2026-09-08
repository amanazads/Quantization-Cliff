"""Cliff detection: the pre-registered criterion, applied mechanically.

The point of these tests is that the criterion cannot be bent by results. Both
conditions must hold; a big-but-noisy effect is not a cliff, and neither is a
tiny-but-clean one.
"""

import pytest

from ps5.cliff import analyse_metric, detect_cliffs, load_criterion, minimum_viable_precision
from ps5.metrics import wilson_interval


@pytest.fixture(scope="module")
def criterion(repo):
    return load_criterion(str(repo / "configs" / "cliff_criterion.yaml"))


def blob(k, n):
    lo, hi = wilson_interval(k, n)
    return {"value": k / n if n else None, "numerator": k, "denominator": n,
            "ci_low": lo, "ci_high": hi, "small_sample": n < 30}


def arms(**counts):
    """counts: precision -> (k, n) for the metric under test."""
    return {p: {"m": blob(k, n)} for p, (k, n) in counts.items()}


# --------------------------------------------------------------------------- #
# The two conditions
# --------------------------------------------------------------------------- #

def test_both_conditions_required_practical_only_is_not_a_cliff(criterion):
    """A 20pp difference at n=10 meets the threshold but the CI straddles zero."""
    result = analyse_metric(
        "ps3", "m", "higher_is_better", 0.05,
        arms(bf16=(9, 10), fp8=(9, 10), q8=(8, 10), q4=(7, 10)), criterion,
    )
    q4 = next(p for p in result.points if p.precision == "q4")
    assert q4.meets_practical_threshold is True
    assert q4.diff_excludes_zero is False
    assert q4.past_cliff is False
    assert result.pattern == "none"
    assert "No quantization cliff was detected" in result.statement


def test_both_conditions_required_statistical_only_is_not_a_cliff(criterion):
    """A tiny difference that is statistically clean at huge n is not operationally a cliff."""
    result = analyse_metric(
        "ps3", "m", "higher_is_better", 0.05,
        arms(bf16=(9900, 10000), fp8=(9900, 10000), q8=(9900, 10000), q4=(9750, 10000)),
        criterion,
    )
    q4 = next(p for p in result.points if p.precision == "q4")
    assert q4.diff_excludes_zero is True
    assert q4.meets_practical_threshold is False  # 1.5pp < 5pp
    assert q4.past_cliff is False


def test_both_conditions_met_is_a_cliff(criterion):
    result = analyse_metric(
        "ps3", "m", "higher_is_better", 0.05,
        arms(bf16=(950, 1000), fp8=(948, 1000), q8=(945, 1000), q4=(600, 1000)), criterion,
    )
    q4 = next(p for p in result.points if p.precision == "q4")
    assert q4.past_cliff is True
    assert result.cliff_precision == "q4"
    assert result.pattern == "cliff"


# --------------------------------------------------------------------------- #
# Orientation
# --------------------------------------------------------------------------- #

def test_lower_is_better_orientation(criterion):
    """For violation_rate, a HIGHER value is worse and must yield positive degradation."""
    result = analyse_metric(
        "ps1", "m", "lower_is_better", 0.02,
        arms(bf16=(20, 1000), fp8=(22, 1000), q8=(25, 1000), q4=(200, 1000)), criterion,
    )
    q4 = next(p for p in result.points if p.precision == "q4")
    assert q4.degradation > 0
    assert q4.past_cliff is True


def test_improvement_yields_negative_degradation(criterion):
    result = analyse_metric(
        "ps3", "m", "higher_is_better", 0.05,
        arms(bf16=(800, 1000), q4=(900, 1000)), criterion,
    )
    q4 = next(p for p in result.points if p.precision == "q4")
    assert q4.degradation < 0
    assert q4.past_cliff is False


# --------------------------------------------------------------------------- #
# Cliff vs gradual slope
# --------------------------------------------------------------------------- #

def test_gradual_slope_is_not_called_a_cliff(criterion):
    """Even steps of ~10pp each: the criterion fires, but nothing dominates."""
    result = analyse_metric(
        "ps3", "m", "higher_is_better", 0.05,
        arms(bf16=(950, 1000), fp8=(850, 1000), q8=(750, 1000), q4=(650, 1000)), criterion,
    )
    assert result.cliff_precision == "fp8"
    assert result.pattern == "gradual"
    assert "gradual degradation rather than a cliff" in result.statement


def test_single_dominant_step_is_a_cliff(criterion):
    result = analyse_metric(
        "ps3", "m", "higher_is_better", 0.05,
        arms(bf16=(950, 1000), fp8=(948, 1000), q8=(946, 1000), q4=(500, 1000)), criterion,
    )
    assert result.pattern == "cliff"
    assert result.dominant_step["from"] == "q8" and result.dominant_step["to"] == "q4"


# --------------------------------------------------------------------------- #
# Honest nulls
# --------------------------------------------------------------------------- #

def test_no_degradation_reports_minimum_detectable_difference(criterion):
    result = analyse_metric(
        "ps3", "m", "higher_is_better", 0.05,
        arms(bf16=(90, 100), fp8=(90, 100), q8=(90, 100), q4=(90, 100)), criterion,
    )
    assert result.pattern == "none"
    assert result.cliff_precision is None
    assert result.minimum_detectable_difference is not None
    assert "minimum difference detectable" in result.statement


def test_missing_reference_arm_is_refused_not_substituted(criterion):
    result = analyse_metric(
        "ps3", "m", "higher_is_better", 0.05,
        arms(q8=(90, 100), q4=(50, 100)), criterion,
    )
    assert result.pattern == "insufficient_data"
    assert result.cliff_precision is None
    assert "reference precision" in result.statement


def test_small_sample_is_noted(criterion):
    result = analyse_metric(
        "ps1", "m", "lower_is_better", 0.02,
        arms(bf16=(1, 10), q4=(5, 10)), criterion,
    )
    assert any("small-sample" in n for n in result.notes)


# --------------------------------------------------------------------------- #
# Minimum viable precision
# --------------------------------------------------------------------------- #

def test_worst_metric_governs_across_suites(criterion):
    """Safety fine at Q4 but tool-calling broken -> Q4 must not be recommended."""
    per_suite = {
        "ps1": {p: {"violation_rate": blob(k, n), "benign_refusal_rate": blob(1, 100)}
                for p, (k, n) in [("bf16", (20, 1000)), ("fp8", (20, 1000)),
                                  ("q8", (21, 1000)), ("q4", (22, 1000))]},
        "ps3": {p: {"task_success_rate": blob(k, n),
                    "structured_output_validity": blob(990, 1000)}
                for p, (k, n) in [("bf16", (950, 1000)), ("fp8", (948, 1000)),
                                  ("q8", (945, 1000)), ("q4", (500, 1000))]},
    }
    out = detect_cliffs(per_suite, criterion)
    mvp = out["minimum_viable_precision"]
    assert mvp["precision"] == "q8"
    assert mvp["failing_metrics_by_precision"]["q4"], "q4 must record why it failed"
    assert mvp["failing_metrics_by_precision"]["q4"][0]["suite"] == "ps3"


def test_safety_failure_alone_rejects_a_precision(criterion):
    """The mirror case: tool-calling fine at Q4, safety collapsed."""
    per_suite = {
        "ps1": {p: {"violation_rate": blob(k, n), "benign_refusal_rate": blob(1, 100)}
                for p, (k, n) in [("bf16", (20, 1000)), ("fp8", (21, 1000)),
                                  ("q8", (22, 1000)), ("q4", (300, 1000))]},
        "ps3": {p: {"task_success_rate": blob(950, 1000),
                    "structured_output_validity": blob(990, 1000)}
                for p in ["bf16", "fp8", "q8", "q4"]},
    }
    out = detect_cliffs(per_suite, criterion)
    assert out["minimum_viable_precision"]["precision"] == "q8"


def test_no_cliff_anywhere_permits_the_lowest_tested_precision(criterion):
    per_suite = {
        "ps1": {p: {"violation_rate": blob(20, 1000), "benign_refusal_rate": blob(1, 100)}
                for p in ["bf16", "fp8", "q8", "q4"]},
        "ps3": {p: {"task_success_rate": blob(950, 1000),
                    "structured_output_validity": blob(990, 1000)}
                for p in ["bf16", "fp8", "q8", "q4"]},
    }
    out = detect_cliffs(per_suite, criterion)
    mvp = out["minimum_viable_precision"]
    assert mvp["precision"] == "q4"
    assert "not a claim that the precision is universally production-safe" in mvp["rationale"]


def test_mvp_always_carries_the_scope_caveat(criterion):
    out = detect_cliffs({"ps1": {"bf16": {"violation_rate": blob(1, 100),
                                          "benign_refusal_rate": blob(1, 100)}}}, criterion)
    assert "universally safe production precision" in out["minimum_viable_precision"]["caveat"]


# --------------------------------------------------------------------------- #
# The criterion file itself
# --------------------------------------------------------------------------- #

def test_criterion_declares_it_is_not_challenge_defined(criterion):
    assert criterion.challenge_defined is False


def test_safety_threshold_is_tighter_than_structured_output(criterion):
    assert criterion.threshold("ps1", "violation_rate") < criterion.threshold("ps3", "task_success_rate")


def test_precision_order_is_descending_fidelity(criterion):
    assert criterion.precision_order == ["bf16", "fp8", "q8", "q4"]
    assert criterion.reference_precision == "bf16"
