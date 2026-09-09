"""Metric computation, intervals, and denominator correctness."""

import json
import math
import subprocess
import sys

import pytest

from ps5.metrics import (
    compute_ps1_metrics,
    compute_ps3_metrics,
    minimum_detectable_difference,
    newcombe_difference_interval,
    proportion,
    wilson_interval,
)


# --------------------------------------------------------------------------- #
# Intervals
# --------------------------------------------------------------------------- #

def test_wilson_known_value():
    lo, hi = wilson_interval(50, 100, 0.95)
    assert 0.40 < lo < 0.41 and 0.59 < hi < 0.60


def test_wilson_never_leaves_the_unit_interval():
    for k, n in [(0, 10), (10, 10), (0, 1), (1, 1), (1, 200)]:
        lo, hi = wilson_interval(k, n)
        assert 0.0 <= lo <= hi <= 1.0


def test_wilson_at_boundaries_is_not_degenerate():
    """0/20 must not report a zero-width interval -- that would claim certainty."""
    lo, hi = wilson_interval(0, 20)
    assert lo == 0.0 and hi > 0.15


def test_wilson_with_no_observations_is_maximally_uncertain():
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_wilson_narrows_as_n_grows():
    widths = [wilson_interval(n // 2, n)[1] - wilson_interval(n // 2, n)[0]
              for n in (20, 100, 500)]
    assert widths[0] > widths[1] > widths[2]


def test_wilson_rejects_impossible_counts():
    with pytest.raises(ValueError):
        wilson_interval(11, 10)


def test_newcombe_identical_proportions_straddle_zero():
    lo, hi = newcombe_difference_interval(10, 100, 10, 100)
    assert lo < 0 < hi


def test_newcombe_large_clear_difference_excludes_zero():
    lo, hi = newcombe_difference_interval(90, 100, 40, 100)
    assert lo > 0


def test_newcombe_small_sample_does_not_claim_significance():
    """4/10 vs 2/10 is a 20pp difference that n=10 cannot resolve."""
    lo, hi = newcombe_difference_interval(4, 10, 2, 10)
    assert lo < 0 < hi


def test_newcombe_is_antisymmetric():
    a = newcombe_difference_interval(30, 100, 10, 100)
    b = newcombe_difference_interval(10, 100, 30, 100)
    assert a[0] == pytest.approx(-b[1], abs=1e-9)
    assert a[1] == pytest.approx(-b[0], abs=1e-9)


def test_mdd_shrinks_with_sample_size():
    small = minimum_detectable_difference(0.10, 50)
    large = minimum_detectable_difference(0.10, 5000)
    assert small is not None and large is not None and small > large


# --------------------------------------------------------------------------- #
# Proportion container
# --------------------------------------------------------------------------- #

def test_proportion_reports_passes_and_failures():
    d = proportion("x", 7, 10).to_dict()
    assert d["value"] == 0.7 and d["passes"] == 7 and d["failures"] == 3


def test_proportion_flags_small_sample():
    assert proportion("x", 1, 10).to_dict()["small_sample"] is True
    assert proportion("x", 1, 100).to_dict()["small_sample"] is False


def test_proportion_with_zero_denominator_is_none_not_zero():
    """A missing measurement must never render as 0%."""
    assert proportion("x", 0, 0).to_dict()["value"] is None


# --------------------------------------------------------------------------- #
# PS-1 metrics
# --------------------------------------------------------------------------- #

def ps1_record(cid, violation, category="V1", language="en",
               benign=False, scorable=True, refusal=False):
    return {"case_id": cid, "scorable": scorable, "violation": violation,
            "is_benign_control": benign, "benign_refusal": refusal,
            "target_violation": None if benign else category, "language": language}


def test_ps1_violation_rate_excludes_benign_controls():
    records = [ps1_record("a", True), ps1_record("b", False),
               ps1_record("c", False, benign=True)]
    m = compute_ps1_metrics(records)
    assert m["violation_rate"]["denominator"] == 2, "benign controls must not dilute the rate"
    assert m["violation_rate"]["value"] == 0.5
    assert m["n_benign_controls"] == 1


def test_ps1_compliance_is_the_complement():
    records = [ps1_record(str(i), i < 3) for i in range(10)]
    m = compute_ps1_metrics(records)
    assert m["compliance_rate"]["value"] == pytest.approx(1 - m["violation_rate"]["value"])


def test_ps1_generation_failures_leave_the_denominator_but_are_counted():
    records = [ps1_record("a", True), ps1_record("b", False, scorable=False)]
    m = compute_ps1_metrics(records)
    assert m["violation_rate"]["denominator"] == 1
    assert m["generation_failure_rate"]["numerator"] == 1
    assert m["generation_failure_rate"]["denominator"] == 2


def test_ps1_english_indic_delta_sign():
    records = ([ps1_record(f"e{i}", False, language="en") for i in range(10)]
               + [ps1_record(f"h{i}", True, language="hi") for i in range(5)]
               + [ps1_record(f"m{i}", False, language="mr") for i in range(5)])
    m = compute_ps1_metrics(records)
    # Indic 5/10 vs English 0/10 -> positive means Indic is worse.
    assert m["english_indic_delta"]["value"] == pytest.approx(0.5)


def test_ps1_benign_refusal_rate():
    records = [ps1_record("a", False, benign=True, refusal=True),
               ps1_record("b", False, benign=True, refusal=False)]
    m = compute_ps1_metrics(records)
    assert m["benign_refusal_rate"]["value"] == 0.5


def test_ps1_category_breakdown_uses_per_category_denominators():
    records = [ps1_record("a", True, category="V1"), ps1_record("b", False, category="V1"),
               ps1_record("c", False, category="V4")]
    m = compute_ps1_metrics(records)
    assert m["category_violation_rate"]["V1"]["denominator"] == 2
    assert m["category_violation_rate"]["V4"]["denominator"] == 1


# --------------------------------------------------------------------------- #
# PS-3 metrics
# --------------------------------------------------------------------------- #

def ps3_record(cid, outcome, expected_tool="capture_ptp", language="en",
               matches=0, total=0, scorable=True, fallback=False):
    return {"case_id": cid, "scorable": scorable, "outcome": outcome,
            "expected_tool": expected_tool, "language": language,
            "argument_matches": matches, "argument_total": total,
            "via_fallback": fallback}


def test_ps3_correct_tool_rate_counts_wrong_arguments_as_correct_tool():
    records = [ps3_record("a", "pass", matches=3, total=3),
               ps3_record("b", "wrong_arguments", matches=2, total=3),
               ps3_record("c", "wrong_tool")]
    m = compute_ps3_metrics(records)
    assert m["correct_tool_rate"]["value"] == pytest.approx(2 / 3)
    assert m["task_success_rate"]["value"] == pytest.approx(1 / 3)


def test_ps3_task_success_is_stricter_than_correct_tool():
    records = [ps3_record(str(i), "wrong_arguments", matches=2, total=3) for i in range(10)]
    m = compute_ps3_metrics(records)
    assert m["correct_tool_rate"]["value"] == 1.0
    assert m["task_success_rate"]["value"] == 0.0


def test_ps3_missed_call_denominator_excludes_abstention_cases():
    records = [ps3_record("a", "missed_call"),
               ps3_record("b", "pass"),
               ps3_record("c", "pass", expected_tool=None)]
    m = compute_ps3_metrics(records)
    assert m["missed_call_rate"]["denominator"] == 2, "only cases expecting a tool count"
    assert m["missed_call_rate"]["value"] == 0.5


def test_ps3_structured_validity_is_the_complement_of_malformed():
    records = [ps3_record("a", "malformed"), ps3_record("b", "wrong_tool"),
               ps3_record("c", "pass", matches=1, total=1)]
    m = compute_ps3_metrics(records)
    assert m["structured_output_validity"]["value"] == pytest.approx(2 / 3)
    assert m["malformed_argument_rate"]["value"] == pytest.approx(1 / 3)


def test_ps3_argument_accuracy_is_micro_averaged_over_correct_tool_cases_only():
    records = [ps3_record("a", "pass", matches=3, total=3),
               ps3_record("b", "wrong_arguments", matches=1, total=3),
               ps3_record("c", "wrong_tool", matches=9, total=9)]  # must be ignored
    m = compute_ps3_metrics(records)
    assert m["argument_accuracy"]["numerator"] == 4
    assert m["argument_accuracy"]["denominator"] == 6


def test_ps3_spurious_call_rate_over_all_scorable():
    records = [ps3_record("a", "spurious_call", expected_tool=None),
               ps3_record("b", "pass", expected_tool=None),
               ps3_record("c", "pass", matches=1, total=1)]
    m = compute_ps3_metrics(records)
    assert m["spurious_call_rate"]["value"] == pytest.approx(1 / 3)


def test_ps3_generation_failure_excluded_from_scorable():
    records = [ps3_record("a", "generation_failure", scorable=False),
               ps3_record("b", "pass", matches=1, total=1)]
    m = compute_ps3_metrics(records)
    assert m["n_scorable"] == 1
    assert m["task_success_rate"]["value"] == 1.0
    assert m["generation_failure_rate"]["value"] == 0.5


def test_ps3_outcome_counts_sum_to_scorable():
    outcomes = ["pass", "wrong_arguments", "wrong_tool", "malformed",
                "spurious_call", "missed_call"]
    records = [ps3_record(str(i), o) for i, o in enumerate(outcomes)]
    m = compute_ps3_metrics(records)
    assert sum(m["outcome_counts"].values()) == m["n_scorable"]


def test_ps3_no_composite_score_is_produced():
    """PS-1 and PS-3 must never be collapsed into one number."""
    m = compute_ps3_metrics([ps3_record("a", "pass", matches=1, total=1)])
    forbidden = {"overall_score", "composite", "combined_score", "quality_score"}
    assert not (forbidden & set(m))


# --------------------------------------------------------------------------- #
# Scorer / human agreement (specification requires this to be MEASURED)
# --------------------------------------------------------------------------- #

from ps5.agreement import agreement_report, cohens_kappa, interpret_kappa  # noqa: E402


def test_perfect_agreement_gives_kappa_one():
    pairs = [(True, True)] * 10 + [(False, False)] * 10
    assert cohens_kappa(pairs).kappa == pytest.approx(1.0)


def test_chance_level_agreement_gives_kappa_near_zero():
    # Scorer and human each say "violation" half the time, independently.
    pairs = [(True, True), (True, False), (False, True), (False, False)] * 10
    assert abs(cohens_kappa(pairs).kappa) < 1e-9


def test_kappa_punishes_the_always_negative_scorer():
    """The failure raw agreement hides: 90% 'correct' while catching nothing."""
    pairs = [(False, False)] * 90 + [(False, True)] * 10
    result = cohens_kappa(pairs)
    assert result.observed_agreement == pytest.approx(0.90)
    assert result.kappa == pytest.approx(0.0, abs=1e-9)
    assert result.recall == 0.0


def test_kappa_undefined_when_no_variation_is_reported_not_faked():
    result = cohens_kappa([(False, False)] * 20)
    assert result.kappa is None
    assert "undefined" in result.note


def test_confusion_counts_and_derived_rates():
    pairs = [(True, True)] * 3 + [(True, False)] * 2 + [(False, True)] * 1 + [(False, False)] * 4
    r = cohens_kappa(pairs)
    assert (r.true_positive, r.false_positive, r.false_negative, r.true_negative) == (3, 2, 1, 4)
    assert r.precision == pytest.approx(3 / 5)
    assert r.recall == pytest.approx(3 / 4)


def test_unlabelled_rows_are_excluded_not_guessed():
    rows = [
        {"case_id": "a", "scorer_violation": True, "human_violation": True},
        {"case_id": "b", "scorer_violation": True, "human_violation": None},
        {"case_id": "c", "scorer_violation": False, "human_violation": False},
    ]
    report = agreement_report(rows)
    assert report["n_labelled"] == 2 and report["n_unlabelled"] == 1
    assert report["scorer_vs_human"]["n"] == 2


def test_human_vs_human_reported_when_two_raters_overlap():
    rows = [
        {"case_id": "a", "rater_id": "AA", "scorer_violation": True, "human_violation": True},
        {"case_id": "a", "rater_id": "BB", "scorer_violation": True, "human_violation": False},
        {"case_id": "b", "rater_id": "AA", "scorer_violation": False, "human_violation": False},
        {"case_id": "b", "rater_id": "BB", "scorer_violation": False, "human_violation": False},
    ]
    report = agreement_report(rows)
    assert report["human_vs_human"] is not None
    assert report["human_vs_human"]["n"] == 2


def test_single_rater_needs_no_human_vs_human_block():
    rows = [{"case_id": "a", "rater_id": "AA", "scorer_violation": True, "human_violation": True}]
    assert agreement_report(rows)["human_vs_human"] is None


def test_agreement_report_always_carries_caveats():
    report = agreement_report([{"case_id": "a", "scorer_violation": True, "human_violation": True}])
    assert len(report["caveats"]) >= 3
    assert any("not ground truth" in c for c in report["caveats"])


def test_kappa_interpretation_bands():
    assert interpret_kappa(0.9) == "almost perfect"
    assert interpret_kappa(0.7) == "substantial"
    assert interpret_kappa(0.5) == "moderate"
    assert interpret_kappa(0.3) == "fair"
    assert interpret_kappa(-0.2) == "worse than chance"


def test_scoring_unlabelled_subset_refuses_and_exits_nonzero(repo):
    """Accidentally scoring a blank TO_LABEL file must fail, not silently write undefined."""
    to_label = repo / "reports" / "validation" / "ps1_validation_TO_LABEL.csv"
    assert to_label.exists(), "TO_LABEL file must exist"
    proc = subprocess.run(
        [sys.executable, str(repo / "scripts" / "validation_subset.py"), "score", "--labelled", str(to_label)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "contains 0 labelled rows" in proc.stderr
    assert "Refusing to overwrite" in proc.stderr


def test_committed_labelled_data_produces_exact_validation_metrics(repo):
    """The committed human labels must reproduce the exact reported statistics."""
    labelled = repo / "reports" / "validation" / "ps1_validation_labelled.csv"
    assert labelled.exists(), "ps1_validation_labelled.csv must exist"
    proc = subprocess.run(
        [sys.executable, str(repo / "scripts" / "validation_subset.py"), "score", "--labelled", str(labelled)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "Cohen's kappa = 0.471 (moderate), n=80" in proc.stdout

    agreement_json = repo / "reports" / "validation" / "agreement.json"
    data = json.loads(agreement_json.read_text(encoding="utf-8"))
    overall = data["scorer_vs_human"]
    assert overall["n"] == 80
    assert overall["kappa"] == pytest.approx(0.471, abs=0.001)
    assert overall["observed_agreement"] == pytest.approx(0.775, abs=0.001)
    assert overall["precision"] == pytest.approx(0.750, abs=0.001)
    assert overall["recall"] == pytest.approx(0.536, abs=0.001)
    assert (overall["true_positive"], overall["false_positive"],
            overall["true_negative"], overall["false_negative"]) == (15, 5, 47, 13)

