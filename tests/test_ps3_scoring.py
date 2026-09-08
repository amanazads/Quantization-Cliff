"""PS-3 scorer edge cases.

These are the cases the PS-5 brief calls out by name. If any of these regress,
a headline metric silently changes meaning, so they are pinned hard.
"""

from conftest import NO_CALL_CASE, PTP_CASE, call, failed, ok

from ps5.scoring.ps3_toolcalls import PS3Outcome, normalise_value, score_ps3_case


# --------------------------------------------------------------------------- #
# The seven outcomes
# --------------------------------------------------------------------------- #

def test_pass_exact(schemas):
    score = score_ps3_case(PTP_CASE, ok(calls=[call("capture_ptp", dict(PTP_CASE["expected_arguments"]))]), schemas)
    assert score.outcome == PS3Outcome.PASS
    assert score.is_pass and score.correct_tool
    assert score.argument_matches == score.argument_total == 2


def test_missed_call(schemas):
    score = score_ps3_case(PTP_CASE, ok(text="Noted, I will look into it."), schemas)
    assert score.outcome == PS3Outcome.MISSED_CALL
    assert score.scorable and not score.correct_tool


def test_wrong_tool(schemas):
    score = score_ps3_case(
        PTP_CASE,
        ok(calls=[call("escalate_human", {"reason": "distress"})]),
        schemas,
    )
    assert score.outcome == PS3Outcome.WRONG_TOOL
    assert score.actual_tool == "escalate_human"
    assert not score.correct_tool


def test_correct_tool_wrong_argument_is_a_failure(schemas):
    """The brief's own example: capture_ptp(5000) vs capture_ptp(50000)."""
    args = dict(PTP_CASE["expected_arguments"], promised_amount=50000)
    score = score_ps3_case(PTP_CASE, ok(calls=[call("capture_ptp", args)]), schemas)
    assert score.outcome == PS3Outcome.WRONG_ARGUMENTS
    assert not score.is_pass
    # Tool identity was right, so it still counts toward correct_tool_rate --
    # which is exactly why that metric must never be reported alone.
    assert score.correct_tool
    assert score.argument_matches == 1 and score.argument_total == 2
    assert score.argument_mismatches[0]["field"] == "promised_amount"
    assert score.argument_mismatches[0]["expected"] == 5000
    assert score.argument_mismatches[0]["actual"] == 50000


def test_malformed_json_arguments(schemas):
    score = score_ps3_case(
        PTP_CASE,
        ok(calls=[call("capture_ptp", None, raw='{"promised_amount": 5000,}',
                       parse_error="arguments not valid JSON")]),
        schemas,
    )
    assert score.outcome == PS3Outcome.MALFORMED
    assert score.schema_errors


def test_spurious_call_when_none_expected(schemas):
    score = score_ps3_case(
        NO_CALL_CASE,
        ok(calls=[call("log_disposition", {"code": "NO_CONTACT"})]),
        schemas,
    )
    assert score.outcome == PS3Outcome.SPURIOUS_CALL


def test_correct_abstention_passes(schemas):
    score = score_ps3_case(NO_CALL_CASE, ok(text="Your total outstanding is INR 13,750."), schemas)
    assert score.outcome == PS3Outcome.PASS


def test_multiple_calls_one_unexpected_is_spurious(schemas):
    """A correct call plus an extra one is NOT a partial success."""
    result = ok(calls=[
        call("capture_ptp", dict(PTP_CASE["expected_arguments"])),
        call("send_payment_link", {"channel": "sms", "amount": 5000}),
    ])
    score = score_ps3_case(PTP_CASE, result, schemas)
    assert score.outcome == PS3Outcome.SPURIOUS_CALL
    assert score.n_calls == 2


def test_generation_failure_is_not_scorable(schemas):
    score = score_ps3_case(PTP_CASE, failed(), schemas)
    assert score.outcome == PS3Outcome.GENERATION_FAILURE
    assert score.scorable is False


# --------------------------------------------------------------------------- #
# Schema validation -> malformed
# --------------------------------------------------------------------------- #

def test_missing_required_field_is_malformed(schemas):
    score = score_ps3_case(PTP_CASE, ok(calls=[call("capture_ptp", {"promised_amount": 5000})]), schemas)
    assert score.outcome == PS3Outcome.MALFORMED
    assert any("promised_date" in e for e in score.schema_errors)


def test_enum_violation_is_malformed(schemas):
    case = {"case_id": "T", "suite": "ps3", "expected_tool": "send_payment_link",
            "expected_arguments": {"channel": "sms", "amount": 100}}
    score = score_ps3_case(
        case, ok(calls=[call("send_payment_link",
                             {"channel": "carrier_pigeon", "amount": 100})]),
        schemas)
    assert score.outcome == PS3Outcome.MALFORMED


def test_unknown_tool_name_is_malformed(schemas):
    score = score_ps3_case(PTP_CASE, ok(calls=[call("delete_account", {"code": "PTP"})]), schemas)
    assert score.outcome == PS3Outcome.MALFORMED
    assert any("unknown tool" in e for e in score.schema_errors)


def test_extra_property_is_tolerated_because_the_spec_schemas_allow_it(schemas):
    """The published Section 6.3 schemas do NOT set additionalProperties:false.

    An earlier revision of this repository added that constraint, which made
    extra keys count as `malformed` and would have made this harness stricter
    than every other team's on the same fixed schemas. The constraint was
    removed; extras are ignored for accuracy and do not invalidate the call.
    """
    args = dict(PTP_CASE["expected_arguments"], interest_waiver=True)
    score = score_ps3_case(PTP_CASE, ok(calls=[call("capture_ptp", args)]), schemas)
    assert score.outcome == PS3Outcome.PASS


def test_non_numeric_amount_is_malformed(schemas):
    args = dict(PTP_CASE["expected_arguments"], promised_amount="five thousand")
    score = score_ps3_case(PTP_CASE, ok(calls=[call("capture_ptp", args)]), schemas)
    assert score.outcome == PS3Outcome.MALFORMED


# --------------------------------------------------------------------------- #
# Argument normalisation
# --------------------------------------------------------------------------- #

def test_number_formatting_variants_all_match(schemas):
    for variant in ["5000", 5000, 5000.0, "5,000", "₹5000", "Rs 5,000", "INR 5000"]:
        args = dict(PTP_CASE["expected_arguments"], promised_amount=variant)
        score = score_ps3_case(PTP_CASE, ok(calls=[call("capture_ptp", args)]), schemas)
        assert score.outcome == PS3Outcome.PASS, f"{variant!r} should normalise to 5000"


def test_magnitude_error_never_forgiven(schemas):
    for wrong in [50000, 500, 5001, 4999.99]:
        args = dict(PTP_CASE["expected_arguments"], promised_amount=wrong)
        score = score_ps3_case(PTP_CASE, ok(calls=[call("capture_ptp", args)]), schemas)
        assert score.outcome == PS3Outcome.WRONG_ARGUMENTS, f"{wrong!r} must not pass"


def test_date_format_variants(schemas):
    for variant in ["2026-09-12", "12/09/2026", "12-09-2026", "12 Sep 2026"]:
        args = dict(PTP_CASE["expected_arguments"], promised_date=variant)
        score = score_ps3_case(PTP_CASE, ok(calls=[call("capture_ptp", args)]), schemas)
        assert score.outcome == PS3Outcome.PASS, f"{variant!r} should normalise"


def test_relative_date_is_not_resolved_by_the_scorer(schemas):
    """The scorer must not do the model's job of resolving 'tomorrow'."""
    args = dict(PTP_CASE["expected_arguments"], promised_date="tomorrow")
    score = score_ps3_case(PTP_CASE, ok(calls=[call("capture_ptp", args)]), schemas)
    assert score.outcome == PS3Outcome.WRONG_ARGUMENTS


def test_enum_comparison_is_case_insensitive(schemas):
    case = {"case_id": "T", "suite": "ps3", "expected_tool": "escalate_human",
            "expected_arguments": {"reason": "distress"}}
    score = score_ps3_case(
        case, ok(calls=[call("escalate_human", {"reason": "Distress"})]),
        schemas)
    assert score.outcome == PS3Outcome.PASS


def test_free_text_fields_are_not_scored(schemas):
    case = {"case_id": "T", "suite": "ps3", "expected_tool": "mark_dispute",
            "expected_arguments": {"dispute_type": "already_paid",
                                   "borrower_statement": "paid on 12 August"}}
    score = score_ps3_case(
        case, ok(calls=[call("mark_dispute", {"dispute_type": "already_paid",
                                              "borrower_statement": "totally different wording"})]),
        schemas)
    assert score.outcome == PS3Outcome.PASS
    assert score.argument_total == 1, "borrower_statement must be excluded from scored fields"


def test_missing_optional_expected_argument_is_a_mismatch(schemas):
    """An expected field the model simply omitted is a failure, not a free pass."""
    case = {"case_id": "T", "suite": "ps3", "expected_tool": "capture_ptp",
            "expected_arguments": {"promised_amount": 5000, "promised_date": "2026-09-12",
                                   "confidence": "firm"}}
    score = score_ps3_case(
        case, ok(calls=[call("capture_ptp", {"promised_amount": 5000,
                                             "promised_date": "2026-09-12"})]),
        schemas)
    assert score.outcome == PS3Outcome.WRONG_ARGUMENTS
    assert score.argument_mismatches[0]["reason"] == "missing"


def test_normalise_dispatches_on_expected_type():
    _e, _a, kind = normalise_value(5000, "5,000")
    assert kind == "number"
    _e, _a, kind = normalise_value("2026-09-12", "12/09/2026")
    assert kind == "date"
    _e, _a, kind = normalise_value("sms", "SMS")
    assert kind == "string"


# --------------------------------------------------------------------------- #
# Fallback extraction
# --------------------------------------------------------------------------- #

def test_prose_json_is_recovered_and_flagged(schemas):
    text = ('I will note that down.\n```json\n'
            '{"name": "capture_ptp", "arguments": '
            '{"promised_amount": 5000, "promised_date": "2026-09-12"}}\n```')
    score = score_ps3_case(PTP_CASE, ok(text=text), schemas)
    assert score.outcome == PS3Outcome.PASS
    assert score.via_fallback is True, "recovery must be flagged, never silent"


def test_tool_call_tag_is_recovered(schemas):
    text = ('<tool_call>{"name": "capture_ptp", "arguments": '
            '{"promised_amount": 5000, "promised_date": "2026-09-12"}}</tool_call>')
    score = score_ps3_case(PTP_CASE, ok(text=text), schemas)
    assert score.outcome == PS3Outcome.PASS and score.via_fallback


def test_prose_without_a_call_stays_a_missed_call(schemas):
    """Ordinary prose containing braces must not be invented into a call."""
    text = "I have noted your request {and will follow up}. Thank you."
    score = score_ps3_case(PTP_CASE, ok(text=text), schemas)
    assert score.outcome == PS3Outcome.MISSED_CALL
    assert score.via_fallback is False


def test_structured_channel_wins_over_prose(schemas):
    """When a real tool call exists, prose is not also scraped."""
    text = '{"name": "escalate_human", "arguments": {"reason": "dispute"}}'
    result = ok(text=text, calls=[call("capture_ptp", dict(PTP_CASE["expected_arguments"]))])
    score = score_ps3_case(PTP_CASE, result, schemas)
    assert score.outcome == PS3Outcome.PASS
    assert score.n_calls == 1 and score.via_fallback is False
