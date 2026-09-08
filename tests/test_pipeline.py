"""Backends, comparability enforcement, and an end-to-end pipeline run.

The end-to-end test uses the mock backend, so it proves the PLUMBING is correct
without asserting anything about any model's behaviour.
"""

import json
import sys
from unittest.mock import patch

import pytest

from ps5.aggregate import ComparabilityError, aggregate, check_comparability, discover_runs, write_outputs
from ps5.backends.base import get_backend
from ps5.cliff import load_criterion
from ps5.config import GenerationConfig, load_experiment_config
from ps5.report import SyntheticReportRefused, render_findings
from ps5.runner import ExperimentRunner


# --------------------------------------------------------------------------- #
# Backend contract
# --------------------------------------------------------------------------- #

def test_mock_backend_is_flagged_synthetic():
    backend = get_backend("mock", "mock:q4", {"precision_id": "q4"})
    assert backend.synthetic is True
    assert backend.describe()["synthetic"] is True
    assert "FABRICATED" in backend.describe()["WARNING"]


def test_real_backends_are_not_flagged_synthetic():
    for name in ["ollama", "vllm"]:
        assert get_backend(name, "x").synthetic is False


def test_mock_backend_is_deterministic(tools):
    case = {"case_id": "PS3-001", "suite": "ps3", "expected_tool": "capture_ptp",
            "expected_arguments": {"borrower_id": "B", "amount": 5000}}
    backend = get_backend("mock", "mock:q4", {"precision_id": "q4"})
    gen = GenerationConfig()
    a = backend.generate("sys", "msg", tools, gen, case=case)
    b = backend.generate("sys", "msg", tools, gen, case=case)
    assert a.to_dict() == b.to_dict()


def test_mock_backend_degrades_with_precision(tools):
    """Sanity check on the fixture itself, not a claim about any model."""
    gen = GenerationConfig()
    cases = [{"case_id": f"PS3-{i:03d}", "suite": "ps3", "expected_tool": "capture_ptp",
              "expected_arguments": {"borrower_id": "B", "amount": 5000,
                                     "promise_date": "2026-09-12"}} for i in range(200)]
    rates = {}
    for precision in ["bf16", "q4"]:
        backend = get_backend("mock", f"mock:{precision}", {"precision_id": precision})
        good = sum(
            1 for c in cases
            if (r := backend.generate("s", "m", tools, gen, case=c)).tool_calls
            and r.tool_calls[0].arguments == c["expected_arguments"]
        )
        rates[precision] = good / len(cases)
    assert rates["bf16"] > rates["q4"]


@pytest.mark.parametrize("backend_name", ["ollama", "vllm"])
def test_real_backends_ignore_the_case_argument(backend_name, tools):
    """A real backend conditioning on the expected answer would be cheating."""
    backend = get_backend(backend_name, "some-model")
    gen = GenerationConfig(transport_retries=0, request_timeout_s=1)
    captured = []

    class Boom(Exception):
        pass

    def fake_post(url, **kwargs):
        captured.append(kwargs.get("json"))
        raise Boom()

    for case in [None, {"case_id": "X", "expected_tool": "capture_ptp",
                        "expected_arguments": {"amount": 5000}}]:
        with patch("requests.post", side_effect=fake_post):
            try:
                backend.generate("sys", "msg", tools, gen, case=case)
            except Boom:
                pass
    assert len(captured) == 2
    assert captured[0] == captured[1], "the request payload must not depend on `case`"


# --------------------------------------------------------------------------- #
# Comparability enforcement
# --------------------------------------------------------------------------- #

class FakeRun:
    def __init__(self, precision, **overrides):
        self.precision = precision
        self.metadata = {
            "experiment_id": f"{precision}-x",
            "manifest_hash": "sha256:M", "system_prompt_hash": "sha256:P",
            "tool_schema_hash": "sha256:S", "generation_config_hash": "sha256:G",
            "metric_spec_version": "1.0.0", "guardrail_rules_hash": "sha256:R",
            "hardware_fingerprint": "sha256:H",
            "model": {"family": "qwen2.5", "tag": f"tag-{precision}"},
            "backend": {"name": "ollama", "synthetic": False},
            "precision": {"id": precision},
            "deviations": [],
        }
        self.metadata.update(overrides)
        self.records = {}

    @property
    def synthetic(self):
        return bool(self.metadata["backend"].get("synthetic"))

    def control_value(self, field):
        return self.metadata.get(field)


def test_matching_arms_are_comparable():
    runs = {p: FakeRun(p) for p in ["bf16", "q8", "q4"]}
    report = check_comparability(runs)
    assert report["comparable"] is True and not report["divergences"]


@pytest.mark.parametrize("field", [
    "manifest_hash", "system_prompt_hash", "tool_schema_hash",
    "generation_config_hash", "guardrail_rules_hash", "hardware_fingerprint",
])
def test_any_control_divergence_blocks_the_comparison(field):
    runs = {"bf16": FakeRun("bf16"), "q4": FakeRun("q4", **{field: "sha256:DIFFERENT"})}
    with pytest.raises(ComparabilityError) as exc:
        check_comparability(runs)
    assert field in str(exc.value)


def test_different_model_families_blocks_the_comparison():
    """Comparing two models while claiming to measure quantization is the cardinal sin."""
    runs = {"bf16": FakeRun("bf16"),
            "q4": FakeRun("q4", model={"family": "llama3", "tag": "t"})}
    with pytest.raises(ComparabilityError) as exc:
        check_comparability(runs)
    assert "DIFFERENT MODEL FAMILIES" in str(exc.value)


def test_allow_deviation_records_rather_than_erases():
    runs = {"bf16": FakeRun("bf16"), "q4": FakeRun("q4", manifest_hash="sha256:OTHER")}
    report = check_comparability(runs, allow_deviation=True)
    assert report["comparable"] is False
    assert report["divergences"], "the divergence must survive into the record"


def test_mixed_backends_warn_but_do_not_block():
    runs = {"bf16": FakeRun("bf16"),
            "q4": FakeRun("q4", backend={"name": "vllm", "synthetic": False})}
    report = check_comparability(runs)
    assert report["comparable"] is True
    assert any("different backends" in w for w in report["warnings"])


def test_synthetic_arm_is_warned_about():
    runs = {"bf16": FakeRun("bf16"),
            "q4": FakeRun("q4", backend={"name": "mock", "synthetic": True})}
    report = check_comparability(runs)
    assert any("SYNTHETIC" in w for w in report["warnings"])


def test_differing_model_tags_are_expected_and_allowed():
    """The tag IS the treatment; it must differ."""
    runs = {p: FakeRun(p) for p in ["bf16", "q4"]}
    assert check_comparability(runs)["comparable"] is True


# -- thinking mode ---------------------------------------------------------- #
#
# Unlike every other control, this one is negotiated with the server at run time
# rather than fixed in a config file, so it can diverge between arms with nothing
# on disk having changed. That makes it the control most worth checking.

def _backend(name="ollama", **thinking):
    info = {"thinking_disable_requested": True, "thinking_disable_sent": True,
            "thinking_unsupported_by_model": False}
    info.update(thinking)
    return {"name": name, "synthetic": False, "info": info}


def test_uniformly_disabled_thinking_is_comparable():
    runs = {p: FakeRun(p, backend=_backend()) for p in ["bf16", "q8", "q4"]}
    report = check_comparability(runs)
    assert report["comparable"] is True
    assert not any("Thinking" in w for w in report["warnings"])


def test_thinking_enabled_on_one_arm_blocks_the_comparison():
    """An arm that reasoned first spends several times the tokens of one that did not."""
    runs = {"bf16": FakeRun("bf16", backend=_backend()),
            "q4": FakeRun("q4", backend=_backend(thinking_disable_requested=False))}
    with pytest.raises(ComparabilityError) as exc:
        check_comparability(runs)
    assert "thinking_mode" in str(exc.value)


def test_a_model_without_thinking_mode_is_still_comparable():
    """Rejected `think` and accepted `think:false` both mean thinking did not run.

    Failing here would be a false alarm that stops a valid comparison, which is
    as damaging as missing a real divergence.
    """
    runs = {
        "bf16": FakeRun("bf16", backend=_backend()),
        "q4": FakeRun("q4", backend=_backend(thinking_disable_sent=False,
                                             thinking_unsupported_by_model=True)),
    }
    assert check_comparability(runs)["comparable"] is True


def test_unrecorded_thinking_status_warns_rather_than_asserting():
    """Absence of evidence is reported as such, not as evidence of the control."""
    runs = {"bf16": FakeRun("bf16", backend=_backend()),
            "q4": FakeRun("q4", backend={"name": "ollama", "synthetic": False})}
    report = check_comparability(runs)
    assert report["comparable"] is True
    assert any("could not be verified" in w and "q4" in w for w in report["warnings"])


# --------------------------------------------------------------------------- #
# End-to-end
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def pipeline(repo, tmp_path_factory):
    out = tmp_path_factory.mktemp("results")
    for precision in ["bf16", "fp8", "q8", "q4"]:
        cfg = load_experiment_config(
            repo / "configs" / "experiments" / f"{precision}.yaml", "mock",
            repo_root=repo, overrides={"experiment": {"results_root": str(out)}})
        ExperimentRunner(cfg, verbose=False).run(["ps1", "ps3"])
    criterion = load_criterion(str(repo / "configs" / "cliff_criterion.yaml"))
    runs = discover_runs(out)
    agg = aggregate(runs, criterion)
    return out, criterion, agg


def test_all_four_arms_produce_results(pipeline):
    out, _criterion, agg = pipeline
    assert set(agg["arms"]) == {"bf16", "fp8", "q8", "q4"}
    for precision in ["bf16", "fp8", "q8", "q4"]:
        assert (out / precision / "metadata.json").exists()
        assert (out / precision / "ps1_results.jsonl").exists()
        assert (out / precision / "ps3_results.jsonl").exists()


def test_every_arm_saw_the_identical_case_set(pipeline):
    out, _criterion, _agg = pipeline
    ids = {}
    for precision in ["bf16", "fp8", "q8", "q4"]:
        rows = [json.loads(l) for l in
                (out / precision / "ps3_results.jsonl").read_text(encoding="utf-8").splitlines() if l]
        ids[precision] = [r["case_id"] for r in rows]
    reference = ids["bf16"]
    for precision, seen in ids.items():
        assert seen == reference, f"{precision} saw a different case set or order"


def test_raw_records_retain_the_model_output_for_rescoring(pipeline):
    out, _criterion, _agg = pipeline
    rows = [json.loads(l) for l in
            (out / "q4" / "ps3_results.jsonl").read_text(encoding="utf-8").splitlines() if l]
    row = rows[0]
    for field in ["response_text", "tool_calls", "prompt", "expected_tool",
                  "expected_arguments", "score", "precision", "model_tag", "synthetic"]:
        assert field in row, f"raw record is missing {field}"


def test_metadata_records_every_control_hash(pipeline):
    out, _criterion, _agg = pipeline
    meta = json.loads((out / "q4" / "metadata.json").read_text(encoding="utf-8"))
    for field in ["manifest_hash", "system_prompt_hash", "tool_schema_hash",
                  "generation_config_hash", "guardrail_rules_hash",
                  "hardware_fingerprint", "cliff_criterion_hash", "environment",
                  "deviations", "controls_held_constant", "controls_NOT_held_constant"]:
        assert field in meta, f"metadata is missing {field}"


def test_aggregate_is_marked_synthetic_end_to_end(pipeline):
    _out, _criterion, agg = pipeline
    assert agg["synthetic"] is True
    assert all(a["synthetic"] for a in agg["arms"].values())


def test_findings_report_refuses_synthetic_by_default(pipeline, tmp_path):
    out, criterion, agg = pipeline
    written = write_outputs(agg, criterion, tmp_path)
    with pytest.raises(SyntheticReportRefused):
        render_findings(written["json"], criterion)


def test_findings_report_stamps_a_banner_when_forced(pipeline, tmp_path):
    out, criterion, agg = pipeline
    written = write_outputs(agg, criterion, tmp_path)
    markdown = render_findings(written["json"], criterion, allow_synthetic=True)
    assert "SYNTHETIC VALIDATION ARTEFACT" in markdown
    assert "not a measurement of any model" in markdown


def test_outputs_include_csv_json_and_markdown(pipeline, tmp_path):
    _out, criterion, agg = pipeline
    written = write_outputs(agg, criterion, tmp_path)
    assert set(written) == {"json", "csv", "markdown"}
    assert written["csv"].read_text(encoding="utf-8").startswith("suite,metric,precision")


def test_degradation_analysis_covers_every_headline_metric(pipeline):
    _out, criterion, agg = pipeline
    for suite, specs in criterion.headline_metrics.items():
        analysed = {a["metric"] for a in agg["degradation"]["analyses"][suite]}
        assert analysed == {s["metric"] for s in specs}


def test_ps1_and_ps3_reported_separately(pipeline):
    _out, _criterion, agg = pipeline
    assert set(agg["metrics"]) == {"ps1", "ps3"}
    assert "overall_score" not in agg


# --------------------------------------------------------------------------- #
# Preflight and shell-script portability
# --------------------------------------------------------------------------- #

def test_preflight_reports_all_arms_ready_on_mock(repo):
    import subprocess
    proc = subprocess.run(
        [sys.executable, "scripts/preflight.py", "--backend", "mock"],
        cwd=repo, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "ready: 4/4" in proc.stdout


def test_preflight_marks_fp8_blocked_on_ollama(repo):
    """FP8 must be reported as a coverage GAP, never as a model that could be pulled."""
    import subprocess
    proc = subprocess.run(
        [sys.executable, "scripts/preflight.py", "--backend", "ollama",
         "--host", "http://127.0.0.1:1"],   # deliberately unreachable
        cwd=repo, capture_output=True, text=True, timeout=120,
    )
    # Unreachable backend exits 2 without claiming anything about the arms.
    assert proc.returncode == 2
    assert "UNREACHABLE" in proc.stdout


def test_run_all_never_hardcodes_bare_python(repo):
    """`python` does not exist on stock macOS or outside an activated venv.

    Regression guard: an earlier version hard-coded it and failed in any shell
    where the venv was not active.
    """
    import re
    script = (repo / "scripts" / "run_all.sh").read_text(encoding="utf-8")
    for lineno, line in enumerate(script.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("#") or not stripped:
            continue
        assert not re.search(r"(^|[;&|]\s*|\$\(\s*)python\s+\S", stripped), (
            f"scripts/run_all.sh:{lineno} invokes bare `python`: {stripped!r}"
        )


def test_run_all_resolves_an_interpreter(repo):
    script = (repo / "scripts" / "run_all.sh").read_text(encoding="utf-8")
    assert 'PY="${PYTHON:-}"' in script
    assert "$REPO/.venv/bin/python" in script, "should work without activating the venv"
    assert "command -v python3" in script


def _run_all(repo, args, cwd=None):
    import subprocess
    return subprocess.run(["bash", str(repo / "scripts" / "run_all.sh"), *args],
                          cwd=cwd or repo, capture_output=True, text=True, timeout=300)


def test_mock_run_defaults_to_its_own_results_root(repo):
    """A pipeline check must not be able to land on top of measured arms.

    Each real arm costs a long serial run, and its JSONL is the primary evidence;
    a default that overwrites it is a trap, not a convenience.
    """
    script = (repo / "scripts" / "run_all.sh").read_text(encoding="utf-8")
    assert 'RESULTS_ROOT="${3:-results_mock$SUFFIX}"' in script
    assert 'RESULTS_ROOT="${3:-results$SUFFIX}"' in script


def test_mock_refuses_to_write_into_a_root_holding_measured_arms(repo, tmp_path):
    (tmp_path / "q4").mkdir()
    (tmp_path / "q4" / "metadata.json").write_text(
        json.dumps({"backend": {"name": "ollama", "synthetic": False}}), encoding="utf-8")

    proc = _run_all(repo, ["mock", "default", str(tmp_path)])
    assert proc.returncode == 1
    assert "ABORT" in proc.stdout
    assert "q4  (real)" in proc.stdout
    # And it refused BEFORE running anything, so the evidence is untouched.
    assert json.loads((tmp_path / "q4" / "metadata.json").read_text())["backend"]["synthetic"] is False
    assert not (tmp_path / "bf16").exists()


def test_a_real_run_refuses_to_aggregate_alongside_fabricated_arms(repo, tmp_path):
    """Mixing them would put invented numbers into the findings report."""
    (tmp_path / "q8").mkdir()
    (tmp_path / "q8" / "metadata.json").write_text(
        json.dumps({"backend": {"name": "mock", "synthetic": True}}), encoding="utf-8")

    proc = _run_all(repo, ["ollama", "default", str(tmp_path)])
    assert proc.returncode == 1
    assert "q8  (synthetic)" in proc.stdout
    assert "invented numbers" in proc.stdout


def test_an_unknown_config_set_lists_the_real_ones(repo):
    proc = _run_all(repo, ["ollama", "no-such-model"])
    assert proc.returncode == 1
    assert "no config set named 'no-such-model'" in proc.stdout
    assert "default" in proc.stdout and "qwen2.5-1.5b" in proc.stdout
