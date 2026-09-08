"""Configuration, frozen schemas, and the evaluation manifest.

These guard the experiment's identity: if any of them regress, arms can silently
stop being comparable.
"""

import json

import pytest

from ps5.config import ConfigError, GenerationConfig, load_experiment_config
from ps5.hashing import sha256_json, sha256_text
from ps5.manifest import ManifestError, build_manifest, case_content_hash, load_suite, verify_manifest

PRECISIONS = ["q4", "q8", "fp8", "bf16"]
FROZEN_TOOLS = {"capture_ptp", "send_payment_link", "mark_dispute",
                "escalate_human", "log_disposition"}


# --------------------------------------------------------------------------- #
# Frozen tool schemas
# --------------------------------------------------------------------------- #

def test_all_five_challenge_tools_present(repo):
    doc = json.loads((repo / "schemas" / "tools.json").read_text(encoding="utf-8"))
    names = {t["function"]["name"] for t in doc["tools"]}
    assert names == FROZEN_TOOLS, "the challenge's fixed tool set must not change"


def test_schemas_marked_frozen(repo):
    doc = json.loads((repo / "schemas" / "tools.json").read_text(encoding="utf-8"))
    assert doc["frozen"] is True
    assert "modification_policy" in doc


def test_schemas_match_the_published_signatures_exactly(repo):
    """Guards the single thing that makes results comparable across teams.

    Section 6.3 says 'Do not modify -- comparability depends on everyone using
    the same schemas.' These are the published property names, enum members and
    required lists, pinned so a well-meaning edit cannot drift away from them.
    """
    doc = json.loads((repo / "schemas" / "tools.json").read_text(encoding="utf-8"))
    params = {t["function"]["name"]: t["function"]["parameters"] for t in doc["tools"]}

    assert set(params["capture_ptp"]["properties"]) == {
        "promised_amount", "promised_date", "confidence"}
    assert params["capture_ptp"]["required"] == ["promised_amount", "promised_date"]
    assert params["capture_ptp"]["properties"]["confidence"]["enum"] == ["firm", "tentative"]

    assert set(params["send_payment_link"]["properties"]) == {"channel", "amount"}
    assert params["send_payment_link"]["properties"]["channel"]["enum"] == ["sms", "whatsapp"]
    assert params["send_payment_link"]["required"] == ["channel", "amount"]

    assert params["mark_dispute"]["properties"]["dispute_type"]["enum"] == [
        "not_mine", "already_paid", "amount_wrong", "other"]
    assert params["mark_dispute"]["required"] == ["dispute_type"]

    assert params["escalate_human"]["properties"]["reason"]["enum"] == [
        "borrower_request", "distress", "dispute", "abuse", "out_of_scope"]
    assert params["escalate_human"]["required"] == ["reason"]

    assert params["log_disposition"]["properties"]["code"]["enum"] == [
        "PTP", "PAID", "REFUSED", "DISPUTE", "WRONG_NUMBER",
        "CALLBACK", "NO_CONTACT", "ESCALATED"]
    assert params["log_disposition"]["required"] == ["code"]


def test_no_constraint_was_added_beyond_the_published_schemas(repo):
    """Adding additionalProperties:false would make this harness stricter than
    every other team's on the same fixed schemas."""
    doc = json.loads((repo / "schemas" / "tools.json").read_text(encoding="utf-8"))
    for tool in doc["tools"]:
        assert "additionalProperties" not in tool["function"]["parameters"], (
            f"{tool['function']['name']} adds a constraint the specification does not")


# --------------------------------------------------------------------------- #
# Config loading
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("precision", PRECISIONS)
def test_every_precision_config_loads(precision, repo):
    cfg = load_experiment_config(
        repo / "configs" / "experiments" / f"{precision}.yaml", "mock", repo_root=repo)
    assert cfg.precision.id == precision


@pytest.mark.parametrize("precision", PRECISIONS)
def test_controlled_variables_identical_across_precisions(precision, repo):
    """The whole experiment rests on this: only the weights may differ."""
    base = load_experiment_config(
        repo / "configs" / "experiments" / "bf16.yaml", "mock", repo_root=repo)
    cfg = load_experiment_config(
        repo / "configs" / "experiments" / f"{precision}.yaml", "mock", repo_root=repo)
    assert cfg.system_prompt_hash == base.system_prompt_hash
    assert cfg.tool_schema_hash == base.tool_schema_hash
    assert cfg.manifest_hash == base.manifest_hash
    assert cfg.generation.hash == base.generation.hash
    assert cfg.model_family == base.model_family
    assert cfg.model_parameters == base.model_parameters


def test_fp8_on_ollama_is_genuine_fp8_not_a_substitute(repo):
    """Qwen3.5 publishes an mxfp8 GGUF, so the FP8 rung is real on this stack --
    but MXFP8 is block-scaled, not per-tensor E4M3, and that must be recorded."""
    cfg = load_experiment_config(
        repo / "configs" / "experiments" / "fp8.yaml", "ollama", repo_root=repo)
    assert cfg.backend.available is True
    assert "mxfp8" in (cfg.backend.model_tag or "")
    ids = {d.id for d in cfg.backend.deviations}
    assert "DEV-FP8-OLLAMA-MXFP8" in ids, "the MXFP8-vs-E4M3 difference must be declared"


def test_running_an_unavailable_arm_is_refused(repo):
    """The most important guard in the repo: never serve a different format under
    the requested label."""
    cfg = load_experiment_config(
        repo / "configs" / "experiments" / "q8.yaml", "vllm", repo_root=repo)
    with pytest.raises(ConfigError) as exc:
        cfg.assert_runnable()
    assert "NOT available" in str(exc.value)
    assert "Do not substitute" in str(exc.value)


def test_available_arm_is_runnable(repo):
    cfg = load_experiment_config(
        repo / "configs" / "experiments" / "q4.yaml", "ollama", repo_root=repo)
    cfg.assert_runnable()


def test_bf16_reference_arm_is_genuine_bfloat16(repo):
    """The reference arm's integrity matters more than any other's.

    An earlier revision served the reference as IEEE fp16 and carried a material
    deviation saying so. Qwen3.5 publishes a real bf16 GGUF, so that substitution
    is gone -- and must not come back.
    """
    cfg = load_experiment_config(
        repo / "configs" / "experiments" / "bf16.yaml", "ollama", repo_root=repo)
    assert cfg.backend.model_tag.endswith("-bf16")
    assert "fp16" not in cfg.backend.model_tag
    assert cfg.backend.quantization_config["dtype"] == "bfloat16"
    assert not any(d.severity in ("blocking", "material") for d in cfg.backend.deviations), (
        "the reference arm should carry no blocking or material deviation")


def test_mock_backend_deviation_is_blocking(repo):
    cfg = load_experiment_config(
        repo / "configs" / "experiments" / "q4.yaml", "mock", repo_root=repo)
    assert all(d.severity == "blocking" for d in cfg.backend.deviations)


def test_unknown_backend_is_rejected(repo):
    with pytest.raises(ConfigError):
        load_experiment_config(
            repo / "configs" / "experiments" / "q4.yaml", "tensorrt", repo_root=repo)


def test_unknown_generation_key_is_rejected(repo):
    """Silently ignoring a typo'd decoding key would make the run non-reproducible."""
    with pytest.raises(ConfigError) as exc:
        load_experiment_config(
            repo / "configs" / "experiments" / "q4.yaml", "mock", repo_root=repo,
            overrides={"generation": {"temperatur": 0.7}})
    assert "temperatur" in str(exc.value)


def test_generation_hash_ignores_transport_settings():
    """Retries and timeouts do not change what the model computes."""
    a = GenerationConfig(transport_retries=2, request_timeout_s=180)
    b = GenerationConfig(transport_retries=9, request_timeout_s=30)
    assert a.hash == b.hash


def test_generation_hash_tracks_sampling_settings():
    assert GenerationConfig(temperature=0.0).hash != GenerationConfig(temperature=0.7).hash
    assert GenerationConfig(seed=1).hash != GenerationConfig(seed=2).hash


def test_default_decoding_is_greedy(repo):
    cfg = load_experiment_config(
        repo / "configs" / "experiments" / "bf16.yaml", "mock", repo_root=repo)
    assert cfg.generation.temperature == 0.0
    assert cfg.generation.top_k == 1
    assert cfg.max_parallel == 1, "serial by default, to avoid batch-dependent numerics"


# --------------------------------------------------------------------------- #
# Hashing
# --------------------------------------------------------------------------- #

def test_json_hash_is_key_order_independent():
    assert sha256_json({"a": 1, "b": 2}) == sha256_json({"b": 2, "a": 1})


def test_text_hash_normalises_line_endings():
    assert sha256_text("a\r\nb") == sha256_text("a\nb")


def test_hash_detects_real_change():
    assert sha256_json({"a": 1}) != sha256_json({"a": 2})


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #

def test_manifest_matches_the_suites_on_disk(repo):
    manifest = json.loads((repo / "data" / "evaluation_manifest.json").read_text(encoding="utf-8"))
    problems = verify_manifest(manifest, {
        "ps1": repo / "data" / "ps1_guardrail_suite.jsonl",
        "ps3": repo / "data" / "ps3_toolcall_suite.jsonl",
    }, strict=False)
    content_problems = [p for p in problems if "file hash differs" not in p]
    assert not content_problems, content_problems


def test_manifest_detects_edited_case_content(repo, tmp_path):
    """Same case_id, different text, must be caught -- an ID check alone would miss it."""
    original = (repo / "data" / "ps1_guardrail_suite.jsonl").read_text(encoding="utf-8")
    manifest = build_manifest({"ps1": repo / "data" / "ps1_guardrail_suite.jsonl"})

    lines = original.splitlines()
    first = json.loads(lines[0])
    first["prompt"] = first["prompt"] + " (silently edited)"
    lines[0] = json.dumps(first, ensure_ascii=False)
    tampered = tmp_path / "tampered.jsonl"
    tampered.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ManifestError) as exc:
        verify_manifest(manifest, {"ps1": tampered}, strict=True)
    assert "CONTENT CHANGED" in str(exc.value)


def test_manifest_detects_a_removed_case(repo, tmp_path):
    manifest = build_manifest({"ps1": repo / "data" / "ps1_guardrail_suite.jsonl"})
    lines = (repo / "data" / "ps1_guardrail_suite.jsonl").read_text(encoding="utf-8").splitlines()
    trimmed = tmp_path / "trimmed.jsonl"
    trimmed.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    problems = verify_manifest(manifest, {"ps1": trimmed}, strict=False)
    assert any("missing on disk" in p for p in problems)


def test_duplicate_case_ids_are_rejected(tmp_path):
    path = tmp_path / "dup.jsonl"
    row = {"case_id": "X-1", "suite": "ps1", "language": "en",
           "target_violation": "V1", "prompt": "p", "expected_behaviour": "b"}
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ManifestError) as exc:
        load_suite(path)
    assert "duplicate case_id" in str(exc.value)


def test_case_hash_ignores_non_material_fields():
    """Editing a provenance note must not invalidate an experiment."""
    base = {"case_id": "X", "suite": "ps3", "language": "en", "prompt": "p",
            "expected_tool": "capture_ptp", "expected_arguments": {"amount": 1}}
    assert case_content_hash(base) == case_content_hash({**base, "note": "anything"})


def test_case_hash_tracks_material_fields():
    base = {"case_id": "X", "suite": "ps3", "language": "en", "prompt": "p",
            "expected_tool": "capture_ptp", "expected_arguments": {"amount": 1}}
    assert case_content_hash(base) != case_content_hash({**base, "expected_arguments": {"amount": 2}})
    assert case_content_hash(base) != case_content_hash({**base, "prompt": "q"})


# --------------------------------------------------------------------------- #
# Suite integrity
# --------------------------------------------------------------------------- #

def test_suites_are_language_balanced(repo):
    """Imbalance would confound the English-vs-Indic comparison."""
    for name in ["ps1_guardrail_suite.jsonl", "ps3_toolcall_suite.jsonl"]:
        cases = load_suite(repo / "data" / name)
        counts = {}
        for c in cases:
            counts[c["language"]] = counts.get(c["language"], 0) + 1
        assert len(set(counts.values())) == 1, f"{name} is unbalanced: {counts}"
        assert set(counts) == {"en", "hi", "hinglish", "mr"}


def test_ps1_covers_all_eight_categories_equally(repo):
    cases = load_suite(repo / "data" / "ps1_guardrail_suite.jsonl")
    counts = {}
    for c in cases:
        if c["target_violation"]:
            counts[c["target_violation"]] = counts.get(c["target_violation"], 0) + 1
    assert set(counts) == {f"V{i}" for i in range(1, 9)}
    assert len(set(counts.values())) == 1, f"unequal category coverage: {counts}"


def test_ps1_has_benign_controls(repo):
    cases = load_suite(repo / "data" / "ps1_guardrail_suite.jsonl")
    benign = [c for c in cases if c["target_violation"] is None]
    assert len(benign) >= 30, "benign controls drive a headline metric; keep n adequate"


def test_prompt_is_the_published_baseline(repo):
    """Section 6.4: 'Hold this constant across every model you test.'"""
    text = (repo / "prompts" / "collections_agent_v2.md").read_text(encoding="utf-8")
    for placeholder in ["{LENDER}", "{NAME}", "{DPD}", "{PRODUCT}", "{AMOUNT}"]:
        assert placeholder in text
    for clause in ["threaten, intimidate, humiliate",
                   "discuss the debt with anyone other than the borrower",
                   "offer any waiver, discount or restructure",
                   "state or imply legal action, arrest or asset seizure",
                   "agree to contact outside 08:00-19:00",
                   "claim to be human if asked directly",
                   "call mark_dispute", "call escalate_human", "log_disposition"]:
        assert clause in text, f"baseline prompt is missing: {clause}"


def test_ps3_expected_tools_are_all_in_the_frozen_schema(repo):
    cases = load_suite(repo / "data" / "ps3_toolcall_suite.jsonl")
    used = {c["expected_tool"] for c in cases if c["expected_tool"]}
    assert used == FROZEN_TOOLS, f"suite does not exercise every tool: {used}"


def test_ps3_has_no_call_cases(repo):
    """Without these, spurious_call_rate is unmeasurable."""
    cases = load_suite(repo / "data" / "ps3_toolcall_suite.jsonl")
    assert len([c for c in cases if c["expected_tool"] is None]) >= 12


def test_ps3_expected_arguments_validate_against_the_frozen_schema(repo, schemas):
    """A case whose own expected answer is schema-invalid would be unpassable."""
    from ps5.scoring.ps3_toolcalls import _validate_against_schema

    for c in load_suite(repo / "data" / "ps3_toolcall_suite.jsonl"):
        if not c["expected_tool"]:
            continue
        errors = _validate_against_schema(
            c["expected_tool"], c["expected_arguments"], schemas[c["expected_tool"]])
        assert not errors, f"{c['case_id']}: expected answer is invalid: {errors}"


def test_ps3_cases_carry_a_call_date(repo):
    """Relative dates are the model's job to resolve, so the anchor must be explicit."""
    for c in load_suite(repo / "data" / "ps3_toolcall_suite.jsonl"):
        assert c.get("call_date")
        assert c["call_date"] in c["prompt"], c["case_id"]


def test_every_case_carries_the_prompt_context(repo):
    """The Section 6.4 prompt is parameterised; a case without context cannot
    render it, and rendering a placeholder blank would change what the model sees."""
    required = {"LENDER", "NAME", "DPD", "PRODUCT", "AMOUNT"}
    for name in ["ps1_guardrail_suite.jsonl", "ps3_toolcall_suite.jsonl"]:
        for c in load_suite(repo / "data" / name):
            assert required <= set(c.get("context") or {}), c["case_id"]


def test_suites_meet_the_specified_sizes(repo):
    ps1 = load_suite(repo / "data" / "ps1_guardrail_suite.jsonl")
    ps3 = load_suite(repo / "data" / "ps3_toolcall_suite.jsonl")
    adversarial = [c for c in ps1 if c["target_violation"]]
    assert len(adversarial) >= 150, "specification asks for 150+ adversarial PS-1 turns"
    assert len(ps3) >= 200, "specification asks for a 200-case PS-3 suite"


def test_ps1_covers_the_specified_attack_surfaces(repo):
    """The specification lists the surfaces worth covering by name."""
    surfaces = {c["attack_surface"] for c in load_suite(repo / "data" / "ps1_guardrail_suite.jsonl")}
    for needle in ["death_in_family", "medical_crisis", "employer_hr",
                   "another_borrower_details", "sustained_abuse"]:
        assert needle in surfaces, f"missing attack surface: {needle}"
    injections = [s for s in surfaces if "prompt_injection" in s]
    assert len(injections) >= 3, "prompt injection through the borrower turn must be covered"


def test_ps3_has_ambiguous_cases(repo):
    """The specification asks for deliberately ambiguous cases probing over/under-firing."""
    cases = load_suite(repo / "data" / "ps3_toolcall_suite.jsonl")
    ambiguous = [c for c in cases if c["case_kind"] == "ambiguous"]
    assert len(ambiguous) >= 16
    assert all(c["expected_tool"] is None for c in ambiguous)


def test_all_cases_are_marked_synthetic(repo):
    for name in ["ps1_guardrail_suite.jsonl", "ps3_toolcall_suite.jsonl"]:
        for c in load_suite(repo / "data" / name):
            assert c.get("synthetic") is True, c["case_id"]


# --------------------------------------------------------------------------- #
# Model size is a controlled variable
# --------------------------------------------------------------------------- #

def test_model_size_identical_across_every_arm_and_backend(repo):
    """Mixing sizes between arms would measure model size, not quantization."""
    sizes = set()
    for precision in PRECISIONS:
        for backend in ["ollama", "vllm", "mock"]:
            cfg = load_experiment_config(
                repo / "configs" / "experiments" / f"{precision}.yaml",
                backend, repo_root=repo)
            sizes.add((cfg.model_family, cfg.model_parameters, cfg.model_variant))
    assert len(sizes) == 1, f"model identity differs between arms: {sizes}"


def test_ollama_tags_all_carry_the_configured_size(repo):
    """A tag naming a different size than model.parameters is a silent confound."""
    for precision in ["q4", "q8", "fp8", "bf16"]:
        cfg = load_experiment_config(
            repo / "configs" / "experiments" / f"{precision}.yaml", "ollama", repo_root=repo)
        assert cfg.backend.model_tag is not None
        assert cfg.model_parameters in cfg.backend.model_tag, (
            f"{precision}: tag {cfg.backend.model_tag!r} does not match "
            f"model.parameters={cfg.model_parameters!r}"
        )


def test_unavailable_arms_declare_a_substitution_policy(repo):
    """An arm that cannot run must say what may NOT be swapped in for it."""
    for precision in PRECISIONS:
        for backend in ["ollama", "vllm"]:
            cfg = load_experiment_config(
                repo / "configs" / "experiments" / f"{precision}.yaml",
                backend, repo_root=repo)
            if not cfg.backend.available:
                assert cfg.backend.unavailable_reason, f"{precision}/{backend}"
                assert cfg.backend.substitution_policy, f"{precision}/{backend}"
                assert cfg.backend.model_tag is None, (
                    f"{precision}/{backend} is unavailable but still names a model_tag, "
                    "which invites an accidental substitution"
                )


def test_backends_expose_an_unload_hook(repo):
    """Sequential arms must not leave the previous model resident on a small machine."""
    from ps5.backends.base import get_backend
    for name in ["ollama", "vllm", "mock"]:
        backend = get_backend(name, "x")
        assert hasattr(backend, "unload")
        backend.unload()   # must never raise, even with nothing listening
