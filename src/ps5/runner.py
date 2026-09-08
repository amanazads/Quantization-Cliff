"""The controlled experiment runner.

One code path serves every precision. The only thing that differs between arms is
which weights the backend loads -- everything else (prompt, schemas, manifest,
decoding parameters, scoring, ordering) is identical by construction, and the
identity of all of it is recorded in metadata.json so the claim is checkable
rather than merely asserted.

Raw records carry the model's full output alongside the score, so a scoring bug
can be fixed by re-scoring (scripts/rescore.py) without re-generating.
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .backends.base import Backend, BackendError, GenerationResult, get_backend
from .config import ExperimentConfig
from .environment import capture_environment
from .hashing import sha256_file, sha256_json, short
from .manifest import load_suite, verify_manifest
from .scoring import ps1_guardrails, ps3_toolcalls

__all__ = ["ExperimentRunner", "RunSummary"]


class RunSummary(dict):
    """Plain dict; named for readability at call sites."""


class ExperimentRunner:
    def __init__(
        self,
        config: ExperimentConfig,
        backend: Optional[Backend] = None,
        guardrail_rules_path: Optional[Path] = None,
        limit: Optional[int] = None,
        verbose: bool = True,
    ):
        self.cfg = config
        self.limit = limit
        self.verbose = verbose
        self.guardrail_rules_path = (
            guardrail_rules_path or (config.repo_root / "configs" / "guardrail_rules.json")
        )

        options = dict(config.backend.options)
        options.setdefault("precision_id", config.precision.id)
        self.backend = backend or get_backend(
            config.backend.name, config.backend.model_tag or "", options
        )

        self.tools, self.schemas_by_name = ps3_toolcalls.load_tool_schemas(str(config.schemas_path))
        self.rules = ps1_guardrails.load_rules(str(self.guardrail_rules_path))
        self.experiment_id = self._make_experiment_id()

    # -- identity ----------------------------------------------------------- #

    def _make_experiment_id(self) -> str:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return (
            f"{self.cfg.precision.id}-{self.cfg.backend.name}-"
            f"{short(self.cfg.manifest_hash, 8)}-{stamp}"
        )

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message, flush=True)

    # -- suites ------------------------------------------------------------- #

    def _suite_paths(self) -> Dict[str, Path]:
        manifest = json.loads(self.cfg.manifest_path.read_text(encoding="utf-8"))
        return {
            suite_id: (self.cfg.repo_root / spec["path"]).resolve()
            for suite_id, spec in (manifest.get("suites") or {}).items()
        }

    def _load_cases(self, suite_id: str) -> List[Dict[str, Any]]:
        paths = self._suite_paths()
        if suite_id not in paths:
            raise BackendError(
                f"Suite '{suite_id}' is not in the evaluation manifest. "
                f"Available: {sorted(paths)}"
            )
        cases = load_suite(paths[suite_id])
        # Fixed manifest order, never shuffled: identical presentation order
        # across arms removes ordering as an uncontrolled variable.
        return cases[: self.limit] if self.limit else cases

    # -- metadata ----------------------------------------------------------- #

    def build_metadata(self, suites_run: Sequence[str], case_counts: Dict[str, int]) -> Dict[str, Any]:
        env = capture_environment(str(self.cfg.repo_root))
        backend_info = self.backend.describe()

        deviations = [d.to_dict() for d in self.cfg.backend.deviations]
        # Record hardware-conditional deviations only when their condition holds,
        # so the list reflects this run rather than every hypothetical run.
        cc = env.get("accelerator", {}).get("compute_capability")
        if cc:
            try:
                if float(cc) < 8.9:
                    for dev in deviations:
                        if dev.get("condition") == "gpu_compute_capability < 8.9":
                            dev["condition_met"] = True
                            dev["observed_compute_capability"] = cc
            except ValueError:
                pass

        return {
            "experiment_id": self.experiment_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),

            "precision": {
                "id": self.cfg.precision.id,
                "label": self.cfg.precision.label,
                "nominal_bits": self.cfg.precision.nominal_bits,
                "family": self.cfg.precision.family,
                "is_reference": self.cfg.precision.is_reference,
            },
            "model": {
                "id": self.cfg.model_id,
                "family": self.cfg.model_family,
                "parameters": self.cfg.model_parameters,
                "variant": self.cfg.model_variant,
                "tag": self.cfg.backend.model_tag,
                "quantization_format": self.cfg.backend.quantization_format,
                "quantization_config": self.cfg.backend.quantization_config,
                # Resolved at run time: digest / revision of the weights actually loaded.
                "resolved": backend_info.get("resolved", {}),
            },
            "backend": {
                "name": self.cfg.backend.name,
                "synthetic": bool(getattr(self.backend, "synthetic", False)),
                "info": backend_info,
            },

            "generation_config": self.cfg.generation.to_dict(),
            "generation_config_hash": self.cfg.generation.hash,
            "concurrency": {"max_parallel": self.cfg.max_parallel},

            "system_prompt_path": str(self.cfg.prompt_path.relative_to(self.cfg.repo_root)),
            "system_prompt_version": self.cfg.prompt_version,
            "system_prompt_hash": self.cfg.system_prompt_hash,
            "tool_schema_path": str(self.cfg.schemas_path.relative_to(self.cfg.repo_root)),
            "tool_schema_hash": self.cfg.tool_schema_hash,
            "manifest_path": str(self.cfg.manifest_path.relative_to(self.cfg.repo_root)),
            "manifest_hash": self.cfg.manifest_hash,
            "cliff_criterion_hash": self.cfg.cliff_criterion_hash,
            "guardrail_rules_hash": sha256_file(self.guardrail_rules_path),

            "metric_spec_version": self.cfg.metric_spec_version,
            "scorer_versions": {
                "ps1": ps1_guardrails.SCORER_VERSION,
                "ps3": ps3_toolcalls.SCORER_VERSION,
            },

            "suites_run": list(suites_run),
            "case_counts": dict(case_counts),
            "repeats": self.cfg.repeats,
            "case_limit_applied": self.limit,

            "environment": env,
            "hardware_fingerprint": env["hardware_fingerprint"],

            "deviations": deviations,
            "controls_held_constant": [
                "system prompt (hash recorded; identical across arms)",
                "tool schemas (hash recorded; frozen)",
                "evaluation manifest (hash recorded; identical case set and order)",
                "decoding parameters (hash recorded; greedy, fixed seed)",
                "scorer versions and guardrail rule set (hashes recorded)",
                "case presentation order (manifest order, never shuffled)",
                "concurrency (serial by default, to avoid batch-dependent numerics)",
                "context window (num_ctx pinned in the backend)",
            ],
            "controls_NOT_held_constant": [
                "wall-clock time and machine thermal state across arms",
                "any OS-level background load during the run",
                "llama.cpp / vLLM kernel selection, which can differ per quantization "
                "format by design -- this is inherent to the treatment, not a flaw "
                "in the control",
            ],
        }

    # -- execution ---------------------------------------------------------- #

    def run_suite(self, suite_id: str, out_dir: Path) -> Dict[str, Any]:
        cases = self._load_cases(suite_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{suite_id}_results.jsonl"

        # The baseline prompt is a TEMPLATE (spec Section 6.4). It is rendered per
        # case with that case's borrower context. The template never varies, and
        # its hash is what the comparability check compares across arms.
        tools = self.tools if suite_id == "ps3" else None

        n_total = len(cases) * self.cfg.repeats
        started = time.time()
        counts: Dict[str, int] = {}
        n_failures = 0

        with out_path.open("w", encoding="utf-8") as fh:
            index = 0
            for repeat_idx in range(self.cfg.repeats):
                for case in cases:
                    index += 1
                    payload = dict(case)
                    payload["_repeat_idx"] = repeat_idx

                    result = self.backend.generate(
                        system_prompt=self.cfg.render_system_prompt(case.get("context")),
                        user_message=case["prompt"],
                        tools=tools,
                        generation=self.cfg.generation,
                        case=payload,
                    )

                    if suite_id == "ps1":
                        score = ps1_guardrails.score_ps1_case(case, result, self.rules)
                        score_dict = score.to_dict()
                        key = ("violation" if score.violation else
                               "benign_refusal" if score.benign_refusal else
                               "ok" if score.scorable else "generation_failure")
                    else:
                        score = ps3_toolcalls.score_ps3_case(case, result, self.schemas_by_name)
                        score_dict = score.to_dict()
                        key = score.outcome

                    counts[key] = counts.get(key, 0) + 1
                    if not result.ok:
                        n_failures += 1

                    fh.write(json.dumps(self._record(
                        suite_id, case, repeat_idx, result, score_dict
                    ), ensure_ascii=False) + "\n")
                    # Flushed per record rather than left to the 8 KB buffer. An
                    # arm is hundreds of serial generations; without this the
                    # results file sits at 0 bytes for minutes and `wc -l` cannot
                    # distinguish a slow run from a hung one. It also means a run
                    # interrupted part-way leaves every completed case on disk.
                    fh.flush()

                    # Dense at the start, sparse later, deliberately: the first
                    # case carries the model load -- often tens of seconds on a
                    # memory-constrained machine -- and is exactly where a run
                    # looks hung. Once a rate is established every tenth is plenty.
                    if index <= 3 or index % 10 == 0 or index == n_total:
                        elapsed = time.time() - started
                        rate = index / elapsed if elapsed else 0.0
                        eta_min = (n_total - index) / rate / 60 if rate else 0.0
                        self._log(
                            f"  [{self.cfg.precision.id}/{suite_id}] {index}/{n_total} "
                            f"({rate:.2f} case/s, ~{eta_min:.0f} min left) counts={counts}"
                        )

        elapsed = time.time() - started
        self._log(
            f"  [{self.cfg.precision.id}/{suite_id}] wrote {out_path.name} "
            f"({n_total} records, {elapsed:.1f}s, {n_failures} generation failures)"
        )
        return {
            "suite": suite_id,
            "path": str(out_path),
            "n_records": n_total,
            "n_cases": len(cases),
            "repeats": self.cfg.repeats,
            "elapsed_s": round(elapsed, 2),
            "outcome_counts": counts,
            "n_generation_failures": n_failures,
        }

    def _record(
        self,
        suite_id: str,
        case: Dict[str, Any],
        repeat_idx: int,
        result: GenerationResult,
        score: Dict[str, Any],
    ) -> Dict[str, Any]:
        """One raw JSONL record: everything needed to re-score without re-running."""
        record: Dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "suite": suite_id,
            "case_id": case.get("case_id"),
            "repeat_idx": repeat_idx,

            "precision": self.cfg.precision.id,
            "precision_label": self.cfg.precision.label,
            "model": self.cfg.model_id,
            "model_tag": self.cfg.backend.model_tag,
            "backend": self.cfg.backend.name,
            "synthetic": bool(getattr(self.backend, "synthetic", False)),

            "language": case.get("language"),
            "prompt": case.get("prompt"),
            "expected_behaviour": case.get("expected_behaviour"),

            "response_text": result.text,
            "tool_calls": [tc.to_dict() for tc in result.tool_calls],
            "finish_reason": result.finish_reason,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "latency_ms": result.latency_ms,
            "attempts": result.attempts,
            "generation_ok": result.ok,
            "generation_error": result.error,
            "generation_error_kind": result.error_kind,

            "score": score,
            "scored_at_utc": datetime.now(timezone.utc).isoformat(),
        }

        # Promote the fields the metric layer reads to the top level, so metrics
        # never has to know the shape of a score object.
        if suite_id == "ps1":
            record.update({
                "target_violation": case.get("target_violation"),
                "scorable": score.get("scorable"),
                "violation": score.get("violation"),
                "is_benign_control": score.get("is_benign_control"),
                "benign_refusal": score.get("benign_refusal"),
            })
        else:
            record.update({
                "expected_tool": case.get("expected_tool"),
                "expected_arguments": case.get("expected_arguments"),
                "scorable": score.get("scorable"),
                "outcome": score.get("outcome"),
                "actual_tool": score.get("actual_tool"),
                "actual_arguments": score.get("actual_arguments"),
                "argument_matches": score.get("argument_matches"),
                "argument_total": score.get("argument_total"),
                "via_fallback": score.get("via_fallback"),
            })
        return record

    def run(self, suite_ids: Optional[Sequence[str]] = None) -> RunSummary:
        suite_ids = list(suite_ids or self.cfg.suite_ids)

        # Refuse an arm the backend cannot genuinely serve, before doing any work.
        self.cfg.assert_runnable()

        # Verify the suites on disk match the manifest before generating anything.
        manifest = json.loads(self.cfg.manifest_path.read_text(encoding="utf-8"))
        verify_manifest(manifest, self._suite_paths(), strict=True)

        self._log(f"experiment_id : {self.experiment_id}")
        self._log(f"precision     : {self.cfg.precision.id} ({self.cfg.precision.label})")
        self._log(f"backend       : {self.cfg.backend.name}  model={self.cfg.backend.model_tag}")
        self._log(f"manifest      : {short(self.cfg.manifest_hash)}  "
                  f"prompt={short(self.cfg.system_prompt_hash)}  "
                  f"schemas={short(self.cfg.tool_schema_hash)}  "
                  f"gen={short(self.cfg.generation.hash)}")

        if getattr(self.backend, "synthetic", False):
            self._log(
                "\n  *** SYNTHETIC BACKEND -- output is FABRICATED and is a pipeline\n"
                "  *** validation fixture only. It is not a measurement of any model\n"
                "  *** and must never be cited as a result.\n"
            )

        self.backend.health_check()

        for dev in self.cfg.backend.deviations:
            self._log(f"  DEVIATION [{dev.severity}] {dev.id}: {dev.description.strip()[:200]}")

        out_dir = self.cfg.results_dir()
        out_dir.mkdir(parents=True, exist_ok=True)

        suite_summaries = []
        case_counts: Dict[str, int] = {}
        try:
            for suite_id in suite_ids:
                summary = self.run_suite(suite_id, out_dir)
                suite_summaries.append(summary)
                case_counts[suite_id] = summary["n_cases"]
        finally:
            # Free the weights before the next arm loads, even if this one
            # failed part-way. See Backend.unload for why this is a control
            # concern and not merely tidiness.
            self.backend.unload()

        metadata = self.build_metadata(suite_ids, case_counts)
        metadata["suite_summaries"] = suite_summaries
        (out_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )
        self._log(f"  wrote {out_dir / 'metadata.json'}")

        return RunSummary({
            "experiment_id": self.experiment_id,
            "precision": self.cfg.precision.id,
            "backend": self.cfg.backend.name,
            "synthetic": bool(getattr(self.backend, "synthetic", False)),
            "results_dir": str(out_dir),
            "suites": suite_summaries,
        })
