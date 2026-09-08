"""CLI entry point:  python -m ps5.run --precision q4 --backend ollama --suite ps1"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from .config import (
    DEFAULT_CONFIG_SET,
    ConfigError,
    config_set_dir,
    config_set_suffix,
    load_experiment_config,
)
from .backends.base import BackendError
from .runner import ExperimentRunner

PRECISIONS = ["f16", "q4", "q8", "fp8", "bf16"]
BACKENDS = ["ollama", "mock"]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _write_not_run_marker(cfg, args, error: str) -> None:
    """Leave a machine-readable record of a refused arm beside the results.

    A precision that was deliberately refused and one that was simply never
    attempted look identical in a results directory, and both render as "not
    run" in the report. They are not the same thing: one is a documented gap
    with a reason, the other is an omission. The aggregator reads this file so
    the findings document can tell a reader which it was.

    Best-effort: failing to write it must never change the exit code, which is
    what the runner script keys on.
    """
    try:
        out_dir = Path(cfg.results_root) / cfg.precision.id
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "NOT_RUN.json").write_text(json.dumps({
            "precision": cfg.precision.id,
            "precision_label": cfg.precision.label,
            "backend": args.backend,
            "model_tag": cfg.backend.model_tag,
            "refused_at_utc": datetime.now(timezone.utc).isoformat(),
            "reason": (cfg.backend.unavailable_reason or "").strip() or None,
            "substitution_policy": (cfg.backend.substitution_policy or "").strip() or None,
            "runner_message": error.strip(),
            "note": ("This arm was REFUSED, not skipped. It is a documented gap in "
                     "precision coverage and must never be read as a null result."),
        }, indent=2), encoding="utf-8")
    except Exception:  # pragma: no cover - never let bookkeeping fail a run
        pass


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m ps5.run",
        description="Run one PS-5 precision arm over the PS-1 and/or PS-3 suites.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples
--------
  # dry-run with mock backend
  python -m ps5.run --precision f16 --backend mock --suite ps1 ps3

  # run on local Ollama
  python -m ps5.run --precision f16 --backend ollama --config-set qwen2.5-1.5b
  python -m ps5.run --precision q8 --backend ollama --config-set qwen2.5-1.5b
  python -m ps5.run --precision q4 --backend ollama --config-set qwen2.5-1.5b
""",
    )
    p.add_argument("--precision", required=True, choices=PRECISIONS)
    p.add_argument("--backend", required=True, choices=BACKENDS)
    p.add_argument("--suite", nargs="+", default=None, choices=["ps1", "ps3"],
                   help="suites to run (default: whatever the config lists)")
    p.add_argument("--config-set", default=DEFAULT_CONFIG_SET,
                   help="which experiment to run (default: 'qwen2.5-1.5b'). Each "
                        "set writes to its own results root, so two models can never "
                        "be aggregated into one comparison.")
    p.add_argument("--config", default=None,
                   help="override the experiment config path (bypasses --config-set)")
    p.add_argument("--results-root", default=None,
                   help="override the results directory root")
    p.add_argument("--limit", type=int, default=None,
                   help="run only the first N cases per suite. FOR SMOKE TESTS ONLY: "
                        "a limited run must never be compared against a full one.")
    p.add_argument("--repeats", type=int, default=None,
                   help="override experiment.repeats, to estimate run-to-run variance")
    p.add_argument("--host", default=None, help="Ollama host, e.g. http://127.0.0.1:11434")
    p.add_argument("--base-url", default=None, help="OpenAI-compatible base URL")
    # Two separate things, because conflating them is how a multi-hour run ends
    # up with no output at all and no way to tell slow from hung.
    p.add_argument("--quiet", action="store_true",
                   help="suppress per-case progress. An arm is hundreds of serial "
                        "generations, so this makes a live run unobservable; the "
                        "results JSONL is flushed per case either way.")
    p.add_argument("--no-summary", action="store_true",
                   help="keep progress, but skip the JSON summary dump at the end")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    root = _repo_root()

    try:
        if args.config:
            config_path = Path(args.config)
        else:
            config_path = config_set_dir(root, args.config_set) / f"{args.precision}.yaml"
    except ConfigError as exc:
        print(f"\nCONFIGURATION ERROR\n{'-' * 70}\n{exc}\n", file=sys.stderr)
        return 2

    overrides = {}
    if args.repeats is not None:
        overrides.setdefault("experiment", {})["repeats"] = args.repeats
    if args.results_root:
        overrides.setdefault("experiment", {})["results_root"] = args.results_root

    try:
        cfg = load_experiment_config(config_path, args.backend, repo_root=root, overrides=overrides)
    except ConfigError as exc:
        print(f"\nCONFIGURATION ERROR\n{'-' * 70}\n{exc}\n", file=sys.stderr)
        return 2

    # Recorded into every arm's metadata, so the findings report can print the
    # command that reproduces THIS experiment rather than the default one.
    cfg.config_set = None if args.config else args.config_set

    # Give each config set its own results root, so an alternate model cannot land
    # on top of the default set's arms. Applied after loading rather than as an
    # override, so it suffixes whatever base.yaml declares instead of assuming it.
    # An explicit --results-root or --config means the caller has taken charge of
    # the destination, and neither is second-guessed.
    if not args.results_root and not args.config:
        suffix = config_set_suffix(args.config_set)
        if suffix and not cfg.results_root.name.endswith(suffix):
            cfg.results_root = cfg.results_root.parent / (cfg.results_root.name + suffix)

    if args.host:
        cfg.backend.options["host"] = args.host
    if args.base_url:
        cfg.backend.options["base_url"] = args.base_url

    if args.limit:
        print(
            f"\n  WARNING: --limit {args.limit} is in effect. This arm will NOT be "
            "comparable\n  with a full arm; the limit is recorded in metadata.json.\n"
        )

    try:
        runner = ExperimentRunner(cfg, limit=args.limit, verbose=not args.quiet)
        summary = runner.run(args.suite)
    except ConfigError as exc:
        print(f"\nARM NOT RUNNABLE\n{'-' * 70}\n{exc}\n", file=sys.stderr)
        # Record WHY, next to the results, so the gap is self-documenting. The
        # reason otherwise lives only in this terminal's scrollback and in a
        # config file, and the findings report could say no more than "not run"
        # -- which reads like an omission rather than a refusal on principle.
        _write_not_run_marker(cfg, args, str(exc))
        return 3
    except BackendError as exc:
        print(f"\nBACKEND ERROR\n{'-' * 70}\n{exc}\n", file=sys.stderr)
        return 4

    if not args.quiet and not args.no_summary:
        print("\n" + json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
