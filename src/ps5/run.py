"""CLI entry point:  python -m ps5.run --precision q4 --backend ollama --suite ps1"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from .config import ConfigError, load_experiment_config
from .backends.base import BackendError
from .runner import ExperimentRunner

PRECISIONS = ["q4", "q8", "fp8", "bf16"]
BACKENDS = ["ollama", "vllm", "openai_compat", "mock"]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m ps5.run",
        description="Run one PS-5 precision arm over the PS-1 and/or PS-3 suites.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples
--------
  # validate the whole pipeline with no GPU and no model (FABRICATED output)
  python -m ps5.run --precision bf16 --backend mock --suite ps1 ps3

  # a real arm on local Ollama
  python -m ps5.run --precision q4 --backend ollama --suite ps1 ps3

  # FP8 requires a CUDA GPU with compute capability >= 8.9, served by vLLM
  python -m ps5.run --precision fp8 --backend vllm --suite ps1 ps3
""",
    )
    p.add_argument("--precision", required=True, choices=PRECISIONS)
    p.add_argument("--backend", required=True, choices=BACKENDS)
    p.add_argument("--suite", nargs="+", default=None, choices=["ps1", "ps3"],
                   help="suites to run (default: whatever the config lists)")
    p.add_argument("--config", default=None,
                   help="override the experiment config path")
    p.add_argument("--results-root", default=None,
                   help="override the results directory root")
    p.add_argument("--limit", type=int, default=None,
                   help="run only the first N cases per suite. FOR SMOKE TESTS ONLY: "
                        "a limited run must never be compared against a full one.")
    p.add_argument("--repeats", type=int, default=None,
                   help="override experiment.repeats, to estimate run-to-run variance")
    p.add_argument("--host", default=None, help="Ollama host, e.g. http://127.0.0.1:11434")
    p.add_argument("--base-url", default=None, help="OpenAI-compatible base URL")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    root = _repo_root()
    config_path = Path(args.config) if args.config else root / "configs" / "experiments" / f"{args.precision}.yaml"

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
        return 3
    except BackendError as exc:
        print(f"\nBACKEND ERROR\n{'-' * 70}\n{exc}\n", file=sys.stderr)
        return 4

    if not args.quiet:
        print("\n" + json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
