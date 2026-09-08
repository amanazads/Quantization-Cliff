#!/usr/bin/env python3
"""Check every arm's readiness BEFORE running anything.

Without this, a missing model tag is discovered one arm at a time: you fix the
first, wait, hit the second, fix it, wait again. Worse, a partially-completed
sweep invites the temptation to compare arms with unequal coverage.

This reports the whole picture in one pass and prints the exact commands needed
to close the gaps. It never runs a model and never changes anything.

  python scripts/preflight.py --backend ollama
  python scripts/preflight.py --backend vllm
  python scripts/preflight.py --backend ollama --config-set qwen2.5-1.5b

Exit codes:
  0  every arm that CAN run on this backend is ready
  1  at least one runnable arm is missing its model
  2  the backend itself is unreachable
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ps5.config import (  # noqa: E402
    DEFAULT_CONFIG_SET,
    ConfigError,
    available_config_sets,
    config_set_dir,
    load_experiment_config,
)

REPO = Path(__file__).resolve().parents[1]
PRECISIONS = ["bf16", "fp8", "q8", "q4"]


def _available_models(backend: str, host: Optional[str], base_url: Optional[str]
                      ) -> Tuple[Optional[List[str]], Optional[str]]:
    """Return (served model names, error). None for the list means unreachable."""
    import requests

    try:
        if backend == "ollama":
            url = (host or "http://127.0.0.1:11434").rstrip("/") + "/api/tags"
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            return sorted(m.get("name", "") for m in resp.json().get("models", [])), None
        if backend in ("vllm", "openai_compat"):
            url = (base_url or "http://127.0.0.1:8000/v1").rstrip("/") + "/models"
            resp = requests.get(url, timeout=20, headers={"Authorization": "Bearer EMPTY"})
            resp.raise_for_status()
            return sorted(m.get("id", "") for m in resp.json().get("data", [])), None
        return [], None  # mock: nothing to check
    except Exception as exc:  # noqa: BLE001 -- any failure means "cannot verify"
        return None, str(exc)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", required=True,
                        choices=["ollama", "vllm", "openai_compat", "mock"])
    parser.add_argument("--host", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--config-set", default=DEFAULT_CONFIG_SET,
                        help="which four-arm experiment to check "
                             f"(available: {', '.join(available_config_sets(REPO)) or 'none'})")
    args = parser.parse_args()

    try:
        config_dir = config_set_dir(REPO, args.config_set)
    except ConfigError as exc:
        print(f"\nCONFIGURATION ERROR\n{'-' * 70}\n{exc}\n", file=sys.stderr)
        return 2

    served, error = _available_models(args.backend, args.host, args.base_url)

    print(f"backend: {args.backend}")
    print(f"config set: {args.config_set}  ({config_dir.relative_to(REPO)})")
    if served is None:
        print(f"\n  UNREACHABLE: {error}\n")
        if args.backend == "ollama":
            print("  Start it with `ollama serve`, or check that the Ollama app is running.")
        else:
            print("  Start your vLLM server first; see README.md section 4.5.")
        return 2
    if args.backend != "mock":
        print(f"models present: {served or '(none)'}\n")

    ready: List[str] = []
    missing: List[Tuple[str, str]] = []
    blocked: List[Tuple[str, str]] = []

    for precision in PRECISIONS:
        try:
            cfg = load_experiment_config(
                config_dir / f"{precision}.yaml", args.backend, repo_root=REPO)
        except ConfigError as exc:
            blocked.append((precision, str(exc)))
            continue

        if not cfg.backend.available:
            reason = (cfg.backend.unavailable_reason or "declared unavailable").strip()
            blocked.append((precision, reason))
            print(f"  {precision:>5}  BLOCKED   {reason.splitlines()[0][:88]}")
            continue

        tag = cfg.backend.model_tag or ""
        if args.backend == "mock" or tag in served:
            ready.append(precision)
            print(f"  {precision:>5}  READY     {tag}")
        else:
            missing.append((precision, tag))
            print(f"  {precision:>5}  MISSING   {tag}")

        for dev in cfg.backend.deviations:
            if dev.severity in ("material", "blocking"):
                print(f"         └─ DEVIATION [{dev.severity}] {dev.id}: "
                      f"{dev.description.strip().splitlines()[0][:80]}")

    print()
    print(f"ready: {len(ready)}/{len(PRECISIONS)}   "
          f"missing: {len(missing)}   blocked: {len(blocked)}")

    if missing:
        print("\nFetch the missing models before running, so every arm covers the same cases:\n")
        for _precision, tag in missing:
            if args.backend == "ollama":
                print(f"  ollama pull {tag}")
            else:
                print(f"  serve {tag}")
        print()

    if blocked:
        print("Blocked arms are a documented GAP in precision coverage, not a null result.")
        print("They will be reported as NOT RUN in the findings report.\n")

    if len(ready) < 2:
        print("WARNING: fewer than two arms are ready. Degradation and cliff analysis "
              "need at least the reference arm plus one other.\n")

    if "bf16" not in ready and args.backend != "mock":
        print("WARNING: the REFERENCE arm (bf16) is not ready. Every delta is measured "
              "against it, so without it no degradation can be computed at all.\n")

    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
