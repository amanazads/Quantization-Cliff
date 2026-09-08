"""Capture the hardware and software environment of a run.

PS-5's primary judging criterion is experimental control, and control that is not
recorded is indistinguishable from control that did not happen. Everything here
is collected automatically so that a run's metadata cannot drift from reality
through someone forgetting to update a hand-written note.

Fields that genuinely cannot be determined are recorded as null with a reason,
never guessed.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .hashing import sha256_json

__all__ = ["capture_environment", "hardware_fingerprint", "git_state"]

_TIMEOUT = 10


def _run(cmd: List[str]) -> Optional[str]:
    """Run a command, returning stripped stdout or None. Never raises."""
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=_TIMEOUT, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    out = proc.stdout.strip()
    return out or None


def _package_versions() -> Dict[str, Optional[str]]:
    """Versions of libraries whose numerics or parsing could affect results."""
    names = [
        "ollama", "openai", "httpx", "requests", "vllm", "torch",
        "transformers", "numpy", "matplotlib", "pyyaml", "jsonschema", "pytest",
    ]
    try:
        from importlib.metadata import version, PackageNotFoundError
    except ImportError:  # pragma: no cover
        return {n: None for n in names}

    out: Dict[str, Optional[str]] = {}
    for name in names:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
        except Exception:
            out[name] = None
    return out


def _cpu_info() -> Dict[str, Any]:
    system = platform.system()
    info: Dict[str, Any] = {
        "arch": platform.machine(),
        "processor": platform.processor() or None,
        "logical_cores": os.cpu_count(),
        "model": None,
        "physical_cores": None,
    }
    if system == "Darwin":
        info["model"] = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        cores = _run(["sysctl", "-n", "hw.physicalcpu"])
        info["physical_cores"] = int(cores) if cores and cores.isdigit() else None
    elif system == "Linux":
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as fh:
                for line in fh:
                    if line.lower().startswith("model name"):
                        info["model"] = line.split(":", 1)[1].strip()
                        break
        except OSError:
            pass
    return info


def _memory_info() -> Dict[str, Any]:
    system = platform.system()
    total: Optional[int] = None
    if system == "Darwin":
        raw = _run(["sysctl", "-n", "hw.memsize"])
        total = int(raw) if raw and raw.isdigit() else None
    elif system == "Linux":
        try:
            with open("/proc/meminfo", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("MemTotal:"):
                        total = int(line.split()[1]) * 1024
                        break
        except OSError:
            pass
    return {
        "total_bytes": total,
        "total_gb": round(total / 1024**3, 2) if total else None,
    }


def _accelerator_info() -> Dict[str, Any]:
    """Describe the compute device actually used for inference.

    On Apple silicon this is the unified-memory GPU reached through Metal; there
    is no discrete VRAM figure, and reporting one would be false. On NVIDIA we
    read nvidia-smi. Compute capability matters specifically because FP8 requires
    >= 8.9 for native execution.
    """
    system = platform.system()
    info: Dict[str, Any] = {
        "kind": None,
        "name": None,
        "memory_total_mb": None,
        "memory_note": None,
        "driver_version": None,
        "cuda_version": None,
        "compute_capability": None,
        "fp8_native_supported": None,
    }

    if shutil.which("nvidia-smi"):
        info["kind"] = "cuda"
        raw = _run([
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version,compute_cap",
            "--format=csv,noheader,nounits",
        ])
        if raw:
            first = raw.splitlines()[0]
            parts = [p.strip() for p in first.split(",")]
            if len(parts) >= 4:
                info["name"] = parts[0]
                info["memory_total_mb"] = int(parts[1]) if parts[1].isdigit() else None
                info["driver_version"] = parts[2]
                info["compute_capability"] = parts[3]
                try:
                    info["fp8_native_supported"] = float(parts[3]) >= 8.9
                except ValueError:
                    info["fp8_native_supported"] = None
        nvcc = _run(["nvcc", "--version"])
        if nvcc:
            for line in nvcc.splitlines():
                if "release" in line:
                    info["cuda_version"] = line.split("release")[-1].split(",")[0].strip()
                    break
        if info["cuda_version"] is None:
            smi = _run(["nvidia-smi"])
            if smi and "CUDA Version:" in smi:
                info["cuda_version"] = smi.split("CUDA Version:")[1].split()[0].strip()

    elif system == "Darwin":
        info["kind"] = "metal"
        chip = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        info["name"] = chip
        info["memory_note"] = (
            "Apple silicon uses unified memory; there is no discrete VRAM figure. "
            "GPU-addressable memory is bounded by system RAM and the Metal working-set limit."
        )
        info["fp8_native_supported"] = False

    return info


def git_state(repo_root: Optional[str] = None) -> Dict[str, Any]:
    """Record the exact code revision that produced a result."""
    cwd = repo_root or os.getcwd()

    def _git(*args: str) -> Optional[str]:
        try:
            proc = subprocess.run(
                ["git", "-C", cwd, *args],
                capture_output=True, text=True, timeout=_TIMEOUT, check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return proc.stdout.strip() if proc.returncode == 0 else None

    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {
        "commit": commit,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "is_repo": commit is not None,
        # A dirty tree means the recorded commit does not fully describe the code
        # that ran. Surfaced rather than hidden.
        "dirty": bool(status) if status is not None else None,
        "dirty_files": status.splitlines() if status else [],
    }


def capture_environment(repo_root: Optional[str] = None) -> Dict[str, Any]:
    """Full environment snapshot embedded in every run's metadata.json."""
    env: Dict[str, Any] = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "platform": platform.platform(),
        },
        "python": {
            "version": sys.version.split()[0],
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "cpu": _cpu_info(),
        "memory": _memory_info(),
        "accelerator": _accelerator_info(),
        "packages": _package_versions(),
        "git": git_state(repo_root),
        "hostname": platform.node(),
    }
    env["hardware_fingerprint"] = hardware_fingerprint(env)
    return env


def hardware_fingerprint(env: Dict[str, Any]) -> str:
    """Stable hash of the parts of the environment that must not vary.

    "Identical hardware" is a PS-5 requirement. This makes it checkable: the
    aggregator compares fingerprints across arms and flags any divergence.

    Deliberately excluded: timestamps, hostname, and package versions -- the
    first two vary between runs on the same machine, and package versions are
    compared separately so a matplotlib upgrade does not invalidate a comparison
    of model outputs.
    """
    material = {
        "os_system": env.get("os", {}).get("system"),
        "os_release": env.get("os", {}).get("release"),
        "cpu_model": env.get("cpu", {}).get("model"),
        "cpu_arch": env.get("cpu", {}).get("arch"),
        "logical_cores": env.get("cpu", {}).get("logical_cores"),
        "memory_total_bytes": env.get("memory", {}).get("total_bytes"),
        "accelerator_kind": env.get("accelerator", {}).get("kind"),
        "accelerator_name": env.get("accelerator", {}).get("name"),
        "compute_capability": env.get("accelerator", {}).get("compute_capability"),
    }
    return sha256_json(material)


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(capture_environment(), indent=2, default=str))
