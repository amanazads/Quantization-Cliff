"""Configuration loading for PS-5 experiments.

Design intent: precision is a *configuration value*, never a branch in the
business logic. Nothing outside `backends/` is allowed to ask "am I running Q4?".
The runner, the scorers, the metrics and the aggregator all operate on the same
code path regardless of precision -- which is what makes the comparison a
controlled one.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .hashing import sha256_file, sha256_json

__all__ = [
    "GenerationConfig",
    "PrecisionSpec",
    "BackendSpec",
    "Deviation",
    "ExperimentConfig",
    "load_experiment_config",
    "ConfigError",
    "DEFAULT_CONFIG_SET",
    "config_set_dir",
    "config_set_suffix",
    "available_config_sets",
]


class ConfigError(RuntimeError):
    """Raised when a configuration is malformed or describes an impossible run."""


# --------------------------------------------------------------------------- #
# Config sets
#
# A config set defines one complete experiment configuration: a model family,
# parameters, and the precision arms evaluating it. The primary configuration
# is `qwen2.5-1.5b` (configs/experiments-qwen2.5-1.5b/).
# --------------------------------------------------------------------------- #

DEFAULT_CONFIG_SET = "qwen2.5-1.5b"
_CONFIG_SET_PREFIX = "experiments"


def config_set_dir(repo_root: str | Path, name: str = DEFAULT_CONFIG_SET) -> Path:
    """Directory holding one set's per-precision configs. Raises if it is absent."""
    root = Path(repo_root)
    name = (name or DEFAULT_CONFIG_SET).strip()
    if name in ("", DEFAULT_CONFIG_SET, "default"):
        path = root / "configs" / f"{_CONFIG_SET_PREFIX}-{DEFAULT_CONFIG_SET}"
        if not path.is_dir():
            path = root / "configs" / _CONFIG_SET_PREFIX
    else:
        if "/" in name or "\\" in name or name.startswith("."):
            raise ConfigError(f"Invalid config set name {name!r}: it is a name, not a path.")
        path = root / "configs" / f"{_CONFIG_SET_PREFIX}-{name}"
    if not path.is_dir():
        known = available_config_sets(root)
        raise ConfigError(
            f"No config set named {name!r} (looked in {path}).\n"
            f"Available sets: {', '.join(known) if known else '(none found)'}"
        )
    return path


def config_set_suffix(name: str = DEFAULT_CONFIG_SET) -> str:
    """Suffix appended to results roots so two sets cannot share a directory."""
    name = (name or DEFAULT_CONFIG_SET).strip()
    return "" if name in ("", DEFAULT_CONFIG_SET, "default") else f"-{name}"


def available_config_sets(repo_root: str | Path) -> List[str]:
    configs = Path(repo_root) / "configs"
    names = []
    for child in sorted(configs.iterdir() if configs.is_dir() else []):
        if not child.is_dir():
            continue
        if child.name == _CONFIG_SET_PREFIX:
            names.append(DEFAULT_CONFIG_SET)
        elif child.name.startswith(f"{_CONFIG_SET_PREFIX}-"):
            names.append(child.name[len(_CONFIG_SET_PREFIX) + 1:])
    return names


# --------------------------------------------------------------------------- #
# Dataclasses
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GenerationConfig:
    """Decoding parameters. Identical across every precision by construction."""

    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 1
    seed: int = 20260907
    max_tokens: int = 512
    stop: List[str] = field(default_factory=list)
    transport_retries: int = 2
    transport_retry_backoff_s: float = 2.0
    request_timeout_s: float = 180.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def hash(self) -> str:
        """Hash of the *sampling* parameters only.

        Transport settings (retries, timeouts) are deliberately excluded: they
        affect how hard we try to obtain a response, not what the model computes,
        so a difference there must not block a comparison.
        """
        sampling = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "seed": self.seed,
            "max_tokens": self.max_tokens,
            "stop": list(self.stop),
        }
        return sha256_json(sampling)


@dataclass(frozen=True)
class Deviation:
    """A recorded departure from perfect experimental control.

    Deviations are surfaced in metadata, in the aggregate output and in the
    findings report. They are never suppressed.
    """

    id: str
    severity: str  # "blocking" | "material" | "minor"
    description: str
    impact: Optional[str] = None
    remediation: Optional[str] = None
    condition: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass(frozen=True)
class PrecisionSpec:
    id: str
    label: str
    nominal_bits: int
    family: str
    is_reference: bool = False


@dataclass(frozen=True)
class BackendSpec:
    name: str
    available: bool
    model_tag: Optional[str]
    quantization_format: Optional[str]
    quantization_config: Dict[str, Any]
    deviations: List[Deviation]
    unavailable_reason: Optional[str] = None
    substitution_policy: Optional[str] = None
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    precision: PrecisionSpec
    backend: BackendSpec
    generation: GenerationConfig
    prompt_path: Path
    prompt_version: str
    schemas_path: Path
    manifest_path: Path
    cliff_criterion_path: Path
    metric_spec_version: str
    suite_ids: List[str]
    repeats: int
    results_root: Path
    max_parallel: int
    model_family: str
    model_parameters: str
    model_variant: str
    repo_root: Path
    #: Which config set this arm came from ("default", "qwen2.5-1.5b", ...).
    #: Set by the CLI, so a config loaded directly by path leaves it None.
    config_set: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    # -- derived identity ---------------------------------------------------- #

    @property
    def system_prompt(self) -> str:
        """The Section 6.4 baseline prompt TEMPLATE, verbatim and unrendered."""
        return self.prompt_path.read_text(encoding="utf-8")

    def render_system_prompt(self, context: Optional[Dict[str, Any]] = None) -> str:
        """Substitute a case's borrower context into the baseline prompt.

        The specification's prompt is parameterised with {LENDER}, {NAME}, {DPD},
        {PRODUCT} and {AMOUNT}. Only those five placeholders are substituted; the
        wording around them is never touched, and the hash recorded for
        comparability is the hash of the TEMPLATE, so a differing borrower context
        between cases cannot be mistaken for a differing prompt between arms.

        A missing placeholder value is an error rather than a silent blank: an
        arm that ran with "outstanding {AMOUNT}" literally in the prompt would be
        measuring something different from one that did not.
        """
        template = self.system_prompt
        placeholders = set(re.findall(r"\{([A-Z_]+)\}", template))
        context = context or {}
        missing = placeholders - set(context)
        if missing:
            raise ConfigError(
                f"Baseline prompt placeholders {sorted(missing)} have no value in this "
                "case's `context`. Every case must supply LENDER, NAME, DPD, PRODUCT "
                "and AMOUNT; rendering a placeholder as blank would silently change "
                "the prompt the model sees."
            )
        rendered = template
        for key in placeholders:
            rendered = rendered.replace("{" + key + "}", str(context[key]))
        return rendered

    @property
    def system_prompt_hash(self) -> str:
        return sha256_file(self.prompt_path)

    @property
    def tool_schema_hash(self) -> str:
        return sha256_file(self.schemas_path)

    @property
    def manifest_hash(self) -> str:
        return sha256_file(self.manifest_path)

    @property
    def cliff_criterion_hash(self) -> str:
        return sha256_file(self.cliff_criterion_path)

    @property
    def model_id(self) -> str:
        return f"{self.model_family}-{self.model_parameters}-{self.model_variant}"

    def results_dir(self) -> Path:
        return self.results_root / self.precision.id

    def assert_runnable(self) -> None:
        """Refuse, loudly, to run an arm the backend genuinely cannot serve.

        This is the guard that stops the single most damaging failure mode in a
        quantization study: quietly serving a different format under the label of
        the one that was requested.
        """
        if not self.backend.available:
            raise ConfigError(
                f"Precision '{self.precision.id}' is NOT available on backend "
                f"'{self.backend.name}'.\n\n"
                f"Reason: {self.backend.unavailable_reason}\n\n"
                f"Substitution policy: {self.backend.substitution_policy}\n\n"
                "This arm must be reported as NOT RUN. Do not substitute a "
                "different precision under this label."
            )
        if self.backend.model_tag is None:
            raise ConfigError(
                f"Precision '{self.precision.id}' on backend '{self.backend.name}' "
                "declares no model_tag; nothing can be served."
            )


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _load_yaml_with_inheritance(path: Path, _seen: Optional[set] = None) -> Dict[str, Any]:
    _seen = _seen or set()
    resolved = path.resolve()
    if resolved in _seen:
        raise ConfigError(f"Circular config inheritance involving {resolved}")
    _seen.add(resolved)

    if not resolved.exists():
        raise ConfigError(f"Config file not found: {resolved}")

    data = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Config {resolved} must be a mapping at top level")

    parent_ref = data.pop("inherits", None)
    if parent_ref:
        parent_path = (resolved.parent / parent_ref).resolve()
        parent = _load_yaml_with_inheritance(parent_path, _seen)
        data = _deep_merge(parent, data)
    return data


def _parse_deviations(items: Any) -> List[Deviation]:
    out: List[Deviation] = []
    for item in items or []:
        if not isinstance(item, dict):
            raise ConfigError(f"Deviation entries must be mappings, got {type(item)}")
        missing = {"id", "severity", "description"} - set(item)
        if missing:
            raise ConfigError(f"Deviation missing required keys: {sorted(missing)}")
        if item["severity"] not in {"blocking", "material", "minor"}:
            raise ConfigError(
                f"Deviation {item['id']} has invalid severity {item['severity']!r}; "
                "expected one of blocking/material/minor"
            )
        out.append(
            Deviation(
                id=item["id"],
                severity=item["severity"],
                description=item["description"],
                impact=item.get("impact"),
                remediation=item.get("remediation"),
                condition=item.get("condition"),
            )
        )
    return out


def load_experiment_config(
    config_path: str | Path,
    backend_name: str,
    repo_root: str | Path | None = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> ExperimentConfig:
    """Load an experiment config for one precision on one backend."""
    config_path = Path(config_path)
    root = Path(repo_root) if repo_root else _infer_repo_root(config_path)
    data = _load_yaml_with_inheritance(config_path)
    if overrides:
        data = _deep_merge(data, overrides)

    # -- precision ---------------------------------------------------------- #
    p = data.get("precision")
    if not isinstance(p, dict) or "id" not in p:
        raise ConfigError(f"{config_path} must define a `precision` mapping with an `id`")
    precision = PrecisionSpec(
        id=p["id"],
        label=p.get("label", p["id"].upper()),
        nominal_bits=int(p.get("nominal_bits", 0)),
        family=p.get("family", "unknown"),
        is_reference=bool(p.get("is_reference", False)),
    )

    # -- backend ------------------------------------------------------------ #
    backends = data.get("backends") or {}
    if backend_name not in backends:
        raise ConfigError(
            f"Backend '{backend_name}' is not defined for precision "
            f"'{precision.id}'. Defined backends: {sorted(backends)}"
        )
    b = backends[backend_name] or {}
    backend = BackendSpec(
        name=backend_name,
        available=bool(b.get("available", False)),
        model_tag=b.get("model_tag"),
        quantization_format=b.get("quantization_format"),
        quantization_config=b.get("quantization_config") or {},
        deviations=_parse_deviations(b.get("deviations")),
        unavailable_reason=b.get("unavailable_reason"),
        substitution_policy=b.get("substitution_policy"),
        options=b.get("options") or {},
    )

    # -- generation --------------------------------------------------------- #
    g = data.get("generation") or {}
    known = set(GenerationConfig.__dataclass_fields__)
    unknown = set(g) - known
    if unknown:
        raise ConfigError(
            f"Unknown generation keys in {config_path}: {sorted(unknown)}. "
            "Silently ignoring these would make the run non-reproducible."
        )
    generation = GenerationConfig(**g)

    exp = data.get("experiment") or {}
    prompt = data.get("prompt") or {}
    scoring = data.get("scoring") or {}
    model = data.get("model") or {}

    def _p(value: str) -> Path:
        return (root / value).resolve()

    cfg = ExperimentConfig(
        precision=precision,
        backend=backend,
        generation=generation,
        prompt_path=_p(prompt.get("path", "prompts/collections_agent_v1.md")),
        prompt_version=prompt.get("version", "unknown"),
        schemas_path=_p((data.get("schemas") or {}).get("path", "schemas/tools.json")),
        manifest_path=_p((data.get("manifest") or {}).get("path", "data/evaluation_manifest.json")),
        cliff_criterion_path=_p(scoring.get("cliff_criterion_path", "configs/cliff_criterion.yaml")),
        metric_spec_version=scoring.get("metric_spec_version", "unknown"),
        suite_ids=list(exp.get("suite_ids", ["ps1", "ps3"])),
        repeats=int(exp.get("repeats", 1)),
        results_root=_p(exp.get("results_root", "results")),
        max_parallel=int((data.get("concurrency") or {}).get("max_parallel", 1)),
        model_family=model.get("family", "unknown"),
        model_parameters=model.get("parameters", "unknown"),
        model_variant=model.get("variant", "unknown"),
        repo_root=root,
        raw=data,
    )

    if cfg.repeats < 1:
        raise ConfigError("experiment.repeats must be >= 1")
    if cfg.max_parallel < 1:
        raise ConfigError("concurrency.max_parallel must be >= 1")
    return cfg


def _infer_repo_root(config_path: Path) -> Path:
    """Walk upward until a directory containing `configs/` and `src/` is found."""
    for candidate in [config_path.resolve()] + list(config_path.resolve().parents):
        if (candidate / "configs").is_dir() and (candidate / "src").is_dir():
            return candidate
    return config_path.resolve().parents[2]
