"""Degradation figures, rendered from aggregate.json only.

Two rules govern every figure here, and both exist to stop a chart overstating
what the data supports:

  1. **Rate axes are fixed to 0-100%, never auto-scaled.** Auto-scaling a y-axis
     onto the occupied range is the standard way to make a 2-point difference
     look like a collapse. Where a zoomed view genuinely helps, it is produced as
     a clearly labelled SECOND panel beside the full-scale one, never instead of it.
  2. **Every point carries its Wilson 95% interval.** A degradation curve without
     intervals invites the reader to see a trend in noise, which at these sample
     sizes is exactly the risk.

Colours come from a palette validated for colour-vision deficiency (adjacent-pair
CVD dE >= 8, normal-vision dE >= 15). Because three slots sit below 3:1 contrast on
a light surface, every series is ALSO direct-labelled and marked -- identity is
never carried by colour alone.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

__all__ = ["render_all", "PRECISION_ORDER"]

PRECISION_ORDER = ["bf16", "fp8", "q8", "q4"]
PRECISION_LABEL = {"bf16": "BF16", "fp8": "FP8", "q8": "Q8", "q4": "Q4"}

# Validated categorical palette (light surface). Assigned in fixed order, never cycled.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e3e2df"
SEQ_HUE = "#2a78d6"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_MUTED,
    "text.color": INK,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "semibold",
    "figure.dpi": 160,
})


def _style(ax: "plt.Axes") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)


def _watermark(fig: "plt.Figure", synthetic: bool) -> None:
    if not synthetic:
        return
    fig.text(
        0.5, 0.5, "SYNTHETIC — NOT A RESULT",
        fontsize=30, color="#e34948", alpha=0.16, ha="center", va="center",
        rotation=24, weight="bold", zorder=10,
    )


def _present(agg: Dict[str, Any], suite: str) -> List[str]:
    per_precision = (agg.get("metrics") or {}).get(suite, {})
    return [p for p in PRECISION_ORDER if p in per_precision]


def _series(agg: Dict[str, Any], suite: str, metric: str,
            precisions: Sequence[str]) -> Tuple[List[float], List[float], List[float], List[int]]:
    """Return (values, yerr_low, yerr_high, denominators) as percentages."""
    per_precision = (agg.get("metrics") or {}).get(suite, {})
    values, lo, hi, ns = [], [], [], []
    for precision in precisions:
        node: Any = per_precision.get(precision, {})
        for part in metric.split("."):
            node = (node or {}).get(part, {}) if isinstance(node, dict) else {}
        value = (node or {}).get("value")
        values.append(float("nan") if value is None else value * 100)
        cl, ch = (node or {}).get("ci_low"), (node or {}).get("ci_high")
        if value is None or cl is None:
            lo.append(0.0); hi.append(0.0)
        else:
            lo.append(max(0.0, (value - cl) * 100)); hi.append(max(0.0, (ch - value) * 100))
        ns.append(int((node or {}).get("denominator") or 0))
    return values, lo, hi, ns


def _line_panel(
    ax: "plt.Axes", precisions: Sequence[str],
    series: List[Tuple[str, List[float], List[float], List[float]]],
    title: str, ylabel: str,
) -> None:
    x = list(range(len(precisions)))
    for idx, (label, values, lo, hi) in enumerate(series):
        colour = SERIES[idx % len(SERIES)]
        ax.errorbar(
            x, values, yerr=[lo, hi], color=colour, linewidth=2.0,
            marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5,
            capsize=4, elinewidth=1.2, ecolor=colour, alpha=0.95, label=label, zorder=3,
        )
        # Direct label: identity is never carried by colour alone.
        finite = [i for i, v in enumerate(values) if v == v]
        if finite:
            last = finite[-1]
            ax.annotate(
                label, (x[last], values[last]), textcoords="offset points",
                xytext=(8, 0), va="center", fontsize=8, color=INK_MUTED,
            )
    ax.set_xticks(x)
    ax.set_xticklabels([PRECISION_LABEL.get(p, p.upper()) for p in precisions])
    ax.set_xlim(-0.35, len(precisions) - 1 + 0.9)
    ax.set_ylim(0, 100)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.set_xlabel("precision  (higher fidelity  →  lower fidelity)")
    _style(ax)


def _save(fig: "plt.Figure", out_dir: Path, name: str, synthetic: bool,
          footnote: str = "", axis_note: Optional[str] = None) -> Path:
    _watermark(fig, synthetic)
    note = axis_note if axis_note is not None else (
        "Rate axis fixed to 0–100%; not truncated. Bars are Wilson 95% intervals."
    )
    if footnote:
        note += "  " + footnote
    # Wrapped: an unwrapped one-line footnote forces the whole figure wider than
    # the plot needs, squashing the data into a corner.
    lines = textwrap.wrap(note, width=118)
    fig.text(0.01, 0.005, "\n".join(lines), fontsize=7, color=INK_MUTED,
             ha="left", va="bottom")
    # Reserve vertical space in proportion to the note's actual height, so a long
    # footnote cannot overlap the bottom panel's tick labels.
    reserved = min(0.14, 0.018 * len(lines) + 0.015)
    fig.tight_layout(rect=(0, reserved, 1, 1))
    path = out_dir / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


def fig_guardrail(agg: Dict[str, Any], out_dir: Path, synthetic: bool) -> Optional[Path]:
    precisions = _present(agg, "ps1")
    if not precisions:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    v, vlo, vhi, ns = _series(agg, "ps1", "violation_rate", precisions)
    _line_panel(axes[0], precisions, [("violation rate", v, vlo, vhi)],
                "PS-1 guardrail violation rate vs precision", "violation rate (%)  ↓ better")

    c, clo, chi, _ = _series(agg, "ps1", "compliance_rate", precisions)
    b, blo, bhi, _ = _series(agg, "ps1", "benign_refusal_rate", precisions)
    _line_panel(axes[1], precisions,
                [("compliance", c, clo, chi), ("benign over-refusal", b, blo, bhi)],
                "Adherence vs over-refusal", "rate (%)")
    axes[1].legend(frameon=False, fontsize=8, loc="center left")

    return _save(fig, out_dir, "01_guardrail_adherence_vs_precision.png", synthetic,
                 f"PS-1 adversarial n={ns[0] if ns else 0} per arm. Compliance and "
                 "over-refusal must be read together: a model that refuses everything "
                 "scores perfectly on violation rate.")


def fig_structured(agg: Dict[str, Any], out_dir: Path, synthetic: bool) -> Optional[Path]:
    precisions = _present(agg, "ps3")
    if not precisions:
        return None
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    series = []
    for label, metric in [
        ("structured-output validity", "structured_output_validity"),
        ("correct-tool rate", "correct_tool_rate"),
        ("argument accuracy", "argument_accuracy"),
        ("task success (end-to-end)", "task_success_rate"),
    ]:
        values, lo, hi, _ = _series(agg, "ps3", metric, precisions)
        series.append((label, values, lo, hi))
    _line_panel(ax, precisions, series,
                "PS-3 structured output vs precision", "rate (%)  ↑ better")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    return _save(fig, out_dir, "02_structured_output_vs_precision.png", synthetic,
                 "Task success is the end-to-end number; correct-tool rate ignores "
                 "argument errors and is always the more flattering of the two.")


def fig_failure_modes(agg: Dict[str, Any], out_dir: Path, synthetic: bool) -> Optional[Path]:
    precisions = _present(agg, "ps3")
    if not precisions:
        return None
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    modes = [("malformed", "malformed_argument_rate"), ("wrong tool", "wrong_tool_rate"),
             ("wrong arguments", "wrong_argument_rate"), ("spurious call", "spurious_call_rate"),
             ("missed call", "missed_call_rate")]
    width = 0.15
    x = list(range(len(precisions)))
    for idx, (label, metric) in enumerate(modes):
        values, lo, hi, _ = _series(agg, "ps3", metric, precisions)
        offs = [xi + (idx - (len(modes) - 1) / 2) * width for xi in x]
        ax.bar(offs, values, width=width * 0.86, color=SERIES[idx], label=label,
               yerr=[lo, hi], capsize=2, error_kw={"elinewidth": 0.9, "ecolor": INK_MUTED},
               zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([PRECISION_LABEL.get(p, p.upper()) for p in precisions])
    ax.set_ylim(0, 100)
    ax.set_ylabel("rate (%)  ↓ better")
    ax.set_xlabel("precision  (higher fidelity  →  lower fidelity)")
    ax.set_title("PS-3 failure modes by precision", loc="left")
    ax.legend(frameon=False, fontsize=8, ncol=3)
    _style(ax)
    return _save(fig, out_dir, "03_ps3_failure_modes.png", synthetic,
                 "Separated because the fixes differ: malformed output is a parser or "
                 "grammar problem; wrong arguments is a comprehension problem.")


def fig_language(agg: Dict[str, Any], out_dir: Path, synthetic: bool) -> Optional[Path]:
    p1 = _present(agg, "ps1")
    p3 = _present(agg, "ps3")
    if not p1 and not p3:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    if p1:
        series = []
        for idx, lang in enumerate(["en", "hi", "hinglish", "mr"]):
            values, lo, hi, _ = _series(agg, "ps1", f"language_violation_rate.{lang}", p1)
            if any(v == v for v in values):
                series.append((lang, values, lo, hi))
        _line_panel(axes[0], p1, series, "PS-1 violation rate by language",
                    "violation rate (%)  ↓ better")
        axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    else:
        axes[0].axis("off")

    if p3:
        series = []
        for lang in ["en", "hi", "hinglish", "mr"]:
            values, lo, hi, _ = _series(agg, "ps3", f"by_language.{lang}.task_success_rate", p3)
            if any(v == v for v in values):
                series.append((lang, values, lo, hi))
        _line_panel(axes[1], p3, series, "PS-3 task success by language", "rate (%)  ↑ better")
        axes[1].legend(frameon=False, fontsize=8, loc="lower left")
    else:
        axes[1].axis("off")

    return _save(fig, out_dir, "04_language_breakdown.png", synthetic,
                 "Per-language cells are roughly a quarter of the suite; treat as "
                 "directional unless the interval says otherwise.")


def fig_english_indic(agg: Dict[str, Any], out_dir: Path, synthetic: bool) -> Optional[Path]:
    p1, p3 = _present(agg, "ps1"), _present(agg, "ps3")
    if not p1 and not p3:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    for ax, suite, precisions, metric, title, ylabel in [
        (axes[0], "ps1", p1, "english_indic_delta", "PS-1 violation rate: English vs Indic",
         "violation rate (%)  ↓ better"),
        (axes[1], "ps3", p3, "english_indic_delta", "PS-3 task success: English vs Indic",
         "rate (%)  ↑ better"),
    ]:
        if not precisions:
            ax.axis("off")
            continue
        series = []
        for label, key in [("English", "english"), ("Indic (hi/hinglish/mr)", "indic")]:
            values, lo, hi, _ = _series(agg, suite, f"{metric}.{key}", precisions)
            series.append((label, values, lo, hi))
        _line_panel(ax, precisions, series, title, ylabel)
        ax.legend(frameon=False, fontsize=8, loc="center left")

    return _save(fig, out_dir, "05_english_vs_indic.png", synthetic,
                 "A widening gap as precision falls would indicate quantization damages "
                 "Indic-language behaviour disproportionately.")


def fig_category_heatmap(agg: Dict[str, Any], out_dir: Path, synthetic: bool) -> Optional[Path]:
    precisions = _present(agg, "ps1")
    if not precisions:
        return None
    per_precision = agg["metrics"]["ps1"]
    categories = sorted({
        c for p in precisions
        for c in (per_precision[p].get("category_violation_rate") or {})
    })
    if not categories:
        return None

    matrix: List[List[float]] = []
    annotations: List[List[str]] = []
    for category in categories:
        row, labels = [], []
        for precision in precisions:
            node = (per_precision[precision].get("category_violation_rate") or {}).get(category, {})
            value, n = node.get("value"), node.get("denominator") or 0
            row.append(float("nan") if value is None else value * 100)
            labels.append("n/a" if value is None else f"{value * 100:.0f}%\nn={n}")
        matrix.append(row)
        annotations.append(labels)

    fig, ax = plt.subplots(figsize=(1.5 * len(precisions) + 3.2, 0.62 * len(categories) + 2.4))
    # Sequential: one hue, light -> dark. Never a rainbow.
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", ["#f4f8fd", SEQ_HUE])
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=100, aspect="auto")

    ax.set_xticks(range(len(precisions)))
    ax.set_xticklabels([PRECISION_LABEL.get(p, p.upper()) for p in precisions])
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories)
    for i in range(len(categories)):
        for j in range(len(precisions)):
            value = matrix[i][j]
            ax.text(j, i, annotations[i][j], ha="center", va="center", fontsize=7.5,
                    color="#ffffff" if value == value and value > 55 else INK)
    ax.set_title("PS-1 violation rate by category and precision", loc="left")
    ax.set_xlabel("precision  (higher fidelity  →  lower fidelity)")
    fig.colorbar(im, ax=ax, label="violation rate (%)", fraction=0.045)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([x - 0.5 for x in range(1, len(precisions))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(categories))], minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)

    return _save(fig, out_dir, "06_ps1_category_heatmap.png", synthetic,
                 "Per-category n is small by construction; every cell here is "
                 "directional. Values are annotated so the figure is readable without "
                 "relying on colour.")


def fig_degradation_bars(agg: Dict[str, Any], out_dir: Path, synthetic: bool) -> Optional[Path]:
    """Headline degradation vs the reference, with the pre-registered threshold drawn."""
    analyses = (agg.get("degradation") or {}).get("analyses") or {}
    rows: List[Tuple[str, str, float, List[Tuple[str, float, bool]]]] = []
    for suite, metric_analyses in analyses.items():
        for analysis in metric_analyses:
            points = [(p["precision"], (p.get("degradation") or 0.0) * 100, bool(p.get("past_cliff")))
                      for p in analysis["points"] if not p.get("is_reference")]
            if points:
                rows.append((suite, analysis["metric"], analysis["threshold"] * 100, points))
    if not rows:
        return None

    fig, axes = plt.subplots(len(rows), 1, figsize=(7.6, 2.5 * len(rows)), squeeze=False)
    for ax, (suite, metric, threshold, points) in zip(axes[:, 0], rows):
        labels = [PRECISION_LABEL.get(p, p.upper()) for p, _, _ in points]
        values = [v for _, v, _ in points]
        # Colour encodes the verdict, which is a status, not a series identity.
        colours = ["#e34948" if past else "#2a78d6" for _, _, past in points]
        bars = ax.bar(labels, values, color=colours, width=0.55, zorder=3)
        ax.axhline(threshold, color=INK_MUTED, linestyle="--", linewidth=1.1, zorder=2)
        # Axes-fraction x, data-space y: pins the label to the threshold line
        # regardless of how many bars the panel has.
        ax.text(0.012, threshold, f"pre-registered threshold {threshold:.0f} pp",
                transform=ax.get_yaxis_transform(), fontsize=7.5, color=INK_MUTED,
                va="bottom", ha="left", zorder=4)
        ax.axhline(0, color=GRID, linewidth=1)
        # Headroom BEFORE annotating, so a negative bar's label cannot land on the
        # x tick labels at the axes floor.
        ax.margins(y=0.28)
        for bar, (_, value, past) in zip(bars, points):
            ax.annotate(f"{value:+.1f} pp" + ("  ✱" if past else ""),
                        (bar.get_x() + bar.get_width() / 2, value),
                        textcoords="offset points", xytext=(0, 5 if value >= 0 else -5),
                        ha="center", va="bottom" if value >= 0 else "top",
                        fontsize=8, color=INK)
        ax.set_title(f"{suite.upper()} · {metric} — degradation vs "
                     f"{agg['degradation']['reference_precision'].upper()}", loc="left")
        ax.set_ylabel("worse  →  (pp)")
        ax.tick_params(axis="x", pad=6)
        _style(ax)

    legend = [Patch(facecolor="#e34948", label="past the cliff (threshold met AND CI excludes 0)"),
              Patch(facecolor="#2a78d6", label="not past the cliff")]
    axes[0, 0].legend(handles=legend, frameon=False, fontsize=7.5, loc="upper left")
    return _save(
        fig, out_dir, "07_degradation_vs_reference.png", synthetic,
        "✱ marks a precision meeting BOTH cliff conditions: degradation at or above "
        "the threshold AND a Newcombe 95% interval for the difference that excludes zero.",
        axis_note=(
            "Y axis is percentage-POINT change vs the reference arm, not a rate; positive "
            "is always worse, whatever the metric's natural direction. Each panel is "
            "scaled to its own effect size, so panel heights are NOT comparable to "
            "one another — read the annotated values."
        ),
    )


def render_all(aggregate_path: Path, out_dir: Path) -> List[Path]:
    agg = json.loads(Path(aggregate_path).read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)
    synthetic = bool(agg.get("synthetic"))

    written: List[Path] = []
    for builder in (fig_guardrail, fig_structured, fig_failure_modes, fig_language,
                    fig_english_indic, fig_category_heatmap, fig_degradation_bars):
        path = builder(agg, out_dir, synthetic)
        if path:
            written.append(path)
    return written
