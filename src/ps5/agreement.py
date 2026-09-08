"""Inter-rater agreement between the automated PS-1 scorer and human labels.

The specification requires this and judges it explicitly: an automated scorer is
acceptable, but "you must validate it against human labels on a subset and report
the agreement", and judge quality is "measured by agreement with human raters
rather than asserted".

This module computes that agreement. It does not, and cannot, make the scorer
better -- it tells you how much to trust it, which is the point. A high violation
rate measured by a scorer with kappa 0.3 is not a finding.

Cohen's kappa is used rather than raw percent agreement because these labels are
heavily imbalanced: most responses are compliant, so a scorer that answered "no
violation" every time would post ~90% raw agreement while being useless. Kappa
corrects for agreement expected by chance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .metrics import wilson_interval

__all__ = ["cohens_kappa", "agreement_report", "interpret_kappa", "AgreementResult"]


def interpret_kappa(kappa: Optional[float]) -> str:
    """Landis & Koch bands, stated so the number is not over-read."""
    if kappa is None:
        return "undefined"
    if kappa < 0.0:
        return "worse than chance"
    if kappa < 0.21:
        return "slight"
    if kappa < 0.41:
        return "fair"
    if kappa < 0.61:
        return "moderate"
    if kappa < 0.81:
        return "substantial"
    return "almost perfect"


@dataclass
class AgreementResult:
    n: int
    observed_agreement: Optional[float]
    expected_agreement: Optional[float]
    kappa: Optional[float]
    interpretation: str
    # Confusion against the human label treated as ground truth.
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int
    precision: Optional[float]
    recall: Optional[float]
    f1: Optional[float]
    agreement_ci_low: float
    agreement_ci_high: float
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def cohens_kappa(pairs: Sequence[Tuple[bool, bool]]) -> AgreementResult:
    """Cohen's kappa for two binary raters.

    `pairs` are (scorer_label, human_label). The human label is treated as the
    reference for precision/recall, which is a convention rather than a claim
    that humans are infallible -- with a single human rater the "ground truth" is
    one person's judgement, and that limitation is reported alongside.
    """
    n = len(pairs)
    if n == 0:
        return AgreementResult(
            n=0, observed_agreement=None, expected_agreement=None, kappa=None,
            interpretation="undefined", true_positive=0, false_positive=0,
            true_negative=0, false_negative=0, precision=None, recall=None, f1=None,
            agreement_ci_low=0.0, agreement_ci_high=1.0,
            note="No labelled pairs supplied; agreement is unmeasured.",
        )

    tp = sum(1 for s, h in pairs if s and h)
    fp = sum(1 for s, h in pairs if s and not h)
    tn = sum(1 for s, h in pairs if not s and not h)
    fn = sum(1 for s, h in pairs if not s and h)

    agree = tp + tn
    po = agree / n

    # Chance agreement from the marginals.
    scorer_pos = (tp + fp) / n
    human_pos = (tp + fn) / n
    pe = scorer_pos * human_pos + (1 - scorer_pos) * (1 - human_pos)

    if math.isclose(pe, 1.0):
        # Both raters used a single label throughout: kappa is undefined, and
        # reporting 0 or 1 here would both be misleading.
        kappa = None
        note = ("Kappa is undefined: both raters assigned the same label to every "
                "item, so there is no variation for chance correction to work on. "
                "Enlarge or re-stratify the subset.")
    else:
        kappa = (po - pe) / (1 - pe)
        note = ""

    lo, hi = wilson_interval(agree, n)
    return AgreementResult(
        n=n,
        observed_agreement=po,
        expected_agreement=pe,
        kappa=kappa,
        interpretation=interpret_kappa(kappa),
        true_positive=tp, false_positive=fp, true_negative=tn, false_negative=fn,
        precision=(tp / (tp + fp)) if (tp + fp) else None,
        recall=(tp / (tp + fn)) if (tp + fn) else None,
        f1=(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else None,
        agreement_ci_low=lo, agreement_ci_high=hi,
        note=note,
    )


def agreement_report(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Full agreement report from labelled rows.

    Each row needs `scorer_violation` (bool), `human_violation` (bool) and,
    optionally, `target_violation`, `language` and `rater_id`. Rows without a
    human label are counted as unlabelled and excluded rather than assumed.
    """
    labelled = [r for r in rows if r.get("human_violation") is not None]
    unlabelled = len(rows) - len(labelled)

    pairs = [(bool(r["scorer_violation"]), bool(r["human_violation"])) for r in labelled]
    overall = cohens_kappa(pairs)

    def _subgroup(key: str) -> Dict[str, Any]:
        groups: Dict[str, List[Tuple[bool, bool]]] = {}
        for r in labelled:
            groups.setdefault(str(r.get(key) or "unknown"), []).append(
                (bool(r["scorer_violation"]), bool(r["human_violation"]))
            )
        return {k: cohens_kappa(v).to_dict() for k, v in sorted(groups.items())}

    # If more than one human rater labelled the SAME case, we can also measure
    # human-vs-human agreement, which bounds how well any scorer could do.
    by_case: Dict[str, List[Dict[str, Any]]] = {}
    for r in labelled:
        by_case.setdefault(str(r.get("case_id")), []).append(r)
    human_pairs: List[Tuple[bool, bool]] = []
    for _cid, group in by_case.items():
        raters = {str(g.get("rater_id") or "r1"): bool(g["human_violation"]) for g in group}
        if len(raters) >= 2:
            values = list(raters.values())
            human_pairs.append((values[0], values[1]))
    human_vs_human = cohens_kappa(human_pairs) if human_pairs else None

    return {
        "scorer_vs_human": overall.to_dict(),
        "by_category": _subgroup("target_violation"),
        "by_language": _subgroup("language"),
        "human_vs_human": human_vs_human.to_dict() if human_vs_human else None,
        "n_rows": len(rows),
        "n_labelled": len(labelled),
        "n_unlabelled": unlabelled,
        "raters": sorted({str(r.get("rater_id") or "r1") for r in labelled}),
        "caveats": [
            "Human labels are one or two people's judgement, not ground truth. "
            "With a single rater there is no way to separate scorer error from "
            "rater error.",
            "Agreement measured on this subset is assumed to hold on the full "
            "suite. That assumption is only as good as the subset's stratification.",
            "Scorer error is CONSTANT across precision arms, so it biases absolute "
            "violation rates more than it biases the between-precision comparison "
            "that PS-5 asks about. It does not licence quoting the absolute rate "
            "as a safety rate.",
        ],
    }
