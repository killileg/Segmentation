"""Registry of branches and fusion methods.

This module is what makes methods individually selectable. A *branch* is a trainable base
network; a *method* is a reported result, defined by which branches it consumes and how it
combines them. Selecting a subset of methods therefore determines a minimal set of branches to
train -- asking only for ``late_fusion`` trains two branches instead of five.

Adding a new method usually means adding one entry to :data:`METHODS`; adding a new base network
means one entry in :data:`BRANCHES`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class BranchSpec:
    """A trainable base network.

    Attributes:
        backbone: key into :data:`msfusion.models.BACKBONES` -- ``"plain"`` or ``"ssfb"``.
        modalities: input volumes stacked as channels. One entry == single-modality,
            two entries == early fusion inside a single network.
        roi: which ROI mask supervises and masks this branch.
        splits: which modality's train/val/test ranges this branch follows.
    """

    backbone: str
    modalities: Tuple[str, ...]
    roi: str
    splits: str

    @property
    def in_channels(self) -> int:
        return len(self.modalities)


@dataclass(frozen=True)
class MethodSpec:
    """A reported result.

    Attributes:
        branches: branch names this method consumes, in order.
        fusion: how the branch outputs are combined.
            ``"identity"`` -- use the single branch's probabilities as-is.
            ``"mean"`` -- average the branches' softmax probabilities (late fusion).
            ``"meta"`` -- train a :class:`~msfusion.models.MetaLearner` on the branches'
            probabilities and logits over the validation range, then apply it to test.
        description: one-line summary, printed by ``--list-methods``.
    """

    branches: Tuple[str, ...]
    fusion: str
    description: str = ""

    @property
    def needs_meta(self) -> bool:
        return self.fusion == "meta"

    @property
    def needs_logits(self) -> bool:
        return self.fusion == "meta"


# ---------------------------------------------------------------------------
# Branches
# ---------------------------------------------------------------------------
BRANCHES: Dict[str, BranchSpec] = {
    "xray_plain": BranchSpec("plain", ("xray",), roi="xray", splits="xray"),
    "neutron_plain": BranchSpec("plain", ("neutron",), roi="neutron", splits="neutron"),
    "xray_ssfb": BranchSpec("ssfb", ("xray",), roi="xray", splits="xray"),
    "neutron_ssfb": BranchSpec("ssfb", ("neutron",), roi="neutron", splits="neutron"),
    "early_fusion": BranchSpec("plain", ("xray", "neutron"), roi="neutron", splits="neutron"),
}


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------
METHODS: Dict[str, MethodSpec] = {
    "single_modality_x": MethodSpec(
        branches=("xray_plain",),
        fusion="identity",
        description="X-ray only, no fusion at all -- the baseline every other method must beat.",
    ),
    "early_fusion": MethodSpec(
        branches=("early_fusion",),
        fusion="identity",
        description="One network, X-ray and neutron concatenated as two input channels.",
    ),
    "late_fusion": MethodSpec(
        branches=("xray_plain", "neutron_plain"),
        fusion="mean",
        description="Mean of the X-ray-branch and neutron-branch softmax probabilities.",
    ),
    "late_fusion_ssfb": MethodSpec(
        branches=("xray_ssfb", "neutron_ssfb"),
        fusion="mean",
        description="Late fusion on SSFB-backbone branches -- isolates the backbone under simple averaging.",
    ),
    "meta_learner": MethodSpec(
        branches=("xray_plain", "neutron_plain"),
        fusion="meta",
        description="Learned fusion of both branches' logits and probabilities, plus the raw images.",
    ),
    "meta_learner_ssfb": MethodSpec(
        branches=("xray_ssfb", "neutron_ssfb"),
        fusion="meta",
        description="Learned fusion on SSFB-backbone branches -- backbone on top of the best fusion strategy.",
    ),
}

#: order results are reported in, regardless of the order they were requested
CANONICAL_ORDER: List[str] = list(METHODS)

#: convenience groups accepted wherever a method name is
METHOD_GROUPS: Dict[str, List[str]] = {
    "all": CANONICAL_ORDER,
    "plain": ["single_modality_x", "early_fusion", "late_fusion", "meta_learner"],
    "ssfb": ["late_fusion_ssfb", "meta_learner_ssfb"],
    "fusion": ["early_fusion", "late_fusion", "late_fusion_ssfb", "meta_learner", "meta_learner_ssfb"],
    "baselines": ["single_modality_x", "early_fusion", "late_fusion"],
}


# ---------------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------------
def resolve_methods(requested: Iterable[str] | None) -> List[str]:
    """Expand group names, de-duplicate, validate and sort into canonical order."""
    if not requested:
        return list(CANONICAL_ORDER)

    expanded: List[str] = []
    for name in requested:
        name = name.strip()
        if not name:
            continue
        if name in METHOD_GROUPS:
            expanded.extend(METHOD_GROUPS[name])
        elif name in METHODS:
            expanded.append(name)
        else:
            raise KeyError(
                f"unknown method {name!r}. Available methods: {', '.join(CANONICAL_ORDER)}. "
                f"Available groups: {', '.join(METHOD_GROUPS)}."
            )

    seen = set()
    unique = [m for m in expanded if not (m in seen or seen.add(m))]
    return [m for m in CANONICAL_ORDER if m in unique]


def required_branches(methods: Sequence[str]) -> List[str]:
    """Minimal set of branches needed to evaluate ``methods``, in registry order."""
    needed = set()
    for m in methods:
        needed.update(METHODS[m].branches)
    return [b for b in BRANCHES if b in needed]


def needs_logits(methods: Sequence[str]) -> bool:
    """Whether any selected method consumes raw branch logits (only the meta-learners do)."""
    return any(METHODS[m].needs_logits for m in methods)


def meta_methods(methods: Sequence[str]) -> List[str]:
    return [m for m in methods if METHODS[m].needs_meta]


def format_catalogue() -> str:
    """Human-readable listing for ``--list-methods``."""
    lines = ["Methods:"]
    width = max(len(m) for m in CANONICAL_ORDER)
    for name in CANONICAL_ORDER:
        spec = METHODS[name]
        lines.append(f"  {name:<{width}}  {spec.description}")
        lines.append(f"  {'':<{width}}  branches: {', '.join(spec.branches)}  |  fusion: {spec.fusion}")
    lines.append("")
    lines.append("Groups:")
    for name, members in METHOD_GROUPS.items():
        lines.append(f"  {name:<{width}}  {', '.join(members)}")
    return "\n".join(lines)
