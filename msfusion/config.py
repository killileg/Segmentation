"""Configuration objects for the X-ray / neutron fusion pipelines.

Precedence (lowest to highest):  dataclass defaults  <  YAML file  <  environment
variables  <  command-line flags.

Environment-variable names are kept identical to the original standalone scripts
(``K_FOLDS``, ``FOLD_INDEX``, ``NUM_EPOCHS_B``, ``REPEAT_B``, ``META_EPOCHS``,
``META_REPEAT``, ``FINAL_DIR``, ``WEIGHTS_DIR``) so existing cluster submit
scripts keep working unchanged.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------
#: input files that must be supplied by the user -- there are no defaults, since the dataset
#: is not distributed with this repository. See the README for the expected format.
REQUIRED_INPUTS = ("xray", "neutron", "seg")


@dataclass
class PathConfig:
    """Where the input volumes live and where results are written.

    The three required inputs have no default value on purpose: this repository ships code, not
    data, so a run must always be pointed at a dataset explicitly (via ``--xray`` etc. or a YAML
    config). ``neutron_seg`` is optional -- supply it only if you have a second, independently
    drawn segmentation you want exported alongside the predictions for comparison.
    """

    xray: Optional[Path] = None
    neutron: Optional[Path] = None
    seg: Optional[Path] = None
    neutron_seg: Optional[Path] = None

    final_dir: Path = Path("./outputs")
    weights_dir: Path = Path("./weights")

    def __post_init__(self) -> None:
        for f in ("xray", "neutron", "seg", "neutron_seg"):
            value = getattr(self, f)
            if value is not None:
                setattr(self, f, Path(value).expanduser())
        self.final_dir = Path(self.final_dir).expanduser()
        self.weights_dir = Path(self.weights_dir).expanduser()

    def validate(self) -> None:
        """Fail early, with an actionable message, rather than deep inside data loading."""
        missing = [f for f in REQUIRED_INPUTS if getattr(self, f) is None]
        if missing:
            flags = ", ".join(f"--{f.replace('_', '-')}" for f in missing)
            raise ValueError(
                f"no input path given for: {', '.join(missing)}. "
                f"Pass {flags} on the command line, or set them in a YAML config "
                f"(see configs/example.yaml). This repository does not ship a dataset."
            )
        not_found = [
            (f, getattr(self, f))
            for f in REQUIRED_INPUTS + ("neutron_seg",)
            if getattr(self, f) is not None and not getattr(self, f).exists()
        ]
        if not_found:
            details = "\n".join(f"  {f}: {path}" for f, path in not_found)
            raise FileNotFoundError(f"input file(s) not found:\n{details}")


@dataclass
class DataConfig:
    """Volume interpretation: labels, ROI derivation and the usable neutron window."""

    #: integer value -> human-readable clast type in the segmentation volume
    label_names: Dict[int, str] = field(default_factory=lambda: {1: "dark", 2: "medium", 3: "bright"})
    #: which label ids are trained/evaluated on (background is added automatically)
    class_ids: List[int] = field(default_factory=lambda: [1, 2, 3])
    #: the ROI segmentation is stored at half resolution -- upsample by this factor
    seg_zoom: int = 2
    #: dilation applied to the non-zero segmentation to build the X-ray ROI mask
    roi_dilation_iters: int = 1
    #: neutron field-of-view actually used, in slice indices along H
    neutron_h_start: int = 750
    neutron_h_end: int = 1800

    @property
    def class_names(self) -> List[str]:
        return [self.label_names[c] for c in self.class_ids]

    @property
    def out_channels(self) -> int:
        """Number of clast channels reported (background excluded)."""
        return len(self.class_ids)

    @property
    def model_out_channels(self) -> int:
        """Network output channels: clast channels + one explicit background channel."""
        return self.out_channels + 1


@dataclass
class SplitConfig:
    """K-fold cross-validation over contiguous blocks of slices."""

    k_folds: int = 3
    fold_index: int = 0
    #: buffer (in slices) carved out between neighbouring blocks of different roles,
    #: so train and val/test never touch across a boundary
    gap: int = 64

    def __post_init__(self) -> None:
        if not 0 <= self.fold_index < self.k_folds:
            raise ValueError(f"fold_index={self.fold_index} out of range for k_folds={self.k_folds}")


@dataclass
class TrainConfig:
    """Branch (base network) training hyper-parameters."""

    num_epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-5
    eta_min: float = 1e-6
    dropout: float = 0.2
    batch_size: int = 4
    num_workers: int = 2
    #: 2D only -- every slice is zero-padded to this (D, W) so shapes are uniform
    pad_size: Tuple[int, int] = (360, 1456)
    #: 3D only -- random crop / sliding-window patch size
    patch_size: Tuple[int, int, int] = (128, 128, 128)
    #: 3D only -- how many random patches are drawn per training slab per epoch
    repeat: int = 50
    #: sliding-window inference overlap
    sw_overlap: float = 0.25


@dataclass
class MetaConfig:
    """Meta-learner (learned stacking) hyper-parameters."""

    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-5
    eta_min: float = 1e-6
    hidden: int = 16
    batch_size: int = 4
    #: 3D only -- number of random patches drawn from the single val slab per epoch
    repeat: int = 200
    #: branch logits handed to the meta-learner are clamped to +/- this
    logit_clip: float = 10.0


@dataclass
class Config:
    """Top-level configuration for one pipeline run (one dimensionality, one fold)."""

    dim: str = "2d"                       # "2d" or "3d"
    methods: List[str] = field(default_factory=list)   # empty == all registered methods
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    meta: MetaConfig = field(default_factory=MetaConfig)
    seed: Optional[int] = None
    #: reuse an existing branch checkpoint instead of retraining it, when present
    resume: bool = False

    # -- derived, fold-scoped output directories ---------------------------
    @property
    def fold_final_dir(self) -> Path:
        return self.paths.final_dir / f"fold{self.split.fold_index}"

    @property
    def fold_weights_dir(self) -> Path:
        return self.paths.weights_dir / f"fold{self.split.fold_index}"

    @property
    def suffix(self) -> str:
        """Filename suffix distinguishing 2D from 3D artefacts."""
        return self.dim

    def validate(self) -> None:
        self.paths.validate()

    def make_dirs(self) -> None:
        self.fold_final_dir.mkdir(parents=True, exist_ok=True)
        self.fold_weights_dir.mkdir(parents=True, exist_ok=True)

    # -- construction ------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: Path | str) -> "Config":
        import yaml  # imported lazily: the package works without pyyaml if no config file is used

        with open(path) as fh:
            raw = yaml.safe_load(fh) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict) -> "Config":
        sub = {
            "paths": PathConfig,
            "data": DataConfig,
            "split": SplitConfig,
            "train": TrainConfig,
            "meta": MetaConfig,
        }
        kwargs = {}
        for key, value in raw.items():
            if key in sub:
                kwargs[key] = sub[key](**value)
            else:
                kwargs[key] = value
        cfg = cls(**kwargs)
        # tuples survive a YAML round-trip as lists; normalise so shape maths works
        cfg.train.pad_size = tuple(cfg.train.pad_size)
        cfg.train.patch_size = tuple(cfg.train.patch_size)
        return cfg

    def apply_env(self) -> "Config":
        """Overlay the legacy environment variables used by the cluster submit scripts."""

        def _int(name, current):
            return int(os.environ[name]) if name in os.environ else current

        self.split.k_folds = _int("K_FOLDS", self.split.k_folds)
        self.split.fold_index = _int("FOLD_INDEX", self.split.fold_index)
        self.split.__post_init__()
        self.train.num_epochs = _int("NUM_EPOCHS_B", self.train.num_epochs)
        self.train.repeat = _int("REPEAT_B", self.train.repeat)
        self.meta.epochs = _int("META_EPOCHS", self.meta.epochs)
        self.meta.repeat = _int("META_REPEAT", self.meta.repeat)
        if "FINAL_DIR" in os.environ:
            self.paths.final_dir = Path(os.environ["FINAL_DIR"])
        if "WEIGHTS_DIR" in os.environ:
            self.paths.weights_dir = Path(os.environ["WEIGHTS_DIR"])
        return self

    def to_dict(self) -> dict:
        out = asdict(self)

        def _clean(obj):
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, dict):
                return {str(k): _clean(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_clean(v) for v in obj]
            return obj

        return _clean(out)

    def describe(self) -> str:
        lines = [
            f"dimensionality      : {self.dim}",
            f"methods             : {', '.join(self.methods) if self.methods else '<all>'}",
            f"fold                : {self.split.fold_index} / {self.split.k_folds}  (gap={self.split.gap})",
            f"classes             : {self.data.class_names}",
            f"branch epochs       : {self.train.num_epochs}",
            f"meta epochs         : {self.meta.epochs}",
            f"results  -> {self.fold_final_dir}",
            f"weights  -> {self.fold_weights_dir}",
        ]
        return "\n".join("  " + line for line in lines)


