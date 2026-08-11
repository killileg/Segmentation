"""Command-line interface.

GPU selection has to happen before torch is imported, so :func:`main` picks the device and only
then pulls in the heavy modules.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import Config
from .methods import CANONICAL_ORDER, METHOD_GROUPS, format_catalogue, resolve_methods


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="msfusion",
        description="X-ray / neutron multimodal clast segmentation — fusion-method comparison.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  msfusion --dim 2d --methods late_fusion meta_learner --fold 0\n"
            "  msfusion --dim 3d --methods all --k-folds 5 --fold 2\n"
            "  msfusion --list-methods\n"
        ),
    )

    p.add_argument("--dim", choices=["2d", "3d"], default="2d", help="pipeline dimensionality (default: 2d)")
    p.add_argument(
        "-m",
        "--methods",
        nargs="+",
        metavar="NAME",
        help=(
            "methods or groups to run; only the branches these need are trained. "
            f"methods: {', '.join(CANONICAL_ORDER)}. groups: {', '.join(METHOD_GROUPS)}. "
            "(default: all)"
        ),
    )
    p.add_argument("--list-methods", action="store_true", help="print the method catalogue and exit")
    p.add_argument("-c", "--config", type=Path, help="YAML config file; CLI flags override it")

    g = p.add_argument_group("cross-validation")
    g.add_argument("--fold", type=int, help="fold index to run (0-based)")
    g.add_argument("--k-folds", type=int, help="total number of folds")
    g.add_argument("--gap", type=int, help="buffer in slices between blocks of different roles")

    g = p.add_argument_group("data paths")
    g.add_argument("--xray", type=Path, help="X-ray volume (.nii)")
    g.add_argument("--neutron", type=Path, help="neutron volume (.nii)")
    g.add_argument("--seg", type=Path, help="X-ray-derived segmentation labels")
    g.add_argument("--neutron-seg", type=Path, help="independent neutron-drawn segmentation")
    g.add_argument("--final-dir", type=Path, help="root directory for metrics, plots and predictions")
    g.add_argument("--weights-dir", type=Path, help="root directory for checkpoints")

    g = p.add_argument_group("training")
    g.add_argument("--epochs", type=int, help="branch training epochs")
    g.add_argument("--meta-epochs", type=int, help="meta-learner training epochs")
    g.add_argument("--repeat", type=int, help="3D: patches drawn per slab per epoch")
    g.add_argument("--meta-repeat", type=int, help="3D: meta-learner patches per epoch")
    g.add_argument("--batch-size", type=int, help="2D: slices per batch")
    g.add_argument("--num-workers", type=int, help="dataloader worker processes")
    g.add_argument("--dropout", type=float, help="backbone dropout")
    g.add_argument("--seed", type=int, help="random seed (default: unseeded)")
    g.add_argument(
        "--resume",
        action="store_true",
        help="reuse existing branch checkpoints instead of retraining them",
    )
    g.add_argument("--no-gpu-select", action="store_true", help="do not touch CUDA_VISIBLE_DEVICES")

    return p


def make_config(args: argparse.Namespace) -> Config:
    """Layer YAML, environment and CLI flags into a single :class:`Config`."""
    cfg = Config.from_yaml(args.config) if args.config else Config()
    cfg.dim = args.dim
    cfg.apply_env()

    for flag, target in [
        ("fold", ("split", "fold_index")),
        ("k_folds", ("split", "k_folds")),
        ("gap", ("split", "gap")),
        ("xray", ("paths", "xray")),
        ("neutron", ("paths", "neutron")),
        ("seg", ("paths", "seg")),
        ("neutron_seg", ("paths", "neutron_seg")),
        ("final_dir", ("paths", "final_dir")),
        ("weights_dir", ("paths", "weights_dir")),
        ("epochs", ("train", "num_epochs")),
        ("repeat", ("train", "repeat")),
        ("batch_size", ("train", "batch_size")),
        ("num_workers", ("train", "num_workers")),
        ("dropout", ("train", "dropout")),
        ("meta_epochs", ("meta", "epochs")),
        ("meta_repeat", ("meta", "repeat")),
    ]:
        value = getattr(args, flag, None)
        if value is not None:
            setattr(getattr(cfg, target[0]), target[1], value)

    if args.seed is not None:
        cfg.seed = args.seed
    cfg.resume = bool(args.resume)
    cfg.split.__post_init__()
    cfg.methods = resolve_methods(args.methods)
    return cfg


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_methods:
        print(format_catalogue())
        return 0

    try:
        cfg = make_config(args)
    except (KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # must precede the torch import
    from .utils import select_gpu

    if not args.no_gpu_select:
        select_gpu()

    print("=" * 72)
    print(f"msfusion — {cfg.dim.upper()} fusion-method comparison")
    print(cfg.describe())
    print("=" * 72)

    try:
        cfg.validate()
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    cfg.make_dirs()

    from .data import load_volumes
    from .pipelines import get_pipeline
    from .utils import set_seed

    set_seed(cfg.seed)
    volumes = load_volumes(cfg.paths, cfg.data)
    pipeline = get_pipeline(cfg.dim)(cfg, volumes, cfg.methods)
    pipeline.run()

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
