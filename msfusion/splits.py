"""Train / validation / test splits along the H axis.

Slices adjacent in H are highly correlated, so the folds are contiguous *blocks*
rather than shuffled slices, and neighbouring blocks of different roles are
separated by a ``gap`` buffer that belongs to neither.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

Range = Tuple[int, int]


@dataclass
class Splits:
    """One fold's slice ranges. ``train`` is a list of ranges; ``val``/``test`` are single ranges."""

    train: List[Range]
    val: Range
    test: Range

    def __str__(self) -> str:
        return f"Train {self.train}  Val {self.val}  Test {self.test}"


def make_kfold_splits(good_start: int, good_end: int, k: int, fold_index: int, gap: int = 64) -> Splits:
    """Split ``[good_start, good_end)`` into ``k`` equal blocks and assign roles for one fold.

    Block ``fold_index`` becomes test, block ``fold_index + 1`` (wrapping) becomes val, the rest
    are train. Wherever two adjacent blocks have different roles, ``gap`` slices straddling the
    boundary are removed from both sides.
    """
    n = good_end - good_start
    block_size = n // k
    bounds = [good_start + i * block_size for i in range(k)] + [good_end]
    starts, ends = bounds[:-1], bounds[1:]

    roles = ["train"] * k
    test_block, val_block = fold_index, (fold_index + 1) % k
    roles[test_block] = "test"
    roles[val_block] = "val"

    for i in range(k - 1):
        if roles[i] != roles[i + 1]:
            mid = ends[i]
            ends[i] = mid - gap // 2
            starts[i + 1] = mid + (gap - gap // 2)

    train: List[Range] = []
    val = test = None
    for i in range(k):
        rng = (starts[i], ends[i])
        if roles[i] == "train":
            train.append(rng)
        elif roles[i] == "val":
            val = rng
        else:
            test = rng

    return Splits(train=_merge_adjacent(train), val=val, test=test)


def make_anchored_splits(
    full_start: int,
    full_end: int,
    val_block: Range,
    test_block: Range,
    gap: int = 64,
) -> Splits:
    """Build splits over ``[full_start, full_end)`` that *reuse* another modality's val/test blocks.

    The X-ray branch spans the full native volume, far wider than the neutron field of view. If
    its blocks were computed independently they would land at different boundaries, and part of
    X-ray's train would fall inside the neutron val/test window -- meaning every fusion method
    built on the X-ray branch would be evaluated on slices that branch had already trained on.

    Anchoring solves this: X-ray takes the *same absolute* val/test ranges as neutron, and trains
    on its full native range minus those two gap-buffered holes. It still keeps every X-ray-only
    slice outside the neutron window.
    """
    holes = []
    for h0, h1 in (val_block, test_block):
        b0 = max(full_start, h0 - gap // 2)
        b1 = min(full_end, h1 + (gap - gap // 2))
        holes.append((b0, b1))

    merged: List[Range] = []
    for b0, b1 in sorted(holes):
        if merged and b0 <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b1))
        else:
            merged.append((b0, b1))

    train: List[Range] = []
    cursor = full_start
    for h0, h1 in merged:
        if h0 > cursor:
            train.append((cursor, h0))
        cursor = max(cursor, h1)
    if cursor < full_end:
        train.append((cursor, full_end))

    return Splits(train=train, val=val_block, test=test_block)


def drop_short_ranges(splits: Splits, min_length: int, label: str = "") -> Splits:
    """Remove train ranges shorter than ``min_length`` (3D only).

    Carving the val/test holes out of the X-ray range can leave a thin edge sliver -- e.g. 56
    slices -- which is too short in H for ``RandCropByPosNegLabeld`` to fit a 128-deep patch and
    crashes with "proposed random crop ROI is larger than the image size". 2D trains per slice
    and has no such constraint. Dropping such a sliver loses well under 3% of X-ray training data.
    """
    dropped = [r for r in splits.train if (r[1] - r[0]) < min_length]
    if dropped:
        print(f"Dropping {label} train range(s) shorter than {min_length}: {dropped}")
    kept = [r for r in splits.train if (r[1] - r[0]) >= min_length]
    return Splits(train=kept, val=splits.val, test=splits.test)


def build_splits(
    volume_h: int,
    neutron_h_start: int,
    neutron_h_end: int,
    k_folds: int,
    fold_index: int,
    gap: int = 64,
) -> Dict[str, Splits]:
    """Produce the ``{'xray': ..., 'neutron': ...}`` split pair used by every branch."""
    neutron = make_kfold_splits(neutron_h_start, neutron_h_end, k_folds, fold_index, gap)
    xray = make_anchored_splits(0, volume_h, neutron.val, neutron.test, gap)
    return {"xray": xray, "neutron": neutron}


def _merge_adjacent(ranges: Sequence[Range]) -> List[Range]:
    merged: List[Range] = []
    for h0, h1 in ranges:
        if merged and merged[-1][1] == h0:
            merged[-1] = (merged[-1][0], h1)
        else:
            merged.append((h0, h1))
    return merged
