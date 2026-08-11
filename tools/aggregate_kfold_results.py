#!/usr/bin/env python
"""Combine per-fold metrics into a mean +/- std table.

Each fold writes ``fold<N>/metrics_<dim>.csv``; this script reads them all and reports, per
method, the mean and sample standard deviation of every metric across folds.

    python tools/aggregate_kfold_results.py --dim 2d --final-dir ./outputs
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

METRIC_COLUMNS = ["dice", "f1", "precision", "recall", "iou", "fragmentation_pct"]


def read_folds(final_dir: Path, dim: str):
    rows = []
    for path in sorted(final_dir.glob(f"fold*/metrics_{dim}.csv")):
        with open(path) as fh:
            for row in csv.DictReader(fh):
                row["_source"] = str(path)
                rows.append(row)
    return rows


def aggregate(rows, class_filter=None):
    """Group by (method, class) and collect each metric's values across folds."""
    grouped = defaultdict(lambda: defaultdict(list))
    folds = defaultdict(set)
    for row in rows:
        cls = row.get("class", "any")
        if class_filter and cls != class_filter:
            continue
        key = (row["method"], cls)
        folds[key].add(row.get("fold"))
        for col in METRIC_COLUMNS:
            if row.get(col) not in (None, ""):
                grouped[key][col].append(float(row[col]))
    return grouped, folds


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--final-dir", type=Path, required=True, help="directory containing fold*/ subdirectories")
    p.add_argument("--dim", choices=["2d", "3d"], default="2d")
    p.add_argument(
        "--class",
        dest="class_filter",
        help="show only one class row: 'any' (binary detection), a clast type name, or 'macro_avg'",
    )
    p.add_argument("--out", type=Path, help="optional CSV to write the aggregate table to")
    args = p.parse_args()

    rows = read_folds(args.final_dir, args.dim)
    if not rows:
        print(f"no metrics_{args.dim}.csv found under {args.final_dir}/fold*/")
        return 1

    grouped, folds = aggregate(rows, args.class_filter)
    if not grouped:
        print(f"no rows matched class={args.class_filter!r}")
        return 1

    label_width = max(len(f"{m} [{c}]") for m, c in grouped)
    header = f"{'method [class]':<{label_width}}  " + "  ".join(f"{c:>18}" for c in METRIC_COLUMNS) + "   folds"
    print(header)
    print("-" * len(header))

    out_rows = []
    last_method = None
    for (method, cls), metrics in grouped.items():
        # blank line between methods keeps the per-class blocks readable
        if last_method is not None and method != last_method:
            print()
        last_method = method

        cells = []
        record = {"method": method, "class": cls, "n_folds": len(folds[(method, cls)])}
        for col in METRIC_COLUMNS:
            values = metrics.get(col, [])
            if not values:
                cells.append(f"{'-':>18}")
                continue
            mean = statistics.fmean(values)
            std = statistics.stdev(values) if len(values) > 1 else 0.0
            cells.append(f"{mean:>10.4f}\u00b1{std:<7.4f}")
            record[f"{col}_mean"] = mean
            record[f"{col}_std"] = std
        label = f"{method} [{cls}]"
        print(f"{label:<{label_width}}  " + "  ".join(cells) + f"   {len(folds[(method, cls)])}")
        out_rows.append(record)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fixed = ["method", "class", "n_folds"]
        fieldnames = fixed + sorted({k for r in out_rows for k in r} - set(fixed))
        with open(args.out, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(out_rows)
        print(f"\n[saved] {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
