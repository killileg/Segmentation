# msfusion — multimodal X-ray / neutron clast segmentation

Segmenting clasts in a rock core imaged twice, by X-ray CT and by neutron tomography. The two
modalities see different things: X-ray responds to density, neutron to hydrogen content, and a
clast invisible in one is often obvious in the other. The question this repository answers is
**how best to combine them** — and whether combining them helps at all.

The comparison holds the backbone architecture fixed and varies only the *fusion strategy*, so
differences in the results table are attributable to fusion rather than to network capacity. A
separate arm swaps in a custom SSFB backbone to measure the architecture's own contribution on
top of the best fusion strategy.

---

## Description

Six methods are trained and scored under identical K-fold cross-validation:

| Method | What it does | Branches trained |
|---|---|---|
| `single_modality_x` | X-ray only, no fusion. The baseline everything else must beat. | `xray_plain` |
| `early_fusion` | One network, X-ray and neutron stacked as two input channels. | `early_fusion` |
| `late_fusion` | Mean of the two branches' softmax probabilities. | `xray_plain`, `neutron_plain` |
| `late_fusion_ssfb` | Same averaging, on SSFB-backbone branches. | `xray_ssfb`, `neutron_ssfb` |
| `meta_learner` | A small network learns to combine both branches' logits and probabilities. | `xray_plain`, `neutron_plain` |
| `meta_learner_ssfb` | Same learned fusion, on SSFB-backbone branches. | `xray_ssfb`, `neutron_ssfb` |

The four rows without `ssfb` in the name all use a plain `DynUNet`, so they isolate fusion
strategy. The two `ssfb` rows reuse the best fusion strategies on a different backbone, so the
pairwise differences (`late_fusion` vs `late_fusion_ssfb`, `meta_learner` vs `meta_learner_ssfb`)
isolate the architecture.

The neutron-only branch is trained internally — late fusion and the meta-learners need something
to fuse with — but its standalone score is deliberately **not** reported. Every reported method
is either X-ray-anchored or a joint prediction, so all six are scored against the same
X-ray-derived ground truth.

Both a 2D pipeline (each slice an independent sample) and a 3D pipeline (volumetric 128³ patches)
are implemented. They share configuration, splits, metrics, models and reporting; only batching
and inference tiling differ.

---

## Requirements

- Python ≥ 3.9, CUDA-capable GPU
- PyTorch ≥ 2.0, MONAI ≥ 1.3, nibabel, scipy, numpy, matplotlib, PyYAML

```bash
pip install -r requirements.txt
# or, to get the `msfusion` command on your PATH:
pip install -e .
```

### Input data

**This repository ships code, not data.** No dataset is included, and the input paths have no
defaults — a run must always be pointed at your own files. To use it on your own material you
need to supply the following.

| Argument | Required | What it is |
|---|---|---|
| `--xray` | yes | First imaging modality, a 3D volume (`.nii` / `.nii.gz`) |
| `--neutron` | yes | Second imaging modality, same shape and grid as the first |
| `--seg` | yes | Integer label volume: `0` = background/matrix, `1..N` = your classes |
| `--neutron-seg` | no | A second, independently drawn label volume, exported for comparison but never trained on |

Requirements on those files:

- **Same grid.** All volumes must share one voxel grid and be co-registered. Axis order is
  `(H, D, W)`, where `H` is the axis the cross-validation folds are cut along — so `H` should be
  the long axis of your sample, with enough slices to split into blocks.
- **Labels are integers.** Every non-zero value in `--seg` must appear in `data.label_names` in
  your config. Values not listed in `class_ids` are treated as background.
- **Half-resolution labels are fine.** If `--seg` is stored at half resolution, leave
  `data.seg_zoom: 2`; it is upsampled with nearest-neighbour on load. Set it to `1` if your labels
  are already full resolution.
- **No intensity rescaling is applied.** Both modalities are read as raw float32. Normalise
  beforehand if your data needs it.
- **The second modality may cover only part of the volume.** Voxels where it reads exactly zero
  are treated as "no signal". Set `data.neutron_h_start` / `neutron_h_end` to the slice range where
  it is actually usable — the auto-detected extent is printed at start-up, and you generally want
  to narrow inwards from it to drop edge artefacts.

Two derived masks fall out of this and drive everything downstream: the **first-modality ROI**
(the label volume, dilated by one voxel) and the **second-modality ROI** (that ROI intersected
with where the second modality has signal). Outside these, predictions are forced to background.

Adapting to a different pair of modalities is a config change, not a code change — the names
"xray" and "neutron" are just labels for "modality A" and "modality B".

### Configuration

Copy `configs/example.yaml`, fill in your paths, and pass it with `--config`:

```bash
cp configs/example.yaml configs/my_config.yaml
$EDITOR configs/my_config.yaml
python run.py --config configs/my_config.yaml --dim 2d --fold 0
```

Or pass everything on the command line:

```bash
python run.py --dim 2d --xray data/a.nii --neutron data/b.nii --seg data/labels.nii.gz
```

Precedence, lowest first: dataclass defaults → YAML → environment variables → CLI flags.

---

## Usage

Run everything, fold 0:

```bash
python run.py --dim 2d --fold 0 --k-folds 3
```

Run only the methods you care about — this is the point of the refactor. Only the branches those
methods actually need get trained, so asking for one late-fusion result trains two networks
instead of five:

```bash
python run.py --dim 2d --methods late_fusion
python run.py --dim 3d --methods meta_learner meta_learner_ssfb --fold 2 --k-folds 5
python run.py --dim 2d --methods ssfb          # a named group
```

Groups: `all`, `plain`, `ssfb`, `fusion`, `baselines`. See everything available with:

```bash
python run.py --list-methods
```

Full K-fold sweep on a cluster, then aggregate:

```bash
./scripts/submit_kfold.sh 2d 5 all
./scripts/aggregate.sh 2d ./outputs
```

`--resume` reuses any branch checkpoint already on disk, which is what you want when adding a
method that shares branches with a run you have already done.

### Reproducibility note

No fixed random seed was used for the experiments reported with this repository. Training is
therefore not bit-for-bit reproducible and results may vary between runs. For future runs, pass
`--seed <integer>` to seed Python, NumPy and PyTorch; exact reproducibility can still depend on
the CUDA, cuDNN, PyTorch and MONAI versions and on nondeterministic GPU operations.

For cluster use, the environment variables `K_FOLDS`, `FOLD_INDEX`, `NUM_EPOCHS_B`, `REPEAT_B`,
`META_EPOCHS`, `META_REPEAT`, `FINAL_DIR` and `WEIGHTS_DIR` are also honoured, so a scheduler can
set the fold without rewriting the command.

### Outputs

Written to `<final_dir>/fold<N>/`:

| File | Contents |
|---|---|
| `metrics_<dim>.csv` | One row per (method, class). See below. |
| `confusion_<dim>.csv` | Class-confusion counts per method, long format |
| `convergence_<dim>_<name>.png` | Training curves, one per branch and meta-learner |
| `per_slice_dice_methods_<dim>.png` | Binary Dice per slice across the test range |
| `per_class_dice_methods_<dim>.png` | Grouped bar chart of Dice per class |
| `test_preds_<dim>.npz` | float16 predictions, ground truth, class names |
| `run_config_<dim>.json` | Fully resolved configuration and the actual splits used |

Checkpoints go to `<weights_dir>/fold<N>/`.

### Metrics

`metrics_<dim>.csv` has a `class` column, giving three levels of detail per method:

- **`any`** — binary clast detection: any class vs. background. The headline number, answering
  *did the model find the object at all*. This is the only row carrying `fragmentation_pct`.
- **one row per class** (`dark`, `medium`, …) — *and did it assign the right type*. Scored by
  argmax over the class channels, so classes are mutually exclusive, matching how the softmax
  output is meant to be read.
- **`macro_avg`** — unweighted mean across classes, which treats a rare class as being as
  important as a common one. Compare it against `any` to see how much of the error is
  classification rather than detection.

Per-class Dice alone doesn't tell you *which* class a mistake went to, so `confusion_<dim>.csv`
records the full ground-truth × prediction voxel counts. Confusion concentrated between two
adjacent classes means something quite different from confusion spread evenly.

The **fragmentation rate** is the fraction of predicted voxels landing where the second modality
recorded no signal at all — a direct measure of hallucination into dead regions.

The aggregation tool understands the class column:

```bash
python tools/aggregate_kfold_results.py --final-dir ./outputs --dim 2d
python tools/aggregate_kfold_results.py --final-dir ./outputs --dim 2d --class any
python tools/aggregate_kfold_results.py --final-dir ./outputs --dim 2d --class macro_avg
```

---

## Repository layout

```
msfusion/
  config.py         Dataclass configuration; YAML / env / CLI layering
  data.py           NIfTI loading, ROI derivation, label stacks
  splits.py         K-fold blocks, X-ray anchoring, gap buffering
  metrics.py        Dice / F1 / precision / recall / IoU / fragmentation
  methods.py        Branch and method registry  <- add new methods here
  utils.py          GPU selection, padding, ROI masking, plots
  models/
    blocks2d.py     SSFB2D, SSFBUpBlock2D, DynUNetSSFB2D
    blocks3d.py     SSFB3D, SSFBUpBlock3D, DynUNetSSFB3D
    networks2d.py   PlainUNet2D, SSFBUNet2D, MetaLearner2D
    networks3d.py   PlainUNet3D, SSFBUNet3D, MetaLearner3D
  pipelines/
    base.py         Shared four-stage orchestration
    pipeline2d.py   Per-slice batching, padded inference
    pipeline3d.py   Patch cropping, sliding-window inference
  cli.py            Argument parsing and entry point
configs/            YAML presets
scripts/            Local run, cluster submit, aggregate
tools/              K-fold result aggregation
run.py              Entry point
```

---

## How does it work?

### Stage 1 — train the base branches

Only the branches required by the selected methods are trained. Each is a `DynUNet` with filters
`[32, 64, 128, 256]`, trained with Dice loss on a softmax output.

The output has an explicit **background channel at index 0** in addition to the three clast
classes. Without it a softmax would force every voxel — including pure matrix — to commit to one
of the three clast types, since softmax normalises to sum 1 across channels with no "none of the
above" option.

Outside the ROI mask, logits are overwritten with a fixed tensor that strongly favours background
(`+20` on channel 0, `-20` elsewhere). This makes the network's output there match the guaranteed
target exactly, so those voxels contribute nothing to the gradient rather than adding noise.

### Stage 2 — freeze and cache

Branches are frozen and run over the validation and test ranges. Two things are kept: the softmax
probabilities, and the raw pre-softmax logits clamped to ±10. Logits carry confidence information
that probabilities have already squashed away, which is exactly what a learned fusion needs;
clamping stops a single saturated branch from dominating the meta-learner's input scale. Logits
are only computed when a meta-learner is actually selected.

### Stage 3 — fuse

- **Late fusion** averages the two branches' probability maps. No parameters, no training.
- **The meta-learner** is a three-layer convolutional network (~a few thousand parameters) taking
  both branches' probabilities, both branches' clamped logits, and the two raw modality images. It
  is trained on the **validation** range, which the frozen branches never saw, so it learns to
  combine them rather than to memorise their training-set behaviour. It is kept deliberately tiny:
  its job is to learn *how to weigh* two frozen branches, not to re-learn segmentation from a
  small validation range.

### Stage 4 — evaluate

Two levels, both against the X-ray-derived ground truth: binary clast detection at threshold 0.5,
and per-clast-type classification by argmax. Splitting them matters, because a method can detect
clasts well while confusing dark with medium, and a single averaged number would hide that. See
[Metrics](#metrics) above.

### Cross-validation and the anchoring problem

Slices adjacent in H are highly correlated, so folds are contiguous blocks, not shuffled slices,
and a 64-slice `gap` straddling each role boundary belongs to neither side.

The subtle part: the neutron field of view is a narrow window (slices 750–1800, trimmed inward
from the auto-detected extent to drop edge artefacts), while X-ray spans the full volume. If each
modality computed its own blocks independently, the boundaries would not line up, and part of
X-ray's *train* would fall inside neutron's *val/test* window. Every fusion method built on the
X-ray branch would then be evaluated on slices that branch had already trained on.

So X-ray's blocks are **anchored** to neutron's: X-ray takes the same absolute val/test ranges and
trains on its full native range minus those two gap-buffered holes. It still keeps every
X-ray-only slice outside the neutron window, which is most of its training data.

In 3D one extra step is needed: carving those holes out can leave a sliver too short in H to fit a
128-deep patch, which crashes the random crop. Ranges shorter than the patch depth are dropped —
under 3% of X-ray training data. 2D trains per slice and has no such constraint.

---

## SSFB — Skip and Spatial Feature Blend

A drop-in replacement for the concat-based skip connection in `DynUNet`. The encoder feature map
arriving on a skip is first re-weighted channel-wise by a gate conditioned on the global context
of *both* the encoder and decoder maps. It is then optionally refined by low-rank cross-attention
against a spatially pooled version of itself, with the decoder map supplying the queries. A
learned scalar `alpha`, initialised so `sigmoid(alpha) = 0.5`, blends the two paths — so the block
can fall back to pure gating if attention isn't earning its keep.

Attention is pooled rather than full because a full attention map over the finest feature grid is
prohibitive in 3D. Pool sizes run coarse-to-fine (`[8, 4, None]` in 2D, `[4, 2, None]` in 3D);
`None` at the finest stage disables the attention path there entirely.

---

## Extending

**A new fusion method** usually means one entry in `msfusion/methods.py`:

```python
METHODS["late_fusion_max"] = MethodSpec(
    branches=("xray_plain", "neutron_plain"),
    fusion="max",
    description="Element-wise max of the two branches' probabilities.",
)
```

then handle `"max"` in `BasePipeline.build_method_predictions`. The CLI, branch scheduling,
metrics, plotting and export all pick it up automatically.

**A new backbone** means one entry in `BACKBONES_2D` / `BACKBONES_3D` (in
`msfusion/models/networks2d.py` and `networks3d.py`) and one `BranchSpec` per branch using it.

Note that `PlainUNet2D` and `PlainUNet3D` are stock MONAI `DynUNet`s with the hyperparameters
pinned — they add no behaviour of their own. They exist so that every "plain" branch is
guaranteed architecturally identical, which is what keeps the fusion comparison honest.

---

## Parameters

**Backbone** — filters `[32, 64, 128, 256]`, residual blocks, dropout 0.2, Dice loss with softmax,
Adam at 1e-3 with cosine annealing and weight decay 1e-5.

**2D** — slices padded to 360 × 1456, batch size 4, 50 epochs. Augmentation: flips on both axes
(p=0.5), rotation (p=0.2, ±0.3 rad).

**3D** — 128³ patches, `RandCropByPosNegLabeld` with `pos=1, neg=0`, so every crop center is
sampled from inside the ROI. There are 50 patches per slab per epoch and 20 epochs. Augmentation:
flips on all three axes (p=0.5 per axis). The same ROI-only sampling is used for meta-learner crops.

**Meta-learner** — 3 conv layers, 16 hidden channels, 20 epochs, logits clamped to ±10.

**Cross-validation** — 3 folds by default, 64-slice gap, neutron window 750–1800.

---

## Future work

- Evaluate against the independent neutron-drawn segmentation, not only the X-ray-derived GT
- Boundary-aware metrics (Hausdorff, surface Dice) alongside the overlap ones
- Uncertainty-weighted late fusion instead of a plain mean
- Test-time augmentation for the frozen branches

---

## License

MIT
