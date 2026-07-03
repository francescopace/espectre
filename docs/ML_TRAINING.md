# ML Training Guide

Train, evaluate, and validate the production ML detector after collecting
labeled CSI datasets.

Use [ML_DATA_COLLECTION.md](ML_DATA_COLLECTION.md) first to build and validate
the `empty`, `static_presence`, and `motion` datasets. This guide covers the
next step: training `tools/10_train_ml_model.py`, interpreting its outputs, and
running the key regressions before promoting new artifacts.

## Prerequisites

Install the ML training stack:

```bash
pip install -r requirements-ml.txt
```

The main repository workflow and the  ML training stack both target
Python `3.14`.

## Basic Training Workflow

Run the default trainer:

```bash
python tools/10_train_ml_model.py
```

Useful variants:

```bash
python tools/10_train_ml_model.py --info
python tools/10_train_ml_model.py --scaler clipped_standard
python tools/10_train_ml_model.py --feature-set robust_relative --no-export
python tools/10_train_ml_model.py --device mps
python tools/10_train_ml_model.py --exclude-chip ESP32
python tools/10_train_ml_model.py --gain-stress-gate
python tools/10_train_ml_model.py --gain-stress-gate --environment bedroom
python tools/10_train_ml_model.py --seed-search-until-improvement 20
```

For exploratory architecture campaigns:

```bash
python tools/10_train_ml_model.py --experiment
python tools/10_train_ml_model.py --experiment --experiment-promote
python tools/10_train_ml_model.py --experiment --experiment-architectures "16,8;24,12;32,16;24;24,12,6"
```

For SHAP diagnostics:

```bash
python tools/10_train_ml_model.py --shap
python tools/10_train_ml_model.py --shap 500
```

## Default Behavior

The binary production trainer loads `empty`, `static_presence`, and `motion`.
`empty` and `static_presence` are both IDLE targets; `motion` is the MOTION
target.

Current default training settings:

- `--fp-weight 2.0`
- `--scaler standard`
- `--batch-size 1024`
- `--device cpu`
- `--feature-set production`

Values above `1.0` for `--fp-weight` reduce false positives at the cost of
slightly lower recall.

CUDA and Apple MPS are available only when requested explicitly through
`--device cuda` or `--device mps`; this small MLP usually runs fastest and most
predictably on CPU. The trainer caches the derived feature matrix and base
sample weights for repeat runs; use `--no-cache` to force a rebuild.

## What The Trainer Does

The training pipeline:

1. Loads all `.npz` files from `data/` for `empty`, `static_presence`, and
   `motion`.
2. Uses gain-mode-aware turbulence: raw std for gain-locked files and
   CV-normalized turbulence for files without gain lock.
3. Applies context-aware MVS-guided sample weighting on the default subcarrier
   set.
4. Extracts 8 relative ML features per sliding window.
5. Runs grouped cross-validation by paired capture/session, with blocked
   scoring to reduce overlap optimism.
6. Reports worst-group metrics for session, chip, environment, and source file.
7. Trains the selected MLP architecture with PyTorch, early stopping, and dropout.
8. Exports artifacts for both Python and C++ runtimes plus a regression dataset.

## Exported Artifacts

Default exports:

- `src/python/micro_espectre/ml_weights.py`
- `src/cpp/core/ml_weights.h`
- `data/auto_generated/ml_test_data.npz`

Use `--seed <number>` for reproducible training. The seed is saved in the
generated weight files.

`ml_test_data.npz` is an inference-regression artifact, not the main
model-selection metric. Architecture and scaler choices should follow the
grouped blocked-CV report emitted by `10_train_ml_model.py`.

## Promotion Guidance

For production artifact promotion, prefer one of these gated flows instead of a
plain export:

- `python tools/10_train_ml_model.py --seed-search-until-improvement <N>`
- `python tools/10_train_ml_model.py --experiment --experiment-promote`

A plain training run always exports the current seed, while the gated flows
replace artifacts only after a stricter grouped-CV improvement.

For exploratory sweeps, `--scaler clipped_standard`, `--feature-set
robust_relative`, alternate `--device` choices, `--no-cache`, and smaller
`--batch-size` values are available, but
non-production feature sets should be run with `--no-export` until they pass
the validation checks below.

## Gain-Shift Robustness Check

The production ML path deliberately keeps Python/C++ runtime inference aligned
by deriving all neural-detector inputs from the same raw turbulence signal. The
exported default feature set is relative to the local turbulence mean, so the
model is structurally less sensitive to absolute amplitude gain changes.

Use the exported-artifact gain-stress gate to quantify this risk without
retraining or exporting a new model:

```bash
python tools/10_train_ml_model.py --gain-stress-gate
python tools/10_train_ml_model.py --gain-stress-gate --environment bedroom
python tools/10_train_ml_model.py --gain-stress-gate --gain-stress-scales 0.75,1.0,1.25
```

`--gain-stress-gate` does not train or export. It loads the current exported
`src/python/micro_espectre/ml_weights.py`, scales only the amplitude-gain-sensitive input
features, and reports recall/FP degradation overall plus worst chip,
environment, session, and source-file groups.

Current finding for the relative `1890407301` export (`8 -> 32 -> 16 -> 1`,
`fp_weight=2.0`): all-environment gain stress is flat at `1.00x`, `1.25x`, and
`1.50x`. The remaining worst-session weakness is nominal dataset difficulty,
not gain-shift sensitivity. Treat this gate as the primary diagnostic for
comparing future raw, relative, or hybrid feature sets.

## Empty-Room Regression Check

The 2026-06-30 production retrain was motivated by a C3 ESPHome runtime log
that produced noisy ML scores in a static room. Offline analysis showed that
the new C3 `static_presence` capture was not the failing case; the new C3
`empty` capture reproduced the problem. The fix was to include `empty` in the
binary ML training labels instead of training only on `static_presence` versus
`motion`. A later C6 bedroom `empty` capture exposed the same class of domain
coverage issue, so the regression now covers all available empty-room files,
not only C3.

Run the dedicated regression for newly collected empty-room data:

```bash
pytest test/python/test_validation_real_data.py::TestPerformanceMetrics::test_ml_empty_false_positive_rate -v
```

The current target is below `5%` false positives for every
`data/empty/empty_*_64sc_*.npz` file.

## Post-Training Checks

Recommended validations before promoting new artifacts:

```bash
pytest test/python/test_validation_real_data.py::TestPerformanceMetrics::test_ml_detection_accuracy -v
pytest test/python/test_validation_real_data.py::TestPerformanceMetrics::test_ml_empty_false_positive_rate -v
python tools/10_train_ml_model.py --gain-stress-gate
python tools/7_compare_detection_methods.py
```

Add `--plot` to `7_compare_detection_methods.py` to visualize the comparison.

## Runtime Notes

The ML pipeline matches runtime gain handling.
`MLDetector::set_cv_normalization(true)` enables CV-normalized turbulence for
no-gain-lock streams; gain-locked streams keep raw turbulence. The exported
feature set remains the 8 relative features used by the neural detector.

To switch the Python runtime to ML detection:

```python
DETECTION_ALGORITHM = "ml"
```

For algorithm details, see [ALGORITHMS.md](ALGORITHMS.md#ml-neural-network-detector).
