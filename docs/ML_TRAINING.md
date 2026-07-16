# ML Training Guide

Train, evaluate, and validate the production ML detector after collecting
labeled CSI datasets.

Use [ML_DATA_COLLECTION.md](ML_DATA_COLLECTION.md) first to build and validate
the `empty`, `static_presence`, and `motion` datasets. This guide covers the
next step: training `tools/train_ml_model.py`, interpreting its outputs, and
running the key regressions before promoting new artifacts.

## Prerequisites

Install the ML training stack:

```bash
pip install -r requirements-ml.txt
```

The main repository workflow and the ML training stack both target
Python `3.14`.

## Basic Training Workflow

Run the default trainer:

```bash
python tools/train_ml_model.py
```

This evaluates grouped CV, trains the final candidate in memory, and runs the
paired and long-recording gates. It does not replace runtime artifacts. Export
the same configuration only with an explicit promotion:

```bash
python tools/train_ml_model.py --promote
```

Useful variants:

```bash
python tools/train_ml_model.py --info
python tools/train_ml_model.py --scaler clipped_standard
python tools/train_ml_model.py --device mps
python tools/train_ml_model.py --exclude-chip ESP32
python tools/train_ml_model.py --gain-stress-gate
python tools/train_ml_model.py --gain-stress-gate --environment bedroom
python tools/train_ml_model.py --seed-search-until-improvement 20
```

For exploratory architecture campaigns:

```bash
python tools/train_ml_model.py --experiment
python tools/train_ml_model.py --experiment --experiment-promote
python tools/train_ml_model.py --experiment --experiment-architectures "16,8;24,12;32,16;24;24,12,6"
```

For a gated FP-weight campaign:

```bash
python tools/train_ml_model.py --experiment-fp-weights "1,1.5,2,2.5,3"
python tools/train_ml_model.py --experiment-fp-weights "1.5,2,2.5,3" --experiment-promote
```

Both campaigns use the exported seed for single-seed screening, retain the
baseline among the finalists, and then apply three- and five-seed robustness
comparisons. Without `--experiment-promote`, they only write their JSON report.

For feature diagnostics:

```bash
python tools/train_ml_model.py --correlation
python tools/train_ml_model.py --shap 500 --seed 1386543369 --no-export
python tools/train_ml_model.py --ablation-feature turb_skewness --seed 1386543369
```

Correlation is a fast marginal screen over the full training matrix. SHAP runs
inside grouped cross-validation: each fold uses a class-, chip-, and
session-balanced background from its training partition and explains only
balanced, blocked windows from the held-out partition. Supplying `--seed` makes
training, sampling, and permutation SHAP reproducible. Use `--no-export` for
diagnostic runs so the current runtime artifacts remain unchanged.
`--ablation-feature` compares Core-6 against one feature removal using the same
seed, grouped CV, paired validation, and all curated long recordings. It also
leaves the exported runtime artifacts unchanged.
The broader `--ablation` command remains a CV-only screening tool; do not use
its ranking for feature promotion until the finalist passes
`--ablation-feature`.

The latest diagnostic snapshot and interpretation live in
[ALGORITHMS.md](ALGORITHMS.md). Recompute the values after changing the dataset,
feature set, preprocessing, model architecture, or training policy.

## Default Behavior

The binary production trainer loads `empty`, `static_presence`, and `motion`.
`empty` and `static_presence` are both IDLE targets; `motion` is the MOTION
target.

Current default training settings:

- `--fp-weight 2.0`
- `--scaler standard`
- `--batch-size 1024`
- `--device cpu`
- `--sample-weight-mode none`

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
2. Uses the shared CV-normalized turbulence path (`std/mean`) across all files.
3. Applies the selected sample-weight policy. The default production retrain
   uses `none` so the first clean AGC-active baseline does not inherit
   support-detector threshold bias.
4. Extracts the selected ML feature set per sliding window. The production
   default is the Core-6 set.
5. Runs grouped cross-validation by paired capture/session, with blocked
   scoring to reduce overlap optimism.
6. Optionally computes balanced SHAP explanations on the held-out blocked
   windows from each fold.
7. Reports worst-group metrics for session, chip, environment, and source file.
8. Trains the selected MLP architecture with PyTorch, early stopping, and dropout.
9. Evaluates the in-memory candidate on paired captures and curated long recordings.
10. Exports Python and C++ runtime artifacts plus a regression dataset only
    when promotion is explicitly requested.

Support-detector-guided weighting is analysis-only until the clean AGC-active
dataset is recollected and re-evaluated. The guided modes now score windows with
the l1_delta runtime replay, which keeps a near-static quiet floor across
AGC and RF-interference changes.

## Exported Artifacts

Promoted exports:

- `src/python/micro_espectre/ml_weights.py`
- `src/cpp/core/ml_weights.h`
- `data/auto_generated/ml_test_data.npz`

Use `--seed <number>` for reproducible training. The seed is saved in the
generated weight files.

`ml_test_data.npz` is an inference-regression artifact, not the main
model-selection metric. Architecture, weighting, and scaler choices should
treat grouped blocked CV as a diagnostic. Paired validation is a
non-regression constraint, and deploy-like long recordings drive final model
selection.

## Promotion Guidance

For production artifact promotion, prefer one of these gated flows instead of a
plain export:

- `python tools/train_ml_model.py --promote`
- `python tools/train_ml_model.py --seed-search-until-improvement <N>`
- `python tools/train_ml_model.py --experiment --experiment-promote`
- `python tools/train_ml_model.py --experiment-fp-weights "..." --experiment-promote`

A plain training run leaves artifacts unchanged. `--promote` is required for a
normal export. It evaluates both the candidate and current exported arrays,
then writes artifacts only when paired metrics do not regress and at least one
long-recording cost improves without worsening the others.
Experiment campaigns leave artifacts unchanged unless `--experiment-promote`
is supplied. Seed search and experiment promotions confirm the selected
artifacts with the full pytest verification before keeping them.

The long gate also replays the production evaluation cadence and consecutive-hit
policy. Candidate ranking is false-positive-first: effective IDLE-to-MOTION
alarms, time spent in false MOTION measured in policy evaluations,
worst-recording FP rate, and raw false positives precede recall, F1, and CV.
Paired recall and FP are enforced as non-regression constraints. Raw metrics remain
useful diagnostics, while policy metrics represent user-visible false alarms
more directly. Event recall and detection latency require long recordings with
an annotated motion start and are not inferred from quiet-only captures.

For exploratory retrains, `--scaler clipped_standard`, alternate `--device`
choices, `--no-cache`, and smaller `--batch-size` values are available, but
promotable artifacts should still pass the validation checks below.

## Gain-Shift Robustness Check

The production ML path deliberately keeps Python/C++ runtime inference aligned
by deriving all neural-detector inputs from the same raw turbulence signal. The
exported Core-6 feature set uses gain-invariant turbulence and L1-delta
statistics, so the model is structurally less sensitive to absolute amplitude
gain changes.

Use the exported-artifact gain-stress gate to quantify this risk without
retraining or exporting a new model:

```bash
python tools/train_ml_model.py --gain-stress-gate
python tools/train_ml_model.py --gain-stress-gate --environment bedroom
python tools/train_ml_model.py --gain-stress-gate --gain-stress-scales 0.75,1.0,1.25
```

`--gain-stress-gate` does not train or export. It loads the current exported
`src/python/micro_espectre/ml_weights.py`, scales only the amplitude-gain-sensitive input
features, and reports recall/FP degradation overall plus worst chip,
environment, session, and source-file groups.

Current finding for the exported Core-6 model: all-environment gain stress is
flat at `1.00x`, `1.25x`, and `1.50x`. The remaining worst-session weakness is
nominal dataset difficulty, not gain-shift sensitivity. Treat this gate as the
primary diagnostic before promoting future retrains.

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
pytest test/python/test_validation_long_recordings.py -v
python tools/train_ml_model.py --gain-stress-gate
python tools/generate_performance_report.py
python tools/compare_detection_methods.py
```

`tools/generate_performance_report.py` rebuilds `docs/performance/README.md` from the
same shared replay helpers used by the Python real-data validation suites, so
the published performance tables stay aligned with the checked runtime behavior.

Add `--plot` to `compare_detection_methods.py` to visualize the comparison.

## Runtime Notes

The ML pipeline matches the runtime's AGC-active design. Turbulence is always
normalized before the same Core-6 features are extracted for the neural
detector.

To switch the Python runtime to ML detection:

```python
DETECTION_ALGORITHM = "ml"
```

For algorithm details, see [ALGORITHMS.md](ALGORITHMS.md#ml-detector).
