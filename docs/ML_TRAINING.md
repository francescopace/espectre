# ML Training Guide

Train, evaluate, and validate the production ML detector after collecting
labeled CSI datasets.

Use [ML_DATA_COLLECTION.md](ML_DATA_COLLECTION.md) first to build and validate
the `empty`, `static_presence`, and `motion` datasets. This guide covers the
next step: training `tools/train_ml_model.py`, interpreting its outputs, and
running the key regressions before promoting new artifacts. For the host CLI
entry points that drive collection and related workflows, see
[CLI.md](CLI.md).

## Prerequisites

Install the ML training stack:

```bash
pip install -r requirements-ml.txt
```

The main repository workflow and the ML training stack both target
Python `3.14`.

Before training, refresh and admit the corpus:

```bash
python tools/validate_dataset_quality.py
```

That command updates pair metadata when needed, writes
`data/auto_generated/DATASET_QUALITY_CHECK.md`, and fails only on admission
checks. Classic indicative scores are review-only.

## Basic Training Workflow

Run the default trainer:

```bash
python tools/train_ml_model.py
```

This evaluates grouped CV, trains the final candidate, runs the paired gate,
and exports runtime artifacts when that gate passes and does not regress
against the current baseline. Use `--no-export` to evaluate without replacing
artifacts.

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
seed, grouped CV, and paired validation. It also leaves the exported runtime
artifacts unchanged.
The broader `--ablation` command remains a CV-only screening tool; do not use
its ranking for feature promotion until the finalist passes
`--ablation-feature`.

The latest diagnostic snapshot and interpretation live in
[ALGORITHMS.md](ALGORITHMS.md). Recompute the values after changing the dataset,
feature set, preprocessing, model architecture, or training policy.
The production ML path also stays aligned with the same fixed 12-tone HT20 band
used by Classic. For why the project keeps exactly that band, instead of
switching count or adopting adjacent-tone averaging, see
[`2026-07-20-keep-the-12-tone-ht20-classic-band.md`](adr/2026-07-20-keep-the-12-tone-ht20-classic-band.md).

## Default Behavior

The binary production trainer loads `empty`, `static_presence`, and `motion`.
`empty` and `static_presence` are both IDLE targets; `motion` is the MOTION
target.

Current default training settings:

- `--fp-weight 2.0`
- `--scaler standard`
- `--batch-size 1024`
- `--device cpu`
- seed: reused from the current exported model when `--seed` is omitted

Values above `1.0` for `--fp-weight` reduce false positives at the cost of
slightly lower recall.

When `--seed` is omitted, diagnostic and training entry points reuse the seed
embedded in `ml_weights.py` / `ml_weights.h`. Pass an explicit `--seed` to
override that, or use `--seed-search-until-improvement` to sample fresh seeds.
If no exported seed is available, the trainer falls back to a random seed.

Use `--augment` to train with:
feature jitter (`0.10`) plus moderate packet gain/noise/loss. Augmentation is
train-only; paired validation and runtime inference stay on clean features.
Combine it with seed search when promoting a new export:

```bash
python tools/train_ml_model.py --augment
python tools/train_ml_model.py --augment --seed-search-until-improvement 10
```

CUDA and Apple MPS are available only when requested explicitly through
`--device cuda` or `--device mps`; this small MLP usually runs fastest and most
predictably on CPU. The trainer caches the derived feature matrix for repeat
runs; use `--no-cache` to force a rebuild.

## What The Trainer Does

The training pipeline:

1. Loads all `.npz` files from `data/` for `empty`, `static_presence`, and
   `motion`.
2. Uses the shared CV-normalized turbulence path (`std/mean`) across all files.
3. Extracts the selected ML feature set per sliding window. The production
   default is the Core-6 set. When Hampel is enabled, the trainer filters both
   base streams before feature extraction: turbulence for all `turb_*`
   features and per-packet L1 deltas for all `l1_delta*` features.
   Feature extraction uses the same fixed HT20 subcarrier band as the runtime,
   rather than re-optimizing the band independently for ML.
4. Runs grouped cross-validation by paired capture/session, with blocked
   scoring to reduce overlap optimism.
5. Optionally computes balanced SHAP explanations on the held-out blocked
   windows from each fold.
6. Reports worst-group metrics for session, chip, environment, and source file.
7. Trains the selected MLP architecture with PyTorch, early stopping, and dropout.
8. Evaluates the in-memory candidate on paired captures.
9. Exports Python and C++ runtime artifacts plus a regression dataset unless
   `--no-export` is set or the paired gate rejects the candidate.

Training uses uniform sample weights by default. Optional `--positive-chip-boost`
can still reweight motion windows for specific chips. Detector-guided sample
weighting was evaluated and rejected; see the related ADR.

## Exported Artifacts

Promoted exports:

- `src/python/micro_espectre/ml_weights.py`
- `src/cpp/core/ml_weights.h`
- `data/auto_generated/ml_test_data.npz`

When `--seed` is omitted, training reuses the seed saved in the current
exported weight files. Pass `--seed <number>` to override it; promoted exports
write the chosen seed back into those files.

`ml_test_data.npz` is an inference-regression artifact, not the main
model-selection metric. Architecture, weighting, and scaler choices should
treat grouped blocked CV as a diagnostic. Paired validation is the real-data
promotion gate and ranking signal. Long-recording checks stay in the
performance report and dedicated pytest suites.

## Promotion Guidance

For production artifact updates, prefer one of these gated flows:

- `python tools/train_ml_model.py`
- `python tools/train_ml_model.py --seed-search-until-improvement <N>`
- `python tools/train_ml_model.py --experiment --experiment-promote`
- `python tools/train_ml_model.py --experiment-fp-weights "..." --experiment-promote`

A normal training run exports when the paired gate passes and does not regress
against the exported baseline. Use `--no-export` to leave runtime artifacts
unchanged. Experiment campaigns leave artifacts unchanged unless
`--experiment-promote` is supplied. Seed search and experiment promotions
confirm the selected artifacts with the paired gate before keeping them.

Candidate ranking is paired-first: pass count, max FP rate, worst-chip recall,
and worst-chip F1 precede grouped CV. Long-recording FP policy metrics remain
useful in `generate_performance_report` and
`test_validation_long_recordings.py`, but they do not block trainer promotion.
Event recall and detection latency still require long recordings with an
annotated motion start and are not inferred from quiet-only captures.

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

When you want a synthetic weak-link proxy instead of a pure amplitude-gain
stress, use `tools/analyze_low_rssi_degradation.py`. That tool can attenuate
CSI I/Q payloads, lower reported RSSI independently, inject bounded additive
noise, perturb per-packet subcarrier profiles, and drop packets before
replaying the current Classic and ML detectors on the degraded streams. Treat
it as an exploratory robustness study, not as a substitute for real low-RSSI
captures. Keep the tool-specific scenario catalog, real-capture notes, and CLI
examples in [README.md](../tools/README.md).

```bash
python tools/analyze_low_rssi_degradation.py --chip ESP32 --scenario clean low_rssi_proxy_medium
```

## Cross-Environment And Cross-Chip Generalization Checks

Grouped CV splits by paired capture and session, so windows from the same room
or the same chip can land in both the training and the held-out fold. That makes
grouped CV optimistic about how well the detector transfers to a room or a chip
it has never seen.

Use the leave-one-group-out gates to measure true generalization without
exporting a new model:

```bash
python tools/train_ml_model.py --cross-environment
python tools/train_ml_model.py --cross-chip
python tools/train_ml_model.py --cross-environment --seed 42
```

`--cross-environment` holds out one named environment at a time; `--cross-chip`
holds out one chip at a time. With no `--seed`, both reuse the exported model
seed so the diagnostic matches the currently promoted weights. For each held-out
group the command trains on all other groups and evaluates on the held-out one,
then reports recall, false-positive rate, precision, and F1 per group plus a
macro-average. Held-out scoring reuses the same block subsampling as grouped CV,
so the numbers stay comparable to the trainer's own report. False positives are
also broken down by `empty` versus `static_presence`, since an unseen group's
idle windows are the most common cross-group false-positive source. Each fold
additionally reports its worst sub-group (worst chip for a held-out room, worst
room for a held-out chip).

Neither gate trains a promotable model or exports runtime artifacts. They are
mutually exclusive with each other and cannot be combined with `--environment`
(which holds nothing out), experiment flows, seed search, or the diagnostic
feature-analysis flags.

## Core-6 Robustness Campaign

Run the staged, non-destructive normalization and augmentation campaign with:

```bash
python tools/train_ml_model.py --experiment-robustness
```

The campaign screens the standard, robust, and session-balanced robust scalers;
relative L1 descriptors; normalized feature noise, block jitter, and feature
dropout; and packet-level frequency-selective gain, amplitude noise, and packet
loss. Every candidate is evaluated with all environment and chip holdouts.
Screening uses one seed, the shortlist uses three seeds, and the final baseline
comparison uses five seeds. Incremental results are written to
`data/auto_generated/ml_robustness_experiment.json`.

Runtime artifacts are never exported by this command. A final candidate must
still pass paired validation, gain stress, long recordings, and Python/C++
feature parity before a separate production promotion.

To keep the production Core-6 features and standard scaler fixed and evaluate
only feature-space and packet-level augmentation, use:

```bash
python tools/train_ml_model.py --experiment-robustness \
  --robustness-augmentation-only
```

The campaign winner
(`baseline_standard__feature_jitter_010__packet_packet_combined_moderate`) is
available for production training through `--augment`.

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
