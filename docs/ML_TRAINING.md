# ML training guide

This guide explains how to train, test, and promote the High Accuracy model. It is for ML contributors and assumes you know supervised classification. Collect the data first with the [data collection guide](ML_DATA_COLLECTION.md).

For all options, run `python tools/train_ml_model.py --help`. Feature evidence is in the [feature ledger](FEATURES.md), current results in the [performance report](performance/README.md).

Terms:

- **Lineage:** recordings so closely related that they must stay in the same validation group.
- **Grouped cross-validation:** each lineage stays entirely in one fold, so related recordings cannot leak between training and evaluation.
- **OOF:** out-of-fold predictions, made on samples the fold was not trained on.
- **Replay gate:** a check that runs recordings through the production pipeline and can block promotion.
- **Promotion:** replacing the Python and C++ model weights, once every gate passes.

## Prerequisites

Install the ML training stack:

```bash
pip install -r requirements-ml.txt
```

Use Python `3.14`. Before training, validate the dataset and refresh its quality report:

```bash
python tools/validate_dataset_quality.py
```

The trainer accepts only HT20 data (`phy_mode=ht`, `ltf_type=ht-ltf`, `channel_width=20`, 64 subcarriers). Old recordings without PHY fields are accepted only if they already have that layout; anything else fails with an error.

## Dataset roles

`data/dataset_info.json` assigns each recording one role:

| Role | Purpose | Used to fit weights |
| --- | --- | --- |
| `train` | Training matrix and lineage-grouped cross-validation | Yes |
| `selection` | Candidate comparison and deployment safety gates | No |
| `holdout` | Final validation of the selected winner | No |
| `exclude` | Retained provenance or diagnostics outside model selection | No |

- A missing role counts as `exclude`, so an incomplete catalog never leaks into training. The quality validator is stricter and fails until every entry has a role. Roles are always set by hand.
- Long quiet recordings (`long_recording: true`) are never trained on. With role `selection` or `holdout`, the quiet gate replays them in full and can block promotion.
- A single production run tests its model on `selection` and `holdout` before export. A seed search tests every candidate on `selection`, picks one winner, and opens `holdout` only for it.
- **Keep the holdout sealed.** Looking at holdout results while you keep changing the model turns it into selection data.

The split policy and its rationale are recorded in [2026-06-30-separate-ml-training-data-from-promotion-replays.md](adr/2026-06-30-separate-ml-training-data-from-promotion-replays.md).

## Production training workflow

Inspect the admitted corpus and split first:

```bash
python tools/train_ml_model.py --info
```

Without `--features`, the trainer uses the promoted Subband 8F production order: `turb_iqr_over_mean_aggr`, `turb_autocorr`, `turb_zcr`, `l1_delta_lag_ratio`, `chan_shape_spread_subband`, `chan_shape_coherent_innovation_energy`, `chan_shape_excess_path`, and `chan_shape_subband_kendall_lag_excess`.

Use read-only variants while investigating a change:

```bash
python tools/train_ml_model.py --augment --no-export
python tools/train_ml_model.py --augment --seed SEED --evaluate-selection
```

- `--no-export` trains and runs grouped CV only: nothing is replaced and no replay gate is opened.
- `--evaluate-selection` also runs the `selection` gates (normal and occupancy-70%), keeping `holdout` sealed.
- Without `--seed`, the trainer reuses the seed of the current exported model.

Once features, settings, and seed are final, promote with one production run:

```bash
python tools/train_ml_model.py --augment --seed SEED
```

This runs grouped CV, fits the model, runs the `selection` and `holdout` gates, compares against the current model, and exports only if everything passes.

For a host-only finalist that cannot be exported, run the same final gates once without exporting:

```bash
python tools/train_ml_model.py --augment --seed SEED --evaluate-gates
```

Either way, the result validates a finished candidate. Do not use it to tune further.

For seed search:

```bash
python tools/train_ml_model.py --augment --seed-search-until-improvement TRIALS
python tools/train_ml_model.py --augment --seed-search-until-improvement TRIALS --no-export
```

The report is updated after each trial in `data/auto_generated/mlp_seed_search.json`. The winner is exported unless you pass `--no-export` or use host-only features.

When comparing two changes, keep everything else the same: data, roles, features, preprocessing, augmentation, and seed. Results without `--augment` cannot be compared with the production workflow.

## Training contract

`empty` and `static_presence` are IDLE; `motion` is MOTION. Features are computed exactly as on the device (`stream_dense`): same slot grid, same resets, one row per packet after warm-up. Missing slots are skipped in turbulence statistics, and trajectory features use the packet timestamps.

The trainer:

1. loads admitted `train` recordings and applies the requested timing-quality policy;
2. builds the runtime-aligned feature stream;
3. runs blocked, lineage-grouped cross-validation so related recordings cannot cross folds;
4. reports overall, worst-group, and worst-five-tail metrics;
5. fits the final model on the complete training matrix;
6. evaluates paired and quiet deployment replays; and
7. exports runtime artifacts only after the promotion gates pass.

Feature definitions and evidence are in the [feature ledger](FEATURES.md); detector behavior is in the [algorithms reference](ALGORITHMS.md). Changing a production feature, its subcarriers, preprocessing, or arithmetic means updating Python and C++ together, then retraining and checking parity.

### Timing-quality policies

Timing quality describes each recording; it is not a model input. It decides which recordings to use or down-weight:

```bash
python tools/train_ml_model.py --augment --timing-quality-policy keep
python tools/train_ml_model.py --augment --timing-quality-policy exclude-fail
python tools/train_ml_model.py --augment --timing-quality-policy downweight-warn
python tools/train_ml_model.py --augment --timing-quality-policy exclude-fail-downweight-warn
```

`--timing-warn-weight` works only with a down-weighting policy. The performance report groups results by timing quality, so you can see the effect of a policy.

## Training augmentation

Bare `--augment` enables the `base,drift,burst-loss` recipe:

- `base` applies moderate feature jitter, packet-domain noise, loss, and stutter, and a stable packet-rate scale from `0.7` to `1.0` with a 70 pps floor that matches the temporal-admission occupancy envelope;
- `drift` injects a slow correlated packet-domain drift episode; and
- `burst-loss` injects short packet-drop bursts.

The exported High Accuracy artifact uses the occupancy-70% `base` scale `0.7-1.0` with a 70 pps floor.

```bash
python tools/train_ml_model.py --augment --seed 656446646 --evaluate-selection
```

How augmentation works:

- Two augmented views are built with fixed packet seeds `20260807` and `20260808`, and rows alternate between them within each recording. The result is about one augmented set, not two, but it covers the stress cases of both seeds. The model seed does not change augmentation.
- Augmentation is used only for fitting. CV scores, `selection`, `holdout`, and the device use clean data. The performance report also shows a non-blocking check that applies the same augmentation to `selection + holdout` and compares ML and Lightweight.
- Rate scaling produces a clean, slower stream (new timestamps and sequence numbers), not packet loss. Slots still come from the recorded `csi_target_pps` and the window stays `1000 ms`. Loss and burst loss leave real gaps.

Explicit component lists are useful for controlled ablations:

```bash
python tools/train_ml_model.py --augment base --no-export
python tools/train_ml_model.py --augment base,drift --no-export
python tools/train_ml_model.py --augment base,drift,burst-loss --no-export
```

Historical augmentation comparisons and their measured outcomes belong in ADRs and the [feature ledger](FEATURES.md), not in this guide.

## Model selection and promotion

Promotion is safety-first. The current stable gate policy is:

| Replay class | Recall | Raw FP | Effective alarms |
| --- | ---: | ---: | ---: |
| Normal-link paired replay | `>95%` | `<5%` | At most one per static-presence replay |
| Low-RSSI paired stress replay | `>90%` | `<10%` | Must not regress against the exported baseline |
| Quiet `empty` replay | N/A | `<5%` | Zero |
| Occupancy-70% paired replay | same absolute cuts | same absolute cuts | same alarm rules, after deterministic thinning of reserved pairs to the production occupancy envelope |
| Occupancy-70% quiet replay | N/A | `<5%` | Zero, on the same thinned empty reserved set |

The occupancy-70% gate removes packets evenly until about 70% of the slots are filled (the minimum the device accepts), then scores the model. The normal replays are still required. Even thinning approximates real conditions; it does not simulate Bluetooth interference.

Passing these limits is required but not enough: the candidate must not get noticeably worse than the current model on any recording. Among safe candidates, the trainer compares paired replays, the worst session and chip, the five worst cases, and OOF metrics. Synthetic data can block a regression but cannot justify a promotion.

`--force-promote --seed SEED` exports even when gates fail (the failures are still printed). Use it only for a deliberate reset, and record why.

Mutable performance numbers belong in the [performance report](performance/README.md). Durable feature and model decisions belong in the [feature ledger](FEATURES.md) and the relevant ADR.

## Research and diagnostic workflows

Architecture and false-positive-weight campaigns are read-only and write JSON reports:

```bash
python tools/train_ml_model.py --augment --experiment
python tools/train_ml_model.py --augment --experiment --experiment-architectures "16,8;24,12;32,16"
python tools/train_ml_model.py --augment --experiment-fp-weights "1,1.5,2,2.5,3"
```

Feature diagnostics also leave runtime artifacts unchanged:

```bash
python tools/train_ml_model.py --correlation
python tools/train_ml_model.py --augment --shap 500 --seed SEED --no-export
python tools/train_ml_model.py --augment --ablation-feature FEATURE_OR_JOINT_REMOVAL --seed SEED
```

Candidate features live in `tools/lib/candidate_features.py` and can be selected with `--features`. They cannot be exported until they have matching Python and C++ implementations and a feature ID. Evidence for retired candidates stays in the [feature ledger](FEATURES.md); their code is removed.

Trajectory-bin experiments use the same host streaming path and keep the production `80 ms` default unless explicitly overridden:

```bash
python tools/train_ml_model.py --augment --seed SEED --trajectory-bin-ms 50 --evaluate-selection
```

Other bin sizes never export, and comparisons with the current model always use the production `80 ms` bin.

Use leave-one-group-out diagnostics to estimate transfer to unseen rooms or chips:

```bash
python tools/train_ml_model.py --augment --cross-environment
python tools/train_ml_model.py --augment --cross-chip
```

They never export. Grouped CV can still have the same room or chip on both sides of a fold, so it does not replace these checks.

Use the gain-stress gate to inspect the current exported artifacts without training:

```bash
python tools/train_ml_model.py --gain-stress-gate
python tools/train_ml_model.py --gain-stress-gate --environment bedroom
```

It checks how the model reacts to amplitude gain changes. It does not simulate weak signals; use real `low_rssi` recordings for that.

## Exported artifacts

A successful promotion updates:

- `tools/lib/ml_weights.py`;
- `src/cpp/core/ml_weights.h`; and
- `data/auto_generated/ml_test_data.npz`.

The weight files store the seed, time, feature order, scaler, topology, and weights, but not the dataset revision or the selection settings. Record those (dataset revision, roles, timing policy, augmentation, and fitting parameters) in the production section of the [feature ledger](FEATURES.md). `ml_test_data.npz` is a regression test for inference, not a quality score.

Never edit these files by hand: export them with the trainer so Python, C++, and test data stay in sync.

## Cache maintenance

Training and replay tools cache computed features under `.cache/npz/`, so repeated runs and new feature variants only compute what is missing. The cache checks its inputs automatically. All outputs are written atomically, and the three model files are published together (a seed search rolls back files it created).

- `--no-cache` forces a cold run of the feature rows.
- `python tools/prune_npz_cache.py` removes entries that can no longer be used.
- `ESPECTRE_NPZ_CACHE_PROGRESS=0` or `=1` turns progress lines off or on.

Report freshness checks cover the detector, model, tool, and capture inputs, not just `dataset_info.json`. See [cache maintenance](../tools/README.md#cache-maintenance) in the tools guide for details.

## Required validation

After changing detection logic, features, preprocessing, or exported weights, run both required parity gates:

```bash
cmake -S test/cpp -B test/cpp/build
cmake --build test/cpp/build
ctest --test-dir test/cpp/build -R test_motion_detection --output-on-failure
.venv/bin/pytest test/python/performance/test_validation_real_data.py::TestPerformanceMetrics -v
```

Validate long quiet recordings and regenerate the performance report:

```bash
.venv/bin/pytest test/python/performance/test_validation_long_recordings.py -v
.venv/bin/python tools/generate_performance_report.py
.venv/bin/python tools/generate_performance_report.py --check-current
```

When the corpus, roles, or dataset-quality logic changes, also regenerate and verify the quality report:

```bash
.venv/bin/python tools/validate_dataset_quality.py
.venv/bin/python tools/validate_dataset_quality.py --check-current
.venv/bin/pytest test/python/host/dataset/test_dataset_quality_validation.py -v
```

A promotion is complete only when the reports are current, every Python and C++ gate passes, and the [feature ledger](FEATURES.md) records the dataset revision and training settings of the promoted run.

## Related documentation

- [data collection guide](ML_DATA_COLLECTION.md): collection and labeling workflow
- [feature ledger](FEATURES.md): production feature set, research ledger, and retained evidence
- [High-Accuracy implementation: HighAccuracyDetector](ALGORITHMS.md#high-accuracy-implementation-highaccuracydetector): runtime detector behavior
- [Performance report](performance/README.md): generated current performance
- [Tools guide](../tools/README.md): complete tool reference and cache operations
- [2026-06-30-separate-ml-training-data-from-promotion-replays.md](adr/2026-06-30-separate-ml-training-data-from-promotion-replays.md): split and promotion rationale
