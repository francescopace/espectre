# ML Training Guide

Train, evaluate, and promote the production High-Accuracy detector after collecting labeled CSI datasets.

Use [ML_DATA_COLLECTION.md](ML_DATA_COLLECTION.md) to collect and validate `empty`, `static_presence`, and `motion` recordings. This guide covers the training workflow, dataset roles, model-selection gates, exported artifacts, and required validation. Use `python tools/train_ml_model.py --help` for the complete option reference, [FEATURES.md](FEATURES.md) for feature evidence, and [performance/README.md](performance/README.md) for current detector results.

This guide is for ML contributors and assumes familiarity with supervised classification. Project-specific terms:

- **Lineage:** recordings that share enough provenance to remain in one validation group.
- **Grouped cross-validation:** fitting and evaluation folds keep each lineage together to reduce leakage.
- **OOF:** out-of-fold predictions produced for samples not used to fit that fold.
- **Replay gate:** a production-aligned recorded-data check that can block model promotion.
- **Promotion:** replacing the exported Python and C++ runtime weights after every required gate passes.

## Prerequisites

Install the ML training stack:

```bash
pip install -r requirements-ml.txt
```

The repository and ML workflows target Python `3.14`.

Before training, validate the corpus and refresh its generated quality report:

```bash
python tools/validate_dataset_quality.py
```

The production trainer admits only the HT20 sensing contract: `phy_mode=ht`, `ltf_type=ht-ltf`, `channel_width=20`, and the stored 64-subcarrier HT20 layout. Historical captures without per-record PHY metadata are admitted only when their payload already matches that layout. Incompatible files fail explicitly.

## Dataset Roles

`data/dataset_info.json` assigns each recording one role:

| Role | Purpose | Used to fit weights |
| --- | --- | --- |
| `train` | Training matrix and lineage-grouped cross-validation | Yes |
| `selection` | Candidate comparison and deployment safety gates | No |
| `holdout` | Final validation of the selected winner | No |
| `exclude` | Retained provenance or diagnostics outside model selection | No |

Entries without an explicit role default to `exclude` and must be admitted explicitly. The quality validator never assigns roles automatically.

An `empty` recording marked `long_recording: true` never enters the training matrix. When its role is `selection` or `holdout`, the quiet gate evaluates the complete recording and can block promotion. A long recording in `exclude` remains available only for explicit diagnostics and the generated quality report.

A normal one-candidate production run evaluates its final candidate on the configured selection and holdout replays before export. Seed search evaluates every candidate on `selection`, chooses one winner, and opens `holdout` only for that winner. Repeatedly consulting holdout results while changing the model turns the holdout into selection data and invalidates its role.

The split policy and its rationale are recorded in [2026-07-23-separate-ml-training-data-from-promotion-replays.md](adr/2026-07-23-separate-ml-training-data-from-promotion-replays.md).

## Production Training Workflow

Inspect the admitted corpus and split first:

```bash
python tools/train_ml_model.py --info
```

Run a production-compatible train with the promoted augmentation recipe:

```bash
python tools/train_ml_model.py --augment
```

Without `--features`, the trainer uses the promoted Subband 7F production order: `turb_iqr_over_mean_aggr`, `turb_autocorr`, `turb_zcr`, `l1_delta_lag_ratio`, `chan_shape_spread_subband`, `chan_shape_coherent_innovation_energy`, and `chan_shape_excess_path`. The trainer runs grouped cross-validation, fits the final candidate, evaluates the deployment replay gates, compares the candidate with the exported baseline, and exports new artifacts only when every promotion requirement passes.

Use read-only variants while investigating a change:

```bash
python tools/train_ml_model.py --augment --no-export
python tools/train_ml_model.py --augment --evaluate-gates
python tools/train_ml_model.py --augment --seed SEED --evaluate-gates
```

`--no-export` runs the training and grouped-CV path without replacing artifacts. `--evaluate-gates` also evaluates the in-memory candidate on deployment replays without exporting it. Pass an explicit seed when a controlled comparison must be reproducible; when omitted, the trainer reuses the seed embedded in the exported model when available.

For seed search:

```bash
python tools/train_ml_model.py --augment --seed-search-until-improvement TRIALS
python tools/train_ml_model.py --augment --seed-search-until-improvement TRIALS --no-export
```

Seed search writes its report after every trial to `data/auto_generated/mlp_seed_search.json` by default. Runtime-supported feature sets may export the selected winner unless `--no-export` is present. Searches containing host-only candidate features always remain in memory.

Use the same corpus, roles, features, preprocessing, augmentation components, and seed when comparing two training changes unless one of those variables is the subject of the experiment. A result produced without `--augment` is not directly comparable with the promoted augmented workflow.

## Training Contract

The binary target maps `empty` and `static_presence` to IDLE and `motion` to MOTION. Training uses the canonical `stream_dense` contract: it follows the runtime streaming feature path and timing resets, then emits one training row per packet after warmup.

The trainer:

1. loads admitted `train` recordings and applies the requested timing-quality policy;
2. builds the runtime-aligned feature stream;
3. runs blocked, lineage-grouped cross-validation so related recordings cannot cross folds;
4. reports overall, worst-group, and worst-five-tail metrics;
5. fits the final model on the complete training matrix;
6. evaluates paired and quiet deployment replays; and
7. exports runtime artifacts only after the promotion gates pass.

The current production feature definitions, topology, and retained evidence belong in [FEATURES.md](FEATURES.md). Runtime detector behavior belongs in [ALGORITHMS.md](ALGORITHMS.md). Changing a production feature, its subcarrier band, preprocessing, or runtime arithmetic requires aligned Python and C++ implementations followed by retraining and parity validation.

### Timing-quality policies

Timing quality is provenance, not a model input. The supported policies are:

```bash
python tools/train_ml_model.py --augment --timing-quality-policy keep
python tools/train_ml_model.py --augment --timing-quality-policy exclude-fail
python tools/train_ml_model.py --augment --timing-quality-policy downweight-warn
python tools/train_ml_model.py --augment --timing-quality-policy exclude-fail-downweight-warn
```

Use `--timing-warn-weight` only with a policy that downweights degraded recordings. The performance report groups replay results by timing-quality bucket so policy changes can be evaluated against the underlying provenance.

## Training Augmentation

Bare `--augment` enables the promoted `base,drift,burst-loss` recipe:

- `base` applies moderate feature jitter, packet-domain noise, loss, and stutter, and a stable packet-rate scale from `0.8` to `1.0`;
- `drift` injects a slow correlated packet-domain drift episode; and
- `burst-loss` injects short packet-drop bursts.

Production training builds two deterministic packet views with seeds `20260807` and `20260808`, then keeps alternating row positions from the two views within each source recording. This produces approximately one augmented row set rather than doubling the synthetic sample count, while exposing the model to the complementary false-positive and weak-recall stress tails of both seeds. The seed order and per-file modulo assignment are fixed; model seeds do not alter packet augmentation.

Augmentation is train-only for fitting and promotion gates. Cross-validation scoring, selection, holdout, and runtime inference use clean replay features. The generated performance report additionally labels a non-gating robustness diagnostic that applies the same two-seed packet recipe to the combined `selection + holdout` corpus and compares the exported ML and Lightweight detectors on matching alternating replay positions.

Stable rate scaling is not packet loss. It selects samples across the source interval, rewrites timestamps and sequence numbers to the lower clean cadence, and lets the shared `1000 ms` detector window resolve to fewer samples. Loss and burst-loss augmentations retain gaps and contamination semantics.

Explicit component lists are useful for controlled ablations:

```bash
python tools/train_ml_model.py --augment base --no-export
python tools/train_ml_model.py --augment base,drift --no-export
python tools/train_ml_model.py --augment base,drift,burst-loss --no-export
```

Historical augmentation comparisons and their measured outcomes belong in ADRs and [FEATURES.md](FEATURES.md), not in this guide.

## Model Selection And Promotion

Promotion is safety-first. The current stable gate policy is:

| Replay class | Recall | Raw FP | Effective alarms |
| --- | ---: | ---: | ---: |
| Normal-link paired replay | `>95%` | `<5%` | At most one per static-presence replay |
| Low-RSSI paired stress replay | `>90%` | `<10%` | Must not regress against the exported baseline |
| Quiet `empty` replay | N/A | `<5%` | Zero |

Passing the absolute targets is necessary but not sufficient. A candidate must also avoid material per-recording regressions against the exported model. Among safe candidates, the trainer compares paired replay quality, worst-session and worst-chip behavior, worst-five tails, and blocked out-of-fold metrics. Synthetic sessions may protect against regressions but cannot justify promotion over real-data evidence.

`--force-promote --seed SEED` bypasses gate failures and exports a fixed candidate while still printing the failed checks. Reserve it for a deliberate baseline reset whose rationale and evidence will be recorded separately.

Mutable performance numbers belong in [performance/README.md](performance/README.md). Durable feature and model decisions belong in [FEATURES.md](FEATURES.md) and the relevant ADR.

## Research And Diagnostic Workflows

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

Candidate features live in `tools/lib/candidate_features.py`. They may be selected with `--features`, but they cannot be exported until they have matching Python and C++ runtime implementations and a published feature ID. Retired candidate evidence remains in `docs/FEATURES.md`; retired implementations are not kept executable solely for historical comparisons. Use `--evaluate-gates` or `--no-export` while evaluating current candidates.

Use leave-one-group-out diagnostics to estimate transfer to unseen rooms or chips:

```bash
python tools/train_ml_model.py --augment --cross-environment
python tools/train_ml_model.py --augment --cross-chip
```

These commands train diagnostic folds and never export a promotable model. Grouped CV can still contain the same room or chip on both sides of a fold, so it does not replace these transfer checks.

Use the gain-stress gate to inspect the current exported artifacts without training:

```bash
python tools/train_ml_model.py --gain-stress-gate
python tools/train_ml_model.py --gain-stress-gate --environment bedroom
```

The gain-stress gate measures sensitivity to explicit amplitude-gain dimensions. It does not model low-RSSI feature-floor drift; validate weak links with real `low_rssi` recordings.

## Exported Artifacts

A successful promotion updates:

- `src/python/micro_espectre/ml_weights.py`;
- `src/cpp/core/ml_weights.h`; and
- `data/auto_generated/ml_test_data.npz`.

The exported weight files store the training seed and complete runtime arrays. `ml_test_data.npz` is an inference-regression artifact, not a model-selection score.

Do not edit generated weight files manually. Export them through the trainer so Python, C++, and regression data remain aligned.

## Cache Maintenance

Training and replay tools persist runtime-aligned feature artifacts under `.npz_cache/`. Runtime-supported replay rows remain cached as complete matrices. Host-side experiments use a shared row spine (`packet_index`, evaluation index, reset index, and evaluation cadence) plus one independently keyed column per feature. Each column identity contains its feature-local formula and provider contract, so adding a sibling variant computes only the new column; existing columns remain reusable across reordered feature sets and model comparisons. Shared provider-contract changes invalidate only columns owned by that provider. The row-spine check rejects columns whose replay coordinates diverge rather than silently combining incompatible data.

Cache keys include source data, timing behavior, implementation identity, and augmentation provenance. Lightweight source-admission metadata is cached separately, and warm host-column hits do not materialize the CSI packet stream. Mixed production augmentation remains cached per source under `ml_training_augmentation_rows`; host-only augmented views also persist their full row spine and feature columns before deterministic row selection. Producers take a per-artifact process lock and recheck after acquiring it, preventing concurrent seed searches or reports from rebuilding the same cold key. Cache entries, seed-search JSON, generated reports, dataset metadata, and individual exports use atomic replacement. The three model outputs are staged and published as one rollback-capable set, and seed-search rollback removes outputs that did not exist before the search.

Use `--no-cache` for a cold feature-row diagnostic run; source-admission metadata remains cached because it does not change the matrix. Use `python tools/prune_npz_cache.py` to remove artifacts that are no longer reachable, and add an explicit age or size limit when historical parameter variants should also be retired. Generated report freshness checks include a digest of detector, model, tool, and capture inputs in addition to `dataset_info.json`. Detailed cache behavior and pruning options belong in [tools/README.md](../tools/README.md).

## Required Validation

After changing detection logic, features, preprocessing, or exported weights, run both required parity gates:

```bash
cmake -S test/cpp -B test/cpp/build
cmake --build test/cpp/build
ctest --test-dir test/cpp/build -R test_motion_detection --output-on-failure
.venv/bin/pytest test/python/test_validation_real_data.py::TestPerformanceMetrics -v
```

Validate long quiet recordings and regenerate the performance report:

```bash
.venv/bin/pytest test/python/test_validation_long_recordings.py -v
.venv/bin/python tools/generate_performance_report.py
.venv/bin/python tools/generate_performance_report.py --check-current
```

When the corpus, roles, or dataset-quality logic changes, also regenerate and verify the quality report:

```bash
.venv/bin/python tools/validate_dataset_quality.py
.venv/bin/python tools/validate_dataset_quality.py --check-current
.venv/bin/pytest test/python/test_dataset_quality_validation.py -v
```

Do not claim a promotion is complete until the generated reports are current and every required Python/C++ gate passes.

## Related Documentation

- [ML_DATA_COLLECTION.md](ML_DATA_COLLECTION.md): collection and labeling workflow
- [FEATURES.md](FEATURES.md): production feature set, research ledger, and retained evidence
- [ALGORITHMS.md](ALGORITHMS.md#ml-detector): runtime detector behavior
- [performance/README.md](performance/README.md): generated current performance
- [tools/README.md](../tools/README.md): complete tool reference and cache operations
- [2026-07-23-separate-ml-training-data-from-promotion-replays.md](adr/2026-07-23-separate-ml-training-data-from-promotion-replays.md): split and promotion rationale
