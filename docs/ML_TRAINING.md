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

The production trainer admits only the HT20 sensing contract: `phy_mode=ht`,
`ltf_type=ht-ltf`, `channel_width=20`, and the stored 64-subcarrier HT20
layout. Historical captures that omit all per-record PHY metadata are allowed
only when their payload already matches that exact layout. Any other PHY or
layout now fails explicitly instead of falling back silently.

## Basic Training Workflow

Run the default trainer:

```bash
python tools/train_ml_model.py
```

This evaluates grouped CV, trains the final candidate, runs the deployment
replay gates, and exports runtime artifacts when those gates pass and do not
regress against the current baseline. Use `--no-export` to evaluate without
replacing artifacts.

Useful variants:

```bash
python tools/train_ml_model.py --info
python tools/train_ml_model.py --scaler clipped_standard
python tools/train_ml_model.py --device mps
python tools/train_ml_model.py --exclude-chip ESP32
python tools/train_ml_model.py --timing-quality-policy exclude-fail
python tools/train_ml_model.py --timing-quality-policy exclude-fail-downweight-warn --timing-warn-weight 0.25
python tools/train_ml_model.py --no-export
python tools/train_ml_model.py --gain-stress-gate
python tools/train_ml_model.py --gain-stress-gate --environment bedroom
python tools/train_ml_model.py --seed-search-until-improvement 20
python tools/train_ml_model.py --features turb_autocorr,turb_zcr,l1_delta_autocorr --no-export
python tools/train_ml_model.py --features turb_mad_over_mean,turb_autocorr,turb_zcr,l1_delta_autocorr,l1_delta_lag_ratio --experiment
```

`--features` selects a subset of the canonical runtime feature surface and is
propagated to architecture and FP-weight campaigns. Host-only candidates are
evaluated through the time-aware `replay_classic_candidates.py` and
`benchmark_classic_candidate_pairs.py` research tools; they no longer enter a
separate trainer matrix. Every removed production feature and the measurement that
rejected it is listed in
[2026-07-27-reduce-the-feature-surface-to-the-production-set.md](adr/2026-07-27-reduce-the-feature-surface-to-the-production-set.md).

For exploratory architecture campaigns:

```bash
python tools/train_ml_model.py --experiment
python tools/train_ml_model.py --experiment --experiment-architectures "16,8;24,12;32,16;24;24,12,6"
```

For a gated FP-weight campaign:

```bash
python tools/train_ml_model.py --experiment-fp-weights "1,1.5,2,2.5,3"
```

Both campaigns use the exported seed for single-seed screening, retain the
baseline among the finalists, and then apply three- and five-seed robustness
comparisons. They only write their JSON report and never replace the exported
runtime artifacts.

For feature diagnostics:

```bash
python tools/train_ml_model.py --correlation
python tools/train_ml_model.py --shap 500 --seed 1386543369 --no-export
python tools/train_ml_model.py --ablation-feature l1_delta_autocorr --seed 1386543369
```

Correlation is a fast marginal screen over the full training matrix. SHAP runs
inside grouped cross-validation: each fold uses a class-, chip-, and
session-balanced background from its training partition and explains only
balanced, blocked windows from the held-out partition. Supplying `--seed` makes
training, sampling, and permutation SHAP reproducible. Use `--no-export` for
diagnostic runs so the current runtime artifacts remain unchanged.
`--ablation-feature` compares the production set against one feature removal
using the same seed, grouped CV, and paired validation. It also leaves the exported runtime
artifacts unchanged.

The latest diagnostic snapshot and interpretation live in
[ALGORITHMS.md](ALGORITHMS.md). Recompute the values after changing the dataset,
feature set, preprocessing, model architecture, or training policy.
The production ML path also stays aligned with the same fixed 12-tone HT20 band
used by Classic, so retrain and re-export whenever that band changes. For how
the band was derived from channel coherence, and why the count stays at 12, see
[`2026-07-25-select-the-classic-band-from-channel-coherence.md`](adr/2026-07-25-select-the-classic-band-from-channel-coherence.md).

## Default Behavior

The binary production trainer loads `empty`, `static_presence`, and `motion`.
`empty` and `static_presence` are both IDLE targets; `motion` is the MOTION
target. Quiet long-run `empty` captures marked with `long_recording: true` stay
evaluation-only and do not enter the training matrix.

If format filtering removes every packet from a selected file, training stops
with an explicit error so incompatible captures cannot contaminate the dataset.

Current default training settings:

- `--fp-weight 1.75`
- `--hidden-layers 24,12`
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

Use `--augment` to enable one or more train-time augmentation components.
`--augment` with no value is shorthand for `--augment base`, which keeps the
validated production recipe. Augmentation is train-only; paired validation and
runtime inference stay on clean features. See
[Training Augmentation](#training-augmentation) before reaching for it.

```bash
python tools/train_ml_model.py --seed-search-until-improvement 10
python tools/train_ml_model.py --augment --seed-search-until-improvement 10
python tools/train_ml_model.py --augment base,drift --seed-search-until-improvement 10
```

Seed search writes `data/auto_generated/mlp_seed_search.json` after every
trial, so a run that crashes or is interrupted still leaves its evidence
behind. Override the location with `--seed-search-output`. The report holds
the per-replay rows for the baseline and for each candidate, and, for any
candidate the gate rejected, the exact comparisons that blocked it:

```json
{"replay": "C6:selection:normal.npz", "metric": "fp_rate",
 "candidate": 1.46, "baseline": 0.44, "margin": 0.730,
 "candidate_evaluations": 10, "baseline_evaluations": 3, "eval_count": 685}
```

The same lines are printed to the console. Percentages hide how small these
differences are, so the report carries the evaluation counts that produced
them: the margin above is five evaluations out of `685`, set from measured
seed-to-seed dispersion rather than chosen. Analyse a finished run with
`tools/analyze_seed_dispersion.py` and re-derive it when the corpus changes.

Architecture and false-positive-weight campaigns write their own reports, to
`data/auto_generated/mlp_architecture_experiment.json` and
`mlp_fp_weight_experiment.json`; override either with `--experiment-output` or
`--fp-weight-experiment-output`. Both are gitignored. `--hidden-layers` sets
the MLP widths for a single run, for example `--hidden-layers 64,32`.

CUDA and Apple MPS are available only when requested explicitly through
`--device cuda` or `--device mps`; this small MLP usually runs fastest and most
predictably on CPU. Host-side tooling uses two cache layers: an in-process
runtime memo for loaded arrays and packet views, and persisted artifacts under
`.npz_cache/`. The canonical persisted ML artifact is one time-aware ready-packet row
stream per capture. It stores every ready runtime feature row, contamination
reset identity, and whether the row is a deployment evaluation tick.
`stream_dense` training consumes every row; dataset-quality validation uses the
same canonical stream; trainer gates, performance reporting, and replay tests
project the marked runtime ticks. Numeric weight changes do not invalidate this
feature-only artifact. Dataset-quality idle summaries are recomputed from these
rows rather than persisted in a separate summary cache. Deterministic
packet-augmentation rows are variants of the same artifact, keyed by the
complete packet recipe, augmentation seed, and implementation digest. Repeating
a research run with identical provenance therefore reuses them, while a seed,
recipe, or implementation change cannot silently alias another run. Delete
`.npz_cache/` to force a cold rebuild of persisted artifacts, or use
`--no-cache` to bypass them for one run. The legacy per-feature
`feature_column` cache, idle-baseline summary cache, and non-time-aware `dense`
contract have been removed. Pruning is an explicit maintenance operation:
`python tools/prune_npz_cache.py` scans all known persisted artifacts, while
repeated `--artifact NAME` options restrict it to selected artifact types.
Normal cache reads do not create `.npz_cache/` or artifact directories.

Training now has one contract: `stream_dense` keeps the runtime streaming
feature path and timing resets, but emits one training row per packet after
warmup. The fully sparse `replay_tick`
training path proved too sample-starved on the current corpus, while
`stream_dense` preserved the runtime streaming semantics closely enough to pass
deployment gates and outperform the retired legacy path under the stronger
structured augmentation mixes. Track the rollout sequence and remaining promotion evidence
in
[`timing-aware-training-review-2026-07-30.md`](review/timing-aware-training-review-2026-07-30.md).

The first timing-aware controls are provenance-only, not new model inputs:

- `--timing-quality-policy keep` records timing quality in the training summary
  but leaves the matrix untouched
- `--timing-quality-policy exclude-fail` drops files whose timing metadata
  crosses the existing continuity fail thresholds
- `--timing-quality-policy downweight-warn` keeps degraded files but applies a
  lower per-window training weight
- `--timing-quality-policy exclude-fail-downweight-warn` combines both

`tools/generate_performance_report.py` now also renders a timing-quality audit
that groups replay results by `clean`, `degraded`, and `poor` timing buckets,
so provenance can be inspected before changing the sample contract.

Training supports the runtime ML feature surface only. Host-only exploratory
features use time-aware replay ticks in the dedicated Classic candidate tools
until they are either promoted into the runtime contract or retired.

## What The Trainer Does

The training pipeline:

1. Loads `train`-role `.npz` files from `data/` for `empty`,
   `static_presence`, and `motion`. Entries without a role remain `train` for
   backward compatibility.
2. Uses the shared CV-normalized turbulence path (`std/mean`) across all files.
3. Extracts the selected ML feature set per sliding window. The production
   default is the Invariant-5 set. When Hampel is enabled, the trainer filters both
   base streams before feature extraction: turbulence for all `turb_*`
   features and per-packet L1 deltas for all `l1_delta*` features.
   Feature extraction uses the same fixed HT20 subcarrier band as the runtime,
   rather than re-optimizing the band independently for ML.
4. Runs grouped cross-validation by provenance lineage, with blocked scoring
   to reduce overlap optimism. Synthetic derivatives share the lineage of
   their real source, so they cannot cross the train/validation boundary.
5. Optionally computes balanced SHAP explanations on the held-out blocked
   windows from each fold.
6. Reports worst-group and worst-five-tail metrics for lineage, session, chip,
   environment, and source file. When synthetic derivatives are present, the
   session metrics are additionally split by provenance into real and
   synthetic reports.
7. Trains the selected MLP architecture with PyTorch, early stopping, and dropout.
8. Evaluates the in-memory candidate on real paired captures and any explicitly
   reserved quiet recordings, using production cadence and hit filtering.
9. Exports Python and C++ runtime artifacts plus a regression dataset unless
   `--no-export` is set or the paired gate rejects the candidate.

Training uses uniform sample weights. Detector-guided sample weighting was
evaluated and rejected; see the related ADR.

## Exported Artifacts

Promoted exports:

- `src/python/micro_espectre/ml_weights.py`
- `src/cpp/core/ml_weights.h`
- `data/auto_generated/ml_test_data.npz`

When `--seed` is omitted, training reuses the seed saved in the current
exported weight files. Pass `--seed <number>` to override it; promoted exports
write the chosen seed back into those files.

`ml_test_data.npz` is an inference-regression artifact, not the main
model-selection metric. Paired and reserved quiet replays are deployment
safety gates. Among candidates that pass those gates without a material
per-recording regression, grouped blocked CV worst-group, worst-five-tail, and
OOF metrics determine promotion. Long-recording checks stay in the performance
report and dedicated pytest suites.

## Promotion Guidance

For production artifact updates, prefer one of these gated flows:

- `python tools/train_ml_model.py`
- `python tools/train_ml_model.py --seed-search-until-improvement <N>`

A normal training run exports when the deployment replay gates pass and do not
regress against the exported baseline. Use `--no-export` to leave runtime
artifacts unchanged. Experiment campaigns are always read-only and leave
artifacts unchanged.

The train/evaluation separation, dataset roles, and link-class policy are a
durable decision; see
[2026-07-23-separate-ml-training-data-from-promotion-replays.md](adr/2026-07-23-separate-ml-training-data-from-promotion-replays.md)
for the rationale and the alternatives that were rejected.

Seed search evaluates every requested seed. Safety comes first: on normal-link
replays paired recall must remain above `95%`, raw FP must remain below `5%`,
runtime-filtered effective alarms must stay within a budget of one per
static-presence replay, and each recording may move by at most one scored
evaluation relative to the baseline. The one-alarm budget exists because a
sustained micro-motion of the present person is genuine motion, not model
noise; quiet `empty` replays keep a zero-alarm requirement, and the
per-recording non-regression checks still forbid exceeding the exported
baseline's alarms on any individual replay. Real weak-link
(`low_rssi: true`) replays are stress diagnostics: motion is barely separable
from the noise floor at very low RSSI, so they gate with the relaxed stress
targets (recall above `90%`, FP below `10%`), and the same split applies to
the validation suite and the performance report. In the per-recording
non-regression checks, weak replays ratchet only their alarm count against
the baseline; their recall and FP move freely within the stress targets,
because at `-75/-77 dBm` both jitter by whole events between equally healthy
models. Safe candidates are then compared first on paired replay quality,
prioritizing worst-chip recall before false-positive burden that still remains
within budget, and then on worst-session recall/FP, the mean of the five worst
sessions, worst-chip recall/FP, and blocked OOF F1. When synthetic derivatives
exist, real sessions lead those grouped-CV comparisons; synthetic session
metrics act only as regression guards and cannot justify promotion on their
own. A candidate still needs at least one material grouped-CV improvement and
no material regression before that paired-first ranking is allowed to decide
the winner. When explicit `holdout` data exists, it stays sealed throughout
selection and is evaluated only once on the chosen winner.

`--force-promote --seed <number>` exports a specific candidate even when the
deployment safety gates fail or regress. The gates still run and print their
results, and the bypass is reported loudly. Reserve it for deliberate baseline
resets, such as replacing a model whose gate results predate the reserved
`selection`/`holdout` split and are therefore in-sample.

Long-recording FP policy metrics remain useful in `generate_performance_report` and
`test_validation_long_recordings.py`, but they do not block trainer promotion.
Event recall and detection latency still require long recordings with an
annotated motion start and are not inferred from quiet-only captures.

For exploratory retrains, `--scaler clipped_standard`, alternate `--device`
choices, `--no-cache`, and smaller `--batch-size` values are available, but
promotable artifacts should still pass the validation checks below.

## Evaluating A Candidate Feature

Candidate features live in `tools/lib/candidate_features.py`, on the host side
only. They are selectable through `--features` but are absent from
`CPP_FEATURE_IDS`, so any flow that would export runtime artifacts refuses to
run until the candidate is ported; use `--no-export` or one of the read-only
flows. The production surface in `csi_features.py` stays exactly the exported
set. See [FEATURES.md](FEATURES.md) for the current production and candidate
inventories.

Screen redundancy before anything else, because a candidate earns its place by
what it adds rather than by how well it separates alone:

```bash
python tools/train_ml_model.py --features "turb_mad_over_mean,turb_autocorr,turb_zcr,l1_delta_autocorr,l1_delta_lag_ratio,chan_coh_lag_ratio" --correlation
```

The report prints each candidate's strongest pairwise correlation against the
production members and the `R2` of a least-squares fit on all of them. A high
`R2` means the production set already reconstructs the candidate, and no
downstream gate will recover value it does not carry.

Candidates run through the same streaming path as production features, so
`--cross-chip`, `--gain-stress-gate`, and the replay gates measure them the way
a runtime would. `--evaluate-gates --no-export` also reports the exact
per-recording regressions against the exported baseline. Only a candidate that
survives the promotion protocol above earns a calc function in both languages,
an `MLFeatureId`, and a `CPP_FEATURE_IDS` entry.

The authoritative inventory, definitions, retained metrics, verdicts, and
future physical axes are in [FEATURES.md](FEATURES.md). All currently
implemented candidates are scale-invariant and remain research-only until
their incremental value and replay robustness justify a production port.
See [LITERATURE.md](LITERATURE.md) for the primary-paper evidence behind those
physical axes and for the hardware assumptions that limit transfer.

## Gain-Shift Robustness Check

The production ML path deliberately keeps Python/C++ runtime inference aligned
by deriving all neural-detector inputs from the same raw CSI stream and shared
tracker families. The exported ten-feature phaseless set combines gain- and
offset-invariant turbulence, L1-delta, channel-shape, and
delay-compensated-coherence statistics, so the model is structurally less
sensitive to absolute amplitude gain changes than the older energy-like
baselines were. Host-side gates, reports, and replay tests call the same
float32 array-inference helper, including the runtime sigmoid saturation rules;
the generated regression artifact and C++ parity tests guard the exported
implementation.

Use the exported-artifact gain-stress gate to quantify this risk without
retraining or exporting a new model:

```bash
python tools/train_ml_model.py --gain-stress-gate
python tools/train_ml_model.py --gain-stress-gate --environment bedroom
python tools/train_ml_model.py --gain-stress-gate --gain-stress-scales 0.75,1.0,1.25
```

`--gain-stress-gate` does not train or export. It loads the current exported
`src/python/micro_espectre/ml_weights.py`, scales only the amplitude-gain-sensitive input
features when present, and reports recall/FP degradation overall plus worst
chip, environment, session, and source-file groups.

Current finding for the exported model: the table is flat at `1.00x`, `1.25x`,
and `1.50x`, and the summary reports `Scaled features: none`. That means the
current export has no gain-sensitive feature dimensions left, so this gate is
now mainly an informational regression guard against reintroducing them. The
remaining worst-session weakness is nominal dataset difficulty, not
gain-shift sensitivity.

Amplitude-gain stress does not model the feature-floor drift seen on weak Wi-Fi
links. Validate low-RSSI behavior separately with real captures registered in
`data/dataset_info.json` and production-path detector regressions. Classic has
a startup-centered safeguard for this case; ML low-RSSI behavior remains a
separate promotion problem that requires additional real-data evidence.

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

## Training Augmentation

`--augment` enables one or more train-time augmentation components:

- `base`: the current validated non-scale recipe, combining feature jitter
  (`jitter_sigma=0.10`) with moderate packet perturbation
  (`noise_sigma=0.01`, `packet_loss=0.05`, `stutter_probability=0.08`)
- `drift`: one slow correlated packet-domain drift episode per source file,
  lasting about `20` to `60` seconds without breaking transport continuity
- `burst-loss`: short packet-drop bursts (`2` to `6` packets) injected at a low
  per-minute rate

The packet path no longer applies gain scaling; instead it injects structured
value noise, random drops, optional drift, and optional burst loss. Inference
stays clean; the exported runtime does not apply augmentation.

**The current promoted compact export uses
`--augment base,drift,burst-loss`.** The stronger structured packet-domain
recipe won the historical legacy-versus-time-aware comparison and passed the
paired plus quiet deployment gates before export.

Current comparison status for the `dense` versus `stream_dense` training
contracts, using the same seed and feature set:

- without augmentation, `dense` remains stronger (`98.1%` blocked OOF F1 versus
  `97.0%`)
- with `base`, the two are effectively tied (`97.4%` versus `97.3%`)
- with `base,drift`, `stream_dense` edges ahead (`97.9%` versus `97.8%`)
- with `base,drift,burst-loss`, `stream_dense` is clearly stronger (`98.0%`
  versus `97.5%`)

That makes `stream_dense + base,drift,burst-loss` the current promoted
runtime-aware baseline when the goal is robustness under structured
packet-domain perturbations. The fixed-seed export run that promoted it was:

```bash
python tools/train_ml_model.py --augment base,drift,burst-loss --seed 1876849819
```

Use these components on normal training runs, seed search, and the
`--cross-environment` / `--cross-chip` diagnostics when you want to measure
whether a specific augmentation mix improves generalization for the set you are
testing:

```bash
python tools/train_ml_model.py --augment
python tools/train_ml_model.py --augment --seed-search-until-improvement 10
python tools/train_ml_model.py --augment drift
python tools/train_ml_model.py --augment base,drift
python tools/train_ml_model.py --augment base,drift,burst-loss
python tools/train_ml_model.py --augment --cross-environment
python tools/train_ml_model.py --augment --cross-chip
```

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
normalized before the same production features are extracted for the neural
detector.

To switch the Python runtime to ML detection:

```python
DETECTION_ALGORITHM = "ml"
```

For algorithm details, see [ALGORITHMS.md](ALGORITHMS.md#ml-detector).
