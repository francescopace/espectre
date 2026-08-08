# Timing-Aware Training Review — Contract, Provenance, and Cache Reuse

Date: 2026-07-30
Branch: `v3.0`
Scope: the ML training, replay, and cache surfaces that would be affected by moving from the current dense training matrix toward a timing-aware sample contract aligned with deployment-time replay semantics. Primary files: `tools/train_ml_model.py`, `tools/lib/performance_report.py`, `src/python/micro_espectre/runtime_policy.py`, `tools/lib/npz_cache.py`, and `tools/validate_dataset_quality.py`.

This review does not argue that timing must become a model input. It reviews whether timing-aware training is a worthwhile contract change, what cache architecture that would imply, and which risks have to be controlled before such a change could be shipped.

Review status: **Complete.** The tracker below reflects current implementation progress. Remaining unchecked items are still open design and validation work.

---

## 1. How To Read This Document

Findings carry a stable id so they can be referenced from commits, issues, and future ADRs:

| Prefix | Theme |
| --- | --- |
| `T-n` | Training contract and sample semantics |
| `P-n` | Timing provenance and data-quality controls |
| `C-n` | Cache architecture and reuse |
| `V-n` | Validation and rollout discipline |

Severity uses three levels:

- **High**: changing or failing to define the contract would invalidate the meaning of training/evaluation comparisons or produce the wrong reusable cache layer.
- **Medium**: the current design remains correct, but it duplicates expensive work, hides quality signals, or invites the wrong next optimization.
- **Low**: cleanup, terminology, or workflow clarification.

Section 2 is the single source of truth for progress. Detailed findings do not repeat their status.

---

## 2. Progress Tracker

### Batch 1 — Define the training contract before optimizing around it

- [x] **T-1** (High) — Decide whether the ML sample unit remains the dense sliding window or becomes the time-aware evaluation tick → [§3](#t-1)
- [x] **T-2** (High) — Keep timing as provenance/control first, not as a model side-input by default → [§3](#t-2)

### Batch 2 — Use timing to control data quality before changing the model

- [x] **P-1** (Medium) — Add timing-quality stratification to the baseline reporting surface → [§4](#p-1)
- [x] **P-2** (Medium) — Test exclusion/downweighting of poor-timing windows or files before any feature-surface change → [§4](#p-2)

### Batch 3 — Unify cache layers only after the sample contract is stable

- [x] **C-1** (High) — Introduce one canonical time-aware ML replay-row cache instead of optimizing each consumer independently → [§5](#c-1)
- [x] **C-2** (Medium) — Keep augmentation, Classic, and aggregate report summaries as derived layers rather than forcing one literal cache for all consumers → [§5](#c-2)

### Batch 4 — Treat rollout as a behavior change, not a local speed tweak

- [x] **V-1** (High) — Prove parity and deployment safety across trainer gates, performance reporting, and runtime replay before switching defaults → [§6](#v-1)

---

## 3. Training Contract

<a id="t-1"></a>

### T-1 — Dense training windows and time-aware deployment ticks are different sample contracts

**Severity**: High

**Where**

- [train_ml_model.py](../../tools/train_ml_model.py) — `extract_features`, `load_training_matrix`
- [performance_report.py](../../tools/lib/performance_report.py) — `compute_ml_packet_result`, `replay_idle_stream`, `evaluate_ml_long_recording`
- [runtime_policy.py](../../src/python/micro_espectre/runtime_policy.py) — `PacketTimingTracker`, `RuntimeMotionPolicy`

**Finding**

Current ML training builds a dense per-window matrix from CSI-derived features. Deployment-style replay does not score every dense window. It evaluates only on time-aware ticks, and can reset cadence or detector state when the packet stream is contaminated or stalls.

That means the current train/eval split is only partially aligned:

- the feature extractor is intentionally shared,
- but the *sample unit* is not,
- and the deploy-time replay contract is stricter than the training matrix.

**Why it matters**

If timing quality genuinely explains replay instability, then a dense training matrix can reward windows the runtime would never score or would explicitly reset away. Optimizing cache reuse without deciding which contract is the source of truth risks speeding up the wrong representation.

**How to resolve**

Make an explicit contract decision:

1. either keep dense-window training as a deliberate approximation, and treat time-aware replay as evaluation-only, or
2. promote the time-aware evaluation tick to the canonical ML sample unit and let training, report generation, and test replays consume that same sample representation.

Do not merge cache layers before that decision is written down.

**Implementation follow-up**

The repository now keeps one host-side ML training contract: `stream_dense`. The fully sparse `replay_tick` training mode proved too sample-starved on the current corpus and was retired as a public training option, while `stream_dense` keeps the runtime streaming feature path and reset semantics without collapsing to one supervised sample every evaluation tick. On the current corpus, `stream_dense` matches `dense` under `base`, slightly beats it under `base,drift`, clearly beats it under `base,drift,burst-loss`, passes the paired plus quiet deployment gates, and has now been exported as the new baseline with a fixed-seed `stream_dense + base,drift,burst-loss` run. The legacy `dense` fallback and its `feature_column` cache were subsequently removed so every active calculation preserves stream timing and reset semantics.

---

<a id="t-2"></a>

### T-2 — Timing should start as provenance and control, not a default model feature

**Severity**: High

**Where**

- [train_ml_model.py](../../tools/train_ml_model.py) — `_load_training_file_records`
- [validate_dataset_quality.py](../../tools/validate_dataset_quality.py) — `validate_capture_continuity`

**Finding**

Timing provenance already exists in the corpus, but the current trainer uses it only for optional file-level sync-metadata filtering. The stronger hypothesis is not yet "timing is a motion feature"; it is "timing quality marks which samples should be trusted, weighted, or suppressed."

**Why it matters**

Treating timing as an input too early risks acquisition-source leakage. Treating it first as provenance control gives a safer path: the model stays on the current physical feature surface while bad timing slices stop contaminating the fit.

**How to resolve**

Evaluate timing in this order:

1. reporting/stratification,
2. exclusion or downweighting,
3. replay/runtime gating,
4. only then side-input experiments.

If time-aware training is pursued, start with sample selection/weighting before adding new model inputs.

---

## 4. Timing Provenance Controls

<a id="p-1"></a>

### P-1 — Timing-quality evidence is not yet surfaced where model decisions are discussed

**Severity**: Medium

**Where**

- [performance_report.py](../../tools/lib/performance_report.py)
- [FEATURES.md](../../docs/FEATURES.md)
- [ALGORITHMS.md](../../docs/ALGORITHMS.md)

**Finding**

The repository can already detect cadence quality and contamination during replay, but the baseline ML reporting does not yet join that evidence to the same paired, weak-link, quiet, cross-chip, and cross-environment slices used to discuss detector quality.

**Why it matters**

Without stratification, a future time-aware training change risks being driven by intuition rather than evidence. The first question is whether bad replay behavior clusters around timing quality at all.

**How to resolve**

Add timing-quality audit outputs to the current no-phase compact baseline before changing training behavior. Make the evidence visible in the same places where replay quality is already compared.

---

<a id="p-2"></a>

### P-2 — Conservative provenance controls are a cheaper test than a full contract rewrite

**Severity**: Medium

**Where**

- [train_ml_model.py](../../tools/train_ml_model.py)
- [validate_dataset_quality.py](../../tools/validate_dataset_quality.py)

**Finding**

If timing quality mostly identifies pathological windows or recordings, then a full time-aware training matrix may be unnecessary. Excluding or downweighting those spans could capture most of the benefit while leaving the current feature surface and optimizer unchanged.

**Why it matters**

This is the lowest-risk experiment that still tests the core hypothesis: "runtime-like timing quality helps because it removes misleading training examples."

**How to resolve**

Prototype timing-aware filtering/downweighting before replacing the dense matrix contract.

---

## 5. Cache Architecture

<a id="c-1"></a>

### C-1 — The cache split currently mirrors tooling boundaries instead of one canonical ML replay layer

**Severity**: High

**Where**

- [npz_cache.py](../../tools/lib/npz_cache.py)
- [train_ml_model.py](../../tools/train_ml_model.py)
- [performance_report.py](../../tools/lib/performance_report.py)

**Finding**

Today the repository has:

- per-feature training columns,
- packet-level replay for several validation paths,
- detector-replay aggregate metric caches,
- and a newly optimized trainer gate cache path.

These are good local optimizations, but they are not yet one coherent ML replay representation.

**Why it matters**

If training becomes time-aware, the most valuable reusable artifact is no longer "feature column per dense window" or "aggregate metric per replay." It is a canonical **time-aware ML replay row**: one cadence-aligned feature row plus the metadata needed to preserve resets, stream boundaries, and evaluation semantics.

That artifact could then feed:

- training,
- trainer gates,
- performance reporting,
- and ML replay tests.

**How to resolve**

Define a shared time-aware ML replay-row cache below all four consumers. Build derived caches above it only where the consumer genuinely needs something else.

**Resolution**

Resolved in the staged implementation. One canonical dense artifact now stores all runtime ML features, reset identity, packet identity, and the deployment evaluation-tick marker. `stream_dense` training and dataset-quality validation consume the dense projection; trainer gates, performance generation, and ML tests consume the runtime-tick projection. Both projections address the same cache key.

---

<a id="c-2"></a>

### C-2 — One shared layer does not mean one literal cache artifact for everything

**Severity**: Medium

**Where**

- [npz_cache.py](../../tools/lib/npz_cache.py)
- [performance_report.py](../../tools/lib/performance_report.py)
- [train_ml_model.py](../../tools/train_ml_model.py)

**Finding**

Even if the ML sample contract is unified, not every consumer should read the exact same artifact:

- augmentation still needs a derived variant,
- Classic has a different detector contract,
- aggregate report summaries and parity payloads remain downstream products.

**Why it matters**

Trying to force a single literal artifact for all tooling would couple unrelated consumers and make invalidation unsafe. The right target is one shared *canonical layer*, not one monolithic cache blob.

**How to resolve**

Keep the architecture layered:

1. canonical time-aware ML replay rows,
2. optional augmented or summarized derivatives,
3. consumer-specific final products.

**Resolution**

Resolved. The incompatible legacy `dense` contract and its per-feature cache were removed. Classic detector results and dataset-quality summaries remain downstream artifacts. ML aggregate detector-result caching was removed; ML consumers recompute inexpensive inference from the shared time-aware rows.

---

## 6. Validation And Rollout

<a id="v-1"></a>

### V-1 — A unified time-aware training pipeline is a behavior change and must be validated as such

**Severity**: High

**Where**

- [train_ml_model.py](../../tools/train_ml_model.py)
- [performance_report.py](../../tools/lib/performance_report.py)
- [test_validation_real_data.py](../../test/python/test_validation_real_data.py)
- [cpp_parity.py](../../tools/lib/cpp_parity.py)

**Finding**

Moving training onto a time-aware sample contract would not be a local performance refactor. It would change:

- sample count,
- class balance,
- which windows are admissible,
- and how directly the training objective matches deployment replay.

**Why it matters**

A change this deep cannot be justified by blocked OOF F1 alone. It must be measured on the deployment-facing surfaces that motivated it: paired replay safety, quiet false positives, weak-link behavior, cross-chip/cross-environment stability, and the C++/Python parity gates.

**How to resolve**

Treat any move to time-aware training as its own measured rollout:

- baseline report-only audit,
- provenance-control experiments,
- cadence-aligned training prototype,
- gate and parity validation,
- only then any default flip.

**Resolution**

Resolved for the ML rollout: `stream_dense + base,drift,burst-loss` passed the paired, quiet, exported-inference, and Python/C++ ML parity surfaces before the default changed. Repository-wide green status is a separate concern: the time-aware Classic replay now exposes existing per-session guardrail failures, which must be fixed in the Classic behavior or explicitly re-decided rather than attributed to the ML model.

---

## 7. Overall Recommendation

The implementation followed the recommended sequence:

1. treat timing as provenance first,
2. select `stream_dense` as the sole training sample contract,
3. introduce one shared time-aware replay-row cache below training, quality validation, reporting, and tests,
4. keep timing out of the model input surface.

That order preserves the current feature surface while testing the part of the hypothesis most likely to help deployment behavior.
