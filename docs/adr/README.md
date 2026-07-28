# ADR Guide

This directory contains Architecture Decision Records (ADRs) for durable
project-level and architectural decisions.

The goal is to preserve decision history in a form that stays understandable
over time, even when the rest of the documentation evolves.

## What Belongs Here

An ADR belongs here when it captures a durable choice that shaped the project,
for example:

- a production architecture change
- a runtime or protocol direction
- a detector or ML baseline promotion
- a long-lived workflow or validation policy
- a historically important decision that was later superseded

An ADR should not be used for:

- routine implementation churn
- temporary experiments
- benchmark sweeps without a durable project decision
- details that only belong in current operational documentation

## Relationship With Other Docs

Use each document for its intended role:

- `docs/adr/`: stable decisions and decision history
- versioned changelog snapshots: historical release framing
- `FEATURES.md`: cross-baseline ML feature inventory, retained measurements,
  current verdicts, and research backlog
- `LITERATURE.md`: mutable paper digest, reported methods and results, and
  ESPectre transferability notes
- current docs such as `ALGORITHMS.md`, `ARCHITECTURE.md`, or `ML_TRAINING.md`:
  current-state explanations, not stable historical references

Important rule: in ADR `Related` sections, prefer links to sibling ADRs,
versioned changelog snapshots, and commit hashes. Avoid generic links to
mutable docs that may describe something different in future releases.
`FEATURES.md` and `LITERATURE.md` are deliberate exceptions for ML feature
evidence and reviewed research: the ADR retains the decision-time rationale,
while the catalogs compare that snapshot with earlier and later feature work
and source evidence.

## Topic Index

Use this index when a current document should point to a durable decision
instead of repeating its historical rationale.

### Architecture

- [`2025-12-06-adopt-a-dual-platform-development-model.md`](2025-12-06-adopt-a-dual-platform-development-model.md)
- [`2025-12-06-adopt-esphome-as-the-production-integration-surface.md`](2025-12-06-adopt-esphome-as-the-production-integration-surface.md)
- [`2026-06-03-adopt-the-core-runtime-frontend-firmware-split.md`](2026-06-03-adopt-the-core-runtime-frontend-firmware-split.md)
- [`2026-07-03-adopt-a-dedicated-cpp-streamer-frontend-for-high-rate-csi-collection.md`](2026-07-03-adopt-a-dedicated-cpp-streamer-frontend-for-high-rate-csi-collection.md)
- [`2026-07-15-adopt-persisted-runtime-detector-selection.md`](2026-07-15-adopt-persisted-runtime-detector-selection.md)

### Detection

- [`2025-11-16-adopt-segmentation-first-mvs-architecture.md`](2025-11-16-adopt-segmentation-first-mvs-architecture.md)
- [`2025-12-03-adopt-nbvi-for-runtime-subcarrier-selection.md`](2025-12-03-adopt-nbvi-for-runtime-subcarrier-selection.md)
- [`2026-06-09-replace-runtime-nbvi-with-fixed-shared-subcarriers.md`](2026-06-09-replace-runtime-nbvi-with-fixed-shared-subcarriers.md)
- [`2026-07-07-adopt-gated-startup-threshold-calibration-for-classic-detector.md`](2026-07-07-adopt-gated-startup-threshold-calibration-for-classic-detector.md)
- [`2026-07-08-promote-classic-detector-and-retire-legacy-baselines.md`](2026-07-08-promote-classic-detector-and-retire-legacy-baselines.md)
- [`2026-07-22-adopt-session-centered-l1-excursion-for-low-rssi.md`](2026-07-22-adopt-session-centered-l1-excursion-for-low-rssi.md)
- [`2026-07-23-adopt-classifier-first-ht20-sensing-contract.md`](2026-07-23-adopt-classifier-first-ht20-sensing-contract.md)

### ML

- [`2025-11-28-prototype-in-python-before-porting-to-production-firmware.md`](2025-11-28-prototype-in-python-before-porting-to-production-firmware.md)
- [`2026-02-15-adopt-an-exportable-mlp-runtime-for-on-device-ml.md`](2026-02-15-adopt-an-exportable-mlp-runtime-for-on-device-ml.md)
- [`2026-02-15-share-ml-model-artifacts-between-python-and-cpp.md`](2026-02-15-share-ml-model-artifacts-between-python-and-cpp.md)
- [`2026-03-08-use-host-side-validation-gates-for-detector-promotion.md`](2026-03-08-use-host-side-validation-gates-for-detector-promotion.md)
- [`2026-05-20-retain-mlp-over-small-temporal-models.md`](2026-05-20-retain-mlp-over-small-temporal-models.md)
- [`2026-06-29-record-historical-ml-baseline-evolution-before-core-6.md`](2026-06-29-record-historical-ml-baseline-evolution-before-core-6.md)
- [`2026-07-02-use-pytorch-as-the-host-training-stack.md`](2026-07-02-use-pytorch-as-the-host-training-stack.md)
- [`2026-07-07-use-core-6-as-the-production-ml-feature-set.md`](2026-07-07-use-core-6-as-the-production-ml-feature-set.md)
- [`2026-07-07-reject-detector-guided-sample-weighting-as-the-default-ml-baseline-policy.md`](2026-07-07-reject-detector-guided-sample-weighting-as-the-default-ml-baseline-policy.md)
- [`2026-07-23-separate-ml-training-data-from-promotion-replays.md`](2026-07-23-separate-ml-training-data-from-promotion-replays.md)
- [`2026-07-23-adopt-coherence-6-as-the-production-ml-feature-set.md`](2026-07-23-adopt-coherence-6-as-the-production-ml-feature-set.md)
- [`2026-07-27-add-the-lag-ratio-to-the-production-ml-feature-set.md`](2026-07-27-add-the-lag-ratio-to-the-production-ml-feature-set.md)
- [`2026-07-27-reduce-the-feature-surface-to-the-production-set.md`](2026-07-27-reduce-the-feature-surface-to-the-production-set.md)
- [`2026-07-28-drop-the-absolute-l1-features.md`](2026-07-28-drop-the-absolute-l1-features.md)

### Protocol And Frontends

- [`2025-11-01-adopt-standalone-esp-idf-mqtt-firmware-as-the-initial-product-shape.md`](2025-11-01-adopt-standalone-esp-idf-mqtt-firmware-as-the-initial-product-shape.md)
- [`2026-07-02-use-a-shared-espectre-protocol-across-esp-idf-frontends.md`](2026-07-02-use-a-shared-espectre-protocol-across-esp-idf-frontends.md)
- [`2026-07-15-persist-per-device-matter-onboarding-data.md`](2026-07-15-persist-per-device-matter-onboarding-data.md)
- [`2026-07-19-preserve-per-record-phy-provenance-in-streamer-datasets.md`](2026-07-19-preserve-per-record-phy-provenance-in-streamer-datasets.md)

### Data Collection And Dataset Contract

- [`2026-06-30-keep-empty-captures-as-first-class-idle-training-data.md`](2026-06-30-keep-empty-captures-as-first-class-idle-training-data.md)
- [`2026-07-04-keep-agc-active-and-standardize-cv-normalization.md`](2026-07-04-keep-agc-active-and-standardize-cv-normalization.md)
- [`2026-07-17-separate-dataset-admission-from-classic-diagnostics.md`](2026-07-17-separate-dataset-admission-from-classic-diagnostics.md)
- [`2026-07-19-preserve-per-record-phy-provenance-in-streamer-datasets.md`](2026-07-19-preserve-per-record-phy-provenance-in-streamer-datasets.md)
- [`2026-07-23-adopt-classifier-first-ht20-sensing-contract.md`](2026-07-23-adopt-classifier-first-ht20-sensing-contract.md)
- [`2026-07-23-separate-ml-training-data-from-promotion-replays.md`](2026-07-23-separate-ml-training-data-from-promotion-replays.md)

## How To Name ADR Files

ADR filenames should be ordered by the decision date, not by the date the ADR
file was written.

Rules:

- use the filename format `YYYY-MM-DD-slug.md`
- keep one decision per file
- use a short descriptive slug in the filename
- if multiple ADRs share the same date, order them by the best available
  historical sequence from release notes and git history, expressed through the
  slug rather than an ordinal
- if chronology is later reconstructed more accurately, prefer keeping the same
  decision date in the filename; only rename when the date itself was wrong or
  the slug no longer reflects the decision

Example:

- `2026-07-07-adopt-gated-startup-threshold-calibration-for-classic-detector.md`

## Retrospective ADRs

Many ADRs in this project are retrospective, reconstructed after the fact from:

1. versioned changelog snapshots
2. git history and commit messages
3. surviving current docs, comments, and test expectations, when useful as
   secondary context
4. earlier ADRs that already preserve superseded decision history

For retrospective ADRs:

- keep the original decision date in `Date:`
- add `Recorded: <date> (retrospective)`
- use explicit `Supersedes:` or `Superseded by:` metadata when applicable

## Recommended File Structure

Use this structure:

```md
# ADR: short title

- Status: Accepted | Superseded | Superseded in part
- Date: YYYY-MM-DD
- Recorded: YYYY-MM-DD (retrospective)
- Supersedes: YYYY-MM-DD-slug.md   # optional
- Superseded by: YYYY-MM-DD-slug.md # optional

## Context

Why this decision was needed.

## Decision

What was chosen.

## Alternatives Considered

### Option name

Rejected. Why it was not chosen.

## Consequences

Benefits:

- ...

Trade-offs:

- ...

## Related

- related ADRs
- versioned changelog snapshots
- git commits
```

## Writing Guidelines

Keep ADRs:

- concise
- technical
- historically grounded
- focused on why the decision mattered

Prefer:

- release-specific historical evidence
- specific sibling ADR links when one decision supersedes, amends, or rejects
  another
- explicit supersession chains

Avoid:

- copying large benchmark tables from source notes or campaign logs
- duplicating full changelog prose
- linking generic mutable docs in `Related`
- mixing multiple unrelated decisions into one ADR

## Quality Bar

Create an ADR only when at least one of these is clearly true, and ideally more
than one:

- the decision changed the project direction for multiple releases
- later code or docs are hard to understand without it
- it affected both Python and C++, or multiple frontends
- it was later superseded and that supersession matters historically

## Updating Existing ADRs

Historical ADRs should be treated as stable records.

Prefer:

- adding a new ADR when a decision changes
- marking an older ADR as superseded
- adding narrowly scoped cross-references

Avoid:

- rewriting old ADRs to match the latest architecture
- replacing historical rationale with current rationale
- removing evidence of past mistakes or superseded directions
