# ADR Guide

This directory contains Architecture Decision Records (ADRs) for durable project-level and architectural decisions.

The catalog is intentionally organized around current decisions. Earlier choices that belong to the same decision lineage are summarized inside the current ADR instead of remaining as separate navigation entries.

## What Belongs Here

An ADR belongs here when it captures a durable choice that shaped the project, for example:

- a production architecture change
- a runtime or protocol direction
- a detector or ML baseline promotion
- a long-lived workflow or validation policy
- a rejected direction whose rationale prevents costly repetition

An ADR should not be used for:

- routine implementation cleanup
- temporary experiments or parameter sweeps
- generated benchmark results
- exact model seeds, weights, or other operational values already owned by generated artifacts or current documentation

Use `FEATURES.md` for detector-feature evidence and baseline lineage, `LITERATURE.md` for external research, versioned changelog snapshots for release history, and current documents such as `ALGORITHMS.md`, `ARCHITECTURE.md`, and `ML_TRAINING.md` for the deployed behavior.

## Current Decision Index

### Architecture And Product Surfaces

- [`2025-12-06-adopt-esphome-as-the-production-integration-surface.md`](2025-12-06-adopt-esphome-as-the-production-integration-surface.md)
- [`2026-06-03-adopt-the-core-runtime-frontend-firmware-split.md`](2026-06-03-adopt-the-core-runtime-frontend-firmware-split.md)
- [`2026-07-02-use-a-shared-espectre-protocol-across-esp-idf-frontends.md`](2026-07-02-use-a-shared-espectre-protocol-across-esp-idf-frontends.md)
- [`2026-07-03-adopt-a-dedicated-cpp-streamer-frontend-for-high-rate-csi-collection.md`](2026-07-03-adopt-a-dedicated-cpp-streamer-frontend-for-high-rate-csi-collection.md)
- [`2026-07-15-adopt-persisted-runtime-detector-selection.md`](2026-07-15-adopt-persisted-runtime-detector-selection.md)
- [`2026-07-15-persist-per-device-matter-onboarding-data.md`](2026-07-15-persist-per-device-matter-onboarding-data.md)
- [`2026-08-13-adopt-goal-oriented-detector-profile-names.md`](2026-08-13-adopt-goal-oriented-detector-profile-names.md)

### Sensing And Detection

- [`2026-07-04-keep-agc-active-and-standardize-cv-normalization.md`](2026-07-04-keep-agc-active-and-standardize-cv-normalization.md)
- [`2026-07-08-promote-classic-detector-and-retire-legacy-baselines.md`](2026-07-08-promote-classic-detector-and-retire-legacy-baselines.md)
- [`2026-07-23-adopt-classifier-first-ht20-sensing-contract.md`](2026-07-23-adopt-classifier-first-ht20-sensing-contract.md)
- [`2026-07-25-select-the-classic-band-from-channel-coherence.md`](2026-07-25-select-the-classic-band-from-channel-coherence.md)
- [`2026-07-26-recover-the-startup-threshold-once-a-session-settles.md`](2026-07-26-recover-the-startup-threshold-once-a-session-settles.md)
- [`2026-08-10-configure-detector-windows-in-milliseconds.md`](2026-08-10-configure-detector-windows-in-milliseconds.md)
- [`2026-08-13-use-aggregated-turbulence-iqr-for-classic.md`](2026-08-13-use-aggregated-turbulence-iqr-for-classic.md)

### ML And Validation

- [`2025-11-28-prototype-in-python-before-porting-to-production-firmware.md`](2025-11-28-prototype-in-python-before-porting-to-production-firmware.md)
- [`2026-02-15-adopt-an-exportable-mlp-runtime-for-on-device-ml.md`](2026-02-15-adopt-an-exportable-mlp-runtime-for-on-device-ml.md)
- [`2026-03-08-use-host-side-validation-gates-for-detector-promotion.md`](2026-03-08-use-host-side-validation-gates-for-detector-promotion.md)
- [`2026-06-30-keep-empty-captures-as-first-class-idle-training-data.md`](2026-06-30-keep-empty-captures-as-first-class-idle-training-data.md)
- [`2026-07-02-use-pytorch-as-the-host-training-stack.md`](2026-07-02-use-pytorch-as-the-host-training-stack.md)
- [`2026-07-23-separate-ml-training-data-from-promotion-replays.md`](2026-07-23-separate-ml-training-data-from-promotion-replays.md)
- [`2026-08-11-promote-channel-shape-trajectory-ml-features.md`](2026-08-11-promote-channel-shape-trajectory-ml-features.md)

### Data And Delivery

- [`2026-07-18-remove-qemu-smoke-tests-from-firmware-ci.md`](2026-07-18-remove-qemu-smoke-tests-from-firmware-ci.md)
- [`2026-07-19-preserve-per-record-phy-provenance-in-streamer-datasets.md`](2026-07-19-preserve-per-record-phy-provenance-in-streamer-datasets.md)
- [`2026-07-29-make-dataset-quality-review-detector-agnostic.md`](2026-07-29-make-dataset-quality-review-detector-agnostic.md)

## File And Metadata Rules

- Use `YYYY-MM-DD-slug.md`, ordered by the date of the first durable decision in the lineage.
- Keep one coherent decision topic per file.
- Use `Status: Accepted`, `Superseded`, or `Superseded in part`.
- Add `Recorded: YYYY-MM-DD (retrospective)` when the record was reconstructed later.
- Use `Supersedes` metadata only when both independently useful records remain in the repository.
- Keep superseded records out of the current decision index.

## Recommended Structure

```md
# ADR: short title

- Status: Accepted
- Date: YYYY-MM-DD
- Recorded: YYYY-MM-DD (retrospective)
- Updated: YYYY-MM-DD

## Context

Why the decision was needed.

## Decision

What is currently accepted.

## Decision History

| Date | Direction | Resolution |
| --- | --- | --- |
| YYYY-MM-DD | Earlier choice | Why it changed |

## Alternatives Considered

### Option name

Rejected. Why it was not chosen.

## Consequences

Benefits and trade-offs.

## Related

- current documentation
- feature or literature evidence
- versioned changelog snapshots
- relevant commits
```

Include `Decision History` only when the topic has a meaningful earlier direction. Keep the table concise; detailed experimental results belong in their owning ledger or generated report.

## Updating ADRs

When a choice evolves within the same coherent topic:

- update the current decision;
- retain the previous direction in `Decision History`;
- move feature measurements and rejected experiments to `FEATURES.md`;
- update current documentation and inbound links; and
- avoid creating a new ADR solely for a model seed, coefficient change, cleanup, or intermediate baseline.

Create a new ADR when the new decision is independently useful, changes a different architectural concern, or supersedes a decision that was already published and must remain addressable as a stable historical record.

## Quality Bar

Create or retain an ADR only when at least one of these is clearly true, and preferably more than one:

- the decision changed project direction across releases;
- later code or documentation is difficult to understand without the rationale;
- it affects both Python and C++, multiple frontends, or a published compatibility surface;
- the rejected alternative is likely to recur and was expensive enough to justify a durable record.
