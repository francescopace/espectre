# ADR guide

Architecture Decision Records for durable project and architecture decisions. Each record describes the current decision; earlier choices in the same line of work live in its decision history instead of separate files.

## What belongs here

Write an ADR for a choice that shaped the project: a production architecture change, a runtime or protocol direction, a detector or ML baseline promotion, a long-lived workflow or validation policy, or a rejected direction that would be costly to try again.

Not for routine cleanup, temporary experiments or sweeps, generated benchmark results, or exact seeds, weights, and other values owned by generated artifacts or current docs. Feature evidence goes in `FEATURES.md`, external research in `LITERATURE.md`, release history in the versioned changelogs, and deployed behavior in `ALGORITHMS.md`, `ARCHITECTURE.md`, and `ML_TRAINING.md`.

## Current decisions

### Architecture and product surfaces

- [`2025-12-06-adopt-esphome-as-the-primary-home-assistant-integration.md`](2025-12-06-adopt-esphome-as-the-primary-home-assistant-integration.md)
- [`2026-06-03-adopt-the-core-runtime-frontend-firmware-split.md`](2026-06-03-adopt-the-core-runtime-frontend-firmware-split.md)
- [`2026-07-02-use-one-resource-oriented-protocol-across-transports.md`](2026-07-02-use-one-resource-oriented-protocol-across-transports.md)
- [`2026-07-15-define-the-detection-profiles.md`](2026-07-15-define-the-detection-profiles.md)
- [`2026-07-15-persist-per-device-matter-onboarding-data.md`](2026-07-15-persist-per-device-matter-onboarding-data.md)
- [`2026-08-17-adopt-improv-serial-and-direct-http-for-local-control.md`](2026-08-17-adopt-improv-serial-and-direct-http-for-local-control.md)
- [`2026-09-16-consume-sdk-through-public-facades.md`](2026-09-16-consume-sdk-through-public-facades.md)

### Sensing and detection

- [`2026-07-04-keep-agc-active-and-standardize-cv-normalization.md`](2026-07-04-keep-agc-active-and-standardize-cv-normalization.md)
- [`2026-07-08-define-the-lightweight-detector.md`](2026-07-08-define-the-lightweight-detector.md)
- [`2026-07-23-adopt-classifier-first-ht20-sensing-contract.md`](2026-07-23-adopt-classifier-first-ht20-sensing-contract.md)
- [`2026-07-25-select-the-classic-band-from-channel-coherence.md`](2026-07-25-select-the-classic-band-from-channel-coherence.md)
- [`2026-08-15-use-fixed-temporal-csi-admission.md`](2026-08-15-use-fixed-temporal-csi-admission.md)
- [`2026-08-23-standardize-managed-csi-traffic-sources.md`](2026-08-23-standardize-managed-csi-traffic-sources.md)

### ML and validation

- [`2025-11-28-prototype-in-python-before-porting-to-production-firmware.md`](2025-11-28-prototype-in-python-before-porting-to-production-firmware.md)
- [`2026-02-15-adopt-an-exportable-mlp-runtime-for-on-device-ml.md`](2026-02-15-adopt-an-exportable-mlp-runtime-for-on-device-ml.md)
- [`2026-03-08-use-host-side-validation-gates-for-detector-promotion.md`](2026-03-08-use-host-side-validation-gates-for-detector-promotion.md)
- [`2026-06-30-separate-ml-training-data-from-promotion-replays.md`](2026-06-30-separate-ml-training-data-from-promotion-replays.md)
- [`2026-08-11-promote-channel-shape-trajectory-ml-features.md`](2026-08-11-promote-channel-shape-trajectory-ml-features.md)

### Data and delivery

- [`2026-07-03-unify-raw-csi-collection-over-http.md`](2026-07-03-unify-raw-csi-collection-over-http.md)
- [`2026-07-18-remove-qemu-smoke-tests-from-firmware-ci.md`](2026-07-18-remove-qemu-smoke-tests-from-firmware-ci.md)
- [`2026-07-29-make-dataset-quality-review-detector-agnostic.md`](2026-07-29-make-dataset-quality-review-detector-agnostic.md)
- [`2026-09-16-sign-published-firmware-and-verify-browser-downloads.md`](2026-09-16-sign-published-firmware-and-verify-browser-downloads.md)

## Files and metadata

- Name files `YYYY-MM-DD-slug.md`, dated by the first decision in the line of work. One topic per file.
- `Status` is `Accepted`, `Superseded`, or `Superseded in part`. A partly superseded ADR opens with one sentence on what still holds and a link to the replacement.
- Add `Recorded: YYYY-MM-DD (retrospective)` when the record was written later, and `Updated: YYYY-MM-DD` when the decision changed.
- Use `Supersedes` only when both records stay useful. Superseded records leave the index above.

## Template

```md
# ADR: short title

- Status: Accepted
- Date: YYYY-MM-DD
- Updated: YYYY-MM-DD

## Context

What problem forced a decision.

## Decision

What we do now.

## Decision history

| Date | Direction | Resolution |
| --- | --- | --- |
| YYYY-MM-DD | Earlier choice | What replaced it and why |

## Evidence

Only the measurements a reader needs to trust the decision.

## Alternatives considered

- **Option.** Why not.

## Consequences

What gets easier, and what it costs.

## Related

- current docs and other ADRs
- Commits `abcdef12` (what it did)
```

Skip `Decision history` when there was no earlier direction, and `Evidence` when the context already carries the reasoning.

## Writing style

- Short sentences and plain words. Say what was decided and why, then stop.
- Keep the decision readable without the evidence. Put numbers in `Evidence` or the decision history, and only the ones that justify the choice.
- Leave out per-run details, checksums, and image hashes. Do not point to local or untracked folders: readers of the repository cannot open them, so bring any detail the decision needs into the ADR itself.
- Do not repeat mutable values (coefficients, thresholds, column definitions) that another document or a generated artifact owns; link to it.
- Cite commits by short hash, and check that they exist before adding them.

## Updating an ADR

When the decision changes within the same topic, update `Decision`, add a row to the decision history, move feature measurements and rejected experiments to `FEATURES.md`, and fix current docs and inbound links. Do not open a new ADR just for a seed, coefficient, cleanup, or intermediate baseline.

Write a new ADR when the new decision stands on its own, concerns a different architectural question, or replaces a published decision that must stay addressable.

## Quality bar

Keep or create an ADR only if at least one of these holds, ideally more:

- it changed project direction across releases;
- later code or docs are hard to understand without the reasoning;
- it touches both Python and C++, several frontends, or a published compatibility surface;
- the rejected alternative is likely to come back and was expensive enough to record.
