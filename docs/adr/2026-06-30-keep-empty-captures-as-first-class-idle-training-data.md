# ADR: keep empty captures as first-class idle training data

- Status: Accepted
- Date: 2026-06-30
- Recorded: 2026-07-09 (retrospective)

## Context

The C3 empty-room retrain incident showed that the deployed task was broader than `static_presence` versus `motion`. A runtime that looks correct on paired static-presence data can still fail in a truly empty room if the training and validation sets do not treat that domain as part of IDLE.

The incident established the core lesson twice:

- a new C3 `empty` capture reproduced a runtime false-positive problem that the corresponding `static_presence` capture did not
- a later C6 `empty` capture repeated the same pattern, confirming that this was not a one-off quirk of one device or room

Retraining with `empty` mapped into the IDLE class materially reduced those false positives, even though broader domain-coverage problems still had to be managed through holdouts and long-recording gates.

## Decision

Treat `empty` captures as first-class IDLE training and validation data for the ML detector.

Concretely:

- map both `empty` and `static_presence` to IDLE
- keep empty-room captures in regression gates and dataset curation
- interpret new empty-domain failures as dataset-coverage problems first, not as evidence that the deployed task excludes empty rooms

## Alternatives Considered

### Train only on `static_presence` versus `motion`

Rejected. That target does not match the deployed problem, which must suppress both empty-room and static-presence false positives.

### Keep `empty` only as an optional smoke-test domain

Rejected. The incidents showed that empty-room behavior is not an edge case; it is part of the normal IDLE distribution the deployed detector must handle.

## Consequences

Benefits:

- the ML target now better matches the deployed sensing task
- dataset collection and validation become more explicit about IDLE coverage
- later baseline promotions inherit a clearer definition of what counts as quiet

Trade-offs:

- data curation must keep empty-room captures in scope
- stronger empty-domain coverage can reveal additional dataset weaknesses that simple paired validation would otherwise miss

## Related

- `docs/ML_DATA_COLLECTION.md`
- `docs/adr/2026-03-08-use-host-side-validation-gates-for-detector-promotion.md`
- [`2026-08-11-promote-channel-shape-trajectory-ml-features.md`](2026-08-11-promote-channel-shape-trajectory-ml-features.md)
