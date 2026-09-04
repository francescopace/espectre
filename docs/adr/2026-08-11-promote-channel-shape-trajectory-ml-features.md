# ADR: promote channel-shape trajectory ML features

- Status: Accepted
- Date: 2026-08-11
- Updated: 2026-08-26

## Context

Motion recorded in a vacation home crossed repeatable Wi-Fi blind spots. Packet loss and weak-link intervals reduced the apparent motion fraction even while the operator moved continuously. The earlier High Accuracy schema relied more heavily on scalar turbulence, L1, and frequency-coherence features and did not transfer as well across that environment.

A host-side screen found that physical-time channel-shape trajectories recovered more of the difficult motion while keeping quiet replay safe. The first promoted trajectory used a standalone full-band spread tracker beside an eight-subband trajectory. That duplicated profile work and reserved substantially more device state than the subband formulation.

Guarded Kendall lag-excess later added an ordinal view of subband turnover using pairwise-order masks already derived from the trajectory. It supplied information that was not interchangeable with the existing path and spread features. Detailed feature definitions, campaigns, ablations, seeds, metrics, and current artifact metadata belong in `FEATURES.md`, the generated performance report, and the exported weight files.

## Decision

Use the gain-normalized, physical-time eight-subband trajectory as the shared source for High Accuracy channel-shape features. The production feature family includes:

- adjacent-profile subband spread;
- coherent innovation energy;
- excess path; and
- guarded Kendall lag-excess.

The current ordered model schema combines those channel-shape inputs with the promoted turbulence and L1 inputs recorded in `FEATURES.md`. Host, MicroPython, and C++ implementations share the trajectory geometry, pairwise-order masks, feature formulas, and generated-artifact parity gates.

Use physical-time bins, duplicate suppression, missing-bin skipping, and DCT-backed trajectory storage. Compute Kendall signatures when a bin is finalized and reuse the existing trajectory window at inference time.

Remove the standalone full-band `chan_shape_spread` tracker and its production feature ID. Historical comparisons remain in `FEATURES.md`; retired implementations do not stay in the executable runtime or host candidate registry solely for rollback convenience. Lightweight does not activate the High Accuracy trajectory tracker.

## Decision History

| Date | Direction | Resolution |
| --- | --- | --- |
| 2026-06-29 to 2026-08-07 | Evolve raw, relative, coherence, invariant, and aggregated-IQR ML baselines | Preserved as feature evidence; none remains the current complete schema |
| 2026-08-11 | Promote physical-time channel-shape trajectory features | Accepted after environment-transfer and deployment replay gates |
| 2026-08-12 | Replace standalone full-band spread with subband spread from the shared trajectory | Accepted to remove duplicate state and packet work |
| 2026-08-16 | Add guarded Kendall lag-excess as an eighth input | Accepted after sealed quiet, occupancy, parity, and promotion gates |

## Alternatives Considered

### Retain the standalone full-band spread tracker

Rejected. The subband form captures the promoted participation signal from state already required by the trajectory features.

### Remove excess path after its low marginal importance

Rejected. Fresh ablations could not preserve the quiet and cross-environment gates without it.

### Replace excess path with Kendall

Rejected. Substitution failed sealed quiet and stationary-domain gates. Kendall is complementary rather than a safe replacement.

### Keep exact rank-gap or Spearman features in production

Rejected. Their runtime cost and missing shared implementation were not justified by promotion evidence. They remain host-side research directions.

## Consequences

- High Accuracy obtains its channel-shape inputs from one DCT-backed physical-time tracker;
- production runtimes no longer allocate or update the full-band lag-profile history;
- Kendall adds bounded mask state and inference work without another tracker family;
- the MLP artifact changes when the ordered schema changes, so generated C++/Python parity remains mandatory; and
- new feature proposals stay host-only until they pass data-role, replay, resource, generated-artifact, and cross-runtime gates.

## Related

- [`../FEATURES.md`](../FEATURES.md)
- [`../ALGORITHMS.md`](../ALGORITHMS.md)
- [`../ML_TRAINING.md`](../ML_TRAINING.md)
- [`2026-06-30-separate-ml-training-data-from-promotion-replays.md`](2026-06-30-separate-ml-training-data-from-promotion-replays.md)
- [`2026-03-08-use-host-side-validation-gates-for-detector-promotion.md`](2026-03-08-use-host-side-validation-gates-for-detector-promotion.md)
- [`2026-08-15-use-fixed-temporal-csi-admission.md`](2026-08-15-use-fixed-temporal-csi-admission.md)
