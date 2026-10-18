# ADR: promote channel-shape trajectory ML features

- Status: Accepted
- Date: 2026-08-11
- Updated: 2026-08-26

## Context

Motion recorded in a vacation home kept crossing the same Wi-Fi blind spots. Packet loss and weak-link stretches made the motion fraction look lower even while the person moved the whole time. The High Accuracy schema of the time leaned on scalar turbulence, L1, and frequency-coherence features, and did not transfer well to that home.

A host-side screen showed that channel-shape trajectories over physical time recovered more of that motion while quiet replay stayed safe. The first version used a separate full-band spread tracker next to an eight-subband trajectory, which duplicated profile work and needed much more device state than the subband form alone. Later, a guarded Kendall lag-excess added an ordinal view of subband turnover from masks the trajectory already computes, and it carried information the path and spread features did not.

Feature definitions, campaigns, ablations, seeds, and metrics live in `FEATURES.md`, the performance report, and the exported weight files.

## Decision

High Accuracy takes its channel-shape features from one gain-normalized, physical-time, eight-subband trajectory:

- adjacent-profile subband spread;
- coherent innovation energy;
- excess path;
- guarded Kendall lag-excess.

These sit next to the turbulence and L1 inputs listed in `FEATURES.md`. Host, MicroPython, and C++ share the trajectory geometry, order masks, formulas, and parity gates.

The trajectory uses physical-time bins, drops duplicates, skips missing bins, and is stored as DCT coefficients. Kendall signatures are computed when a bin closes and reuse the existing window at inference.

The full-band `chan_shape_spread` tracker and its feature ID are removed. Old comparisons stay in `FEATURES.md`; retired code is not kept around for rollback. Lightweight does not run this tracker.

## Decision history

| Date | Direction | Resolution |
| --- | --- | --- |
| 2026-06-29 to 2026-08-07 | Raw, relative, coherence, invariant, and aggregated-IQR ML baselines | Kept as feature evidence; none is the current schema |
| 2026-08-11 | Physical-time channel-shape trajectory features | Accepted after environment-transfer and deployment replay gates |
| 2026-08-12 | Subband spread from the shared trajectory replaces full-band spread | Accepted; removes duplicate state and per-packet work |
| 2026-08-16 | Guarded Kendall lag-excess as an eighth input | Accepted after sealed quiet, occupancy, parity, and promotion gates |

## Alternatives considered

- **Keep the full-band spread tracker.** The subband form gives the same signal from state the trajectory already needs.
- **Drop excess path for its low marginal importance.** Without it, fresh ablations failed the quiet and cross-environment gates.
- **Replace excess path with Kendall.** Failed the sealed quiet and stationary gates. Kendall adds to it, it does not replace it.
- **Exact rank-gap or Spearman features.** Too costly at runtime for what they gained, and no shared implementation. They stay host-side research.

## Consequences

High Accuracy gets all its channel-shape inputs from one tracker, and the runtimes no longer keep a full-band lag-profile history. Kendall adds a small bounded mask state and some inference work, but no new tracker. Any change to the ordered schema changes the MLP artifact, so C++/Python parity remains mandatory. New features stay host-only until they pass data-role, replay, resource, artifact, and cross-runtime gates.

## Related

- [Feature ledger](../FEATURES.md), [algorithms](../ALGORITHMS.md), and [ML training](../ML_TRAINING.md)
- [`2026-06-30-separate-ml-training-data-from-promotion-replays.md`](2026-06-30-separate-ml-training-data-from-promotion-replays.md)
- [`2026-03-08-use-host-side-validation-gates-for-detector-promotion.md`](2026-03-08-use-host-side-validation-gates-for-detector-promotion.md)
- [`2026-08-15-use-fixed-temporal-csi-admission.md`](2026-08-15-use-fixed-temporal-csi-admission.md)
- Commits `08e45e70` (trajectory features), `1edad9f2` (subband spread), `1a3147c2` (Kendall lag-excess)
