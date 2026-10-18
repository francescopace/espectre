# ADR: keep AGC active and standardize CV normalization

- Status: Accepted
- Date: 2026-07-04
- Recorded: 2026-07-09 (retrospective)

## Context

Detection and ML kept running into gain-dependent behavior. The gain-shift diagnostic made it obvious on the old raw-feature ML baseline: quality looked good at nominal gain, but a uniform artificial gain shift swung the false-positive rate wildly. Scaling features globally is not the same as being gain-invariant.

Later work confirmed this. The ML baseline only became gain-stable after moving to relative statistics (the relative-8 topology with an `fp_weight` sweep). Detector-guided sample weighting helped some long runs, but it did not replace a clean, gain-robust signal path.

Locking the gain had its own costs: instability, and Wi-Fi RX/TX problems that could add packet loss.

## Decision

No hardware gain lock in production. AGC stays on for every chip. Coefficient-of-variation turbulence (`std / mean`) is the one normalization used by runtime detection, collection and datasets, host tooling and validation, and ML training and inference.

## Alternatives considered

- **Keep gain lock.** The experiments did not show it as a reliable base across devices, and it brought operational problems.
- **Keep several normalization paths.** Harder to reason about, weaker Python/C++ alignment, and datasets and models that cannot be compared.

## Consequences

Runtime, datasets, and ML share one normalization, cross-chip and cross-session results are easier to read, and we avoid the transport problems caused by forced gain. Some earlier experiments and models became history rather than baselines, and later evaluations have to be read as starting from the AGC-active reset.

## Related

- [`2026-08-11-promote-channel-shape-trajectory-ml-features.md`](2026-08-11-promote-channel-shape-trajectory-ml-features.md)
- [`2026-06-30-separate-ml-training-data-from-promotion-replays.md`](2026-06-30-separate-ml-training-data-from-promotion-replays.md)
- Commits `52859c06` (gain-shift diagnostic) and `8b150797` (gain lock removed, CV unified)
