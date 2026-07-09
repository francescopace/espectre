# ADR: keep agc active and standardize cv normalization

- Status: Accepted
- Date: 2026-07-04
- Recorded: 2026-07-09 (retrospective)

## Context

Earlier motion-detection and ML work had to cope with gain-dependent behavior.
The gain-shift diagnostic on the older raw-feature ML baseline showed the core
failure mode directly: nominal-gain quality looked strong, but artificial
uniform gain shifts caused large false-positive swings, proving that global
feature scaling was not the same thing as structural gain invariance.

The later relative-feature and weighting work reinforced that conclusion rather
than replacing it. The relative-8 topology and `fp_weight` sweep recovered a
gain-stable ML baseline only after the feature path moved to relative
statistics. The later MVS-guided weighting retrain improved some long-run
results, but it also confirmed that detector-guided weighting is not a
substitute for a clean, gain-robust signal path.

At the same time, forced gain management introduced operational downsides,
including instability and Wi-Fi RX/TX issues that could contribute to packet
loss.

## Decision

Remove hardware gain lock from the production path, keep AGC active on all
chips, and use coefficient-of-variation turbulence (`std / mean`) as the single
shared normalization path across:

- runtime motion detection support signals
- collection and dataset workflows
- host-side tooling and validation
- ML training and inference

## Alternatives Considered

### Continue with gain-locked paths

Rejected. The experiments did not support gain lock as a robust cross-device
foundation, and it introduced operational costs of its own.

### Maintain multiple normalization paths

Rejected. Split paths complicated reasoning, weakened cross-stack alignment, and
made datasets and models harder to compare.

## Consequences

Benefits:

- runtime, datasets, and ML share one normalization story
- cross-chip and cross-session behavior is easier to interpret
- the project avoids forced-gain side effects that can harm transport stability

Trade-offs:

- some earlier experiments and models became historical rather than active
  baselines
- later evaluations must be read in the context of the AGC-active reset

## Related

- `docs/adr/2026-07-07-use-core-6-as-the-production-ml-feature-set.md`
- `docs/adr/2026-07-07-reject-detector-guided-sample-weighting-as-the-default-ml-baseline-policy.md`
- git commits: `bf395397`, `aac68d9d`, `86d934fe`
