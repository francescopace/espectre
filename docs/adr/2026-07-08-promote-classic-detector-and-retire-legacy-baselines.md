# ADR: promote Classic and retire legacy detector baselines

- Status: Accepted
- Date: 2026-07-08
- Recorded: 2026-07-09 (retrospective)
- Updated: 2026-08-12

## Context

The project accumulated a multi-state detector, moving-variance variants, threshold modes, and several L1-based Classic formulations. They were useful during exploration, but keeping them in the production runtime multiplied configuration, calibration, tests, and frontend behavior.

The durable decision is the production role of `ClassicDetector`, not any intermediate feature pair. The current feature definition is owned by the aggregated-turbulence ADR, while current calibration behavior is documented in `ALGORITHMS.md` and the settled-session recovery ADR.

The public profile and class were later renamed Lightweight Detection and `LightweightDetector`. Historical `Classic` names below identify the decision lineage described by the goal-oriented naming ADR.

An inert startup variance-floor sampling path was also removed from the shared C++ and Python calibration implementations. It never influenced detector output: replay metrics remained bit-for-bit identical, while `sizeof(StartupThresholdCalibrator)` fell from `4,636` to `328` bytes, saving `4,308` bytes during calibration. This removed path was distinct from the motion-level floor still used by the current threshold metric.

## Decision

Use the detector now exposed as `LightweightDetector` as the only production non-ML detector, and remove legacy detector baselines from the active runtime surface.

Lightweight must:

- expose the same `0.0-1.0` probability threshold scale as High Accuracy;
- use one automatic startup-calibration path plus session-only runtime overrides;
- restore its automatic threshold after recalibration or restart;
- keep Python and C++ feature, filtering, fusion, reset, and calibration behavior aligned; and
- remain a compact deterministic alternative for deployments that do not select High Accuracy.

Historical MVS, moving-variance, L1-primary, recovery-vote, threshold-mode, and low-RSSI L1-blend formulations remain research history, not selectable runtime modes.

## Decision History

| Date | Direction | Resolution |
| --- | --- | --- |
| 2025-11-16 | Use a segmentation-first MVS architecture | Superseded by the simpler production Classic path |
| 2026-07-07 | Use gated startup calibration for an L1-primary detector | Replaced by probability calibration and later current threshold adaptation |
| 2026-07-08 | Promote L1-primary Classic and retire legacy baselines | Preserved as the product-direction decision |
| 2026-07-16 | Add a guarded MVS recovery vote | Replaced by direct weighted two-feature fusion |
| 2026-07-22 | Blend to a session-centered L1 excursion at low RSSI | Retired when Classic stopped consuming L1 features |
| 2026-07-28 | Remove inert startup variance-floor sampling | Accepted after bit-for-bit replay validation and a measured `4,308`-byte calibration-memory reduction |

## Alternatives Considered

### Keep multiple production baselines

Rejected. It would preserve comparison convenience at the cost of duplicated configuration, calibration, reset behavior, and frontend support.

### Keep build-time threshold modes

Rejected. Automatic startup placement and session overrides cover the production contract without persistent mode complexity.

### Keep historical feature branches inside Classic

Rejected. Candidate features belong in host-side tools and `FEATURES.md` until a promotion decision replaces the current pair.

## Consequences

Benefits:

- the project has one non-ML detector contract;
- frontend controls and calibration lifecycle do not branch by historical baseline; and
- algorithm evolution can replace Lightweight internals without reviving old runtime modes.

Trade-offs:

- historical comparisons must use host-side tooling or Git history; and
- current feature and calibration rationale is split into the narrow ADRs that own those topics.

## Related

- [`2026-03-08-use-host-side-validation-gates-for-detector-promotion.md`](2026-03-08-use-host-side-validation-gates-for-detector-promotion.md)
- [`2026-07-26-recover-the-startup-threshold-once-a-session-settles.md`](2026-07-26-recover-the-startup-threshold-once-a-session-settles.md)
- [`2026-08-13-use-aggregated-turbulence-iqr-for-lightweight.md`](2026-08-13-use-aggregated-turbulence-iqr-for-lightweight.md)
- [`../ALGORITHMS.md`](../ALGORITHMS.md)
- git commits: `dc0658ed`, `5b871159`, `dbbe21dd`, `b2e0de00`, `8641425d`, `acec4a2c`, `a593edb1`
