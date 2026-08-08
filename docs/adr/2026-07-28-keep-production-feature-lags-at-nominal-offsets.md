# ADR: keep production feature lags at nominal offsets

- Status: Accepted
- Date: 2026-07-28
- Supersedes: 2026-07-25-derive-detector-timing-from-the-measured-packet-rate.md

## Context

The earlier timing decision made the detector lags duration-based and left the deployment wiring unfinished. The C++ review completed that wiring, then tested the result outside the `80-133 pps` dead band.

The L1 feature is not one lag. It is a ratio between displacement over a lagged offset and displacement over the previous packet:

```text
mean(|profile[t] - profile[t-lag]|)
----------------------------------
mean(|profile[t] - profile[t-1]|)
```

Deriving only the numerator offset changes the fitted `10:1` relation. At 70 pps, for example, it becomes `7:1`.

## Decision

Deployed C++ and MicroPython runtimes keep the production feature offsets fixed at their nominal packet counts:

- L1 displacement ratio: `10:1`
- turbulence autocorrelation: lag `1`
- detector window: the configured sample count

The evaluation cadence remains time-relative, and calibration uses the same arrival-time schedule as steady-state detection. The supported v3 detector envelope is `80-133 pps`, which contains the recorded production corpus.

`derive_detector_timing()` remains available to host replay and validation tools. It is not called by deployed runtimes. A future rate-independent L1 ratio must scale both offsets together, refit Classic, retrain ML, and pass the per-session non-regression gates before replacing this decision.

## Validation

The low-rate arm decimated 22 normal-link pairs to 75, 65, and 55 pps, for 66 comparable cells:

| Measure | Result |
| --- | --- |
| Cells where derived lags were worse | 13 |
| Cells where derived lags were better | 3 |
| Mean recall delta | `-0.76` points |
| Worst recall delta | `-11.0` points |
| Mean false-positive delta | `-0.14` points |

There was no false-positive improvement to trade against the recall loss. Deriving only the turbulence-autocorrelation lag reproduced the fixed-lag low-rate results, confirming that the partial L1-ratio rescaling caused the regression.

At 500 pps, derived lags reduced false positives from `9.2%` to `0.0%` on the pair with margin, but cost `16.5` recall points. That rate is replay-only and is outside the supported device envelope.

## Consequences

- Startup calibration begins immediately after connection; it no longer waits for enough packets to rebuild the detector.
- Production feature geometry stays aligned with the fitted Classic coefficients and exported ML model.
- High-rate support remains research work and requires a feature change, not a runtime-only lag substitution.

## Alternatives Considered

### Scale only the lagged L1 offset

Rejected by the low-rate A/B measurement because it changes the ratio and regresses recall.

### Scale both L1 offsets

Deferred. This is the physically consistent design, but it changes the feature definition and therefore requires a Classic refit and an ML retrain. The existing corpus does not justify that work for v3.

### Keep the measured-lag runtime wiring but limit it to autocorrelation

Rejected for v3. It adds a session warm-up, detector rebuild, and calibration delay for a high-rate regime that supported hardware does not deliver.
