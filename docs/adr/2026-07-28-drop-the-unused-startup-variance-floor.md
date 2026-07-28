# ADR: drop the unused startup variance floor

- Status: Accepted
- Date: 2026-07-28

## Context

`StartupThresholdCalibrator` retained a variance-floor path in both C++ and
MicroPython. The detector interface always supplied `0.0`, no detector
implemented the feature, and the host replay path only probed for an attribute
that did not exist. The path therefore wrote, reordered, and retained samples
that could not influence a threshold or verdict.

The inactive C++ state included a 1,000-float ring and three smaller bootstrap
buffers. It also added per-packet writes and per-chunk copies to the CSI
callback.

## Decision

Remove the startup variance-floor contract, storage, processing, and test-only
accessors from both runtimes. Keep the separate motion-level floor used by
`threshold_metric()`; despite the similar name, it participates in the active
calibration decision.

## Validation

- `sizeof(StartupThresholdCalibrator)` fell from 4,636 to 328 bytes, saving
  4,308 bytes.
- All 356 replay metric lines were bit-for-bit identical before and after the
  removal.
- The long-quiet-prefix calibration scenario remains covered directly on both
  runtimes.

## Consequences

- Startup calibration no longer pays memory or callback work for unreachable
  behavior.
- Restoring a variance-floor vote would be a new detector feature. It would
  require an explicit metric contract, Python and C++ parity, and replay
  evidence, rather than reactivating the deleted scaffolding.

## Alternatives Considered

### Repair the floor snapshot

Rejected. Fixing the snapshot mutation would preserve code with no caller and
no effect on detector output.

### Keep the path for future experiments

Rejected. Host-only feature candidates belong under `tools/` until evidence
justifies a production implementation.
