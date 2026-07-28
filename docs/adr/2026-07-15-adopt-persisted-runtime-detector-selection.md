# ADR: adopt persisted runtime detector selection

- Status: Superseded in part
- Date: 2026-07-15

The persisted runtime-selection decision remains active for ESPHome and Native.
Matter remains read-only but changed from the ML default recorded below to the
Classic default on 2026-07-28.

## Context

ESPectre supported `classic` and `ml`, but firmware selected the detector only
before runtime setup. Comparing detectors on deployed ESPHome and Native
devices therefore required rebuilding or reflashing firmware. A frontend-local
solution would have duplicated validation, persistence, threshold reset, and
calibration behavior across integration surfaces.

The frontends also have different control contracts. ESPHome and Native expose
writable runtime controls, Matter is intentionally read-only, and Streamer
transports CSI without running a detector.

## Decision

Add detector selection to the shared runtime contract and gate it through an
explicit runtime capability.

Concretely:

- let ESPHome and Native select `classic` or `ml` at runtime
- persist their selection in a shared ESP-IDF NVS store and restore it at boot
- reset the threshold to the selected detector's default when switching
- start calibration automatically when switching to `classic`, and cancel any
  active calibration when switching to `ml`
- keep Matter without a writable detector surface and use `ml` as its
  frontend-owned firmware default
- keep Streamer detector-free and report runtime detector selection as
  unsupported

The switch remains an explicit control-path operation and adds no detector
selection work to the CSI packet hot path.

## Alternatives Considered

### Keep detector selection build-time only

Rejected. It would preserve a smaller runtime surface, but deployed detector
comparisons would continue to require rebuilding or reflashing firmware.

### Persist detector selection in each frontend

Rejected. Separate ESPHome and Native stores would duplicate behavior and risk
different validation, boot restoration, threshold, and calibration semantics.

### Expose the same writable control on every frontend

Rejected. Matter's product surface is intentionally read-only, and Streamer
does not own a detector to configure.

## Consequences

Benefits:

- ESPHome and Native devices can compare detectors without reflashing
- persistence, validation, threshold reset, and calibration behavior stay
  aligned in the shared runtime
- capabilities make unsupported frontend behavior explicit
- detector switching does not add continuous CSI processing overhead

Trade-offs:

- the runtime contract, events, protocol, and frontend controls must evolve
  together
- persisted detector state can override the build-time default on supported
  frontends
- Matter intentionally differs by using a fixed `ml` default without consuming
  the shared persisted selection

## Related

- `docs/adr/2026-06-03-adopt-the-core-runtime-frontend-firmware-split.md`
- `docs/adr/2026-07-02-use-a-shared-espectre-protocol-across-esp-idf-frontends.md`
- `docs/adr/2026-07-03-adopt-a-dedicated-cpp-streamer-frontend-for-high-rate-csi-collection.md`
- git commit: `52a6f350`
