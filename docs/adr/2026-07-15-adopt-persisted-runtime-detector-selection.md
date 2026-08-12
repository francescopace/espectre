# ADR: define detector selection capabilities and frontend defaults

- Status: Accepted
- Date: 2026-07-15
- Updated: 2026-08-12

## Context

ESPectre supports Classic and ML, but its frontends have different control contracts. ESPHome and Native expose writable runtime controls, Matter is intentionally read-only, and Streamer transports CSI without running a detector. A frontend-local implementation would duplicate validation, persistence, threshold reset, and calibration behavior.

## Decision

Keep detector selection in the shared runtime and expose it through explicit frontend capabilities.

| Frontend | Detector capability | Default and persistence |
| --- | --- | --- |
| ESPHome | Writable `classic` or `ml` | Persist the selected value in the shared ESP-IDF store |
| Native | Writable `classic` or `ml` | Persist the selected value in the shared ESP-IDF store |
| Matter | Read-only | Use Classic as the frontend-owned fixed default |
| Streamer | Unsupported | Run no detector |

On a supported runtime switch:

- validate the requested detector;
- reset the threshold to that detector's automatic default;
- start calibration when switching to Classic;
- cancel active Classic calibration when switching to ML; and
- emit the shared runtime and protocol state change without adding work to the CSI hot path.

Matter intentionally exposes no persisted writable selection. Its original ML default changed to Classic before the v3 release candidate so the read-only occupancy frontend follows the platform's non-ML default.

## Decision History

| Date | Direction | Resolution |
| --- | --- | --- |
| 2026-07-15 | Add shared persisted selection; use ML as Matter's fixed default | Persisted selection retained for writable frontends |
| 2026-07-28 | Change Matter's fixed default from ML to Classic | Amended the capability matrix without superseding shared persisted selection |

## Alternatives Considered

### Keep detector selection build-time only

Rejected. Deployed comparisons would continue to require rebuilding or reflashing.

### Persist selection separately in every frontend

Rejected. It would duplicate lifecycle and validation semantics.

### Expose the same writable control everywhere

Rejected. Matter's surface is read-only, and Streamer owns no detector.

## Consequences

- ESPHome and Native can switch detectors without reflashing.
- Persistence, threshold reset, and calibration remain aligned in the shared runtime.
- Capability differences are explicit instead of appearing as frontend drift.
- Choosing ML on Matter still requires a firmware-level product change.

## Related

- [`2026-06-03-adopt-the-core-runtime-frontend-firmware-split.md`](2026-06-03-adopt-the-core-runtime-frontend-firmware-split.md)
- [`2026-07-02-use-a-shared-espectre-protocol-across-esp-idf-frontends.md`](2026-07-02-use-a-shared-espectre-protocol-across-esp-idf-frontends.md)
- [`2026-07-08-promote-classic-detector-and-retire-legacy-baselines.md`](2026-07-08-promote-classic-detector-and-retire-legacy-baselines.md)
- git commit: `52a6f350`
