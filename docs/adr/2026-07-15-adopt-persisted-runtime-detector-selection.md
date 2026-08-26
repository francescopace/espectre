# ADR: define detector selection capabilities and frontend defaults

- Status: Accepted
- Date: 2026-07-15
- Updated: 2026-08-26

## Context

ESPectre supports Lightweight and High Accuracy, but its frontends expose different integration surfaces. ESPHome has entities and Direct HTTP, Native has Direct HTTP and optional MQTT, and Matter has standard occupancy clusters plus Direct HTTP for ESPectre-specific tuning. A frontend-local implementation would duplicate validation, persistence, threshold reset, and calibration behavior.

## Decision

Keep detector selection in the shared runtime and expose it through explicit frontend capabilities.

| Frontend | Detector capability | Default and persistence |
| --- | --- | --- |
| ESPHome | Writable through ESPHome entities or Direct HTTP | Start from the configured value, then persist runtime selection in the shared ESP-IDF store |
| Native | Writable through Direct HTTP or MQTT | Start from the configured value, then persist runtime selection in the shared ESP-IDF store |
| Matter | Writable through Direct HTTP; not represented in standard Matter occupancy clusters | Published firmware starts with Lightweight, then persists runtime selection in the shared ESP-IDF store |

On a supported runtime switch:

- validate the requested detector;
- reset the threshold to that detector's automatic default;
- start calibration when switching to Lightweight;
- cancel active Lightweight calibration when switching to High Accuracy; and
- emit the shared runtime and protocol state change without adding work to the CSI hot path.

Matter keeps detector selection out of its standard occupancy clusters because those clusters do not define an equivalent control. Its shared Direct HTTP bridge advertises `set_detector` when the runtime capability is enabled and persists the accepted value through the same store as the other ESP-IDF frontends.

## Decision History

| Date | Direction | Resolution |
| --- | --- | --- |
| 2026-07-15 | Add shared persisted selection; use ML as Matter's fixed default | Persisted selection retained for writable frontends |
| 2026-07-28 | Change Matter's fixed default from ML to Classic | Amended the initial frontend default; the profiles were later renamed High Accuracy and Lightweight |
| 2026-08-26 | Enable persisted Matter detector selection through Direct HTTP | Accepted without adding a non-standard Matter cluster |

## Alternatives Considered

### Keep detector selection build-time only

Rejected. Deployed comparisons would continue to require rebuilding or reflashing.

### Persist selection separately in every frontend

Rejected. It would duplicate lifecycle and validation semantics.

### Add detector selection to the Matter occupancy surface

Rejected. Standard occupancy clusters do not define this ESPectre-specific control. Direct HTTP provides it without inventing a non-standard Matter data model.

## Consequences

- ESPHome, Native, and Matter can switch detectors without reflashing through their advertised control surfaces.
- Persistence, threshold reset, and calibration remain aligned in the shared runtime.
- Capability differences are explicit instead of appearing as frontend drift.
- Matter controllers continue to see the standard occupancy surface, while Direct clients can select the detector independently.

## Related

- [`2026-06-03-adopt-the-core-runtime-frontend-firmware-split.md`](2026-06-03-adopt-the-core-runtime-frontend-firmware-split.md)
- [`2026-07-02-use-one-message-model-and-command-engine-across-transports.md`](2026-07-02-use-one-message-model-and-command-engine-across-transports.md)
- [`2026-07-08-promote-classic-detector-and-retire-legacy-baselines.md`](2026-07-08-promote-classic-detector-and-retire-legacy-baselines.md)
- git commit: `52a6f350`
