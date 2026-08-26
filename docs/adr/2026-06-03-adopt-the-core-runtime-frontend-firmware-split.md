# ADR: adopt the core-runtime-frontend firmware split

- Status: Accepted
- Date: 2026-06-03
- Recorded: 2026-07-09 (retrospective)
- Updated: 2026-08-26

## Context

Before the split, firmware concerns were clustered too closely around the ESPHome component layout. Detection logic, runtime orchestration, and ecosystem-specific integration surfaces were harder to evolve independently, which made new firmware targets expensive and risked leaking integration details into reusable sensing code.

The resulting layout supports ESPHome, Native, and Matter without duplicating the sensing pipeline. The former Streamer frontend used the same boundaries before raw collection moved into the maintained sensing frontends; removing that frontend did not change the layer contract.

## Decision

Split the firmware-side C++ code into three explicit layers:

- `src/cpp/core/` for reusable detectors, signal processing, and domain logic
- `src/cpp/runtime/` for portable lifecycle, event, snapshot, and boundary contracts, with ESP-IDF implementations under `runtime/esp_idf/`
- `src/cpp/frontend/` for integration-specific surfaces such as ESPHome, Native, and Matter

Keep `core` frontend-agnostic, keep platform orchestration in `runtime`, and keep ecosystem-specific behavior in the relevant `frontend`.

Python retains a complementary role: Micro-ESPectre provides the device-side prototype path, while host-side Python owns research, validation, training, and export. Production behavior that graduates from Python must still land in the shared C++ layers and relevant frontends.

## Decision History

| Date | Direction | Resolution |
| --- | --- | --- |
| 2025-12-06 | Use a dual C++ production and Python experimentation model | Retained as the development workflow |
| 2026-06-03 | Split production C++ into Core, Runtime, and Frontend layers | Accepted as the firmware architecture |
| 2026-08-25 | Remove the dedicated Streamer frontend and move raw collection into the shared runtime | Retained the layer split while reducing the frontend set |

## Alternatives Considered

### Keep the ESPHome-centered layout

Rejected. It would reduce short-term work for ESPHome but require Native, Matter, and future frontends to copy or depend on ESPHome-specific logic.

### Split only by platform or chip family

Rejected. That would organize build targets, but would not separate reusable motion-detection logic from runtime and integration concerns.

## Consequences

Benefits:

- new frontends can reuse the same sensing and calibration foundations
- algorithm work in `core` no longer depends on a single integration surface
- runtime behavior can be shared across multiple ESP-IDF frontends

Trade-offs:

- cross-layer boundaries must stay disciplined
- some refactors become larger because contracts have to remain explicit

## Related

- [`2025-12-06-adopt-esphome-as-the-primary-home-assistant-integration.md`](2025-12-06-adopt-esphome-as-the-primary-home-assistant-integration.md)
- [`2025-11-28-prototype-in-python-before-porting-to-production-firmware.md`](2025-11-28-prototype-in-python-before-porting-to-production-firmware.md)
- [`2026-07-02-use-one-message-model-and-command-engine-across-transports.md`](2026-07-02-use-one-message-model-and-command-engine-across-transports.md)
- [`2026-07-03-unify-raw-csi-collection-over-http.md`](2026-07-03-unify-raw-csi-collection-over-http.md)
- git commits: `57b126ba`, `c43d51b8`, `77fa9f48`
