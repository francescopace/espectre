# ADR: adopt the core-runtime-frontend firmware split

- Status: Accepted
- Date: 2026-06-03
- Recorded: 2026-07-09 (retrospective)
- Updated: 2026-08-30

## Context

The firmware code had grown around the ESPHome component. Detection, orchestration, and integration code were tangled together, so every new firmware target was expensive and integration details kept leaking into the sensing code.

## Decision

The C++ firmware code is split into three layers:

- `src/cpp/core/`: detectors, signal processing, and domain logic. Knows nothing about frontends.
- `src/cpp/runtime/`: portable lifecycle, events, snapshots, and boundary contracts, with the ESP-IDF implementation in `runtime/esp_idf/`.
- `src/cpp/frontend/`: ESPHome, Native, Matter, and anything ecosystem-specific.

Logging follows the same rule. `core` and `runtime` write to a portable sink and never include a platform logger. Each frontend registers its own sink before runtime setup; without one, the SDK stays silent. Micro-ESPectre does the same: its native MicroPython frontend owns the ESP-IDF sink.

Python keeps its role as the prototype and research path (see the Python-first ADR). Anything that graduates to production lands in the shared C++ layers.

## Decision history

| Date | Direction | Resolution |
| --- | --- | --- |
| 2025-12-06 | C++ for production, Python for experiments | Kept as the development workflow |
| 2026-06-03 | Split C++ into core, runtime, and frontend | Accepted |
| 2026-08-25 | Remove the Streamer frontend; raw collection moves into the shared runtime | Layer split unchanged, one fewer frontend |
| 2026-08-30 | Replace the shared `esp_log` dependency with frontend-owned sinks | Logging policy now lives above the SDK layers |

## Alternatives considered

- **Keep the ESPHome-centered layout.** Cheaper for ESPHome in the short term, but Native, Matter, and later frontends would have to copy or depend on ESPHome code.
- **Split by platform or chip family.** That organizes build targets, not responsibilities.

## Consequences

New frontends reuse the same sensing and calibration code, algorithm work in `core` does not depend on any one integration, and SDK users can drop logging or plug in their own backend. The price is discipline: layer boundaries must hold, some refactors get bigger because contracts stay explicit, and sink callbacks must respect the runtime's task and capture-context limits.

## Related

- [`2025-12-06-adopt-esphome-as-the-primary-home-assistant-integration.md`](2025-12-06-adopt-esphome-as-the-primary-home-assistant-integration.md)
- [`2025-11-28-prototype-in-python-before-porting-to-production-firmware.md`](2025-11-28-prototype-in-python-before-porting-to-production-firmware.md)
- [`2026-07-02-use-one-resource-oriented-protocol-across-transports.md`](2026-07-02-use-one-resource-oriented-protocol-across-transports.md)
- [`2026-07-03-unify-raw-csi-collection-over-http.md`](2026-07-03-unify-raw-csi-collection-over-http.md)
- Commits `605e8e37` (split), `9b5ae041` (Streamer removal), `dbfbd2a5` (frontend log sinks)
