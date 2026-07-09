# ADR: adopt the core-runtime-frontend firmware split

- Status: Accepted
- Date: 2026-06-03
- Recorded: 2026-07-09 (retrospective)

## Context

Before the split, firmware concerns were clustered too closely around the
ESPHome component layout. Detection logic, runtime orchestration, and
ecosystem-specific integration surfaces were harder to evolve independently,
which made new firmware targets expensive and risked leaking integration
details into reusable sensing code.

The repository history and later architecture documentation show a clear shift
toward a platform that can support ESPHome, native firmware, Matter, and the
streamer frontend without duplicating the sensing pipeline.

## Decision

Split the firmware-side C++ code into three explicit layers:

- `src/cpp/core/` for reusable detectors, signal processing, and domain logic
- `src/cpp/runtime/` for CSI acquisition, calibration, Wi-Fi, and platform
  orchestration
- `src/cpp/frontend/` for integration-specific surfaces such as ESPHome,
  native, Matter, and streamer

Keep `core` frontend-agnostic, keep platform orchestration in `runtime`, and
keep ecosystem-specific behavior in the relevant `frontend`.

## Alternatives Considered

### Keep the ESPHome-centered layout

Rejected. It would keep shipping velocity acceptable for one integration, but
would make native, Matter, and streamer support harder to maintain without
copying logic.

### Split only by platform or chip family

Rejected. That would organize build targets, but would not separate reusable
motion-detection logic from runtime and integration concerns.

## Consequences

Benefits:

- new frontends can reuse the same sensing and calibration foundations
- algorithm work in `core` no longer depends on a single integration surface
- runtime behavior can be shared across multiple ESP-IDF frontends

Trade-offs:

- cross-layer boundaries must stay disciplined
- some refactors become larger because contracts have to remain explicit

## Related

- `docs/adr/2025-12-06-adopt-esphome-as-the-production-integration-surface.md`
- `docs/adr/2025-12-06-adopt-a-dual-platform-development-model.md`
- `docs/adr/2026-07-03-adopt-a-dedicated-cpp-streamer-frontend-for-high-rate-csi-collection.md`
- git commits: `57b126ba`, `c43d51b8`, `77fa9f48`
