# ADR: remove QEMU smoke tests from firmware CI

- Status: Accepted
- Date: 2026-07-18
- Updated: 2026-08-26

## Context

When this decision was made, ESPectre built ESPHome, Native, Matter, and the former Streamer frontend across their supported target matrices. The maintained C++ set is now ESPHome, Native, and Matter, and the exact chip matrix differs by frontend. QEMU smoke coverage multiplied that CI surface with extra matrix branches, artifact handling, log uploads, and frontend-specific workarounds.

The intended value of the QEMU smoke tests was limited boot validation:

- the merged flash image is structurally valid
- the bootloader starts
- the partition table is readable
- the application image loads
- a minimal startup marker may be printed before later hardware init

That signal is weak for ESPectre's actual product risk. The frontends depend on hardware paths which Espressif QEMU does not fully emulate:

- Wi-Fi PHY and modem-clock bring-up
- Bluetooth controller and NimBLE
- Matter-over-BLE commissioning
- ESPHome Improv / BLE provisioning
- full chip coverage across the shipped targets

Local validation confirmed the mismatch:

- `Native` product firmware on `ESP32-C3` booted in QEMU only until Bluetooth controller initialization, then asserted before the frontend could exercise any real BLE fallback logic
- the former `Streamer` frontend could reach the early startup marker, but still failed later in the known Wi-Fi PHY path that QEMU does not model
- QEMU never exercised the real runtime value of CSI sensing, provisioning, Wi-Fi association, BLE, or Matter commissioning

As a result, QEMU increased CI time and complexity while not covering the frontends' highest-risk behavior. It also covered only a subset of the shipped chips, creating an asymmetric and potentially misleading quality signal.

## Decision

Remove QEMU smoke tests from firmware CI.

This includes:

- deleting QEMU-specific workflow branches and log artifacts
- deleting QEMU-specific helper actions and configs
- deleting code or config changes that existed only to support QEMU smoke paths
- keeping the supported product-build matrix for each maintained frontend in CI

CI remains responsible for:

- building all supported firmware targets
- producing publishable artifacts for snapshot and release workflows
- verifying non-firmware test surfaces already covered by host-side tests

## Alternatives Considered

### Keep QEMU only for some frontends

Rejected. The former Streamer frontend could provide a narrow early-boot signal, but that signal was too small to justify the custom CI path and did not cover real Wi-Fi or CSI behavior.

### Keep QEMU only for some chips

Rejected. Partial chip coverage adds maintenance cost and makes the resulting status check look more representative than it really is.

### Keep QEMU as a non-blocking informational job

Rejected. Even as advisory-only, it still consumes CI time, adds workflow complexity, and invites future QEMU-specific code or configuration drift.

## Consequences

Benefits:

- shorter and simpler firmware CI
- fewer frontend-specific CI exceptions and artifacts
- less pressure to keep production firmware shaped around emulator limitations
- clearer signal: successful CI means the product matrix built, not that an emulator partially booted

Trade-offs:

- CI no longer has an emulator-based early-boot sanity check
- some image-assembly regressions may be discovered later than before

Mitigations:

- keep host-side tests strong
- rely on the product build itself as the primary artifact-integrity check
- prefer targeted hardware smoke testing when radio behavior matters

## Related

- `.github/workflows/ci.yml`
- `.github/workflows/snapshot.yml`
- `.github/workflows/release.yml`
- `docs/CHANGELOG.md`
