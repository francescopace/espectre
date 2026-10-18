# ADR: remove QEMU smoke tests from firmware CI

- Status: Accepted
- Date: 2026-07-18
- Updated: 2026-08-26

## Context

CI builds every maintained frontend (then ESPHome, Native, Matter, and Streamer; now the first three) across its chip matrix. QEMU smoke tests multiplied that with extra matrix branches, artifacts, log uploads, and per-frontend workarounds.

What QEMU could check was early boot: a valid merged image, a running bootloader, a readable partition table, a loaded app, maybe a startup marker. That is not where ESPectre breaks. Espressif QEMU does not emulate Wi-Fi PHY and modem-clock bring-up, the Bluetooth controller and NimBLE, Matter or Improv over BLE, or all the chips we ship.

Local runs confirmed it. Native on ESP32-C3 asserted at Bluetooth controller init, before any real BLE logic ran. Streamer reached its startup marker and then failed in the Wi-Fi PHY path. CSI sensing, provisioning, association, BLE, and Matter commissioning were never exercised. And since only some chips were covered, a green check looked more meaningful than it was.

## Decision

QEMU smoke tests are removed from firmware CI, together with their workflow branches, helper actions, configs, log artifacts, and any code that existed only for them.

CI still builds every supported firmware target, produces the snapshot and release artifacts, and runs the host-side tests.

## Alternatives considered

- **Keep QEMU for some frontends.** Streamer gave a thin early-boot signal, not worth a custom CI path, and it said nothing about Wi-Fi or CSI.
- **Keep QEMU for some chips.** More maintenance, and a status check that looks more representative than it is.
- **Keep it as a non-blocking job.** Still costs CI time and complexity, and invites QEMU-specific code to creep back in.

## Consequences

Firmware CI is shorter and simpler, with fewer per-frontend exceptions, and production firmware no longer bends around emulator limits. A green CI means the product matrix built, not that an emulator half-booted.

We lose the early-boot sanity check, so some image-assembly bugs may surface later. To compensate: strong host-side tests, the product build itself as the integrity check, and targeted hardware smoke tests when radio behavior matters.

## Related

- `.github/workflows/ci.yml` and `.github/workflows/cd.yml`
- Commits `3eff278c` (QEMU extended to Native and Streamer) and `af9060a5` (QEMU removed)
