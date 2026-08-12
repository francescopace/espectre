# ESPectre Matter Frontend

Matter is a smart-home interoperability standard that lets devices work across major ecosystems such as Apple Home, Google Home, Amazon Alexa, Samsung SmartThings, Home Assistant's Matter integration, and other Matter-compatible controllers. ESPectre exposes the standard occupancy sensor device type, so any controller that supports Matter occupancy sensors can consume it without a dedicated ESPectre integration.

Use this guide to flash, commission, or maintain the Matter frontend. First-time users can jump to [Getting Started](#getting-started) and [Exposed Matter Surface](#exposed-matter-surface); firmware developers can continue through runtime ownership and implementation details.

## Scope

The Matter frontend is responsible for:

- Matter-specific surface mapping
- Matter firmware app startup and commissioning flow
- standard occupancy publishing over Matter
- target-specific build, flash, and monitor workflow

## Getting Started

If you came from the shared setup hub, this README covers the Matter workflow after flashing or when building locally.

### Browser-Flashed Firmware

Start from [`SETUP.md`](../../../../docs/SETUP.md) for the shared browser-flash entry point and supported image flow.

Each release and snapshot publishes one full-flash Matter image per supported chip. The current Matter frontend does not publish a separate OTA image.

After flashing a Matter image:

1. power-cycle if needed and wait for the device to boot
2. use **Read Matter QR** on the web flasher, or run `./espectre matter qr --port <port>`, to retrieve the device-specific code
3. use a Matter controller that supports BLE commissioning
4. commission the device into your target fabric
5. use the standard Matter occupancy surface exposed by the firmware

The first boot generates a random setup passcode, discriminator, and SPAKE2+ salt in the dedicated `matter_factory` partition. Normal browser and CLI flashes preserve that partition, so every surface shows the same QR for the physical device. A full flash erase intentionally creates a new QR on the next boot.

### Local ESP-IDF Workflow

Before building locally, complete the shared [`Local Build Prerequisites`](../../../../docs/SETUP.md#local-build-prerequisites). The repository CLI prefers a reusable local ESP-IDF installation and falls back to the pinned Docker build environment when local ESP-IDF is absent; use [`CLI.md`](../../../../docs/CLI.md) for backend controls and command syntax.

Repository CLI:

```bash
./espectre matter build --chip c3 --clean
./espectre matter flash --port /dev/cu.usbmodemXXXX
./espectre matter qr --port /dev/cu.usbmodemXXXX
./espectre monitor --port /dev/cu.usbmodemXXXX
```

Notes:

- On Windows, use `.\espectre.cmd matter ...` and `.\espectre.cmd monitor --port COM5`.
- Docker can replace local ESP-IDF for `build`; `flash` and `doctor` continue to use the local environment.
- Shared sensing options are selected through the shared ESPectre sensing `sdkconfig` menu.
- the first build downloads managed components and compiles `esp_matter`, so it is significantly slower than incremental builds

<details>
<summary>Advanced raw ESP-IDF flow</summary>

```bash
cd src/cpp/frontend/matter/app
idf.py set-target esp32c3
idf.py build
idf.py -p /dev/cu.usbmodemXXXX flash
idf.py -p /dev/cu.usbmodemXXXX monitor
```

</details>

## Commissioning and Runtime Ownership

The Matter frontend keeps ownership boundaries explicit:

- the Matter stack starts first
- the shared ESPectre runtime is initialized after `esp_matter::start()`
- Wi-Fi ownership remains with `esp-matter`
- the firmware keeps the shared Wi-Fi transport baseline active during commissioning, including AMPDU enabled plus the larger Wi-Fi and lwIP queues
- CSI services remain disarmed until commissioning completes
- after commissioning, the reused runtime layers CSI Wi-Fi policy and capture setup on top of the initialized station stack

That ordering is visible in [`app_main.cpp`](app/main/app_main.cpp).

The Matter frontend also uses the same shared periodic progress-bar sensing status log helper used by the ESPHome and standalone native frontends, so runtime serial diagnostics stay aligned across the ecosystem-facing firmware targets.

### Commissioning Window Behavior

The firmware opens a basic commissioning window for uncommissioned devices and re-opens it when the last fabric is removed.

Current behavior from the firmware app:

- commissioning data is generated locally with the ESP32 hardware RNG
- onboarding data persists in the `matter_factory` partition at the end of flash and is independent from the application image
- every boot emits `MATTER_QR` and `MATTER_MANUAL_CODE` markers on serial
- the browser and CLI read those markers rather than generating competing codes
- an uncommissioned device opens a `300` second commissioning window
- the commissioning window advertises all supported discovery transports, including BLE
- DNS-SD includes the commissionable device type for an occupancy sensor
- commissioning completion is logged
- a failed commissioning attempt is logged when the fail-safe timer expires
- removing the last fabric re-opens the commissioning window automatically

Only the firmware-owned behavior is documented here. The exact controller UX, QR/manual-pairing presentation, and fabric-management screens depend on the Matter controller you use.

## Exposed Matter Surface

The current frontend exposes:

| Feature | Matter mapping | Type | Access |
|---------|----------------|------|--------|
| Motion detected | `OccupancySensing` occupancy bitmap | bitmap | read-only |

## What You Can Configure Today

The shared sensing options, defaults, and ranges are documented in [`SETUP.md`](../../../../docs/SETUP.md). This README covers only the Matter-owned surface.

What is not currently exposed as a Matter configuration surface:

- writable ESPectre runtime controls
- full detector parameter parity with the ESPHome YAML surface
- an end-user Matter-native workflow for every runtime knob
- a separate frontend-owned tuning guide beyond the shared [`TUNING.md`](../../../../docs/TUNING.md)

Matter supports both `classic` and `ml` as build-time detector choices. Choose Classic to leave more detector CPU and working memory for the Matter stack or other product work; choose ML for higher detection accuracy, stronger generalization, and startup without Classic's quiet calibration of up to about 10 seconds. ML still waits for CSI readiness and feature-window warmup. The published firmware selects Classic, while a local build can select ML through the shared ESP-IDF sensing configuration. Unlike ESPHome and Native, Matter does not expose runtime detector selection or persist an end-user detector choice.

In practice, this frontend is best understood as:

- a Matter-native occupancy surface
- without ESPectre-specific writable controls
- over the shared ESPectre runtime

## Targets and Validation

Current published Matter targets:

- `ESP32`
- `ESP32-S3`
- `ESP32-C3`
- `ESP32-C5`
- `ESP32-C6`

Validation notes:

| Area | Current status |
| --- | --- |
| Firmware hardware smoke | Recorded on `ESP32-C3` |
| Controller commissioning | Limited; no complete cross-controller validation matrix has been published |

Published target availability does not imply that every controller and target combination has been commissioned successfully. Add verified controller results to this table only with a reproducible hardware test record.

## Implementation Map

This map is for frontend maintainers; it is not required for commissioning an existing image.

- [`matter_frontend.cpp`](espectre/matter_frontend.cpp), [`matter_frontend.h`](espectre/matter_frontend.h): frontend adapter over the shared runtime frontend controller
- [`matter_surface.h`](espectre/matter_surface.h): cluster and attribute IDs plus Matter mapping helpers
- [`matter_bindings.h`](espectre/matter_bindings.h): boundary between the adapter and the Matter transport layer
- [`app/`](app/): standalone ESP-IDF firmware app
- [`app_main.cpp`](app/main/app_main.cpp): Matter node setup, endpoint creation, commissioning window behavior, and startup order
- [`idf_component.yml`](app/main/idf_component.yml): `esp_matter` dependency declaration

## Dependencies and Firmware Layout

- firmware app: [`app/`](app/)
- dependency manager: ESP-IDF Component Manager
- declared external dependency: `espressif/esp_matter`
- Matter device type: occupancy sensor (`0x0107`)
- development VID/PID: `0xFFF1` / `0x8000`
- partition layout: [`partitions.csv`](app/partitions.csv)
- defaults: [`sdkconfig.defaults`](app/sdkconfig.defaults)

The per-device onboarding flow removes the shared Matter test passcode, but the published firmware still uses development VID/PID and example device attestation credentials. Production certification requires a manufacturing pipeline for unique DAC credentials in addition to this onboarding partition.

No manual `esp_matter` clone is required.

## OTA

Matter OTA is not supported in the current firmware scope.

Current behavior:

- the Matter frontend does not expose the Matter OTA requestor path
- the shared ESPectre MQTT-triggered HTTPS OTA service is not reused by Matter
- published Matter images are full firmware images intended for manual flashing and commissioning workflows

Future Matter OTA work, if it returns, should come back as a complete Requestor-plus-Provider design rather than a direct firmware download path.

## Matter-Specific Troubleshooting

### The device does not appear for commissioning

Check these first:

1. the controller supports BLE commissioning
2. the device is uncommissioned or the previous fabric was removed
3. serial logs show the Matter firmware started successfully

### Commissioning fails and times out

The firmware logs fail-safe expiration events. Retry with:

1. the board close to the controller during BLE commissioning
2. a fresh power cycle
3. a controller that supports the target platform cleanly

### Google Home commissioning is slow

Google Home is stricter than local development controllers about the advertised device type, commissioning transports, and Wi-Fi behavior during network commissioning.

Check that:

1. the serial log shows CSI services as `waiting for commissioning` before pairing completes
2. the commissioning window advertises BLE through the all-supported transport mode
3. the image was built from `sdkconfig.defaults` with commissionable device type enabled and device type `0x0107`
4. Wi-Fi CSI policy logs appear only after `Commissioning complete`

### Runtime values are not exposed as writable Matter controls

That is expected in the current frontend. The Matter surface is intentionally kept to standard occupancy behavior instead of mirroring the broader ESPectre runtime control plane.
