# ESPectre Matter Frontend

This directory contains the ESPectre Matter frontend.

It maps the shared ESPectre runtime to a Matter surface built on top of
`esp_matter`, without pulling Matter-specific concepts into `core` or the
shared runtime contract.

## Scope

The Matter frontend is responsible for:

- Matter-specific surface mapping
- Matter firmware app startup and commissioning flow
- standard occupancy publishing over Matter
- target-specific build, flash, and monitor workflow

## Directory Layout

- [`matter_frontend.cpp`](espectre/matter_frontend.cpp),
  [`matter_frontend.h`](espectre/matter_frontend.h):
  frontend adapter over the shared runtime frontend controller
- [`matter_surface.h`](espectre/matter_surface.h):
  cluster and attribute IDs plus Matter mapping helpers
- [`matter_bindings.h`](espectre/matter_bindings.h):
  boundary between the adapter and the Matter transport layer
- [`app/`](app/):
  standalone ESP-IDF firmware app
- [`app_main.cpp`](app/main/app_main.cpp):
  Matter node setup, endpoint creation, commissioning window behavior, and
  startup order
- [`idf_component.yml`](app/main/idf_component.yml):
  `esp_matter` dependency declaration

## Getting Started

If you came from the shared setup hub, this README covers the Matter workflow
after flashing or when building locally.

### Browser-Flashed Firmware

Start from [`SETUP.md`](../../../../docs/SETUP.md) for the
shared browser-flash entry point and supported image flow.

Each release and snapshot publishes one full-flash Matter image per supported
chip. The current Matter frontend does not publish a separate OTA image.

After flashing a Matter image:

1. power-cycle if needed and wait for the device to boot
2. use a Matter controller that supports BLE commissioning
3. commission the device into your target fabric
4. use the standard Matter occupancy surface exposed by the firmware

### Local ESP-IDF Workflow

Before building locally, complete the shared
[`ESP-IDF Local Build Prerequisite`](../../../../docs/SETUP.md#esp-idf-local-build-prerequisite).
The repository CLI auto-detects a reusable ESP-IDF install, so the wrapper-first
workflow does not require a separate setup check before build.
See [`CLI.md`](../../../../docs/CLI.md) for shared CLI syntax, host-side
tools, and wrapper behavior.

Repository CLI:

```bash
./espectre matter build --chip c3 --clean
./espectre matter flash --port /dev/cu.usbmodemXXXX
./espectre monitor --port /dev/cu.usbmodemXXXX
```

Notes:

- On Windows, use `.\espectre.cmd matter ...` and `.\espectre.cmd monitor --port COM5`.
- If the wrapper cannot find or validate ESP-IDF, run `.\espectre.cmd doctor`
  or `./espectre doctor` for troubleshooting.
- shared sensing options are selected through the shared ESPectre sensing
  `sdkconfig` menu; `matter` currently overrides `ESPECTRE_DETECTION_ALGORITHM`
  to `ML` in [`sdkconfig.defaults`](app/sdkconfig.defaults)
- the first build downloads managed components and compiles `esp_matter`, so it
  is significantly slower than incremental builds

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
- the firmware keeps the shared Wi-Fi transport baseline active during
  commissioning, including AMPDU enabled plus the larger Wi-Fi and lwIP queues
- CSI services remain disarmed until commissioning completes
- after commissioning, the reused runtime layers CSI Wi-Fi policy and capture
  setup on top of the initialized station stack

That ordering is visible in [`app_main.cpp`](app/main/app_main.cpp).

The Matter frontend also uses the same shared periodic progress-bar sensing
status log helper used by the ESPHome and standalone native frontends, so runtime
serial diagnostics stay aligned across the ecosystem-facing firmware targets.

### Commissioning Window Behavior

The firmware opens a basic commissioning window for uncommissioned devices and
re-opens it when the last fabric is removed.

Current behavior from the firmware app:

- an uncommissioned device opens a `300` second commissioning window
- the commissioning window advertises all supported discovery transports,
  including BLE
- DNS-SD includes the commissionable device type for an occupancy sensor
- commissioning completion is logged
- a failed commissioning attempt is logged when the fail-safe timer expires
- removing the last fabric re-opens the commissioning window automatically

Only the firmware-owned behavior is documented here. The exact controller UX,
QR/manual-pairing presentation, and fabric-management screens depend on the
Matter controller you use.

## Exposed Matter Surface

The current frontend exposes:

| Feature | Matter mapping | Type | Access |
|---------|----------------|------|--------|
| Motion detected | `OccupancySensing` occupancy bitmap | bitmap | read-only |

## What You Can Configure Today

The shared sensing options, defaults, and ranges are documented in
[`SETUP.md`](../../../../docs/SETUP.md). This README covers only the Matter-owned
surface.

What is not currently exposed as a Matter configuration surface:

- writable ESPectre runtime controls
- full detector parameter parity with the ESPHome YAML surface
- an end-user Matter-native workflow for every runtime knob
- a separate frontend-owned tuning guide beyond the shared
  [`TUNING.md`](../../../../docs/TUNING.md)

In practice, this frontend is best understood as:

- a Matter-native occupancy surface
- without ESPectre-specific writable controls
- over the shared ESPectre runtime

Matter firmware uses `ml` as its frontend default and does not load or expose
the persisted detector selection used by ESPHome and Native.

## Targets and Validation

Current published Matter targets:

- `ESP32`
- `ESP32-S3`
- `ESP32-C3`
- `ESP32-C5`
- `ESP32-C6`

Validation notes:

- the current recorded hardware smoke target is `ESP32-C3`
- CI QEMU smoke currently covers `ESP32`, `ESP32-S3`, and `ESP32-C3`
- QEMU uses a dedicated no-BLE overlay and does not represent normal hardware
  commissioning behavior

## Dependencies and Firmware Layout

- firmware app: [`app/`](app/)
- dependency manager: ESP-IDF Component Manager
- declared external dependency: `espressif/esp_matter`
- Matter device type: occupancy sensor (`0x0107`)
- development VID/PID: `0xFFF1` / `0x8000`
- partition layout: [`partitions.csv`](app/partitions.csv)
- defaults: [`sdkconfig.defaults`](app/sdkconfig.defaults)

No manual `esp_matter` clone is required.

## OTA

Matter OTA is not supported in the current firmware scope.

Current behavior:

- the Matter frontend does not expose the Matter OTA requestor path
- the shared ESPectre MQTT-triggered HTTPS OTA service is not reused by Matter
- published Matter images are full firmware images intended for manual flashing
  and commissioning workflows

Future Matter OTA work, if it returns, should come back as a complete
Requestor-plus-Provider design rather than a direct firmware download path.

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

Google Home is stricter than local development controllers about the advertised
device type, commissioning transports, and Wi-Fi behavior during network
commissioning.

Check that:

1. the serial log shows CSI services as `waiting for commissioning` before
   pairing completes
2. the commissioning window advertises BLE through the all-supported transport
   mode
3. the image was built from `sdkconfig.defaults` with commissionable device type
   enabled and device type `0x0107`
4. Wi-Fi CSI policy logs appear only after `Commissioning complete`

### Runtime values are not exposed as writable Matter controls

That is expected in the current frontend. The Matter surface is intentionally
kept to standard occupancy behavior instead of mirroring the broader ESPectre
runtime control plane.
