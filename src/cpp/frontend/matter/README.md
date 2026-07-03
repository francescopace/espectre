# ESPectre Matter Frontend

This directory contains the ESPectre Matter frontend.

It maps the shared ESPectre runtime to a Matter surface built on top of
`esp_matter`, without pulling Matter-specific concepts into `core` or the
shared runtime contract.

## Scope

The Matter frontend is responsible for:

- Matter-specific surface mapping
- Matter firmware app startup and commissioning flow
- vendor-cluster diagnostics and runtime controls
- target-specific build, flash, and monitor workflow

## Directory Layout

- [`espectre/matter_frontend.cpp`](espectre/matter_frontend.cpp),
  [`espectre/matter_frontend.h`](espectre/matter_frontend.h):
  frontend adapter over the shared runtime frontend controller
- [`espectre/matter_surface.h`](espectre/matter_surface.h):
  cluster and attribute IDs plus Matter mapping helpers
- [`espectre/matter_bindings.h`](espectre/matter_bindings.h):
  boundary between the adapter and the Matter transport layer
- [`app/`](app/):
  standalone ESP-IDF firmware app
- [`app/main/app_main.cpp`](app/main/app_main.cpp):
  Matter node setup, endpoint creation, commissioning window behavior, and
  startup order
- [`app/main/idf_component.yml`](app/main/idf_component.yml):
  `esp_matter` dependency declaration

## Getting Started

If you came from the shared setup hub, this README is now the source of truth
for the Matter workflow after flashing or when building locally.

### Browser-Flashed Firmware

After flashing a Matter image:

1. power-cycle if needed and wait for the device to boot
2. use a Matter controller that supports BLE commissioning
3. commission the device into your target fabric
4. continue using the Matter surface documented below for runtime visibility and
   control

The current implementation relies on BLE commissioning, so `ESP32-S2` is not
part of the supported target set.

### Local ESP-IDF Workflow

One-time repository setup:

```bash
python3 -m venv .venv
python -m pip install -r requirements.txt
```

Per-shell environment setup:

```bash
source .venv/bin/activate
source <ESP_IDF_PATH>/export.sh
```

If ESP-IDF was installed through PlatformIO or ESPHome, a common export path is:

```bash
source ~/.platformio/packages/framework-espidf/export.sh
```

Repository CLI:

```bash
./espectre matter build --chip c3
./espectre matter flash --port /dev/cu.usbmodemXXXX
./espectre monitor --port /dev/cu.usbmodemXXXX
```

Notes:

- On Windows, use `.\espectre.cmd matter ...` for build/flash and
  `.\espectre.cmd monitor --port COM5` for serial logs.
- `./espectre` / `.\espectre.cmd` still requires the repository Python dependencies from
  `requirements.txt`
- `idf.py` must already be available in the shell through the ESP-IDF export
  script
- the Matter frontend detector is selected through `sdkconfig`; the default is
  `ML`
- the first build downloads managed components and compiles `esp_matter`, so it
  is significantly slower than incremental builds

<details>
<summary>Raw ESP-IDF flow</summary>

```bash
source <ESP_IDF_PATH>/export.sh
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
- the firmware leaves normal Wi-Fi aggregation enabled for commissioning
- CSI services remain disarmed until commissioning completes
- after commissioning, the reused runtime layers CSI Wi-Fi policy and capture
  setup on top of the initialized station stack

That ordering is visible in [`app/main/app_main.cpp`](app/main/app_main.cpp).

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

This README intentionally documents only the firmware-owned behavior. The exact
controller UX, QR/manual-pairing presentation, and fabric-management screens
depend on the Matter controller you use.

## Exposed Matter Surface

The current frontend exposes:

| Feature | Matter mapping |
|---------|----------------|
| Motion detected | `OccupancySensing` occupancy bitmap |
| Movement metric | Vendor cluster `0xFFF1FC01`, attribute `0x0000` |
| Threshold | Vendor cluster writable attribute `0x0001` |
| Calibrating | Vendor cluster attribute `0x0002` |
| Ready-to-publish | Vendor cluster attribute `0x0003` |
| Best Pxx | Vendor cluster attribute `0x0004` |
| Gain locked | Vendor cluster attribute `0x0005` |
| Manual recalibration trigger | Vendor cluster writable attribute `0x0006` |

Relevant constants live in [`espectre/matter_surface.h`](espectre/matter_surface.h).

## What You Can Configure Today

Today the Matter surface exposes runtime control for:

- threshold updates through the writable vendor attribute
- manual recalibration through the writable vendor attribute

What is not currently exposed as a Matter configuration surface:

- full detector parameter parity with the ESPHome YAML surface
- an end-user Matter-native workflow for every runtime knob
- a separate frontend-owned tuning guide beyond the shared
  [`../../../../docs/TUNING.md`](../../../../docs/TUNING.md)

In practice, this frontend is best understood as:

- a Matter-native occupancy and diagnostics surface
- with limited writable runtime controls
- over the shared ESPectre runtime

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
- partition layout: [`app/partitions.csv`](app/partitions.csv)
- defaults: [`app/sdkconfig.defaults`](app/sdkconfig.defaults)

No manual `esp_matter` clone is required.

## Matter-Specific Troubleshooting

### The device does not appear for commissioning

Check these first:

1. the target is not `ESP32-S2`
2. the controller supports BLE commissioning
3. the device is uncommissioned or the previous fabric was removed
4. serial logs show the Matter firmware started successfully

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

### Runtime values are visible but not all tuning knobs are writable

That is expected in the current frontend. The Matter surface exposes threshold
and recalibration today, not the full ESPHome parameter surface.
