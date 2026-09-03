# ESPectre Matter Frontend

ESPectre's Matter firmware publishes the standard occupancy sensor device type. A controller that implements that type can consume it without an ESPectre-specific integration.

Start with [Getting Started](#getting-started) to flash and commission a device. [Exposed Matter Surface](#exposed-matter-surface) documents the controller-visible contract; the later sections cover firmware ownership and implementation.

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

The `release`, `preview`, and `develop` channels publish one full-flash Matter image per supported chip. The current Matter frontend does not publish a separate OTA image.

ESP32-S2 is intentionally excluded. It has no Bluetooth radio, while the supported `esp-matter` onboarding path used by this frontend commissions Wi-Fi over Bluetooth. A different non-Bluetooth commissioning design would require a separate architectural decision and implementation.

After flashing a Matter image:

1. power-cycle if needed and wait for the device to boot
2. use **Read the onboarding QR over USB** on the web flasher, or run `./espectre matter qr --chip <chip> --port <port>`, to retrieve the device-specific code
3. use a Matter controller that supports BLE commissioning
4. commission the device into your target fabric
5. use the standard Matter occupancy surface exposed by the firmware

The first boot generates a random setup passcode, discriminator, and SPAKE2+ salt in the dedicated `matter_factory` partition. Normal browser and CLI flashes preserve that partition, so every surface shows the same QR for the physical device. A full flash erase intentionally creates a new QR on the next boot.

### Local ESP-IDF Workflow

Before building locally, complete the shared [`Local Build Prerequisites`](../../../../docs/SETUP.md#local-build-prerequisites). Use [`CLI.md`](../../../../docs/CLI.md) for backend controls and shared command syntax.

Repository CLI:

```bash
./espectre matter build --chip c3
./espectre matter flash --chip c3 --port /dev/cu.usbmodemXXXX
./espectre monitor --chip c3 --frontend matter --port /dev/cu.usbmodemXXXX
```

The flash command prints the onboarding codes captured from the first boot. To retrieve the persisted codes later, run `./espectre matter qr --chip c3 --port /dev/cu.usbmodemXXXX`.

Notes:

- On Windows, use `.\espectre.cmd matter ...` and `.\espectre.cmd monitor --chip c3 --frontend matter --port COM5`.
- Shared sensing options are selected through the shared ESPectre sensing `sdkconfig` menu.
- The first build downloads managed components and compiles `esp_matter`, so it is significantly slower than incremental builds.
- Subsequent builds reuse the target-specific build directory; use `--clean` only when changing an incompatible toolchain or recovering from stale build state.

## Commissioning and Runtime Ownership

The Matter frontend keeps ownership boundaries explicit:

- the Matter stack starts first
- the shared ESPectre runtime is initialized after `esp_matter::start()`
- Wi-Fi ownership remains with `esp-matter`
- the firmware keeps the standalone sensing Wi-Fi transport baseline active during commissioning, including disabled TX and RX AMPDU, Wi-Fi buffer counts `10/32/32`, and the enlarged lwIP queues
- CSI, Direct HTTP, ESPectre DNS-SD, and the one-shot bootstrap responder remain stopped until commissioning completes
- commissioning and fabric events cross into the ESPectre loop through pending events, so the CHIP task never starts, stops, or reconfigures the sensing runtime directly
- a newly commissioned device gives the controller a 10-second completion grace before the sensing runtime can reconfigure Wi-Fi; an already commissioned boot starts without that grace
- the ESPectre runtime allocation is deferred until a fabric exists, preserving heap for SPAKE2+ and operational-certificate validation on constrained targets
- Matter reserves two dynamic endpoints and two device types, matching the root and occupancy endpoint instead of the framework defaults of 16 each
- event history and event queues are sized for the attribute-only occupancy application, and non-ISR FreeRTOS helpers reside in flash to preserve heap during CASE
- Direct discovery idempotently joins the Matter-owned ESP-IDF mDNS responder and never frees that shared responder
- Matter packet buffers use on-demand lwIP RAM allocation, and the station retains the ESP-IDF default dynamic RX count so operational mDNS and CASE do not exhaust the fixed packet pool during commissioning
- the commissionee-only NimBLE profile keeps one peripheral connection, omits unused central and observer roles, preserves BLE security defaults, and releases BLE memory before operational CASE
- after commissioning, the ESPectre loop starts the operational services and layers the reused runtime's CSI Wi-Fi policy and capture setup on top of the initialized station stack
- station IPv4 changes reach the bootstrap responder through IP events instead of polling `esp_netif_get_ip_info()` every 10 ms

That ordering is visible in [`app_main.cpp`](app/main/app_main.cpp).

The Matter frontend uses the shared periodic progress-bar sensing status helper, as do ESPHome and Native. The runtime uses that same one-second heartbeat to cache one CSI and Wi-Fi rate sample consumed by every C++ frontend and returned by Direct diagnostics.

High-rate telemetry follows the detector evaluation cadence only while a Direct SSE client is connected. Runtime callbacks retain snapshots only; Direct serialization happens after the CSI drain, and edge-triggered occupancy updates are scheduled onto the CHIP work queue.

### Commissioning Window Behavior

The firmware opens a basic commissioning window for uncommissioned devices and re-opens it when the last fabric is removed.

Current behavior from the firmware app:

- commissioning data is generated locally with the ESP32 hardware RNG
- onboarding data persists in the `matter_factory` partition at the end of flash and is independent from the application image
- every boot emits `MATTER_QR` and `MATTER_MANUAL_CODE` markers on serial
- the browser and CLI read those markers rather than generating competing codes
- an uncommissioned device opens a `300` second commissioning window
- the commissioning window advertises all supported discovery transports, including BLE
- DNS-SD includes the standard commissionable device type for an occupancy sensor, while the separate `_espectre._tcp.local.` service advertises the Direct HTTP endpoint used by `./espectre devices --frontend matter` after commissioning
- commissioning completion is logged
- a failed commissioning attempt is logged when the fail-safe timer expires
- removing the last fabric stops the ESPectre operational services and re-opens the commissioning window automatically

Only the firmware-owned behavior is documented here. The exact controller UX, QR/manual-pairing presentation, and fabric-management screens depend on the Matter controller you use.

## Exposed Matter Surface

The current frontend exposes:

| Feature | Matter mapping | Type | Access |
|---------|----------------|------|--------|
| Motion detected | `OccupancySensing` occupancy bitmap | bitmap | read-only |

## What You Can Configure Today

The shared sensing options, defaults, and ranges are documented in [`SETUP.md`](../../../../docs/SETUP.md). This README covers only the Matter-owned surface.

The standard Matter surface remains intentionally narrow. It does not expose:

- writable ESPectre detector controls through Matter clusters
- full detector parameter parity with the ESPHome YAML surface
- an end-user Matter-native workflow for every runtime knob
- a separate frontend-owned tuning guide beyond the shared [`TUNING.md`](../../../../docs/TUNING.md)

The firmware exposes `http://<device>:62587/espectre/v1/request` as its local tuning plane. Direct HTTP provides the shared runtime controls, diagnostics, Wi-Fi association inspection, BSSID selection, Basic Information `NodeLabel` editing, peer discovery, and raw CSI advertised by the capability catalog. [`ESPECTRE_PROTOCOL.md`](../../../../docs/ESPECTRE_PROTOCOL.md) owns the method catalog and [peer-assisted browser discovery](../../../../docs/ESPECTRE_PROTOCOL.md#peer-assisted-browser-discovery).

Matter still owns Wi-Fi credentials, commissioning, and fabric access. Direct cannot reset the Wi-Fi configuration or replace the read-only Matter occupancy attribute. It remains available after commissioning, while `_matterc` is advertised only to Matter controllers during an open commissioning window.

ESPectre stores the Direct BSSID preference separately from Matter credentials and binds it to the current Matter-provisioned SSID. Direct acknowledges the staged mutation with the current BSSID before Matter changes the station association. When the requested BSSID is already active, the default behavior only persists the preference; `force=true` performs the reassociation anyway. A new candidate is saved to non-volatile storage before sensing is suspended, CSI is disabled, and its callback is detached. The shared Wi-Fi lifecycle then disconnects, applies the RAM-backed station configuration, and reconnects with promiscuous mode disabled, so Matter-owned credentials remain unchanged. ESPectre commits the candidate only after the station associates with that BSSID and obtains IPv4; otherwise, it reconnects once with the previous preference. After IPv4 returns, the shared runtime runs a non-blocking scan to refresh the CSI receive path, registers a fresh CSI callback, enables a fresh CSI session, and starts a new calibration without rebooting the device. After restart, ESPectre resumes a pending transaction and reapplies a confirmed preference only when the commissioned SSID matches. The preference remains dormant while Matter uses another SSID, and an explicit BSSID clear removes it.

The Direct adapter uses the same `FrontendCommandEngine` as the other C++ frontends; only frontend-owned operations differ.

Matter supports both `lightweight` and `high_accuracy`. Published firmware starts with Lightweight, while a local build can select another initial profile through the shared ESP-IDF sensing configuration. Direct HTTP changes and persists the runtime selection; standard Matter occupancy clusters do not expose that control. [`TUNING.md`](../../../../docs/TUNING.md#startup-and-detection-profile) owns the profile trade-offs and startup procedure.

Matter also supports the shared internal and external traffic modes and the bearer-bound raw HTTP surface. [`SETUP.md`](../../../../docs/SETUP.md#traffic-generation) owns traffic configuration, and [`ESPECTRE_PROTOCOL.md`](../../../../docs/ESPECTRE_PROTOCOL.md#direct-raw-csi) owns raw-session behavior and framing.

## Targets and Validation

Current published Matter targets:

- `ESP32`
- `ESP32-S3`
- `ESP32-C3`
- `ESP32-C5`
- `ESP32-C6`

Generated firmware snapshots record the latest per-target scope and result: [ESP32](../../../../docs/performance/ESP32.md), [ESP32-S3](../../../../docs/performance/ESP32-S3.md), [ESP32-C3](../../../../docs/performance/ESP32-C3.md), [ESP32-C5](../../../../docs/performance/ESP32-C5.md), and [ESP32-C6](../../../../docs/performance/ESP32-C6.md).

Validation notes:

| Area | Current status |
| --- | --- |
| Firmware hardware smoke | Recorded for every published target; the generated chip snapshot defines the exact scope |
| Controller commissioning | Limited; see [Matter Controller Compatibility](#matter-controller-compatibility) |

### Matter Controller Compatibility

The following matrix separates support documented by each controller ecosystem from compatibility validated with ESPectre. Vendor support for the standard Occupancy Sensor device type or Occupancy Sensing cluster does not, by itself, prove that ESPectre commissions, reports state, and triggers automations correctly in that ecosystem. Vendor documentation was last reviewed on 2026-08-26.

| Controller ecosystem | Vendor-documented Matter support | ESPectre validation |
| --- | --- | --- |
| Google Home | Lists the Occupancy Sensor device type (`0x0107`) and Occupancy Sensing cluster (`0x0406`) in its [supported-device matrix](https://developers.home.google.com/matter/supported-devices) | Not yet recorded |
| Amazon Alexa | Maps a Matter motion detector using Occupancy Sensing to `Alexa.MotionSensor` in its [supported-category matrix](https://developer.amazon.com/docs/alexaplus/smarthome/supported-matter-device-categories.html) | Not yet recorded |
| Apple Home | Lists Matter motion sensors among the categories supported by Apple Home in its [Matter accessory guidance](https://developer.apple.com/apple-home/works-with-apple-home/) | Not yet recorded |
| Samsung SmartThings | Provides a standard Matter [`motionSensor`](https://developer.smartthings.com/docs/edge-device-drivers/matter/defaults/motionSensor.html) handler in its Edge driver API | Not yet recorded |
| Home Assistant | Maps `OccupancySensing.Occupancy` to an occupancy binary sensor in its [Matter integration source](https://github.com/home-assistant/core/blob/dev/homeassistant/components/matter/binary_sensor.py) | Not yet recorded |

Published target availability does not imply that every controller and target combination has been commissioned successfully. Current images are uncertified development accessories, so an ecosystem may require a developer workflow or an explicit acknowledgement. [Dependencies and Firmware Layout](#dependencies-and-firmware-layout) records the identifiers and credentials used by published firmware.

Mark a controller as validated only with a reproducible hardware record that identifies the controller app and hub versions, ESP32 target, firmware identity, and results for commissioning, occupancy-state updates, and an automation trigger.

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
- upstream notice preserved for firmware compliance archives: [`NOTICE`](third_party/esp_matter/NOTICE)
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

## Matter-Specific Troubleshooting

Use [`SETUP.md`](../../../../docs/SETUP.md#direct-http-connectivity) for Direct HTTP, browser permission, address, and discovery failures. Use [`TUNING.md`](../../../../docs/TUNING.md#troubleshooting) for missing motion, false positives, calibration, packet health, placement, or unstable detection.

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

### Commissioning remains open or progresses slowly

Check the firmware-owned state before attributing the delay to the controller:

1. the serial log shows CSI services as `waiting for commissioning` before pairing completes
2. the commissioning window advertises BLE through the all-supported transport mode
3. the image was built from `sdkconfig.defaults` with commissionable device type enabled and device type `0x0107`
4. Wi-Fi CSI policy logs appear only after `Commissioning complete`
