# ESPectre Matter Frontend

ESPectre's Matter firmware publishes the standard occupancy sensor device type. A controller that implements that type can consume it without an ESPectre-specific integration.

This guide covers commissioning, the Matter and Direct controls, controller validation, and local builds. Start with [SETUP.md](../../../../docs/SETUP.md) for supported boards, browser installation, placement, and the first sensing check.

## Getting Started

### Browser-Flashed Firmware

The `release`, `preview`, and `develop` channels publish full-flash Matter images. OTA is not supported.

ESP32-S2 is excluded because it has no Bluetooth radio; this frontend commissions Wi-Fi over BLE.

After flashing a Matter image:

1. Wait for the device to boot, and use the setup codes read by the installer.
2. Commission it with a Matter controller that supports BLE commissioning.
3. Check that the controller receives occupancy updates.

To retrieve the codes later, reconnect to the installer and choose **Matter QR code**, or run `./espectre matter qr --chip <chip> --port <port>`.

The first boot generates per-device onboarding codes in the `matter_factory` partition. Browser and CLI flashes preserve that partition unless you choose a full erase, which creates new codes on the next boot.

### Local ESP-IDF Workflow

Complete the prerequisites in [CLI.md](../../../../docs/CLI.md#local-build-prerequisites), then run:

```bash
./espectre matter build --chip c3
./espectre matter flash --chip c3 --port /dev/cu.usbmodemXXXX
./espectre monitor --chip c3 --frontend matter --port /dev/cu.usbmodemXXXX
```

The flash command prints the onboarding codes captured from the first boot. To retrieve the persisted codes later, run `./espectre matter qr --chip c3 --port /dev/cu.usbmodemXXXX`.

The first build downloads and compiles the managed `esp_matter` dependency; no manual clone is required. See [CLI.md](../../../../docs/CLI.md#native-and-matter) for Windows commands, build backends, and cache controls.

### Commissioning Window Behavior

An uncommissioned device opens a 300-second window with BLE discovery. Removing the last fabric stops ESPectre services and schedules a restart after two seconds to restore BLE. The device reuses its persisted onboarding codes; the restart is canceled if a fabric exists when the timer expires. Pairing and fabric-management screens depend on the controller.

Improv Serial supports firmware discovery and retrieval of Matter codes, but does not accept Wi-Fi provisioning commands. Matter owns network commissioning. Serial logs report commissioning completion or fail-safe expiration.

## Exposed Matter Surface

| Feature | Matter mapping | Type | Access |
|---------|----------------|------|--------|
| Motion detected | `OccupancySensing` occupancy bitmap | bitmap | read-only |

## What You Can Configure Today

Matter exposes read-only occupancy. To change detector settings, open Device settings or Monitor through Direct HTTP after commissioning. Run `./espectre devices --frontend matter` to find the endpoint.

Direct provides sensing controls, diagnostics, BSSID selection, Basic Information `NodeLabel` editing, peer discovery, and raw CSI collection. See [API.md](../../../../docs/API.md) for the resource contract and [DISCOVERY.md](../../../../docs/DISCOVERY.md) for discovery.

Matter still owns Wi-Fi credentials, commissioning, and fabric access. Direct cannot reset the Wi-Fi configuration or replace the read-only Matter occupancy attribute. It remains available after commissioning, while `_matterc` is advertised only to Matter controllers during an open commissioning window.

ESPectre stores a BSSID preference separately from Matter credentials and binds it to the commissioned SSID. A change reconnects without rebooting; failure restores the previous preference. Pending changes resume after restart. A confirmed pin applies only while the SSID matches and remains dormant on another network until explicitly cleared. See [API.md](../../../../docs/API.md#wi-fi-scan-and-bssid-selection) for requests and [CSI.md](../../../../docs/CSI.md#wi-fi-and-capture-lifecycle) for capture restart behavior.

Set initial sensing options through the shared ESP-IDF menu described in [SDK.md](../../../../docs/SDK.md#shared-sensing-options). Direct changes to the detector and traffic selections persist across reboot. Use [TROUBLESHOOTING.md](../../../../docs/TROUBLESHOOTING.md#tuning-essentials) for tuning and [CLI.md](../../../../docs/CLI.md#collect) for collection.

## Targets and Validation

Hardware smoke results are recorded for every published target. The generated snapshots define the exact scope: [ESP32.md](../../../../docs/performance/ESP32.md), [ESP32-S3.md](../../../../docs/performance/ESP32-S3.md), [ESP32-C3.md](../../../../docs/performance/ESP32-C3.md), [ESP32-C5.md](../../../../docs/performance/ESP32-C5.md), and [ESP32-C6.md](../../../../docs/performance/ESP32-C6.md). Controller commissioning coverage is separate.

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

## Commissioning and Runtime Ownership

`esp-matter` owns Wi-Fi, commissioning, and fabrics. ESPectre defers runtime allocation until a fabric exists, keeping heap available for commissioning. CSI, Direct HTTP, and ESPectre discovery start after commissioning, with a 10-second grace for a newly commissioned device. An already commissioned boot skips that grace.

Commissioning and fabric events are passed to the ESPectre loop; the CHIP task never reconfigures the sensing runtime directly. Occupancy changes are scheduled onto the CHIP work queue. Direct serialization runs after CSI processing, and discovery shares the Matter-owned mDNS responder. See [app_main.cpp](app/main/app_main.cpp) for the startup sequence and [SDK.md](../../../../docs/SDK.md#threading) for the shared runtime contract.

[sdkconfig.defaults](app/sdkconfig.defaults) contains the Matter-specific endpoint, queue, and network memory budgets. BLE is used only for commissioning and released afterward, which is why removing the last fabric requires a restart.

## Implementation Map

This map is for frontend maintainers; it is not required for commissioning an existing image.

- [`matter_frontend.cpp`](espectre/matter_frontend.cpp), [`matter_frontend.h`](espectre/matter_frontend.h): frontend adapter over the shared runtime frontend controller
- [`matter_surface.h`](espectre/matter_surface.h): cluster and attribute IDs plus Matter mapping helpers
- [`matter_bindings.h`](espectre/matter_bindings.h): boundary between the adapter and the Matter transport layer
- [`app/`](app/): standalone ESP-IDF firmware app
- [`app_main.cpp`](app/main/app_main.cpp): Matter node setup, endpoint creation, commissioning window behavior, and startup order
- [matter_commissioning_data.cpp](app/main/matter_commissioning_data.cpp): random per-device onboarding data and `matter_factory` persistence
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

## OTA

Update Matter over USB using a full firmware image. The frontend implements neither a Matter OTA requestor nor Native's HTTPS OTA service.

## Matter-Specific Troubleshooting

Use [TROUBLESHOOTING.md](../../../../docs/TROUBLESHOOTING.md) for browser connectivity and sensing problems.

### The device does not appear for commissioning

Check these first:

1. the controller supports BLE commissioning
2. the device is uncommissioned or the previous fabric was removed
3. serial logs show the Matter firmware started successfully

### Commissioning fails and times out

The firmware logs fail-safe expiration events. Power-cycle the board, move it close to the controller, and retry BLE commissioning. If it still fails, record the controller, hub, firmware version, and serial failure log; the [controller matrix](#matter-controller-compatibility) shows the current validation coverage.

### Commissioning remains open or progresses slowly

Check the firmware-owned state before attributing the delay to the controller:

1. the serial log shows CSI services as `waiting for commissioning` before pairing completes
2. the commissioning window advertises BLE through the all-supported transport mode
3. the image was built from `sdkconfig.defaults` with commissionable device type enabled and device type `0x0107`
4. Wi-Fi CSI policy logs appear only after `Commissioning complete`
