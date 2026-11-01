# ESPectre Matter frontend

ESPectre's Matter firmware publishes the standard occupancy sensor device type. A controller that implements that type can consume it without an ESPectre-specific integration.

This guide covers commissioning, the Matter and Direct controls, controller validation, and local builds. Start with the [setup guide](../../../../docs/SETUP.md) for supported boards, browser installation, placement, and the first sensing check.

## Getting started

### Browser-flashed firmware

The `release`, `preview`, and `develop` channels publish full-flash Matter images. OTA is not supported.

ESP32-S2 is excluded because it has no Bluetooth radio; this frontend commissions Wi-Fi over BLE.

After flashing a Matter image:

1. Wait for the device to boot, and use the setup codes read by the installer.
2. Commission it with a Matter controller that supports BLE commissioning.
3. Check that the controller receives occupancy updates.

To retrieve the codes later, reconnect to the installer and choose **Matter QR code**, or run `./espectre matter qr --chip <chip> --port <port>`.

The first boot generates per-device onboarding codes in the `matter_factory` partition. Browser and CLI flashes preserve that partition unless you choose a full erase, which creates new codes on the next boot.

### Local ESP-IDF workflow

Complete the prerequisites in [local build prerequisites](../../../../docs/CLI.md#local-build-prerequisites), then run:

```bash
./espectre matter build --chip c3
./espectre matter flash --chip c3 --port /dev/cu.usbmodemXXXX
./espectre monitor --chip c3 --frontend matter --port /dev/cu.usbmodemXXXX
```

The flash command prints the onboarding codes captured from the first boot. To retrieve the persisted codes later, run `./espectre matter qr --chip c3 --port /dev/cu.usbmodemXXXX`.

The first build downloads and compiles the managed `esp_matter` dependency; no manual clone is required. See [`native` and `matter` commands](../../../../docs/CLI.md#native-and-matter) for Windows commands, build backends, and cache controls.

Per-chip settings live in `app/sdkconfig.defaults.<idf_target>`, which the CLI loads after the shared defaults. CPU frequency is explicit for every supported chip: 240 MHz on ESP32, ESP32-S3, and ESP32-C5, and 160 MHz on ESP32-C3 and ESP32-C6. Add future chip-specific overrides to these files; ESP32-S2 is not supported by this frontend.

### Commissioning window behavior

An uncommissioned device opens a 300-second window with BLE discovery. Removing the last fabric stops ESPectre services and schedules a restart after two seconds to restore BLE. The device reuses its persisted onboarding codes; the restart is canceled if a fabric exists when the timer expires. Pairing and fabric-management screens depend on the controller.

Improv Serial supports firmware discovery and retrieval of Matter codes, but does not accept Wi-Fi provisioning commands. Matter owns network commissioning. Serial logs report commissioning completion or fail-safe expiration.

## Exposed Matter surface

| Feature | Matter mapping | Type | Access |
|---------|----------------|------|--------|
| Motion detected | `OccupancySensing` occupancy bitmap | bitmap | read-only |
| Occupancy hold time | `OccupancySensing` `HoldTime` and `HoldTimeLimits` | seconds, 1-300 | read-write |

## What you can configure today

Matter exposes read-only occupancy, which means motion: a person sitting still can read as unoccupied. Stationary presence will not change this attribute when it ships; it will be reported separately. Occupancy clears 1 second after the detector reports idle, which already waits for `motion_off_hits`. A controller can set a longer `HoldTime`, up to 300 seconds, and the device keeps it across reboots. To change detector or traffic settings, open [Monitor](https://espectre.dev/tools/monitor/) through Direct HTTP after commissioning. Run `./espectre devices --frontend matter` to find the endpoint.

Direct provides sensing controls, diagnostics, BSSID selection, Basic Information `NodeLabel` editing, peer discovery, and raw CSI collection. See the [API reference](../../../../docs/API.md) for the resource contract and the [discovery reference](../../../../docs/DISCOVERY.md) for discovery.

Matter still owns Wi-Fi credentials, commissioning, and fabric access. Direct cannot reset the Wi-Fi configuration or replace the read-only Matter occupancy attribute. It remains available after commissioning, while `_matterc` is advertised only to Matter controllers during an open commissioning window.

You can pin the device to one access point. The pin is tied to the commissioned network: it is ignored on another SSID until you clear it. Changing it reconnects without a reboot, and a failed change restores the previous pin. See [Wi-Fi scan and BSSID selection](../../../../docs/API.md#wi-fi-scan-and-bssid-selection) for requests and [Wi-Fi and capture lifecycle](../../../../docs/CSI.md#wi-fi-and-capture-lifecycle) for capture restart behavior.

Set build-time defaults in menuconfig; see [shared sensing options](../../../../docs/SDK.md#shared-sensing-options). Detector, traffic, and packet spacing changes made through Direct survive a reboot. See [tuning essentials](../../../../docs/TROUBLESHOOTING.md#tuning-essentials).

## Targets and validation

Hardware smoke results are recorded for every published target. The generated snapshots define the exact scope: [ESP32.md](../../../../docs/performance/ESP32.md), [ESP32-S3.md](../../../../docs/performance/ESP32-S3.md), [ESP32-C3.md](../../../../docs/performance/ESP32-C3.md), [ESP32-C5.md](../../../../docs/performance/ESP32-C5.md), and [ESP32-C6.md](../../../../docs/performance/ESP32-C6.md). Controller commissioning coverage is separate.

### Matter controller compatibility

The table shows what each ecosystem documents and what we have actually tested with ESPectre. Vendor support for occupancy sensors does not guarantee that ESPectre works there. Vendor documentation was last checked on 2026-08-26.

| Controller ecosystem | Vendor-documented Matter support | ESPectre validation |
| --- | --- | --- |
| Google Home | Lists the Occupancy Sensor device type (`0x0107`) and Occupancy Sensing cluster (`0x0406`) in its [supported-device matrix](https://developers.home.google.com/matter/supported-devices) | Validated on ESP32-S3; needs a [Developer Console project](#google-home-with-community-images). See the [validation records](#controller-validation-records) |
| Amazon Alexa | Maps a Matter motion detector using Occupancy Sensing to `Alexa.MotionSensor` in its [supported-category matrix](https://developer.amazon.com/docs/alexaplus/smarthome/supported-matter-device-categories.html) | Not yet recorded |
| Apple Home | Lists Matter motion sensors among the categories supported by Apple Home in its [Matter accessory guidance](https://developer.apple.com/apple-home/works-with-apple-home/) | Validated on ESP32-S3; see the [validation records](#controller-validation-records) |
| Samsung SmartThings | Provides a standard Matter [`motionSensor`](https://developer.smartthings.com/docs/edge-device-drivers/matter/defaults/motionSensor.html) handler in its Edge driver API | Not yet recorded |
| Home Assistant | Maps `OccupancySensing.Occupancy` to an occupancy binary sensor in its [Matter integration source](https://github.com/home-assistant/core/blob/dev/homeassistant/components/matter/binary_sensor.py) | Validated on ESP32-S3; see the [validation records](#controller-validation-records) |

Not every controller and chip combination has been tested. Current images are uncertified development devices, so some ecosystems ask you to confirm before adding them, and Google Home needs a one-time setup; see [Google Home with community images](#google-home-with-community-images). [Dependencies and Firmware Layout](#dependencies-and-firmware-layout) records the identifiers and credentials used by published firmware.

Mark a controller as validated only with a reproducible hardware record that identifies the controller app and hub versions, ESP32 target, firmware identity, and results for commissioning, occupancy-state updates, and an automation trigger.

Record each session with these fields:

| Field | Value |
| --- | --- |
| Controller and app version | |
| Hub or border router model and firmware | |
| ESP32 target and firmware identity | |
| Commissioning result and time to complete | |
| Occupancy updates seen in the controller | |
| Automation trigger fired | |
| Limits, prompts, or workarounds | |

### Controller validation records

Tested on 2026-10-31 with an ESP32-S3 on a 2.4 GHz Wi-Fi network. The firmware was a 3.1.0 development build that includes `HoldTime`. Each controller passed commissioning, occupancy updates in both directions, and an automation triggered by occupancy.

| Field | Google Home | Home Assistant | Apple Home |
| --- | --- | --- | --- |
| Controller and app version | Google Home app, version not recorded | Home Assistant OS 18.4, Core 2026.10.4, Companion app 2026.4.0 on iPhone 15 | Home app on iPhone 15, iOS 27.0.1 |
| Hub or border router model and firmware | Not recorded | Matter Server add-on, version not recorded | Not recorded |
| Commissioning result | Passed from the QR code as the first controller | Passed with a pairing code from Google Home | Passed with a pairing code from Google Home, and earlier from the QR code as the first controller |
| Occupancy updates seen in the controller | Occupied and unoccupied | Occupied and unoccupied | Occupied and unoccupied |
| Automation trigger fired | Yes | Yes | Yes |
| Limits, prompts, or workarounds | Needs the Developer Console project. | None | Asks to confirm an uncertified accessory. Removing the accessory in Apple Home also removed every other controller's fabric, so the device restarted for commissioning |

The device holds up to five Matter fabrics. Google Home used two of them, while Apple Home and Home Assistant used one each.

### Google Home with community images

Google Home adds an uncertified device only for members of a Google Home Developer Console project that has a Matter integration for its vendor and product ID. Published images use the development IDs, so set this up once per Google account:

1. Open the [Google Home Developer Console](https://console.home.google.com/) with the account you use in the Google Home app, and create a project.
2. Add a Matter integration with vendor ID `0xFFF1` and product ID `0x8000`.
3. Commission the device from the Google Home app with the same account, using its QR code.

Other people in the home need to be members of the same project. Apple Home asks you to confirm an uncertified accessory, and Home Assistant needs no extra step. Only a certified product removes this requirement; see [Google's pairing rules](https://developers.home.google.com/matter/integration/pair) and [building a certified product](#building-a-certified-product).

## Commissioning and runtime ownership

`esp-matter` owns Wi-Fi, commissioning, and fabrics. ESPectre defers runtime allocation until a fabric exists, keeping heap available for commissioning. CSI, Direct HTTP, and ESPectre discovery start after commissioning, with a 10-second grace for a newly commissioned device. An already commissioned boot skips that grace.

Commissioning and fabric events are passed to the ESPectre loop; the CHIP task never reconfigures the sensing runtime directly. Occupancy changes are scheduled onto the CHIP work queue. Direct serialization runs after CSI processing, and discovery shares the Matter-owned mDNS responder. See [app_main.cpp](app/main/app_main.cpp) for the startup sequence and [SDK threading](../../../../docs/SDK.md#threading) for the shared runtime contract.

[sdkconfig.defaults](app/sdkconfig.defaults) contains the Matter-specific endpoint, queue, and network memory budgets. BLE is used only for commissioning and released afterward, which is why removing the last fabric requires a restart.

## Implementation map

The frontend uses public SDK headers. See [building against an SDK bundle](../../../../docs/CLI.md#building-against-an-sdk-bundle) for builds against an extracted SDK bundle and [frontend layer](../../../../docs/ARCHITECTURE.md#srccppfrontend) for source groups and ownership.

Native and Matter pin `improv/improv` to `1.2.7` from the ESP Component Registry in their frontend manifests. The shared Improv Serial service uses this dependency, which the SDK excludes.

This map is for frontend maintainers; it is not required for commissioning an existing image.

- [`matter_frontend.cpp`](espectre/matter_frontend.cpp), [`matter_frontend.h`](espectre/matter_frontend.h): frontend adapter over the shared runtime frontend controller
- [`matter_surface.h`](espectre/matter_surface.h): cluster and attribute IDs plus Matter mapping helpers
- [`matter_bindings.h`](espectre/matter_bindings.h): boundary between the adapter and the Matter transport layer
- [`app/`](app/): standalone ESP-IDF firmware app
- [`app_main.cpp`](app/main/app_main.cpp): Matter node setup, endpoint creation, commissioning window behavior, and startup order
- [matter_commissioning_data.cpp](app/main/matter_commissioning_data.cpp): random per-device onboarding data and `matter_factory` persistence
- [matter_device_instance_info.cpp](app/main/matter_device_instance_info.cpp): MAC-derived serial number for builds without factory data
- [`idf_component.yml`](app/main/idf_component.yml): `esp_matter` dependency declaration

## Dependencies and firmware layout

- firmware app: [`app/`](app/)
- dependency manager: ESP-IDF Component Manager
- declared external dependency: `espressif/esp_matter`
- upstream notice preserved for firmware compliance archives: [`NOTICE`](third_party/esp_matter/NOTICE)
- Matter device type: occupancy sensor (`0x0107`)
- development VID/PID: `0xFFF1` / `0x8000`
- partition layout: [`partitions.csv`](app/partitions.csv)
- defaults: [`sdkconfig.defaults`](app/sdkconfig.defaults)

Published images use development IDs and the SDK's example attestation credentials. [Building a certified product](#building-a-certified-product) covers what changes in a product build.

## Building a certified product

Published images are development devices. A certified Matter product also needs a vendor ID, per-device credentials written in the factory, and certification tests, which only the company that builds and sells the hardware can provide. The firmware is ready for that product build; the integrator does the rest.

| Area | ESPectre firmware | Integrator |
| --- | --- | --- |
| Vendor and product ID | Development `0xFFF1` / `0x8000`, set by `CONFIG_DEVICE_VENDOR_ID` and `CONFIG_DEVICE_PRODUCT_ID` | A Connectivity Standards Alliance vendor ID and a product ID |
| Onboarding data | Generated on first boot in `matter_factory`; read from the `fctry` partition in a factory-data build | Per-device passcode verifier and discriminator, with the QR code printed on the product |
| Device attestation | The SDK's example DAC; a factory-data build reads the DAC from `fctry` or `esp_secure_cert` | A unique DAC per device under a PAA in the CSA trust store, and the product's certification declaration |
| Serial number | Derived from the base MAC; read from `fctry` in a factory-data build | A unique serial number per device |
| Software version | `SoftwareVersion` and its string come from the release identity at build time | Its own numbering only if it ships a modified firmware |
| Test hooks | Test Event Triggers are off | Turn them on with its own key for certification tests, and off again for shipping |
| Firmware protection | Secure Boot and flash encryption are off | Secure Boot and flash encryption as part of manufacturing, to protect the DAC key |
| Firmware updates | USB only | An update method, usually a Matter OTA Requestor with two app slots; C5 and C6 images need 8 MB of flash for that |
| Certification | Hardware smoke results | Certification tests at an authorized lab and the product's entry in the Distributed Compliance Ledger |

### Factory-data builds

1. Copy [sdkconfig.factory.example](app/sdkconfig.factory.example), set the vendor and product ID, and choose where the DAC lives.
2. Generate each device's factory partition with Espressif's [Matter manufacturing tool](https://github.com/espressif/esp-matter-mfg-tool), and write it to `fctry` at `0x13000`. A DAC kept in `esp_secure_cert` goes to `0x11000`.
3. Build with the copy listed after the shared defaults:

   ```bash
   SDKCONFIG_DEFAULTS="sdkconfig.defaults;sdkconfig.defaults.esp32s3;/path/to/product.defaults" ./espectre matter build --chip s3 --clean
   ```

The factory partition holds the passcode verifier, not the passcode, so the device cannot show its QR code. The web installer, `./espectre matter flash`, and `./espectre matter qr` report the codes as unavailable; use the codes the manufacturing tool writes for the product label. The certification declaration given to the tool must name the same vendor ID as the DAC, or commissioning fails device attestation.

## OTA

Update Matter over USB using a full firmware image. The frontend implements neither a Matter OTA requestor nor Native's HTTPS OTA service, and the firmware does not check an application signature. A product that needs updates in the field adds them; see [building a certified product](#building-a-certified-product). See [official images and personal builds](../../../../docs/SETUP.md#official-images-and-personal-builds) for catalog verification and the shared USB workflow.

## Matter-specific troubleshooting

Use the [troubleshooting guide](https://espectre.dev/guides/troubleshooting/) for browser connectivity and sensing problems, and the [troubleshooting reference](../../../../docs/TROUBLESHOOTING.md) for the device log and configuration options.

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
