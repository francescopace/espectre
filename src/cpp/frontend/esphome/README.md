# ESPectre ESPHome frontend

The ESPHome frontend turns ESPectre into a YAML component with Home Assistant entities. This guide covers Wi-Fi setup, entities, adoption, and builds. For installation, placement, and the first detection check, start with the [setup guide](../../../../docs/SETUP.md).

## Getting started

After flashing, configure Wi-Fi with one of these provisioning paths:

| Method | How |
|--------|-----|
| USB | Use Improv Serial with `./espectre provision --ssid MyNetwork` or any Improv Serial-compatible web flasher, such as the [ESPectre web flasher](https://espectre.dev/tools/flash/) |
| Captive portal | Connect to the `ESPectre Fallback` network and finish setup in the browser |

The examples keep the SSID, password, and BSSID out of YAML. After provisioning, Improv Serial returns a link to Device settings. On ESP32-S2, the logger and Improv Serial share the TinyUSB console.

Once Wi-Fi is set up, Home Assistant discovers the device through ESPHome.

If the device roams between access points, you can pin it to one without editing YAML or rebooting; see [mesh Wi-Fi instability](../../../../docs/TROUBLESHOOTING.md#mesh-wi-fi-instability). The pin survives restarts and stops roaming scans. If a new pin fails, the device goes back to the previous one. The requests are in [Wi-Fi scan and BSSID selection](../../../../docs/API.md#wi-fi-scan-and-bssid-selection).

Official images add the last three MAC bytes to the ESPHome hostname, for example `espectre-a1b2c3.local`. This lets one image serve multiple devices on the same network. The Home Assistant device and entity IDs use the same suffix.

## Integration surface

Declaring `espectre:` creates the [entities](#integrated-entities) automatically.

The device also runs ESPectre's Direct HTTP API next to the ESPHome API. Direct offers the same controls plus local management and raw CSI collection; find it with `./espectre devices --frontend esphome`. See the [API reference](../../../../docs/API.md) and [discovery reference](../../../../docs/DISCOVERY.md).

Direct is on by default. To keep only the ESPHome API and entities, turn it off:

```yaml
espectre:
  direct_api: false
```

This turns off Direct requests, live telemetry, raw CSI streaming, and ESPectre discovery. The ESPHome API and entities keep working.

Changes made through Direct show up in Home Assistant right away. Wi-Fi credentials, OTA, and API encryption stay under ESPHome's control. Renaming the device in ESPectre does not change the ESPHome hostname or entity IDs.

## Configuration surface

Sensing options go under `espectre:`, with the names, defaults, and ranges listed in [shared sensing options](../../../../docs/SDK.md#shared-sensing-options). The full YAML schema is in [__init__.py](components/espectre/__init__.py). See [tuning essentials](../../../../docs/TROUBLESHOOTING.md#tuning-essentials) for what to adjust.

Set the capture profile in YAML and rebuild to change it:

```yaml
espectre:
  csi_capture_profile: lltf # auto (default), lltf, or ht-vht
```

The profile cannot be changed from Home Assistant or Direct. The `wifi_raw` source needs `auto` or `lltf`. See [capture profiles](../../../../docs/CSI.md#capture-profiles) for what each value selects.

### Diagnostic telemetry

Diagnostic sensors update only when you press `Refresh Diagnostics`. Direct also reports performance, memory, and detector timing; see [API diagnostics](../../../../docs/API.md#diagnostics). To read the rates, see [check the sensing input](../../../../docs/TROUBLESHOOTING.md#check-the-sensing-input).

### Detection profile selection

```yaml
wifi:
  band_mode: AUTO  # ESP32-C5 only; optional because AUTO is the default

espectre:
  detection_algorithm: lightweight  # or high_accuracy
```

On ESP32-C5, `wifi.band_mode` accepts `2.4GHz`, `5GHz`, or `AUTO` (default). Other chips use 2.4 GHz.

`detection_algorithm` is only the starting profile. The `detector_select` entity changes it live and remembers the choice. Switching to Lightweight starts a calibration, shown by `calibration_active_sensor`. See [detection profile](../../../../docs/TROUBLESHOOTING.md#detection-profile) to choose.

## Entity customization

### Integrated entities

| Sensor config | Type | Default name | Description |
|---------------|------|--------------|-------------|
| `movement_sensor` | sensor | `Movement Score` | Current movement score (0.0–1.0), published every `evaluation_interval_ms` |
| `motion_sensor` | binary_sensor | `Motion Detected` | Edge-driven motion state; resets to idle when sensing stops or CSI restarts |
| `threshold_number` | number | `Threshold` | Runtime probability threshold (0.0–1.0) |
| `motion_on_hits_number` | number | `Motion On Hits` | Runtime motion-on debounce count (1–20) |
| `motion_off_hits_number` | number | `Motion Off Hits` | Runtime motion-off debounce count (1–20) |
| `detector_select` | select | `Detection Profile` | Runtime `lightweight` / `high_accuracy` selection |
| `traffic_generator_mode_select` | select | `CSI Traffic Source` | Runtime `ping` / `dns` (UDP) / `dns_tcp` / `wifi_raw` / `external` selection |
| `sensing_switch` | switch | `Sensing Enabled` | Enables or pauses sensing through the common command engine; publishes the runtime state at startup |
| `recalibrate_button` | button | `Recalibrate` | Starts runtime recalibration |
| `calibration_active_sensor` | binary_sensor | `Calibration Active` | Whether calibration is running (read-only) |
| `diagnostics_button` | button | `Refresh Diagnostics` | Publishes the latest cached diagnostic sample on demand |
| `generator_rate_sensor` | sensor | `Generator Rate` | Successful internal generator sends; zero in external mode |
| `traffic_rate_sensor` | sensor | `Traffic TX Rate` | Station network packets accepted by the driver |
| `traffic_rx_rate_sensor` | sensor | `Traffic RX Rate` | Station network packets delivered by the driver |
| `csi_callback_rate_sensor` | sensor | `CSI Callback Rate` | Raw CSI callback rate; diagnostic-only |
| `csi_accepted_rate_sensor` | sensor | `CSI Accepted Rate` | Raw identity-accepted capture rate before temporal admission; diagnostic-only |
| `csi_admitted_rate_sensor` | sensor | `CSI Admitted Rate` | Rate admitted to the detector's temporal grid; diagnostic-only |
| `csi_filtered_rate_sensor` | sensor | `CSI Filtered Rate` | Capture rejection rate; diagnostic-only |
| `csi_missing_rate_sensor` | sensor | `CSI Missing Slot Rate` | Missing detector slots per second; diagnostic-only |
| `csi_excess_rate_sensor` | sensor | `CSI Excess Rate` | Non-selected same-slot candidates per second, including candidates replaced by one nearer the slot center; diagnostic-only |
| `csi_stale_rate_sensor` | sensor | `CSI Stale Rate` | Packets discarded as stale per second; diagnostic-only |
| `csi_out_of_order_rate_sensor` | sensor | `CSI Out-of-Order Rate` | Duplicate or backward-timestamp packets discarded per second; diagnostic-only |
| `csi_occupancy_sensor` | sensor | `CSI Temporal Occupancy` | Valid-slot occupancy of the active detector window; diagnostic-only |
| `wifi_channel_sensor` | sensor | `WiFi Channel` | Current associated Wi-Fi channel; diagnostic-only, with zero decimal places and `measurement` state class |
| `wifi_rssi_sensor` | sensor | `WiFi RSSI` | Current associated Wi-Fi RSSI; diagnostic-only |

All entities support standard ESPHome options such as `name`, `internal`, `icon`, and `disabled_by_default`. The `movement_sensor` also supports ESPHome [sensor filters](https://esphome.io/components/sensor/#sensor-filters):

```yaml
espectre:
  movement_sensor:
    name: "Living Room Movement"
    internal: true
    icon: "mdi:sine-wave"
    filters:
      - multiply: 100
      - clamp:
          min_value: 0
          max_value: 100
      - round: 1
  motion_sensor:
    name: "Living Room Motion"
    icon: "mdi:motion-sensor"
  threshold_number:
    name: "Living Room Threshold"
```

Use `internal: true` on `movement_sensor` when you want to keep the binary motion entity for automations without publishing the raw score to Home Assistant.

## Home Assistant integration

Once the device is flashed and connected to Wi-Fi:

1. Home Assistant discovers it through ESPHome
2. Go to **Settings** -> **Devices & Services** -> **ESPHome**
3. Configure the discovered device
4. The default entities are added automatically

How entities update:

- Writable entities always show the value the device is using, even after a rejected change.
- Movement Score updates at every evaluation (250 ms by default). Motion Detected updates only when the state changes.
- Threshold also updates after calibration and when Lightweight lowers it automatically.

Movement Score produces a lot of history. To keep the recorder small, exclude `sensor.*_movement_score` rather than slowing the evaluation interval.

To manage configuration and OTA updates, adopt the device in ESPHome Device Builder. The adopted configuration builds from the GitHub `main` branch and reports version `0.0.0-main`. To stay on a release, use a prebuilt image instead: download `espectre-esphome-<channel-or-version>-<chip>-ota.bin` from GitHub Releases and upload it:

```bash
./espectre esphome flash --chip c6 --device espectre-<mac-suffix>.local --firmware espectre-esphome-3.0.0-esp32c6-ota.bin
```

An older installation without the MAC suffix is still at `espectre.local` for this upload. Afterwards, use `espectre-<mac-suffix>.local`.

### Dashboard examples

[home-assistant-dashboard.yaml](examples/home-assistant-dashboard.yaml) provides motion, movement score, history, controls, and diagnostics.

![ESPectre Home Assistant dashboard](../../../../docs/web/assets/images/guides/home-assistant-dashboard.png)

*Home Assistant dashboard with motion state, movement score, movement-versus-threshold history, detection profile, threshold, calibration, and diagnostics.*

To import a dashboard:

1. Go to **Settings** -> **Dashboards** -> **Add Dashboard**
2. Open the dashboard and choose **Edit**
3. Open the raw configuration editor
4. Replace the default content with the YAML from the example file
5. Save the dashboard

Official images add the MAC suffix to entity IDs, so update the IDs in the dashboard to match your device (and the `espectre` prefix, if you renamed it). Check the exact IDs in Home Assistant: a name clash can add a suffix such as `_2`.

## Traffic configuration

Traffic settings go under `espectre:`. See [traffic sources](../../../../docs/CSI.md#traffic-sources) for how they work.

### Internal traffic generator

```yaml
espectre:
  csi_target_pps: 100
  traffic_generator_mode: ping
  traffic_generator_target_ip: "" # Empty uses the gateway; set an IPv4 address to override
```

The `traffic_generator_mode_select` entity changes the source at runtime, including `external`, and the device remembers it. ESP32-C6 does not offer `wifi_raw`; see [compatibility limits](../../../../docs/CSI.md#compatibility-limits).

### External traffic mode

On Home Assistant OS, the ESPectre Traffic Generator add-on can send the traffic and switch devices to external mode from its panel. See the [web guide](https://espectre.dev/guides/home-assistant/#ha-traffic-generator) and the [add-on documentation](../../../../tools/ha_traffic_generator_addon/DOCS.md).

To use external traffic in YAML:

```yaml
espectre:
  csi_target_pps: 100
  traffic_generator_mode: external
  csi_traffic_multicast_group: "239.255.0.1"
```

Raw collection needs Direct; see the [`collect` command](../../../../docs/CLI.md#collect). Collection switches the device to external mode, and it stays there afterwards.

## Build and consumption

The `release`, `preview`, and `develop` channels publish a full-flash image and an OTA image for each chip. Both detection profiles are included; Lightweight is the starting one.

The example YAML and normal Device Builder builds are not signed, and a device with an official image rejects them over OTA. Install the first one over USB (see [official images and personal builds](../../../../docs/SETUP.md#official-images-and-personal-builds)), or enable ESPHome's `signed_ota_verification` with your own key.

### As an ESPHome external component

Each maintained chip has one canonical example. By default it includes `espectre-source-github.yaml`, which resolves the component from GitHub:

```yaml
external_components:
  - source:
      type: git
      url: https://github.com/francescopace/espectre
      path: src/cpp/frontend/esphome/components
    components: [espectre]
```

Omit `ref` to follow the default branch, or set a tag or commit to pin the component. An empty `ref: ""` is invalid.

`esphome.project.version` sets the firmware version (up to 31 bytes). If you omit it, a numeric `ref` such as `3.0.0-rc1` is used. The SDK version is separate; see [versioning](../../../../docs/SDK.md#versioning).

Repository development selects `espectre-source-local.yaml` instead, which resolves the same component from the local checkout:

```yaml
external_components:
  - source:
      type: local
      path: ../components
    components: [espectre]
```

### Repository CLI

See the [CLI reference](../../../../docs/CLI.md) for all options.

```bash
./espectre esphome build --chip c6 --clean
./espectre esphome flash --chip c6
./espectre esphome config --chip c6
./espectre esphome monitor --chip c6 --device /dev/cu.usbmodemXXXX
```

On Windows, use `.\espectre.cmd esphome ...` from the repository root and pass a COM port such as `COM5` to `--device` when serial access is needed.

The CLI builds the chip's example YAML with the component from your local checkout, and sets the version from `git describe`. Use `flash` to upload without building and `monitor` for logs.

## Hardware and packaging notes

### Build toolchain

The ESPHome examples use the native ESP-IDF backend from the ESPHome version pinned in [`requirements.txt`](../../../../requirements.txt). [`__init__.py`](components/espectre/__init__.py) registers this directory with ESP-IDF's component manager. Its [`CMakeLists.txt`](components/espectre/CMakeLists.txt) compiles the selected SDK through the canonical build definition at [`CMakeLists.txt`](../../CMakeLists.txt). No toolchain override or separate library is needed.

### Automatic SDK configuration

The component sets the CSI, Wi-Fi, buffer, and lwIP options it needs in [__init__.py](components/espectre/__init__.py), and sends runtime logs to the ESPHome logger. Put board-specific overrides under `esp32.framework.sdkconfig_options`.

The examples use Improv Serial instead of BLE provisioning, which saves flash and memory and avoids radio contention. Keep it that way unless you need BLE.

### Flash size and partitions

The firmware fits in 4 MB of flash with OTA and uses the default partition table. You can override it with `esp32.partitions` if needed.

## ESPHome-specific troubleshooting

For sensing and connection problems, see the [troubleshooting guide](../../../../docs/TROUBLESHOOTING.md). If the board does not enter download mode, see [web flash](../../../../docs/SETUP.md#web-flash-no-coding-required).

### Bluetooth proxy and CSI occupancy

Bluetooth and Wi-Fi share the radio, so a Bluetooth proxy can lower CSI occupancy. If it does, compare with Bluetooth disabled first. For a proxy that only forwards advertisements, these experimental settings had the smallest impact in our tests. Merge them into your existing configuration:

```yaml
esp32:
  cpu_frequency: 240MHz
  framework:
    type: esp-idf
    sdkconfig_options:
      CONFIG_ESP_COEX_SW_COEXIST_ENABLE: n

bluetooth_proxy:
  active: false

esp32_ble_tracker:
  software_coexistence: false
  scan_parameters:
    interval: 100ms
    window: 5ms
    duration: 5min
    active: false
    continuous: true

espectre:
  traffic_generator_mode: wifi_raw
```

- The `sdkconfig_options` line is required. With ESPHome 2026.8.2 and ESP-IDF 5.5.5, `software_coexistence: false` alone left ESP-IDF coexistence enabled, even after a clean build.
- Run **Clean Build Files**, rebuild, and install. The generated SDK configuration must contain `# CONFIG_ESP_COEX_SW_COEXIST_ENABLE is not set`.
- `bluetooth_proxy.active: false` turns off active GATT connections but still forwards advertisements. `scan_parameters.active: false` selects passive scanning.
- Short scan windows receive fewer advertisements. Check that the BLE devices you rely on still report.
- Active GATT connections were not tested with these settings. For `wifi_raw` limits, see [compatibility limits](../../../../docs/CSI.md#compatibility-limits).

The tests for [issue #165](https://github.com/francescopace/espectre/issues/165) used one ESP32-S3 at 240 MHz, a fixed position and access point, and `wifi_raw` at 100 pps. Each run lasted 60–120 seconds, and the first 15 seconds were excluded:

| Scan window / interval | Scan type | Tracker coexistence | ESP-IDF coexistence | Mean CSI occupancy per run |
|------------------------|-----------|---------------------|---------------------|----------------------------|
| BLE disabled | — | — | — | 94.1%, 88.8%, 94.0% |
| 320 / 320 ms | Active | On | On | 52.3% |
| 30 / 320 ms | Passive | On | On | 53.2% |
| 10 / 100 ms | Passive | On | On | 53.3%, 54.4% |
| 5 / 100 ms | Passive | On | On | 53.6%, 51.9% |
| 10 / 100 ms | Active | On | On | 54.0% |
| 10 / 100 ms | Passive | Off | On | 56.7% |
| 5 / 100 ms | Passive | Off | On | 56.5% |
| 30 / 320 ms | Passive | Off | Off | 84.4% |
| 10 / 100 ms | Passive | Off | Off | 88.5%, 87.6% |
| 5 / 100 ms | Passive | Off | Off | 93.4%, 91.1%, 92.6% |

In the final two-minute run with the 5 ms window, the proxy still received 433 advertisements from 26 devices. The 10 ms window received about twice as many, with lower occupancy. These are short tests on one board and network, not a general compatibility result.

### View logs

The runtime logs an `IDLE | csi:` or `MOTION | csi:` status line every second. It appears on USB serial and in `esphome logs` when the `espectre.runtime` tag allows INFO messages.

```bash
esphome logs <your-config>.yaml
esphome logs <your-config>.yaml --device espectre.local
./espectre monitor --port /dev/cu.usbmodem*
```

## Implementation map

The frontend uses public SDK headers. See [building against an SDK bundle](../../../../docs/CLI.md#building-against-an-sdk-bundle) for builds against an extracted SDK bundle and [frontend layer](../../../../docs/ARCHITECTURE.md#srccppfrontend) for source groups and ownership.

This section is for component maintainers. `sensing_schema.py` is generated from the SDK schema; CMake checks that it matches the selected SDK before compiling.

- [`__init__.py`](components/espectre/__init__.py): YAML schema, validation, codegen, native ESP-IDF component registration, and ESPHome build flags
- [`CMakeLists.txt`](components/espectre/CMakeLists.txt): native ESP-IDF bridge to the canonical shared SDK build definition
- [`espectre.cpp`](components/espectre/espectre.cpp), [`espectre.h`](components/espectre/espectre.h): ESPHome adapter over the shared runtime frontend controller
- [`sensor_publisher.cpp`](components/espectre/sensor_publisher.cpp): movement and motion publishing
- [`threshold_number.cpp`](components/espectre/threshold_number.cpp): runtime threshold control
- [`motion_hits_number.cpp`](components/espectre/motion_hits_number.cpp): runtime motion-hit debounce control
- [`detector_select.cpp`](components/espectre/detector_select.cpp): persisted runtime detector selection
- [`sensing_switch.cpp`](components/espectre/sensing_switch.cpp): sensing lifecycle control
- [`recalibrate_button.cpp`](components/espectre/recalibrate_button.cpp): runtime recalibration action
- [`traffic_mode_select.cpp`](components/espectre/traffic_mode_select.cpp): runtime traffic generator mode control
- [`examples/`](examples/): production and local-development configurations for ESP32, ESP32-S2, ESP32-S3, ESP32-C3, ESP32-C5, and ESP32-C6, plus the Home Assistant dashboard
