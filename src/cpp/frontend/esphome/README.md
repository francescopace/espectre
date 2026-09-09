# ESPectre ESPHome Frontend

The ESPHome frontend exposes ESPectre as a YAML component with Home Assistant entities. This guide covers provisioning, entity configuration, adoption, and builds. For installation, placement, and the first sensing check, start with [SETUP.md](../../../../docs/SETUP.md).

## Getting Started

After flashing, configure Wi-Fi with one of these provisioning paths:

| Method | How |
|--------|-----|
| USB | Use Improv Serial with `./espectre provision --ssid MyNetwork` or any Improv Serial-compatible web flasher, such as the [ESPectre web flasher](https://espectre.dev/tools/flash/) |
| Captive portal | Connect to the `ESPectre Fallback` network and finish setup in the browser |

The maintained examples enable Improv Serial, return a Device settings URL after provisioning, and keep SSID, password, and BSSID out of YAML.

For an unstable access-point association, follow [TROUBLESHOOTING.md](../../../../docs/TROUBLESHOOTING.md#mesh-wi-fi-instability). ESPectre stores a BSSID pin separately from ESPHome's Wi-Fi credentials and applies it without rewriting YAML or rebooting. A failed update reconnects once with the previous pin. The request contract is in [API.md](../../../../docs/API.md#wi-fi-scan-and-bssid-selection), and the capture lifecycle is in [CSI.md](../../../../docs/CSI.md#wi-fi-and-capture-lifecycle).

Once Wi-Fi is configured, the device is discovered automatically by Home Assistant through ESPHome.

## Integration Surface

The frontend maps runtime state and controls into the ESPHome entities listed under [Integrated Entities](#integrated-entities). Those entities are created automatically when the `espectre:` component is declared.

ESPHome advertises its native API alongside ESPectre's Direct HTTP service. Run `./espectre devices --frontend esphome` to find the Direct endpoint. Direct and the ESPHome entities use the same runtime controls; Direct also provides local management and raw CSI collection. See [API.md](../../../../docs/API.md) for capabilities and [DISCOVERY.md](../../../../docs/DISCOVERY.md) for discovery.

Direct API is enabled by default. To expose only ESPHome's native API and Home Assistant entities, disable it in the component configuration:

```yaml
espectre:
  direct_api: false
```

This disables Direct HTTP requests, SSE telemetry, raw CSI streaming, `_espectre._tcp.local.` discovery, and the peer-assisted browser bootstrap responder. It does not disable ESPHome's native API or ESPectre entities.

A successful Direct mutation republishes the affected number or select state, so Home Assistant and Direct clients observe the same runtime configuration. Wi-Fi credentials, OTA, and ESPHome API encryption remain owned by ESPHome. Changing the ESPectre label does not alter the ESPHome hostname, adopted YAML, or entity IDs.

## Configuration Surface

Shared sensing options go under `espectre:` with the names, defaults, and ranges in [SDK.md](../../../../docs/SDK.md#shared-sensing-options). The YAML schema is in [__init__.py](components/espectre/__init__.py); the [Integrated Entities](#integrated-entities) table lists the runtime controls. Use [TROUBLESHOOTING.md](../../../../docs/TROUBLESHOOTING.md#tuning-essentials) to decide what to adjust.

### Diagnostic Telemetry

Press `Refresh Diagnostics` to publish the latest cached rate sample to Home Assistant. Diagnostic sensors are available in production builds and publish only on request. Direct HTTP also exposes performance, heap, load, and detector timing; see [API.md](../../../../docs/API.md#diagnostics). For interpreting input rates and occupancy, follow [TROUBLESHOOTING.md](../../../../docs/TROUBLESHOOTING.md#check-the-sensing-input).

### Detection Profile Selection

```yaml
wifi:
  band_mode: AUTO  # ESP32-C5 only; optional because AUTO is the default

espectre:
  detection_algorithm: lightweight  # or high_accuracy
```

Set the Wi-Fi band under `wifi:`, separately from ESPectre's detector selection. On ESP32-C5, `band_mode` accepts `2.4GHz`, `5GHz`, or `AUTO` and defaults to `AUTO`. Other supported targets use 2.4 GHz. Capture profiles and the limits of 5 GHz sensing are documented in [CSI.md](../../../../docs/CSI.md#capture-profiles).

The YAML value is the initial profile when no persisted selection exists. The Home Assistant `detector_select` changes it live and persists the choice across reboot. `high_accuracy -> lightweight` starts calibration automatically, and `calibration_active_sensor` reflects automatic and user-triggered calibration state.

See [TROUBLESHOOTING.md](../../../../docs/TROUBLESHOOTING.md#detection-profile) for profile selection and startup, and [ALGORITHMS.md](../../../../docs/ALGORITHMS.md) for detector behavior.

## Entity Customization

### Integrated Entities

| Sensor config | Type | Default name | Description |
|---------------|------|--------------|-------------|
| `movement_sensor` | sensor | `Movement Score` | Current movement score (0.0–1.0), published every `evaluation_interval_ms` |
| `motion_sensor` | binary_sensor | `Motion Detected` | Edge-driven motion state; resets to idle when sensing stops or CSI restarts |
| `threshold_number` | number | `Threshold` | Runtime probability threshold (0.0–1.0) |
| `motion_on_hits_number` | number | `Motion On Hits` | Runtime motion-on debounce count (1–20) |
| `motion_off_hits_number` | number | `Motion Off Hits` | Runtime motion-off debounce count (1–20) |
| `detector_select` | select | `Detection Profile` | Runtime `lightweight` / `high_accuracy` selection |
| `csi_traffic_mode_select` | select | `CSI Traffic Ownership` | Runtime `internal` / `external` selection |
| `traffic_generator_mode_select` | select | `CSI Traffic Source` | Runtime `ping` / `dns` (UDP) / `dns_tcp` / `wifi_raw` selection |
| `sensing_switch` | switch | `Sensing Enabled` | Enables or pauses sensing through the common command engine; publishes the runtime state at startup |
| `recalibrate_button` | button | `Recalibrate` | Starts runtime recalibration |
| `calibration_active_sensor` | binary_sensor | `Calibration Active` | Read-only authoritative calibration state |
| `diagnostics_button` | button | `Refresh Diagnostics` | Publishes the latest cached diagnostic sample on demand |
| `traffic_rate_sensor` | sensor | `Traffic TX Rate` | Diagnostic traffic rate |
| `csi_callback_rate_sensor` | sensor | `CSI Callback Rate` | Raw CSI callback rate; diagnostic-only |
| `csi_accepted_rate_sensor` | sensor | `CSI Accepted Rate` | Raw identity-accepted capture rate before temporal admission; diagnostic-only |
| `csi_admitted_rate_sensor` | sensor | `CSI Admitted Rate` | Rate admitted to the detector's temporal grid; diagnostic-only |
| `csi_filtered_rate_sensor` | sensor | `CSI Filtered Rate` | Capture rejection rate; diagnostic-only |
| `csi_missing_rate_sensor` | sensor | `CSI Missing Slot Rate` | Missing detector slots per second; diagnostic-only |
| `csi_excess_rate_sensor` | sensor | `CSI Excess Rate` | Non-selected same-slot candidates per second, including candidates replaced by one nearer the slot center; diagnostic-only |
| `csi_stale_rate_sensor` | sensor | `CSI Stale Rate` | Packets discarded as stale per second; diagnostic-only |
| `csi_out_of_order_rate_sensor` | sensor | `CSI Out-of-Order Rate` | Duplicate or backward-timestamp packets discarded per second; diagnostic-only |
| `csi_occupancy_sensor` | sensor | `CSI Temporal Occupancy` | Valid-slot occupancy of the active detector window; diagnostic-only |
| `wifi_channel_sensor` | sensor | `WiFi Channel` | Current associated Wi-Fi channel; diagnostic-only |
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

## Home Assistant Integration

Once the device is flashed and connected to Wi-Fi:

1. Home Assistant discovers it through ESPHome
2. Go to **Settings** -> **Devices & Services** -> **ESPHome**
3. Configure the discovered device
4. The default entities are added automatically

Writable entities report the authoritative runtime state, including when a change is rejected. Direct mutations synchronize the affected entities immediately.

Movement Score updates on the detector evaluation cadence, 250 ms by default. The high-rate path runs while the Movement Score entity exists or a Direct SSE client is connected, so Direct-only configurations do not need an unused Home Assistant sensor. Motion Detected publishes only on filtered state edges.

Threshold also updates after calibration and Lightweight settled-level recovery. Motion-hit controls publish on change, and traffic selects synchronize on connect and after accepted changes.

To reduce Home Assistant recorder history, exclude `sensor.*_movement_score` from recording. Keep the evaluation cadence set for the detector.

To manage configuration and OTA updates, install ESPHome Device Builder and adopt the discovered device. The adopted configuration uses the GitHub source profile, follows `main`, and identifies that rolling build as `0.0.0-main`. Repository builds launched through `./espectre esphome` resolve `project_version` from the same numeric `git describe` identity used by the other frontends. First-party CI and release builds override it with the detected build version or release tag.

To install a prebuilt OTA image from GitHub Releases instead, download the `espectre-esphome-<channel-or-version>-<chip>-ota.bin` asset and upload it over the network:

```bash
./espectre esphome flash --chip c6 --device espectre.local --firmware espectre-esphome-3.0.0-esp32c6-ota.bin
```

To stay on a released version, use the matching prebuilt image rather than the rolling `main` example.

### Dashboard Examples

[home-assistant-dashboard.yaml](examples/home-assistant-dashboard.yaml) provides motion, movement score, history, controls, and diagnostics.

![ESPectre Home Assistant dashboard](../../../../docs/web/assets/images/guides/home-assistant-dashboard.png)

*Home Assistant dashboard with motion state, movement score, movement-versus-threshold history, detection profile, threshold, calibration, and diagnostics.*

To import a dashboard:

1. Go to **Settings** -> **Dashboards** -> **Add Dashboard**
2. Open the dashboard and choose **Edit**
3. Open the raw configuration editor
4. Replace the default content with the YAML from the example file
5. Save the dashboard

If you changed the device name from `espectre`, update the entity IDs in the YAML. If you enabled `name_add_mac_suffix: true`, include the MAC suffix in the entity names as well. Inspect the exact IDs under the Home Assistant device before adapting the dashboard because an existing registry collision can add a suffix such as `_2`.

## Traffic Configuration

The ESPHome surface exposes the shared runtime traffic settings under `espectre:`. [`CSI.md`](../../../../docs/CSI.md#traffic-sources) describes traffic modes and pacing; [API.md](../../../../docs/API.md#external-csi-traffic) defines ports and external markers; [`TROUBLESHOOTING.md`](../../../../docs/TROUBLESHOOTING.md#no-csi-or-insufficient-input) owns rate and occupancy guidance.

### Internal Traffic Generator

```yaml
espectre:
  csi_target_pps: 100
  csi_traffic_mode: internal
  traffic_generator_mode: ping
  traffic_generator_target_ip: "" # Empty uses the gateway; set an IPv4 address to override
```

The `traffic_generator_mode_select` entity can change the internal source at runtime, and `csi_traffic_mode_select` can switch between internal and external ownership. Both selections persist after an accepted change.

### External Traffic Mode

To disable the internal generator and rely on external traffic:

```yaml
espectre:
  csi_target_pps: 100
  csi_traffic_mode: external
  csi_traffic_multicast_group: "239.255.0.1"
```

For raw collection, keep Direct API enabled and follow [CLI.md](../../../../docs/CLI.md#collect). The collector selects external traffic mode, which persists after collection.

## Build and Consumption

The `release`, `preview`, and `develop` channels publish one full-flash image and one OTA image per supported chip, with `lightweight` as the initial detector. Both `lightweight` and `high_accuracy` are available in the image and can be selected through the persisted runtime detector entity. After adoption, ESPHome Device Builder can compile and install updates wirelessly from the device YAML; `detection_algorithm` sets the initial detector for a fresh configuration rather than limiting which detector the firmware supports.

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

`esphome.project.version` sets the application version. If omitted, a numeric source `ref`, such as `3.0.0-rc1`, supplies it; other refs and local sources retain ESPHome's default. Application versions are limited to 31 bytes. SDK identity is separate; see [SDK.md](../../../../docs/SDK.md#versioning).

Omit `ref` to use the repository's default branch, or set a valid tag or commit to pin the component. An empty `ref: ""` is invalid.

Repository development selects `espectre-source-local.yaml` instead, which resolves the same component from the local checkout:

```yaml
external_components:
  - source:
      type: local
      path: ../components
    components: [espectre]
```

### Repository CLI

See [`CLI.md`](../../../../docs/CLI.md) for shared CLI syntax, host-side tools, and wrapper behavior.

```bash
./espectre esphome build --chip c6 --clean
./espectre esphome flash --chip c6
./espectre esphome config --chip c6
./espectre esphome monitor --chip c6 --device /dev/cu.usbmodemXXXX
```

On Windows, use `.\espectre.cmd esphome ...` from the repository root and pass a COM port such as `COM5` to `--device` when serial access is needed.

The repository CLI keeps the selected canonical YAML and loads the ESPectre component from the local checkout. Use `flash` for upload-only and `monitor` for logs.

## Hardware and Packaging Notes

### Build Toolchain

The ESPHome examples use the native ESP-IDF backend from the ESPHome version pinned in [`requirements.txt`](../../../../requirements.txt). [`__init__.py`](components/espectre/__init__.py) registers this component directory with ESP-IDF's component manager. Its [`CMakeLists.txt`](components/espectre/CMakeLists.txt) reuses the canonical SDK build definition at [`CMakeLists.txt`](../../CMakeLists.txt), so ESPHome compiles `src/cpp/core/` and `src/cpp/runtime/esp_idf/` directly. No toolchain override or separate library package is required.

### Automatic SDK Configuration

The component sets the required CSI, Wi-Fi power, aggregation, buffer, and lwIP options in [__init__.py](components/espectre/__init__.py). It routes runtime logs through the ESPHome logger and selects TinyUSB CDC for logging and Improv Serial on maintained USB-OTG configurations that need it. Board-specific overrides can go under `esp32.framework.sdkconfig_options`.

The examples use Improv Serial to avoid the flash, memory, and radio contention costs of a BLE provisioning stack. Keep that choice when adapting an example unless your product requires BLE.

### Flash Size and Partitions

The ESPHome frontend fits in `4 MB` flash with OTA and uses the board and framework default partition table. A custom project can override it with `esp32.partitions`; ESPectre itself does not require a custom table.

## ESPHome-Specific Troubleshooting

Use [TROUBLESHOOTING.md](../../../../docs/TROUBLESHOOTING.md#check-the-sensing-input) for sensing and connectivity problems, and [SETUP.md](../../../../docs/SETUP.md#web-flash-no-coding-required) when the board does not enter download mode.

### View logs

Home Assistant entity updates do not replace the serial status log. The shared runtime forwards its 1 Hz `IDLE | csi:` / `MOTION | csi:` heartbeats through the ESPHome sink, so they should appear on USB serial and in `esphome logs` when the `espectre.runtime` tag permits INFO messages.

```bash
esphome logs <your-config>.yaml
esphome logs <your-config>.yaml --device espectre.local
./espectre monitor --port /dev/cu.usbmodem*
```

## Implementation Map

This map is for component maintainers; it is not required for normal installation or tuning.

- [`__init__.py`](components/espectre/__init__.py): YAML schema, validation, codegen, native ESP-IDF component registration, and ESPHome build flags
- [`CMakeLists.txt`](components/espectre/CMakeLists.txt): native ESP-IDF bridge to the canonical shared SDK build definition
- [`espectre.cpp`](components/espectre/espectre.cpp), [`espectre.h`](components/espectre/espectre.h): ESPHome adapter over the shared runtime frontend controller
- [`sensor_publisher.cpp`](components/espectre/sensor_publisher.cpp): movement and motion publishing
- [`threshold_number.cpp`](components/espectre/threshold_number.cpp): runtime threshold control
- [`motion_hits_number.cpp`](components/espectre/motion_hits_number.cpp): runtime motion-hit debounce control
- [`detector_select.cpp`](components/espectre/detector_select.cpp): persisted runtime detector selection
- [`sensing_switch.cpp`](components/espectre/sensing_switch.cpp): sensing lifecycle control
- [`recalibrate_button.cpp`](components/espectre/recalibrate_button.cpp): runtime recalibration action
- [`traffic_mode_select.cpp`](components/espectre/traffic_mode_select.cpp): runtime CSI traffic ownership and generator control
- [`examples/`](examples/): production and local-development configurations for ESP32, ESP32-S2, ESP32-S3, ESP32-C3, ESP32-C5, and ESP32-C6, plus the Home Assistant dashboard
