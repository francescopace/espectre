# ESPectre ESPHome Frontend

Installing an ESPHome image? Start with [Getting Started](#getting-started). If the device is already adopted, [Configuration Surface](#configuration-surface) lists the YAML options and entity controls. The implementation and packaging sections are for component maintainers.

## Scope

The ESPHome frontend is responsible for:

- YAML schema and code generation
- packaging as an ESPHome `external_components` source
- Home Assistant-facing entities
- ESPHome provisioning and dashboard-oriented usage
- ESPHome-specific SDK configuration defaults and troubleshooting

## Getting Started

If you want the browser-flash path, start from [`SETUP.md`](../../../../docs/SETUP.md) and come back here after flashing `ESPHome`.

After flashing, configure Wi-Fi with one of these provisioning paths:

| Method | How |
|--------|-----|
| USB | Use Improv Serial with `./espectre provision --ssid MyNetwork` or any Improv Serial-compatible web flasher, such as the [ESPectre web flasher](https://espectre.dev/tools/flash/) |
| Captive portal | Connect to the `ESPectre Fallback` network and finish setup in the browser |

All maintained ESPHome example configurations enable Improv Serial.

Once Wi-Fi is configured, the device is discovered automatically by Home Assistant through ESPHome.

ESPHome continues to advertise its native API as `_esphomelib._tcp.local.`. ESPectre also publishes the canonical `_espectre._tcp.local.` record for its Direct HTTP endpoint on the shared port `62587`. Run `./espectre devices --frontend esphome` to list that first-party record with the standard ESPectre `device_id`; the CLI does not inspect or depend on ESPHome's upstream TXT schema. [`ESPECTRE_PROTOCOL.md`](../../../../docs/ESPECTRE_PROTOCOL.md#mdnsdns-sd-discovery) owns the shared record contract.

Direct HTTP exposes the same runtime threshold, motion-hit counts, detector selection, recalibration, CSI traffic ownership, and traffic-generator controls as the ESPHome entities when the runtime advertises them. It also exposes the current Wi-Fi association, access-point scans, BSSID pinning and pin removal, an editable persisted ESPectre label, peer discovery, and raw CSI. Shared Auto-discovery, including Micro-ESPectre peer results, is documented in [Peer-assisted browser discovery](../../../../docs/ESPECTRE_PROTOCOL.md#peer-assisted-browser-discovery). A successful Direct mutation republishes the affected number or select state, so Home Assistant and Direct clients observe one current runtime configuration. Wi-Fi credentials, OTA, and ESPHome API encryption remain owned by ESPHome rather than this Direct surface; changing the ESPectre label does not alter the ESPHome hostname, adopted YAML, or entity IDs.

The `release`, `preview`, and `develop` channels publish one full-flash image and one OTA image per supported chip, with `lightweight` as the initial detector. Both `lightweight` and `high_accuracy` are available in the image and can be selected through the persisted runtime detector entity. After adoption, ESPHome Device Builder can compile and install updates wirelessly from the device YAML; `detection_algorithm` sets the initial detector for a fresh configuration rather than limiting which detector the firmware supports.

## Integration Surface

The frontend maps runtime state into ESPHome and Home Assistant entities.

| Runtime state/event | ESPHome surface | Cadence |
|---------------------|-----------------|---------|
| movement metric | `movement_sensor` | Detector evaluation (`evaluation_interval_ms`, default 250 ms) |
| motion state | `motion_sensor` | Filtered state edges |
| runtime threshold write | `threshold_number` | On change |
| runtime motion-hit debounce write | `motion_on_hits_number`, `motion_off_hits_number` | On change |
| runtime detector selection | `detector_select` | On change |
| sensing lifecycle | `sensing_switch` | On change |
| runtime recalibration trigger | `recalibrate_button` | On press |
| calibration state | `calibration_active_sensor` | On change |
| CSI traffic ownership | `csi_traffic_mode_select` | On change |
| internal traffic generator type | `traffic_generator_mode_select` | On change |
| on-demand CSI diagnostics | diagnostic sensors and `diagnostics_button` | On request |

The default entities are created automatically when the `espectre:` component is declared.

## Configuration Surface

The ESPHome YAML schema is defined in [`__init__.py`](components/espectre/__init__.py). This README covers ESPHome-specific syntax and entity mapping. See [`SETUP.md`](../../../../docs/SETUP.md) for the shared configuration overview and [`TUNING.md`](../../../../docs/TUNING.md) for the "when and why" of tuning.

### Core Parameters

The shared sensing options, with their defaults and ranges, are documented in the [`Shared Sensing Options`](../../../../docs/SETUP.md#shared-sensing-options) table in `SETUP.md`. In ESPHome, those options live under the `espectre:` section with the same names, as shown in the example below.

These options are applied from YAML during firmware configuration. Runtime control is exposed separately through the entities below:

| Runtime surface | Config key | Runtime behavior |
|-----------------|------------|------------------|
| Movement score | `movement_sensor` | Read-only Home Assistant sensor; evaluation cadence |
| Motion state | `motion_sensor` | Read-only Home Assistant binary sensor; edge-published |
| Threshold | `threshold_number` | Writable runtime threshold control |
| Motion On Hits | `motion_on_hits_number` | Writable runtime motion-on debounce control |
| Motion Off Hits | `motion_off_hits_number` | Writable runtime motion-off debounce control |
| Detection profile | `detector_select` | Writable, persisted `lightweight` / `high_accuracy` selection |
| Sensing | `sensing_switch` | Writable runtime sensing lifecycle |
| Recalibration | `recalibrate_button` | Writable runtime recalibration action |
| Calibration state | `calibration_active_sensor` | Read-only runtime calibration state |
| CSI traffic ownership | `csi_traffic_mode_select` | Writable, persisted `internal` / `external` selection |
| Traffic generator | `traffic_generator_mode_select` | Writable, persisted `ping` / `dns` selection |

### Diagnostic Telemetry

Diagnostic entities are always available in production builds. ESPectre refreshes their cached rate sample from the existing sensing update that also feeds the periodic status log, without adding a diagnostic timer or periodically publishing new Home Assistant states. Direct returns the same cached rate sample. Press `Refresh Diagnostics` to publish it to Home Assistant on demand:

| Entity | Meaning |
|--------|---------|
| `Traffic TX Rate` | Successful internal generator or observed external marker packets per second |
| `CSI Callback Rate` | Raw ESP-IDF CSI callbacks per second |
| `CSI Accepted Rate` | CSI packets per second accepted by the sensing pipeline |
| `CSI Filtered Rate` | CSI packets per second rejected by capture validation |
| `WiFi Channel` | Current primary channel reported by the associated access point |
| `WiFi RSSI` | Current RSSI reported by the Wi-Fi association |

Comparing the three main rates localizes failures: traffic without callbacks points at capture/radio state, callbacks without accepted packets points at validation or identity filtering, and accepted packets without stable detector output points above the capture layer.

Runtime performance, heap, load, and detector timing are collected as production diagnostics and are available through Direct HTTP without a build-time switch or periodic debug logs.

### Detection Profile Selection

```yaml
wifi:
  band_mode: 2.4GHz  # ESP32-C5: also accepts 5GHz or AUTO

espectre:
  detection_algorithm: lightweight  # or high_accuracy
```

ESPHome owns Wi-Fi association policy through `wifi.band_mode`; it is not an `espectre:` property. On ESP32-C5 it accepts `2.4GHz`, `5GHz`, or `AUTO` and is optional; when omitted, ESPectre follows ESPHome's `AUTO` default. Other supported targets are single-band and remain fixed to 2.4 GHz. ESPectre mirrors the effective ESPHome selection into its runtime and keeps the production sensing contract at HT20 on the selected band. The examples select `2.4GHz` because detection quality on 5 GHz is not yet characterized.

Threshold behavior:

- range: `0.0-1.0` for both detectors
- `lightweight`: automatic session-adapted startup threshold, then possible quiet-stretch lowering published through `on_threshold_changed`
- `high_accuracy` default: `0.5`

Lightweight Detection uses less active detector CPU and working memory, making it suitable when the ESPHome node also runs resource-intensive components. High-Accuracy Detection uses more feature state and inference work but provides higher accuracy and skips Lightweight's threshold calibration. Lightweight requires about 10 seconds of clean, ready quiet-room coverage after temporal warmup; insufficient occupancy extends that wall-clock duration. High Accuracy still waits for CSI readiness and its feature window to fill.

The YAML value is the initial profile when no persisted selection exists. The Home Assistant `detector_select` changes it live and persists the choice across reboot. `high_accuracy -> lightweight` starts calibration automatically, and `calibration_active_sensor` reflects automatic and user-triggered calibration state.

See [`ALGORITHMS.md`](../../../../docs/ALGORITHMS.md) for how the two detectors differ and [`TUNING.md`](../../../../docs/TUNING.md) for choosing between them.

### Example

```yaml
espectre:
  detection_algorithm: lightweight
  csi_target_pps: 100
  csi_traffic_mode: internal
  csi_traffic_multicast_group: "239.255.0.1"
  traffic_generator_mode: ping
  segmentation_window_size_ms: 1000
  motion_on_hits: 4
  motion_off_hits: 3
```

## Entity Customization

### Integrated Entities

| Sensor config | Type | Default name | Description |
|---------------|------|--------------|-------------|
| `movement_sensor` | sensor | `Movement Score` | Current movement score (0.0–1.0), published every `evaluation_interval_ms` |
| `motion_sensor` | binary_sensor | `Motion Detected` | Edge-driven motion state |
| `threshold_number` | number | `Threshold` | Runtime probability threshold (0.0–1.0) |
| `motion_on_hits_number` | number | `Motion On Hits` | Runtime motion-on debounce count (1–20) |
| `motion_off_hits_number` | number | `Motion Off Hits` | Runtime motion-off debounce count (1–20) |
| `detector_select` | select | `Detection Profile` | Runtime `lightweight` / `high_accuracy` selection |
| `csi_traffic_mode_select` | select | `CSI Traffic Ownership` | Runtime `internal` / `external` selection |
| `traffic_generator_mode_select` | select | `CSI Traffic Source` | Runtime `ping` / `dns` selection |
| `sensing_switch` | switch | `Sensing Enabled` | Enables or pauses sensing through the common command engine |
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

All entities support standard ESPHome options such as:

- `name`
- `internal`
- `icon`
- `disabled_by_default`

The `movement_sensor` also supports ESPHome [sensor filters](https://esphome.io/components/sensor/#sensor-filters).

Common filters:

| Filter | Example | Description |
|--------|---------|-------------|
| `multiply` | `multiply: 10` | Scale values |
| `round` | `round: 1` | Round to N decimals |
| `clamp` | `min_value: 0, max_value: 100` | Limit the value range |
| `offset` | `offset: -0.5` | Add or subtract a constant |
| `sliding_window_moving_average` | `window_size: 5` | Smooth noisy readings |

Example:

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

The ESPHome frontend exposes movement, motion, sensing state, threshold control, motion-hit debounce control, recalibration, calibration state, CSI traffic ownership, traffic generator selection, and on-demand CSI diagnostics as Home Assistant entities. Every writable entity invokes the common command engine and republishes authoritative state when a command is rejected. Direct mutations use the same engine and immediately synchronize the affected entities. Movement Score updates on the detector evaluation cadence (default 250 ms). The high-rate path runs while the Movement Score entity exists or a Direct SSE client is connected, so Direct-only configurations do not need to add an unused Home Assistant sensor. Motion Detected publishes only on filtered state edges. Threshold publishes on operator writes, calibration, and Lightweight settled-level recovery; motion-hit controls publish on change. Calibration Active reports the read-only runtime state, and the traffic selects mirror runtime state on connect and each accepted change. Diagnostic sensors publish only when Refresh Diagnostics runs the canonical `diagnostics` query. If the Home Assistant recorder is a concern, exclude `sensor.*_movement_score` rather than lowering `evaluation_interval_ms`.

To manage configuration and OTA updates, install ESPHome Device Builder and adopt the discovered device. The adopted configuration uses the GitHub source profile, follows `main`, and identifies that rolling build as `0.0.0-main`. First-party CI and release builds use the local checkout and override `project_version` with `git describe` or the release tag.

To install a prebuilt OTA image from GitHub Releases instead, download the `espectre-esphome-<channel-or-version>-<chip>-ota.bin` asset and upload it over the network:

```bash
./espectre esphome flash --chip c6 --device espectre.local --firmware espectre-esphome-3.0.0-esp32c6-ota.bin
```

To stay on a released version, use the matching prebuilt image rather than the rolling `main` example.

### Dashboard Examples

Examples live in:

| File | Description |
|------|-------------|
| [`home-assistant-dashboard.yaml`](examples/home-assistant-dashboard.yaml) | Production dashboard with motion, movement score, history, controls, and diagnostics |

![ESPectre Home Assistant dashboard](../../../../docs/web/assets/images/guides/home-assistant-dashboard.png)

*Home Assistant dashboard with motion state, movement score, movement-versus-threshold history, detection profile, threshold, calibration, and diagnostics. Native MQTT Discovery reuses these cards after replacing the `espectre_` prefix.*

To import a dashboard:

1. Go to **Settings** -> **Dashboards** -> **Add Dashboard**
2. Open the dashboard and choose **Edit**
3. Open the raw configuration editor
4. Replace the default content with the YAML from the example file
5. Save the dashboard

If you changed the device name from `espectre`, update entity IDs in the YAML. If you enabled `name_add_mac_suffix: true`, include the MAC suffix in the entity names as well. Home Assistant generates Native MQTT entity IDs when it first registers them, so inspect the exact IDs under the device before adapting this dashboard. A default Native device can produce an ID such as `sensor.espectre_c3_223333_movement_score`, and an existing registry collision can add a suffix such as `_2`.

## Traffic Generator and Runtime Notes

The ESPHome surface exposes the shared runtime traffic-generation settings. By default, the device continuously generates traffic for CSI collection while powered on.

### Internal Traffic Generator

```yaml
espectre:
  csi_target_pps: 100
  csi_traffic_mode: internal
  traffic_generator_mode: ping
```

`csi_target_pps` defines the temporal detector grid and the internal managed-traffic target. `csi_traffic_mode` independently selects `internal` or `external`; a rate of zero is invalid. Internal traffic uses a fixed DNS or ICMP send rate at that target. Occupancy does not change the send rate; if occupancy stays below 70%, repair the traffic path or lower `csi_target_pps` explicitly.

Available modes:

| Mode | Protocol | Notes |
|------|----------|-------|
| `ping` | ICMP | Default and usually the safest choice |
| `dns` | TCP | Persistent, non-blocking root queries to gateway port `53`; use only when the router accepts DNS over TCP |

### External Traffic Mode

To disable the internal generator and rely on external traffic:

```yaml
espectre:
  csi_target_pps: 100
  csi_traffic_mode: external
  csi_traffic_multicast_group: "239.255.0.1"
  evaluation_interval_ms: 250
```

In that mode the runtime opens a UDP listener on port `5555` and joins multicast group `239.255.0.1` by default. Drive it with unicast UDP to each device IP, or with one datagram to `239.255.0.1`. Use [`espectre_traffic_generator.py`](../../../../tools/espectre_traffic_generator.py) and set `TARGETS` to a device IP, a list of addresses, or `['239.255.0.1']`. Set `csi_traffic_multicast_group: ""` to disable the join. Subnet and limited broadcast (`x.x.x.255`, `255.255.255.255`) do not produce reliable CSI: access points typically send those frames at legacy rates, which the HT20 capture contract drops.

For raw collection, use `./espectre collect` with this device's IP, hostname, Direct endpoint, or device ID. ESPHome exposes Direct and raw HTTP on the shared port `62587`; the collector persistently selects external mode and drives the port-`5555` marker source.

For rate recommendations, airtime tradeoffs, and placement guidance, see [`TUNING.md`](../../../../docs/TUNING.md).

## Startup Calibration

In `lightweight` mode, keep the room quiet after boot so the runtime can complete the startup threshold bootstrap; a later quiet stretch can still lower the live threshold, and the Home Assistant number follows it. `high_accuracy` skips the bootstrap and starts once CSI capture is ready and its feature window has filled. For the startup workflow and budget details, see [`TUNING.md`](../../../../docs/TUNING.md).

Runtime recalibration is exposed as the `recalibrate_button` entity in Home Assistant. `calibration_active_sensor` reports the authoritative in-progress state.

## Build and Consumption

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

The ESPHome examples use ESPHome 2026.7's native ESP-IDF backend. The external component registers the shared sensing tree as a local ESP-IDF component, so no toolchain override or separate library package is required.

### Automatic SDK Configuration

The frontend automatically sets the ESP-IDF options required by the runtime, including CSI enablement, disabled Wi-Fi power save, TX AMPDU, the shared high-rate Wi-Fi buffer profile, lwIP IRAM optimization, and enlarged TCP/IP and UDP mailboxes. RX AMPDU remains disabled so sensing receives individual CSI frames. ESPHome keeps the ESP-IDF default log level at ERROR so Wi-Fi and lwIP stay quiet; the shared SDK compiles INFO/DEBUG only in its own sources and restores the `espectre` and `espectre.runtime` tags at runtime so the periodic `IDLE | csi:` status lines reach USB serial. In most cases you do not need to set these options manually.

The supplied examples deliberately use Improv Serial instead of BLE provisioning. Omitting BLE keeps its provisioning stack out of the firmware, reducing flash and memory pressure. BLE and Wi-Fi share the ESP32's 2.4 GHz radio; while BLE is active, coexistence can interrupt the Wi-Fi packet flow and reduce the CSI occupancy required for reliable sensing. Disabling BLE after provisioning ends that radio contention, but it does not remove the compiled-in stack or the added provisioning lifecycle.

For board-specific tweaks, you can still add `sdkconfig_options` in YAML:

```yaml
esp32:
  variant: ESP32C6
  framework:
    type: esp-idf
    sdkconfig_options:
      CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ: "160"
```

### Flash Size and Partitions

The ESPHome frontend fits in `4 MB` flash with OTA. It uses the board and framework default partition table unless you override it in your own project.

If you need a custom table:

```yaml
esp32:
  variant: ESP32C6
  partitions: /absolute/path/to/partitions_custom.csv
```

The frontend itself does not require a custom partition table.

## ESPHome-Specific Troubleshooting

### No motion detection

1. Verify Wi-Fi is connected
2. Verify traffic generation is active, or provide unicast or multicast (`239.255.0.1`) external traffic to port `5555`
3. Wait for startup calibration to complete in `lightweight`
4. Lower the Threshold number entity if the detector is too conservative

### False positives

1. Raise the Threshold number entity
2. Check for fans, AC, curtains, or other interference
3. Increase `segmentation_window_size_ms` for a longer, steadier analysis interval

### Mesh Wi-Fi instability

If the device roams between access points, lock it to a specific BSSID.

For local configurations:

```yaml
wifi_bssid: "AA:BB:CC:DD:EE:FF"
```

Then reference it in the `wifi` block:

```yaml
wifi:
  networks:
    - ssid: !secret wifi_ssid
      password: !secret wifi_password
      bssid: !secret wifi_bssid
```

### Flash failed

1. Hold the `BOOT` button
2. Press `RESET`
3. Release `BOOT`
4. Retry the flash

### View logs

Home Assistant entity updates do not replace the serial status log. Movement and motion can be live in Home Assistant while `espectre monitor` stays quiet if the shared runtime `ESP_LOGI` lines are compiled out. After a current ESPHome build, the 1 Hz `IDLE | csi:` / `MOTION | csi:` heartbeats should appear on USB serial as well as in `esphome logs`.

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

## Packaging Notes

[`__init__.py`](components/espectre/__init__.py) registers this component directory with ESP-IDF's component manager. Its [`CMakeLists.txt`](components/espectre/CMakeLists.txt) reuses the canonical SDK build definition at [`CMakeLists.txt`](../../CMakeLists.txt), so ESPHome compiles `src/cpp/core/` and `src/cpp/runtime/esp_idf/` directly through the native ESP-IDF backend.
