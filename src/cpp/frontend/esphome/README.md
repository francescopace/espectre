# ESPectre ESPHome Frontend

Use this guide to install, configure, or maintain the ESPHome integration for Home Assistant. First-time users can jump to [Getting Started](#getting-started) and [Configuration Surface](#configuration-surface); component maintainers can continue through the implementation and packaging sections.

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
| BLE | Use the ESPHome or Home Assistant Companion app |
| USB | Go to [web.esphome.io](https://web.esphome.io) and use **Connect** -> **Configure WiFi** |
| Captive portal | Connect to the `ESPectre Fallback` network and finish setup in the browser |

Once Wi-Fi is configured, the device is discovered automatically by Home Assistant through ESPHome.

Release and snapshot channels publish one full-flash image per supported chip, with `lightweight` as the initial detector. Both `lightweight` and `high_accuracy` are available in the image and can be selected through the persisted runtime detector entity. After adoption, ESPHome Device Builder compiles and installs updates wirelessly from the device YAML; `detection_algorithm` sets the initial detector for a fresh configuration rather than limiting which detector the firmware supports.

## Integration Surface

The frontend maps runtime state into ESPHome and Home Assistant entities.

| Runtime state/event | ESPHome surface |
|---------------------|-----------------|
| movement metric | `movement_sensor` |
| movement vs threshold | `intensity_sensor` |
| motion state | `motion_sensor` |
| runtime threshold write | `threshold_number` |
| runtime detector selection | `detector_select` |
| runtime recalibration trigger | `calibrate_switch` |
| on-demand CSI diagnostics | diagnostic sensors and `diagnostics_button` |

The default entities are created automatically when the `espectre:` component is declared.

## Configuration Surface

The ESPHome YAML schema is defined in [`__init__.py`](components/espectre/__init__.py). This README covers ESPHome-specific syntax and entity mapping. See [`SETUP.md`](../../../../docs/SETUP.md) for the shared configuration overview and [`TUNING.md`](../../../../docs/TUNING.md) for the "when and why" of tuning.

### Core Parameters

The shared sensing options, with their defaults and ranges, are documented in the [`Shared Sensing Options`](../../../../docs/SETUP.md#shared-sensing-options) table in `SETUP.md`. In ESPHome, those options live under the `espectre:` section with the same names, as shown in the example below.

These options are applied from YAML during firmware configuration. Runtime control is exposed separately through the entities below:

| Runtime surface | Config key | Runtime behavior |
|-----------------|------------|------------------|
| Movement score | `movement_sensor` | Read-only Home Assistant sensor |
| Intensity | `intensity_sensor` | Read-only movement-vs-threshold percent (0–200) |
| Motion state | `motion_sensor` | Read-only Home Assistant binary sensor |
| Threshold | `threshold_number` | Writable runtime threshold control |
| Detection profile | `detector_select` | Writable, persisted `lightweight` / `high_accuracy` selection |
| Recalibration | `calibrate_switch` | Writable runtime recalibration trigger |

### Diagnostic Telemetry

Diagnostic entities are always available in production builds. ESPectre refreshes their cached rate sample from the existing sensing update that also feeds the periodic status log, without adding a diagnostic timer or periodically publishing new Home Assistant states. Press `Refresh Diagnostics` to publish the latest cached sample on demand:

| Entity | Meaning |
|--------|---------|
| `Traffic TX Rate` | Successful internal traffic-generator or external pacing packets per second |
| `CSI Callback Rate` | Raw ESP-IDF CSI callbacks per second |
| `CSI Accepted Rate` | CSI packets per second accepted by the sensing pipeline |
| `CSI Filtered Rate` | CSI packets per second rejected by capture validation |
| `WiFi Channel` | Current primary channel reported by the associated access point |
| `WiFi RSSI` | Current RSSI reported by the Wi-Fi association |

Comparing the three main rates localizes failures: traffic without callbacks points at capture/radio state, callbacks without accepted packets points at validation or identity filtering, and accepted packets without stable detector output points above the capture layer.

The optional `debug_telemetry: true` setting is separate: it enables periodic runtime DEBUG logs with heap, load, and timing metrics, but it is not required for these diagnostic entities or their sampling.

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
- `lightweight`: automatic session-adapted startup threshold
- `high_accuracy` default: `0.5`

Lightweight Detection uses less active detector CPU and working memory, making it suitable when the ESPHome node also runs resource-intensive components. High-Accuracy Detection uses more feature state and inference work but provides higher accuracy and skips Lightweight's threshold calibration. Lightweight requires about 10 seconds of clean, ready quiet-room coverage after temporal warmup; insufficient occupancy extends that wall-clock duration. High Accuracy still waits for CSI readiness and its feature window to fill.

The YAML value is the initial profile when no persisted selection exists. The Home Assistant `detector_select` changes it live and persists the choice across reboot. `high_accuracy -> lightweight` starts calibration automatically, and the `calibrate_switch` reflects automatic and user-triggered calibration state.

See [`ALGORITHMS.md`](../../../../docs/ALGORITHMS.md) for how the two detectors differ and [`TUNING.md`](../../../../docs/TUNING.md) for choosing between them.

### Example

```yaml
espectre:
  detection_algorithm: lightweight
  csi_target_pps: 100
  csi_traffic_mode: internal
  traffic_generator_adaptive: false
  traffic_generator_mode: ping
  segmentation_window_size_ms: 1000
  motion_on_hits: 4
  motion_off_hits: 3
```

## Entity Customization

### Integrated Entities

| Sensor config | Type | Default name | Description |
|---------------|------|--------------|-------------|
| `movement_sensor` | sensor | `Movement Score` | Current movement score (0.0–1.0) |
| `intensity_sensor` | sensor | `Intensity` | Movement relative to threshold (`min(200, movement / threshold × 100)`); 100% is at threshold |
| `motion_sensor` | binary_sensor | `Motion Detected` | Edge-driven motion state |
| `threshold_number` | number | `Threshold` | Runtime probability threshold (0.0–1.0) |
| `detector_select` | select | `Detection Profile` | Runtime `lightweight` / `high_accuracy` selection |
| `calibrate_switch` | switch | `Calibrate` | Startup recalibration trigger |
| `diagnostics_button` | button | `Refresh Diagnostics` | Publishes the latest cached diagnostic sample on demand |
| `traffic_rate_sensor` | sensor | `Traffic TX Rate` | Diagnostic traffic rate |
| `csi_callback_rate_sensor` | sensor | `CSI Callback Rate` | Raw CSI callback rate; diagnostic-only |
| `csi_accepted_rate_sensor` | sensor | `CSI Accepted Rate` | Raw identity-accepted capture rate before temporal admission; diagnostic-only |
| `csi_admitted_rate_sensor` | sensor | `CSI Admitted Rate` | Rate admitted to the detector's temporal grid; diagnostic-only |
| `csi_filtered_rate_sensor` | sensor | `CSI Filtered Rate` | Capture rejection rate; diagnostic-only |
| `csi_missing_rate_sensor` | sensor | `CSI Missing Rate` | Missing detector slots per second; diagnostic-only |
| `csi_excess_rate_sensor` | sensor | `CSI Excess Rate` | Same-slot packets discarded per second; diagnostic-only |
| `csi_stale_rate_sensor` | sensor | `CSI Stale Rate` | Packets discarded as stale per second; diagnostic-only |
| `csi_out_of_order_rate_sensor` | sensor | `CSI Out-of-Order Rate` | Duplicate or backward-timestamp packets discarded per second; diagnostic-only |
| `csi_occupancy_sensor` | sensor | `CSI Occupancy` | Valid-slot occupancy of the active detector window; diagnostic-only |
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

The ESPHome frontend exposes movement, intensity, motion, threshold control, and recalibration as Home Assistant entities.

To manage configuration and OTA updates, install ESPHome Device Builder and adopt the discovered device. The adopted configuration compiles the component from the `git_ref` substitution, which defaults to `main` and therefore tracks the latest release, so the device follows each new release without manual edits.

To stay on one version instead, declare `git_ref` in the adopted configuration:

```yaml
substitutions:
  git_ref: "3.0.0"
```

The same value also drives the import URL the device republishes after the next build, so this one declaration is enough.

The `@` suffix of the adopted `packages` URL is a separate ref, and it selects which revision of the example YAML is downloaded. Change it to `@${git_ref}` to keep both on the same revision. This matters for snapshot builds, whose URL carries their source commit while the component still follows `main`.

### Dashboard Examples

Examples live in:

| File | Description |
|------|-------------|
| [`home-assistant-dashboard.yaml`](examples/home-assistant-dashboard.yaml) | Production dashboard with motion entities |

To import a dashboard:

1. Go to **Settings** -> **Dashboards** -> **Add Dashboard**
2. Open the dashboard and choose **Edit**
3. Open the raw configuration editor
4. Replace the default content with the YAML from the example file
5. Save the dashboard

If you changed the device name from `espectre`, update entity IDs in the YAML. If you enabled `name_add_mac_suffix: true`, include the MAC suffix in the entity names as well.

## Traffic Generator and Runtime Notes

The ESPHome surface exposes the shared runtime traffic-generation settings. By default, the device continuously generates traffic for CSI collection while powered on.

### Internal Traffic Generator

```yaml
espectre:
  csi_target_pps: 100
  csi_traffic_mode: internal
  traffic_generator_adaptive: false
  traffic_generator_mode: ping
```

`csi_target_pps` defines the temporal detector grid and the managed-traffic target. `csi_traffic_mode` independently selects `internal`, `external`, `pacing`, or `disabled`; a rate of zero is invalid. Internal traffic uses a fixed DNS or ICMP send rate by default. Set `traffic_generator_adaptive: true` only when raw accepted CSI is a validated feedback signal for the deployment; unrelated Home Assistant or application traffic can otherwise reduce the managed sensing cadence.

Available modes:

| Mode | Protocol | Notes |
|------|----------|-------|
| `ping` | ICMP | Default and usually the safest choice |
| `dns` | UDP | Lower-overhead alternative when the router responds consistently |

### External Traffic Mode

To disable the internal generator and rely on external traffic:

```yaml
espectre:
  csi_target_pps: 100
  csi_traffic_mode: external
  publish_interval_ms: 1000
  evaluation_interval_ms: 250
```

In that mode the runtime opens a UDP listener on port `5555`. Use [`espectre_traffic_generator.py`](../../../../tools/espectre_traffic_generator.py) to drive one or more devices from the network.

For rate recommendations, airtime tradeoffs, and placement guidance, see [`TUNING.md`](../../../../docs/TUNING.md).

## Startup Calibration

In `lightweight` mode, keep the room quiet after boot so the runtime can complete the startup threshold bootstrap; `high_accuracy` skips the bootstrap and starts once CSI capture is ready and its feature window has filled. For the startup workflow and budget details, see [`TUNING.md`](../../../../docs/TUNING.md).

Runtime recalibration is exposed as the `calibrate_switch` entity in Home Assistant.

## Build and Consumption

### As an ESPHome external component

Production examples consume this frontend with:

```yaml
external_components:
  - source:
      type: git
      url: https://github.com/francescopace/espectre
      path: src/cpp/frontend/esphome/components
    components: [espectre]
```

Local development examples consume it with:

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

Add `--dev` to use the local development YAML mapping. Use `flash` for upload-only and `monitor` for logs.

## Hardware and Packaging Notes

### Build Toolchain

The ESPHome examples use ESPHome 2026.7's native ESP-IDF backend. The external component registers the shared sensing tree as a local ESP-IDF component, so no toolchain override or separate library package is required.

### Automatic SDK Configuration

The frontend automatically sets the ESP-IDF options required by the runtime, including CSI enablement, disabled Wi-Fi power save, TX AMPDU, the Streamer high-rate Wi-Fi buffer profile, lwIP IRAM optimization, and enlarged TCP/IP and UDP mailboxes. RX AMPDU remains disabled so sensing receives individual CSI frames. The supplied examples do not enable Bluetooth. In most cases you do not need to set these options manually.

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
2. Verify traffic generation is active, or provide external traffic
3. Wait for startup calibration to complete in `lightweight`
4. Lower the Threshold number entity if the detector is too conservative

### False positives

1. Raise the Threshold number entity
2. Check for fans, AC, curtains, or other interference
3. Increase `segmentation_window_size_ms` for a longer, steadier analysis interval

### Mesh Wi-Fi instability

If the device roams between access points, lock it to a specific BSSID.

For development YAML files:

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

### ESP32-C3 Super Mini

Common fixes for low-cost C3 boards:

1. if USB logs are missing, force `UART0` in the logger
2. if calibration hangs, keep `csi_target_pps` at `94` or below and inspect temporal occupancy
3. if flash mode is unreliable, switch from `qio` to `dio`

Logger example:

```yaml
logger:
  hardware_uart: UART0
```

Flash mode example:

```yaml
esp32:
  variant: ESP32C3
  flash_mode: dio
```

### Flash failed

1. Hold the `BOOT` button
2. Press `RESET`
3. Release `BOOT`
4. Retry the flash

### View logs

```bash
esphome logs <your-config>.yaml
esphome logs <your-config>.yaml --device espectre.local
```

## Implementation Map

This map is for component maintainers; it is not required for normal installation or tuning.

- [`__init__.py`](components/espectre/__init__.py): YAML schema, validation, codegen, native ESP-IDF component registration, and ESPHome build flags
- [`CMakeLists.txt`](components/espectre/CMakeLists.txt): native ESP-IDF bridge to the canonical shared SDK build definition
- [`espectre.cpp`](components/espectre/espectre.cpp), [`espectre.h`](components/espectre/espectre.h): ESPHome adapter over the shared runtime frontend controller
- [`sensor_publisher.cpp`](components/espectre/sensor_publisher.cpp): movement and motion publishing
- [`threshold_number.cpp`](components/espectre/threshold_number.cpp): runtime threshold control
- [`detector_select.cpp`](components/espectre/detector_select.cpp): persisted runtime detector selection
- [`calibrate_switch.cpp`](components/espectre/calibrate_switch.cpp): runtime recalibration trigger
- [`examples/`](examples/): production, local-development, S3 variant, and Home Assistant dashboard examples

## Packaging Notes

[`__init__.py`](components/espectre/__init__.py) registers this component directory with ESP-IDF's component manager. Its [`CMakeLists.txt`](components/espectre/CMakeLists.txt) reuses the canonical SDK build definition at [`CMakeLists.txt`](../../CMakeLists.txt), so ESPHome compiles `src/cpp/core/` and `src/cpp/runtime/esp_idf/` directly through the native ESP-IDF backend.
