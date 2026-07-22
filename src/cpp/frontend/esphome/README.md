# ESPectre ESPHome Frontend

This directory contains the ESPHome frontend for ESPectre.

It is the production-oriented integration surface for Home Assistant and keeps
the ESPHome-specific concerns separate from the shared `core` and `runtime`
layers.

## Scope

The ESPHome frontend is responsible for:

- YAML schema and code generation
- packaging as an ESPHome `external_components` source
- Home Assistant-facing entities
- ESPHome provisioning and dashboard-oriented usage
- ESPHome-specific SDK configuration defaults and troubleshooting

## Directory Layout

- [`__init__.py`](espectre/__init__.py):
  YAML schema, validation, codegen, shared local PlatformIO library registration, and ESPHome build flags
- [`espectre.cpp`](espectre/espectre.cpp),
  [`espectre.h`](espectre/espectre.h):
  ESPHome adapter over the shared runtime frontend controller
- [`sensor_publisher.cpp`](espectre/sensor_publisher.cpp):
  movement and motion publishing
- [`threshold_number.cpp`](espectre/threshold_number.cpp):
  runtime threshold control
- [`detector_select.cpp`](espectre/detector_select.cpp):
  persisted runtime detector selection
- [`calibrate_switch.cpp`](espectre/calibrate_switch.cpp):
  runtime recalibration trigger

## Getting Started

If you want the browser-flash path, start from
[`SETUP.md`](../../../../docs/SETUP.md) and come back here
after flashing `ESPHome`.

After flashing, configure Wi-Fi with one of these provisioning paths:

| Method | How |
|--------|-----|
| BLE | Use the ESPHome or Home Assistant Companion app |
| USB | Go to [web.esphome.io](https://web.esphome.io) and use **Connect** -> **Configure WiFi** |
| Captive portal | Connect to the `ESPectre Fallback` network and finish setup in the browser |

Once Wi-Fi is configured, the device is discovered automatically by Home
Assistant through ESPHome.

Release and snapshot channels publish one full-flash image per supported chip,
using the default `classic` detector. After adoption, ESPHome Device Builder
compiles and installs updates wirelessly from the device YAML. To use `ml`,
build locally with `detection_algorithm: ml`; there is no separate precompiled
ML image.

## Integration Surface

The frontend maps runtime state into ESPHome and Home Assistant entities.

| Runtime state/event | ESPHome surface |
|---------------------|-----------------|
| movement metric | `movement_sensor` |
| motion state | `motion_sensor` |
| runtime threshold write | `threshold_number` |
| runtime detector selection | `detector_select` |
| runtime recalibration trigger | `calibrate_switch` |

The default entities are created automatically when the `espectre:` component
is declared.

## Configuration Surface

The ESPHome YAML schema is defined in [`__init__.py`](espectre/__init__.py).
This README covers ESPHome-specific syntax and entity mapping. See
[`SETUP.md`](../../../../docs/SETUP.md) for the shared configuration overview
and [`TUNING.md`](../../../../docs/TUNING.md) for the "when and why" of tuning.

### Core Parameters

The shared sensing options, with their defaults and ranges, are documented in
the [`Shared Sensing Options`](../../../../docs/SETUP.md#shared-sensing-options)
table in `SETUP.md`. In ESPHome, those options live under the `espectre:`
section with the same names, as shown in the example below.

These options are applied from YAML during firmware configuration. Runtime
control is exposed separately through the entities below:

| Runtime surface | Config key | Runtime behavior |
|-----------------|------------|------------------|
| Movement score | `movement_sensor` | Read-only Home Assistant sensor |
| Motion state | `motion_sensor` | Read-only Home Assistant binary sensor |
| Threshold | `threshold_number` | Writable runtime threshold control |
| Detector | `detector_select` | Writable, persisted `classic` / `ml` selection |
| Recalibration | `calibrate_switch` | Writable runtime recalibration trigger |

### Detection Algorithm Selection

```yaml
espectre:
  detection_algorithm: classic  # or ml
```

Threshold behavior:

- range: `0.0-1.0` for both detectors
- `classic`: automatic session-adapted startup threshold
- `ml` default: `0.5`

The YAML value is the initial detector when no persisted selection exists.
The Home Assistant `detector_select` changes it live and persists the choice
across reboot. `ml -> classic` starts calibration automatically, and the
`calibrate_switch` reflects automatic and user-triggered calibration state.

See [`ALGORITHMS.md`](../../../../docs/ALGORITHMS.md) for how the two
detectors differ and [`TUNING.md`](../../../../docs/TUNING.md) for choosing
between them.

### Example

```yaml
espectre:
  detection_algorithm: classic
  traffic_generator_rate: 100
  traffic_generator_adaptive: true
  traffic_generator_mode: ping
  segmentation_window_size: 100
  motion_on_hits: 4
  motion_off_hits: 3
```

## Entity Customization

### Integrated Entities

| Sensor config | Type | Default name | Description |
|---------------|------|--------------|-------------|
| `movement_sensor` | sensor | `Movement Score` | Current movement score (0.0–1.0) |
| `motion_sensor` | binary_sensor | `Motion Detected` | Edge-driven motion state |
| `threshold_number` | number | `Threshold` | Runtime probability threshold (0.0–1.0) |
| `detector_select` | select | `Detector` | Runtime `classic` / `ml` selection |
| `calibrate_switch` | switch | `Calibrate` | Startup recalibration trigger |

All entities support standard ESPHome options such as:

- `name`
- `internal`
- `icon`
- `disabled_by_default`

The `movement_sensor` also supports ESPHome
[sensor filters](https://esphome.io/components/sensor/#sensor-filters).

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

Use `internal: true` on `movement_sensor` when you want to keep the binary
motion entity for automations without publishing the raw score to Home
Assistant.

## Home Assistant Integration

Once the device is flashed and connected to Wi-Fi:

1. Home Assistant discovers it through ESPHome
2. Go to **Settings** -> **Devices & Services** -> **ESPHome**
3. Configure the discovered device
4. The default entities are added automatically

The ESPHome frontend exposes movement, motion, threshold control, and
recalibration as Home Assistant entities.

To manage configuration and OTA updates, install ESPHome Device Builder and
adopt the discovered device. The imported configuration keeps the Git ref
embedded by the installed firmware: release builds remain pinned to their
release tag, while snapshot builds remain pinned to their source commit. Change
the ref after `@` in the adopted `packages` URL when you want ESPHome to compile
and install a newer version:

```yaml
packages:
  francescopace.espectre: github://francescopace/espectre/examples/espectre-c6.yaml@3.0.0
```

### Dashboard Examples

Examples live in:

| File | Description |
|------|-------------|
| [`home-assistant-dashboard.yaml`](../../../../examples/home-assistant-dashboard.yaml) | Production dashboard with motion entities |

To import a dashboard:

1. Go to **Settings** -> **Dashboards** -> **Add Dashboard**
2. Open the dashboard and choose **Edit**
3. Open the raw configuration editor
4. Replace the default content with the YAML from the example file
5. Save the dashboard

If you changed the device name from `espectre`, update entity IDs in the YAML.
If you enabled `name_add_mac_suffix: true`, include the MAC suffix in the
entity names as well.

## Traffic Generator and Runtime Notes

The ESPHome surface exposes the shared runtime traffic-generation settings.
By default, the device continuously generates traffic for CSI collection while
powered on.

### Internal Traffic Generator

```yaml
espectre:
  traffic_generator_rate: 100
  traffic_generator_adaptive: true
  traffic_generator_mode: ping
```

`traffic_generator_rate` is the target rate of valid local CSI callbacks. The
adaptive controller is enabled by default and changes the network send pace to
hold that target. Set `traffic_generator_adaptive: false` to interpret the
configured rate as a fixed DNS or ICMP send rate.

Available modes:

| Mode | Protocol | Notes |
|------|----------|-------|
| `ping` | ICMP | Default and usually the safest choice |
| `dns` | UDP | Lower-overhead alternative when the router responds consistently |

### External Traffic Mode

To disable the internal generator and rely on external traffic:

```yaml
espectre:
  traffic_generator_rate: 0
  publish_interval: 100
  evaluation_interval: 25
```

In that mode the runtime opens a UDP listener on port `5555`. Use
[`espectre_traffic_generator.py`](../../../../tools/espectre_traffic_generator.py)
to drive one or more devices from the network.

For rate recommendations, airtime tradeoffs, and placement guidance, see
[`TUNING.md`](../../../../docs/TUNING.md).

## Startup Calibration

In `classic` mode, keep the room quiet after boot so the runtime can complete
the startup threshold bootstrap; `ml` skips the bootstrap and starts as soon
as CSI capture is ready. For the startup workflow and budget details, see
[`TUNING.md`](../../../../docs/TUNING.md).

Runtime recalibration is exposed as the `calibrate_switch` entity in Home
Assistant.

## Build and Consumption

### As an ESPHome external component

Production examples consume this frontend with:

```yaml
external_components:
  - source:
      type: git
      url: https://github.com/francescopace/espectre
      path: src/cpp/frontend/esphome
    components: [espectre]
```

Local development examples consume it with:

```yaml
external_components:
  - source:
      type: local
      path: ../src/cpp/frontend/esphome
    components: [espectre]
```

### Repository CLI

See [`CLI.md`](../../../../docs/CLI.md) for shared CLI syntax, host-side
tools, and wrapper behavior.

```bash
./espectre esphome build --chip c6 --clean
./espectre esphome flash --chip c6
./espectre esphome config --chip c6
./espectre esphome monitor --chip c6 --device /dev/cu.usbmodemXXXX
```

On Windows, use `.\espectre.cmd esphome ...` from the repository root and pass
a COM port such as `COM5` to `--device` when serial access is needed.

Add `--dev` to use the local development YAML mapping.
Use `flash` for upload-only and `monitor` for logs.

## Hardware and Packaging Notes

### Automatic SDK Configuration

The frontend automatically sets the ESP-IDF options required by the runtime,
including CSI enablement and timing-related defaults. In most cases you do not
need to set these manually.

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

The ESPHome frontend fits in `4 MB` flash with OTA. It uses the board and
framework default partition table unless you override it in your own project.

If you need a custom table:

```yaml
esphome:
  name: my-device
  platformio_options:
    board_build.partitions: /absolute/path/to/partitions_custom.csv
```

The frontend itself does not require a custom partition table.

## ESPHome-Specific Troubleshooting

### No motion detection

1. Verify Wi-Fi is connected
2. Verify traffic generation is active, or provide external traffic
3. Wait for startup calibration to complete in `classic`
4. Lower the Threshold number entity if the detector is too conservative

### False positives

1. Raise the Threshold number entity
2. Check for fans, AC, curtains, or other interference
3. Increase `segmentation_window_size` for more stability

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
2. if calibration hangs, keep `traffic_generator_rate` at `94` or below
3. if flash mode is unreliable, switch from `qio` to `dio`

Logger example:

```yaml
logger:
  hardware_uart: UART0
```

Flash mode example:

```yaml
esphome:
  platformio_options:
    board_build.flash_mode: dio
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

## Packaging Notes

[`__init__.py`](espectre/__init__.py) registers the local
[`library.json`](../../library.json) package so PlatformIO builds the
canonical shared sources directly from `src/cpp/core/` and
`src/cpp/runtime/esp_idf/`. This keeps ESPHome packaging aligned with the main
repository layout across platforms.
