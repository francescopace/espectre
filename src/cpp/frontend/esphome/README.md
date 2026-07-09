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

- [`espectre/__init__.py`](espectre/__init__.py):
  YAML schema, validation, codegen, shared local PlatformIO library registration, and ESPHome build flags
- [`espectre/espectre.cpp`](espectre/espectre.cpp),
  [`espectre/espectre.h`](espectre/espectre.h):
  ESPHome adapter over the shared runtime frontend controller
- [`espectre/sensor_publisher.cpp`](espectre/sensor_publisher.cpp):
  movement and motion publishing
- [`espectre/threshold_number.cpp`](espectre/threshold_number.cpp):
  runtime threshold control
- [`espectre/calibrate_switch.cpp`](espectre/calibrate_switch.cpp):
  runtime recalibration trigger

## Getting Started

If you want the browser-flash path, start from
[`../../../../docs/SETUP.md`](../../../../docs/SETUP.md) and come back here
after flashing `ESPHome`.

After flashing, configure Wi-Fi with one of these provisioning paths:

| Method | How |
|--------|-----|
| BLE | Use the ESPHome or Home Assistant Companion app |
| USB | Go to [web.esphome.io](https://web.esphome.io) and use **Connect** -> **Configure WiFi** |
| Captive portal | Connect to the `ESPectre Fallback` network and finish setup in the browser |

Once Wi-Fi is configured, the device is discovered automatically by Home
Assistant through ESPHome.

## Integration Surface

The frontend maps runtime state into ESPHome and Home Assistant entities.

| Runtime state/event | ESPHome surface |
|---------------------|-----------------|
| movement metric | `movement_sensor` |
| motion state | `motion_sensor` |
| runtime threshold write | `threshold_number` |
| runtime recalibration trigger | `calibrate_switch` |

The default entities are created automatically when the `espectre:` component
is declared.

## Configuration Surface

The ESPHome YAML schema is defined in [`espectre/__init__.py`](espectre/__init__.py).
This README is the source of truth for ESPHome-specific syntax and entity
mapping. Use [`../../../../docs/SETUP.md`](../../../../docs/SETUP.md) for the
shared configuration overview and [`../../../../docs/TUNING.md`](../../../../docs/TUNING.md)
for the "when and why" of tuning.

### Core Parameters

All frontend parameters live under the `espectre:` section:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `detection_algorithm` | string | `classic` | Detection algorithm: `classic` or `ml` |
| `traffic_generator_rate` | int | `100` | Packets per second for CSI generation (`0-1000`, `0` disables the internal generator) |
| `traffic_generator_mode` | string | `ping` | Traffic generator mode: `ping` or `dns` |
| `publish_interval` | int | `auto` | Packets between periodic movement/log updates |
| `evaluation_interval` | int | `25` | Packets between internal detector evaluations |
| `motion_on_hits` | int | `3` | Consecutive hits required before entering `MOTION` |
| `motion_off_hits` | int | `3` | Consecutive hits required before returning to `IDLE` |
| `segmentation_threshold` | string/float | `auto` | Threshold mode: `auto`, `min`, or a numeric manual threshold (`classic`: `0.0-10.0`, `ml`: `0.0-1.0`) |
| `segmentation_window_size` | int | `100` | Shared detector window in packets for classic variance recovery and ML features (`10-200`) |
| `lowpass_enabled` | bool | `false` | Enable low-pass filtering |
| `lowpass_cutoff` | float | `11.0` | Low-pass cutoff in Hz (`5-20`) |
| `hampel_enabled` | bool | `true` | Enable Hampel outlier filtering |
| `hampel_window` | int | `7` | Hampel window size (`3-11`) |
| `hampel_threshold` | float | `5.0` | Hampel sensitivity (`1.0-10.0`) |
These options are applied from YAML during firmware configuration. Runtime
control is exposed separately through the entities below:

| Runtime surface | Config key | Runtime behavior |
|-----------------|------------|------------------|
| Movement score | `movement_sensor` | Read-only Home Assistant sensor |
| Motion state | `motion_sensor` | Read-only Home Assistant binary sensor |
| Threshold | `threshold_number` | Writable runtime threshold control |
| Recalibration | `calibrate_switch` | Writable runtime recalibration trigger |

### Detection Algorithm Selection

| Algorithm | Summary | Shared behavior |
|-----------|---------|-----------------|
| `classic` | L1-Delta primary with variance recovery | Adaptive startup threshold bootstrap |
| `ml` | Neural-network detector | Faster boot, no threshold bootstrap |

```yaml
espectre:
  detection_algorithm: classic  # or ml
```

Threshold behavior:

- range: `classic` `0.0-10.0`, `ml` `0.0-1.0`
- `classic` default: `auto` (shared adaptive startup calibration; motion-first with internal quiet-first fallback)
- `ml` default: `0.5`

### Example

```yaml
espectre:
  detection_algorithm: classic
  traffic_generator_rate: 100
  traffic_generator_mode: ping
  segmentation_threshold: auto
  segmentation_window_size: 100
  motion_on_hits: 3
  motion_off_hits: 3
```

## Entity Customization

### Integrated Entities

| Sensor config | Type | Default name | Description |
|---------------|------|--------------|-------------|
| `movement_sensor` | sensor | `Movement Score` | Current movement score |
| `motion_sensor` | binary_sensor | `Motion Detected` | Edge-driven motion state |
| `threshold_number` | number | `Threshold` | Runtime threshold control |
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

### Dashboard Examples

Examples live in:

| File | Description |
|------|-------------|
| [`../../../../examples/home-assistant-dashboard.yaml`](../../../../examples/home-assistant-dashboard.yaml) | Production dashboard with motion entities |
| [`../../../../examples/home-assistant-dashboard-dev.yaml`](../../../../examples/home-assistant-dashboard-dev.yaml) | Development dashboard with debug entities |

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
  traffic_generator_mode: ping
```

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
[`../../../../examples/espectre_traffic_generator.py`](../../../../examples/espectre_traffic_generator.py)
to drive one or more devices from the network.

For rate recommendations, airtime tradeoffs, and placement guidance, see
[`../../../../docs/TUNING.md`](../../../../docs/TUNING.md).

## Startup Calibration

In `classic` mode, keep the room quiet after boot so the runtime can
complete the startup threshold bootstrap.

Startup behavior:

1. AGC-active startup with detector-specific normalized metrics
2. adaptive threshold bootstrap for `classic`
3. normal motion detection loop

With the default `segmentation_window_size: 100`, `classic` uses a startup
budget of up to `1000` packets. This is a maximum, not a fixed wait, so clean
motion-first startups may finish earlier. `ML` skips threshold bootstrap.

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

Use [`CLI.md`](../../../../docs/CLI.md) as the source of truth
for shared CLI syntax, host-side tools, and wrapper behavior.

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
4. Lower `segmentation_threshold` if the detector is too conservative

### False positives

1. Raise `segmentation_threshold`
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

[`espectre/__init__.py`](espectre/__init__.py) registers the local
[`../../library.json`](../../library.json) package so PlatformIO builds the
canonical shared sources directly from `src/cpp/core/` and
`src/cpp/runtime/esp_idf/`. This keeps ESPHome packaging aligned with the main
repository layout across platforms.
