# Setup Guide

This document is the shared setup hub for choosing a frontend and finding the
right installation path.

ESPectre now exposes multiple frontends, and each frontend owns its own
configuration surface, integration workflow, and troubleshooting. This guide
keeps only the shared entry points and links you to the frontend-specific
source of truth.

Use the `stable` channel for the latest official release, or `main` when you
want the newest development snapshot.

## Choose Your Frontend

| Frontend | Best starting point | Frontend source of truth |
|----------|---------------------|--------------------------|
| `ESPHome` | [Web Flash](#web-flash-no-coding-required) for the quickest start, then the frontend README for YAML, Home Assistant, and local development | [`../src/cpp/frontend/esphome/README.md`](../src/cpp/frontend/esphome/README.md) |
| `Native` | [Web Flash](#web-flash-no-coding-required) for published firmware, then the native frontend README for local ESP-IDF workflow and [`ESPECTRE_PROTOCOL.md`](ESPECTRE_PROTOCOL.md) for the shared protocol surface over BLE | [`../src/cpp/frontend/native/README.md`](../src/cpp/frontend/native/README.md) |
| `Matter` | [Web Flash](#web-flash-no-coding-required) for published firmware, then the frontend README for commissioning and local ESP-IDF workflow | [`../src/cpp/frontend/matter/README.md`](../src/cpp/frontend/matter/README.md) |
| `Streamer` | Frontend README for the dedicated CSI collection workflow | [`../src/cpp/frontend/streamer/README.md`](../src/cpp/frontend/streamer/README.md) |

## Shared Prerequisites

### Hardware

- ESP32 board with CSI support
- USB cable for flashing
- 2.4 GHz Wi-Fi network

Current entry-point support by frontend:

| Frontend | Supported published targets | Notes |
|----------|-----------------------------|-------|
| `ESPHome` | `ESP32-S3`, `ESP32-C6`, `ESP32-C5`, `ESP32-C3`, `ESP32`, `ESP32-S2` (experimental) | Web flasher supports the default `MVS` detector and `ML` assets |
| `Native` | `ESP32`, `ESP32-S3`, `ESP32-C3`, `ESP32-C5`, `ESP32-C6` | Standalone native frontend exposed over BLE and MQTT |
| `Matter` | `ESP32`, `ESP32-S3`, `ESP32-C3`, `ESP32-C5`, `ESP32-C6` | Requires BLE commissioning, so `ESP32-S2` is excluded |
| `Streamer` | local build workflow | Not part of the browser flasher path |

### Software

- Chromium-based browser with Web Serial support for browser flashing
- For local workflows, use the repository CLI namespaces documented in each
  frontend README

## Local CLI Workflows

Use `./espectre` from the repository root for local build, flash, and monitor
tasks:

| Frontend | Commands | Example |
|----------|----------|---------|
| `ESPHome` | `build`, `flash`, `config`, `logs` | `./espectre esphome logs --chip c6 --device /dev/cu.usbmodemXXXX` |
| `Native` | `build`, `flash`, `monitor` | `./espectre native flash --chip c3 --port /dev/cu.usbmodemXXXX` |
| `Matter` | `build`, `flash`, `monitor` | `./espectre matter build --chip c3` |
| `Streamer` | `build`, `flash`, `monitor` | `./espectre streamer monitor --chip s3 --port /dev/cu.usbmodemXXXX` |
| `Micro-ESPectre` | `flash`, `deploy`, `run`, `verify`, `detect`, `collect`, `ui` | `./espectre micro deploy` |

Use the frontend READMEs for complete prerequisites and chip-specific notes:

- [`../src/cpp/frontend/esphome/README.md`](../src/cpp/frontend/esphome/README.md)
- [`../src/cpp/frontend/native/README.md`](../src/cpp/frontend/native/README.md)
- [`../src/cpp/frontend/matter/README.md`](../src/cpp/frontend/matter/README.md)
- [`../src/cpp/frontend/streamer/README.md`](../src/cpp/frontend/streamer/README.md)
- [`MICRO_ESPECTRE.md`](MICRO_ESPECTRE.md)

## Web Flash (no coding required)

Go to [espectre.dev/flash](https://espectre.dev/flash/) and select:

- the firmware frontend
- the firmware channel
- your target chip

For the `ESPHome` frontend, the web flasher also exposes detector variants:

- `MVS` for the default variance-based detector
- `ML` for the neural-network detector assets

To flash:

1. Connect the board over USB
2. Click **Connect**
3. Select the serial port
4. Confirm the browser prompt

If your browser does not support Web Serial, the same page exposes direct
download links for manual flashing.

## After Flashing

The next step depends on the frontend you chose:

| Frontend | Continue here | What that README owns |
|----------|---------------|-----------------------|
| `ESPHome` | [`../src/cpp/frontend/esphome/README.md`](../src/cpp/frontend/esphome/README.md) | Wi-Fi provisioning, YAML parameters, Home Assistant entities, dashboards, ESPHome-specific troubleshooting |
| `Native` | [`../src/cpp/frontend/native/README.md`](../src/cpp/frontend/native/README.md) | Build/flash workflow, Wi-Fi build-time configuration, firmware limits, and links to the shared protocol spec |
| `Matter` | [`../src/cpp/frontend/matter/README.md`](../src/cpp/frontend/matter/README.md) | Commissioning flow, Matter surface, writable attributes, local ESP-IDF workflow |
| `Streamer` | [`../src/cpp/frontend/streamer/README.md`](../src/cpp/frontend/streamer/README.md) | CSI streaming firmware, UDP packet format, Wi-Fi setup via `sdkconfig.wifi` or BLE provisioning, frontend-specific configuration |

## Shared Runtime Concepts

These concepts are shared across the C++ platform, even though each frontend
exposes them differently.

### Detection Algorithms

ESPectre currently supports two detector families:

| Algorithm | Summary | Shared behavior |
|-----------|---------|-----------------|
| `MVS` | Moving-variance detector | Requires startup threshold bootstrap from a quiet room |
| `ML` | Neural-network detector | Skips threshold bootstrap and starts faster |

The algorithm theory belongs in [ALGORITHMS.md](ALGORITHMS.md). Frontend-level
configuration syntax belongs in the README of the frontend you are using.

### Startup Behavior

At boot, the shared runtime may perform:

1. gain lock, when supported by the chip and enabled by the frontend surface
2. startup calibration for `MVS`, which expects the room to stay quiet for
   about 10 seconds
3. transition into steady-state motion detection

For practical tuning guidance, sensor placement, and parameter tradeoffs, see
[TUNING.md](TUNING.md).

### Traffic Generation

Motion detection frontends depend on CSI packets. For the shared detection
runtime, traffic is generated internally by default, but the way that traffic is
configured or exposed belongs to each frontend surface.

The standalone `streamer` frontend is different: it does not own an internal
traffic generator and instead expects collector-driven external UDP stimulus.
Use the streamer frontend README as the source of truth for that workflow and
for its Wi-Fi setup options, including BLE-assisted provisioning through the
shared ESPectre BLE service.

If you are tuning `traffic_generator_rate`, thresholds, or filters, use
[TUNING.md](TUNING.md) for the rationale and the frontend README for the
configuration syntax.

## Generic Troubleshooting

These notes apply regardless of frontend surface.

### Wi-Fi driver logs show protocol or bandwidth as unavailable

Some targets do not expose protocol or bandwidth values through every read API.
Logs such as the following do not automatically mean the Wi-Fi connection
failed:

```text
WiFi Protocol: unavailable (...)
WiFi Bandwidth: unavailable (...)
```

### CSI packet length warnings (`wrong SC count`)

ESPectre expects HT20 CSI payloads normalized to `128 bytes` (64 subcarriers).
The runtime already remaps several common alternate lengths. If warnings remain
frequent, collect the logged metadata and target details before opening an
issue.

### Detection does not behave as expected

Before checking frontend-specific settings:

1. confirm the device is connected to 2.4 GHz Wi-Fi
2. confirm startup calibration had a quiet room in `MVS` mode
3. check sensor placement and interference sources
4. continue in [TUNING.md](TUNING.md) and the README of your frontend

## Frontend-Specific Workflows

This guide is intentionally a shared entry point only. For build commands,
commissioning steps, protocol details, integration behavior, and frontend-level
configuration, continue in the local README of the frontend you selected.

## Next Steps

- [README.md](../README.md) for the project overview and documentation map
- [TUNING.md](TUNING.md) for parameter tradeoffs and environment tuning
- [ALGORITHMS.md](ALGORITHMS.md) for signal-processing and detector theory
- [ARCHITECTURE.md](ARCHITECTURE.md) for `core` / `runtime` / `frontend`
  boundaries

## License

GPLv3 - See [LICENSE](../LICENSE) for details.
