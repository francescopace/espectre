# Setup Guide

This document is the shared setup hub for choosing a frontend and finding the right installation path.

ESPectre now exposes multiple frontends, and each frontend owns its own configuration surface, integration workflow, and troubleshooting. 
This guide keeps only the shared entry points and links you to the frontend-specific source of truth.

Use the `stable` channel for the latest official release, or `main` when you want the newest development snapshot.

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
| `Native` | `ESP32`, `ESP32-S3`, `ESP32-C3`, `ESP32-C5`, `ESP32-C6` | Standalone native frontend exposed over BLE and MQTT, with HTTPS OTA triggered over MQTT |
| `Matter` | `ESP32`, `ESP32-S3`, `ESP32-C3`, `ESP32-C5`, `ESP32-C6` | Requires BLE commissioning, so `ESP32-S2` is excluded; OTA stays in the Matter ecosystem |
| `Streamer` | local build workflow | Not part of the browser flasher path; uses minimal MQTT control plus HTTPS OTA |

### Software

- Chromium-based browser with Web Serial support for browser flashing
- For local workflows, use the repository CLI namespaces documented in each frontend README

### ESP-IDF Local Build Prerequisite

Local `Native`, `Matter`, and `Streamer` firmware builds require ESP-IDF to be
available to the repository CLI.

The repository Python dependencies include ESPHome. ESPHome uses PlatformIO and
can provide a reusable ESP-IDF framework package at
`~/.platformio/packages/framework-espidf` after an ESPHome build has downloaded
it. If that package exists, reuse it instead of installing a second ESP-IDF
copy.

For the current repository baseline, `requirements.txt` pins
`esphome==2026.6.2`, and the matching ESPHome/PlatformIO ESP-IDF framework
package is ESP-IDF `5.5.4` (`framework-espidf` package `3.50504.0`). Use that
same ESP-IDF version for local `Native`, `Matter`, and `Streamer` builds. If
the ESPHome/PlatformIO package does not exist yet, install ESP-IDF `5.5.4` with
the official Espressif setup flow for your host:

- [ESP-IDF Get Started](https://docs.espressif.com/projects/esp-idf/en/stable/esp32/get-started/index.html)

One-time repository setup:

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Fast path for local `Native`, `Matter`, and `Streamer` builds:

1. activate the repository virtual environment
2. run the frontend-specific `build` or `flash` command
3. if ESP-IDF detection fails, run `doctor` for troubleshooting

macOS/Linux:

```bash
source .venv/bin/activate
./espectre native build --chip c3
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
.\espectre.cmd native build --chip c3
```

Optional environment check:

- use `./espectre doctor` or `.\espectre.cmd doctor` when a build fails to find
  or validate ESP-IDF
- use it when you want to see which local ESP-IDF install the wrapper will use

What `doctor` auto-detects today:

| Install source | macOS/Linux | Windows |
|----------------|-------------|---------|
| ESPHome/PlatformIO package | `~/.platformio/packages/framework-espidf` | `%USERPROFILE%\.platformio\packages\framework-espidf` |
| Standard ESP-IDF install | `~/esp/esp-idf` | `%USERPROFILE%\esp\esp-idf` |
| Existing `IDF_PATH` | uses `IDF_PATH` when it points to an ESP-IDF install | uses `IDF_PATH` when it points to an ESP-IDF install |

If a build fails and `doctor` reports that no usable ESP-IDF install was found:

- first choice: reuse the ESP-IDF package downloaded by ESPHome/PlatformIO
- second choice: install official ESP-IDF `5.5.4`, then rerun `doctor`

If the ESPHome/PlatformIO package does not exist yet, any local ESPHome build
will download it:

macOS/Linux:

```bash
./espectre esphome build --chip c3
./espectre doctor
```

Windows PowerShell:

```powershell
.\espectre.cmd esphome build --chip c3
.\espectre.cmd doctor
```

### ESP-IDF Troubleshooting

Use these manual exports only when `doctor` cannot auto-detect or validate your
ESP-IDF install.

macOS/Linux:

```bash
source ~/.platformio/packages/framework-espidf/export.sh
./espectre doctor
```

```bash
source ~/esp/esp-idf/export.sh
./espectre doctor
```

Windows PowerShell:

```powershell
. "$env:USERPROFILE\.platformio\packages\framework-espidf\export.ps1"
.\espectre.cmd doctor
```

```powershell
. "$env:USERPROFILE\esp\esp-idf\export.ps1"
.\espectre.cmd doctor
```

## Local CLI Workflows

Use the repository CLI from the repository root for local build, flash, and monitor tasks.
For `Native`, `Matter`, and `Streamer`, the wrapper auto-detects ESP-IDF during
`build` and `flash`. Use `doctor` only when you want an explicit environment
check or when a build reports an ESP-IDF setup problem:

| Host | CLI launcher |
|------|--------------|
| macOS/Linux | `./espectre` |
| Windows PowerShell/CMD | `.\espectre.cmd` |

| Frontend | Commands | Example |
|----------|----------|---------|
| `ESPHome` | `build`, `flash`, `config`, `monitor` | `./espectre esphome monitor --chip c6 --device /dev/cu.usbmodemXXXX` |
| `Native` | `build`, `flash` | `./espectre native flash --port /dev/cu.usbmodemXXXX` |
| `Matter` | `build`, `flash` | `./espectre matter build --chip c3` |
| `Streamer` | `build`, `flash` | `./espectre streamer flash --port /dev/cu.usbmodemXXXX` |
| `Micro-ESPectre` | `flash`, `deploy`, `run`, `verify` | `./espectre micro deploy` |
| `Host tools` | `collect`, `ui`, `mqtt`, `monitor` | `./espectre collect --stimulus-target 239.1.1.50 --no-save --log-turbulence` |

On Windows, replace `./espectre` with `.\espectre.cmd` and use the COM port shown by Device Manager, for example `COM5`, instead of `/dev/cu...`.

Use the frontend READMEs for complete prerequisites and chip-specific notes:

- [`../src/cpp/frontend/esphome/README.md`](../src/cpp/frontend/esphome/README.md)
- [`../src/cpp/frontend/native/README.md`](../src/cpp/frontend/native/README.md)
- [`../src/cpp/frontend/matter/README.md`](../src/cpp/frontend/matter/README.md)
- [`../src/cpp/frontend/streamer/README.md`](../src/cpp/frontend/streamer/README.md)
- [`../src/python/micro_espectre/README.md`](../src/python/micro_espectre/README.md)

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

If your browser does not support Web Serial, the same page exposes direct download links for manual flashing.

## After Flashing

The next step depends on the frontend you chose:

| Frontend | Continue here | What that README owns |
|----------|---------------|-----------------------|
| `ESPHome` | [`../src/cpp/frontend/esphome/README.md`](../src/cpp/frontend/esphome/README.md) | Wi-Fi provisioning, YAML parameters, Home Assistant entities, dashboards, ESPHome-specific troubleshooting |
| `Native` | [`../src/cpp/frontend/native/README.md`](../src/cpp/frontend/native/README.md) | Build/flash workflow, Wi-Fi and MQTT setup, native control surface, and HTTPS OTA flow |
| `Matter` | [`../src/cpp/frontend/matter/README.md`](../src/cpp/frontend/matter/README.md) | Commissioning flow, Matter surface, Matter-native OTA behavior, and local ESP-IDF workflow |
| `Streamer` | [`../src/cpp/frontend/streamer/README.md`](../src/cpp/frontend/streamer/README.md) | CSI streaming firmware, UDP packet format, Wi-Fi and minimal MQTT setup, and HTTPS OTA flow |

## Shared Runtime Concepts

These concepts are shared across the C++ platform, even though each frontend exposes them differently.

### Detection Algorithms

ESPectre currently supports three detector families:

| Algorithm | Summary | Shared behavior |
|-----------|---------|-----------------|
| `MVS` | Moving-variance detector | Requires startup threshold bootstrap from a quiet room (`max x 1.3`) |
| `L1-Delta` | Normalized profile-displacement detector | Requires startup threshold bootstrap from a quiet room (`max x 1.1`); more stable quiet level across sessions than MVS |
| `ML` | Neural-network detector | Skips threshold bootstrap and starts faster |

The algorithm theory belongs in [ALGORITHMS.md](ALGORITHMS.md). 
Frontend-level configuration syntax belongs in the README of the frontend you are using.

### Startup Behavior

At boot, the shared runtime may perform:

1. AGC-active startup with the shared normalized turbulence path
2. startup calibration for `MVS` and `L1-Delta`, which expects the room to stay quiet for about 10 seconds
3. transition into steady-state motion detection

For practical tuning guidance, sensor placement, and parameter tradeoffs, see [TUNING.md](TUNING.md).

### Traffic Generation

Motion detection frontends depend on CSI packets. 
For the shared detection runtime, traffic is generated internally by default, but the way that traffic is configured or exposed belongs to each frontend surface.

The standalone `streamer` frontend is different: it does not own an internal traffic generator and instead expects collector-driven external UDP stimulus.
Use the streamer frontend README as the source of truth for that workflow and for its Wi-Fi setup options, including BLE-assisted provisioning through the shared ESPectre BLE service.

If you are tuning `traffic_generator_rate`, thresholds, or filters, use [TUNING.md](TUNING.md) for the rationale and the frontend README for the configuration syntax.

## Generic Troubleshooting

These notes apply regardless of frontend surface.

### Wi-Fi driver logs show protocol or bandwidth as unavailable

Some targets do not expose protocol or bandwidth values through every read API.
Logs such as the following do not automatically mean the Wi-Fi connection failed:

```text
WiFi Protocol: unavailable (...)
WiFi Bandwidth: unavailable (...)
```

### CSI packet length warnings (`wrong SC count`)

ESPectre expects HT20 CSI payloads normalized to `128 bytes` (64 subcarriers).
The runtime already remaps several common alternate lengths. If warnings remain frequent, collect the logged metadata and target details before opening an issue.

### Detection does not behave as expected

Before checking frontend-specific settings:

1. confirm the device is connected to 2.4 GHz Wi-Fi
2. confirm startup calibration had a quiet room in `MVS` mode
3. check sensor placement and interference sources
4. continue in [TUNING.md](TUNING.md) and the README of your frontend

## Frontend-Specific Workflows

This guide is intentionally a shared entry point only. 
For build commands, commissioning steps, protocol details, integration behavior, and frontend-level configuration, continue in the local README of the frontend you selected.

## Next Steps

- [README.md](../README.md) for the project overview and documentation map
- [TUNING.md](TUNING.md) for parameter tradeoffs and environment tuning
- [ALGORITHMS.md](ALGORITHMS.md) for signal-processing and detector theory
- [ARCHITECTURE.md](ARCHITECTURE.md) for `core` / `runtime` / `frontend`
  boundaries
