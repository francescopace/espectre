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
| `Streamer` | Frontend README for the dedicated CSI stream workflow | [`../src/cpp/frontend/streamer/README.md`](../src/cpp/frontend/streamer/README.md) |

## Shared Prerequisites

### Hardware

- ESP32 board with CSI support
- USB cable for flashing
- 2.4 GHz Wi-Fi network

Current entry-point support by frontend:

| Frontend | Supported published targets | Notes |
|----------|-----------------------------|-------|
| `ESPHome` | `ESP32-S3`, `ESP32-C6`, `ESP32-C5`, `ESP32-C3`, `ESP32`, `ESP32-S2` (experimental) | Published web-flash images use the default detector profile; the frontend README covers `classic` and `ml` configuration |
| `Native` | `ESP32`, `ESP32-S3`, `ESP32-C3`, `ESP32-C5`, `ESP32-C6` | Standalone native frontend exposed over BLE and MQTT, with HTTPS OTA triggered over MQTT |
| `Matter` | `ESP32`, `ESP32-S3`, `ESP32-C3`, `ESP32-C5`, `ESP32-C6` | Requires BLE commissioning, so `ESP32-S2` is excluded; OTA stays in the Matter ecosystem |
| `Streamer` | local build workflow | Not part of the browser flasher path |

### Software

- Chromium-based browser with Web Serial support for browser flashing
- For local workflows, use the repository [CLI.md](CLI.md) plus the relevant frontend README

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

Use the repository CLI from the repository root for local build, flash, monitor,
and host-tool tasks.

The repository [CLI.md](CLI.md) is the source of truth for:

- launcher syntax on each host
- namespace and command coverage
- shared host-tool behavior, including the interactive MQTT shell
- common wrapper patterns such as `doctor`, serial monitoring, and CLI examples

Use the frontend READMEs for frontend-specific prerequisites, examples, and
notes that depend on the selected firmware surface.

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

- `Classic` for the default non-ML detector
- `ML` for the neural-network detector assets

To flash:

1. Connect the board over USB
2. Click **Connect**
3. Select the serial port
4. Confirm the browser prompt

If your browser does not support Web Serial, the same page exposes direct download links for manual flashing.

For local preview of the web flasher with same-origin firmware assets:

1. Build the firmware you want to test, or download the published binaries into a local directory.
2. Stage a channel manifest and matching binaries under `docs/web/flash/firmware/<channel>/`:

```bash
python .github/scripts/stage_web_firmware.py \
  --firmware-dir /path/to/firmware \
  --output-dir docs/web/flash/firmware/stable \
  --channel stable \
  --version 3.0.0 \
  --release-tag 3.0.0 \
  --url-prefix /flash/firmware/stable
```

3. Serve the site root locally:

```bash
python -m http.server 8080 --directory docs/web
```

4. Open `http://localhost:8080/flash/` in a Chromium-based browser and verify the selected firmware resolves from `/flash/firmware/...`.

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

ESPectre currently supports two runtime detector families:

| Algorithm | Summary | Shared behavior |
|-----------|---------|-----------------|
| `Classic` | L1-delta primary with variance recovery | Uses startup threshold calibration |
| `ML` | Neural-network detector | Starts without threshold bootstrap |

Use:

- [ALGORITHMS.md](ALGORITHMS.md) for detector behavior and formulas
- [TUNING.md](TUNING.md) for the practical startup and threshold workflow
- the frontend README for configuration syntax

### Startup Behavior

At boot:

1. the sensing path starts with AGC active
2. `classic` performs startup threshold calibration
3. `ml` starts once CSI capture is ready
4. the runtime transitions into steady-state detection

Keep this document at the entry-point level. For the actual startup guidance,
including the `quiet -> motion -> quiet` behavior and the quiet-only fallback,
use [TUNING.md](TUNING.md).

### Traffic Generation

Motion detection frontends depend on CSI packets. 
For the shared detection runtime, traffic is generated internally by default, but the way that traffic is configured or exposed belongs to each frontend surface.

The standalone `streamer` frontend is collector-paced: the host sends ordinary
UDP traffic, the device learns the collector IP from the packet source address,
and the stream-specific runtime backend returns one CSI datagram toward the
collector for each accepted pacing step.
Use the streamer frontend README as the source of truth for that workflow and
for its Wi-Fi setup options via the active `sdkconfig` defaults.

If you are tuning `traffic_generator_rate`, thresholds, or filters, use [TUNING.md](TUNING.md) for the rationale and the frontend README for the configuration syntax.

## Frontend-Specific Workflows

This guide is intentionally a shared entry point only. 
For build commands, commissioning steps, protocol details, integration behavior, and frontend-level configuration, continue in the local README of the frontend you selected.

## Next Steps

- [README.md](../README.md) for the project overview and documentation map
- [TUNING.md](TUNING.md) for parameter tradeoffs and environment tuning
- [ALGORITHMS.md](ALGORITHMS.md) for signal-processing and detector theory
- [ARCHITECTURE.md](ARCHITECTURE.md) for `core` / `runtime` / `frontend`
  boundaries
