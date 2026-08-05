# Setup Guide

This document is the shared setup hub for choosing a frontend and finding the right installation path.

ESPectre now exposes multiple frontends, and each frontend owns its own configuration surface, integration workflow, and troubleshooting. 
This guide covers the shared entry points and links you to the frontend-specific README for everything else.

Use `Latest Release` for the newest official firmware, or `Release Preview` for the newest development build from `main`. A separate `Developer Preview` GitHub Release is also published from `develop` for pre-main validation, but GitHub Pages continues to expose only `Latest Release` and `Release Preview`.

The SDK now mirrors the same channel model: `stable` is published at
`https://espectre.dev/sdk/stable/`, `snapshot` at
`https://espectre.dev/sdk/main/`, and `snapshot-dev` remains GitHub-only as the
`snapshot-dev` prerelease.

## Choose Your Frontend

| Frontend | Best starting point | Frontend README |
|----------|---------------------|-----------------|
| `ESPHome` | [Web Flash](#web-flash-no-coding-required) for the quickest start, then the frontend README for YAML, Home Assistant, and local development | [`README.md` (esphome)](../src/cpp/frontend/esphome/README.md) |
| `Native` | [Web Flash](#web-flash-no-coding-required) for standalone BLE/MQTT or Home Assistant MQTT Discovery, then the native frontend README for local ESP-IDF workflow | [`README.md` (native)](../src/cpp/frontend/native/README.md) |
| `Matter` | [Web Flash](#web-flash-no-coding-required) for published preview firmware, then the frontend README for commissioning and local ESP-IDF workflow | [`README.md (matter)`](../src/cpp/frontend/matter/README.md) |
| `Streamer` | Frontend README for the dedicated CSI stream workflow | [`README.md`](../src/cpp/frontend/streamer/README.md) |

## Shared Prerequisites

### Hardware

- ESP32 board with CSI support
- USB cable for flashing
- 2.4 GHz Wi-Fi network

Current chip support by frontend:

| Frontend | Supported chips | Delivery |
|----------|-----------------|----------|
| `ESPHome` | `ESP32-S3`, `ESP32-C6`, `ESP32-C5`, `ESP32-C3`, `ESP32` | Published web-flash images |
| `Native` | `ESP32`, `ESP32-S3`, `ESP32-C3`, `ESP32-C5`, `ESP32-C6` | Published web-flash images |
| `Matter` | `ESP32`, `ESP32-S3`, `ESP32-C3`, `ESP32-C5`, `ESP32-C6` | Published web-flash images |
| `Streamer` | `ESP32`, `ESP32-C3`, `ESP32-C5`, `ESP32-C6`, `ESP32-S3` | Local build workflow |

Use the frontend README for the workflow and surface details after you choose the firmware path.

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
./espectre native build --chip c3 --clean
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
.\espectre.cmd native build --chip c3
```

Build cleanup options:

- `--clean` removes only the selected frontend build before rebuilding.
- `--clean-all` removes all builds for that frontend plus shared generated
  artifacts before rebuilding.
- For ESPHome, these flags delegate to the native `esphome clean` and
  `esphome clean-all` commands for the selected config.

Flash note:

- For ESP-IDF frontends, `flash` prefers the build directory that matches the
  connected chip detected on the selected serial port.
- The wrapper still delegates to `idf.py flash`, so ESP-IDF may configure or
  complete the selected build directory before flashing if that build is not
  already ready.

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

Matter generates a unique onboarding identity on first boot and stores it in a
dedicated factory partition. Retrieve the same QR payload after either a web or
CLI flash with:

```bash
./espectre matter qr --port /dev/cu.usbmodemXXXX
```

Normal flashes preserve the QR. Erasing the complete flash generates a new
identity on the next boot.

The same browser applications published on `espectre.dev` can be served from
localhost when a local MQTT broker exposes an insecure `ws://` listener:

```bash
./espectre ui mqtt
./espectre ui ble
./espectre ui theremin
```

The local server preserves the unified site while allowing a browser to connect
to endpoints such as `ws://homeassistant.local:9001/mqtt`. The public HTTPS
pages also allow selecting `ws://` for compatibility testing, but browsers may
block that connection as mixed content. Use `wss://` for a supported hosted
deployment.

See the repository [CLI.md](CLI.md) for:

- launcher syntax on each host
- namespace and command coverage
- shared host-tool behavior, including the interactive MQTT shell
- common wrapper patterns such as `doctor`, serial monitoring, and CLI examples

Use the frontend READMEs for frontend-specific prerequisites, examples, and
chip-specific notes:

- [`README.md` (esphome)](../src/cpp/frontend/esphome/README.md)
- [`README.md` (native)](../src/cpp/frontend/native/README.md)
- [`README.md` (matter)](../src/cpp/frontend/matter/README.md)
- [`README.md` (streamer)](../src/cpp/frontend/streamer/README.md)
- [`README.md` (micro_espectre)](../src/python/micro_espectre/README.md)

## SDK Bundles

If you want to embed the sensing layers into your own firmware instead of
flashing a published frontend, use the SDK bundle channels:

| Channel | Surface | Best for |
|---------|---------|----------|
| `stable` | `https://espectre.dev/sdk/stable/` and semver GitHub Releases | production integrations and reproducible builds |
| `snapshot` | `https://espectre.dev/sdk/main/` and the rolling `snapshot` prerelease | validating the latest `main` changes before release |
| `snapshot-dev` | `snapshot-dev` GitHub prerelease only | pre-main validation from `develop` |

The bundle is source-first. It includes:

- `src/cpp/espectre_sdk.h`, the single include that reaches the supported
  integration surface
- `src/cpp/espectre_sources.cmake` for CMake / ESP-IDF integration
- `src/cpp/library.json` for PlatformIO metadata
- a component-shaped `src/cpp/` root with `CMakeLists.txt`,
  `idf_component.yml`, and `Kconfig.projbuild`, where the optional MQTT, BLE,
  provisioning, OTA, and stream-runtime groups are selected under the
  "ESPectre SDK" menuconfig menu

Use [EMBEDDING.md](EMBEDDING.md) for the actual integration model and runtime
contracts.

## Web Flash (no coding required)

Go to [espectre.dev/flash](https://espectre.dev/flash/) and select:

- the firmware frontend
- the firmware channel
- your target chip

Release and snapshot publishing provide one full-flash image for each supported
chip on the `ESPHome`, `Native`, and `Matter` frontends. GitHub Releases also
provide application-only OTA payloads for Native. GitHub Pages stages only the
full-flash images used by the browser flasher. The published `ESPHome` image
uses the default `Classic` detector, and CI pins its `git_ref` substitution to
the exact source commit used to build the published binary. Subsequent ESPHome
updates are compiled and installed through ESPHome Device Builder; see
[`README.md` (esphome)](../src/cpp/frontend/esphome/README.md) for which
revision an adopted configuration compiles from.

The published `Matter` image also uses the default `Classic` detector. The
`ML` detector remains available through local firmware builds; it is not
published as a separate precompiled image. `Streamer` is also source-built
because its Wi-Fi credentials are supplied at build time.

| Publication surface | Full-flash images | OTA payloads | Manifests |
|---------------------|------------------:|-------------:|-----------|
| GitHub Release or snapshot | 15 | 5 | unified manifest plus 5 Native per-chip OTA manifests |
| GitHub Pages | 15 | 0 | factory-only web-flash manifest |

The OTA payloads are five Native application binaries. ESPHome Device Builder
produces its OTA image from the adopted device configuration. Matter does not
use this OTA flow, and Streamer firmware is not published.

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
| `ESPHome` | [`README.md`](../src/cpp/frontend/esphome/README.md) | Wi-Fi provisioning, YAML parameters, Home Assistant entities, dashboards, ESPHome-specific troubleshooting |
| `Native` | [`README.md`](../src/cpp/frontend/native/README.md) | Build/flash workflow, Wi-Fi and MQTT setup, Home Assistant MQTT Discovery, native control surface, and HTTPS OTA flow |
| `Matter` | [`README.md`](../src/cpp/frontend/matter/README.md) | Commissioning flow, Matter occupancy surface, and local ESP-IDF workflow |
| `Streamer` | [`README.md`](../src/cpp/frontend/streamer/README.md) | CSI streaming firmware, UDP packet format, build-time Wi-Fi setup, and the frontend's intentionally narrow scope |

## Shared Runtime Concepts

These concepts are shared across the C++ platform, even though each frontend exposes them differently.

### Shared Sensing Options

These options belong to the shared sensing runtime and apply to all sensing
frontends. This table is the canonical reference for names, defaults, and
ranges; the exact user-facing syntax differs by frontend:

- `ESPHome`: YAML under `espectre:`
- `Native`: shared ESP-IDF sensing `sdkconfig` menu, with frontend-local overrides in `app/sdkconfig.defaults`
- `Matter`: shared ESP-IDF sensing `sdkconfig` menu, with frontend-local overrides in `app/sdkconfig.defaults`

Support in this phase:

| Frontend | Shared sensing options available |
|----------|----------------------------------|
| `ESPHome` | yes |
| `Native` | yes |
| `Matter` | yes |
| `Streamer` | no, streamer keeps its own stream/collector runtime profile |

| Option | Type / values | Default | Range / notes |
|--------|---------------|---------|---------------|
| `detection_algorithm` | `classic` or `ml` | `classic`, including Matter | Shared detector family |
| Runtime threshold | probability | detector-specific | Selected automatically at startup; adjustable from the frontend during the session |
| `segmentation_window_size` | int | `100` | `100-200` packets |
| `traffic_generator_rate` | int | `100` | Arithmetic validation range `0-100000`; `0` disables internal traffic generation. Supported ESP32 targets sustain much lower practical CSI rates, normally around the `100` target |
| `traffic_generator_adaptive` | bool | `true` | Adjusts DNS or ICMP send pacing from CSI feedback and local socket backpressure; floor at `70%` of target, overshoot up to about `125%` |
| `traffic_generator_mode` | `ping` or `dns` | `ping` | Shared internal traffic generator mode |
| `publish_interval` | int | `100` | `1-1000` packets between periodic updates |
| `evaluation_interval` | int | `25` | `1-1000` packets between detector evaluations |
| `motion_on_hits` | int | `4` | `1-20` consecutive evaluation hits for `IDLE -> MOTION` (about `1.0 s` at the default `100` pps / `25` interval) |
| `motion_off_hits` | int | `3` | `1-20` consecutive evaluation hits for `MOTION -> IDLE` (about `0.75 s` at the same defaults) |
| `lowpass_enabled` | bool | `false` | Enables low-pass filtering |
| `lowpass_cutoff` | float | `11.0` | `5.0-20.0` Hz |
| `hampel_enabled` | bool | `true` | Enables Hampel outlier filtering |
| `hampel_window` | int | `7` | `3-11` samples |
| `hampel_threshold` | float | `5.0` | `1.0-10.0` MAD units |

See [TUNING.md](TUNING.md) for how evaluation cadence and hit filtering set the
expected publish delay (about `1 s` for `IDLE -> MOTION` with the defaults).

Use the frontend README for the exact syntax and local workflow:

- [`README.md` (esphome)](../src/cpp/frontend/esphome/README.md)
- [`README.md` (native)](../src/cpp/frontend/native/README.md)
- [`README.md` (matter)](../src/cpp/frontend/matter/README.md)

### Detection Algorithms And Startup

ESPectre supports two runtime detector families, `classic` and `ml`. At boot,
the sensing path starts with AGC active: `classic` performs startup threshold
calibration, while `ml` starts as soon as CSI capture is ready.

ESPHome and Native can switch detectors at runtime and persist the selection.
The switch resets the threshold to the selected detector's default;
`ml -> classic` starts calibration automatically. Matter remains read-only,
does not consume that persisted selection, and uses its firmware default of
`classic` to keep the published path conservative while the frontend remains
preview. Streamer has no detector.

See:

- [ALGORITHMS.md](ALGORITHMS.md) for detector behavior and formulas
- [TUNING.md](TUNING.md) for the practical startup and threshold workflow,
  including the `quiet -> motion -> quiet` behavior and the quiet-only fallback
- the frontend README for configuration syntax

### Traffic Generation

Motion detection frontends depend on CSI packets. 
For the shared detection runtime, traffic is generated internally by default, but the way that traffic is configured or exposed belongs to each frontend surface.

The standalone `streamer` frontend does not use the internal generator; it is
collector-paced. See the streamer frontend README for that workflow.

If you are tuning `traffic_generator_rate`, thresholds, or filters, use [TUNING.md](TUNING.md) for the rationale and the frontend README for the configuration syntax.

## Frontend-Specific Workflows

For build commands, commissioning steps, protocol details, integration behavior, and frontend-level configuration, continue in the local README of the frontend you selected.

## Next Steps

- [README.md](../README.md) for the project overview and documentation map
- [TUNING.md](TUNING.md) for parameter tradeoffs and environment tuning
- [ALGORITHMS.md](ALGORITHMS.md) for signal-processing and detector theory
- [ARCHITECTURE.md](ARCHITECTURE.md) for `core` / `runtime` / `frontend`
  boundaries
