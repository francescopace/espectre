# Setup Guide

Use this guide to choose a frontend, flash published firmware, or prepare a local build. If you are embedding ESPectre into another firmware product, go directly to [EMBEDDING.md](EMBEDDING.md).

The fastest path is [Web Flash](#web-flash-no-coding-required). Local builds require the repository environment and, for ESP-IDF frontends, the ESP-IDF prerequisite below. Each frontend README owns its configuration and troubleshooting after installation.

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
- Wi-Fi network on a band the board supports: 2.4 GHz on every supported chip, or 5 GHz on the dual-band ESP32-C5. Firmware defaults to 2.4 GHz; an ESP32-C5 integrator can explicitly select 5 GHz or automatic band selection. The runtime pins the selected band or bands to HT20. Detection quality on 5 GHz is not characterized yet

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

### Local Build Prerequisites

Create the repository environment before any local firmware build:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, create the environment with `py -3 -m venv .venv`, activate `.\.venv\Scripts\Activate.ps1`, and run the same install command.

Native, Matter, and Streamer builds prefer a standard local ESP-IDF installation or an active `IDF_PATH` environment and automatically fall back to the pinned ESP-IDF Docker image when no local installation is detected. ESPHome manages its own native ESP-IDF toolchain separately.

```bash
./espectre native build --chip c3
```

On Windows, use `.\espectre.cmd native build --chip c3`. The same pattern applies to Matter and Streamer.

When the local environment is absent and Docker is running, a cached image is used without prompting. If the image is missing, an interactive build asks before downloading it; non-interactive builds must opt in with `--pull missing`. If Docker is installed but stopped, the CLI asks you to start it and retry. Use `--backend local` or `--backend docker` to require one path, and use `./espectre doctor` to inspect only the local ESP-IDF environment.

Docker currently covers builds only. Flashing through the repository CLI still uses local serial tooling and ESP-IDF. If neither build backend is available, either install Docker or install ESP-IDF `5.5.5` with the official [ESP-IDF Get Started](https://docs.espressif.com/projects/esp-idf/en/stable/esp32/get-started/index.html) flow.

#### Optional Compiler Cache

`ccache` is optional but strongly recommended for local ESP-IDF builds, especially Matter. It reuses unchanged compiler output across rebuilds and clean build directories. Repository Docker builds enable a persistent cache automatically, so no host installation is needed for the Docker backend.

Install `ccache` for the local backend:

- macOS with Homebrew: `brew install ccache`
- Debian or Ubuntu Linux: `sudo apt update && sudo apt install ccache`; on other distributions, install the `ccache` package with the system package manager
- Windows: the official ESP-IDF Tools installation includes `ccache`; verify it from an ESP-IDF PowerShell with `ccache --version`. For a manually managed toolchain, install the [official Windows release](https://ccache.dev/download.html) or run `choco install ccache` when Chocolatey is available

Enable it in the current macOS or Linux shell before building:

```bash
export IDF_CCACHE_ENABLE=1
ccache --version
```

Use the equivalent PowerShell environment variable on Windows:

```powershell
$env:IDF_CCACHE_ENABLE = "1"
ccache --version
```

Add the environment variable to the shell profile or user environment to enable it in future terminals. ESP-IDF also accepts `idf.py --ccache` for an individual raw ESP-IDF invocation.

Build cleanup, chip-matched flash selection, and namespace-specific flags are documented in [CLI.md](CLI.md#frontend-workflow-commands).

## Local CLI Workflows

Use the repository CLI from the repository root for local build, flash, monitor, and host-tool tasks.

Matter generates a unique onboarding identity on first boot and stores it in a dedicated factory partition. Retrieve the same QR payload after either a web or CLI flash with:

```bash
./espectre matter qr --port /dev/cu.usbmodemXXXX
```

Normal flashes preserve the QR. Erasing the complete flash generates a new identity on the next boot.

The same browser applications published on `espectre.dev` can be served from localhost when a local MQTT broker exposes an insecure `ws://` listener:

```bash
./espectre ui mqtt
./espectre ui ble
./espectre ui theremin
```

The local server preserves the unified site while allowing a browser to connect to endpoints such as `ws://homeassistant.local:9001/mqtt`. The public HTTPS pages also allow selecting `ws://` for compatibility testing, but browsers may block that connection as mixed content. Use `wss://` for a supported hosted deployment.

See the repository [CLI.md](CLI.md) for:

- launcher syntax on each host
- namespace and command coverage
- shared host-tool behavior, including the interactive MQTT shell
- common wrapper patterns such as `doctor`, serial monitoring, and CLI examples

Use the frontend READMEs for frontend-specific prerequisites, examples, and chip-specific notes:

- [`README.md` (esphome)](../src/cpp/frontend/esphome/README.md)
- [`README.md` (native)](../src/cpp/frontend/native/README.md)
- [`README.md` (matter)](../src/cpp/frontend/matter/README.md)
- [`README.md` (streamer)](../src/cpp/frontend/streamer/README.md)
- [`README.md` (micro_espectre)](../src/python/micro_espectre/README.md)

## Advanced: SDK Bundles

If you want to embed the sensing layers into your own firmware instead of flashing a published frontend, use the SDK bundle channels:

| Channel | Surface | Best for |
|---------|---------|----------|
| `stable` | `https://espectre.dev/artifacts/sdk/stable/` and semver GitHub Releases | production integrations and reproducible builds |
| `snapshot` | `https://espectre.dev/artifacts/sdk/main/` and the rolling `snapshot` prerelease | validating the latest `main` changes before release |
| `snapshot-dev` | `snapshot-dev` GitHub prerelease only | pre-main validation from `develop` |

The bundle is source-first. It includes:

- `src/cpp/espectre_sdk.h`, the single include that reaches the supported integration surface
- `src/cpp/espectre_sources.cmake` for CMake / ESP-IDF integration
- a component-shaped `src/cpp/` root with `CMakeLists.txt`, `idf_component.yml`, and `Kconfig.projbuild`, where the optional MQTT, BLE, provisioning, OTA, and stream-runtime groups are selected under the "ESPectre SDK" menuconfig menu

Use [EMBEDDING.md](EMBEDDING.md) for the actual integration model and runtime contracts.

## Web Flash (no coding required)

Go to [espectre.dev/flash](https://espectre.dev/flash/) and select:

- the firmware frontend
- the firmware channel
- your target chip

Use `Latest Release` for official firmware or `Release Preview` for the latest build from `main`. Published ESPHome firmware starts with Lightweight Detection and supports persisted runtime switching to High Accuracy. Published Matter firmware starts with Lightweight; High Accuracy is available in local Matter builds and is selected at build time. Streamer is source-built because it needs build-time Wi-Fi configuration.

To flash:

1. Connect the board over USB
2. Click **Connect**
3. Select the serial port
4. Confirm the browser prompt

If your browser does not support Web Serial, the same page exposes direct download links for manual flashing.

Website maintainers can find local preview and artifact-staging instructions in [`docs/web/README.md`](web/README.md).

## After Flashing

The next step depends on the frontend you chose:

| Frontend | Continue here | What that README owns |
|----------|---------------|-----------------------|
| `ESPHome` | [`README.md`](../src/cpp/frontend/esphome/README.md) | Wi-Fi provisioning, YAML parameters, Home Assistant entities, dashboards, ESPHome-specific troubleshooting |
| `Native` | [`README.md`](../src/cpp/frontend/native/README.md) | Build/flash workflow, Wi-Fi and MQTT setup, Home Assistant MQTT Discovery, native control surface, and HTTPS OTA flow |
| `Matter` | [`README.md`](../src/cpp/frontend/matter/README.md) | Commissioning flow, Matter occupancy surface, and local ESP-IDF workflow |
| `Streamer` | [`README.md`](../src/cpp/frontend/streamer/README.md) | CSI streaming firmware, UDP packet format, build-time Wi-Fi setup, and the frontend's intentionally narrow scope |

## Reference: Shared Runtime Concepts

These concepts are shared across the C++ platform, even though each frontend exposes them differently.

### Shared Sensing Options

These options belong to the shared sensing runtime and apply to all sensing frontends. This table is the canonical reference for names, defaults, and ranges; the exact user-facing syntax differs by frontend:

- `ESPHome`: YAML under `espectre:`, except the ESP32-C5 band policy, which uses ESPHome's native `wifi.band_mode`
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
| `wifi.band_mode` (ESPHome) / `RuntimeConfig::wifi_band_policy` | `2.4GHz`, `5GHz`, or `AUTO` in ESPHome; `BAND_2G`, `BAND_5G`, or `AUTO` in the SDK | ESPHome C5: `AUTO` when omitted; other frontends: `2.4GHz` | `5GHz` and `AUTO` require the dual-band ESP32-C5; Native can persist the policy over BLE and applies a changed policy after restart; ESPHome examples select `2.4GHz`, and the production PHY remains HT20 |
| `detection_algorithm` | `lightweight` or `high_accuracy` | `lightweight`, including Matter | Lightweight uses less detector CPU and working memory; High Accuracy improves detection quality and skips quiet-room threshold calibration |
| Runtime threshold | probability | detector-specific | Selected automatically at startup; session-adjustable where the frontend exposes a writable control. Matter currently exposes no writable sensing controls |
| `segmentation_window_size_ms` | int | `1000` | `1000-2000` milliseconds; resolved to samples from measured CSI cadence |
| `traffic_generator_rate` | int | `100` | Arithmetic validation range `0-100000`; `0` disables internal traffic generation. Supported ESP32 targets sustain much lower practical CSI rates, normally around the `100` target |
| `traffic_generator_adaptive` | bool | `true` | Adjusts DNS or ICMP send pacing from CSI feedback and local socket backpressure; floor at `70%` of target, overshoot up to about `125%` |
| `traffic_generator_mode` | `ping` or `dns` | `ping` | Shared internal traffic generator mode |
| `publish_interval_ms` | int | `1000` | `100-60000` milliseconds between periodic updates |
| `evaluation_interval_ms` | int | `250` | `10-10000` milliseconds between detector evaluations |
| `motion_on_hits` | int | `4` | `1-20` consecutive evaluation hits for `IDLE -> MOTION` (about `1.0 s` at the default `250 ms` interval) |
| `motion_off_hits` | int | `3` | `1-20` consecutive evaluation hits for `MOTION -> IDLE` (about `0.75 s` at the same defaults) |
| `lowpass_enabled` | bool | `false` | Enables low-pass filtering |
| `lowpass_cutoff` | float | `11.0` | `5.0-20.0` Hz |
| `hampel_enabled` | bool | `true` | Enables Hampel outlier filtering |
| `hampel_window` | int | `7` | `3-11` samples |
| `hampel_threshold` | float | `5.0` | `1.0-10.0` MAD units |

See [TUNING.md](TUNING.md) for how evaluation cadence and hit filtering set the expected publish delay (about `1 s` for `IDLE -> MOTION` with the defaults).

Use the frontend README for the exact syntax and local workflow:

- [`README.md` (esphome)](../src/cpp/frontend/esphome/README.md)
- [`README.md` (native)](../src/cpp/frontend/native/README.md)
- [`README.md` (matter)](../src/cpp/frontend/matter/README.md)

### Detection Profiles And Startup

ESPectre keeps two production detection profiles because no single choice optimizes both accuracy and resource use. Lightweight runs fewer feature trackers and is the leaner choice when the chip or surrounding firmware needs more CPU time and working memory for other work. High Accuracy uses a larger feature state and neural inference to provide higher accuracy and stronger generalization on the maintained corpus.

At boot, Lightweight adapts its threshold to the room during an initial quiet calibration that can take up to about 10 seconds. High Accuracy uses its trained threshold and skips that calibration; it becomes active after CSI capture is ready and the feature window has filled.

ESPHome, Native, and Matter support both `lightweight` and `high_accuracy`. ESPHome and Native can switch profiles at runtime and persist the selection; the switch resets the threshold to the selected profile's default, and `high_accuracy -> lightweight` starts calibration automatically. Matter selects the profile at build time, exposes no runtime detector control, and uses `lightweight` in published firmware while the frontend remains preview. Streamer has no detector.

See:

- [ALGORITHMS.md](ALGORITHMS.md) for detector behavior and formulas
- [TUNING.md](TUNING.md) for the practical startup and threshold workflow, including the `quiet -> motion -> quiet` behavior and the quiet-only fallback
- the frontend README for configuration syntax

### Traffic Generation

Motion detection frontends depend on CSI packets. For the shared detection runtime, traffic is generated internally by default, but the way that traffic is configured or exposed belongs to each frontend surface.

The standalone `streamer` frontend does not use the internal generator; it is collector-paced. See the streamer frontend README for that workflow.

If you are tuning `traffic_generator_rate`, thresholds, or filters, use [TUNING.md](TUNING.md) for the rationale and the frontend README for the configuration syntax.

## Frontend-Specific Workflows

For build commands, commissioning steps, protocol details, integration behavior, and frontend-level configuration, continue in the local README of the frontend you selected.

## Next Steps

- [README.md](../README.md) for the project overview and documentation map
- [TUNING.md](TUNING.md) for parameter tradeoffs and environment tuning
- [ALGORITHMS.md](ALGORITHMS.md) for signal-processing and detector theory
- [ARCHITECTURE.md](ARCHITECTURE.md) for `core` / `runtime` / `frontend` boundaries
