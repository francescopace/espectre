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

Native, Matter, and Streamer builds prefer an active `IDF_PATH` environment, a standard local ESP-IDF installation, or the pinned ESP-IDF toolchain already managed by ESPHome, and automatically fall back to the pinned ESP-IDF Docker image when none is available. Repository ESPHome commands explicitly select its native `esp-idf` toolchain and never use PlatformIO.

```bash
./espectre native build --chip c3
```

On Windows, use `.\espectre.cmd native build --chip c3`. The same pattern applies to Matter and Streamer.

When the local environment is absent and Docker is running, a cached image is used without prompting. If the image is missing, an interactive build asks before downloading it; non-interactive builds must opt in with `--pull missing`. If Docker is installed but stopped, the CLI asks you to start it and retry. Use `--backend local` or `--backend docker` to require one path, and use `./espectre doctor` to inspect only the local ESP-IDF environment.

Docker currently covers builds only. Flashing through the repository CLI still uses local serial tooling and ESP-IDF. If neither build backend is available, build an ESPHome configuration once to provision its native toolchain, install Docker, or install ESP-IDF `5.5.5` with the official [ESP-IDF Get Started](https://docs.espressif.com/projects/esp-idf/en/stable/esp32/get-started/index.html) flow.

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

Browser tools such as Flash, Configure, Monitor, and Theremin live on [espectre.dev](https://espectre.dev). Hosted HTTPS pages should use `wss://` for MQTT over WebSockets. To preview the same site from this repository, including a local `ws://` broker that browsers would block as mixed content from HTTPS, serve `docs/web` as described in [docs/web/README.md](web/README.md).

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
| `release` | `https://espectre.dev/artifacts/sdk/release/` and semver GitHub Releases | production integrations and reproducible builds |
| `preview` | `https://espectre.dev/artifacts/sdk/preview/` and the rolling `snapshot` prerelease | validating the latest `main` changes before release |
| `develop` | `https://espectre.dev/artifacts/sdk/develop/` and the rolling `snapshot-dev` prerelease | pre-main validation from `develop` |

The bundle is source-first. It includes:

- `src/cpp/espectre_sdk.h`, the single include that reaches the supported integration surface
- `src/cpp/espectre_sources.cmake` for CMake / ESP-IDF integration
- a component-shaped `src/cpp/` root with `CMakeLists.txt`, `espectre_git_version.cmake`, `idf_component.yml`, and `Kconfig.projbuild`, where the optional MQTT, BLE, provisioning, OTA, and stream-runtime groups are selected under the "ESPectre SDK" menuconfig menu

Use [EMBEDDING.md](EMBEDDING.md) for the actual integration model and runtime contracts.

## Web Flash (no coding required)

Go to [espectre.dev/flash](https://espectre.dev/flash/) and select:

- the firmware frontend
- the firmware channel
- your target chip

Use `Latest Release` for official firmware, `Release Preview` for the latest build from `main`, or `Development` for the latest build from `develop`. Published ESPHome firmware starts with Lightweight Detection and supports persisted runtime switching to High Accuracy. Published Matter firmware starts with Lightweight; High Accuracy is available in local Matter builds and is selected at build time. Streamer is source-built because it needs build-time Wi-Fi configuration.

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
| `segmentation_window_size_ms` | int | `1000` | `1000-2000` milliseconds; combined with `csi_target_pps` to define a fixed temporal slot window |
| `csi_target_pps` | int | `100` | `1-500`; defines detector slot cadence and the managed-traffic target, but never enables or disables traffic |
| `csi_traffic_mode` | `internal`, `external`, or `disabled` | `internal` | Selects traffic ownership independently from `csi_target_pps`; `disabled` means unmanaged ambient traffic, not disabled sensing. `pacing` is Streamer collector mode only |
| `csi_traffic_multicast_group` | IPv4 multicast address, or empty | `239.255.0.1` | Joined by the UDP listener in `external` and `pacing`. Empty disables the join. Unicast to the device IP still works |
| `traffic_generator_mode` | `ping` or `dns` | `ping` | Shared internal traffic generator mode |
| `publish_interval_ms` | int | `1000` | `100-60000` milliseconds between periodic status-log and diagnostics samples. Canonical MQTT telemetry and Home Assistant Movement Score follow `evaluation_interval_ms` |
| `evaluation_interval_ms` | int | `250` | `10-10000` milliseconds between detector evaluations |
| `motion_on_hits` | int | `4` | `1-20` consecutive evaluation hits for `IDLE -> MOTION` (about `1.0 s` at the default `250 ms` interval) |
| `motion_off_hits` | int | `3` | `1-20` consecutive evaluation hits for `MOTION -> IDLE` (about `0.75 s` at the same defaults) |
| `lowpass_enabled` | bool | `false` | Enables low-pass filtering |
| `lowpass_cutoff` | float | `11.0` | `5.0-20.0` Hz |
| `hampel_enabled` | bool | `true` | Enables Hampel outlier filtering |
| `hampel_window` | int | `7` | `3-11` samples |
| `hampel_threshold` | float | `5.0` | `1.0-10.0` MAD units |

Migration from earlier v3 snapshots: replace `traffic_generator_rate: N` with `csi_target_pps: N` plus `csi_traffic_mode: internal`. Replace the former zero-rate disable sentinel with a positive target plus `csi_traffic_mode: external` when a UDP source supplies traffic, or `disabled` when ambient traffic is intentionally unmanaged.

See [TUNING.md](TUNING.md) for how evaluation cadence and hit filtering set the expected publish delay (about `1 s` for `IDLE -> MOTION` with the defaults).

Use the frontend README for the exact syntax and local workflow:

- [`README.md` (esphome)](../src/cpp/frontend/esphome/README.md)
- [`README.md` (native)](../src/cpp/frontend/native/README.md)
- [`README.md` (matter)](../src/cpp/frontend/matter/README.md)

### Detection Profiles And Startup

ESPectre keeps two production detection profiles because no single choice optimizes both accuracy and resource use. Lightweight runs fewer feature trackers and is the leaner choice when the chip or surrounding firmware needs more CPU time and working memory for other work. High Accuracy uses a larger feature state and neural inference to provide higher accuracy and stronger generalization on the maintained corpus.

At boot, Lightweight adapts its threshold to the room from about 10 seconds of clean, ready CSI coverage after temporal warmup. Missing or burst-concentrated slots extend wall-clock calibration instead of counting as evidence. After that, a long quiet stretch can still lower the live threshold if the opening was noisier than the rest of the session; Home Assistant, ESPHome, and the website Monitor follow that value. High Accuracy uses its trained threshold and skips threshold calibration; it becomes active after CSI capture is ready and the feature window has filled.

ESPHome, Native, and Matter support both `lightweight` and `high_accuracy`. ESPHome and Native can switch profiles at runtime and persist the selection; the switch resets the threshold to the selected profile's default, and `high_accuracy -> lightweight` starts calibration automatically. Matter selects the profile at build time, exposes no runtime detector control, and uses `lightweight` in published firmware while the frontend remains preview. Streamer has no detector.

See:

- [ALGORITHMS.md](ALGORITHMS.md) for detector behavior and formulas
- [TUNING.md](TUNING.md) for the practical startup and threshold workflow, including the `quiet -> motion -> quiet` behavior and the quiet-only fallback
- the frontend README for configuration syntax

### Traffic Generation

Motion detection frontends depend on CSI packets. For the shared detection runtime, traffic is generated internally by default, but the way that traffic is configured or exposed belongs to each frontend surface.

The fixed temporal admission grid accepts at most one packet per target slot. Same-slot bursts are discarded, missing slots remain missing, and the detector becomes ready only after a complete configured window has at least 70% valid occupancy. Arrival-rate jitter does not resize or reconstruct the detector.

Raw rate near `csi_target_pps` does not prove that the target is usable: an AP may deliver those packets in aggregates, producing both same-slot excess and missing slots. If occupancy stays below 70%, fix the traffic source or choose a lower explicit `csi_target_pps` and revalidate detector quality at that cadence. The runtime never lowers the target automatically because doing so would silently change feature timing.

| Path | Target owner | Traffic source | Detector admission | Pacing notes |
|------|--------------|----------------|--------------------|--------------|
| Native / Matter | `CONFIG_ESPECTRE_CSI_TARGET_PPS` | `csi_traffic_mode`; internal by default | yes | fixed send cadence; local socket backoff only |
| ESPHome | `csi_target_pps` | `csi_traffic_mode`; internal by default | yes | fixed send cadence; local socket backoff only |
| Micro-ESPectre | `CSI_TARGET_PPS` | factory default from `TRAFFIC_GENERATOR_ENABLED`, with session-only MQTT overrides for `csi_traffic_mode` and `traffic_generator_mode` | yes | fixed send cadence; local socket backoff only |
| Streamer firmware | collector `--pps` | collector pacing | no; transports raw timestamped CSI | none on device; host collect owns pacing |
| Collector detector, replay, training, and validation | recorded `csi_target_pps`, collector `--pps`, or a documented legacy fallback | recorded raw stream | yes, through the production Micro-ESPectre sampler | collect slows only on TX backpressure; occupancy is telemetry |

Streamer remains collector-paced and preserves raw CSI. The collector applies the same production temporal admission to its live detector and derived sensing view. Host collect slows only on sustained firmware TX backpressure and recovers toward `--pps`; `--fixed` keeps a constant send rate. Occupancy remains telemetry. Firmware pacing credits and raw capture stay independent from the detector grid.

External UDP traffic can be unicast to each device IP, or sent to multicast group `239.255.0.1`. ESP-IDF frontends join that group automatically in `external` and `pacing` (ESPHome, Native, Matter, and Streamer). Empty `csi_traffic_multicast_group` disables the join. Subnet and limited broadcast (`x.x.x.255`, `255.255.255.255`) do not produce reliable HT20 CSI. ESPHome, Native, and Matter `external` mode listen on port `5555`; use [`espectre_traffic_generator.py`](../tools/espectre_traffic_generator.py) with a unicast `TARGETS` list or `TARGETS = ['239.255.0.1']`. Streamer collection listens on port `9999` and can pace several devices with `./espectre collect --target 239.255.0.1`.

Micro-ESPectre keeps its persisted factory default as `TRAFFIC_GENERATOR_ENABLED` plus `TRAFFIC_GENERATOR_MODE`, then exposes session-only MQTT and Home Assistant runtime control over `csi_traffic_mode` and `traffic_generator_mode`. `internal` starts the local generator, and `external` and `disabled` stop it. Micro does not open a UDP listener, so it does not join the multicast group. Sensing MQTT, Home Assistant, ESPHome, and the website do not offer `pacing`; that mode is Streamer collector pacing only.

If you are tuning `csi_target_pps`, thresholds, or filters, use [TUNING.md](TUNING.md) for the rationale and the frontend README for the configuration syntax.

## Frontend-Specific Workflows

For build commands, commissioning steps, protocol details, integration behavior, and frontend-level configuration, continue in the local README of the frontend you selected.

## Next Steps

- [README.md](../README.md) for the project overview and documentation map
- [TUNING.md](TUNING.md) for parameter tradeoffs and environment tuning
- [ALGORITHMS.md](ALGORITHMS.md) for signal-processing and detector theory
- [ARCHITECTURE.md](ARCHITECTURE.md) for `core` / `runtime` / `frontend` boundaries
