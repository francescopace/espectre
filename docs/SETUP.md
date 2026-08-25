# Setup Guide

Choose a frontend first. If it has a published image, [Web Flash](#web-flash-no-coding-required) is the shortest installation path. Local builds use the repository environment and the relevant frontend README. Firmware integrators should start with [EMBEDDING.md](EMBEDDING.md).

## Choose Your Frontend

| Frontend | Best starting point | Frontend README |
|----------|---------------------|-----------------|
| `ESPHome` | [Web Flash](#web-flash-no-coding-required), Home Assistant entities, and Direct HTTP runtime tuning | [`README.md` (esphome)](../src/cpp/frontend/esphome/README.md) |
| `Native` | [Web Flash](#web-flash-no-coding-required), Improv Serial Wi-Fi provisioning, Direct HTTP, and optional MQTT or Home Assistant MQTT Discovery | [`README.md` (native)](../src/cpp/frontend/native/README.md) |
| `Matter` | [Web Flash](#web-flash-no-coding-required), Matter commissioning, and Direct HTTP detector tuning | [`README.md (matter)`](../src/cpp/frontend/matter/README.md) |
| `Micro-ESPectre` | Frontend README for the maintained MicroPython R&D runtime, project firmware, deployment, and MQTT workflow | [`README.md` (micro_espectre)](../src/python/micro_espectre/README.md) |

## Web Flash (no coding required)

Go to [espectre.dev/flash](https://espectre.dev/flash/) and select:

- the firmware frontend
- the firmware channel
- your target chip

Use `Latest Release` for official firmware, `Release Preview` for the latest build from `main`, or `Development` for the latest build from `develop`. Published ESPHome firmware starts with Lightweight Detection and supports persisted runtime switching to High Accuracy. Published Matter firmware starts with Lightweight; High Accuracy is available in local Matter builds and is selected at build time.

To flash:

1. Connect the board over USB
2. Click **Connect**
3. Select the serial port
4. Confirm the browser prompt

If your browser does not support Web Serial, the same page exposes direct download links for manual flashing.

Website maintainers can find local preview and artifact-staging instructions in [`docs/web/README.md`](web/README.md).

## Shared Prerequisites

### Hardware

- ESP32 board with CSI support
- USB cable for flashing
- Wi-Fi network on a band the board supports: 2.4 GHz on every supported chip, or 5 GHz on the dual-band ESP32-C5. Firmware defaults to 2.4 GHz; an ESP32-C5 integrator can explicitly select 5 GHz or automatic band selection. The runtime pins the selected band or bands to HT20. Detection quality on 5 GHz is not characterized yet

Current chip support by frontend:

| Frontend | Supported chips | Delivery |
|----------|-----------------|----------|
| `ESPHome` | `ESP32-S3`, `ESP32-S2`, `ESP32-C6`, `ESP32-C5`, `ESP32-C3`, `ESP32` | Published web-flash images; ESP32-S2 uses serial or fallback-AP provisioning because it has no Bluetooth radio |
| `Native` | `ESP32`, `ESP32-S3`, `ESP32-S2`, `ESP32-C3`, `ESP32-C5`, `ESP32-C6` | Published web-flash images |
| `Matter` | `ESP32`, `ESP32-S3`, `ESP32-C3`, `ESP32-C5`, `ESP32-C6` | Published web-flash images |
| `Micro-ESPectre` | `ESP32`, `ESP32-C3`, `ESP32-C5`, `ESP32-C6`, `ESP32-S3` | Local project-firmware build and filesystem deployment workflow |

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

Native and Matter builds prefer an active `IDF_PATH` environment, a standard local ESP-IDF installation, or the pinned ESP-IDF toolchain already managed by ESPHome, and automatically fall back to the pinned ESP-IDF Docker image when none is available. Repository ESPHome commands explicitly select its native `esp-idf` toolchain and never use PlatformIO.

```bash
./espectre native build --chip c3
```

On Windows, use `.\espectre.cmd native build --chip c3`. The same pattern applies to Matter.

When the local environment is absent and Docker is running, a cached image is used without prompting. If the image is missing, an interactive build asks before downloading it; non-interactive builds must opt in with `--pull missing`. If Docker is installed but stopped, the CLI asks you to start it and retry. Use `--backend local` or `--backend docker` to require one path, and use `./espectre doctor` to inspect only the local ESP-IDF environment.

Docker currently covers builds only. Flashing through the repository CLI still uses local serial tooling and ESP-IDF. If neither build backend is available, build an ESPHome configuration once to provision its native toolchain, install Docker, or install ESP-IDF `5.5.5` with the official [ESP-IDF Get Started](https://docs.espressif.com/projects/esp-idf/en/stable/esp32/get-started/index.html) flow.

#### Optional Compiler Cache

`ccache` is optional. It shortens repeat ESP-IDF builds, especially Matter builds, by reusing unchanged compiler output across build directories. Repository Docker builds enable a persistent cache automatically, so the Docker backend needs no host installation.

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

Browser tools such as Flash, Configure, Monitor, and Theremin live on [espectre.dev](https://espectre.dev). Configure offers starting device-to-broker presets for Home Assistant with the Mosquitto add-on, a broker on the LAN, EMQX Cloud, HiveMQ Cloud, Flespi, and a custom broker; credentials are never prefilled. Provider presets fill stable MQTT TLS ports and prefill editable `.emqxsl.com` and `.hivemq.cloud` endpoint templates. Provider-defined ports and the fixed Flespi hostname are read-only while their preset is selected; account-specific endpoints, credentials, and topic prefixes remain editable. Configure adds the `mqtts://` scheme automatically when saving a secure preset. Monitor uses Direct HTTP rather than MQTT over WebSockets. To preview the same site from this repository, serve `docs/web` as described in [docs/web/README.md](web/README.md).

Configure and Monitor accept a private device IP, device name, full 16-character device ID, or its last 6 characters. A full ID is translated to the device's unique local address internally; a name or short ID runs the same bounded discovery as the **Auto-discovery** button. One match connects directly, while multiple matches are all displayed for an explicit selection. Native uses an internal nonce-scoped IPv4 bootstrap hostname, performs one fresh browse, and displays validated devices without exposing HTTP or mDNS endpoint syntax in the form. Selecting a result never stores the shared bootstrap hostname or a peer inventory. When automatic discovery is unavailable, enter the device IP or full ID, reuse a remembered device, run `./espectre devices`, or consult the router lease table.

On Configure, click the device ID in the connected-device banner to set the first user-facing name, or click the current name to edit it. The browser saves the value when the field loses focus; Enter saves immediately, and Escape cancels the edit.

See the repository [CLI.md](CLI.md) for:

- launcher syntax on each host
- namespace and command coverage
- shared host-tool behavior, including the interactive MQTT shell
- common wrapper patterns such as `doctor`, serial monitoring, and CLI examples

Use the frontend READMEs for frontend-specific prerequisites, examples, and chip-specific notes:

- [`README.md` (esphome)](../src/cpp/frontend/esphome/README.md)
- [`README.md` (native)](../src/cpp/frontend/native/README.md)
- [`README.md` (matter)](../src/cpp/frontend/matter/README.md)
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
- a component-shaped `src/cpp/` root with `CMakeLists.txt`, `espectre_git_version.cmake`, `idf_component.yml`, and `Kconfig.projbuild`, where the optional MQTT, provisioning, OTA, and stream-runtime groups are selected under the "ESPectre SDK" menuconfig menu

Use [EMBEDDING.md](EMBEDDING.md) for the actual integration model and runtime contracts.

## After Installation

The next step depends on the frontend you chose:

| Frontend | Continue here | What that README owns |
|----------|---------------|-----------------------|
| `ESPHome` | [`README.md`](../src/cpp/frontend/esphome/README.md) | Wi-Fi provisioning, YAML parameters, Home Assistant entities, dashboards, ESPHome-specific troubleshooting |
| `Native` | [`README.md`](../src/cpp/frontend/native/README.md) | Build/flash workflow, Wi-Fi and MQTT setup, Home Assistant MQTT Discovery, native control surface, and HTTPS OTA flow |
| `Matter` | [`README.md`](../src/cpp/frontend/matter/README.md) | Commissioning flow, Matter occupancy surface, and local ESP-IDF workflow |
| `Micro-ESPectre` | [`README.md`](../src/python/micro_espectre/README.md) | Project firmware, filesystem deployment, local configuration, and MQTT operation |

## Reference: Shared Runtime Concepts

These concepts are shared across the C++ platform, even though each frontend exposes them differently.

### Shared Sensing Options

These options belong to the shared sensing runtime and apply to all sensing frontends. This table is the canonical reference for names, defaults, and ranges; the exact user-facing syntax differs by frontend:

- `ESPHome`: YAML under `espectre:`, except the ESP32-C5 band policy, which uses ESPHome's native `wifi.band_mode`
- `Native`: shared ESP-IDF sensing `sdkconfig` menu, with frontend-local overrides in `app/sdkconfig.defaults`
- `Matter`: shared ESP-IDF sensing `sdkconfig` menu, with frontend-local overrides in `app/sdkconfig.defaults`
- `Micro-ESPectre`: constants in `src/python/micro_espectre/config.py`, overridden locally through `config_local.py`; supported MQTT writes are session-only

Frontend coverage:

| Frontend | Shared sensing options available |
|----------|----------------------------------|
| `ESPHome` | yes |
| `Native` | yes |
| `Matter` | yes |
| `Micro-ESPectre` | yes, through its MicroPython configuration surface; runtime MQTT writes are session-only |

| Option | Type / values | Default | Range / notes |
|--------|---------------|---------|---------------|
| `wifi.band_mode` (ESPHome) / `RuntimeConfig::wifi_band_policy` | `2.4GHz`, `5GHz`, or `AUTO` in ESPHome; `BAND_2G`, `BAND_5G`, or `AUTO` in the SDK | ESPHome C5: `AUTO` when omitted; other frontends, including Native C5: `2.4GHz` | `5GHz` and `AUTO` require the dual-band ESP32-C5. Select the policy in ESPHome YAML or an SDK build; Direct HTTP does not expose band selection. ESPHome examples select `2.4GHz`, and the production PHY remains HT20 |
| `detection_algorithm` | `lightweight` or `high_accuracy` | `lightweight`, including Matter | Lightweight uses less detector CPU and working memory; High Accuracy improves detection quality and skips quiet-room threshold calibration |
| Runtime threshold | probability | detector-specific | Selected automatically at startup; session-adjustable where the frontend exposes a writable control. Matter currently exposes no writable sensing controls |
| `segmentation_window_size_ms` | int | `1000` | `1000-2000` milliseconds; combined with `csi_target_pps` to define a fixed temporal slot window |
| `csi_target_pps` | int | `100` | `1-500`; defines detector slot cadence and the managed-traffic target, but never enables or disables traffic |
| `csi_traffic_mode` | `internal` or `external` | `internal` | Selects the configured traffic source independently from `csi_target_pps`; persisted legacy `pacing` or `disabled` values migrate once to `internal` |
| `csi_traffic_multicast_group` | IPv4 multicast address, or empty | `239.255.0.1` | Joined by the UDP listener in `external`. Empty disables the join. Unicast to the device IP still works |
| `traffic_generator_mode` | `ping` or `dns` | `ping` | Shared internal traffic generator mode |
| `publish_interval_ms` | int | `1000` | `100-60000` milliseconds between periodic status-log and diagnostics samples. Canonical MQTT telemetry and Home Assistant Movement Score follow `evaluation_interval_ms` |
| `evaluation_interval_ms` | int | `250` | `10-10000` milliseconds between detector evaluations |
| `motion_on_hits` | int | `4` | `1-20` consecutive evaluation hits for `IDLE -> MOTION` (about `1.0 s` at the default `250 ms` interval) |
| `motion_off_hits` | int | `3` | `1-20` consecutive evaluation hits for `MOTION -> IDLE` (about `0.75 s` at the same defaults) |
| `lowpass_enabled` | bool | `false` | Enables low-pass filtering |
| `lowpass_cutoff` | float | `11.0` | `5.0-20.0` Hz |
| `hampel_enabled` | bool | `true` in the C++ sensing frontends; `false` in Micro-ESPectre | Enables Hampel outlier filtering; Micro keeps it off by default to preserve CPU and heap headroom |
| `hampel_window` | int | `7` | `3-11` samples |
| `hampel_threshold` | float | `5.0` | `1.0-10.0` MAD units |

Migration from earlier v3 snapshots: replace `traffic_generator_rate: N` with `csi_target_pps: N` plus `csi_traffic_mode: internal`. Persisted `pacing` and `disabled` values are migrated once to `internal`; runtime requests using those removed values fail with `invalid_params`.

See [TUNING.md](TUNING.md) for how evaluation cadence and hit filtering set the expected publish delay (about `1 s` for `IDLE -> MOTION` with the defaults).

Use the frontend README for the exact syntax and local workflow:

- [`README.md` (esphome)](../src/cpp/frontend/esphome/README.md)
- [`README.md` (native)](../src/cpp/frontend/native/README.md)
- [`README.md` (matter)](../src/cpp/frontend/matter/README.md)
- [`README.md` (micro_espectre)](../src/python/micro_espectre/README.md)

### Detection Profiles And Startup

ESPectre keeps two production detection profiles because no single choice optimizes both accuracy and resource use. Lightweight runs fewer feature trackers and is the leaner choice when the chip or surrounding firmware needs more CPU time and working memory for other work. High Accuracy uses a larger feature state and neural inference to provide higher accuracy and stronger generalization on the maintained corpus.

At boot, Lightweight adapts its threshold to the room from about 10 seconds of clean, ready CSI coverage after temporal warmup. Missing or burst-concentrated slots extend wall-clock calibration instead of counting as evidence. After that, a long quiet stretch can still lower the live threshold if the opening was noisier than the rest of the session; Home Assistant, ESPHome, and the website Monitor follow that value. High Accuracy uses its trained threshold and skips threshold calibration; it becomes active after CSI capture is ready and the feature window has filled.

ESPHome, Native, Matter, and Micro-ESPectre support both `lightweight` and `high_accuracy`. ESPHome and Native can switch profiles at runtime and persist the selection; the switch resets the threshold to the selected profile's default, and `high_accuracy -> lightweight` starts calibration automatically. Matter selects the profile at build time, exposes no runtime detector control, and uses `lightweight` in published firmware while the frontend remains preview. Micro-ESPectre selects the profile in its deployment configuration and does not expose runtime detector switching.

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
| Native / Matter | `CONFIG_ESPECTRE_CSI_TARGET_PPS` | `csi_traffic_mode`; internal by default | yes | phase-preserving cadence without catch-up bursts; local socket backoff only |
| ESPHome | `csi_target_pps` | `csi_traffic_mode`; internal by default | yes | phase-preserving cadence without catch-up bursts; local socket backoff only |
| Micro-ESPectre | `CSI_TARGET_PPS` | factory default from `TRAFFIC_GENERATOR_ENABLED`, with session-only MQTT overrides for `csi_traffic_mode` and `traffic_generator_mode` | yes | phase-preserving cadence without catch-up bursts; local socket backoff only |
| Collector detector, replay, training, and validation | recorded `csi_target_pps`, collector `--pps`, or a documented legacy fallback | recorded raw HTTP stream | yes, through the production Micro-ESPectre sampler | external generator owns rate; HTTP does not pace |

Raw HTTP collection is available on Native, ESPHome, and Matter. It preserves every classified CSI frame except explicitly counted fixed-ring drops; only the collector's derived live detector view applies temporal admission.

External UDP traffic can be unicast to each device IP, or sent to multicast group `239.255.0.1`. ESP-IDF frontends join that group automatically in `external`. Empty `csi_traffic_multicast_group` disables the join. Subnet and limited broadcast (`x.x.x.255`, `255.255.255.255`) do not produce reliable HT20 CSI. ESPHome, Native, and Matter listen on port `5555` and accept only the exact four-byte UTF-8 marker `"👻".encode("utf-8")` (`F0 9F 91 BB`); use [`espectre_traffic_generator.py`](../tools/espectre_traffic_generator.py) standalone or through `./espectre collect`.

Micro-ESPectre keeps its persisted factory default as `TRAFFIC_GENERATOR_ENABLED` plus `TRAFFIC_GENERATOR_MODE`, then exposes session-only MQTT and Home Assistant runtime control over `csi_traffic_mode` and `traffic_generator_mode`. `internal` starts the local generator, and `external` stops it. Micro does not open a UDP listener, so it does not join the multicast group.

Across Native, Matter, ESPHome, and Micro-ESPectre, internal `ping` mode sends ICMP echo requests, while internal `dns` mode sends DNS root queries through a persistent, non-blocking TCP connection to gateway port `53`. DNS mode requires the gateway resolver to accept TCP queries.

If you are tuning `csi_target_pps`, thresholds, or filters, use [TUNING.md](TUNING.md) for the rationale and the frontend README for the configuration syntax.

## Where to Go Next

- To configure or troubleshoot an installed device, use its frontend README and [TUNING.md](TUNING.md).
- To study detector behavior and formulas, use [ALGORITHMS.md](ALGORITHMS.md).
- To change the shared code, use [ARCHITECTURE.md](ARCHITECTURE.md).
- To integrate the SDK into another firmware product, use [EMBEDDING.md](EMBEDDING.md).
