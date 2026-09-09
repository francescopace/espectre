# Micro-ESPectre

Micro-ESPectre is the MicroPython frontend for sensing research. MicroPython handles device orchestration, while native components run Lightweight detection, temporal admission, traffic generation, and HTTP delivery. This guide covers firmware builds, application deployment, configuration, and device-specific limits.

## Device profile

The deployed runtime provides Lightweight detection and calibration, internal ping or DNS traffic, Direct HTTP monitoring and manual recalibration, mDNS discovery, serial logs, and the MicroPython REPL.

The device does not deploy the High Accuracy ML detector, ML weights, MQTT, Home Assistant discovery, runtime detector switching, raw CSI streaming, OTA, or configuration mutations. The High Accuracy and pure-Python Lightweight implementations live under `tools/lib/` for host-side research and C++/Python validation; `micro deploy` does not copy them to the device.

The project builds a pinned mainline MicroPython revision for ESP32, ESP32-S2, ESP32-S3, ESP32-C3, ESP32-C5, and ESP32-C6. It senses through an associated Wi-Fi station without promiscuous mode. Start with 2.4 GHz; see [CSI.md](../../../docs/CSI.md#micro-espectre-acquisition) for band and capture differences, and [SETUP.md](../../../docs/SETUP.md#sensor-placement) for placement and the first sensing check. Recording requirements are in [ML_DATA_COLLECTION.md](../../../docs/ML_DATA_COLLECTION.md#data-privacy).

## Build and deploy

Complete the prerequisites in [CLI.md](../../../docs/CLI.md#local-build-prerequisites). From the repository root, create your local configuration:

```bash
cp src/python/micro_espectre/config_local.py.example src/python/micro_espectre/config_local.py
```

Set the Wi-Fi credentials in `config_local.py`; do not commit that file.

```python
WIFI_SSID = "YourWiFiSSID"
WIFI_PASSWORD = "YourWiFiPassword"
# WIFI_BSSID = "AA:BB:CC:DD:EE:FF"  # Optional AP lock
# WIFI_CHANNEL = 6  # Optional known channel used with WIFI_BSSID
```

Build and flash the firmware, then deploy and start the application:

```bash
./espectre micro flash --chip c3 --erase
./espectre micro deploy
./espectre micro run
```

`--erase` clears flash for the initial installation. On Windows, use `.\espectre.cmd` in place of `./espectre`. Builds require ESP-IDF 5.5 and use the backend and compiler-cache policy in [CLI.md](../../../docs/CLI.md#local-build-prerequisites). Cached sources, build trees, and images live under `.firmware/` in this directory.

Application and configuration changes need only `micro deploy`, followed by `micro run`. Deployment compiles the complete application to `.mpy -O3`, stages it on the device filesystem, and activates it atomically, with recovery after an interrupted swap. The device and `mpy-cross` use MPY ABI 6.3. Rebuild and flash when changing native components or the MicroPython board profile.

The embedded `micro-espectre` application descriptor reports the ESPectre firmware build version. A later filesystem deployment does not change that version. Older images with empty descriptors need a firmware rebuild and flash to populate them.

## Runtime behavior

Defaults live in [config.py](config.py). Put deployment-specific overrides in `config_local.py`, then deploy and restart. These settings cannot be changed through Direct HTTP.

| Settings | Use |
| --- | --- |
| `DEVICE_LABEL` | Device name; empty keeps the generated name |
| `WIFI_BSSID`, `WIFI_CHANNEL` | Optional AP pin and known-channel hint |
| `CSI_TARGET_PPS` | Managed traffic rate and detector slot cadence |
| `TRAFFIC_GENERATOR_ENABLED`, `TRAFFIC_GENERATOR_MODE`, `TRAFFIC_GENERATOR_TARGET_IP` | Traffic source and destination |
| `CSI_BUFFER_SIZE`, `CSI_CAPTURE_MAX_DATA_LEN`, `CSI_LINK_RECOVERY_TIMEOUT_MS` | Capture buffering and stalled-link recovery |
| `SEGMENTATION_WINDOW_SIZE_MS`, `EVALUATION_INTERVAL_MS` | Detector window and evaluation cadence |
| `MOTION_ON_HITS`, `MOTION_OFF_HITS` | Consecutive evaluated hits required to change motion state |
| `ENABLE_LOWPASS_FILTER`, `LOWPASS_CUTOFF`, `ENABLE_HAMPEL_FILTER`, `HAMPEL_WINDOW`, `HAMPEL_THRESHOLD` | Optional preprocessing; both filters are disabled by default |

`TRAFFIC_GENERATOR_MODE` accepts `ping`, `dns`, or `dns_tcp`, with `ping` as the default. An empty `TRAFFIC_GENERATOR_TARGET_IP` uses the Wi-Fi gateway; a unicast IPv4 address overrides it. Disabling the generator requires external traffic, but Micro has no UDP marker listener or multicast join. See [CSI.md](../../../docs/CSI.md#micro-espectre-acquisition) for traffic, buffer, and recovery details.

Use [ALGORITHMS.md](../../../docs/ALGORITHMS.md) for temporal admission, filters, and calibration, and [TROUBLESHOOTING.md](../../../docs/TROUBLESHOOTING.md#tuning-essentials) for tuning guidance. Apply supported configuration changes through deployment on Micro-ESPectre.

## Direct HTTP surface

Micro-ESPectre provides read-only status and diagnostics, motion events, and manual recalibration. Open Monitor using the device IP or `espectre-<device-id>.local`, or discover the endpoint with:

```bash
./espectre devices --frontend micro
```

Only one Monitor or SSE client can connect at a time. Recalibration affects the current session. [API.md](../../../docs/API.md) defines the supported resources, request limits, events, and Origin policy.

Browser Auto-discovery can list Micro when a Native, ESPHome, or Matter responder is on the LAN. Micro does not provide peer discovery itself; otherwise, enter its IP or unique hostname. See [DISCOVERY.md](../../../docs/DISCOVERY.md#browser-bootstrap).

For local browser development, build with `CONFIG_ESPECTRE_DIRECT_DEV_ORIGINS_ENABLED=y` and follow [README.md](../../../docs/web/README.md#local-preview). Published board profiles keep loopback origins disabled.

## Commands

| Command | Purpose |
| --- | --- |
| `./espectre micro build [--chip <esp32\|c3\|s2\|s3\|c5\|c6>]` | Build the lean project firmware; the default chip is `esp32` |
| `./espectre micro flash --chip <chip> --erase` | Build and flash the canonical project image |
| `./espectre micro deploy` | Compile and upload the complete `.mpy -O3` manifest |
| `./espectre micro run` | Start the device application |
| `./espectre micro verify` | Check firmware, native modules, and deployed bytecode |
| `./espectre monitor --chip <chip> --reset` | Start the application through esptool, then follow serial output |

`--port` is optional for `flash`, `deploy`, `run`, and `verify`. `micro flash` requires `--chip`, builds the canonical project image, and flashes the files listed in its generated `flasher_args.json` metadata in one esptool session. Other device-facing commands accept an optional chip to resolve ambiguous candidates, while `build` defaults to `esp32`. `micro deploy --config <path>` compiles an alternate local override as device `config_local.mpy`, which is useful for isolated laboratory settings.

`micro build --json` emits final artifact metadata, `micro flash --json` adds the selected port after a successful flash, and `micro run --json` streams logs and emits a `direct_ready` event when the application reports its endpoint. Run `./espectre micro <command> --help` for the current flags.

## Troubleshooting

Use [TROUBLESHOOTING.md](../../../docs/TROUBLESHOOTING.md) for browser connectivity and sensing problems.

### Deployed changes do not appear

Run `./espectre micro deploy` again, then restart the application with `./espectre micro run`. If the application still does not start, use `./espectre micro verify` to check CSI firmware support, the MicroPython version, required bytecode, and `config_local.mpy`.

### Monitor cannot open the event stream

Micro-ESPectre retains one SSE client. Close any previous Monitor tab or client before reconnecting with the private IP or unique `.local` hostname.

### Wi-Fi never becomes ready

Confirm that `config_local.py` exists, contains the intended SSID and password, and does not retain a stale optional `WIFI_BSSID`. Deploy the updated configuration and run the application again.

## Implementation

[runtime_main.py](runtime_main.py) owns calibration, diagnostics, and the device loop. Native bindings expose the SDK's Lightweight detector and temporal sampler, plus the shared traffic generator. Startup fails if the core module is absent or incompatible; there is no on-device Python detector fallback. See [SDK.md](../../../docs/SDK.md#core-only) for the core interface and logging contract.

[native_direct.c](firmware/native_components/native_direct.c) owns HTTP framing, CORS, mDNS, and transport counters; MicroPython supplies resource snapshots. Telemetry runs at `EVALUATION_INTERVAL_MS` only while an SSE client is connected. Health refreshes once per second. Full diagnostics and garbage collection prefer an empty CSI ring, with at most 500 ms of additional deferral under backlog.

Board profiles under [boards/](firmware/boards/) contain the memory, Wi-Fi, and peripheral build choices. Classic ESP32 uses smaller Wi-Fi queues and omits lwIP IRAM placement to preserve heap. Bluetooth, ESP-NOW, asyncio, and Ethernet are disabled in the maintained profile.

The `Micro-ESPectre > Advanced task scheduling` Kconfig menu controls native HTTPD and traffic priorities through `CONFIG_ESPECTRE_DIRECT_HTTPD_TASK_PRIORITY` and `CONFIG_ESPECTRE_TRAFFIC_TASK_PRIORITY`. Both accept `1..10` and default to `1`; board overrides require validation. MicroPython owns the VM task priority.

## Validation

Run the focused host tests from the repository environment:

```bash
.venv/bin/pytest test/python/host/cli/test_traffic_generator.py test/python/micro/test_espectre_cli_micro.py test/python/micro/test_micro_protocol.py -q --tb=short
```

Firmware builds use the project wrapper:

```bash
./espectre micro build --chip c3
```

## Contributing

Keep device code MicroPython-compatible and allocations bounded. Host research belongs under `tools/`; shared detector behavior and serialized messages must remain aligned with C++. Follow [CONTRIBUTING.md](../../../CONTRIBUTING.md) for review and full coverage checks, and preserve the licensing notices described in [LICENSING.md](../../../LICENSING.md).
