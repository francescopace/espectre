# Micro-ESPectre

Micro-ESPectre is the small, research-oriented MicroPython sensing frontend. It keeps the device path easy to modify while using native ESP-IDF components only where timing or transport work benefits from fixed memory and predictable scheduling.

Changes to shared detector behavior must remain aligned with the C++ implementation. See the main [README](../../../README.md), [SETUP.md](../../../docs/SETUP.md), [ARCHITECTURE.md](../../../docs/ARCHITECTURE.md), and [ESPECTRE_PROTOCOL.md](../../../docs/ESPECTRE_PROTOCOL.md) for the project-wide contracts.

## Device profile

The deployed runtime intentionally contains only:

- the Lightweight CSI detector and its startup threshold calibration;
- the ESP-IDF ICMP ping traffic generator;
- a bounded, read-only Direct HTTP endpoint for monitoring;
- one SSE telemetry client;
- mDNS/DNS-SD advertisement and a unique `.local` hostname; and
- serial logging and the MicroPython REPL.

The device does not deploy the High Accuracy ML detector, ML weights, MQTT, Home Assistant discovery, the shared C++ DNS-over-TCP generator, runtime detector switching, raw CSI streaming, OTA, or configuration mutations. The High Accuracy Python sources remain in the repository for host-side research and C++/Python validation, but `micro deploy` does not copy them to the device.

ESPectre contributed direct ESP32 Wi-Fi CSI access to mainline MicroPython through [micropython/micropython#18460](https://github.com/micropython/micropython/pull/18460). Micro-ESPectre builds a pinned mainline revision with a lean ESPectre board profile rather than using the earlier CSI fork.

## Build and deploy

Complete the shared prerequisites in [SETUP.md](../../../docs/SETUP.md#local-build-prerequisites), then run the Micro-ESPectre workflow from the repository root:

```bash
cp src/python/micro_espectre/config_local.py.example src/python/micro_espectre/config_local.py
./espectre micro flash --chip c3 --erase
./espectre micro deploy
./espectre micro run
```

Set the Wi-Fi credentials in `config_local.py`; do not commit that file.

```python
WIFI_SSID = "YourWiFiSSID"
WIFI_PASSWORD = "YourWiFiPassword"
# WIFI_BSSID = "AA:BB:CC:DD:EE:FF"  # Optional AP lock
# WIFI_CHANNEL = 6  # Optional known channel used with WIFI_BSSID
```

The firmware image freezes only MicroPython's upstream boot and filesystem helpers. The complete ESPectre application is compiled to optimized `.mpy -O3` bytecode and stored on the filesystem, so research changes require only `micro deploy`, not a firmware rebuild and flash. Deployment uploads the complete manifest to a staging directory and atomically activates it, restoring the previous directory after an interrupted swap. The device and `mpy-cross` use MPY ABI 6.3.

The firmware links the core-only ESPectre SDK as an ESP-IDF component. Its MicroPython binding exposes finalizable `Detector` and `TemporalCsiSampler` objects through the public `espectre_core_sdk.h` facade. The production Lightweight detector and temporal admission hot paths therefore run in C++, while MicroPython owns orchestration, calibration policy, diagnostics, and delivery. The application fails at startup if the core module is absent or incompatible; it does not silently fall back to the Python detector on the device. The same Python implementation remains available under CPython for replay, training, and host-side experimentation. The other native components are the ICMP traffic generator and the Direct HTTP/mDNS service. Bluetooth, ESP-NOW, asyncio, Ethernet, unused peripheral bindings, and unused generic Python modules remain disabled.

## Runtime behavior

The runtime uses this fixed sensing path:

```text
Wi-Fi -> native managed traffic -> CSI temporal sampler -> Lightweight calibration -> detection -> Direct SSE and serial
```

Key settings live in `config.py`:

```python
DEVICE_LABEL = ""
CSI_TARGET_PPS = 100
TRAFFIC_GENERATOR_ENABLED = True
CSI_LINK_RECOVERY_TIMEOUT_MS = 5000
CSI_CAPTURE_MAX_DATA_LEN = 256
SEGMENTATION_WINDOW_SIZE_MS = 1000
EVALUATION_INTERVAL_MS = 250
MOTION_ON_HITS = 4
MOTION_OFF_HITS = 3
```

These `config.py` values are deployment settings rather than runtime mutations. An empty `DEVICE_LABEL` keeps the shared generated name. `CSI_TARGET_PPS` defines the detector grid and ICMP target rate, while setting `TRAFFIC_GENERATOR_ENABLED = False` requires an external CSI traffic source. `CSI_CAPTURE_MAX_DATA_LEN` selects the fixed native ring-record stride: 256 supports the doubled HT20 layout, while 128 is suitable only when every captured frame uses the canonical payload because larger frames are truncated.

In `config_local.py`, `WIFI_CHANNEL` can accompany `WIFI_BSSID` to avoid a scan during association. If no CSI frame arrives for `CSI_LINK_RECOVERY_TIMEOUT_MS`, the runtime first rearms CSI. If the stall persists, it reconnects Wi-Fi, recalibrates, and republishes Direct discovery.

The production `TemporalCsiSampler` retains the packet nearest each slot center, preserves missing slots, and keeps the live detector geometry independent from observed network jitter. See [SETUP.md](../../../docs/SETUP.md#traffic-generation) for shared traffic behavior, [TUNING.md](../../../docs/TUNING.md) for startup and detector operation, and [ALGORITHMS.md](../../../docs/ALGORITHMS.md) for the implementation rationale.

## Direct HTTP surface

Micro-ESPectre listens on port `62587`, advertises `_espectre._tcp.local.`, and exposes:

- `POST /espectre/v1/request` with `capabilities`, `info`, `status`, `config`, and `diagnostics`;
- `GET /espectre/v1/events` with canonical `telemetry` SSE events; and
- CORS and Private Network Access preflight support for the ESPectre website.

The exact capability response is authoritative: the Micro frontend advertises only read-only queries. The Monitor site therefore displays sensing without sending unsupported control requests. Enter the device IP or `espectre-micro-<suffix>.local` in the site, or discover it with:

```bash
./espectre devices --frontend micro
```

Monitor Auto-discovery can also list this device when a Native, ESPHome, or Matter responder is already on the LAN. Micro-ESPectre does not answer the one-shot bootstrap hostname or `discover_peers`; without another eligible responder, use the private IP or unique `.local` hostname. Shared discovery is documented in [Peer-assisted browser discovery](../../../docs/ESPECTRE_PROTOCOL.md#peer-assisted-browser-discovery).

Only one SSE client is retained to bound sockets and heap. Query snapshots are generated by the MicroPython runtime, while HTTP framing, request parsing, CORS, and mDNS run in the native firmware component. Telemetry follows `EVALUATION_INTERVAL_MS` while that SSE client is connected, matching the consumer-aware C++ frontends. A fixed one-second heartbeat logs the current sensing status and refreshes the `status` and `diagnostics` snapshots; the diagnostics payload uses only canonical protocol fields and caches CSI rates, occupancy, heap, runtime-loop timing, and detector timing.

## Commands

| Command | Purpose |
| --- | --- |
| `./espectre micro build --chip <esp32|c3|s2|s3|c5|c6>` | Build the lean project firmware |
| `./espectre micro flash --chip <chip> --erase` | Build and flash the project image |
| `./espectre micro deploy` | Compile and upload the complete `.mpy -O3` manifest |
| `./espectre micro run` | Start the device application |
| `./espectre micro verify` | Check firmware, native modules, and deployed bytecode |
| `./espectre monitor --reset` | Follow serial output with auto-reconnect |

`micro build` and the implicit build in `micro flash` use the shared ESP-IDF backend policy documented in [SETUP.md](../../../docs/SETUP.md#local-build-prerequisites). See [CLI.md](../../../docs/CLI.md) for `--backend` and `--pull` controls.

## Troubleshooting

Use [SETUP.md](../../../docs/SETUP.md#direct-http-connectivity) for Direct HTTP, browser permission, address, and discovery failures. Use [TUNING.md](../../../docs/TUNING.md#troubleshooting) for missing CSI, calibration, placement, false positives, or unstable detection.

### Deployed changes do not appear

Run `./espectre micro deploy` again, then restart the application with `./espectre micro run`. If the application still does not start, use `./espectre micro verify` to check CSI firmware support, the MicroPython version, required bytecode, and `config_local.mpy`.

### Monitor cannot open the event stream

Micro-ESPectre retains one SSE client. Close any previous Monitor tab or client before reconnecting with the private IP or unique `.local` hostname.

### Wi-Fi never becomes ready

Confirm that `config_local.py` exists, contains the intended SSID and password, and does not retain a stale optional `WIFI_BSSID`. Deploy the updated configuration and run the application again.

## Validation

Run the focused host tests from the repository environment:

```bash
.venv/bin/pytest test/python/test_traffic_generator.py test/python/test_espectre_cli_micro.py test/python/test_micro_protocol.py -v
```

Firmware builds use the project wrapper:

```bash
./espectre micro build --chip c3
```
