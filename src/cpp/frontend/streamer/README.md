# ESPectre Streamer Frontend

This directory contains the standalone CSI streamer frontend.

Unlike `esphome`, `ble`, and `matter`, the streamer is not an ecosystem-facing
adapter over `IEspectreRuntime`. It is a dedicated firmware path for raw CSI
collection and transport to host-side tooling.

This file is the source of truth for the streamer frontend surface and UDP
protocol.

## Scope

The streamer frontend is responsible for:

- capturing CSI on-device
- receiving external UDP stimulus
- gain lock before streaming
- packaging CSI into the UDP stream format
- sending packets to the most recent stimulus sender host

Use [`ML_DATA_COLLECTION.md`](../../../../docs/ML_DATA_COLLECTION.md) for ML data collection workflow.

## Important Architectural Note

The streamer frontend uses the lower-level `runtime/esp_idf` modules directly.
It does not go through the `IEspectreRuntime` facade.

That is intentional:

- the goal is raw CSI transport, not motion-detection entity exposure
- the firmware needs a compact packet-oriented streaming path
- the state machine is streamer-specific (`WAIT_WIFI` -> `STREAMING`)

The standalone Wi-Fi setup path is shared with the other ESP-IDF standalone
firmware targets through `StandaloneWifiManager`; only the CSI capture and UDP
streaming workflow is streamer-specific.

The streamer also exposes the shared ESPectre BLE provisioning surface for
Wi-Fi setup only. This lets a browser client save Wi-Fi credentials over Web
Bluetooth without turning the streamer into a full runtime frontend or adding
motion telemetry over BLE.

## Directory Layout

- [`espectre/stream_frontend.cpp`](espectre/stream_frontend.cpp),
  [`espectre/stream_frontend.h`](espectre/stream_frontend.h):
  frontend state machine and orchestration
- [`espectre/csi_stream_protocol.h`](espectre/csi_stream_protocol.h):
  UDP stream header and flags
- [`espectre/csi_udp_sender.cpp`](espectre/csi_udp_sender.cpp):
  queued UDP sender
- [`espectre/Kconfig.projbuild`](espectre/Kconfig.projbuild):
  frontend-specific configuration surface
- [`app/`](app/):
  standalone ESP-IDF firmware app

## Workflow States

The streamer frontend uses these states:

- `WAIT_WIFI`
- `WIFI_READY`
- `CSI_READY`
- `GAIN_LOCK`
- `STREAMING`

This state machine is defined in [`espectre/stream_frontend.h`](espectre/stream_frontend.h).

## UDP Stream Protocol

Protocol constants live in [`espectre/csi_stream_protocol.h`](espectre/csi_stream_protocol.h).

Current version:

- magic: `0x4353`
- version: `2`
- header size: `52` bytes

Header layout:

| Field | Type | Description |
|-------|------|-------------|
| `magic` | `uint16` | Stream magic |
| `version` | `uint8` | Protocol version |
| `header_len` | `uint8` | Header size |
| `chip` | `uint8` | Chip enum |
| `flags` | `uint8` | Packet flags |
| `seq_num` | `uint32` | Wrapping packet sequence |
| `num_subcarriers` | `uint16` | Logical subcarrier count |
| `csi_len_bytes` | `uint16` | CSI payload length |
| `device_id` | `uint64` | Stable device identifier |
| `device_ticks_us` | `uint64` | Device-side timestamp |
| `wifi_rx_ts_us` | `uint32` | Wi-Fi RX timestamp |
| `wifi_rx_start_ts_ns` | `uint64` | Estimated Wi-Fi RX start |
| `stimulus_id` | `uint32` | Optional stimulus identifier |
| `channel` | `uint8` | Wi-Fi channel |
| `rssi_dbm` | `int8` | RSSI |
| `noise_floor_dbm` | `int8` | Noise floor |
| `agc_gain` | `uint8` | AGC gain |
| `fft_gain` | `int8` | FFT gain |

Flags:

| Bit | Constant | Meaning |
|-----|----------|---------|
| 0 | `STREAM_FLAG_GAIN_LOCKED` | Gain lock active |
| 1 | `STREAM_FLAG_FIRST_WORD_INVALID` | Espressif CSI flag |
| 2 | `STREAM_FLAG_WIFI_RX_TS_VALID` | `wifi_rx_ts_us` valid |
| 3 | `STREAM_FLAG_WIFI_RX_START_TS_NS_VALID` | `wifi_rx_start_ts_ns` valid |
| 4 | `STREAM_FLAG_GAIN_INFO_VALID` | Gain metadata valid |
| 5 | `STREAM_FLAG_STIMULUS_ID_VALID` | `stimulus_id` valid |
| 6 | `STREAM_FLAG_REFERENCE_FRAME` | Packet marked as reference frame |

Payload:

- raw I/Q values in Espressif ordering
- typical HT20 packet: `52 + 128 = 180 bytes`
- the sender may concatenate multiple complete stream records into one UDP
  datagram; the host collector parses them sequentially from the datagram body

## Frontend Configuration

Frontend-specific options are declared in [`espectre/Kconfig.projbuild`](espectre/Kconfig.projbuild).

Versioned defaults live in [`app/sdkconfig.defaults`](app/sdkconfig.defaults).
Local Wi-Fi credentials should live in `app/sdkconfig.wifi`, which is gitignored.
Wi-Fi credentials can also be provisioned live over BLE and persisted in NVS;
stored BLE-provisioned values take precedence over build-time defaults.

Typical local override file:

```ini
CONFIG_ESPECTRE_WIFI_SSID="YourSSID"
CONFIG_ESPECTRE_WIFI_PASSWORD="YourPassword"
# CONFIG_ESPECTRE_WIFI_BSSID is not set
# CONFIG_ESPECTRE_WIFI_CHANNEL is not set
```

Recommended workflow for local Wi-Fi configuration:

1. create `src/cpp/frontend/streamer/app/sdkconfig.wifi`
2. set `CONFIG_ESPECTRE_WIFI_SSID` and `CONFIG_ESPECTRE_WIFI_PASSWORD`
3. leave `CONFIG_ESPECTRE_WIFI_BSSID` unset unless you intentionally want to
   pin the streamer to a specific AP radio
4. leave `CONFIG_ESPECTRE_WIFI_CHANNEL` unless you intentionally want to
   pin the streamer to a specific AP channel
5. build via `./espectre streamer build --chip <esp32|c3|c5|c6|s3>`, which automatically passes
   `sdkconfig.defaults;sdkconfig.wifi` to `idf.py`

Alternative BLE provisioning workflow:

1. flash the streamer firmware once
2. open [`tools/web/espectre-ble.html`](../../../../tools/web/espectre-ble.html)
   from a secure browser context
3. connect to `ESPectre Streamer`
4. use `Save Wi-Fi` to send `SET_WIFI_SSID`, `SET_WIFI_PASSWORD`,
   `SET_WIFI_BSSID`, `SET_WIFI_CHANNEL`, and `APPLY_WIFI`
5. request sysinfo and verify `wifi_saved=true`

Example local file:

```ini
CONFIG_ESPECTRE_WIFI_SSID="YourSSID"
CONFIG_ESPECTRE_WIFI_PASSWORD="YourPassword"
# CONFIG_ESPECTRE_WIFI_BSSID is not set
CONFIG_ESPECTRE_WIFI_CHANNEL=0
```

Notes:

- `sdkconfig.wifi` is the recommended place for machine-local credentials
  because it is ignored by git
- keep `CONFIG_ESPECTRE_WIFI_BSSID` unset for normal use; the streamer will
  scan all channels and connect to the strongest matching AP
- set `CONFIG_ESPECTRE_WIFI_BSSID="aa:bb:cc:dd:ee:ff"` only when you need to
  force a specific AP radio for repeatable RF tests; this also enables fast
  scan instead of the default full scan
- set `CONFIG_ESPECTRE_WIFI_CHANNEL=<1-14>` together with BSSID when you want
  deterministic BSSID+channel association for CSI captures
- if you change `sdkconfig.defaults`, `sdkconfig.wifi`, or the frontend Kconfig
  surface and the generated `sdkconfig` appears stale, remove
  `src/cpp/frontend/streamer/app/sdkconfig` and `sdkconfig.old` before
  rebuilding so ESP-IDF regenerates the active config from the defaults
- the active generated files `sdkconfig`, `sdkconfig.old`, and
  `dependencies.lock` are build artifacts and should remain untracked

Key knobs in the frontend surface:

- `ESPECTRE_WIFI_SSID`
- `ESPECTRE_WIFI_PASSWORD`
- `ESPECTRE_WIFI_BSSID`
- `ESPECTRE_WIFI_CHANNEL`
- `ESPECTRE_STREAM_OUTPUT_ENABLED`
- `ESPECTRE_COLLECTOR_PORT`
- `ESPECTRE_TRAFFIC_RX_PORT`
- `ESPECTRE_TRAFFIC_RX_MULTICAST_GROUP`
- `ESPECTRE_GAIN_LOCK_ENABLED`
- `ESPECTRE_GAIN_LOCK_MODE_*`
- `ESPECTRE_STREAM_QUEUE_SLOTS`
- `ESPECTRE_STREAM_BATCH_MAX_RECORDS`
- `ESPECTRE_STREAM_BATCH_MAX_BYTES`
- `ESPECTRE_STREAM_LOG_INTERVAL_MS`

Runtime behavior notes:

- the streamer no longer owns an internal traffic generator
- BLE provisioning handles Wi-Fi setup plus device naming/sysinfo; it does not
  expose streamer CSI data or runtime motion telemetry over BLE
- the collector address is learned from the source IP of the latest UDP stimulus packet
- the UDP stimulus payload may carry the `ESTM` metadata header
  (`magic + version + role + stimulus_id`), which is propagated into the CSI
  stream when present
- the UDP sender uses a bounded queue plus datagram batching, so queue depth and
  queue peak are useful indicators when tuning packet rate

## Collector-Driven Stimulus

The streamer expects external UDP stimulus from the host collector.

The collector is responsible for:

- sending UDP packets to the streamer stimulus port
- choosing the stimulus rate (`pps`)
- assigning `stimulus_id`
- optionally marking packets as reference frames

The streamer is responsible for:

- learning the collector IP from the source address of valid incoming stimulus
- extracting `ESTM` metadata from the packet payload seen in CSI
- copying `stimulus_id` / `reference` markers into the UDP CSI stream

`ESTM` carries:

- `magic`
- `version`
- `role`
- `stimulus_id`

Current roles:

- measurement frame: normal sample used for the session stream
- reference frame: sample marked with `STREAM_FLAG_REFERENCE_FRAME`

Reference frames are controlled entirely by the collector. The streamer does
not generate them on its own and does not reinterpret their meaning.

## Build and Tooling

Repository CLI:

```bash
./espectre streamer build --chip s3
./espectre streamer flash --port /dev/cu.usbmodemXXXX
./espectre monitor --port /dev/cu.usbmodemXXXX
```

On Windows, use `.\espectre.cmd streamer ...` for build/flash and
`.\espectre.cmd monitor --port COM5` for serial logs.

When `app/sdkconfig.wifi` exists, the repository CLI automatically passes
`sdkconfig.defaults;sdkconfig.wifi` to `idf.py` for `build`.

Raw ESP-IDF flow:

```bash
cd src/cpp/frontend/streamer/app
idf.py -DSDKCONFIG_DEFAULTS="sdkconfig.defaults;sdkconfig.wifi" set-target esp32c3
idf.py -DSDKCONFIG_DEFAULTS="sdkconfig.defaults;sdkconfig.wifi" build
```

Current repository CLI target coverage for the streamer frontend includes
`ESP32`, `ESP32-C3`, `ESP32-C5`, `ESP32-C6`, and `ESP32-S3`.

## Observed ESP32-C3 Throughput

The table below summarizes the latest standalone streamer transport benchmark on
`ESP32-C3`, measured with collector-driven UDP stimulus and host-side receive
stats over `4 s` windows.

Benchmark firmware profile:

- `WIFI_PS_MIN_MODEM`
- `CONFIG_ESP_WIFI_DYNAMIC_TX_BUFFER_NUM=128`
- `CONFIG_ESP_WIFI_DYNAMIC_RX_BUFFER_NUM=128`
- `CONFIG_ESP_WIFI_STATIC_RX_BUFFER_NUM=16`
- `CONFIG_LWIP_TCPIP_RECVMBOX_SIZE=64`
- `CONFIG_LWIP_UDP_RECVMBOX_SIZE=32`
- `CONFIG_LWIP_IRAM_OPTIMIZATION=y`
- `CONFIG_ESPECTRE_STREAM_QUEUE_SLOTS=32`
- `CONFIG_ESPECTRE_STREAM_BATCH_MAX_RECORDS=4`
- `CONFIG_ESPECTRE_STREAM_BATCH_MAX_BYTES=1200`

Observed results:

| Requested Stimulus Rate | Observed Host Receive Rate | Host Drop Rate |
|-------------------------|----------------------------|----------------|
| `500 pps` | `~473 pps` | `~1.2%` |
| `650 pps` | `~618 pps` | `~0.0%` |
| `750 pps` | `~707 pps` | `~0.8%` |
| `850 pps` | `~806 pps` | `~0.2%` |
| `1000 pps` | `~935 pps` | `~1.1%` |
| `1200 pps` | `~1136 pps` | `~0.3%` |

Notes:

- host-side `requested pps` is the control target; the Python sender may
  slightly under-run or over-run during short windows
- the transport path is mostly packet-rate bound rather than byte-rate bound
- `queue=0` in the periodic log does not mean the queue never saturated; use
  `peak=<n>/<slots>` to inspect burst pressure between log ticks
- `batch=4` is the recommended default on ESP32-C3; `batch=8` slightly helped
  some `1000 pps` runs but was less robust at `1200 pps`

Practical guidance:

- use `1000 pps` as the recommended high-rate profile for ESP32-C3
- use `1200 pps` as an aggressive profile when a small amount of burst pressure
  is acceptable