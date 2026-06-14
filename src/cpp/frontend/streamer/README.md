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

It is not the source of truth for:

- algorithm theory
- ML data collection workflow as a whole
- general `core` / `runtime` architecture

Use these global docs for those topics:

- [`../../../../docs/ARCHITECTURE.md`](../../../../docs/ARCHITECTURE.md)
- [`../../../../docs/ML_DATA_COLLECTION.md`](../../../../docs/ML_DATA_COLLECTION.md)
- [`../../../../docs/ALGORITHMS.md`](../../../../docs/ALGORITHMS.md)

## Important Architectural Note

The streamer frontend uses the lower-level `runtime/esp_idf` modules directly.
It does not go through the `IEspectreRuntime` facade.

That is intentional:

- the goal is raw CSI transport, not motion-detection entity exposure
- the firmware needs a compact packet-oriented streaming path
- the state machine is streamer-specific (`WAIT_WIFI` -> `STREAMING`)

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

## Frontend Configuration

Frontend-specific options are declared in [`espectre/Kconfig.projbuild`](espectre/Kconfig.projbuild).

Versioned defaults live in [`app/sdkconfig.defaults`](app/sdkconfig.defaults).
Local Wi-Fi credentials should live in `app/sdkconfig.wifi`, which is gitignored.

Typical local override file:

```ini
CONFIG_ESPECTRE_WIFI_SSID="YourSSID"
CONFIG_ESPECTRE_WIFI_PASSWORD="YourPassword"
# CONFIG_ESPECTRE_WIFI_BSSID is not set
```

Key knobs in the frontend surface:

- `ESPECTRE_WIFI_SSID`
- `ESPECTRE_WIFI_PASSWORD`
- `ESPECTRE_WIFI_BSSID`
- `ESPECTRE_STREAM_OUTPUT_ENABLED`
- `ESPECTRE_COLLECTOR_PORT`
- `ESPECTRE_TRAFFIC_RX_PORT`
- `ESPECTRE_TRAFFIC_RX_MULTICAST_GROUP`
- `ESPECTRE_GAIN_LOCK_ENABLED`
- `ESPECTRE_GAIN_LOCK_MODE_*`
- `ESPECTRE_STREAM_QUEUE_SLOTS`
- `ESPECTRE_STREAM_LOG_INTERVAL_MS`

Runtime behavior notes:

- the streamer no longer owns an internal traffic generator
- the collector address is learned from the source IP of the latest UDP stimulus packet
- the UDP stimulus payload may carry the `ESTM` metadata header
  (`magic + version + role + stimulus_id`), which is propagated into the CSI
  stream when present

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
./espectre streamer build --chip c3
./espectre streamer flash --chip c3 --port /dev/cu.usbmodemXXXX
./espectre streamer monitor --chip c3 --port /dev/cu.usbmodemXXXX
```

When `app/sdkconfig.wifi` exists, the repository CLI automatically passes
`sdkconfig.defaults;sdkconfig.wifi` to `idf.py` for `build`.

Raw ESP-IDF flow:

```bash
cd src/cpp/frontend/streamer/app
idf.py -DSDKCONFIG_DEFAULTS="sdkconfig.defaults;sdkconfig.wifi" set-target esp32c3
idf.py -DSDKCONFIG_DEFAULTS="sdkconfig.defaults;sdkconfig.wifi" build
```

Current repository CLI target coverage for the streamer frontend is intentionally
minimal and currently centered on `ESP32-C3`.

## Relationship to ML Data Collection

[`../../../../docs/ML_DATA_COLLECTION.md`](../../../../docs/ML_DATA_COLLECTION.md)
remains the workflow-oriented guide for collecting labeled datasets.

This README owns:

- the streamer firmware surface
- the UDP packet format
- the frontend-specific configuration knobs

`ML_DATA_COLLECTION.md` should refer here for protocol/surface details and stay
focused on collection, labeling, and training workflow.

## Boundaries

The streamer README should not become a second copy of:

- runtime architecture
- algorithm theory
- ML best practices
- host-side analysis workflow

## Related Docs

- [`../../../../docs/ML_DATA_COLLECTION.md`](../../../../docs/ML_DATA_COLLECTION.md):
  collection and dataset workflow
- [`../../../../docs/ARCHITECTURE.md`](../../../../docs/ARCHITECTURE.md):
  shared architecture and frontend boundaries
- [`../ble/README.md`](../ble/README.md):
  standalone BLE frontend
- [`../matter/README.md`](../matter/README.md):
  Matter frontend
- [`../esphome/README.md`](../esphome/README.md):
  ESPHome frontend
