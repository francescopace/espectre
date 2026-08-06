# ESPectre Streamer Frontend

This directory contains the standalone CSI streamer frontend.

Like `esphome`, `native`, and `matter`, the streamer now goes through the shared
`IEspectreRuntime` controller contract. It still remains a dedicated firmware
path for raw CSI collection and transport to host-side tooling through a
stream-specific backend.

This README covers the streamer frontend surface and the UDP stream protocol.

## Scope

The streamer frontend is responsible for:

- capturing CSI on-device
- receiving lightweight host UDP pacing traffic
- immediate AGC-active normalized startup
- packaging CSI into the UDP stream format
- emitting one uplink CSI record toward the collector for each valid UDP
  pacing packet received from the host, batching several records per
  datagram to reduce uplink packet rate

Use [`ML_DATA_COLLECTION.md`](../../../../docs/ML_DATA_COLLECTION.md) for the
ML data collection workflow.

## Important Architectural Note

The streamer now uses `RuntimeFrontendController` with a dedicated
`StreamEspIdfRuntime` backend. That keeps the controller/runtime split aligned
across frontends without forcing the motion-oriented `EspIdfRuntime` onto the
raw CSI transport workflow.

That shape is intentional:

- the goal is raw CSI transport, not motion-detection entity exposure
- the firmware still needs a tight packet-oriented streaming path
- the state machine remains streamer-specific (`WAIT_WIFI` -> `STREAMING`)

The standalone Wi-Fi setup path is shared with the other ESP-IDF standalone
firmware targets through `StandaloneWifiService`, while the stream-specific CSI
capture and UDP transport stay isolated in `StreamEspIdfRuntime` and
`CsiStreamTransport`.

The streamer reads Wi-Fi credentials only from the active `sdkconfig` surface,
typically `app/sdkconfig.wifi`. It does not expose a separate BLE, MQTT, or OTA
control plane.

Because those credentials are supplied at build time, releases, snapshots, and
the web flasher do not publish Streamer binaries. Build and flash this frontend
locally with the repository CLI.

## Directory Layout

- [`streamer_frontend.cpp`](espectre/streamer_frontend.cpp),
  [`streamer_frontend.h`](espectre/streamer_frontend.h):
  thin frontend adapter over `RuntimeFrontendController`
- [`csi_stream_protocol.h`](../../runtime/csi_stream_protocol.h):
  UDP stream header and flags
- [`stream_esp_idf_runtime.cpp`](../../runtime/esp_idf/stream_esp_idf_runtime.cpp),
  [`stream_esp_idf_runtime.h`](../../runtime/esp_idf/stream_esp_idf_runtime.h):
  streamer-specific runtime backend
- [`csi_stream_transport.cpp`](../../runtime/esp_idf/csi_stream_transport.cpp),
  [`csi_stream_transport.h`](../../runtime/esp_idf/csi_stream_transport.h):
  pacing-driven CSI stream transport and telemetry
- [`Kconfig.projbuild`](espectre/Kconfig.projbuild):
  frontend-specific configuration surface
- [`app/`](app/):
  standalone ESP-IDF firmware app

## Workflow States

The streamer frontend uses these states:

- `WAIT_WIFI`
- `WIFI_READY`
- `CSI_READY`
- `STREAMING`

This state machine is implemented in
[`stream_esp_idf_runtime.h`](../../runtime/esp_idf/stream_esp_idf_runtime.h).

## UDP Stream Protocol

Protocol constants live in [`csi_stream_protocol.h`](../../runtime/csi_stream_protocol.h).

### UDP Pacing Packet

The host-side collector sends ordinary UDP datagrams to the firmware pacing
port. The firmware does not require a dedicated application header.

Current behavior:

- any UDP datagram received on `ESPECTRE_TRAFFIC_RX_PORT` is treated as valid
  pacing traffic
- the device learns the collector IP from the UDP source address of the latest
  received pacing packet
- the collector controls the effective stream rate only by sending more or
  fewer UDP packets

### CSI Stream Packet

Current version:

- magic: `0x4353`
- version: `7`
- header size: `64` bytes

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
| `channel` | `uint8` | Wi-Fi channel |
| `rssi_dbm` | `int8` | RSSI |
| `noise_floor_dbm` | `int8` | Noise floor |
| `tx_backpressure_total` | `uint64` | Cumulative TX backpressure events |
| `stream_fresh_total` | `uint32` | Cumulative fresh CSI records |
| `pacing_rx_total` | `uint32` | Cumulative received pacing packets |
| `phy_mode` | `uint8` | Normalized PHY mode enum |
| `ltf_type` | `uint8` | LTF represented by the CSI payload |
| `channel_width` | `uint8` | Normalized channel-width enum |

Flags:

| Bit | Constant | Meaning |
|-----|----------|---------|
| 0 | `STREAM_FLAG_FIRST_WORD_INVALID` | Espressif CSI flag |
| 1 | `STREAM_FLAG_WIFI_RX_TS_VALID` | `wifi_rx_ts_us` valid |
| 2 | `STREAM_FLAG_WIFI_RX_START_TS_NS_VALID` | `wifi_rx_start_ts_ns` valid |
| 3 | `STREAM_FLAG_CSI_FRESH` | fresh CSI sample record |

PHY modes are `unknown`, `legacy`, `HT`, `VHT`, `HE-SU`, `HE-MU`,
`HE-ERSU`, and `HE-TB`. LTF types emitted by the current streamer sensing
contract are `unknown`, `HT-LTF`, `VHT-LTF`, and `HE-LTF`. The current streamer sensing contract emits HT20
records with `phy_mode=HT`, `ltf_type=HT-LTF`, and `channel_width=20`.

Channel-width values are `unknown`, `20`, `40`, `80`, `160`, and `80+80` MHz.
The existing `channel` field distinguishes 2.4 GHz from 5 GHz channels, while
the normalized width enum leaves the protocol ready for wider future captures.

Payload:

- raw I/Q values in Espressif ordering
- typical HT20 record: `64 + 128 = 192 bytes`
- the sender concatenates up to `ESPECTRE_STREAM_TX_BATCH_RECORDS` complete
  records (default 4, maximum 7) into one UDP datagram; the receiver parses
  records back-to-back until the datagram is exhausted
- partial batches are flushed after `100 ms` so low pacing rates keep bounded
  record latency
- pacing slots without a fresh CSI sample produce no stream record

## Frontend Configuration

Frontend-specific options are declared in [`Kconfig.projbuild`](espectre/Kconfig.projbuild).

Versioned defaults live in [`sdkconfig.defaults`](app/sdkconfig.defaults).
The streamer also carries the shared standalone ESP-IDF Wi-Fi transport
baseline: AMPDU enabled, Wi-Fi buffers `16/128/128`, lwIP mailboxes `64/32`,
and `CONFIG_LWIP_IRAM_OPTIMIZATION=y`.
Chip-specific overrides may also live in `app/sdkconfig.defaults.<idf_target>`
when a target needs extra tuning on top of that shared baseline.
Local Wi-Fi credentials should live in `app/sdkconfig.wifi`, which is gitignored.
The streamer reads Wi-Fi credentials from the active `sdkconfig` surface, so
`app/sdkconfig.wifi` is the recommended machine-local override file.

Typical local override file:

```ini
CONFIG_ESPECTRE_WIFI_SSID="YourSSID"
CONFIG_ESPECTRE_WIFI_PASSWORD="YourPassword"
# CONFIG_ESPECTRE_WIFI_BSSID is not set
CONFIG_ESPECTRE_WIFI_CHANNEL=0
CONFIG_ESPECTRE_WIFI_BAND_2G=y
```

Recommended workflow for local Wi-Fi configuration:

1. create `src/cpp/frontend/streamer/app/sdkconfig.wifi`
2. set `CONFIG_ESPECTRE_WIFI_SSID` and `CONFIG_ESPECTRE_WIFI_PASSWORD`
3. leave `CONFIG_ESPECTRE_WIFI_BSSID` unset unless you intentionally want to
   pin the streamer to a specific AP radio
4. keep `CONFIG_ESPECTRE_WIFI_BAND_2G=y`, or explicitly select 5 GHz or AUTO on
   ESP32-C5
5. leave `CONFIG_ESPECTRE_WIFI_CHANNEL=0` unless you intentionally want to hint
   a specific AP channel
6. build via `./espectre streamer build --chip <esp32|c3|c5|c6|s3>`, which
   automatically passes `sdkconfig.defaults`, the matching
   `sdkconfig.defaults.<idf_target>` when present, and `sdkconfig.wifi` to
   `idf.py`;
   add `--clean` when you want a fresh build

Notes:

- `sdkconfig.wifi` is the recommended place for machine-local credentials
  because it is ignored by git
- `sdkconfig.defaults.<idf_target>` is optional and lets a specific chip layer
  transport tuning on top of the shared defaults without affecting other
  streamer targets
- the firmware still initializes `nvs_flash` because ESP-IDF Wi-Fi startup
  requires it, but streamer credentials are no longer loaded from NVS
- keep `CONFIG_ESPECTRE_WIFI_BSSID` unset for normal use; the streamer will
  scan all channels and connect to the strongest matching AP
- keep `CONFIG_ESPECTRE_WIFI_BAND_2G=y` for the validated default; only an
  ESP32-C5 build may select `CONFIG_ESPECTRE_WIFI_BAND_5G` or
  `CONFIG_ESPECTRE_WIFI_BAND_AUTO`, and the sensing PHY remains HT20
- set `CONFIG_ESPECTRE_WIFI_BSSID="aa:bb:cc:dd:ee:ff"` only when you need to
  force a specific AP radio for repeatable RF tests; this also enables fast
  scan instead of the default full scan
- set `CONFIG_ESPECTRE_WIFI_CHANNEL` together with BSSID when you want
  deterministic BSSID+channel association for CSI captures; the channel must
  belong to the selected band
- if you change `sdkconfig.defaults`, `sdkconfig.wifi`, or the frontend Kconfig
  surface and the generated `sdkconfig` appears stale, rebuild with
  `./espectre streamer build --chip <esp32|c3|c5|c6|s3> --clean` so ESP-IDF
  regenerates the active config from the defaults
- the active generated files `sdkconfig`, `sdkconfig.old`, and
  `dependencies.lock` are build artifacts and should remain untracked

Key knobs in the frontend surface:

- `ESPECTRE_WIFI_SSID`
- `ESPECTRE_WIFI_PASSWORD`
- `ESPECTRE_WIFI_BSSID`
- `ESPECTRE_WIFI_CHANNEL`
Shared runtime traffic ingress:

- `ESPECTRE_TRAFFIC_RX_PORT`
- `ESPECTRE_TRAFFIC_RX_MULTICAST_GROUP`

Streamer-local transport:

- `ESPECTRE_COLLECTOR_PORT`
- `ESPECTRE_STREAM_LOG_INTERVAL_MS`

Runtime behavior notes:

- when the shared capture service observes the first valid CSI packet on a new
  Wi-Fi channel, Streamer rejects the transition packet, resets the UDP
  transport session, and rearms CSI capture through its normal workflow
- the streamer no longer uses the shared internal traffic generator to emit the
  CSI stream; the collector controls pacing directly by sending UDP pacing
  packets to `ESPECTRE_TRAFFIC_RX_PORT`
- the collector uses two adaptive pacing signals: `tx_backpressure_total`
  drives protective slowdowns, and the delivered-record rate drives delivery
  targeting; the `stream_fresh_total / pacing_rx_total` delta ratio gates the
  latter and still never triggers a slowdown on its own
- adaptive collection uses the same chip-independent policy for every target:
  it holds the requested rate during CSI-only deficits, spaces reductions
  caused by actual TX backpressure across three control windows, and does not
  fall below 70% of the requested target
- delivery targeting treats `--pps` as the delivered-record goal for
  group-addressed (broadcast or multicast) pacing, whose loss is retry-less by
  design: when delivered records fall below ~95% of the target while the
  device converts received pacing into fresh CSI cleanly (fresh ratio at least
  0.90) with zero TX backpressure in the window, pacing rises above the target
  in proportional steps up to 1.5x to compensate path loss, and trims back
  toward the target once delivery overshoots; unicast pacing never boosts,
  because its delivery is MAC-retransmitted and a deficit there is device-side;
  both act on a smoothed delivery measurement so the rate settles instead of
  chasing per-window RF variance, and the boost is revoked outright when the
  fresh ratio degrades while above target, because excess pacing can itself
  starve CSI conversion
- the collector address is learned from the source IP of the latest valid UDP
  pacing packet
- CSI capture excludes 802.11 ACK frames (`dump_ack_en=0`) and keeps only
  frames transmitted by the associated AP (source MAC equals the BSSID)
- the CSI callback pushes samples into a short FIFO (16 slots, oldest dropped
  on overflow) so bursty arrivals such as DTIM-released broadcast pacing keep
  every sample; each pacing slot drains one queued sample and emitted records
  carry `STREAM_FLAG_CSI_FRESH`
- the stream always carries the full normalized CSI payload
- the stream socket uses the default best-effort access category (TID 0,
  AC_BE), which supports AMPDU and avoids routing sustained CSI traffic
  through the short, non-aggregating voice queue
- AMPDU stays enabled on the streamer firmware; CSI is sourced from ordinary
  collector-paced unicast data frames rather than ACK observations

Periodic telemetry focuses on the traffic-paced flow:

- `csi_ap`: valid CSI callbacks sourced from the associated AP
- `csi_filt`: valid CSI callbacks dropped by the BSSID source filter
- `udp_rx`: valid UDP pacing packets received
- `udp_tx`: stream datagrams accepted by `sendto()`; with record batching each
  datagram carries up to `ESPECTRE_STREAM_TX_BATCH_RECORDS` records
- `fresh`: stream records carrying a new CSI sample
- `repeat`: pacing slots that arrived without a fresh CSI sample ready to send
- `tx_err`: datagrams rejected by `sendto()`
- `tx_bp`: datagrams rejected specifically because the TX path reported
  backpressure (`ENOMEM`, `ENOBUFS`, `EAGAIN`, or `EWOULDBLOCK`)
- `age_ms`: age of the last accepted CSI sample

In a healthy stream, `udp_tx` should sit close to the emitted fresh-record rate
divided by the configured batch size, and `fresh` should track the downlink CSI
opportunities (`csi_ap`). A high `repeat` share means the AP is not generating
enough downlink frames toward the device, so pacing is out-running fresh CSI
arrival and the streamer is intentionally dropping stale slots instead of
re-sending old samples.

## Pacing-Driven Flow

The streamer expects lightweight host UDP pacing traffic plus a paired uplink
CSI stream.

The collector is responsible for:

- sending UDP pacing traffic to the configured target port
- controlling the effective stream rate by changing the pacing packet rate
- receiving the CSI UDP stream on `ESPECTRE_COLLECTOR_PORT`

The streamer is responsible for:

- learning the collector IP from the source address of valid UDP pacing
  traffic
- advertising a one-shot discovery endpoint over mDNS/DNS-SD as
  `_espectre-streamer._udp.local.`
- embedding the latest AP-sourced CSI sample only when it is fresh
- flagging emitted CSI records with `STREAM_FLAG_CSI_FRESH`
- emitting CSI records only for pacing slots that coincide with a fresh sample,
  batching them into uplink datagrams, and retargeting live when the collector
  address changes

The discovery advertisement publishes:

- SRV port = `ESPECTRE_TRAFFIC_RX_PORT` (the pacing target)
- TXT `device_id` = canonical ESPectre device ID already carried in the CSI stream
- TXT `chip` = active ESP-IDF target name
- TXT `traffic_port` = pacing target port
- TXT `collector_port` = CSI uplink port

On clean Wi-Fi disconnects, the firmware disables the mDNS service so peers can
observe a best-effort goodbye. On reconnects and IP changes, it re-announces the
same service identity on the new address. Host discovery still validates the
announced `device_id` against the first CSI packets, so stale records or DHCP IP
reuse cannot silently redirect a capture to the wrong device.

When multiple streamers share the same target, the host collector is
expected to demultiplex incoming CSI by `device_id` and save one dataset file
per device. Mixed-device `.npz` files are not part of the supported workflow.

## Build and Tooling

Before building locally, complete the shared
[`ESP-IDF Local Build Prerequisite`](../../../../docs/SETUP.md#esp-idf-local-build-prerequisite).
The repository CLI auto-detects a reusable ESP-IDF install, so the wrapper-first
workflow does not require a separate setup check before build.
See [`CLI.md`](../../../../docs/CLI.md) for shared CLI syntax, host-side
tools, and wrapper behavior.

Repository CLI:

```bash
./espectre streamer build --chip s3 --clean
./espectre streamer flash --port /dev/cu.usbmodemXXXX
./espectre monitor --port /dev/cu.usbmodemXXXX
```

On Windows, use `.\espectre.cmd streamer ...` and `.\espectre.cmd monitor --port COM5`.
If the wrapper cannot find or validate ESP-IDF, run `.\espectre.cmd doctor`
or `./espectre doctor` for troubleshooting.

When `app/sdkconfig.wifi` exists, the repository CLI automatically passes
`sdkconfig.defaults`, the matching `sdkconfig.defaults.<idf_target>` when
present, and `sdkconfig.wifi` to `idf.py` for `build`.

<details>
<summary>Advanced raw ESP-IDF flow</summary>

```bash
cd src/cpp/frontend/streamer/app
idf.py -DSDKCONFIG_DEFAULTS="sdkconfig.defaults;sdkconfig.wifi" set-target esp32
idf.py -DSDKCONFIG_DEFAULTS="sdkconfig.defaults;sdkconfig.wifi" build
```

</details>

## Firmware Scope

The streamer firmware is intentionally narrow:

- Wi-Fi credentials come from `app/sdkconfig.wifi` or other active build-time
  `sdkconfig` defaults
- there is no separate BLE, MQTT, or OTA control surface in this frontend
- CSI streaming is UDP-only and controlled by host pacing traffic

Current repository CLI target coverage for the streamer frontend includes
`ESP32`, `ESP32-C3`, `ESP32-C5`, `ESP32-C6`, and `ESP32-S3`.

## Observed ESP32-C3 Throughput

The table below summarizes standalone streamer transport benchmark captures on
`ESP32-C3`, recorded on `2026-07-21`. It was measured
with collector-driven fixed UDP target traffic and saved `10 s` captures; 
observed host receive rate is derived from capture duration, while host drop 
rate and max sequence gap are derived from `stream_seq_num` gaps in the saved stream. 
Broader project performance metrics live in
[`docs/performance`](../../../../docs/performance/README.md).

Benchmark firmware profile:

- `CONFIG_ESP_WIFI_DYNAMIC_TX_BUFFER_NUM=128`
- `CONFIG_ESP_WIFI_DYNAMIC_RX_BUFFER_NUM=128`
- `CONFIG_ESP_WIFI_STATIC_RX_BUFFER_NUM=16`
- `CONFIG_LWIP_TCPIP_RECVMBOX_SIZE=64`
- `CONFIG_LWIP_UDP_RECVMBOX_SIZE=32`
- `CONFIG_LWIP_IRAM_OPTIMIZATION=y`
- UDP sender queue depth is fixed in firmware for one-ACK-one-UDP streaming

Observed results:

| Requested Traffic Rate | Observed Host Receive Rate | Host Drop Rate | Max Sequence Gap |
|-------------------------|----------------------------|----------------|------------------|
| `500 pps` | `~496 pps` | `~0.2%` | `4 packets` |
| `750 pps` | `~746 pps` | `~0.1%` | `10 packets` |
| `1000 pps` | `~997 pps` | `~0.2%` | `7 packets` |
| `1250 pps` | `~1248 pps` | `~0.4%` | `32 packets` |
| `1500 pps` | `~1441 pps` | `~0.6%` | `47 packets` |
| `1750 pps` | `~1674 pps` | `~0.1%` | `5 packets` |
| `2000 pps` | `~1853 pps` | `~0.1%` | `10 packets` |

Notes:

- host-side `requested pps` is the control target; the Python sender may
  still land slightly under-run or over-run across a finite capture window
- the transport path is mostly packet-rate bound rather than byte-rate bound
- `queue=0` in the periodic log does not mean the queue never saturated; use
  `peak=<n>/<slots>` to inspect burst pressure between log ticks
- `retry` counts retry-marked frames, while `wifi_dup` counts the repeated
  copies actually filtered early by the streamer; `retry` may therefore be
  higher than `wifi_dup`
- max sequence gap is the worst observed burst loss in one capture, measured as
  the largest count of consecutive `stream_seq_num` values missing between two
  received records
