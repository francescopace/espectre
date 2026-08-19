# Micro-ESPectre

**Python/MicroPython R&D layer for ESPectre Wi-Fi CSI sensing.**

Micro-ESPectre is the fast experimentation path inside the ESPectre platform. It implements the shared motion-detection ideas in Python so algorithms, parameters, MQTT payloads, and data-collection workflows can be tested quickly before stable behavior is promoted into the C++ `core` / `runtime` layers and their frontends.

This guide assumes basic Python and MQTT familiarity. CSI means channel state information, the per-packet Wi-Fi channel measurement used by the detectors.

Use Micro-ESPectre when you want:

- rapid Python-side prototyping without C++ firmware rebuilds
- MicroPython deployment for CSI sensing experiments
- MQTT-based runtime control and inspection
- cross-checks against the C++ detector behavior
- dataset collection and ML validation workflows

For the production firmware paths, start from the main [README](../../../README.md) and [SETUP.md](../../../docs/SETUP.md).

## Role in ESPectre

ESPectre v3 is a multi-frontend C++ sensing platform. Micro-ESPectre is not a separate product surface; it is the Python R&D layer that helps validate sensing changes before they become shared platform behavior.

| Layer | Purpose | Main users |
|-------|---------|------------|
| C++ platform | Shared `core` / `runtime` plus ESPHome, native, Matter, and streamer frontends | Smart home users, firmware developers, integrators |
| Micro-ESPectre | Python/MicroPython R&D and MQTT workflow | Researchers, developers, algorithm contributors |

Validated Micro-ESPectre changes should stay aligned with the C++ detector and runtime behavior. See [ARCHITECTURE.md](../../../docs/ARCHITECTURE.md) for the platform split.

## Upstream MicroPython CSI

Micro-ESPectre uses the [micropython-esp32-csi](https://github.com/francescopace/micropython-esp32-csi) firmware distribution for ESP32 CSI support. ESP32 Wi-Fi CSI support was also contributed upstream and merged into mainline MicroPython in [micropython/micropython#18460](https://github.com/micropython/micropython/pull/18460) for the `1.29.0` release cycle.

This matters because it reduces long-term maintenance risk for the MicroPython-based workflow and shows that ESPectre contributes enabling Wi-Fi sensing infrastructure back to the ecosystem.

## Requirements

Hardware:

- ESP32 board with CSI support (`ESP32`, `ESP32-C3`, `ESP32-C5`, `ESP32-S3`, `ESP32-C6`)
- 2.4 GHz Wi-Fi network

Software:

- repository Python environment, currently Python `3.14`
- MicroPython firmware with ESP32 CSI support
- MQTT broker for telemetry and runtime control

## Quick Start

From the repository root:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

./espectre micro flash --erase
./espectre micro deploy
./espectre micro run
```

On Windows, use `.\espectre.cmd` instead of `./espectre` and COM ports such as `COM5`.

The `micro` namespace owns only MicroPython device operations:

| Command | Purpose |
|---------|---------|
| `./espectre micro flash --erase` | Flash the CSI-enabled MicroPython firmware |
| `./espectre micro deploy` | Copy Micro-ESPectre Python sources to the device |
| `./espectre micro run` | Start the device application |
| `./espectre micro verify` | Check firmware and device readiness |

Host-side workflows live at the repository CLI root:

| Command | Purpose |
|---------|---------|
| `./espectre collect ...` | Unified host CLI for live detection, live recording, and legacy timed dataset collection |
| `./espectre mqtt` | Interactive MQTT command and telemetry console |
| `./espectre monitor ...` | Attach to serial logs with auto-reconnect; add `--reset` for a hard reset on open |

See the repository [CLI.md](../../../docs/CLI.md) for current CLI syntax and host-side workflow behavior, and the shared [SETUP.md](../../../docs/SETUP.md) for setup and frontend selection.

## Configure Wi-Fi and MQTT

Create a local config file:

```bash
cp src/python/micro_espectre/config_local.py.example src/python/micro_espectre/config_local.py
```

Set at least:

```python
WIFI_SSID = "YourWiFiSSID"
WIFI_PASSWORD = "YourWiFiPassword"

MQTT_BROKER = "homeassistant.local"
MQTT_CLIENT_ID = "micro-espectre"
MQTT_TOPIC_PREFIX = "espectre/v1/devices"
MQTT_USERNAME = "mqtt"
MQTT_PASSWORD = "mqtt"
MQTT_DEVICE_LABEL = "Lab prototype"  # Optional
```

Do not commit `config_local.py`.

## Runtime Behavior

Micro-ESPectre follows the same detector direction as the C++ platform:

```text
Boot -> AGC-active startup -> Lightweight threshold bootstrap or High Accuracy startup -> Detection Loop
```

### Detection Profiles

Micro-ESPectre implements the same two detector families as the C++ platform, `lightweight` and `high_accuracy`, described in [ALGORITHMS.md](../../../docs/ALGORITHMS.md).

Lightweight is the leaner path: its Lightweight implementation uses fewer feature trackers and less per-packet computation, but is less accurate and robust than High Accuracy on the maintained corpus. High Accuracy uses the ML implementation, with more working memory and CPU for its eight features and neural inference, but provides better detection quality and skips Lightweight's threshold calibration. Lightweight requires about 10 seconds of clean, ready quiet-room coverage after temporal warmup, and insufficient occupancy extends that wall-clock duration. High Accuracy still waits for CSI readiness and its feature window to fill.

Key config values live in `config.py`:

```python
DETECTION_ALGORITHM = "lightweight"  # "lightweight" or "high_accuracy"
CSI_TARGET_PPS = 100
TRAFFIC_GENERATOR_ENABLED = True  # False expects an external traffic source
SEGMENTATION_WINDOW_SIZE_MS = 1000
PUBLISH_INTERVAL_MS = 1000
EVALUATION_INTERVAL_MS = 250
MOTION_ON_HITS = 4
MOTION_OFF_HITS = 3
```

`CSI_TARGET_PPS` defines both the temporal detector grid and the internal generator target. `TRAFFIC_GENERATOR_ENABLED` selects traffic ownership independently; the target is always positive. Session MQTT `external` and `disabled` stop the local generator. Micro-ESPectre does not open a UDP listener, so it does not join multicast group `239.255.0.1`; ESP-IDF sensing frontends do. The production `TemporalCsiSampler` retains the packet nearest each slot center, enforces a half-slot minimum spacing derived from the target, preserves missing slots, and is imported unchanged by CPython collection, replay, training, and validation workflows. The live loop uses two fixed CSI payload buffers so the previous slot can be emitted while the current candidate is retained without hot-path allocation.

Lightweight selects its threshold automatically during startup calibration; keep the room quiet immediately after boot. A later quiet stretch can still lower that threshold, and the HA number plus Monitor follow the live value. High Accuracy uses its trained default threshold. Both thresholds remain adjustable at runtime. For the practical startup workflow, see [TUNING.md](../../../docs/TUNING.md). For the calibration formulas and detector theory, see [ALGORITHMS.md](../../../docs/ALGORITHMS.md).

### Filters

Both detector paths support the same lightweight filters. In Lightweight and High Accuracy, the single Hampel switch controls the active feature streams:

```python
ENABLE_HAMPEL_FILTER = True
HAMPEL_WINDOW = 7
HAMPEL_THRESHOLD = 5.0

ENABLE_LOWPASS_FILTER = False
LOWPASS_CUTOFF = 11.0
```

For tuning rationale, use [TUNING.md](../../../docs/TUNING.md). For detector theory, use [ALGORITHMS.md](../../../docs/ALGORITHMS.md).

## MQTT Surface

Micro-ESPectre publishes ESPectre Protocol telemetry with `frontend: "micro"`. The exact payload and topic model are defined in [ESPECTRE_PROTOCOL.md](../../../docs/ESPECTRE_PROTOCOL.md).

For protocol identity fields:

- `device_id` comes from `MQTT_CLIENT_ID`
- `device_name` is derived automatically from chip and `device_id`
- `device_label` is optional and can be supplied through `MQTT_DEVICE_LABEL`

Default telemetry topic shape:

```text
espectre/v1/devices/{MQTT_CLIENT_ID}/telemetry
```

Use:

```bash
./espectre mqtt
```

for interactive MQTT inspection and runtime commands. Micro-ESPectre `info` and `stats` use the same protocol schema as Native, including CSI traffic settings on `info` and the cached CSI/Wi-Fi diagnostic fields on `stats`. MQTT connect and the `info` command publish retained `info` so `./espectre mqtt` discovery sees `frontend: micro` instead of a leftover retained identity from another firmware on the same `device_id`. Micro-ESPectre accepts session-only `set_threshold`, `set_motion_hits`, `recalibrate`, `set_csi_traffic_mode`, and `set_traffic_generator_mode` commands over canonical MQTT, and it answers `commands` with that catalog. It advertises `supports_runtime_detector: false`, so detector switching remains out of scope on this frontend. Traffic ownership accepts `internal`, `external`, and `disabled`. For repository CLI behavior, including MQTT shell discovery and selection flow, use [CLI.md](../../../docs/CLI.md). Runtime changes made over MQTT are session-only unless the device code explicitly persists them. Micro-ESPectre advertises `supports_ota: false` and `supports_ble: false`; those commands are omitted from its catalog and rejected if sent.

### Home Assistant MQTT Discovery

Micro-ESPectre can optionally publish a small Home Assistant adapter surface on top of the ESPectre protocol topics. Enable it in `config_local.py` when the same broker is shared with Home Assistant:

```python
MQTT_HA_DISCOVERY_ENABLED = True
MQTT_HA_DISCOVERY_PREFIX = "homeassistant"
```

When enabled, the runtime:

- publishes retained discovery payloads for Motion Detected, Movement Score, Threshold, Motion On Hits, Motion Off Hits, CSI Traffic Ownership, CSI Traffic Source, Trigger Calibration, the ESPHome CSI diagnostic sensors, and Refresh Diagnostics
- publishes empty retained discovery payloads for leftover Intensity and previous Micro object IDs so Home Assistant entity IDs match ESPHome slugs after a prefix swap
- publishes plain HA availability, motion, movement, threshold, motion-hit, calibrate, traffic-control, and on-demand diagnostic state topics under the existing device topic base
- subscribes to `homeassistant/status` and republishes discovery when Home Assistant announces `online`

HA sensing cadences match ESPHome and Native MQTT so the same Home Assistant dashboard can be reused after replacing entity ID prefixes:

| Entity | Topic suffix | Cadence |
|--------|--------------|---------|
| Motion Detected | `ha/motion/state` | Filtered state edges |
| Movement Score | `ha/movement/state` | Detector evaluation (`EVALUATION_INTERVAL_MS`, default 250 ms) |
| Threshold | `ha/threshold/state` and `ha/threshold/set` | On change (operator write, calibration, or Lightweight settled-level recovery), plus connect/birth snapshot; writable 0.0–1.0 number |
| Motion On Hits | `ha/motion_on_hits/state` and `ha/motion_on_hits/set` | On change, plus connect/birth snapshot; writable 1–20 number |
| Motion Off Hits | `ha/motion_off_hits/state` and `ha/motion_off_hits/set` | On change, plus connect/birth snapshot; writable 1–20 number |
| CSI Traffic Ownership | `ha/csi_traffic_mode/state` and `ha/csi_traffic_mode/set` | On change, plus connect/birth snapshot; writable `internal`, `external`, or `disabled` select |
| CSI Traffic Source | `ha/traffic_generator_mode/state` and `ha/traffic_generator_mode/set` | On change, plus connect/birth snapshot; writable `ping` / `dns` select |
| Trigger Calibration | `ha/calibrate/state` and `ha/calibrate/set` | ON while recalibrating; ON starts startup recalibration, OFF is ignored while a session is running |
| Traffic TX Rate, CSI rates, occupancy, Wi-Fi channel, Wi-Fi RSSI | `ha/traffic_tx_rate/state`, `ha/csi_callback_rate/state`, `ha/csi_accepted_rate/state`, `ha/csi_admitted_rate/state`, `ha/csi_filtered_rate/state`, `ha/csi_missing_rate/state`, `ha/csi_excess_rate/state`, `ha/csi_stale_rate/state`, `ha/csi_out_of_order_rate/state`, `ha/csi_occupancy/state`, `ha/wifi_channel/state`, `ha/wifi_rssi/state` | On demand after Refresh Diagnostics; diagnostic category |
| Refresh Diagnostics | `ha/diagnostics/set` | Button; publishes the latest cached diagnostic sample |

Entity IDs look like `sensor.micro_micro_espectre_movement_score`. Copy the ESPHome dashboard from [`home-assistant-dashboard.yaml`](../../cpp/frontend/esphome/examples/home-assistant-dashboard.yaml) and replace the `espectre_` prefix.

![ESPectre Home Assistant dashboard](../../../docs/web/assets/images/guides/home-assistant-dashboard.png)

*Home Assistant dashboard reused from the ESPHome example. Replace the `espectre_` prefix so the entity IDs match this Micro-ESPectre device.*

The canonical ESPectre protocol topics remain unchanged.

## Relevant Paths

```text
src/python/micro_espectre/       MicroPython runtime sources
src/python/espectre_cli/         Repository CLI implementation
tools/                           Host-side analysis and validation tools
docs/web/                       Website
test/python/                     Python test suite
data/                            Local CSI datasets
```

## Testing

Run the Python test suite from the repository root:

```bash
source .venv/bin/activate
pytest test/python -v
```

With coverage:

```bash
pytest test/python -v --cov=src/python/micro_espectre --cov-report=term-missing
```
