# Micro-ESPectre

**Python/MicroPython R&D layer for ESPectre Wi-Fi CSI sensing.**

Micro-ESPectre is the fast experimentation path inside the ESPectre platform. It
implements the shared motion-detection ideas in Python so algorithms,
parameters, MQTT payloads, and data-collection workflows can be tested quickly
before stable behavior is promoted into the C++ `core` / `runtime` layers and
their frontends.

Use Micro-ESPectre when you want:

- rapid Python-side prototyping without C++ firmware rebuilds
- MicroPython deployment for CSI sensing experiments
- MQTT-based runtime control and inspection
- cross-checks against the C++ detector behavior
- dataset collection and ML validation workflows

For the production firmware paths, start from the main [README](../../../README.md)
and [SETUP.md](../../../docs/SETUP.md).

## Role in ESPectre

ESPectre v3 is a multi-frontend C++ sensing platform. Micro-ESPectre is not a
separate product surface; it is the Python R&D layer that helps validate sensing
changes before they become shared platform behavior.

| Layer | Purpose | Main users |
|-------|---------|------------|
| C++ platform | Shared `core` / `runtime` plus ESPHome, native, Matter, and streamer frontends | Smart home users, firmware developers, integrators |
| Micro-ESPectre | Python/MicroPython R&D and MQTT workflow | Researchers, developers, algorithm contributors |

Validated Micro-ESPectre changes should stay aligned with the C++ detector and
runtime behavior. See [ARCHITECTURE.md](../../../docs/ARCHITECTURE.md) for the
platform split.

## Upstream MicroPython CSI

Micro-ESPectre uses the
[micropython-esp32-csi](https://github.com/francescopace/micropython-esp32-csi)
firmware distribution for ESP32 CSI support. ESP32 Wi-Fi CSI support was also
contributed upstream and merged into mainline MicroPython in
[micropython/micropython#18460](https://github.com/micropython/micropython/pull/18460)
for the `1.29.0` release cycle.

This matters because it reduces long-term maintenance risk for the
MicroPython-based workflow and shows that ESPectre contributes enabling Wi-Fi
sensing infrastructure back to the ecosystem.

## Requirements

Hardware:

- ESP32 board with CSI support (`ESP32`, `ESP32-C3`, `ESP32-S3`, `ESP32-C6`)
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

On Windows, use `.\espectre.cmd` instead of `./espectre` and COM ports such as
`COM5`.

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
| `./espectre collect ...` | Collect labeled CSI datasets |
| `./espectre detect ...` | Inspect live ML inference from the streamer |
| `./espectre mqtt` | Interactive MQTT command and telemetry console |
| `./espectre ui ...` | Open local browser tools |
| `./espectre monitor ...` | Attach to serial logs |

Use `./espectre --help` and the shared [SETUP.md](../../../docs/SETUP.md) for
current CLI syntax.

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
```

Do not commit `config_local.py`.

## Runtime Behavior

Micro-ESPectre follows the same detector direction as the C++ platform:

```text
Boot -> Gain Lock -> MVS threshold bootstrap or ML startup -> Detection Loop
```

### Detection Algorithms

| Algorithm | Method | Startup behavior |
|-----------|--------|------------------|
| `mvs` | Moving variance over turbulence | Gain lock plus startup threshold bootstrap |
| `ml` | 8-feature MLP over turbulence windows | Gain lock only, fixed threshold |

Key config values live in `config.py`:

```python
DETECTION_ALGORITHM = "mvs"  # "mvs" or "ml"
GAIN_LOCK_MODE = "auto"      # "auto", "enabled", or "disabled"
SEG_THRESHOLD = "auto"       # "auto", "min", or 0.0-10.0
SEG_WINDOW_SIZE = 100
EVALUATION_INTERVAL = 25
MOTION_ON_HITS = 3
MOTION_OFF_HITS = 3
```

Keep the room quiet after boot in `mvs` mode while threshold bootstrap runs. In
`ml` mode, only gain lock runs.

### Filters

Both detector paths support the same lightweight filters:

```python
ENABLE_HAMPEL_FILTER = True
HAMPEL_WINDOW = 7
HAMPEL_THRESHOLD = 5.0

ENABLE_LOWPASS_FILTER = False
LOWPASS_CUTOFF = 11.0
```

For tuning rationale, use [TUNING.md](../../../docs/TUNING.md). For detector
theory, use [ALGORITHMS.md](../../../docs/ALGORITHMS.md).

## MQTT Surface

Micro-ESPectre publishes ESPectre Protocol telemetry with `frontend: "micro"`.
The exact payload and topic model are defined in
[ESPECTRE_PROTOCOL.md](../../../docs/ESPECTRE_PROTOCOL.md).

Default telemetry topic shape:

```text
espectre/v1/devices/{MQTT_CLIENT_ID}/telemetry
```

Use:

```bash
./espectre mqtt
```

for interactive MQTT inspection and runtime commands. Runtime changes made over
MQTT are session-only unless the device code explicitly persists them.

## Data Collection and ML

For v3, the most useful community datasets are room-state captures:

- `empty`
- `static_presence`
- `motion`

Use:

- [ML_DATA_COLLECTION.md](../../../docs/ML_DATA_COLLECTION.md) for collection and labeling
- [ML_TRAINING.md](../../../docs/ML_TRAINING.md) for training, validation, and export
- [PERFORMANCE.md](../../../docs/PERFORMANCE.md) for current metrics and caveats

The current ML detector uses an `8 -> 32 -> 16 -> 1` MLP with 8 relative
turbulence-window features. The exported model is shared by Python and C++
runtimes.

## Relevant Paths

```text
src/python/micro_espectre/       MicroPython runtime sources
src/python/espectre_cli/         Repository CLI implementation
tools/                           Host-side analysis and validation tools
tools/web/                       Local browser utilities
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
