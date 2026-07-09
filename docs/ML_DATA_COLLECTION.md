# ML Data Collection Guide

**Building labeled CSI datasets for machine learning**

This guide covers how to collect and label CSI data for training ML models.
For ESPectre v3, the priority is robust room-state sensing across chips,
routers, and environments using `empty`, `static_presence`, and `motion`
captures. Gesture recognition, HAR, and people counting remain future research
tracks built on the same collection infrastructure.

## Status

| Feature | Status |
|---------|--------|
| Data collection infrastructure | ✅ Ready |
| Feature extraction | ✅ Ready |
| ML detector (MLP) | ✅ Ready |
| Training script | ✅ Ready |
| Runtime weight export | ✅ Ready |
| Room-state datasets (`empty`, `static_presence`, `motion`) | ✅ Current priority |
| Static presence robustness | 🔜 Planned |
| Gesture recognition | 🔬 Future research |
| Human Activity Recognition (HAR) | 🔬 Future research |
| People counting | 🔬 Future research |

---

## Supported Hardware

**Recommended chips for ML data collection:**
- ESP32
- ESP32-S3
- ESP32-C3
- ESP32-C5
- ESP32-C6
> **Note**: ESPectre now keeps AGC active during collection.

---

## Getting Started

### 1. Activate Virtual Environment

Before running any command, activate the virtual environment:

```bash
cd .
source ../.venv/bin/activate  # Your prompt should show (.venv)
```

On Windows PowerShell, activate the repository environment with
`..\.venv\Scripts\Activate.ps1`, then replace `./espectre` examples with
`.\espectre.cmd` and use COM ports such as `COM5`.

### 2. Flash and Deploy (First Time Only)

If you haven't already flashed the firmware:

```bash
./espectre micro flash --erase
./espectre micro deploy
```

### 3. Start the Streamer Firmware

Start the standalone streamer firmware on the device. The host collector now
drives the UDP target traffic, and the streamer learns the collector IP from
incoming traffic packets (default CSI UDP port: `5001`):

```bash
./espectre streamer build --chip <chip> --clean
./espectre streamer flash --port <serial_port>
./espectre monitor --port <serial_port>
```

The streamer frontend README is the source of truth for the firmware surface,
UDP packet format, and frontend-specific configuration:
[`../src/cpp/frontend/streamer/README.md`](../src/cpp/frontend/streamer/README.md).

Use that README as the source of truth for:

- local streamer Wi-Fi configuration via `sdkconfig.wifi`
- optional streamer Wi-Fi provisioning via Web Bluetooth
- transport tuning knobs such as queue depth and batching
- observed standalone streamer throughput on `ESP32-C3`

**Features:**
- 64 subcarriers (HT20 mode)
- 32-bit sequence numbers for packet loss detection
- collector-driven traffic rate (see streamer README for practical transport profiles and benchmarks)

### 4. Optional: Inspect Live Motion With `collect`

If you want to validate runtime detector behavior before saving data, run the
same host-side pipeline in live mode without saving files:

```bash
./espectre collect --target 192.168.1.50 --no-save
```

`espectre collect` reads threshold mode, the fixed production subcarrier set,
Hampel, low-pass, and hit filtering from `src/python/micro_espectre/config.py` and
`src/python/micro_espectre/config_local.py`, just like the rest of micro-ESPectre. Use
`--target <ip>` to point at the firmware device or shared target
group, and `--bind-ip <local_ip>` only when auto-detection picks the wrong host
interface.

In `--no-save` mode, the command now focuses on rolling status output: startup
calibration, ready-state tracking, packet counters, and one summary line per
device and detector slot. It no longer exposes the old per-publish debug logs
for turbulence windows or ML feature vectors.

For startup-calibrated detectors, the live collect path mirrors the runtime
startup threshold bootstrap. `classic` uses the shared L1-Delta motion-first
startup path with an internal quiet-first fallback, and `auto` applies the
detector factor to the resulting `threshold_metric` instead of assuming one
fixed startup statistic for every session.

If you pass a comma-separated detector list to `--detector`, `espectre collect`
runs the detectors side by side on the same live CSI stream. This is useful for
quick A/B comparisons because each detector keeps its own threshold bootstrap
and status line while sharing the same packets, device metadata, and target
traffic.

The same live path can also save the raw CSI packets it is inspecting. This
uses the ordinary dataset format; no derived ML scores, feature vectors, or
states are stored because they can be reconstructed offline from the raw CSI
and the exported model.

```bash
# Mixed idle/motion/idle smoke-test capture: store under data/test/
./espectre collect \
  --target 192.168.1.50 \
  --label test \
  --duration 45 \
  --description "live collect ML, idle-motion-idle"

# Homogeneous hard-negative capture: store under data/empty/
./espectre collect \
  --target 192.168.1.50 \
  --label empty \
  --duration 60 \
  --description "live collect ML, empty room"
```

Use `test` for mixed sessions where the room state changes during the capture.
Use training labels such as `empty`, `static_presence`, or `motion` only when
the whole capture is label-homogeneous.

---

## Data Collection with `espectre collect`

The `espectre collect` subcommand now covers both live inspection and dataset
capture. Each saved capture window emits one `.npz` per `device_id`, so a
shared-target session can save multiple dataset files without mixing devices
into one file.

### Commands

| Command | Description |
|---------|-------------|
| `./espectre collect --target <ip> --no-save` | Inspect live detector status without saving files |
| `./espectre collect --target <ip> --no-save --detector classic,ml` | Compare multiple detectors side by side on the same live CSI stream, with one status line per detector |
| `./espectre collect --label <name> --duration <sec> --target <ip>` | Run live collect and save the accepted capture window for the specified duration |
| `./espectre collect --label <name> --target <ip>` | Run live collect, wait for the ready gate, then keep saving until `Ctrl+C` |
| `./espectre collect --label <name> --samples <n> --target <ip>` | Legacy timed dataset mode: record `n` timed collections |
| `./espectre collect --label <name> --count <n> --target <ip>` | Alias for `--samples`, useful for repeated timed collections |
| `./espectre collect --label <name> --start-delay <sec> --target <ip>` | Legacy timed dataset mode: wait before starting collection |
| `./espectre collect --label <name> --target <ip> --reference-every <N>` | Mark every `N`th traffic packet as a reference frame |
| `./espectre collect --label <name> --contributor <user>` | Override contributor (auto-detected from git) |
| `./espectre collect --label <name> --description "text"` | Add description to sample |
| `./espectre collect --info` | Show dataset statistics |

When saving is enabled, live collect keeps a pre-recording readiness gate: the
selected detector must stay below its effective threshold for 3 continuous
seconds before packets are accepted into the saved capture. For `classic`, this
happens after the startup calibration phase.

### Options

#### Core modes

| Option | Meaning |
|--------|---------|
| `--info` | Print dataset statistics and exit without starting UDP collection |
| `--no-save` | Live inspection mode: do not save `.npz` files |
| `--label <name>` | Dataset label used when saving live captures or legacy timed samples |
| `--duration <sec>` | In live mode, stop after the accepted recording window reaches the requested duration. In legacy timed dataset mode, duration per sample |
| `--samples`, `--count`, `-n` | Legacy timed dataset mode: save multiple samples |
| `--start-delay <sec>` | Legacy timed dataset mode: wait before starting |

#### Detector selection

| Option | Meaning |
|--------|---------|
| `--detector <name[,name...]>` | Select one or more live detectors, for example `classic`, `ml`, or `classic,ml` (`classic` default) |

#### Transport and dataset metadata

| Option | Meaning |
|--------|---------|
| `--target`, `-t` | One or more IPv4 target destinations, comma-separated for multi-unicast |
| `--bind-ip <ip>` | Override the local bind interface used for UDP reception |
| `--udp-port <port>` | CSI UDP listen port (default `5001`) |
| `--target-port <port>` | UDP port used by the target listener (default `9999`) |
| `--rate <pps>` | Traffic send rate in packets per second (default `100`) |
| `--reference-every <n>` | Mark every `n`th traffic packet as a reference frame |
| `--contributor <user>` | Override contributor metadata for saved files |
| `--description "text"` | Store a human-readable description in dataset metadata |

#### Save semantics

- In live mode with `--label` and no `--no-save`, saving starts only after the
  ready gate reaches `READY`.
- With `--duration`, `Ctrl+C` before the requested duration aborts the run and
  discards the partial live capture instead of saving it.
- Without `--duration`, `Ctrl+C` stops the live run and saves the packets
  already accepted after the ready gate.
- With `--no-save`, `Ctrl+C` only stops the live session; no `.npz` files are created.

### Recording Samples

```bash
# Live inspection only, no files written
./espectre collect --target 192.168.1.50 --no-save --detector classic

# Live comparison on the same stream: one rolling status line per detector
./espectre collect --target 192.168.1.50 --no-save --detector classic,ml

# Live recording: save after the stream stays below threshold for 3s
# and stop automatically after 60 accepted seconds
./espectre collect --label static_presence --duration 60 --target 192.168.1.50

# Record 30 seconds of motion
./espectre collect --label motion --duration 30 --target 192.168.1.50

# Record with explicit contributor override
./espectre collect --label gesture --samples 10 --target 192.168.1.50 --contributor otheruser

# Mark every 20th traffic packet as a reference frame
./espectre collect --label static_presence --duration 30 --target 192.168.1.50 --reference-every 20

# Shared-target session: all streamers subscribed to the multicast group
# save their own per-device files during the same capture window
./espectre collect --label empty --duration 30 --target 239.1.1.50

# Wait 15 seconds, then record 3 timed collections
./espectre collect --label static_presence --duration 10 --count 3 --start-delay 15 --target 192.168.1.50

./espectre collect --label static_presence --duration 10 --target 192.168.1.50

# Continuous live recording until Ctrl+C
./espectre collect --label test --target 192.168.1.50 --detector ml
```

Accepted target forms:

- unicast IPv4, for one streamer
- multicast IPv4, for multiple streamers joined to the same group
- broadcast IPv4, when your network setup intentionally uses broadcast target traffic

Every saved `.npz` is single-device. If packets arrive without `device_id`
metadata, the collector fails instead of emitting a mixed or anonymous file.

### Reference Frames

The host collector can optionally mark some traffic packets as reference
frames with:

```bash
./espectre collect --label static_presence --target 192.168.1.50 --reference-every 20
```

Semantics:

- `--reference-every 0` means measurement-only traffic (default)
- `--reference-every N` means every `N`th traffic packet is sent with the
  `reference` role in the `ESTM` header
- the streamer copies that role into the outgoing CSI UDP packet through
  `STREAM_FLAG_REFERENCE_FRAME`

This is collector-driven metadata. It does not change how the streamer captures
CSI; it only tags frames so downstream tooling can distinguish:

- measurement frames: ordinary collection samples
- reference frames: collector-selected anchor samples for later alignment,
  normalization, or analysis policies

In the current ESPectre workflow, most ordinary ML dataset collection still uses
measurement-only traffic. The main reason to preserve `stimulus_id` and
optional reference markers is to keep datasets usable for future multi-device
host-side experiments, especially phase-coherence studies and temporally aligned
feature fusion. See [`EXPERIMENTS.md`](EXPERIMENTS.md) for the research context
and current limitations of those paths.

Use reference frames only when your host-side processing pipeline has a clear
reason to distinguish them. For ordinary dataset collection, leaving
`--reference-every` at `0` is usually the simplest choice.

For room-state datasets, this means:

- `static_presence` and `motion` are normally collected as measurement-only
  samples
- some imported `empty` datasets may include both measurement and reference
  frames when they come from a multi-device streamer session
- quality checks that compare `empty` against `static_presence` should drop
  reference frames from `empty` first, so the comparison stays aligned with the
  ordinary ESPectre collection format

### Viewing Dataset

```bash
./espectre collect --info
```

Output:
```
  Label                   Samples
  --------------------------------
  static_presence               12
  wave                         10
  swipe                        10
  ...
  --------------------------------
  Total                        47
```

---

## Dataset Format

### Directory Structure

```
data/
├── dataset_info.json          # Global metadata
├── empty/
│   └── ...
├── static_presence/
│   ├── static_presence_c6_64sc_dev0000000000abcdef_20251212_142443_381306_0001.npz
│   └── ...
├── motion/
│   ├── motion_c6_64sc_dev0000000000123456_20251212_142443_381512_0002.npz
│   └── ...
└── ...
```

**Note**: HT20 only - all datasets use 64 subcarriers.

File naming convention:
`{label}_{chip}_{num_sc}sc_{device_token}_{timestamp}_{save_index}.npz`

### Dataset Info (dataset_info.json)

Central metadata file for the dataset:

```json
{
  "format_version": "1.1",
  "labels": {
    "empty": { "description": "Quiet room, no human present" },
    "static_presence": { "description": "Human present in room, remaining still" },
    "motion": { "description": "Human movement in room" }
  },
  "files": {
    "static_presence": [
      {
        "filename": "static_presence_c6_64sc_dev0000000000abcdef_20251212_142443_381306_0001.npz",
        "chip": "C6",
        "subcarriers": 64,
        "device_id": "0x0000000000abcdef",
        "contributor": "francescopace",
        "collected_at": "2025-12-12T14:24:43.381306",
        "duration_ms": 10000,
        "num_packets": 1000,
        "description": "HT20 static presence sample"
      },
      {
        "filename": "static_presence_esp32_64sc_dev000000000000f00d_20260214_183059_355439_0002.npz",
        "chip": "ESP32",
        "subcarriers": 64,
        "device_id": "0x000000000000f00d",
        "contributor": "francescopace",
        "collected_at": "2026-02-14T18:30:59.355439",
        "duration_ms": 9998,
        "num_packets": 961,
        "description": "HT20 static presence, AGC-active normalized pipeline"
      }
    ]
  },
  "environments": [...]
}
```

| Field | Description |
|-------|-------------|
| `filename` | NPZ file name |
| `chip` | ESP32 chip type (C6, S3, ESP32) |
| `subcarriers` | Number of subcarriers (64 for HT20) |
| `device_id` | Canonical `0x...` device identifier for the saved single-device file |
| `contributor` | GitHub username of data collector |
| `collected_at` | ISO timestamp of collection |
| `duration_ms` | Sample duration in milliseconds |
| `num_packets` | Number of CSI packets |
| `description` | Human-readable description |

### Sample Format (.npz)

Each `.npz` file contains a minimal, compact format optimized for ML training:

| Field | Type | Description |
|-------|------|-------------|
| `csi_data` | `int8[N, SC*2]` | Raw I/Q data (N packets × SC subcarriers × 2) |
| `num_subcarriers` | `int` | Number of subcarriers (64 for HT20) |
| `label` | `str` | Sample label (e.g., "static_presence", "motion") |
| `chip` | `str` | ESP32 chip type (e.g., "c6", "s3") |
| `collected_at` | `str` | ISO timestamp of collection |
| `duration_ms` | `float` | Sample duration in milliseconds |
| `format_version` | `str` | NPZ format version ("1.1") |
| `stream_seq_num` | `uint32[N]` | Per-packet stream sequence numbers |
| `device_ticks_us` | `uint64[N]` | Device-side monotonic timestamps in microseconds |
| `device_id` | `uint64` | Device identifier for the single-device capture file; same canonical identity exposed by MQTT/BLE in hex string form |
| `wifi_rx_ts_us` | `uint32[N]` | Optional Wi-Fi RX timestamps when available |
| `wifi_rx_start_ts_ns` | `uint64[N]` | Optional hardware-derived RX-start estimate |
| `channel` | `uint8[N]` | Optional per-packet Wi-Fi channel metadata |
| `rssi_dbm` | `int16[N]` | Optional per-packet RSSI metadata |
| `stimulus_id` | `uint32[N]` | Optional per-packet stimulus identifier |

Amplitudes and phases can be computed on-the-fly from `csi_data`:

```python
# Espressif CSI format: [Imaginary, Real, ...] per subcarrier
Q = csi_data[:, 0::2].astype(float)  # Imaginary (Q) at even indices
I = csi_data[:, 1::2].astype(float)  # Real (I) at odd indices
amplitudes = np.sqrt(I**2 + Q**2)
phases = np.arctan2(Q, I)
```

### Loading Data

```python
import numpy as np

# Load single sample
data = np.load('data/static_presence/static_presence_c6_64sc_dev0000000000abcdef_20251212_142443_381306_0001.npz')
csi_data = data['csi_data']        # Shape: (N, 128) for 64 subcarriers
label = str(data['label'])         # 'static_presence'
num_sc = int(data['num_subcarriers'])  # 64

# Compute amplitudes from raw I/Q data
# Espressif CSI format: [Imaginary, Real, ...] per subcarrier
Q = csi_data[:, 0::2].astype(float)  # Imaginary (Q) - Shape: (N, 64)
I = csi_data[:, 1::2].astype(float)  # Real (I) - Shape: (N, 64)
amplitudes = np.sqrt(I**2 + Q**2)    # Shape: (N, 64)
phases = np.arctan2(Q, I)            # Shape: (N, 64)
```

### Using the tool library

```python
from tools.lib.csi_io import load_npz_as_packets
from pathlib import Path
import numpy as np

# Load a sample file (run from the repo root)
packets = load_npz_as_packets(
    Path('data/static_presence/static_presence_c6_64sc_dev0000000000abcdef_20251212_142443_381306_0001.npz')
)

for pkt in packets:
    csi_data = pkt['csi_data']           # Shape: (128,) - raw I/Q data
    label = pkt['label']
    
    # Calculate amplitudes from I/Q pairs
    Q = csi_data[0::2].astype(float)     # Imaginary (odd indices)
    I = csi_data[1::2].astype(float)     # Real (even indices)
    amplitudes = np.sqrt(I**2 + Q**2)    # Shape: (64,)
    # Process...
```

---

## AGC-Active Data Model

CSI amplitudes may vary with signal strength because AGC remains active by design.  
ESPectre compensates for this by using normalized turbulence and relative ML features rather than relying on forced-gain control.

### Canonical Labels

The dataset now uses three canonical room-state labels:

- `empty`: quiet room, no human present
- `static_presence`: human present in room, remaining still
- `motion`: human movement in room

Older `baseline` captures were migrated to `static_presence`. The old
`baseline` / `movement` naming is no longer used for dataset files,
directories, or `dataset_info.json` metadata.

Imported streamer sessions can also populate the `empty` label. When those
captures come from a multi-device workflow, the stored NPZ may preserve
reference-frame markers in `is_reference` while still using the normal ESPectre
dataset schema.

### How It Works

The ML training script uses normalized turbulence (`std/mean`) for all chips
before the sliding-window features are computed. The exported ML features are
still relative ratios such as
`std/mean`, `iqr/mean`, `mad/mean`, and normalized waveform length.

### Viewing Files and Training Metadata

```bash
python tools/10_train_ml_model.py --info
```

This prints the current dataset summary used by the training pipeline.

---

## Best Practices

### Recording Guidelines

| Aspect | Recommendation |
|--------|----------------|
| **Duration** | 30-60 seconds per sample (packet count depends on the chosen traffic rate) |
| **Repetitions** | 10+ samples per label for variability |
| **Environment** | Same environment for all samples in a session |
| **Position** | Vary position/distance between samples for robustness |
| **Labels** | Use lowercase, no spaces (e.g., `wave`, `swipe_left`) |

### Label Naming Convention

```
# Good labels
static_presence
wave
swipe_left
swipe_right
push
pull
circle_cw
circle_ccw

# Avoid
Wave          # uppercase
swipe left    # spaces
gesture1      # non-descriptive
```

### Session Workflow

1. **Prepare environment**: Ensure room is quiet for `static_presence`
2. **Record static presence first**: `./espectre collect --label static_presence --duration 60 --target <ip>`
3. **Record motion**: `./espectre collect --label motion --duration 60 --target <ip>`
4. **Verify dataset**: `./espectre collect --info`
5. **Backup data**: Copy `data/` to safe location

Note: Contributor is auto-detected from `git config user.name`. Use `--contributor` to override.

---

## Analysis Tools

After collecting data, use the analysis scripts in `tools/`:

```bash
cd tools

# Visualize raw CSI data
python 1_analyze_raw_data.py

# Compare filter placement on your data
python 4_analyze_filter_location.py --plot
```

See [tools/README.md](../tools/README.md) for complete documentation of all analysis scripts.

---

## Training the ML Model

Once you have collected labeled data, move to the dedicated training guide:

- [ML_TRAINING.md](ML_TRAINING.md) - full ML training workflow, trainer flags,
  export artifacts, gain-shift diagnostics, and post-training regressions

Quick start:

```bash
# ML training extras
pip install -r requirements-ml.txt

# Train model
python tools/10_train_ml_model.py
```

---

## Advanced: Custom CSI Receiver (Optional)

For custom real-time processing, you can use `CSIReceiver` as a library:

```python
from tools.lib.csi_io import CSIReceiver

def my_callback(packet):
    # packet is a CSIPacket dataclass with:
    # - timestamp: Reception timestamp (seconds since epoch)
    # - seq_num: Sequence number (0-255)
    # - num_subcarriers: Number of subcarriers (64 for HT20)
    # - iq_raw: Raw I/Q values as int8 array
    # - chip: Chip type (e.g., 'c6', 's3') - auto-detected from stream
    print(f"Chip: {packet.chip}, Seq: {packet.seq_num}, SC: {packet.num_subcarriers}")

receiver = CSIReceiver(port=5001)
receiver.add_callback(my_callback)
receiver.run(timeout=60)  # Run for 60 seconds
```

### UDP Packet Format

The UDP stream format is now documented in the streamer frontend README:
[`../src/cpp/frontend/streamer/README.md`](../src/cpp/frontend/streamer/README.md).

Collection-specific notes that matter here:

- ESPectre uses HT20 mode (64 subcarriers) for consistent cross-chip datasets
- chip type is auto-detected from the stream
- `wifi_rx_start_ts_ns` is a hardware-derived estimate, not a guaranteed
  nanosecond-accurate timestamp

---

## Contributing Your Data

Help build a diverse CSI dataset for the community. For v3, the most useful
contributions are room-state captures that improve cross-device reliability and
reduce false positives in real homes and labs.

### How to Contribute

1. **Collect data** following the [Best Practices](#best-practices) above
2. **Ensure quality**: At least 10 samples per label, 30+ seconds each
3. **Document your setup**:
   - ESP32 model (S3, C6, etc.)
   - Distance from router
   - Room type (living room, office, etc.)
   - Any notable characteristics
4. **Share via GitHub**:
   - Add your data to `data/<label>/`
   - Submit a Pull Request to the `develop` branch

### What We're Looking For

Current v3 dataset priorities:

| Priority | Label | Description | Why it matters |
|----------|-------|-------------|----------------|
| High | `empty` | Empty room, no movement | Reduces false positives and hard-negative failures |
| High | `static_presence` | Person present but mostly still | Helps separate occupancy-like stillness from empty-room noise |
| High | `motion` | Walking or ordinary room movement | Maintains recall across homes, chips, routers, and layouts |

Future research datasets can include gestures, HAR, people counting, and other
advanced sensing labels, but those are not the primary v3 release target.

### Data Privacy

- **CSI data is anonymous** - it contains only radio channel characteristics
- No personal information, images, or audio
- You retain ownership of your contributions
- All contributions will be credited

---

## References

For scientific background on CSI-based gesture recognition and HAR:

- **WiGest**: WiFi-based gesture recognition (IEEE INFOCOM 2015)
- **Widar 3.0**: Cross-domain gesture recognition dataset
- **SignFi**: Sign language recognition with WiFi

See [References](ALGORITHMS.md#references) in the algorithms guide for the
complete bibliography.
