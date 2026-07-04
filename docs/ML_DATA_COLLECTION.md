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
| Feature extraction (8 relative ML features) | ✅ Ready |
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
- ESP32-S3
- ESP32-C3
- ESP32-C5
- ESP32-C6

**Also supported:**
- ESP32 (original) - Does not support AGC gain lock, but data is usable for ML training (ML features are relative to local turbulence mean)

> **Note**: AGC gain lock stabilizes CSI amplitudes during data collection. Without it, amplitudes vary with signal strength. Both MVS and ML use the no-gain-lock CV-normalized turbulence path for streams where `gain_locked=false`; ML then extracts relative per-window features such as `std/mean`, `iqr/mean`, and `mad/mean`.

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
drives the UDP stimulus and the streamer learns the collector IP from incoming
stimulus packets (default CSI UDP port: `5001`):

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
- Gain lock phase (~3s) for stable CSI acquisition
- 64 subcarriers (HT20 mode)
- 32-bit sequence numbers for packet loss detection
- collector-driven stimulus rate (see streamer README for practical transport profiles and benchmarks)

### 4. Optional: Inspect Live ML Motion Detection

If you want to validate runtime ML behavior before recording data, run live
host-side inference from the UDP CSI stream:

```bash
./espectre detect --stimulus-target 192.168.1.50 --log-turbulence
```

`espectre detect` reads threshold, the fixed production subcarrier set,
Hampel, low-pass, and hit filtering from `src/python/micro_espectre/config.py` and
`src/python/micro_espectre/config_local.py`, just like the rest of micro-ESPectre. Use
`--stimulus-target <ip>` to point at the firmware device or shared stimulus
group, and `--bind-ip <local_ip>` only when auto-detection picks the wrong host
interface.

Live detection can also save the raw CSI packets it is inspecting. This uses
the same dataset format as `collect`; no derived ML scores, feature
vectors, or states are stored because they can be reconstructed offline from
the raw CSI and the exported model.

```bash
# Mixed idle/motion/idle smoke-test capture: store under data/test/
./espectre detect \
  --stimulus-target 192.168.1.50 \
  --log-features \
  --capture-label test \
  --capture-duration 45 \
  --description "live detect ML, idle-motion-idle"

# Homogeneous hard-negative capture: store under data/empty/
./espectre detect \
  --stimulus-target 192.168.1.50 \
  --capture-label empty \
  --capture-duration 60 \
  --description "live detect ML, empty room"
```

Use `test` for mixed sessions where the room state changes during the capture.
Use training labels such as `empty`, `static_presence`, or `motion` only when
the whole capture is label-homogeneous.

---

## Data Collection with `espectre collect`

The `espectre collect` subcommand provides a streamlined workflow for recording labeled CSI samples.
Each capture window now saves one `.npz` per `device_id`, so a shared-stimulus
session can emit multiple dataset files without mixing devices into one file.

### Commands

| Command | Description |
|---------|-------------|
| `./espectre collect --label <name> --duration <sec> --stimulus-target <ip>` | Record for the specified duration from one or more devices sharing the target |
| `./espectre collect --label <name> --samples <n> --stimulus-target <ip>` | Record `n` timed collections |
| `./espectre collect --label <name> --count <n> --stimulus-target <ip>` | Alias for `--samples`, useful for repeated timed collections |
| `./espectre collect --label <name> --start-delay <sec> --stimulus-target <ip>` | Wait before starting collection |
| `./espectre collect --label <name> --stimulus-target <ip> --reference-every <N>` | Mark every `N`th stimulus packet as a reference frame |
| `./espectre collect --label <name> --contributor <user>` | Override contributor (auto-detected from git) |
| `./espectre collect --label <name> --description "text"` | Add description to sample |
| `./espectre collect --info` | Show dataset statistics |

Gain lock status is **automatically detected** from the CSI stream and saved in `dataset_info.json`.

`detect --capture-label <name>` is a convenience path for live detector
smoke tests: it records the same raw CSI schema while printing ML output.
Prefer `collect` for ordinary scripted dataset collection and its
pre-recording stable-scene gate.

### Recording Samples

```bash
# Record 60 seconds of static presence from one device
./espectre collect --label static_presence --duration 60 --stimulus-target 192.168.1.50

# Record 30 seconds of motion
./espectre collect --label motion --duration 30 --stimulus-target 192.168.1.50

# Record with explicit contributor override
./espectre collect --label gesture --samples 10 --interactive --stimulus-target 192.168.1.50 --contributor otheruser

# Mark every 20th stimulus packet as a reference frame
./espectre collect --label static_presence --duration 30 --stimulus-target 192.168.1.50 --reference-every 20

# Shared-stimulus session: all streamers subscribed to the multicast group
# save their own per-device files during the same capture window
./espectre collect --label empty --duration 30 --stimulus-target 239.1.1.50

# Wait 15 seconds, then record 3 timed collections
./espectre collect --label static_presence --duration 10 --count 3 --start-delay 15 --stimulus-target 192.168.1.50

# Gain lock status is auto-detected from the CSI stream
# No need to specify --no-gain-lock, it's automatic!
./espectre collect --label static_presence --duration 10 --stimulus-target 192.168.1.50
```

Accepted target forms:

- unicast IPv4, for one streamer
- multicast IPv4, for multiple streamers joined to the same group
- broadcast IPv4, when your network setup intentionally uses broadcast stimulus

Every saved `.npz` is single-device. If packets arrive without `device_id`
metadata, the collector fails instead of emitting a mixed or anonymous file.

### Reference Frames

The host collector can optionally mark some stimulus packets as reference
frames with:

```bash
./espectre collect --label static_presence --stimulus-target 192.168.1.50 --reference-every 20
```

Semantics:

- `--reference-every 0` means measurement-only stimulus (default)
- `--reference-every N` means every `N`th stimulus packet is sent with the
  `reference` role in the `ESTM` header
- the streamer copies that role into the outgoing CSI UDP packet through
  `STREAM_FLAG_REFERENCE_FRAME`

This is collector-driven metadata. It does not change how the streamer captures
CSI; it only tags frames so downstream tooling can distinguish:

- measurement frames: ordinary collection samples
- reference frames: collector-selected anchor samples for later alignment,
  normalization, or analysis policies

In the current ESPectre workflow, most ordinary ML dataset collection still uses
measurement-only stimulus. The main reason to preserve `stimulus_id` and
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
        "device_id": 11259375,
        "device_token": "dev0000000000abcdef",
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
        "device_id": 61453,
        "device_token": "dev000000000000f00d",
        "contributor": "francescopace",
        "gain_locked": false,
        "collected_at": "2026-02-14T18:30:59.355439",
        "duration_ms": 9998,
        "num_packets": 961,
        "description": "HT20 static presence, no gain lock (ESP32 lacks AGC lock support)"
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
| `device_id` | Numeric `uint64` device identifier stored in each single-device file; MQTT/BLE surface the same identity as a `0x...` hex string |
| `device_token` | Stable ASCII token used in filenames |
| `contributor` | GitHub username of data collector |
| `collected_at` | ISO timestamp of collection |
| `duration_ms` | Sample duration in milliseconds |
| `num_packets` | Number of CSI packets |
| `gain_locked` | `true` if AGC gain lock was active during collection |
| `description` | Human-readable description |

### Sample Format (.npz)

Each `.npz` file contains a minimal, compact format optimized for ML training:

| Field | Type | Description |
|-------|------|-------------|
| `csi_data` | `int8[N, SC*2]` | Raw I/Q data (N packets × SC subcarriers × 2) |
| `num_subcarriers` | `int` | Number of subcarriers (64 for HT20) |
| `label` | `str` | Sample label (e.g., "static_presence", "motion") |
| `chip` | `str` | ESP32 chip type (e.g., "c6", "s3") |
| `gain_locked` | `bool` | Whether AGC gain lock was active during collection |
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

### Using csi_utils

```python
from tools.csi_utils import load_npz_as_packets
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

## Data Without Gain Lock

Some ESP32 chips (original ESP32) or data collection sessions may not have AGC gain lock enabled. This causes CSI amplitudes to vary with signal strength rather than just motion.

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

The ML training script uses gain-mode-aware turbulence for all chips:
gain-locked streams use raw turbulence, while streams without gain lock use
CV-normalized turbulence (`std/mean`) before the sliding-window features are
computed. The exported ML features are still relative ratios such as
`std/mean`, `iqr/mean`, `mad/mean`, and normalized waveform length.

### When CV Normalization Is Applied

CV normalization is used by both detectors when gain lock is unavailable:
- **ESP32 (original)**: CV normalization is used since AGC gain lock is not supported
- **Data collected before enabling gain lock**: CV normalization applies for older captures
- **Future compatibility**: Any data where amplitudes are unreliable

### Automatic Detection

The collector **automatically detects** the gain lock status from the CSI stream:

1. The ESP32 firmware sends a `gain_locked` flag in each UDP packet
2. The collector saves this flag in the `.npz` file
3. `dataset_info.json` stores `gain_locked: false` when gain lock was not applied

No manual flags needed - the system handles everything automatically!

### Viewing Files with CV Normalization

```bash
python tools/10_train_ml_model.py --info
```

This shows which files use CV normalization.

---

## Best Practices

### Recording Guidelines

| Aspect | Recommendation |
|--------|----------------|
| **Duration** | 30-60 seconds per sample (packet count depends on the chosen stimulus rate) |
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
2. **Record static presence first**: `./espectre collect --label static_presence --duration 60 --stimulus-target <ip>`
3. **Record motion**: `./espectre collect --label motion --duration 60 --stimulus-target <ip>`
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
from csi_utils import CSIReceiver

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

- `gain_locked` is carried by the stream and saved into dataset metadata
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
