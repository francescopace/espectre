# ML Data Collection Guide

**Building labeled CSI datasets for machine learning**

This guide covers how to collect and label CSI data for training ML models. This infrastructure lays the groundwork for advanced Wi-Fi sensing features (gesture recognition, HAR, people counting) planned for ESPectre 3.x.

## Status

| Feature | Status |
|---------|--------|
| Data collection infrastructure | ✅ Ready |
| Feature extraction (9 features) | ✅ Ready |
| ML detector (MLP) | ✅ Ready |
| Training script | ✅ Ready |
| TFLite export | ✅ Ready |
| Gesture recognition | 🔜 Planned (3.x) |
| Human Activity Recognition (HAR) | 🔜 Planned (3.x) |
| People counting | 🔜 Planned (3.x) |

---

## Supported Hardware

**Recommended chips for ML data collection:**
- ESP32-S3
- ESP32-C3
- ESP32-C6

**Also supported:**
- ESP32 (original) - Does not support AGC gain lock, but data is usable for ML training (raw std is used for all chips)

> **Note**: AGC gain lock stabilizes CSI amplitudes during data collection. Without it, amplitudes vary with signal strength. The ML training pipeline and MLDetector always use raw std (`σ`) for turbulence, regardless of gain lock status. CV normalization (`σ/μ`) is only used by MVS detection.

---

## Getting Started

### 1. Activate Virtual Environment

Before running any command, activate the virtual environment:

```bash
cd .
source ../.venv/bin/activate  # Your prompt should show (.venv)
```

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
./espectre streamer flash --chip <chip> --port <serial_port>
./espectre streamer monitor --chip <chip> --port <serial_port>
```

The streamer frontend README is the source of truth for the firmware surface,
UDP packet format, and frontend-specific configuration:
[`../src/cpp/frontend/streamer/README.md`](../src/cpp/frontend/streamer/README.md).

Use that README as the source of truth for:

- local streamer Wi-Fi configuration via `sdkconfig.wifi`
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
./espectre micro detect --streamer-ip 192.168.1.50 --log-turbulence
```

`espectre micro detect` reads threshold, the fixed production subcarrier set,
Hampel, low-pass, and hit filtering from `src/python/config.py` and
`src/python/config_local.py`, just like the rest of micro-ESPectre. Use
`--streamer-ip <device_ip>` to point at the firmware device and `--bind-ip
<local_ip>` only when auto-detection picks the wrong host interface.

---

## Data Collection with `espectre micro collect`

The `espectre micro collect` subcommand provides a streamlined workflow for recording labeled CSI samples.

### Commands

| Command | Description |
|---------|-------------|
| `./espectre micro collect --label <name> --duration <sec> --streamer-ip <device_ip>` | Record for specified duration |
| `./espectre micro collect --label <name> --samples <n> --streamer-ip <device_ip>` | Record n samples interactively |
| `./espectre micro collect --label <name> --streamer-ip <device_ip> --reference-every <N>` | Mark every `N`th stimulus packet as a reference frame |
| `./espectre micro collect --label <name> --contributor <user>` | Override contributor (auto-detected from git) |
| `./espectre micro collect --label <name> --description "text"` | Add description to sample |
| `./espectre micro collect --info` | Show dataset statistics |

Gain lock status is **automatically detected** from the CSI stream and saved in `dataset_info.json`.

### Recording Samples

```bash
# Record 60 seconds of baseline (contributor auto-detected from git config)
./espectre micro collect --label baseline --duration 60 --streamer-ip 192.168.1.50

# Record 30 seconds of movement
./espectre micro collect --label movement --duration 30 --streamer-ip 192.168.1.50

# Record with explicit contributor override
./espectre micro collect --label gesture --samples 10 --interactive --streamer-ip 192.168.1.50 --contributor otheruser

# Mark every 20th stimulus packet as a reference frame
./espectre micro collect --label baseline --duration 30 --streamer-ip 192.168.1.50 --reference-every 20

# Gain lock status is auto-detected from the CSI stream
# No need to specify --no-gain-lock, it's automatic!
./espectre micro collect --label baseline --duration 10 --streamer-ip 192.168.1.50
```

### Reference Frames

The host collector can optionally mark some stimulus packets as reference
frames with:

```bash
./espectre micro collect --label baseline --streamer-ip 192.168.1.50 --reference-every 20
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

Use reference frames only when your host-side processing pipeline has a clear
reason to distinguish them. For ordinary dataset collection, leaving
`--reference-every` at `0` is usually the simplest choice.

### Viewing Dataset

```bash
./espectre micro collect --info
```

Output:
```
  Label                   Samples
  --------------------------------
  idle                         12
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
├── baseline/
│   ├── baseline_c6_64sc_20251212_142443.npz
│   └── ...
├── movement/
│   ├── movement_c6_64sc_20251212_142443.npz
│   └── ...
└── ...
```

**Note**: HT20 only - all datasets use 64 subcarriers.

File naming convention: `{label}_{chip}_{num_sc}sc_{timestamp}.npz`

### Dataset Info (dataset_info.json)

Central metadata file for the dataset:

```json
{
  "format_version": "1.1",
  "labels": {
    "baseline": { "description": "Quiet room, no motion" },
    "movement": { "description": "Human movement in room" }
  },
  "files": {
    "baseline": [
      {
        "filename": "baseline_c6_64sc_20251212_142443.npz",
        "chip": "C6",
        "subcarriers": 64,
        "contributor": "francescopace",
        "collected_at": "2025-12-12T14:24:43.381306",
        "duration_ms": 10000,
        "num_packets": 1000,
        "description": "HT20 baseline sample"
      },
      {
        "filename": "baseline_esp32_64sc_20260214_183059.npz",
        "chip": "ESP32",
        "subcarriers": 64,
        "contributor": "francescopace",
        "gain_locked": false,
        "collected_at": "2026-02-14T18:30:59.355439",
        "duration_ms": 9998,
        "num_packets": 961,
        "description": "HT20 baseline, no gain lock (ESP32 lacks AGC lock support)"
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
| `label` | `str` | Sample label (e.g., "baseline", "movement") |
| `chip` | `str` | ESP32 chip type (e.g., "c6", "s3") |
| `gain_locked` | `bool` | Whether AGC gain lock was active during collection |
| `collected_at` | `str` | ISO timestamp of collection |
| `duration_ms` | `float` | Sample duration in milliseconds |
| `format_version` | `str` | NPZ format version ("1.1") |
| `stream_seq_num` | `uint32[N]` | Per-packet stream sequence numbers |
| `device_ticks_us` | `uint64[N]` | Device-side monotonic timestamps in microseconds |
| `device_id` | `uint64` | Optional device identifier for multi-device fusion |
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
data = np.load('data/baseline/baseline_c6_64sc_20251212_142443.npz')
csi_data = data['csi_data']        # Shape: (N, 128) for 64 subcarriers
label = str(data['label'])         # 'baseline'
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
packets = load_npz_as_packets(Path('data/baseline/baseline_c6_64sc_20251212_142443.npz'))

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

### How It Works

The ML training script uses **raw std** for all chips, including those without gain lock. CV normalization is not applied during ML training or inference — it is only used by the MVS detector.

### When CV Normalization Is Applied

CV normalization is only used by the **MVS detector**, not by ML:
- **ESP32 (original)**: MVS uses CV normalization since AGC gain lock is not supported
- **Data collected before enabling gain lock**: MVS applies CV normalization for older captures
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
idle
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

1. **Prepare environment**: Ensure room is quiet for baseline
2. **Record baseline first**: `./espectre micro collect --label baseline --duration 60 --streamer-ip <device_ip>`
3. **Record movement**: `./espectre micro collect --label movement --duration 60 --streamer-ip <device_ip>`
4. **Verify dataset**: `./espectre micro collect --info`
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

Once you have collected labeled data, train the ML model:

```bash
# Train model (default uses --fp-weight 1.0, --scaler standard, --batch-size 32)
python tools/10_train_ml_model.py

# Show dataset info (including excluded files)
python tools/10_train_ml_model.py --info

# Compare alternate feature normalization modes
python tools/10_train_ml_model.py --scaler clipped_standard

# Optional chip-exclusion experiment
python tools/10_train_ml_model.py --exclude-chip ESP32
```

The `--fp-weight` parameter multiplies the IDLE class weight during training. Values >1.0 reduce false positives at the cost of slightly lower recall. Current defaults: `--fp-weight 1.0`, `--scaler standard`, `--batch-size 32`.

This will:
1. Load all `.npz` files from `data/`
2. Use raw std for all files (CV normalization disabled for ML)
3. Apply context-aware MVS-guided sample weighting on the default subcarrier set
4. Extract 9 features per sliding window
5. Run grouped cross-validation by paired capture/session, with blocked scoring to reduce overlap optimism
6. Report worst-group metrics (session, chip, source file) alongside mean fold metrics
7. Train the selected MLP architecture with early stopping and dropout
8. Export to:
   - `src/python/ml_weights.py` (MicroPython) - includes seed and timestamp
   - `src/cpp/core/ml_weights.h` (C++ shared core) - includes seed and timestamp
   - `models/motion_detector_small.tflite` (TFLite int8)
   - `models/feature_scaler.npz` (normalization params)
   - `models/ml_test_data.npz` (blocked regression subset for inference validation)

Use `--seed <number>` for reproducible training. The seed is saved in the generated weight files.

> **Note**: The ML pipeline always uses raw std for turbulence, regardless of `gain_locked` status. CV normalization is only applied by the MVS detector at runtime.
>
> **Note**: `--exclude-chip` is an experiment knob for ablations and domain-isolation studies. The default training path keeps all supported chips in the dataset unless you explicitly exclude them.
>
> **Note**: `ml_test_data.npz` is an inference-regression artifact, not the primary model-selection metric. Architecture and scaler choices should follow the grouped blocked-CV report emitted by `10_train_ml_model.py`.
>
> **Tip**: `--scaler clipped_standard` and larger `--batch-size` values are available for exploratory sweeps, but should be validated against `test/python/test_validation_real_data.py::TestPerformanceMetrics::test_ml_detection_accuracy` before being promoted to production artifacts.
>
> **Tip**: For production artifact promotion, prefer `python tools/10_train_ml_model.py --seed-search-until-improvement <N>` over a plain training run. A plain run always exports the current seed, while the seed-search flow only replaces artifacts after a strict grouped-CV improvement.

### Compare Detection Methods

After training, compare ML with MVS:

```bash
python tools/7_compare_detection_methods.py
```

Add `--plot` to visualize results graphically.

### Using the ML Detector

Set in `config.py`:

```python
DETECTION_ALGORITHM = "ml"
```

For algorithm details, see [ALGORITHMS.md](ALGORITHMS.md#ml-neural-network-detector).

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

Help build a diverse CSI dataset for the community! Your contributions will improve ML models for everyone.

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

Gestures useful for Home Assistant / smart home automation:

| Priority | Gesture | Description | Home Automation Use |
|----------|---------|-------------|---------------------|
| 🔴 High | `swipe_left` / `swipe_right` | Hand swipe in air | Change scene, adjust brightness |
| 🔴 High | `push` / `pull` | Push away / pull toward | Turn on/off, open/close |
| 🔴 High | `circle_cw` / `circle_ccw` | Circular hand motion | Dimmer, thermostat up/down |
| 🟡 Medium | `clap` | Hand clap | Toggle lights |
| 🟡 Medium | `sit_down` / `stand_up` | Sitting/standing | TV mode, energy saving |
| 🟡 Medium | `fall` | Person falling | Elderly safety alert |
| 🟢 Low | `idle` | Empty room, no movement | Baseline (always needed) |

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

See [References](MICRO_ESPECTRE.md#references) in the Micro-ESPectre guide for the complete bibliography.

## License

GPLv3 - See [LICENSE](../LICENSE) for details.