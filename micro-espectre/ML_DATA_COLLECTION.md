# ML Data Collection Guide

**Building labeled CSI datasets for machine learning**

This guide covers how to collect and label CSI data for training ML models. This infrastructure lays the groundwork for advanced Wi-Fi sensing features (gesture recognition, HAR, people counting) planned for ESPectre 3.x.

## Roadmap (3.x)

| Feature | Status |
|---------|--------|
| Data collection infrastructure | ✅ Ready (2.2.0) |
| Feature extraction pipeline | 🔜 Planned |
| Model training scripts (RF, CNN, LSTM) | 🔜 Planned |
| Gesture recognition | 🔜 Planned |
| Human Activity Recognition (HAR) | 🔜 Planned |
| People counting | 🔜 Planned |
| Real-time inference | 🔜 Planned |
| Cloud service (API) | 🔜 Planned |

---

## Getting Started

### 1. Activate Virtual Environment

Before running any command, activate the virtual environment:

```bash
cd micro-espectre
source ../venv/bin/activate  # Your prompt should show (venv)
```

### 2. Flash and Deploy (First Time Only)

If you haven't already flashed the firmware:

```bash
./me flash --erase
./me deploy
```

### 3. Start CSI Streaming

Start streaming CSI data from ESP32 to your PC:

```bash
./me stream --ip <your_pc_ip>
```

**Features:**
- Gain lock phase (~3s) for stable CSI acquisition
- 64 subcarriers (HT20 mode)
- Sequence numbers for packet loss detection
- ~100 packets/second

---

## Data Collection with `me collect`

The `me collect` subcommand provides a streamlined workflow for recording labeled CSI samples.

### Commands

| Command | Description |
|---------|-------------|
| `./me collect --label <name> --duration <sec>` | Record for specified duration |
| `./me collect --label <name> --samples <n>` | Record n samples interactively |
| `./me collect --label <name> --contributor <user>` | Override contributor (auto-detected from git) |
| `./me collect --info` | Show dataset statistics |

### Recording Samples

```bash
# Record 60 seconds of baseline (contributor auto-detected from git config)
./me collect --label baseline --duration 60

# Record 30 seconds of movement
./me collect --label movement --duration 30

# Record with explicit contributor override
./me collect --label gesture --samples 10 --interactive --contributor otheruser
```

### Viewing Dataset

```bash
./me collect --info
```

Output:
```
Dataset: 5 labels, 47 samples
  idle: 12 samples (36000 packets)
  wave: 10 samples (15000 packets)
  swipe: 10 samples (15000 packets)
  ...
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
└── baseline_noisy/
    └── ...
```

**Note**: HT20 only - all datasets use 64 subcarriers.

File naming convention: `{label}_{chip}_{num_sc}sc_{timestamp}.npz`

### Dataset Info (dataset_info.json)

Central metadata file for the dataset:

```json
{
  "format_version": "1.0",
  "labels": {
    "baseline": { "id": 0, "description": "Quiet room, no motion" },
    "movement": { "id": 1, "description": "Human movement in room" }
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
      }
    ]
  },
  "environments": [...]
}
```

| Field | Description |
|-------|-------------|
| `filename` | NPZ file name |
| `chip` | ESP32 chip type (C6, S3) |
| `subcarriers` | Number of subcarriers (64 for HT20) |
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
| `label` | `str` | Sample label (e.g., "baseline", "movement") |
| `label_id` | `int` | Numeric label ID for ML |
| `chip` | `str` | ESP32 chip type (e.g., "c6", "s3") |
| `collected_at` | `str` | ISO timestamp of collection |
| `duration_ms` | `float` | Sample duration in milliseconds |
| `format_version` | `str` | NPZ format version ("1.0") |

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
from csi_utils import load_npz_as_packets
from pathlib import Path

# Load a sample file
data_dir = Path('data')
packets = load_npz_as_packets(data_dir / 'baseline' / 'baseline_c6_64sc_20251212_142443.npz')

for pkt in packets:
    amplitudes = pkt['amplitudes']  # Shape: (64,) - computed on load
    label = pkt['label']
    # Process...
```

---

## Best Practices

### Recording Guidelines

| Aspect | Recommendation |
|--------|----------------|
| **Duration** | 30-60 seconds per sample (1500-3000 packets @ 50 pps) |
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
2. **Record baseline first**: `./me collect --label baseline --duration 60`
3. **Record movement**: `./me collect --label movement --duration 60`
4. **Verify dataset**: `./me collect --info`
5. **Backup data**: Copy `data/` to safe location

Note: Contributor is auto-detected from `git config user.name`. Use `--contributor` to override.

---

## Analysis Tools

After collecting data, use the analysis scripts in `tools/`:

```bash
cd tools

# Visualize raw CSI data
python 1_analyze_raw_data.py

# Test MVS detection on your data
python 3_analyze_moving_variance_segmentation.py --plot
```

See [tools/README.md](tools/README.md) for complete documentation of all analysis scripts.

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

```
Header (6 bytes):
  - Magic: 0x4353 ("CS") - 2 bytes
  - Chip type: 1 byte (0=unknown, 1=ESP32, 2=S2, 3=S3, 4=C3, 5=C5, 6=C6)
  - Sequence number: 1 byte (0-255, wrapping)
  - Num subcarriers: 2 bytes (uint16, little-endian)

Payload (N × 2 bytes):
  - I0, Q0, I1, Q1, ... (int8 each)

Example (HT20, 64 SC):
  - 6 + 128 = 134 bytes
```

Note: ESPectre uses HT20 mode (64 subcarriers) for consistent performance across all ESP32 variants. The chip type is automatically detected and included in each packet.

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

See [References](README.md#references) in the main README for complete bibliography.

## License

GPLv3 - See [LICENSE](../LICENSE) for details.