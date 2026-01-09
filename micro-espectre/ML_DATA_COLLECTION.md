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
- Gain lock phase (~3s) to determine dominant subcarrier count (64, 128, or 256)
- Protocol v2 supports full 256 subcarriers (512 bytes payload)
- Sequence numbers for packet loss detection
- Packet filtering to ensure consistent SC count
- ~100 packets/second

---

## Data Collection with `me collect`

The `me collect` subcommand provides a streamlined workflow for recording labeled CSI samples.

### Commands

| Command | Description |
|---------|-------------|
| `./me collect --label <name> --duration <sec>` | Record for specified duration |
| `./me collect --label <name> --samples <n>` | Record n samples interactively |
| `./me collect --info` | Show dataset statistics |

### Recording Samples

```bash
# Record 60 seconds of idle (baseline)
./me collect --label idle --duration 60

# Record 30 seconds of wave gesture
./me collect --label wave --duration 30

# Record 10 samples interactively (press Enter between each)
./me collect --label swipe --samples 10 --interactive
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
├── idle/
│   ├── idle_001.npz
│   ├── idle_002.npz
│   └── ...
├── wave/
│   ├── wave_001.npz
│   └── ...
└── movement/
    └── ...
```

### Sample Format (.npz)

Each `.npz` file contains:

| Field | Type | Description |
|-------|------|-------------|
| `amplitudes` | `float32[N, SC]` | Amplitude per subcarrier (SC = 64, 128, or 256) |
| `phases` | `float32[N, SC]` | Phase per subcarrier (radians) |
| `iq_raw` | `int8[N, SC*2]` | Raw I/Q data (N packets × SC subcarriers × 2) |
| `timestamps` | `float64[N]` | Packet timestamps (seconds, relative to first packet) |
| `label` | `str` | Sample label |
| `label_id` | `int` | Numeric label ID |
| `num_packets` | `int` | Number of packets in sample |
| `num_subcarriers` | `int` | Number of subcarriers (64, 128, or 256) |
| `duration_ms` | `float` | Sample duration in milliseconds |
| `sample_rate_hz` | `float` | Packet rate in Hz |
| `dropped_packets` | `int` | Number of dropped packets detected |
| `collected_at` | `str` | ISO timestamp of collection |
| `subject` | `str` | Subject/contributor ID (optional) |
| `environment` | `str` | Environment description (optional) |
| `notes` | `str` | Additional notes (optional) |
| `format_version` | `int` | NPZ format version (1) |

### Loading Data

```python
import numpy as np

# Load single sample
data = np.load('data/wave/wave_001.npz')
amplitudes = data['amplitudes']  # Shape: (N, 64)
phases = data['phases']          # Shape: (N, 64)
iq_raw = data['iq_raw']          # Shape: (N, 128)
label = str(data['label'])       # 'wave'

# Amplitudes are pre-calculated, but you can also compute from iq_raw:
I = iq_raw[:, 0::2].astype(float)  # Shape: (N, 64)
Q = iq_raw[:, 1::2].astype(float)  # Shape: (N, 64)
amplitudes_manual = np.sqrt(I**2 + Q**2)  # Shape: (N, 64)
```

### Using csi_utils

```python
from csi_utils import load_samples, load_dataset

# Load all samples as list of dicts
samples = load_samples()  # All labels
samples = load_samples(labels=['wave', 'idle'])  # Specific labels

for sample in samples:
    amplitudes = sample['amplitudes']  # Shape: (N, 64)
    label = sample['label']
    # Process...

# Load as ML-ready arrays
X, y = load_dataset(labels=['wave', 'idle'])
# X: list of amplitude arrays
# y: array of label IDs
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
2. **Record baseline first**: `./me collect --label idle --duration 60`
3. **Record gestures**: One gesture type at a time
4. **Verify dataset**: `./me collect --info`
5. **Backup data**: Copy `data/` to safe location

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
    # - num_subcarriers: Number of subcarriers (64)
    # - iq_raw: Raw I/Q values as int8 array
    # - iq_complex: Complex representation
    # - amplitudes: Amplitude per subcarrier
    # - phases: Phase per subcarrier (radians)
    print(f"Seq: {packet.seq_num}, Subcarriers: {packet.num_subcarriers}")

receiver = CSIReceiver(port=5001)
receiver.add_callback(my_callback)
receiver.run(timeout=60)  # Run for 60 seconds
```

### UDP Packet Format

The streamer uses protocol v2 which supports up to 65535 subcarriers:

```
Header v2 (6 bytes):
  - Magic: 0x4353 ("CS") - 2 bytes
  - Version: 0x02 - 1 byte
  - Sequence number: 1 byte (0-255, wrapping)
  - Num subcarriers: 2 bytes (uint16, little-endian)

Payload (N × 2 bytes):
  - I0, Q0, I1, Q1, ... (int8 each)

Examples:
  - 64 SC:  6 + 128 = 134 bytes
  - 128 SC: 6 + 256 = 262 bytes
  - 256 SC: 6 + 512 = 518 bytes
```

The receiver (`csi_utils.py`) also supports legacy v1 format for backward compatibility:

```
Header v1 (4 bytes, legacy):
  - Magic: 0x4353 ("CS") - 2 bytes
  - Num subcarriers: 1 byte (max 255)
  - Sequence number: 1 byte (0-255, wrapping)
```

Note: The streamer performs a gain lock phase (~3 seconds) at startup to determine the dominant subcarrier count (64, 128, or 256). Packets with different SC counts are filtered out during streaming for consistency.

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