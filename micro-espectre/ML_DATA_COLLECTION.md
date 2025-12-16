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
- Auto-calibration NBVI at boot (if subcarriers not configured)
- Raw I/Q data for 12 selected subcarriers
- Sequence numbers for packet loss detection
- ~100 packets/second @ 28 bytes/packet

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
| `csi_data` | `int8[N, 24]` | Raw I/Q data (N packets × 12 subcarriers × 2) |
| `timestamps` | `uint32[N]` | Packet timestamps (ms) |
| `rssi` | `int8[N]` | RSSI values |
| `noise_floor` | `int8[N]` | Noise floor values |
| `label` | `str` | Sample label |

### Loading Data

```python
import numpy as np

# Load single sample
data = np.load('data/wave/wave_001.npz')
csi = data['csi_data']      # Shape: (N, 24)
label = str(data['label'])  # 'wave'

# Extract amplitudes
amplitudes = []
for i in range(12):
    I = csi[:, i*2].astype(float)
    Q = csi[:, i*2+1].astype(float)
    amplitudes.append(np.sqrt(I**2 + Q**2))
amplitudes = np.array(amplitudes).T  # Shape: (N, 12)
```

### Using csi_utils

```python
from csi_utils import load_npz_as_packets

# Load as list of packet dicts (compatible with analysis scripts)
packets = load_npz_as_packets('data/wave/wave_001.npz')

for pkt in packets:
    csi_data = pkt['csi_data']
    rssi = pkt['rssi']
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
2. **Record baseline first**: `./me collect start idle 60`
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
    # packet is a dict with:
    # - timestamp_ms: Reception timestamp
    # - seq_num: Sequence number (0-255)
    # - csi_data: Raw I/Q as numpy array
    # - rssi, noise_floor: Signal quality
    print(f"Seq: {packet['seq_num']}, RSSI: {packet['rssi']}")

receiver = CSIReceiver(port=5001)
receiver.add_callback(my_callback)
receiver.run(timeout=60)  # Run for 60 seconds
```

### UDP Packet Format

```
Header (4 bytes):
  - Magic: 0x4353 ("CS") - 2 bytes
  - Num subcarriers: 1 byte
  - Sequence number: 1 byte (0-255, wrapping)

Payload (N × 2 bytes = 24 bytes for 12 SC):
  - I0, Q0, I1, Q1, ... (int8 each)

Total: 28 bytes per packet
```

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