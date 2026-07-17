# ML Data Collection Guide

This guide covers the current data-collection workflow for ESPectre datasets:
collection, labels, and the dataset format.

Use:

- [`README.md` (streamer)](../src/cpp/frontend/streamer/README.md)
  for streamer firmware setup, UDP protocol, Wi-Fi provisioning, and transport
  tuning
- [`ML_TRAINING.md`](ML_TRAINING.md) for training, export, and validation
- [`ALGORITHMS.md`](ALGORITHMS.md) for detector and feature definitions

For historical rationale behind the current dataset contract, see:

- [`2026-06-30-keep-empty-captures-as-first-class-idle-training-data.md`](adr/2026-06-30-keep-empty-captures-as-first-class-idle-training-data.md)
- [`2026-07-04-keep-agc-active-and-standardize-cv-normalization.md`](adr/2026-07-04-keep-agc-active-and-standardize-cv-normalization.md)
- [`2026-07-17-separate-dataset-admission-from-classic-diagnostics.md`](adr/2026-07-17-separate-dataset-admission-from-classic-diagnostics.md)

Use `tools/validate_dataset_quality.py` before training:

```bash
python tools/validate_dataset_quality.py
python tools/validate_dataset_quality.py --chip C6
python tools/validate_dataset_quality.py --no-report
```

Every run refreshes explicit `static_presence` / `motion` pair metadata in
`data/dataset_info.json` (writes and bumps `updated_at` only on real changes)
and writes `data/auto_generated/DATASET_QUALITY_CHECK.md` unless `--no-report`
is set. Admission checks (integrity, empty/static sanity, ML readiness) can
fail the run. Classic replay adds indicative 0-100 scores for review only, so
the dataset is not filtered to "what Classic already solves". Soft `Breath`
marks use one shared `Breath` ladder (coverage and human-rate Hz weight already
folded into the score), with Empty polarity inverted.
Tooling details live in [`tools/README.md`](../tools/README.md).

## Scope

Current collection priority for v3:

- `empty`
- `static_presence`
- `motion`

Those three labels feed the current production binary ML workflow:

- `empty` and `static_presence` map to `IDLE`
- `motion` maps to `MOTION`

Gesture, HAR, and people-counting datasets are possible, but they are not the
mainline v3 collection target.

## Supported Collection Path

The primary collection path is:

```text
streamer frontend
  -> collector-driven UDP traffic
  -> ./espectre collect
  -> one .npz per device_id
```

See the streamer frontend README for:

- build and flash steps
- local Wi-Fi configuration
- UDP packet format
- transport tuning

This guide assumes that the streamer is already running and reachable.

## Quick Start

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate `.venv\\Scripts\\Activate.ps1` and replace
`./espectre` with `.\espectre.cmd`.

Inspect the live stream first:

```bash
./espectre collect --target 192.168.1.50
```

Then record labeled data:

```bash
./espectre collect --label empty --duration 60 --target 192.168.1.50
./espectre collect --label static_presence --duration 60 --target 192.168.1.50
./espectre collect --label motion --duration 60 --target 192.168.1.50
```

Use `test` only for mixed sessions that are not label-homogeneous.

## `espectre collect`

`./espectre collect` is the host-side entry point for live inspection and
dataset capture in the workflow described above.

For the full command reference, supported modes, options, pacing behavior, and
examples, see [`CLI.md#collect`](CLI.md#collect).

Each saved capture emits one `.npz` per `device_id`. Mixed-device files are not
part of the supported workflow.

### Save Semantics

When saving captures:

- collection starts only after the ready gate is satisfied
- for `classic`, that happens after startup calibration
- `ml` uses its production feature window and does not run startup calibration
- `--detector` chooses the production detector for the ready gate in both live
  and timed collection; timed collection accepts one detector, while live
  inspection can compare `classic,ml`
- `Ctrl+C` before a requested `--duration` finishes aborts the partial live
  capture
- without `--duration`, `Ctrl+C` saves the packets already accepted

## Labels

Current canonical room-state labels:

- `empty`: quiet room, no person present
- `static_presence`: person present but mostly still
- `motion`: ordinary room movement

Use these labels only when the whole capture is homogeneous.

Suggested workflow for one session:

1. collect `empty`
2. collect `static_presence`
3. collect `motion`
4. run `./espectre collect --info`

Recommended starting point:

- 30 to 60 seconds per sample
- at least 10 samples per label
- one environment at a time
- varied positions and distances within the same environment

## Stream Metadata

The clean-break streamer protocol keeps only metadata that is still useful for
analysis and validation:

- `device_ticks_us`
- `wifi_rx_ts_us`, when available
- `wifi_rx_start_ts_ns`, when available
- RF context such as `channel`, `rssi_dbm`, and `noise_floor_dbm`

## Dataset Layout

Directory shape:

```text
data/
├── dataset_info.json
├── empty/
├── static_presence/
└── motion/
```

Typical filename:

```text
{label}_{chip}_{num_sc}sc_{device_token}_{timestamp}_{save_index}.npz
```

All current ESPectre datasets use HT20 CSI with 64 logical subcarriers.

## Metadata

`dataset_info.json` is the dataset-level index. It stores file metadata such as:

- `filename`
- `chip`
- `subcarriers`
- `device_id`
- `contributor`
- `collected_at`
- `duration_ms`
- `num_packets`
- `description`
- `environment`
- `optimal_pair_motion_file` / `optimal_pair_static_presence_file` for
  reciprocal `static_presence` / `motion` pairing

`validate_dataset_quality.py` regenerates those pair fields automatically
before admission and Classic review.

## NPZ Contract

Each `.npz` file stores raw CSI plus capture metadata.

Common fields:

| Field | Type | Meaning |
|-------|------|---------|
| `csi_data` | `int8[N, SC*2]` | Raw I/Q data |
| `num_subcarriers` | `int` | Logical subcarrier count, currently `64` |
| `label` | `str` | Dataset label |
| `chip` | `str` | Chip identifier |
| `collected_at` | `str` | ISO timestamp |
| `duration_ms` | `float` | Capture duration |
| `format_version` | `str` | Dataset format version |
| `stream_seq_num` | `uint32[N]` | Stream sequence numbers |
| `device_ticks_us` | `uint64[N]` | Device monotonic timestamps |
| `device_id` | `uint64` | Stable device identifier |
| `wifi_rx_ts_us` | `uint32[N]` | Optional Wi-Fi RX timestamps |
| `wifi_rx_start_ts_ns` | `uint64[N]` | Optional RX-start estimate |
| `channel` | `uint8[N]` | Optional per-packet Wi-Fi channel |
| `rssi_dbm` | `int16[N]` | Optional RSSI metadata |
| `noise_floor_dbm` | `int16[N]` | Optional noise-floor metadata |

CSI uses the Espressif ordering `[Q0, I0, Q1, I1, ...]`.

Amplitude extraction:

```python
Q = csi_data[:, 0::2].astype(float)
I = csi_data[:, 1::2].astype(float)
amplitudes = np.sqrt(I**2 + Q**2)
phases = np.arctan2(Q, I)
```

## Loading Data

Minimal example:

```python
import numpy as np

data = np.load("data/static_presence/sample.npz")
csi_data = data["csi_data"]
label = str(data["label"])
```

Using the tool library:

```python
from pathlib import Path
from tools.lib.csi_io import load_npz_as_packets

packets = load_npz_as_packets(Path("data/static_presence/sample.npz"))
```

## Collection Notes

- AGC stays active during collection
- the fixed production subcarrier set is applied later in runtime and offline
  tooling; raw captures keep the full HT20 packet layout
- the current ML runtime and training flow use the Core-6 feature set defined in
  [`ALGORITHMS.md`](ALGORITHMS.md)

## Dataset Inspection

Use:

```bash
./espectre collect --info
python tools/validate_dataset_quality.py
python tools/train_ml_model.py --info
```

`collect --info` summarizes collected files.
`validate_dataset_quality.py` refreshes pair metadata, runs admission plus
Classic review, and updates `data/auto_generated/DATASET_QUALITY_CHECK.md`.
`train_ml_model.py --info` shows the dataset view used by the trainer.

## Contributing Data

The most useful contributions for the current project direction are:

- `empty` captures that reduce false positives
- `static_presence` captures that improve idle robustness
- `motion` captures across chips, routers, and room layouts

Before opening a PR:

1. collect at least 10 samples per label when possible
2. keep labels homogeneous
3. note chip, room type, and unusual environmental conditions
4. verify the dataset with `./espectre collect --info`
5. run `python tools/validate_dataset_quality.py` and resolve admission FAILs

## Next Steps

- [`ML_TRAINING.md`](ML_TRAINING.md) for model training, export, and regression
  checks
- [`README.md` (streamer)](../src/cpp/frontend/streamer/README.md)
  for firmware-side streaming details
- [`README.md` (tools)](../tools/README.md) for analysis helpers
