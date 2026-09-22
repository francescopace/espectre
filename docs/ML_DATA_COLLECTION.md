# ML data collection guide

This guide explains how to record CSI datasets for the High Accuracy model: `empty`, `static_presence`, and `motion` recordings. If you are new, read [Quick start](#quick-start), [Labels](#labels), and [Contributing data](#contributing-data). The rest is reference for dataset curators.

Terms:

- **Label:** what was happening in the room during the recording.
- **Pair:** a `static_presence` and a `motion` recording made in comparable conditions.
- **Dataset role:** how a recording may be used: for training, for model selection, for the final test, or not at all.
- **NPZ:** the compressed NumPy file written for each device and recording.

Related: the [setup guide](SETUP.md) to install a device, the [ML training guide](ML_TRAINING.md) to train and validate, and the [algorithms reference](ALGORITHMS.md) for how the detector works.

## What to collect

For v3, collect only three labels. The model maps them to two states:

| Label | Room state | Model state |
|-------|------------|-------------|
| `empty` | Nobody in the room, quiet | `IDLE` |
| `static_presence` | Someone present but mostly still | `IDLE` |
| `motion` | Normal movement | `MOTION` |

## How collection works

```text
ESPectre device
  <- UDP markers from the collector
  -> raw CSI over GET /espectre/v1/csi
  -> ./espectre collect
  -> one .npz per device
```

`./espectre collect` switches the device to external traffic, sends the traffic itself, and saves what the device streams back. You need a running ESPectre device (ESPHome, Native, or Matter) with Direct enabled, reachable from your computer. All flags are in the [`collect` command](CLI.md#collect) reference.

## Quick start

Set up the Python environment once, from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1` and use `.\espectre.cmd` instead of `./espectre`.

Watch the live stream first to check that everything works:

```bash
./espectre collect --target 192.168.1.50
```

Then record:

```bash
./espectre collect --label empty --duration 60 --target 192.168.1.50
./espectre collect --label static_presence --duration 60 --target 192.168.1.50
./espectre collect --label motion --duration 60 --target 192.168.1.50
```

Recording starts only when the detector is ready: after calibration for `lightweight` (the default), or once the feature window is full for `high_accuracy`. Press `Ctrl+C` to stop: with `--duration`, this discards the recording; without it, the packets received so far are saved.

The device stays in external mode afterwards.

## Labels

- `empty`: nobody in the room, and nothing moving.
- `static_presence`: someone in the room, mostly still.
- `motion`: normal movement in the room.

Use a label only if it is true for the whole recording; mixed recordings are not supported. Long quiet recordings used to test false alarms are also `empty`, marked with `long_recording: true` in the catalog. They are never used for training.

A good session:

1. Record `empty`, then `static_presence`, then `motion`.
2. Aim for 30–60 seconds per recording and at least 10 recordings per label.
3. Stay in one room per session, and vary your position and distance.
4. [Add the catalog fields](#add-the-catalog-fields) for each new file.
5. Run `./espectre collect --info` and `python tools/validate_dataset_quality.py`.

## Add the catalog fields

Each recording has an entry in `data/dataset_info.json`. The collector fills in the technical details but cannot know the room or how the recording should be used, so add two fields by hand:

```json
{
  "filename": "static_presence_c6_64sc_dev...npz",
  "environment": "bedroom",
  "dataset_role": "exclude"
}
```

- `environment`: a room name. Use the same name for all recordings from the same room.
- `dataset_role`: start with `exclude` while you review the recording. Changing it to `train`, `selection`, or `holdout` is a deliberate decision about the corpus.

Do not add pair fields yourself: the validator creates them.

## Check the dataset

```bash
./espectre collect --info
python tools/validate_dataset_quality.py
python tools/train_ml_model.py --info
```

- `collect --info` lists the recordings, one table per room.
- `validate_dataset_quality.py` checks every file, creates the pair fields, and updates `data/auto_generated/DATASET_QUALITY_CHECK.md`. A failure (FAIL) blocks training; the feature-space scores are only informative. It never sets dataset roles.
- `train_ml_model.py --info` shows which recordings the trainer will use.

The validator warns when a file has less than 85% occupancy and fails below 70%. A file without usable timing information fails; it is never assumed to be 100 pps. See the [tools guide](../tools/README.md#dataset-inspection-and-validation) for options.

## Contributing data

The most useful recordings right now:

- `empty`, to reduce false alarms
- `static_presence`, to make idle detection more robust
- `motion`, from different chips, routers, and room layouts

Before opening a pull request:

1. Record at least 10 files per label, if you can.
2. Make sure each file has a single label.
3. Add `environment` and `dataset_role` to every catalog entry.
4. Describe the room and anything unusual in `description`.
5. Run `./espectre collect --info`.
6. Run `python tools/validate_dataset_quality.py` and fix every FAIL.

### Data privacy

CSI contains no images or audio, but it is not anonymous. Device IDs, timestamps, names, room labels, radio metadata, and detected activity can identify people or reveal private information.

- Record only where you have the right to, tell the people affected, and follow privacy laws.
- Before the pull request, check the `.npz` metadata and `data/dataset_info.json`. Remove unnecessary details, and use a pseudonym as contributor if you prefer.
- Never submit Wi-Fi passwords, SSIDs, BSSIDs, local IP addresses, serial logs, or unrelated personal data.

You keep ownership of your data and are credited in the dataset documentation. See [DCO and CLA](../CONTRIBUTING.md#dco-and-cla) for the contribution terms.

## Reference

### Dataset layout

```text
data/
├── dataset_info.json
├── empty/
├── static_presence/
└── motion/
```

File names follow `{label}_{chip}_{num_sc}sc_{device_token}_{timestamp}_{save_index}.npz`. Each file holds one device; mixed-device files are not supported.

### Catalog fields

`data/dataset_info.json` stores, for each file:

| Field | Meaning |
|-------|---------|
| `filename`, `chip`, `subcarriers`, `device_id` | Which file and device |
| `contributor`, `collected_at`, `description` | Who, when, and notes |
| `duration_ms`, `num_packets` | Length of the recording |
| `environment` | Room name (set by hand) |
| `dataset_role` | `train`, `selection`, `holdout`, or `exclude` (set by hand). `selection` files are used to choose between candidate models; `holdout` files stay sealed until the final check of the chosen model. The trainer treats a missing role as `exclude`, but the validator fails until every entry has one |
| `optimal_pair_motion_file`, `optimal_pair_static_presence_file` | The matching recording of the pair (set by the validator) |
| `low_rssi: true` | Weak-signal recording. Allows up to 5% missing records instead of 3% (warning above 1% for both) |
| `synthetic: true` | Generated, not a real measurement |
| `long_recording: true` | Long quiet `empty` recording, used only to test false alarms. With role `exclude`, it is kept only for reference and the quality report |

The validator writes the pair fields itself and never pairs a real recording with a synthetic one.

Older synthetic weak-link files live in the normal label folders, marked `low_rssi: true` and `synthetic: true`. The generator that made them is no longer shipped, and model promotion now relies on real recordings.

### NPZ contents

Each `.npz` holds the raw CSI and the capture metadata:

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
| `raw_stream_sequence` | `uint64[N]` | Canonical raw HTTP sequence numbers, including observable gaps |
| `device_ticks_us` | `uint64[N]` | Device monotonic timestamps |
| `phy_mode` | `str[N]` | Per-record PHY mode; current sensing rows use `ht` |
| `ltf_type` | `str[N]` | Per-record LTF type; current sensing rows use `ht-ltf` |
| `channel_width` | `str[N]` | Per-record channel width; current sensing rows use `20` |
| `device_id` | `uint64` | Stable pseudonymous device identifier |
| `transport` | `str` | Live transport, currently `http` for new captures |
| `endpoint`, `transport_target` | `str` | Direct raw endpoint used for collection |
| `requested_pps` | `float` | Requested external generator rate per target |
| `observed_pps`, `effective_pps` | `float` | Observed collector receive rate |
| `raw_protocol_version` | `uint8` | Raw HTTP protocol version, currently `1` |
| `record_version` | `uint8` | CSI record version, currently `8` for live captures |
| `frontend` | `str` | Device frontend (`native`, `esphome`, or `matter`) |
| `firmware_version`, `firmware_identity` | `str` | Firmware provenance reported by the Direct `device` resource |
| `fresh_record_total`, `raw_fresh_record_total` | `uint64` | Final sent-record counter |
| `raw_drop_total` | `uint64` | Final count of raw records not transmitted |
| `send_backpressure_total`, `raw_send_backpressure_total` | `uint64` | Final failed-send backpressure counter |
| `raw_final_stream_sequence` | `uint64` | Final offered-frame sequence used with the final counters to validate the raw-loss invariant |
| `csi_target_pps` | `uint64` | Nominal temporal-admission rate recorded for replay |
| `detector_admitted_packets` | `uint64` | Records accepted by the production temporal sampler during capture review |
| `temporal_missing_slots`, `temporal_excess_packets` | `uint64` | Missing nominal slots and records above the configured slot cadence |
| `temporal_stale_packets`, `temporal_out_of_order_packets` | `uint64` | Records rejected for stale or reversed timing |
| `temporal_occupancy_slots`, `temporal_window_slots` | `uint64` | Occupied and available slots used to calculate mean temporal occupancy |
| `wifi_rx_ts_us` | `uint32[N]` | Optional Wi-Fi RX timestamps |
| `wifi_rx_start_ts_ns` | `uint64[N]` | Optional RX-start estimate |
| `channel` | `uint8[N]` | Optional per-packet Wi-Fi channel |
| `rssi_dbm` | `int16[N]` | Optional RSSI metadata |
| `noise_floor_dbm` | `int16[N]` | Optional noise-floor metadata |

Legacy generated NPZ files may additionally store `synthetic`, `source_dataset`, `low_rssi_profile`, `generation_mode`, `generation_seed`, `generation_group`, `generated_at`, and `generator_version`. They may also embed historical Core-6 feature names, source, target, and achieved medians, normalized fit errors, and fitted impairment parameters. These fields keep old generated files self-describing for ML analysis; the runtime packet loader ignores them.

CSI values are stored in Espressif order, `[Q0, I0, Q1, I1, ...]`:

```python
Q = csi_data[:, 0::2].astype(float)
I = csi_data[:, 1::2].astype(float)
amplitudes = np.sqrt(I**2 + Q**2)
phases = np.arctan2(Q, I)
```

### Loading data

Plain NumPy returns the raw arrays, including rows the detector would skip:

```python
import numpy as np

data = np.load("data/static_presence/sample.npz")
csi_data = data["csi_data"]
label = str(data["label"])
```

The tool library returns the same view the detector uses (HT20, HT-LTF, 64 subcarriers):

```python
from pathlib import Path
from tools.lib.csi_io import load_npz_as_packets

packets = load_npz_as_packets(Path("data/static_presence/sample.npz"))
```

- Old recordings without any PHY fields are accepted only if their data already has the 64-subcarrier HT20 layout.
- A file with only some PHY fields is rejected as suspect.
- Pass `keep_all_phy=True` to see every row, including other capture profiles.

The dataset validator and the C++ test loader use the same view, so skipped rows show up as gaps in the continuity checks.

### Collection notes

- AGC stays on during collection.
- `--pps` sets the rate of the collector's own traffic; the device does not pace or skip records.
- The traffic marker is the four bytes `F0 9F 91 BB` (`"👻".encode("utf-8")`), sent to the device's advertised UDP port.
- Training uses only HT20 + HT-LTF + 64 subcarriers. The device may capture `lltf20` (classic ESP32, ESP32-S2) or `vht20` (5 GHz); those rows keep their PHY fields and are left out of the default view.
- Raw records also carry timing and radio data: `device_ticks_us`, `wifi_rx_ts_us` and `wifi_rx_start_ts_ns` when available, `channel`, `rssi_dbm`, and `noise_floor_dbm`.
- The dataset rules come from earlier decisions recorded in the [ADR index](adr/README.md).

## Next steps

- [ML training guide](ML_TRAINING.md): train, export, and check the model
- [CSI collection](API.md#csi-collection): the raw stream format
- [Tools guide](../tools/README.md): analysis helpers
