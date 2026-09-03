# ML Data Collection Guide

This guide is for dataset contributors collecting `empty`, `static_presence`, and `motion` recordings. New contributors can follow [Quick Start](#quick-start) and [`espectre collect`](#espectre-collect); metadata and validation sections are reference material for curators.

A **label** is the observed room state, a **pair** links comparable static-presence and motion recordings, a **dataset role** controls how a recording may be used during model selection, and an **NPZ** file is the compressed NumPy container written for one device capture.

Use:

- [`SETUP.md`](SETUP.md) for ESPectre firmware setup and external traffic generation
- [`ML_TRAINING.md`](ML_TRAINING.md) for training, export, and validation
- [`ALGORITHMS.md`](ALGORITHMS.md) for detector and feature definitions

Historical rationale behind the dataset contract remains in the [ADR index](adr/README.md); this guide describes the current collection workflow.

## Scope

Current collection priority for v3:

- `empty`
- `static_presence`
- `motion`

Those three labels feed the current production binary ML workflow:

- `empty` and `static_presence` map to `IDLE`
- `motion` maps to `MOTION`

Gesture, HAR, and people-counting datasets are possible, but they are not the mainline v3 collection target.

## Supported Collection Path

The primary collection path is:

```text
raw-capable ESPectre frontend
  -> ExternalTrafficGenerator UDP marker
  -> bearer-bound raw HTTP
  -> ./espectre collect
  -> one .npz per device_id
```

The collector resolves the Direct endpoint, persistently selects `csi_traffic_mode=external`, verifies the device configuration, opens the raw HTTP stream, and imports the same standard-library-only external generator used by `tools/espectre_traffic_generator.py`. This guide assumes that a raw-capable ESPectre device is already running and reachable.

## Quick Start

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate `.venv\\Scripts\\Activate.ps1` and replace `./espectre` with `.\espectre.cmd`.

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

Before validation, curate the new entries in `data/dataset_info.json`. The collector records capture and transport provenance, but it cannot infer the room name or decide how a recording may be used. Add an explicit `environment` and `dataset_role` to every new entry; use `exclude` while reviewing a capture, and assign `train`, `selection`, or `holdout` only as a deliberate corpus decision. One entry should look like this:

```json
{
  "filename": "static_presence_c6_64sc_dev...npz",
  "environment": "bedroom",
  "dataset_role": "exclude"
}
```

Use the same environment name for comparable recordings. Do not add pair fields by hand: the validator derives reciprocal static-presence and motion pairs after the required manual metadata is present.

## `espectre collect`

`./espectre collect` is the host-side entry point for live inspection and dataset capture in the workflow described above.

For the full command reference, supported targets, external generator options, and examples, see [`CLI.md#collect`](CLI.md#collect).

Each saved capture emits one `.npz` per `device_id`. Mixed-device files are not part of the supported workflow.

### Save Semantics

When saving captures:

- collection starts only after the ready gate is satisfied
- for `lightweight`, that happens after startup calibration
- `high_accuracy` uses its production feature window and does not run startup calibration
- `--detector` chooses the production detection profile for the ready gate in both live and timed collection; timed collection accepts one profile, while live inspection can compare `lightweight,high_accuracy`
- `Ctrl+C` before a requested `--duration` finishes aborts the partial live capture
- without `--duration`, `Ctrl+C` saves the packets already accepted

## Labels

Current canonical room-state labels:

- `empty`: quiet room, no person present
- `static_presence`: person present but mostly still
- `motion`: ordinary room movement

Use these labels only when the whole capture is homogeneous.

Quiet long-run replays also live under `empty`. Mark them in `dataset_info.json` with `long_recording: true` so validation and long-recording suites can find them. The trainer excludes these recordings from fitting; `selection` and `holdout` long recordings still participate in the quiet replay gate.

Mixed sessions are not part of the current v3 mainline dataset contract.

Suggested workflow for one session:

1. collect `empty`
2. collect `static_presence`
3. collect `motion`
4. add `environment` and an explicit `dataset_role` to each new `dataset_info.json` entry
5. run `./espectre collect --info`
6. run `python tools/validate_dataset_quality.py`

Recommended starting point:

- 30 to 60 seconds per sample
- at least 10 samples per label
- one environment at a time
- varied positions and distances within the same environment

## Raw Record Metadata

Raw HTTP carries CSI V8 records with metadata useful for analysis and validation:

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

All current ESPectre datasets use HT20 CSI with 64 logical subcarriers. Training and validation loaders therefore label captures without per-record PHY metadata as `ht20`; new raw HTTP captures preserve explicit PHY and LTF metadata.

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
- `optimal_pair_motion_file` / `optimal_pair_static_presence_file` for reciprocal `static_presence` / `motion` pairing
- `low_rssi: true` for real and synthetic weak-link datasets stored under their semantic labels. Stream-continuity admission warns above 1% missing sequence records, fails above 3% for normal recordings, and fails above 5% for `low_rssi` recordings; maximum sequence-gap and inter-packet-gap gates remain unchanged
- `synthetic: true` for generated captures that are not real measurements
- `long_recording: true` for quiet long-run `empty` captures reserved for the long-recording replay suites; these stay evaluation-only and do not enter ML training or the standard empty-room admission table. A long recording with `dataset_role: exclude` remains in the catalog for provenance and quality-report diagnostics only
- `dataset_role: train | selection | holdout | exclude` to control how a recording participates in fitting and deployment replay. The trainer treats a missing role as `exclude` for safety, but dataset validation still fails until every entry declares its role explicitly. `selection` recordings gate candidate selection, `holdout` recordings stay sealed until the trainer evaluates the final winner once, and `exclude` keeps a dataset in the catalog while removing it from the current train/selection/holdout workflow

`validate_dataset_quality.py` regenerates those pair fields automatically before admission and shared feature-space review. It never pairs a real capture with a synthetic capture; generated pair identity is read from the NPZ metadata.

Legacy synthetic low-RSSI derivatives use the standard `data/<label>/` directories. Their `low_rssi: true` and `synthetic: true` catalog markers describe the link condition and generated origin without changing the `empty`, `static_presence`, or `motion` meaning. The repository no longer ships the synthetic generator, and current model promotion relies on real captures.

Existing generated NPZs can retain detailed generation provenance, fitted parameters, and historical Core-6 diagnostics. These fields are a backward-compatible legacy contract for self-describing analysis inputs, not a description of the current production feature set.

## NPZ Contract

Each `.npz` file stores raw CSI plus capture metadata.

Current collector fields:

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
| `firmware_version`, `firmware_identity` | `str` | Firmware provenance reported by Direct `info` |
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

CSI uses the Espressif ordering `[Q0, I0, Q1, I1, ...]`.

Amplitude extraction:

```python
Q = csi_data[:, 0::2].astype(float)
I = csi_data[:, 1::2].astype(float)
amplitudes = np.sqrt(I**2 + Q**2)
phases = np.arctan2(Q, I)
```

## Loading Data

Minimal example (raw on-disk arrays, including any non-HT20 rows):

```python
import numpy as np

data = np.load("data/static_presence/sample.npz")
csi_data = data["csi_data"]
label = str(data["label"])
```

Using the tool library (HT20 sensing view by default):

```python
from pathlib import Path
from tools.lib.csi_io import load_npz_as_packets

packets = load_npz_as_packets(Path("data/static_presence/sample.npz"))
```

`load_npz_as_packets` and `load_npz_csi_data` expose the production sensing view by default: `phy_mode=ht`, `ltf_type=ht-ltf`, `channel_width=20`, and the stored 64-subcarrier HT20 layout. Historical captures that omit all per-record PHY metadata are only accepted when the on-disk payload already matches that same 64-subcarrier contract. Partially missing PHY metadata (some arrays present, others absent) is rejected rather than defaulted: a capture recorded after PHY provenance was introduced should carry every field, so a missing one marks the file as suspect. There is no fallback to `legacy` rows. Pass `keep_all_phy=True` to inspect mixed-PHY or unsupported captures explicitly. Dataset quality validation and the C++ test NPZ loader use the same filtered view, so excessive non-sensing drops show up as stream continuity gaps.

## Collection Notes

- AGC stays active during collection
- `--pps` controls the external UDP generator and nominal dataset rate; HTTP does not pace or decimate records
- the collector intentionally leaves the device in `external` mode after it stops
- the external traffic marker is the exact four-byte UTF-8 payload `"👻".encode("utf-8")` (`F0 9F 91 BB`) on the capability-advertised UDP endpoint
- the fixed training and validation view is HT20 + HT-LTF + 64 subcarriers; the runtime may select `lltf20` on the original ESP32 and ESP32-S2 or `vht20` on a VHT-capable 5 GHz association, but those raw rows retain their PHY metadata and remain outside this default dataset view
- the current ML runtime and training flow use the eight scale-invariant production features defined in [FEATURES.md](FEATURES.md)

## Dataset Inspection

Use:

```bash
./espectre collect --info
python tools/validate_dataset_quality.py
python tools/train_ml_model.py --info
```

`collect --info` summarizes collected files but does not assign environments or dataset roles. `validate_dataset_quality.py` requires those manual fields, refreshes pair metadata, runs admission plus quality review, and updates `data/auto_generated/DATASET_QUALITY_CHECK.md`. Mean valid-slot occupancy warns below 85%, fails admission below 70%, and caps every affected review score; temporal quality and ML-readiness checks require a usable recorded packet rate, or `num_packets` plus `duration_ms`. Insufficient timing metadata is a validation failure and is never interpreted as 100 pps. `train_ml_model.py --info` shows the dataset view used by the trainer.

Run the validator after curating new entries and before training. Admission failures block the workflow; feature-space scores are diagnostic only. Dataset roles remain manual, and the validator never assigns `train`, `selection`, or `holdout`. See [`tools/README.md`](../tools/README.md#dataset-inspection-and-validation) for command variants and report behavior.

## Contributing Data

The most useful contributions for the current project direction are:

- `empty` captures that reduce false positives
- `static_presence` captures that improve idle robustness
- `motion` captures across chips, routers, and room layouts

Before opening a PR:

1. collect at least 10 samples per label when possible
2. keep labels homogeneous
3. add a stable environment name and an explicit dataset role to every catalog entry
4. record room type and unusual environmental conditions in the description
5. verify the dataset with `./espectre collect --info`
6. run `python tools/validate_dataset_quality.py` and resolve admission FAILs

## Next Steps

- [`ML_TRAINING.md`](ML_TRAINING.md) for model training, export, and regression checks
- [`API.md`](API.md#csi-collection) for raw HTTP framing and session ownership
- [`README.md` (tools)](../tools/README.md) for analysis helpers
