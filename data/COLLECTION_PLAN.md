# Collection Plan

Target captures to close the current dataset gaps, sequenced after the
2026-07-23 train/evaluation separation (see
[2026-07-23-separate-ml-training-data-from-promotion-replays.md](../docs/adr/2026-07-23-separate-ml-training-data-from-promotion-replays.md)).

Status 2026-07-23: normal-link bedroom training pairs are collected for C3,
C5, C6, and S3 (two S3 devices), and S3 has a normal-link holdout pair
(Waveshare device). Remaining gaps: the ESP32 kit (no out-of-sample gate, no
`empty` capture) and the C3, C5, and C6 normal-link holdout pairs (the first
C5 attempt was deleted as contaminated; see the P2 notes).

Durations follow the existing corpus: `static_presence` 3 min, `motion`
1.5 min, `empty` 2 min.

## RSSI Targets

- Normal link: `-45..-60 dBm` (matches the healthy historical captures).
- Weak link (P4 only): `-70..-75 dBm`. Do not go beyond `-77 dBm`: at
`-77/-80` the motion/static turbulence ratio collapses to ~1.0x and the
capture is physically unusable for training.
- Check RSSI from the device logs before starting; the value is stored in  
`rssi_dbm` for post-capture verification.



## P1 — ESP32 Kit

ESP32 is the only chip without a role-isolated gate (single real pair, gate is
in-sample via the legacy fallback) and has no `empty` capture at all.


| Done | Captures                | Environment     | RSSI     | dataset_role     |
| ---- | ----------------------- | --------------- | -------- | ---------------- |
| [ ]  | static 3' + motion 1.5' | bedroom         | -45..-60 | selection        |
| [ ]  | empty 2'                | bedroom         | -45..-60 | selection        |
| [ ]  | static 3' + motion 1.5' | living or hobby | -45..-60 | train (no field) |
| [ ]  | empty 2'                | living or hobby | -45..-60 | train (no field) |


Estimated time: ~13 min.

## P2 — Normal-Link Holdout Pairs

No normal-link holdout exists: all 2026-07-22 holdout pairs turned out to be weak-link captures. Collect in a separate session from P0 (different day, or at least a different time slot), otherwise train and holdout become near-duplicates and the holdout loses its value.


| Done | Chip | Captures                | Environment | RSSI     | dataset_role |
| ---- | ---- | ----------------------- | ----------- | -------- | ------------ |
| [ ]  | C3   | static 3' + motion 1.5' | bedroom     | -45..-60 | holdout      |
| [ ]  | C5   | static 3' + motion 1.5' | bedroom     | -45..-60 | holdout      |
| [ ]  | C6   | static 3' + motion 1.5' | bedroom     | -45..-60 | holdout      |
| [x]  | S3   | static 3' + motion 1.5' | bedroom     | -45..-60 | holdout      |


Notes on the 2026-07-23 holdout captures:

- The S3 holdout pair is the Waveshare LCD device (~`-62 dBm`, ~20 dB
enclosure attenuation, healthy 5.9x motion/static ratio): it doubles as a
novel-hardware generalization check and gates at strict normal-link targets.
- A first C5 holdout attempt (14:22/14:24, trimmed for startup gaps) was
deleted entirely: the static was contaminated (turbulence `5.4e-01` at
`-44 dBm`, ~40x the healthy C5 floor, motion/static ratio `1.84x`, and both the old Core-6 model and the swap6 candidate false-fired heavily on it while staying clean on every other C5 replay), and dataset admission requires paired captures, so the healthy motion half went with it. Re-record the full C5 holdout pair on a separate day, and check the Classic review flags (`Ratio`, robust margin) before assigning the role.

Estimated time: ~18 min.

## P3— Optional

Status 2026-07-23: three swap6 seed searches ran; every candidate stayed under
the `5%` FP bar but none reached the zero-alarm full-restore required in
broken-baseline mode, so the sealed holdout was never opened. The weak-link
question below stays undecided until a promoted swap6 model is evaluated on
the S3 weak holdout.

- One `empty` 2' per chip (C3, C5, C6, S3) in a room different from its
current train `empty` (~8 min).
- One real weak-link pair for S3 (optionally C6) at `-70..-75 dBm` for
training: all real weak pairs currently sit in the holdout, so the model
learns the weak regime only from synthetic derivatives. Skip if the swap6
feature set closes the S3 recall gap on its own.



## After Collection

1. Report the new files; `selection` and `holdout` roles are assigned in
  `dataset_info.json` (new entries default to train, which is correct for
   P1 and the train rows above).
2. Run the dataset quality validation to refresh pair metadata and admission.
3. Rerun the gated seed search (with `--augment`, and `--features` for the
  swap6 candidate set).

