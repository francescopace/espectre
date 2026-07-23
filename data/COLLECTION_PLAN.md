# Collection Plan

Target captures to close the current dataset gaps: no normal-link bedroom
pairs in training, no normal-link holdout, no out-of-sample ESP32 gate, and no
ESP32 `empty` capture. Sequenced after the 2026-07-23 train/evaluation
separation (see
[2026-07-23-separate-ml-training-data-from-promotion-replays.md](../docs/adr/2026-07-23-separate-ml-training-data-from-promotion-replays.md)).

Durations follow the existing corpus: `static_presence` 3 min, `motion`
1.5 min, `empty` 2 min.

## RSSI Targets

- Normal link: `-45..-60 dBm` (matches the healthy historical captures).
- Weak link (P4 only): `-70..-75 dBm`. Do not go beyond `-77 dBm`: at
  `-77/-80` the motion/static turbulence ratio collapses to ~1.0x and the
  capture is physically unusable for training.
- Check RSSI from the device logs before starting; the value is stored in
  `rssi_dbm` for post-capture verification.

## P1 — Normal-Link Bedroom Pairs For Training

Highest priority: training has zero normal-link bedroom pairs, which is the
main reason seed-search candidates fail the bedroom selection replays.

Use a different time of day and person position than the reserved 2026-07-22
sessions (different chair, opposite side of the bed). In the S3 motion capture
include slow and small movements: its known gap is recall.

| Done | Chip | Captures | Environment | RSSI | dataset_role |
|---|---|---|---|---|---|
| [ ] | C3 | static 3' + motion 1.5' | bedroom | -45..-60 | train (no field) |
| [ ] | C5 | static 3' + motion 1.5' | bedroom | -45..-60 | train (no field) |
| [ ] | C6 | static 3' + motion 1.5' | bedroom | -45..-60 | train (no field) |
| [ ] | S3 | static 3' + motion 1.5' | bedroom | -45..-60 | train (no field) |

Estimated time: ~18 min.

## P2 — ESP32 Kit

ESP32 is the only chip without a role-isolated gate (single real pair, gate is
in-sample via the legacy fallback) and has no `empty` capture at all.

| Done | Captures | Environment | RSSI | dataset_role |
|---|---|---|---|---|
| [ ] | static 3' + motion 1.5' | bedroom | -45..-60 | selection |
| [ ] | empty 2' | bedroom | -45..-60 | selection |
| [ ] | static 3' + motion 1.5' | living or hobby | -45..-60 | train (no field) |
| [ ] | empty 2' | living or hobby | -45..-60 | train (no field) |

Estimated time: ~13 min.

## P3 — Normal-Link Holdout Pairs

No normal-link holdout exists: all 2026-07-22 holdout pairs turned out to be
weak-link captures. Collect in a separate session from P1 (different day, or
at least a different time slot), otherwise train and holdout become
near-duplicates and the holdout loses its value.

| Done | Chip | Captures | Environment | RSSI | dataset_role |
|---|---|---|---|---|---|
| [ ] | C3 | static 3' + motion 1.5' | bedroom | -45..-60 | holdout |
| [ ] | C5 | static 3' + motion 1.5' | bedroom | -45..-60 | holdout |
| [ ] | C6 | static 3' + motion 1.5' | bedroom | -45..-60 | holdout |
| [ ] | S3 | static 3' + motion 1.5' | bedroom | -45..-60 | holdout |

Estimated time: ~18 min.

## P4 — Optional, Decide After The swap6 Seed Search

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
