# Collection Plan

Remaining dataset gaps under the 2026-07-23 train/evaluation separation (see
[2026-07-23-separate-ml-training-data-from-promotion-replays.md](../docs/adr/2026-07-23-separate-ml-training-data-from-promotion-replays.md))
and the Coherence-6 production feature set (see
[2026-07-23-adopt-coherence-6-as-the-production-ml-feature-set.md](../docs/adr/2026-07-23-adopt-coherence-6-as-the-production-ml-feature-set.md)).

Durations follow the existing corpus: `static_presence` 3 min, `motion`
1.5 min, `empty` 2 min.

## Current State (2026-07-23)

| Role | Real normal-link pairs | Real weak-link pairs | Synthetic weak pairs | Empty |
|------|------------------------|----------------------|----------------------|-------|
| train | 13 (C3/C5/C6/S3 across bedroom/hobby/living, ESP32 bedroom) | 0 | 8 | 4 |
| selection | 4 (C3/C5/C6/S3 bedroom) | 0 | 0 | 4 |
| holdout | 1 (S3 Waveshare bedroom) | 4 (C3/C5/C6/S3 bedroom) | 0 | 4 |

Coherence-6 is promoted (seed `1312857390`). The known open issues this plan
addresses:

- **ESP32 has no role-isolated gate and no `empty` capture.** It has a single
  real pair (train), so its paired gate falls back to the in-sample legacy
  path and it contributes nothing to the quiet gate.
- **Only S3 has a normal-link holdout.** C3, C5, and C6 normal-link holdout
  pairs are missing (the first C5 attempt was deleted as contaminated).
- **Training has zero real weak-link data.** All real weak pairs are reserved
  in holdout (correctly: gates must be real, never synthetic), so the model
  learns the weak regime only from synthetic derivatives. Synthetic fidelity
  in the coherence features was refit for Coherence-6; a real weak pair in
  training would still be the stronger signal.

## RSSI Targets

- Normal link: `-45..-60 dBm` (matches the healthy historical captures).
- Weak link: `-70..-75 dBm`. Do not go beyond `-77 dBm`: at `-77/-80` the
  motion/static turbulence ratio collapses to ~1.0x and the capture is
  physically unusable for training.
- Check RSSI from the device logs before starting; it is stored in `rssi_dbm`
  for post-capture verification.
- Before assigning any role, check the Classic review flags in
  `DATASET_QUALITY_CHECK.md` (`Ratio`, robust margin): a static capture with
  elevated turbulence means the person moved during it and the pair is
  contaminated.

## P1 — ESP32 Kit

The only chip without a role-isolated gate or an `empty` capture.

| Done | Captures | Environment | RSSI | dataset_role |
|------|----------|-------------|------|--------------|
| [ ] | static 3' + motion 1.5' | bedroom | -45..-60 | selection |
| [ ] | empty 2' | bedroom | -45..-60 | selection |
| [ ] | static 3' + motion 1.5' | living or hobby | -45..-60 | train |
| [ ] | empty 2' | living or hobby | -45..-60 | train |

Estimated time: ~13 min.

## P2 — Normal-Link Holdout Pairs (C3, C5, C6)

S3 already has one (Waveshare device). Collect these in a session separate
from the training captures (different day or time slot), otherwise train and
holdout become near-duplicates and the holdout loses its value.

| Done | Chip | Captures | Environment | RSSI | dataset_role |
|------|------|----------|-------------|------|--------------|
| [ ] | C3 | static 3' + motion 1.5' | bedroom | -45..-60 | holdout |
| [ ] | C5 | static 3' + motion 1.5' | bedroom | -45..-60 | holdout |
| [ ] | C6 | static 3' + motion 1.5' | bedroom | -45..-60 | holdout |

Estimated time: ~14 min.

The existing S3 holdout is the Waveshare LCD device (~`-62 dBm`, ~20 dB
enclosure attenuation, healthy 5.9x ratio): it doubles as a novel-hardware
generalization check and gates at strict normal-link targets.

## P3 — Real Weak-Link Pair For Training

Training's weak regime is currently synthetic-only. A real weak pair assigned
to `train` (not holdout) would give the model genuine weak-link structure,
directly targeting the residual weak-link recall gap and the incoherent-motion
weakness (energetic-but-incoherent motion the model under-detects).

| Done | Chip | Captures | Environment | RSSI | dataset_role |
|------|------|----------|-------------|------|--------------|
| [ ] | S3 | static 3' + motion 1.5' | any | -70..-75 | train |
| [ ] | C6 | static 3' + motion 1.5' | any | -70..-75 | train (optional) |

Estimated time: ~5-9 min. Keep the four existing real weak pairs in holdout;
these are additional captures, not moved from holdout.

## P4 — Optional Coverage

- One `empty` 2' per chip in a room different from its current train `empty`
  (~8 min), to broaden the quiet training distribution.
- Normal-link motion with varied, less coherent movement styles (steady walking
  at distance/angle, not only close jerky motion) to cover the
  energetic-but-incoherent motion signature the current model under-detects.

## After Collection

1. Report the new files. New entries default to `train`; assign `selection`
   and `holdout` roles in `dataset_info.json` for the reserved captures.
2. Run `tools/validate_dataset_quality.py` to refresh pair metadata and
   admission, and review the Classic flags before trusting any reserved pair.
3. If real weak or new normal captures were added, recalibrate and regenerate
   the synthetic low-RSSI derivatives so their coherence features stay faithful.
4. Rerun the gated seed search with `--features` (Coherence-6 set) and
   `--augment`, watching the reserved weak-link replays.
