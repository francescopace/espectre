# Dataset Collection Plan

Review basis: `data/auto_generated/DATASET_QUALITY_CHECK.md`, `docs/performance/README.md`, and the current real-data performance tests run on 2026-07-26. The excluded-pair figures were re-measured on 2026-07-27 against Classic on the lag ratio.

Note: `test` below means `holdout` in `data/dataset_info.json`.

## Why the excluded pairs stay excluded

An excluded pair cannot be promoted to `test`/holdout. Each one has been
examined repeatedly while making decisions, so it can no longer answer the
question a holdout exists to answer. That door closes on first use, not on
role assignment.

Promotion to `select` is technically open, because selection pairs are meant
to be inspected during model choice, but the two `100 pps` C3 pairs would
gate Classic on captures Classic currently fails, turning an open problem
into a permanent red light without adding information we do not already have.
They are more useful as a named regression watchlist for the Classic recall
item in `ROADMAP.md`: they are the hardest cases we own, ML clears them, and
their numbers are recorded below.

The `500 pps` and `1000 pps` C3 pairs are excluded on cadence, not quality.
They belong to the high-rate work, not to this plan.

## Priority 1: Replace clearly bad pairs

- [x] Chip: `C6` | Environment: `bedroom` | RSSI: `normal link, target -45 to -55 dBm` | Role: `select` | Labels: `static_presence + motion` | Retain the current pair from `2026-07-22 18:52/18:54`. With the lag-ratio feature its idle/motion medians separate at `1.10` versus `2.54` (`0.99999` AUC), and a seven-feature model trained with the restored C5 normal-link pair reaches `0.88`-`1.02%` FP with no alarms. The old failure exposed missing training coverage rather than a contaminated pair.
- [ ] Chip: `C5` | Environment: `bedroom` | RSSI: `normal link, target -45 to -55 dBm` | Role: `test` | Labels: `static_presence + motion` | Replace the current pair from `2026-07-24 12:59/13:05`, then drop the old pair. It fails the Classic gate hard (`69.2%` recall) even though the link is not weak.
- [ ] Chip: `C3` | Environment: `bedroom` | RSSI: `normal link, target -45 to -55 dBm` | Role: `select` | Labels: `static_presence + motion` | Collect a verified replacement for the pair from `2026-07-22 19:58/20:01`. The old pair is restored with role `exclude`: it is useful to ML (`0.9976` lag-ratio AUC, `98.0%` candidate recall, and `0.0%` FP) but cannot serve as a shared selection gate while production Classic remains at `82.5%` recall. Re-measured on 2026-07-27 against Classic on the lag ratio, that recall is unchanged at `82.5%` with `0.9852` separation and `0.0%` FP, so the lag ratio did not touch this failure. The link is `-39/-38 dBm`, the strongest in the corpus, which rules out the weak-link explanation.
- [ ] Chip: `C3` | Environment: `bedroom` | RSSI: `weak link, target -65 to -75 dBm` | Role: `test` | Labels: `static_presence + motion` | Collect a verified replacement for the low-RSSI pair from `2026-07-25 13:58/14:00`. The old pair is restored with role `exclude`: it retains useful weak-motion evidence (`0.9375` lag-ratio AUC) without entering training or blocking the promotion holdout. Re-measured on 2026-07-27, Classic reaches `74.2%` recall at `0.9872` separation and `0.0%` FP. At `-63/-62 dBm` this is a moderate link, not one of the `-70` to `-80 dBm` captures the weak-link work targeted.
- [ ] Chip: `S3` | Environment: `bedroom` | RSSI: `weak link, target -70 to -80 dBm` | Role: `test` | Labels: `static_presence + motion` | Replace the current low-RSSI pair from `2026-07-22 17:20/17:23`, then drop the old pair. It can dominate reserved holdout FP for otherwise good ML seeds (`36.7%` FP, `30` effective alarms on seed `1975812835`).

## Priority 2: Replace noisy training captures

- [ ] Chip: `S3` | Environment: `bedroom` | RSSI: `normal link, target -45 to -55 dBm` | Role: `train` | Labels: `static_presence + motion` | Recollect a cleaner pair to replace the current `2026-07-23 13:06/13:09` train pair. The static-presence side is the worst normal-link idle capture in the room (`11.1%` FP), so it should not remain the only normal bedroom training pair.
- [ ] Chip: `C3` | Environment: `bedroom` | RSSI: `normal link, target -45 to -55 dBm` | Role: `train` | Label: `empty` | Recollect a cleaner empty-room capture. The current bedroom empty sample is the only outright bad `empty` dataset in the report (`5.6%` FP).
- [ ] Chip: `C6` | Environment: `bedroom` | RSSI: `normal link, target -45 to -55 dBm` | Role: `test` | Label: `test` | Recollect a quieter long-run test capture. The current bedroom quiet test is the weakest long-recording replay (`3.6%` FP, `1.5s` burst).

## Priority 3: Fill structural coverage gaps

- [ ] Chip: `any` | Environment: `bedroom` | RSSI: `weak link, target -70 to -80 dBm` | Role: `select` | Labels: `static_presence + motion` | **The corpus has no weak-link selection pair at all.** Every weak pair is in `train` (five), `holdout` (three), or `exclude` (one), so weak-link behaviour can only be studied by contaminating training or by burning the holdout. That blocks the `low_rssi` non-regression exemption from ever being decided on measured dispersion rather than on analogy with normal links. One reserved weak selection pair unblocks it; two would let a spread be distinguished from a single recording's quirk

- [ ] Chip: `ESP32` | Environment: `bedroom` | RSSI: `normal link, target -45 to -55 dBm` | Role: `select` | Labels: `static_presence + motion` | Add the first reserved normal-link selection pair. `docs/performance/README.md` currently shows `N/A` for ESP32 reserved ML replays because no such pair exists yet.
- [ ] Chip: `ESP32` | Environment: `bedroom` | RSSI: `normal link, target -45 to -55 dBm` | Role: `test` | Labels: `static_presence + motion` | Add the first reserved normal-link holdout pair so ESP32 stops being evaluated only on training data.
- [ ] Chip: `ESP32` | Environment: `bedroom` | RSSI: `normal link, target -45 to -55 dBm` | Role: `train` | Label: `empty` | Add the first empty-room bedroom sample for ESP32. There is currently no ESP32 empty capture in the catalog.
- [ ] Chip: `ESP32` | Environment: `bedroom` | RSSI: `normal link, target -45 to -55 dBm` | Role: `test` | Label: `test` | Add the first long quiet test run for ESP32. The long-recording report is currently `N/A` for ESP32.
- [x] Chip: `C5` | Environment: `bedroom` | RSSI: `normal link, target -45 to -55 dBm` | Role: `train` | Labels: `static_presence + motion` | Restored the `2026-07-23 14:35/14:38` pair. Its seven-feature distribution matches the difficult C6 selection regime, and adding it changes all four previously failing fixed seeds to `5/5` paired passes with `0.88`-`1.02%` max FP and no alarms.

## Priority 4: Nice-to-have cleanup after replacements land

- [ ] Chip: `C6` | Environment: `bedroom` | RSSI: `normal link, target -45 to -55 dBm` | Role: `train` | Labels: `static_presence + motion` | Collect one cleaner spare pair and compare it against the current `2026-07-23 13:33/13:35` train pair, which is still noisy (`5.6%` idle FP) even though aggregate metrics still pass.
- [ ] Chip: `C5` | Environment: `bedroom` | RSSI: `weak link, target -70 to -75 dBm` | Role: `train` | Labels: `static_presence + motion` | Collect one cleaner weak-link training pair before considering whether to retire the current `2026-07-25 14:47/14:49` pair, which has the weakest C5 low-RSSI idle quality in the report.
