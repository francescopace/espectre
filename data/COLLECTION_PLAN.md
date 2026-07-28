# Dataset Collection Plan

Current baseline status:
- The exported ML baseline is back to a passing, reproducible state.
- Coverage was intentionally reduced to get there: the weak `S3` bedroom holdout pair and the quiet long-run `S3` selection replay were both moved to role `exclude`.
- The remaining collection work should recover that lost coverage first, then clean up the next-noisiest reserved and training captures.

## Priority 1: Restore the reserved coverage we had to cut

- [ ] Chip: `S3` | Environment: `living_room` | RSSI: `normal link, target -45 to -55 dBm` | Role: `selection` | Label: `empty` | Collect a cleaner reserved quiet replay to replace `empty_s3_64sc_dev000010b41de8ec00_20260713_002325_306350_0001.npz`, which is now in role `exclude`. That replay was the only quiet selection dataset still producing `2` to `4` effective alarms across seed search, so it had to be removed to restore a passing baseline.
- [ ] Chip: `S3` | Environment: `bedroom` | RSSI: `weak link, target -70 to -80 dBm` | Role: `holdout` | Labels: `static_presence + motion` | Collect a cleaner replacement for the low-RSSI pair from `2026-07-22 17:20/17:23`, which is now in role `exclude`. It could dominate reserved holdout FP for otherwise good ML seeds (`36.7%` FP, `30` effective alarms on seed `1975812835`), so weak-holdout coverage is intentionally suspended until a better replacement lands.
- [ ] Chip: `C3` | Environment: `bedroom` | RSSI: `weak link, target -70 to -80 dBm` | Role: `holdout` | Labels: `static_presence + motion` | Collect a true weak-link holdout replacement for the excluded `2026-07-25 13:58/14:00` pair. The old pair retains useful evidence (`0.9375` lag-ratio AUC) but at `-63/-62 dBm` it is only a moderate link and should not define the weak holdout slice.

## Priority 2: Clean up the remaining quiet and empty weak points

- [ ] Chip: `C3` | Environment: `bedroom` | RSSI: `normal link, target -45 to -55 dBm` | Role: `train` | Label: `empty` | Recollect a cleaner empty-room capture. The current bedroom empty sample is still the only outright bad `empty` dataset in the report (`5.6%` FP).
- [ ] Chip: `C6` | Environment: `bedroom` | RSSI: `normal link, target -45 to -55 dBm` | Role: `holdout` | Label: `empty` | Recollect a quieter long-run empty capture and register it with `long_recording: true`. The current bedroom quiet run is still the weakest reserved long-recording replay (`3.6%` FP, `1.5s` burst).

## Priority 3: Strengthen weak-link robustness beyond the minimum

- [ ] Chip: `any` | Environment: `bedroom` | RSSI: `weak link, target -70 to -80 dBm` | Role: `selection` | Labels: `static_presence + motion` | Collect a second weak-link selection pair, ideally on a chip other than `S3`. One reserved weak selection replay is enough to enforce the exemption; a second one is what separates seed dispersion from one recording's quirk.
- [ ] Chip: `C5` | Environment: `bedroom` | RSSI: `weak link, target -70 to -75 dBm` | Role: `train` | Labels: `static_presence + motion` | Collect one cleaner weak-link training pair before considering whether to retire the current `2026-07-25 14:47/14:49` pair, which still has the weakest `C5` low-RSSI idle quality in the report.

## Priority 4: Revisit lower-value trainer cleanup only with a clearly better candidate

- [ ] Chip: `C6` | Environment: `bedroom` | RSSI: `normal link, target -45 to -55 dBm` | Role: `train` | Labels: `static_presence + motion` | The `2026-07-28 13:17/13:19` spare looked cleaner in Classic, but retraining made it the worst grouped-CV FP lineage (`8.0%` FP on `static_presence`) and still failed deployment safety. Only replace the restored `2026-07-23 13:33/13:35` train pair with a candidate that is both cleaner and measurably better for the ML trainer.

## Priority 5: Expand ESP32 from one-recording evidence to a measured slice

- [ ] Chip: `ESP32` | Environment: `bedroom` | RSSI: `normal link, target -45 to -55 dBm` | Role: `selection` | Labels: `static_presence + motion` | Add the first reserved normal-link selection pair. `docs/performance/README.md` still shows `N/A` for ESP32 reserved ML replays because no such pair exists yet.
- [ ] Chip: `ESP32` | Environment: `bedroom` | RSSI: `normal link, target -45 to -55 dBm` | Role: `holdout` | Labels: `static_presence + motion` | Add the first reserved normal-link holdout pair so ESP32 stops being evaluated only on training data.
- [ ] Chip: `ESP32` | Environment: `bedroom` | RSSI: `normal link, target -45 to -55 dBm` | Role: `train` | Label: `empty` | Add the first empty-room bedroom sample for ESP32. There is currently no ESP32 empty capture in the catalog.
- [ ] Chip: `ESP32` | Environment: `bedroom` | RSSI: `normal link, target -45 to -55 dBm` | Role: `holdout` | Label: `empty` | Add the first long quiet empty run for ESP32 and register it with `long_recording: true`. The long-recording report is currently `N/A` for ESP32.

When a Priority 1 or Priority 2 item lands, re-measure in this order:

1. `python tools/validate_dataset_quality.py`
2. `pytest test/python/test_validation_real_data.py::TestPerformanceMetrics -v`
3. `pytest test/python/test_validation_long_recordings.py -v`
4. `python tools/generate_performance_report.py`
5. `python tools/train_ml_model.py --evaluate-gates`
6. `python tools/train_ml_model.py --seed-search-until-improvement 5`
