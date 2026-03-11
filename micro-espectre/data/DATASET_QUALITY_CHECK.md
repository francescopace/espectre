# Dataset Quality Check

Last update: 2026-03-11
Source: `data/dataset_info.json`

## Extraction reference

- analysis script: `tools/1_analyze_raw_data.py`

## Validation rule

A pair is considered valid when:

- labels are coherent (`baseline` vs `movement`)
- `movement_variance > baseline_variance`

Computed metrics:

- `Baseline Var`: variance of spatial turbulence on baseline file
- `Movement Var`: variance of spatial turbulence on movement file
- `Ratio`: `Movement Var / Baseline Var`
- `Gap end->start`: time between baseline end and movement start (negative means overlap)

## Results (sorted by chip, then ratio desc)

| Chip | File pair (baseline / movement) | Baseline Var | Movement Var | Ratio | Gap end->start | Status |
|---|---|---:|---:|---:|---:|---|
| C3 | `baseline_c3_64sc_20260203_203509.npz` / `movement_c3_64sc_20260203_203533.npz` | 0.18 | 0.72 | 4.06x | 14.64s | PASS |
| C3 | `baseline_c3_64sc_20260310_232427.npz` / `movement_c3_64sc_20260310_232443.npz` | 0.38 | 0.89 | 2.35x | 5.70s | PASS |
| C5 | `baseline_c5_64sc_20260310_233856.npz` / `movement_c5_64sc_20260310_233928.npz` | 0.04 | 1.09 | 24.95x | 21.93s | PASS |
| C5 | `baseline_c5_64sc_20260310_230906.npz` / `movement_c5_64sc_20260310_230925.npz` | 0.21 | 0.71 | 3.42x | 9.40s | PASS |
| C6 | `baseline_c6_64sc_20260310_230414.npz` / `movement_c6_64sc_20260310_230434.npz` | 0.07 | 0.65 | 9.17x | 9.60s | PASS |
| C6 | `baseline_c6_64sc_20260310_234819.npz` / `movement_c6_64sc_20260310_234839.npz` | 0.07 | 0.56 | 8.12x | 10.18s | PASS |
| ESP32 | `baseline_esp32_64sc_20260214_183059.npz` / `movement_esp32_64sc_20260214_183141.npz` | 2.71e-04 | 3.68e-03 | 13.58x | 31.99s | PASS |
| ESP32 | `baseline_esp32_64sc_20260310_232100.npz` / `movement_esp32_64sc_20260310_232137.npz` | 1.03e-04 | 5.14e-04 | 4.98x | 27.13s | PASS |
| S3 | `baseline_s3_64sc_20260117_222606.npz` / `movement_s3_64sc_20260117_222626.npz` | 0.64 | 2.14 | 3.33x | 10.24s | PASS |
| S3 | `baseline_s3_64sc_20260310_231250.npz` / `movement_s3_64sc_20260310_231306.npz` | 0.14 | 0.21 | 1.51x | 5.58s | PASS |

## Summary

- total pairs: 10
- pass: 10
- fail: 0
