# ADR: replace the Classic L1 mean with a lag ratio

- Status: Superseded
- Date: 2026-07-26
- Superseded by: 2026-07-30-adopt-frequency-coherence-for-classic.md

## Context

Classic fused `l1_delta`, the mean normalized-profile displacement at lag `L`, with `turb_autocorr`. The L1 profile is already divided by its own per-packet mean, so per-packet gain cancels and the feature is scale-invariant as intended.

It is not floor-invariant, and that is the difference that mattered. A mean displacement carries a unit, so it inherits whatever noise floor the link happens to have. When the link weakens the floor rises and the feature rises with it, whether or not anything moved. Measured per feature across the corpus:

| feature | AUC normal link | AUC weak link | sign flips |
| --- | --- | --- | --- |
| `turb_autocorr` | 0.9987 | 0.9962 | 0/8 |
| `turb_zcr` | 0.0021 | 0.0038 | 0/8 |
| `l1_delta_autocorr` | 0.9721 | 0.9072 | 0/8 |
| `l1_delta_std` | 1.0000 | 0.8946 | 1/8 |
| **`l1_delta`** | **1.0000** | **0.8705** | **2/8** |

The shape statistics hold; the level statistics degrade and, on two captures, separate in the wrong direction. A weight fitted for one polarity then pushes toward the wrong class.

The consequence reached production. Two captures calibrated their threshold to `0.969` and `0.986`, effectively pinned at the ceiling, and Classic managed `69%` recall on them where ML managed `100%` on a fixed `0.5` threshold. The information was there; the feature was not carrying it.

## Decision

The first Classic feature is now `delta_lag_ratio`: the mean displacement at lag `L` divided by the mean displacement at lag `1`, over the same window.

Noise saturates the displacement immediately, so adjacent packets already differ by the full noise amount and the ratio sits near `1.0`. Real channel evolution keeps accumulating with the lag and lifts it. Both terms carry the same unit, so the floor divides out instead of adding in. The feature has an intrinsic no-motion reference that a mean never had.

The implementation costs one extra running sum and no extra normalization. The lag-1 reference profile is the slot behind the lagged reference in the ring that already exists, so a single pass over the normalized profile accumulates both displacements. The two delta windows share one allocation carved into two views. Both displacements pass through their own Hampel filter, because an outlier surviving only in the denominator would depress the ratio and read as less motion.

`l1_delta` itself is unchanged and still exported to ML as feature id `17`.

## Validation

Refit through `tools/fit_classic_detector.py`, which reads the feature from the production detector rather than a reimplementation, at `--fp-target 3.0`. At the time of this fit, the tool still defaulted to `5.0`; that looser ceiling cleared the pooled rate but put two alarms into the empty-room gate, which binds first.

| | `l1_delta` | `delta_lag_ratio` |
| --- | --- | --- |
| fit F1 | 96.104% | 97.631% |
| fit recall | 97.056% | 98.175% |
| fit worst session recall | 55.056% | **82.022%** |
| replay mean recall | 96.81% | **98.65%** |
| replay worst pair recall | **69.21%** | **90.83%** |
| pairs under 95% | 3 | 2 |
| replay mean FP | 2.12% | **1.23%** |
| empty-room alarms | 0 | 0 |
| empty-room worst FP | 5.14% | **2.45%** |

It wins on every measure, and the two pinned captures come back at `99.68%` and `99.71%`. The Python and C++ report parity gate passes.

**Startup calibration is still needed.** The ratio's idle level varies `1.82x` across the corpus's quiet captures against `14.29x` for the mean, and its calibration prefix tracks its session `2.3x` more closely (dispersion of `log(prefix/rest)` falls from `0.073` to `0.032`), but that is not `1.0`. Disabling the startup shift puts an alarm back into the empty-room gate at every threshold tried and costs `5.5` points of worst-pair recall. Calibration gets easier, not redundant.

## Alternatives Considered

### Keep `l1_delta` and normalize it differently

Rejected on measurement. `std/mean` of the delta series scores `0.7602` AUC on normal links against `1.0000` for the current feature, and a `q95/median` crest scores `0.7822`. Both would pay on good links to buy little on weak ones. `l1_delta_cv` was already listed as a candidate elsewhere and this is the measurement that should retire it.

### Drop `turb_autocorr` and run on the ratio alone

Rejected, and the pooled numbers nearly hid why. A solo fit scores `98.0%` recall on the operating-point sweep, better than what Classic ships today. The per-pair replay then finds one capture at `62%`, falling to `17.8%` as the threshold rises, and an empty-room gate that never closes at any threshold: five alarms at the best point, still two at the worst recall. The two features cover different captures, and the aggregate hid it because twenty-six good pairs outvoted the one that failed. `choose_base_threshold` now reports `worst session recall` alongside the pooled figure and warns when they diverge, so the next attempt of this shape is visible from the fit output.

## Consequences

The worst pair in the corpus moves from `69.21%` to `90.83%` recall with lower false positives everywhere, and the two captures that had prompted a proposal to delete them as bad data are now read correctly.

**The L1 noise-blend safeguard is now dead code.** It engages when the startup L1 floor passes `FEATURE_CENTER[0] + FEATURE_SCALE[0]`, which is `2.022` against a typical idle level of `1.437`, and disabling it reproduces the shipped numbers exactly on all 27 pairs and 12 empty-room recordings. It exists only for the saturation the mean suffered, which the ratio does not. It is left in place here so this change carries one behaviour at a time, and its removal is tracked separately.

The `l1_delta` name survives in the Classic metric dictionary and in `current_l1_delta_`, where it now holds a ratio. That is a naming debt, not a behavioural one, and renaming it touches the ML feature ids that legitimately still mean the mean.

The detector holds one more `capacity`-sized float window and one more Hampel state per Classic instance.
