# ADR: configure detector windows in milliseconds

- Status: Accepted
- Date: 2026-08-10
- Supersedes: 2026-07-25-derive-detector-timing-from-the-measured-packet-rate.md (sample-count window); 2026-07-28-keep-production-feature-lags-at-nominal-offsets.md (sample-count window)

## Context

The deployed detector window was configured as a fixed packet count. Its default of 100 samples represented one second only at the nominal `100 pps`; a slower generator widened the physical interval, while a faster generator shortened it. Runtime evaluation and periodic publishing had already moved to elapsed-time schedules, leaving the analysis window as the remaining public packet-rate-dependent setting.

Earlier evidence against a temporal window shortened the detector window on recordings that still carried their original cadence. That changed detector geometry without reproducing a real lower-rate source. A valid rate sweep must instead remove packets from a stable stream, preserve a clean lower cadence, and resolve the detector window from the resulting timestamps.

The configuration, runtime, replay, validation, and training paths also need one contract. Otherwise, a model can be trained on a fixed number of samples and deployed over a different physical interval whenever generator throughput changes.

## Decision

Replace the public `segmentation_window_size` packet count with `segmentation_window_size_ms`:

- the default is `1000 ms`, and the supported configuration range is `1000-2000 ms`;
- runtime and replay measure clean CSI throughput, divide the requested duration by the measured interval, and round up so the resolved window never covers less than the requested duration;
- the resolved detector storage range is `80-1000` samples;
- at the default duration, the window resolves to 80 samples at `80 pps`, 100 at `100 pps`, and 120 at `120 pps`;
- below `80 pps`, live detection stays on hold at idle until packet supply recovers;
- C++, MicroPython, ESPHome, host replay, offline analysis, validation, and training use the same duration-to-sample conversion; and
- live and replay inputs require advancing packet timestamps; there is no packet-count timing fallback.

The production feature offsets remain fixed at the fitted packet geometry: the L1 displacement ratio stays `10:1`, and turbulence autocorrelation stays at lag `1`. This ADR supersedes only the sample-count window portion of [2026-07-28-keep-production-feature-lags-at-nominal-offsets.md](2026-07-28-keep-production-feature-lags-at-nominal-offsets.md).

Stable packet-rate augmentation is distinct from loss. It selects packets across a source interval, rewrites sequence counters and device/Wi-Fi timestamps to a clean lower cadence, and lets feature extraction resolve the temporal window from that cadence. Packet loss, burst loss, stutter, drift, and feature jitter remain separate augmentation effects.

## Validation

A 60-second sweep over 22 normal-link pairs removed packets from stable streams to create a genuine `80 pps` cadence. With the default temporal window resolved to 80 samples, aggregate ML recall was `98.844%`, aggregate false positives were `0.019%`, and worst-session recall was `92.797%`. The fixed 100-sample control produced `99.546%` recall and `0.041%` false positives. The aggregate result supports `80 pps` as the floor, while the localized worst session argues against extending support below it.

On the explicit high-rate C3 regression pair, stable decimation to `120`, `100`, and `80 pps` with matching one-second windows kept ML at `100%` recall and `0%` false positives at every rate. Classic reached `99.1%` recall and `0%` false positives at `80 pps`.

Correct native-cadence replay also replaced two optimistic Classic results produced by nominal-rate rounding. A normal-link C3 replay moved from `93.64%` to `91.64%` recall, and a weak-link S3 replay moved from `85.06%` to `83.62%`; false positives remained within `0-4.2%`, and the normal-link chip aggregates remained above their production gates.

The Classic settled-level recovery margin was therefore revalidated against the temporal-window geometry. Reducing `CLASSIC_SETTLE_MARGIN_LOGITS` from `2.8` to `2.7` restored the weak-link S3 floor from `83.62%` to `85.06%` recall. Normal-link mean recall moved from `98.365%` to `98.391%`; normal-link mean false positives remained `1.836%`, the weak-link maximum remained `8.857%`, and the long-quiet maximum and alarm count remained `10.794%` and `45`. A lower global startup strength was rejected because it increased quiet false positives and alarms, and a fresh coefficient fit was rejected because its deployment replay reached `24.28%` false positives and 84 alarms on the worst quiet C6 recording.

The complete Python performance gate passed with the temporal contract and the retrained export: Classic aggregate recall remained above `95%` on every chip, and ML aggregate recall remained between `99.0%` and `100%`, with aggregate false-positive rates between `0%` and `0.1%`.

The production ML model was then retrained on 2026-08-10 with seed `1161881508` and the promoted `base+drift+burst-loss` augmentation recipe. Stable packet-rate augmentation used `packet_rate_scale=(0.8, 1.0)` so the shared `1000 ms` window trained on clean lower-rate intervals containing fewer samples, independently of the separate `5%` packet-loss, burst-loss, stutter, drift, and feature-jitter effects. The final training matrix contained `617,883` windows across 27 lineage groups and all five supported chip families; timing provenance was 46 clean, three degraded, zero poor, and zero unknown captures.

Three-fold grouped CV by lineage group produced `98.7%` recall, `99.5%` precision, `0.2%` false positives, and `99.1%` F1, with `99.1%` blocked out-of-fold F1. Worst-chip recall was `97.5%` on C3, worst-chip false positives were `0.9%` on C6, and the worst-five lineage-group mean was `95.8%` recall with `1.0%` false positives. The most difficult individual lineage reached `92.1%` recall, while the highest lineage false-positive rate was `5.1%`; these localized tails remain explicit collection targets rather than reasons to return to a packet-count window.

All 14 reserved paired deployment replays passed with zero alarms, a maximum false-positive rate of `0.29%`, and worst recall of `94.56%`. All nine reserved quiet replays also passed with zero alarms and a maximum false-positive rate of `0.51%`. Against the previously exported model on the paired gate, the new candidate improved maximum false positives from `0.57%` to `0.29%` and worst recall from `93.12%` to `94.56%`. The gain-stress regression remained identical from `0.5x` through `2.0x` amplitude scale at `99.3%` recall, `0.0%` aggregate false positives, and `99.6%` F1, confirming that the export still contains no gain-sensitive input dimensions.

The accepted export regenerated `src/python/micro_espectre/ml_weights.py`, `src/cpp/core/ml_weights.h`, and `data/auto_generated/ml_test_data.npz` from the same trained candidate.

## Alternatives Considered

### Keep a fixed 100-sample window

Rejected. It makes the physical analysis interval depend on generator throughput and prevents deployment, replay, and packet-rate augmentation from sharing one temporal contract.

### Support rates below 80 pps with a smaller window

Rejected. The available evidence shows localized recall loss at 80 samples already; below that supply level, holding detection is more explicit than emitting results outside the validated envelope.

### Stretch the 80-sample minimum over more than one second

Rejected. It would silently violate the configured duration, increase latency, and mix different physical intervals under one setting.

### Resample every stream to 100 pps

Rejected. Decimation can remove excess samples but cannot create missing information on a slower source, and interpolation would introduce synthetic CSI values into both training and evaluation.

## Consequences

Benefits:

- detector latency and physical coverage no longer depend on generator packet rate;
- runtime, replay, validation, and training share one window contract;
- packet-rate augmentation can train clean intervals containing fewer packets without treating them as loss; and
- unsupported packet supply produces an explicit hold state instead of an unvalidated detector result.

Trade-offs:

- the detector may be reconstructed when measured throughput changes enough to alter the resolved sample count;
- Classic startup calibration restarts after a window resize;
- replay and training require trustworthy timestamps; and
- rate-dependent sample counts make results generated before this decision non-comparable unless their replay contract is stated.

## Related

- [2026-07-25-derive-detector-timing-from-the-measured-packet-rate.md](2026-07-25-derive-detector-timing-from-the-measured-packet-rate.md)
- [2026-07-28-keep-production-feature-lags-at-nominal-offsets.md](2026-07-28-keep-production-feature-lags-at-nominal-offsets.md)
