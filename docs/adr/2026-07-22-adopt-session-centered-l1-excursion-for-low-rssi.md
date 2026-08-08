# ADR: adopt session-centered L1 excursion for low-RSSI Classic detection

- Status: Accepted
- Date: 2026-07-22

## Context

Classic uses weighted logistic fusion of `l1_delta` and `turb_autocorr`. `l1_delta` is normally the stronger term, but a real ESP32-C3 HT20 capture at approximately `-80 dBm` exposed an out-of-domain failure:

- the fitted `l1_delta` center is `0.0367`
- the low-RSSI static median was `0.1254`
- the low-RSSI motion median was lower, at `0.1056`
- the raw logit saturated near probability `1.0` during both phases
- startup calibration raised the threshold to approximately `0.9954`, but could not recover class separation

The production replay therefore reported `100%` recall and `100%` false positives, with `33.1%` precision. The failure was not uniform gain: weak-link noise raised the quiet profile-displacement floor, and motion could move that floor either up or down.

Earlier experiments did not solve this combination:

- soft rescaling preserved the wrong direction when motion lowered `l1_delta`
- one-sided excess over the startup floor discarded downward motion evidence
- fixed threshold and temporal-policy sweeps could not restore separation
- shape-only variants and autocorrelation-only decisions lost too much recall

Subsequent real low-RSSI captures from ESP32-C5, ESP32-C6, and ESP32-S3 showed a second, independent effect. Their features remained separable, but the startup threshold followed only `30%` of the quiet-session logit displacement. This left the C6 threshold too high for motion and the S3 threshold too low for its quiet baseline. Across the complete real corpus, C6 recall fell to `92.41%` and S3 false-positive rate rose to `8.59%`.

## Decision

Keep the existing raw two-feature Classic model inside its fitted domain. When the startup L1 floor moves outside that domain, fade continuously to an absolute, session-centered L1 excursion.

The startup calibration records the median L1 floor:

```text
floor = median(startup_l1_delta)
```

The blend is derived from the fitted L1 center and scale:

```text
blend_start = center + scale
blend_end = center + 2.5 * scale
blend = clamp((floor - blend_start) / (blend_end - blend_start), 0, 1)
```

This corresponds to a blend start near `0.0637` and full activation near `0.1042`. The robust term treats displacement in either direction as motion evidence:

```text
robust_l1_norm = 1.5 * abs(l1_delta - floor) / scale
```

Classic blends the raw and robust logits with the same `blend`. It also blends the normal session-adapted threshold back toward the validated global threshold as the robust path becomes active. The detector does not depend on RSSI metadata; it responds directly to the observed reliability of its L1 feature.

For the normal adaptive path, apply `75%` of the startup `q95` displacement from the training idle reference in logit space. This replaces the original `30%` partial correction. The value is shared across chips and was selected on the complete real paired corpus; it is not a low-RSSI or chip-specific profile.

Implement the same formula and constants in Python and C++. Keep one real-data regression per runtime using the retained low-RSSI C3 pair.

## Validation

On the real low-RSSI pair, both runtimes produce:

| Metric | Before | After |
|--------|-------:|------:|
| Recall | 100.00% | 88.70% |
| False-positive rate | 100.00% | 0.70% |
| Precision | 33.12% | 98.43% |
| Effective baseline alarms | 1 | 0 |

The highest startup L1 floor in the current normal paired and long-recording corpus is approximately `0.0406`, below `blend_start`. The blend therefore remains exactly zero on that corpus. The stronger startup threshold adaptation changes the decision threshold, but preserves one common Classic path for normal and weak-link sessions.

Across all real paired datasets, the previously failing chip aggregates change as follows:

| Chip | Metric | 30% startup shift | 75% startup shift |
|------|--------|------------------:|------------------:|
| ESP32-C6 | Recall | 92.41% | 97.35% |
| ESP32-C6 | False-positive rate | 0.43% | 2.83% |
| ESP32-S3 | Recall | 98.57% | 98.57% |
| ESP32-S3 | False-positive rate | 8.59% | 1.62% |

The other chip aggregates remain within the same `>95%` recall and `<5%` false-positive targets. The retained low-RSSI C3 pair remains on the fully session-centered path and is therefore numerically unchanged.

Before expanding the real weak-link corpus, the validation baseline was:

- paired recall: `98.66%`
- paired false-positive rate: `0.087%`
- long-quiet false-positive rate: `0.168%`
- long-quiet effective alarms: `6`

## Alternatives Considered

### Keep raw L1 and the saturated adaptive threshold

Rejected. Threshold movement cannot recover separation when both phases saturate and the primary feature changes in the opposite direction.

### Promote asymmetric soft normalization

Rejected. It improves some noisy sessions but preserves the assumption that motion raises `l1_delta`, which the real low-RSSI pair disproves.

### Subtract only positive excess over the startup floor

Rejected. `max(l1_delta - floor, 0)` removes the downward excursion that carries most motion evidence in the low-RSSI pair.

### Use turbulence autocorrelation alone

Rejected. It avoids L1 saturation but reaches only approximately `69%` recall at an acceptable false-positive rate on the real pair.

### Apply session centering to every Classic session

Rejected. It increases false alarms on long quiet recordings. Continuous out-of-domain blending preserves the validated normal path.

## Consequences

- Classic remains a two-feature, vote-free detector.
- Normal sessions below `blend_start` are numerically unchanged.
- Weak-link sessions gain bidirectional motion evidence without RSSI metadata.
- Startup calibration stores one additional bounded L1 sample array.
- Retained real pairs and aligned C++/Python regressions remain the promotion evidence. Clearly marked synthetic derivatives may broaden Classic stress testing, but do not replace real weak-link validation.
- This decision covers Classic only. ML low-RSSI robustness remains a separate follow-up requiring its own real-data evidence and runtime design.

## Related

- `2026-07-08-promote-classic-detector-and-retire-legacy-baselines.md`
- `2026-07-20-keep-the-12-tone-ht20-classic-band.md`
- `2026-07-07-adopt-gated-startup-threshold-calibration-for-classic-detector.md`
