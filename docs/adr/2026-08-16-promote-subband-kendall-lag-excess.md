# ADR: promote subband Kendall lag-excess as an eighth ML input

- Status: Accepted
- Date: 2026-08-16
- Extends: `2026-08-11-promote-channel-shape-trajectory-ml-features.md`

## Context

The Subband-7 production set already stores an eight-subband, `80 ms`-binned Hellinger trajectory. Guarded Kendall lag-excess reuses that path: each bin encodes 28 pairwise subband orders, treats differences within `2%` of the profile maximum as ties, requires at least eight commonly ordered pairs, and reports the median positive excess of the `240 ms` Kendall distance over the mean of its three constituent `80 ms` distances.

A substitution of `chan_shape_excess_path` with Kendall failed sealed quiet and cross-domain stationary gates. Adding Kendall as an eighth input, after the later fixed temporal-admission contract, passed a ten-seed in-memory search (`8/10` eligible). Rank-gap substitution and six-feature spread removal remained unsafe, so Kendall does not replace an existing slot.

The feature is sparse (about `83%` exact zeros on clean train), complementary to excess path (`r≈0.27`), and cheap: two `uint32` masks per trajectory bin plus XOR/popcount at extraction. The MLP grows from `7 -> 24 -> 12 -> 1` (505 parameters) to `8 -> 24 -> 12 -> 1` (529 parameters).

## Decision

Export `chan_shape_subband_kendall_lag_excess` as the eighth High-Accuracy input, in this order:

1. `turb_iqr_over_mean_aggr`;
2. `turb_autocorr`;
3. `turb_zcr`;
4. `l1_delta_lag_ratio`;
5. `chan_shape_spread_subband`;
6. `chan_shape_coherent_innovation_energy`;
7. `chan_shape_excess_path`; and
8. `chan_shape_subband_kendall_lag_excess`.

The committed artifact uses topology `8 -> 24 -> 12 -> 1`, seed `656446646`, standard scaling, false-positive weight `1.75`, and the `base,drift,burst-loss` augmentation recipe. This seed was chosen for the strongest grouped-OOF balance among eligible 8F trials rather than the search's paired-recall ranking winner. The occupancy-70% export reached blocked OOF F1 `99.0%`, worst-session recall / FP `93.7%` / `2.5%`, paired `14/14` with worst recall `97.13%` and maximum FP `0.29%`, quiet maximum FP `0.31%`, and zero alarms. Host, MicroPython, and C++ extractors share the pairwise-order masks and lag-excess formula. Rank-gap and the seven-input Kendall substitution remain host-only research.

## Consequences

- High Accuracy allocates `104` bytes of Kendall masks on the existing trajectory window and `96` bytes of extra MLP weights.
- Packet-path work stays the existing DCT; Kendall signatures are computed on bin finalize and XOR/popcount runs at evaluation.
- The production occupancy floor is independent and is recorded in the temporal-admission ADR; this export was trained after that floor moved to seven tenths.

## Related

- [FEATURES.md](../FEATURES.md)
- [ALGORITHMS.md](../ALGORITHMS.md)
- [2026-08-11-promote-channel-shape-trajectory-ml-features.md](2026-08-11-promote-channel-shape-trajectory-ml-features.md)
- [2026-08-15-use-fixed-temporal-csi-admission.md](2026-08-15-use-fixed-temporal-csi-admission.md)
