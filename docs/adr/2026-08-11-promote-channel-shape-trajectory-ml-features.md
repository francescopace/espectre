# ADR: promote channel-shape trajectory ML features

- Status: Accepted
- Date: 2026-08-11
- Updated: 2026-08-16

## Context

Motion collected in a vacation home crossed repeatable WiFi blind spots. The operator moved continuously, but packet loss and weak-link intervals reduced the apparent motion fraction. This is a production-relevant domain-transfer problem: supported commercial chip families are represented in the corpus, while unseen rooms, routers, RSSI regimes, and intermittent packet delivery remain unpredictable.

The Aggregated-IQR-7 model used L1-delta autocorrelation and frequency-coherence curve standard deviation. A host-only screen found that two physical-time channel-shape trajectory features recovered more vacation-home motion and generalized better across environments when they replaced both inputs. Leave-one-chip-out C3 recall fell by `1.36` percentage points, but supported-chip low-RSSI and quiet replay are the deployment gates; chip exclusion remains a conservative diagnostic rather than a veto.

The initial Trajectory-7 promotion used a standalone full-band `chan_shape_spread` tracker alongside two features derived from an eight-subband physical-time trajectory. That full-band path reserved approximately `22.1 KiB` of device state and repeated profile work on every CSI packet. A subband formulation measures the same participation concept from adjacent profiles already retained by the trajectory tracker, so it can remove the separate history.

Ten-seed selection showed more seed sensitivity than the full-band baseline (`4/10` safe seeds versus `10/10`) and a higher unseen-bedroom false-positive tail. The selected subband seed nevertheless passed the blocked OOF and every sealed deployment replay gate without an override, improved difficult `exclude` recall, preserved zero effective alarms, and passed exact host, MicroPython, and C++ DCT-mode parity. Host benchmarks measured `21.39%` less packet-path time, unchanged pure MLP time, and an `18.73%` lower modeled packet-plus-evaluation total. Requested dynamic float storage falls exactly from `24,720` to `2,320` bytes at the default window, a reduction of `22,400` bytes; allocator overhead and on-device peak RAM are not included.

The Subband-7 set already stores an eight-subband, `80 ms`-binned Hellinger trajectory. Guarded Kendall lag-excess reuses that path: each bin encodes 28 pairwise subband orders, treats differences within `2%` of the profile maximum as ties, requires at least eight commonly ordered pairs, and reports the median positive excess of the `240 ms` Kendall distance over the mean of its three constituent `80 ms` distances. The feature is sparse (about `83%` exact zeros on clean train), complementary to excess path (`r≈0.27`), and cheap: two `uint32` masks per trajectory bin plus XOR/popcount at extraction.

This record incorporates the later Kendall eighth-input promotion. That file is removed rather than kept as an extension because this ADR is the cumulative current High Accuracy feature decision.

## Decision

Promote this ordered eight-input High Accuracy schema, which is the current default when `tools/train_ml_model.py` runs without `--features`:

1. `turb_iqr_over_mean_aggr`;
2. `turb_autocorr`;
3. `turb_zcr`;
4. `l1_delta_lag_ratio`;
5. `chan_shape_spread_subband`;
6. `chan_shape_coherent_innovation_energy`;
7. `chan_shape_excess_path`; and
8. `chan_shape_subband_kendall_lag_excess`.

The committed production artifact uses topology `8 -> 24 -> 12 -> 1`, seed `656446646`, standard scaling, false-positive weight `1.75`, and the `base,drift,burst-loss` augmentation recipe. This seed was chosen for the strongest grouped-OOF balance among eligible 8-feature trials rather than the search's paired-recall ranking winner. Host, MicroPython, and C++ extractors share the pairwise-order masks and lag-excess formula. Rank-gap and the seven-input Kendall substitution remain host-only research.

The trajectory tracker uses a gain-normalized eight-subband energy profile, `80 ms` physical-time median bins, a one-second window, exact duplicate suppression, and missing-bin skipping. Coherent innovation measures positive low-order DCT energy left after a constant-velocity prediction and high-order noise subtraction. Excess path measures positive two-step path length beyond its chord after subtracting high-order DCT path excess. Subband spread measures the participation of adjacent trajectory-profile differences. Kendall lag-excess is computed on bin finalize from the stored pairwise-order masks.

Remove `chan_shape_spread`, its feature ID, source routing, and its standalone tracker from the C++ and MicroPython production surfaces. Historical comparisons and rollback evidence remain auditable through the recorded experiment configuration and results in `docs/FEATURES.md`; the retired implementation is not retained in the executable host candidate registry. Lightweight does not activate the High Accuracy channel-shape tracker; its current aggregated-turbulence decision is recorded separately.

The production occupancy floor is independent and is recorded in the temporal-admission ADR; the committed eight-feature export was trained after that floor moved to seven tenths.

## Decision History

Detailed measurements for every baseline and individual feature remain in [`FEATURES.md`](../FEATURES.md). The durable production lineage is:

| Date | Baseline | Resolution |
| --- | --- | --- |
| 2026-06-29 | Raw and relative pre-Core-6 baselines | Preserved as historical evidence only |
| 2026-07-07 | Core-6 | Replaced after absolute and energy-like inputs proved weak-link and seed fragile |
| 2026-07-23 | Coherence-6 | Replaced after the lag ratio improved reserved replay behavior |
| 2026-07-27 | Coherence-7 and a seven-feature-only runtime surface | Replaced after absolute L1 members inverted weak-link behavior |
| 2026-07-28 | Invariant-5 | Retained the gain-invariance direction but expanded the physical feature surface |
| 2026-08-07 | Aggregated-IQR-7 | Replaced after trajectory features improved environment transfer |
| 2026-08-11 | Trajectory-7 | Promoted the physical-time trajectory, then replaced its standalone full-band spread history |
| 2026-08-12 | Subband-7 | Promoted DCT-backed subband spread in place of the full-band history; topology `7 -> 24 -> 12 -> 1`, seed `1584727888` |
| 2026-08-16 | Subband-8 | Added guarded Kendall lag-excess as an eighth input; topology `8 -> 24 -> 12 -> 1`, seed `656446646` |

## Validation

The initial Trajectory-7 promotion passed all paired selection and holdout replays, all quiet replays, gain stress, and the environment-transfer gates. Its augmented export reached blocked grouped CV F1 `99.1%`, `94.84%` worst paired recall, `0.34%` maximum quiet raw FP, and zero effective alarms.

The promoted Subband-7 seed reached blocked OOF F1 `99.187%`, CV worst-session recall / FP `88.764%` / `1.163%`, selection minimum recall `96.264%`, selection maximum paired FP `0.284%`, holdout minimum recall `97.971%`, holdout maximum paired FP `0%`, and zero effective selection or holdout alarms. On `exclude`, it improved worst recall from `5.556%` to `7.639%` and reduced motion misses from `163` to `146`, with `0%` FP and zero alarms. Leave-one-environment-out macro recall / FP / F1 was `98.609%` / `0.213%` / `99.055%`.

The only ablation to pass the initial robust CV comparison removed `chan_shape_excess_path`, but a fresh search produced no selection-safe seed (`0/10` versus `4/10` for Subband-7), and every seed caused a quiet selection alarm. The selected six-feature seed also caused two holdout quiet alarms and one `exclude` quiet alarm, while cross-environment macro FP increased from `0.213%` to `0.296%`. The production schema therefore retains `chan_shape_excess_path` despite its low marginal importance.

A substitution of `chan_shape_excess_path` with Kendall failed sealed quiet and cross-domain stationary gates. Adding Kendall as an eighth input, after the later fixed temporal-admission contract, passed a ten-seed in-memory search (`8/10` eligible). Rank-gap substitution and six-feature spread removal remained unsafe, so Kendall does not replace an existing slot. The occupancy-70% export reached blocked OOF F1 `99.0%`, worst-session recall / FP `93.7%` / `2.5%`, paired `14/14` with worst recall `97.13%` and maximum FP `0.29%`, quiet maximum FP `0.31%`, and zero alarms.

## Consequences

- High Accuracy obtains all four channel-shape inputs from one DCT-backed physical-time tracker.
- The production runtime no longer allocates, updates, or exposes the full-band lag-profile and motion-energy rings.
- The MLP grows from 505 to 529 parameters. Packet-path work stays the existing DCT; Kendall signatures are computed on bin finalize and XOR/popcount runs at evaluation. High Accuracy allocates `104` bytes of Kendall masks on the existing trajectory window and `96` bytes of extra MLP weights.
- Finalized trajectory bins store orthonormal DCT coefficients rather than median profiles. DCT linearity preserves the innovation residual, and Parseval's identity preserves the full-profile L2 distances in excess path, so only the current bin requires a DCT during each extraction. A profile-space reference test and the generated C++/Python replay-parity gate protect this equivalence.
- A fresh independent-room replay and representative-device CPU, allocator, and peak-RAM measurements remain follow-up validation, not blockers to this accepted promotion.
- DS2 and slow EMA remain host-only rollback candidates if hardware measurements or new-room data invalidate the subband choice.

## Related

- [FEATURES.md](../FEATURES.md)
- [ALGORITHMS.md](../ALGORITHMS.md)
- [ML_TRAINING.md](../ML_TRAINING.md)
- [2026-07-23-separate-ml-training-data-from-promotion-replays.md](2026-07-23-separate-ml-training-data-from-promotion-replays.md)
- [2026-03-08-use-host-side-validation-gates-for-detector-promotion.md](2026-03-08-use-host-side-validation-gates-for-detector-promotion.md)
- [2026-08-15-use-fixed-temporal-csi-admission.md](2026-08-15-use-fixed-temporal-csi-admission.md)
