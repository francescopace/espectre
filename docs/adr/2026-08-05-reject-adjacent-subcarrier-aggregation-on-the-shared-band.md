# ADR: reject adjacent-subcarrier aggregation on the shared twelve-tone band

- Status: Accepted
- Date: 2026-08-05

## Context

`2026-07-25-select-the-classic-band-from-channel-coherence.md` measured that
quiet fluctuation is nearly independent tone to tone while the motion
perturbation stays coherent over about 10 subcarriers. That is exactly the
condition under which averaging a few adjacent bins into each selected tone
should raise the per-tone signal-to-noise ratio: the noise partially cancels,
the signal does not.

The open question was whether that amplitude-level gain reaches the detector, or
whether it destroys the short-timescale and frequency-local structure the
features are built on.

Scope matters here, because the "12-of-64 path" is narrower than it sounds.
Classic fuses `turb_autocorr` and `chan_freq_coh_curve_std`, and only the first
reads the 12-tone amplitude buffer; `chan_freq_coh_curve_std` reads the 56-bin
live complex profile and cannot move under this change. Of the ten production ML
features, five read the buffer and five do not.

This is distinct from the pair averaging archived on 2026-07-20, which averaged
*selected* tones together and so halved the number of spatial looks. Aggregation
keeps twelve profile entries and widens only what feeds each one.

## Decision

Do not adopt adjacent-subcarrier aggregation on the shared twelve-tone band.
Keep single-bin sampling at each of the twelve selected subcarriers.

The scope is deliberate. Classic rides that band, and Classic degrades, so the
shared path stays as it is. Whether the ML feature set alone would benefit is a
separate question that this ADR measures but does not decide; see Follow-up.

## Validation

All measurements below are reproducible with
`tools/benchmark_subcarrier_aggregation.py`, one mode per section. Aggregation
was injected at the production amplitude-buffer fill, so the whole runtime chain
replays unchanged behind it. Group windows are centered on the
selected bin, clamped inside the usable range, and never include the DC null:
bins 3 and 61 are guard nulls, so an unclamped 3-wide window at the edge tones
would average a hard zero into two of the twelve entries.

**The amplitude-level gain is real.** Per-tone fluctuation was measured as the
packet-to-packet first difference, which is the band the lag-1 autocorrelation
actually consumes, over 20 empty and 20 motion recordings.

The dominant per-tone noise term is not per-tone at all. Raw quiet fluctuation
is `0.11402`; after dividing each packet by its own live-band mean it falls to
`0.02410`, so `78.9%` of it is the common-mode per-packet CSI scale factor.
Cross-tone averaging cannot touch that component, and the production features
already discard it by construction.

Within the remaining per-tone residual, the coherence split holds:

| lag (sc) | MHz | quiet | motion |
| --- | --- | --- | --- |
| 1 | 0.31 | 0.238 | 0.615 |
| 2 | 0.62 | 0.182 | 0.563 |
| 3 | 0.94 | 0.125 | 0.512 |
| 5 | 1.56 | 0.178 | 0.488 |
| 10 | 3.12 | 0.130 | 0.331 |
| 14 | 4.38 | 0.019 | 0.170 |

The quiet lag-1 value reproduces the `0.272` in the band ADR from an independent
measurement. Folding these correlations into the variance of a W-bin mean gives
the predicted gain, and it is worth having:

| W | noise factor | signal factor | SNR gain |
| --- | --- | --- | --- |
| 2 | 0.787 | 0.899 | +14% |
| 3 | 0.692 | 0.855 | +24% |
| 5 | 0.593 | 0.809 | +36% |

**The detector still loses, monotonically.** Replaying the 19 train pairs
through the production `ClassicDetector`, with the fusion coefficients refit per
configuration so each band is scored under its own coefficients rather than the
baseline's:

| configuration | `turb_autocorr` worst pair | fused worst pair |
| --- | --- | --- |
| baseline, single bin | 0.9734 | 0.9800 |
| W=2 magnitude | 0.9599 | 0.9703 |
| W=3 magnitude | 0.9502 | 0.9632 |
| W=5 magnitude | 0.9374 | 0.9524 |
| W=2 coherent | 0.9596 | 0.9700 |
| W=3 coherent | 0.9487 | 0.9626 |
| W=5 coherent | 0.9354 | 0.9511 |

Per-pair AUC medians sit at `0.999` for every configuration, so the median
cannot discriminate and the worst pair carries the evidence. It degrades
monotonically in W under both magnitude and coherent averaging, and coherent
averaging is consistently the worse of the two. Mean paired deltas are small
(`-0.0007` to `-0.0022` fused), but the ordering across an ordered sweep is the
signal, not the size of any single step.

**The mechanism is that a cleaner input is the wrong thing here.**
`turb_autocorr` is a temporal autocorrelation of the turbulence series, and it
is invariant to that series' scale. Its discriminating power comes from the
quiet series being *white*: quiet sits near zero because per-tone noise
decorrelates it, motion sits high because motion varies smoothly. Removing the
white component raises the quiet floor toward motion instead of lowering it:

| configuration | quiet mean | motion mean | gap | d' |
| --- | --- | --- | --- | --- |
| baseline | 0.0716 | 0.7437 | 0.6722 | 3.823 |
| W=3 magnitude | 0.1223 | 0.7780 | 0.6557 | 3.573 |
| W=5 magnitude | 0.1382 | 0.7838 | 0.6456 | 3.464 |

Quiet `turb_autocorr` nearly doubles while motion moves `5%`, so the separation
narrows exactly as the noise improves. No aggregation width escapes this,
because the effect grows with the amount of noise removed.

**The effect is not uniform across the features on that buffer**, which is why
the result should not be generalized past Classic. Scoring each production
feature as `max(AUC, 1-AUC)` so inverted polarity stays comparable, at W=3
magnitude:

| feature | worst pair, baseline | worst pair, W=3 | delta | same limiting pair | mean paired delta | pairs improved |
| --- | --- | --- | --- | --- | --- | --- |
| `turb_mad_over_mean` | 0.6190 | 0.8155 | +0.1965 | yes | +0.0139 | 8/19 |
| `turb_zcr` | 0.9685 | 0.9457 | -0.0229 | yes | -0.0006 | 9/19 |
| `turb_autocorr` | 0.9734 | 0.9502 | -0.0233 | yes | -0.0011 | 9/19 |
| `l1_delta_autocorr` | 0.8403 | 0.8687 | +0.0284 | no | +0.0042 | 11/19 |
| `l1_delta_lag_ratio` | 0.9824 | 0.9576 | -0.0248 | no | -0.0009 | 5/19 |

The five channel-shape and coherence features are bit-identical under
aggregation, which confirms the injection reached only the intended path.

Among the five that read the buffer, only three carry weight. The two statistics
computed on the turbulence series, `turb_autocorr` and `turb_zcr`, both lose on
the same limiting pair in both configurations, as the mechanism above predicts.
`turb_mad_over_mean` gains sharply on its own limiting pair, which is also
consistent: its quiet floor *is* the noise being reduced, rather than the thing
that keeps the quiet series structureless.

The two `l1_delta` features are not evidence in either direction. Their worst
pair is a different recording in the two configurations, so those deltas are a
minimum taken over pairs rather than a paired comparison, and their mean paired
deltas are small in both directions. Do not read a tidy two-family law into this
table: the mechanism is established for the turbulence series, where it was
measured directly, and the L1 family was neither confirmed nor refuted.

## Alternatives Considered

### Coherent (complex) averaging instead of magnitude averaging

Rejected. It is the better noise suppressor in principle, since complex noise is
zero-mean, but it is worse in every measured configuration. With an RMS delay
spread near 70 ns, the channel phase rotates about 24 degrees across a 3-bin
group, so the group partially cancels itself by an amount that depends on the
delay.

### A wider or narrower group

Rejected. The loss is monotone in W across `2`, `3`, and `5`, and the mechanism
predicts that: any width that removes more noise raises the quiet floor further.
There is no width at which the trade turns positive.

### Aggregating the coherence path too

Not evaluated, and out of scope. The coherence features read the full 56-bin
profile precisely to measure frequency-local structure; averaging adjacent bins
before that measurement would remove the quantity being measured.

## Consequences

- The 12-tone amplitude buffer keeps single-bin sampling, so there is no change
  to `csi_format.h`, `segmentation.py`, or the C++/Python parity surface.
- The band ADR's coherence measurement is independently reproduced, which raises
  confidence in the band selection that rests on it.
- The per-packet CSI scale factor is now quantified as `78.9%` of raw quiet
  per-tone jitter, reinforcing why every production feature is scale-invariant.

## Follow-up

**`turb_mad_over_mean` on the weakest pair is a real open lead.** Its worst-pair
separation rises from `0.6190` to `0.8155` under W=3 aggregation, which is the
largest single-feature movement seen on this corpus, and it lands on the pair
that limits the corpus rather than on an already-easy one. Classic does not use
that feature, so this ADR does not settle it. Deciding it needs an ML retrain
under the promotion protocol in
`2026-03-08-use-host-side-validation-gates-for-detector-promotion.md`, with the
two gaining features weighed against the three losing ones; per-feature AUC on
the train corpus cannot predict that net effect. Note that adopting aggregation
for ML alone would mean the two detectors no longer share one amplitude path,
which is a cost the retrain has to clear.

The C++/Python parity and detector performance validations were **not** run for
this ADR. Nothing in the detection path changed, and no configuration survived
the screen to be worth gating.

## Related

- [`2026-07-25-select-the-classic-band-from-channel-coherence.md`](2026-07-25-select-the-classic-band-from-channel-coherence.md)
- [`2026-06-09-replace-runtime-nbvi-with-fixed-shared-subcarriers.md`](2026-06-09-replace-runtime-nbvi-with-fixed-shared-subcarriers.md)
- [`2026-03-08-use-host-side-validation-gates-for-detector-promotion.md`](2026-03-08-use-host-side-validation-gates-for-detector-promotion.md)
- [`FEATURES.md`](../FEATURES.md)
