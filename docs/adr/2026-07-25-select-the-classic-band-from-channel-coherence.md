# ADR: select the Classic band from channel coherence

- Status: Accepted
- Date: 2026-07-25

## Context

`2026-07-20-keep-the-12-tone-ht20-classic-band.md` locked
`DEFAULT_SUBCARRIERS = (14, 17, 20, 23, 26, 29, 35, 38, 41, 44, 47, 50)` on four
stated design rules and a count sweep across `N = 10..16`. Both the rules and
the sweep were derived under a bin-ordering assumption that turned out to be
wrong on three of five chip families.

Classic-MAC parts (ESP32, C3, S3) deliver HT20 CSI in Espressif's native
`0~31, -32~-1` order, with DC in bin 0. Wi-Fi 6 parts (C5, C6) deliver it
centered, with DC in bin 32. The code assumed the centered convention
everywhere. Every rule in the previous ADR fails under the corrected reading:

| rule as written | actual meaning |
| --- | --- |
| uniform spacing across the usable HT20 range | spans `+/-18` of `+/-28`, 64% of the range |
| symmetric placement around DC index `32` | true on C5/C6 only; DC sat in bin 0 elsewhere |
| explicit skip of the DC-null region `30..34` | the real null region is `29..35`, and the band **included** 29 and 35 |
| 3-index margin inside guard-band limits `11..52` | the real limits are `4..60`; the margin wasted 7 subcarriers per side |

The consequence for the sweep is decisive. On ESP32, C3, and S3 the payload
bins 29 and 35 are identically zero, so the `N=12` row measured ten live tones
on those chips, and different `N` placed a different number of tones on dead
bins. The independent variable was not the tone count, so the seven rows are
not comparable and the "flat within noise" conclusion is unsupported — not
refuted, simply never measured.

Selecting a replacement empirically on the same corpus was rejected as the
method. A per-band F1/AUC ranking over the recorded datasets put the band that
classic chips had been using *by accident* in first place, while the physical
ranking below puts it fourth. That divergence is the hazard: the accidental
band scores well for reasons specific to these recordings.

## Decision

Select the band from measured channel and radio properties, and keep the count
at 12.

`DEFAULT_SUBCARRIERS = (4, 8, 13, 18, 23, 28, 36, 41, 46, 51, 56, 60)`, that is
subcarriers `+/-4, +/-9, +/-14, +/-19, +/-24, +/-28`. `GUARD_BAND_LOW` and
`GUARD_BAND_HIGH` are corrected to `4` and `60`.

## Validation

All measurements below come from channel and radio statistics on quiet and
motion recordings. None uses a detection metric, and none was used to pick
between near-equivalent candidates.

**No tone is contaminated.** Per-tone amplitude and relative jitter across four
chip families show no significant LO leakage near DC, no filter roll-off at the
edges, and no anomaly at the pilot positions `+/-7` and `+/-21` — expected,
since CSI comes from the HT-LTF where every tone carries a known symbol. Two
mild gradients remain against a flat floor of `0.0235`: relative jitter rises
about 10-13% for `|sc| <= 3` and 13-20% for `|sc| >= 21`.

**Motion and noise live on opposite coherence scales.** Correlating each tone's
temporal fluctuation against its neighbours:

| lag (sc) | MHz | quiet | motion |
| --- | --- | --- | --- |
| 1 | 0.31 | 0.272 | 0.901 |
| 5 | 1.56 | 0.210 | 0.787 |
| 10 | 3.12 | 0.144 | 0.498 |
| 14 | 4.38 | 0.032 | 0.189 |

Quiet fluctuation is nearly independent tone to tone, while the motion
perturbation stays coherent over about 10 subcarriers (3.1 MHz), implying an
RMS delay spread near 70 ns. The coherence distance is stable across the three
recorded environments (0.5 crossing between lag 6 and 9), so it is a property
of indoor propagation rather than of one room.

**Independent looks saturate.** With 3.1 MHz of coherence inside 17.5 MHz of
usable bandwidth, HT20 offers roughly four independent views of the channel,
and no tone count raises that:

| tones | spacing | neighbour r | independent looks |
| --- | --- | --- | --- |
| 4 | 18.3 | 0.000 | 4.00 |
| 6 | 11.0 | 0.425 | 3.51 |
| 12 | 5.0 | 0.787 | 3.49 |
| 24 | 2.4 | 0.881 | 3.43 |

**Span is therefore the scarce resource**, and the candidate bands rank:

| band | span | mean r | jitter | independent looks |
| --- | --- | --- | --- | --- |
| **selected `+/-4..+/-28`** | 56 | 0.216 | 0.0253 | **3.55** |
| `+/-3..+/-28`, spacing 5 | 56 | 0.222 | 0.0258 | 3.49 |
| inner `\|sc\| <= 20` | 40 | 0.316 | 0.0238 | 2.68 |
| accidental classic-chip band | 52 | 0.295 | 0.0254 | 2.83 |
| previous ADR band | 36 | 0.356 | 0.0240 | 2.44 |

The selected band spans the full usable range, so it keeps all the available
independent looks, and stops short of `|sc| <= 3` to avoid the near-DC jitter
rise. Accepting the edge tones costs 15-20% jitter there and buys 45% more
independent information than the previous band, which is the trade the coherence
measurement says to make.

**The count was later re-tested end to end, and 12 held.** On 2026-07-26 the
tone count was swept at `16`, `20`, `24`, and `32` against the `12`-tone
control, each band refitted with the production recipe, because comparing bands
under one coefficient set measures the mismatch rather than the band.

Per-chip aggregates on the healthy corpus flattered the wider bands: at `16`
tones the single ESP32 capture went from `91.9%` to `99.4%` recall and the
worst per-chip recall rose from `91.9%` to `97.1%`. Two checks removed the
illusion.

Per-pair gates, which the aggregates hide: at the operating point where `16`
tones keep the empty-room recordings silent, individual pairs fall to
`89.3-92.2%` recall, well under the `95%` target the aggregate suggested it had
cleared.

The high-rate stress capture, which is `exclude` and therefore absent from the
corpus the sweep scored. Recall on it degrades monotonically with the count:

| decimated rate | 12 tones | 16 tones | 32 tones |
| --- | --- | --- | --- |
| 500 pps | 88.2% | 79.6% | 49.6% |
| 400 pps | 93.3% | 85.2% | 54.1% |
| 300 pps | 96.0% | 92.3% | 61.6% |
| 100 pps | 97.5% | 93.5% | 67.9% |

This is the independent-looks table above, seen from the detection side. Tighter
spacing raises the neighbour correlation (`0.787` at 12 tones, `0.881` at 24),
so the extra tones add correlated information and compress the dynamic range of
the spatial-variance feature. The one thing they do buy, averaging down the
per-tone quiet noise, is small: the quiet coefficient of variation moves only
from `0.1219` at 12 tones to `0.1135` at 32, and it does not compensate.

The channel statistics predicted this and the detection measurement confirmed
it, so the count stays at 12 on two independent grounds rather than one.

## Alternatives Considered

### Keep the previous band

Rejected. It spans 36 of 56 usable subcarriers and ranks last on independent
looks. Its stated justification does not survive the corrected bin convention.

### Adopt the band classic chips were using accidentally

Rejected. It ranks first on this corpus by F1 and AUC but fourth on the physical
measure. Promoting it would lock in a coincidence of these recordings.

### Change the tone count at the same time

Rejected for now, and recorded as follow-up below. The span correction moves
independent looks from 2.44 to 3.55; any count change inside 12-20 moves far
less. Changing one thing at a time keeps the next regression attributable.

## Consequences

Benefits:

- the band spans the full usable HT20 range on every chip family
- band selection now rests on measured coherence and noise rather than on rules
  derived from a wrong bin layout
- the corpus is reserved for verification instead of selection, so a fortuitous
  dataset artifact cannot pick the production band

Trade-offs and open regressions:

- **Classic does not currently clear the promotion gate on every chip.** After
  refitting with `tools/fit_classic_detector.py`, the gate-constrained operating
  point holds false positives under 5% on every chip, but recall lands at 93.7%
  on C3 and 94.2% on ESP32 from a single pair, against a 95% target. Moving the
  single global threshold trades recall on one chip for false positives on
  another, so the residual gap is a per-chip separability question, not an
  operating-point question.
- **Weak links now fail on false alarms rather than on missed motion.** Recall
  holds from 85.4% to 100% across the nine real weak-link pairs, but five of them
  produce one to eleven effective alarms on their quiet segment, and the
  lowest-quality capture breaches the false-positive ceiling. This inverts the
  failure mode recorded when this ADR was written: the pathological C3 captures
  that produced sub-62% recall were replaced with clean ones on 2026-07-25, and
  with those gone the recall problem went with them.
- **ML was retrained on the corrected band** (2026-07-25, same seed `368496409`,
  `--fp-weight 1.5 --augment`, no seed search) and improved on every reserved
  replay: recall 97.3-100.0% at 0.2-1.9% false positives with zero effective
  alarms, against 97.9-100.0% at up to 0.9% before. Long-quiet false positives
  stay at or below 0.32% average. The one regression is the C3 weak-link pair,
  where ML recall sits at 83.5%, so `test_ml_detection_accuracy` fails on that
  pair and on nothing else.
- edge tones carry 15-20% more relative jitter than mid-band tones, which the
  span gain is judged to outweigh but which is a real cost at weak links.

## Follow-up

**The Classic gap is accepted and documented, not closed.** Since the threshold
is a single global value and ML clears the targets on the same recordings, the
gap is a two-feature separability limit rather than an operating-point one, and
no threshold move fixes it. The project therefore publishes the per-chip spread
as a known limit in `docs/ALGORITHMS.md`, `docs/TUNING.md`, and the generated
performance report, and points deployments that need the recall or the alarm
quietness at `ml`. Raising the Classic ceiling stays open as feature-side work:
the 2026-07-23 coherence-oriented candidate (`turb_zcr` plus
`l1_delta_autocorr`) is the leading direction and needs its own fit and gate run.

The `85%` recall floor in `test_low_rssi_classic.py` and `test_low_rssi.cpp`
stays where it is. Both tests now cover every real weak-link pair rather than one
C3 pair, and every pair clears the floor, so the recall assertion is green; the
margin on the weakest C3 pair is under a point, which is why 85 rather than 90 is
the level the corpus supports today. The floor should move up when the feature
work lands and must never move down to accommodate a regression.

What keeps those tests red is the false-alarm side: `effective_alarms == 0` fails
on five weak-link pairs across C5, C6, and S3, and `fp_rate <= 5.0` fails on one
C5 capture. That capture is also the lowest-scoring pair in the corpus at `42.7`,
with a quiet-baseline `q95` logit of `+1.48`, meaning its static-presence segment
sits above the decision boundary; it is a candidate for the same treatment the
pathological captures received rather than evidence about the detector.

Note that these assertions are stricter than the published stress policy, which
calls weak-link Classic results report-only and sets the ML stress targets at
recall `>90%` and false positives `<10%`. Tightening the test contract past the
documented policy is deliberate for now, so the gap stays visible, but the two
should be reconciled once the feature work settles.

**Evaluate the tone count on its own.** The count was left at 12 because the
physics does not discriminate inside 12-20: independent looks saturate for any
`N >= 6`, so extra tones buy only noise averaging, and that averaging is weak
because quiet fluctuation is only partly independent — quiet dispersion falls
just 13% going from 12 tones to 56, against the 54% fully independent noise
would give. The knee sits near 16-20. A separability measure (`d'`) over the
same sweep was non-monotone with about 10% spread and no trend, so it cannot
arbitrate either.

The count question therefore needs a refit-based comparison rather than a
physical argument, run once the band and the Classic gate are settled. Note that
raising it requires removing the hard 12-tone assumptions in `segmentation.py`
and `csi_features.py`, and the `len(selected_band) == 12` assertion in
`test_validation_real_data.py` — another reason it was not bundled here.

## Related

- `2026-07-20-keep-the-12-tone-ht20-classic-band.md` (superseded by this ADR)
- `2026-06-09-replace-runtime-nbvi-with-fixed-shared-subcarriers.md`
- `2026-07-08-promote-classic-detector-and-retire-legacy-baselines.md`
