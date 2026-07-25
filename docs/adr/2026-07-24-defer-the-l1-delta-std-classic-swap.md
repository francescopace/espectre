# ADR: defer the `l1_delta_std` Classic swap

- Status: Accepted; the decision stands, the evidence below does not
- Date: 2026-07-24

> The ranking in this ADR was computed on the band
> `DEFAULT_SUBCARRIERS = (14, 17, 20, ...)` and on the Classic coefficients of
> the time. Both changed the next day: see
> [2026-07-25-select-the-classic-band-from-channel-coherence.md](2026-07-25-select-the-classic-band-from-channel-coherence.md),
> after which both detectors were refit. The decision below still holds, because
> production Classic is still `l1_delta + turb_autocorr` and none of the
> promotion reasoning depended on the band. The numbers do not: anyone
> revisiting the swap has to re-run the ranking on the current band before
> quoting them.

## Context

Classic currently uses weighted logistic fusion of `l1_delta` and
`turb_autocorr`. A grouped blocked-CV feature ranking over the complete
non-`exclude` corpus showed that the strongest replacement pair is
`turb_autocorr + l1_delta_std`:

- best single feature: `turb_autocorr`
- best pair by blocked out-of-fold F1: `turb_autocorr + l1_delta_std`
- blocked OOF F1: `93.88%` for `turb_autocorr + l1_delta_std` versus `91.04%`
  for the current `turb_autocorr + l1_delta`
- real low-RSSI slice: `92.75%` F1 and `3.90%` false-positive rate for
  `turb_autocorr + l1_delta_std` versus `84.28%` and `15.88%` for the current
  pair

That ranking made the swap look like the best next Classic simplification:
still two features, still no ML runtime cost, but with much better weak-link
behavior.

The open question was whether that analytical win survives the production
runtime contract:

- startup-calibrated thresholding
- fixed 12-tone HT20 band
- paired normal-link promotion gates (`>95%` recall, `<5%` false positives)
- long-quiet replay stability
- aligned Python and C++ implementations

## Decision

Do not replace production Classic with `turb_autocorr + l1_delta_std` yet.

Keep the current production pair, `l1_delta + turb_autocorr`, as the only
promoted Classic path. Record `turb_autocorr + l1_delta_std` as an
experiment-ready candidate that still needs its own runtime design and
promotion evidence.

In particular:

- do not ship the pair as a direct drop-in replacement for `ClassicDetector`
- do not weaken the existing per-pair promotion suite to make the swap pass
- do not treat grouped-CV ranking alone as sufficient promotion evidence for a
  threshold detector with startup calibration

## Validation

A scratch implementation of the new pair was built in both Python and C++
style, using:

- logistic coefficients fitted on the full non-`exclude` corpus
- `l1_delta_std` plus `turb_autocorr` as the only fused features
- the existing Classic startup-calibration flow, adapted to the new feature
  pair

The result was promising in aggregate, but it failed the current normal-link
promotion suite as a direct replacement. The candidate operating point that
best balanced weak-link and long-quiet behavior still failed five real paired
datasets:

| Dataset | Failure |
|--------|--------:|
| `c3_bedroom_static_presence_c3_64sc_dev0000acebe64adb64_20260722_195858_502587_0001` | recall `86.8%` |
| `c6_bedroom_static_presence_c6_64sc_dev00007c2c6742bbac_20260722_185259_158062_0001` | FP `7.2%` |
| `c5_bedroom_static_presence_c5_64sc_dev000030eda0e46278_20260723_143540_186526_0001` | FP `7.4%` |
| `c6_bedroom_static_presence_c6_64sc_dev00007c2c6742bbac_20260723_133317_279759_0001` | recall `92.2%` |
| `s3_bedroom_static_presence_s3_64sc_dev000010b41de8ec00_20260723_130651_930086_0001` | FP `5.1%` |

The failure pattern matters:

- some sessions required lower thresholds to recover recall
- others required higher thresholds to suppress false positives
- the startup-calibrated threshold movement therefore remained unstable across
  nominally normal sessions, even though the pair ranked well in blocked CV

Because the promotion suite is the contract for the production non-ML path, the
swap was not landed.

## Follow-up Experiment

After the direct swap was deferred, follow-up work kept the promoted Classic
pair but made the weak-link policy explicitly RF-aware instead of relying only
on the internal L1-floor proxy.

The final experiment that passed the current replay contract was:

- keep the normal Classic `l1_delta + turb_autocorr` branch
- keep `l1_delta_std + turb_autocorr` as the weak-link-only branch
- drive the branch choice from the per-window median `rssi_dbm`
- simplify the RSSI gate to a hard weak-link switch at `-76 dBm`
- use the same RSSI decision for both score blending and startup-threshold
  preparation, so threshold and steady-state scoring stay coherent
- keep Python and C++ replay paths aligned, including long-quiet datasets

Two intermediate findings mattered:

- a broad RSSI ramp was too aggressive
- using RSSI only for the score while leaving the threshold path
  RSSI-agnostic created an incoherent operating point and replay drift

Once the same RSSI-aware policy was applied to both score and threshold, the
context-dependent Classic path stayed promotion-safe on the current corpus:

- normal-link aggregate remained effectively unchanged
- low-RSSI aggregate improved materially versus the non-RSSI-aware baseline
- Python and C++ paired plus long-quiet replay parity remained green

This follow-up does not reverse the main ADR decision. It shows that the
analytically stronger `l1_delta_std` feature can help in production only when
it is introduced as part of a runtime RF-context policy, not as a pure global
feature swap.

## Alternatives Considered

### Replace `l1_delta` with `l1_delta_std` immediately

Rejected. The pair improves weak-link slices and blocked-CV ranking, but it
does not yet meet the current per-pair production gates under the real runtime
calibration path.

### Keep the pair only as a weak-link-specific runtime branch

Accepted in restricted form. The validated path is now an RSSI-aware Classic
branch that keeps the promoted default pair for normal links and activates the
`l1_delta_std` branch only in weak-link conditions.

### Introduce a separate experimental detector instead of changing Classic

Accepted as the preferred follow-up path if the project wants to continue this
line. A distinct experiment avoids destabilizing the promoted Classic baseline
while preserving the evidence and code path for future trials.

## Consequences

Benefits:

- the promoted Classic baseline stays green on the current paired and long-quiet
  suites
- the project keeps a durable record that the swap was tested and rejected for
  now, not forgotten
- future work can reuse the ranking evidence without repeating the same direct
  replacement attempt

Trade-offs:

- Classic keeps the weaker low-RSSI secondary feature in production for now
- the best analytical pair remains unavailable to end users until it gets a
  runtime policy that also passes normal-link promotion
- weak-link improvements require either a separate detector variant or a
  context-dependent Classic branch rather than a pure coefficient swap
- thresholding and scoring for that branch must stay RF-aware together; making
  only one of them context-dependent is not stable enough

## Related

- `2026-07-22-adopt-session-centered-l1-excursion-for-low-rssi.md`
- `2026-07-23-adopt-coherence-6-as-the-production-ml-feature-set.md`
- `2026-07-20-keep-the-12-tone-ht20-classic-band.md`
