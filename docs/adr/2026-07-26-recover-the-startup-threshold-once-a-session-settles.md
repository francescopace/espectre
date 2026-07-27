# ADR: recover the startup threshold once a session settles

- Status: Accepted
- Date: 2026-07-26

## Context

Classic calibrates its threshold once, from the opening of a session, and never
revisits it. That is fine when the opening represents the session. When it does
not, the threshold stays wrong for the entire run.

One ESP32 capture showed the failure clearly, and it was the last pair missing
the project recall target at `94.2%` against `95%`. Three measurements located
the cause.

**The information was never missing.** Measured threshold-free, that capture's
window features separate motion from idle almost perfectly: `0.9999` AUC for
`l1_delta` and `0.9994` for `turb_autocorr`. A wider selected band cannot help,
because there is nothing left to collect; see
[2026-07-25-select-the-classic-band-from-channel-coherence.md](2026-07-25-select-the-classic-band-from-channel-coherence.md).

**The loss is threshold placement, and only there.** Against the best threshold
achievable at the same false-positive cost, the corpus leaves `+0.34` points of
recall on the table on average, and `16` of `17` normal-link pairs leave
`0.0-0.3`. ESP32 leaves `+4.7`.

**The prefix is the culprit.** On that capture the calibration prefix is `4.14x`
noisier than the rest of the session, so the threshold settles at `0.4212`
against a session whose idle metric never exceeds `0.1104`, a factor of `3.82`.
Elsewhere the prefix is representative or quieter, which is why nothing else
loses anything.

Everything that acted inside the prefix failed. Raising the startup shift
strength above `0.75` lifts ESP32 recall but starts producing empty-room alarms
at `0.78`, so the shipped value was already the largest safe one. Capping the
calibrated threshold at the session's own quiet ceiling hit the same wall.
Refitting the coefficients lost on every chip, ESP32 worst at `-2.3` points.
They all fail together because they read the same unrepresentative prefix.

## Decision

Let the runtime lower the threshold once a session has demonstrated, over a long
stretch, that it runs quieter than its opening.

Every `CLASSIC_SETTLE_BLOCK_EVALUATIONS` evaluations the detector records the
maximum metric logit in that block and keeps the last `CLASSIC_SETTLE_BLOCKS`
of them. Once the ring is full it takes the median of those maxima as the level
the session has settled at. If that level plus `CLASSIC_SETTLE_MARGIN_LOGITS`
converts to a probability below the live threshold, the threshold drops to it.

`12` blocks of `20` evaluations is a `60 s` dwell at the nominal cadence, and
the margin is `3.0` logits.

Three properties carry the safety, and none of them is a tuning choice:

1. **One-sided.** The rule can only lower. It can never hide motion that the
   calibrated threshold would have caught.
2. **Motion holds it up.** During activity the block maxima are high, so the
   candidate lands above the current threshold and nothing happens. The rule
   moves only after a long quiet stretch, which is the evidence that the
   threshold is too high.
3. **Median of block maxima.** A single spike cannot pull the level down, and a
   single quiet block cannot either.

The evidence is dropped on every restart: `reset()`, `clear_buffer()`,
`on_startup_calibration_begin()`, and `set_adaptive_threshold()` all clear it,
so a contaminated stream or a fresh calibration has to earn the lowering again.

## Validation

| metric | before | after |
| --- | --- | --- |
| ESP32 recall | 94.2% | 98.0% |
| worst per-chip recall | 94.2% | 97.7% |
| C3 / C5 / C6 / S3 recall | 97.6 / 99.8 / 99.8 / 99.5 | 97.7 / 99.8 / 99.8 / 99.5 |
| per-chip false positives | 0.1 / 0.1 / 3.6 / 3.3 | unchanged |
| weak-link slice | 99.4% / 3.2% | 99.6% / 3.2% |
| empty-room alarms | 0 | 0 |
| empty-room worst FP | 5.14% | 5.14% |

`98.0%` is close to the `98.8%` that the achievable-threshold analysis said was
available on that capture, so the rule collects most of what was measured to be
there.

**The margin is the safety knob.** Swept over the corpus, holding everything
else fixed:

| margin | ESP32 recall | worst pair recall | pairs below `95%` | mean FP | worst FP | empty-room alarms |
| --- | --- | --- | --- | --- | --- | --- |
| off | 94.2% | 94.2% | 1 | 2.15% | 10.6% | 0 |
| 4.0 | 94.2% | 94.2% | 1 | 2.15% | 10.6% | 0 |
| **3.0** | **98.0%** | **95.4%** | **0** | **2.20%** | **10.6%** | **0** |
| 2.5 | 98.3% | 95.4% | 0 | 2.22% | 10.6% | 0 |
| 2.0 | 98.8% | 96.3% | 0 | 2.25% | 10.6% | 0 |
| 1.5 | 99.1% | 96.8% | 0 | 2.35% | 10.6% | 0 |
| 1.0 | 100.0% | 97.4% | 0 | 2.92% | 12.3% | 0 |
| 0.5 | 100.0% | 98.0% | 0 | 3.97% | 15.6% | 3 |
| 0.0 | 100.0% | 99.1% | 0 | 5.52% | 17.5% | 5 |

`4.0` is not quite inert: it fires on one C3 pair for `+0.9` points and leaves
ESP32 where it was. The lower wall is further down than the shipped value
suggests. False positives are what bind first: the worst pair breaches the
`12%` weak-link ceiling at `1.0`. The empty-room gate holds all the way to
`1.0` and only breaks at `0.5`.

So the usable range runs to roughly `1.5`, and `3.0` ships as its conservative
end rather than as the last value that works. Tightening it is a decision to
take on the re-collected corpus, not on this one.

**The rule does not rescue the five excluded bedroom pairs.** Temporarily
restoring the `[TO BE REPLACED]` captures to an evaluation role and replaying
them gives, in recall:

| pair | calibrated threshold | off | 3.0 | 2.0 | 1.5 | 1.0 | 0.0 | ML |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C3 `...adb64` 07-22 | 0.528 | 83.0% | 83.0% | 83.0% | 84.5% | 87.1% | 92.5% | 68.5% |
| C3 `...ae708` 07-25 low-RSSI | 0.133 | 85.4% | 85.4% | 86.8% | 89.1% | 90.8% | 96.0% | 94.5% |
| C5 `...e46278` 07-24 | 0.969 | 69.2% | 69.2% | 69.2% | 69.2% | 69.2% | 69.2% | 100.0% |
| C6 `...42bbac` 07-22 | 0.986 | 69.3% | 69.3% | 69.3% | 69.3% | 79.0% | 91.4% | 100.0% |
| S3 `...e8ec00` 07-22 low-RSSI | 0.395 | 92.0% | 92.0% | 92.0% | 92.0% | 95.4% | 97.1% | 98.3% |

At the shipped margin the rule leaves all five exactly where the rule-off run
puts them. They only begin to move at `1.5` and below, past the point where the
healthy corpus breaches the weak-link false-positive ceiling.

They are not one failure mode. The C3 `...adb64` capture defeats both detectors,
ML worse than Classic, so its motion really is weak and excluding it is right.
The other four are Classic-specific: ML reaches `94.5%` to `100%` on the same
recordings, so the information is present and Classic is not reading it.

The two worst, C5 and C6, calibrate to `0.969` and `0.986`, effectively pinned
at `CLASSIC_MAX_THRESHOLD`. On C5 the settled rule never fires at any margin
down to `0.0`, so its recall is not threshold-limited at all: the fused metric
does not respond to that motion. This is the startup-calibration failure mode in
its extreme form, and the settled-level rule is not the instrument that fixes
it. It is tracked with the re-collection rather than resolved here.

**The statistic matters as much as the margin.** Using the maximum of the dwell
instead of the median of block maxima costs `1.8` points of the recovery
(`96.2%` against `98.0%`), because one spike then governs the whole window.

**A shorter dwell is not better.** `8` blocks of `30` evaluations, the same `240`
evaluations arranged more coarsely, recovers only `96.2%`, because the median is
taken over fewer, longer blocks and follows the spikes more.

The exact-quantile form used to explore the design, a `240`-sample ring with a
`p95`, gives the same numbers as the `12`-block form for `20x` the memory, so
the block form ships.

## Alternatives Considered

### Raise `CLASSIC_STARTUP_STRENGTH`

Rejected on measurement. It lifts ESP32 recall, but `0.75` is already the last
value that keeps the empty-room recordings silent; `0.78` produces the first
alarm. It acts on the same unrepresentative prefix.

### Cap the calibrated threshold at the session's quiet ceiling

Rejected. Same wall: at the margin that moves ESP32 the empty rooms alarm and S3
false positives breach the ceiling. Also prefix-bound.

### Refit the coefficients

Rejected. On the current corpus a refit loses on every chip and is worst exactly
where it is needed, `-2.3` points on ESP32.

### Recalibrate periodically from scratch

Rejected for now. A second full calibration pass would have to assume the window
it samples is idle, and the runtime has no way to know that. The settled-level
rule needs no such assumption, because motion keeps the level high on its own.

## Consequences

The last pair missing the recall target now clears it, and the worst per-chip
recall moves from `94.2%` to `97.7%` with no cost to false positives, the
weak-link slice, or the empty-room gate.

A room that becomes genuinely noisier after the threshold has come down cannot
push it back up. This is the exact mirror of the property that makes the rule
safe, and recovery needs a recalibration.

The firmware pays `CLASSIC_SETTLE_BLOCKS` floats and four counters per Classic
detector, `62` bytes, with no allocation and no work outside the evaluation
that already runs.

Both runtimes carry the same rule and the report parity gate covers it: the
Classic side of `tools/generate_performance_report.py` shows no drift.
