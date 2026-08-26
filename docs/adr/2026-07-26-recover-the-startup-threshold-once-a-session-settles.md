# ADR: recover the startup threshold once a session settles

- Status: Accepted
- Date: 2026-07-26
- Updated: 2026-08-26

## Context

Lightweight calibrates its threshold from the opening of a sensing session. One ESP32 recording showed that an unusually noisy calibration prefix could leave the threshold too high after the room became quiet, reducing recall even though the underlying features still separated motion from idle.

Prefix-only interventions could not distinguish a genuinely noisy room from an unrepresentative opening. Raising startup strength or capping the threshold from the same prefix improved the failing recording only at operating points that weakened empty-room or weak-link safety. Refitting the feature coefficients did not address the observed threshold-placement failure.

The runtime needed later evidence that the session had settled without assuming that an arbitrary recalibration window was idle.

## Decision

Allow Lightweight to lower its live threshold after a sustained quiet dwell shows that the session is materially quieter than its startup prefix.

Partition evaluations into fixed blocks. Record the maximum metric logit in each block, retain a bounded ring of recent block maxima, and use their median as the settled session level. Add the configured safety margin and lower the live threshold only when that candidate is below the current value.

The rule has three constraints:

1. It is one-sided: it can lower the threshold but never raise it.
2. Motion holds the candidate up because active blocks contribute high maxima.
3. A median of block maxima prevents one spike or one quiet block from controlling the result.

`reset()`, `clear_buffer()`, `on_startup_calibration_begin()`, and `set_adaptive_threshold()` clear the accumulated evidence. A restarted, switched, or recalibrated detector must observe another complete dwell before lowering the threshold. Current dwell, block, and margin values are operational detector constants documented in `ALGORITHMS.md` and protected by replay and parity gates.

## Decision History

| Date | Direction | Resolution |
| --- | --- | --- |
| 2026-07-26 | Recover from an unrepresentative startup prefix with a bounded median-of-block-maxima rule | Accepted after paired, weak-link, and empty-room replay |
| 2026-08-10 | Revalidate the safety margin under millisecond-window native-cadence replay | Retuned the operating value without changing the rule |
| 2026-08-15 | Feed the same temporally admitted stream into calibration and evaluation | Retained the settled-session rule under fixed temporal admission |

## Alternatives Considered

### Increase startup calibration strength

Rejected. It still depends on the unrepresentative prefix and reached the empty-room safety boundary before recovering the failing session safely.

### Cap the calibrated threshold from prefix statistics

Rejected. It uses the same incomplete evidence and reproduced the false-positive trade-off.

### Refit Lightweight coefficients

Rejected. The measured failure was threshold placement, and the refit reduced recall elsewhere without resolving it safely.

### Recalibrate periodically from scratch

Rejected. A new calibration interval would have to assume that the sampled window is idle. The settled rule lets motion prevent the threshold change instead.

### Track an exact long-window quantile

Rejected. The block-maxima form reproduced the useful decision with bounded, much smaller state.

## Consequences

- a noisy startup prefix no longer fixes an unnecessarily high threshold for the whole session;
- motion cannot cause the rule to lower the threshold;
- a room that becomes noisier after recovery cannot raise the threshold automatically and requires recalibration;
- the detector adds a small fixed block-history state with no allocation; and
- C++ and Python must keep the rule, reset boundaries, constants, and replay behavior aligned.

## Related

- [`../ALGORITHMS.md`](../ALGORITHMS.md)
- [`../performance/README.md`](../performance/README.md)
- [`2026-08-15-use-fixed-temporal-csi-admission.md`](2026-08-15-use-fixed-temporal-csi-admission.md)
- [`2026-03-08-use-host-side-validation-gates-for-detector-promotion.md`](2026-03-08-use-host-side-validation-gates-for-detector-promotion.md)
