# ADR: adopt gated startup threshold calibration for classic detector

- Status: Accepted
- Date: 2026-07-07
- Recorded: 2026-07-09 (retrospective)

## Context

Once `l1_delta` had been promoted as the primary non-ML metric, the remaining
open question was not the metric itself, but how `ClassicDetector` should place
its startup threshold in real deployments. Clean paired datasets supported the
simple policy `max(calibration) x 1.1`, but the later experiments showed that
startup contamination could push the threshold to motion level and fail recall
closed.

The repo history shows a clear progression:

- a same-day threshold sweep confirmed that clean sessions still favor
  `max x 1.1`, while quantile-only replacements and online adapters introduce
  their own false-positive costs
- a contaminated-calibration sweep then promoted a rolling-chunk consistency
  gate with calibration extension
- a follow-up quiet-tail rescue amended that gate after a long quiet C6 run
  exposed a case where extension could lock onto an accepted but too-low quiet
  tail
- a later post-startup drift-tracker re-test confirmed that the decaying-peak
  line helps only true startup-overshoot cases and regresses the long-quiet
  gate on the sessions that drove the production decision

## Decision

Adopt a gated startup threshold calibration policy for `ClassicDetector`.

The accepted policy is:

- keep the threshold formula `max(accepted_window) x 1.1`
- validate startup calibration with a rolling chunk gate before accepting the
  window
- use chunk maxima with spread and floor-anchor checks to reject contaminated
  startup windows
- extend calibration one chunk at a time when the gate rejects the current
  window
- on accepted extension, rescue one discarded quiet-tail chunk when it stays
  inside the floor-anchor band
- on budget exhaustion, fall back conservatively to `median(ring) x 1.1`
- do not promote online threshold adaptation into production

Promoted parameters:

- `k = 6` chunks
- spread gate: `max(ring) <= 1.10 x median(ring)`
- floor anchor: `median(ring) <= 1.5 x min_chunk_ever`
- extension cap: `+2000` packets

## Alternatives Considered

### Keep ungated static `max x 1.1`

Rejected. It is strong on clean startup, but collapses badly when calibration is
contaminated by real motion.

### Replace `max` with startup quantiles such as `p95` or `p98`

Rejected. Lower quantiles improve contaminated startup tolerance, but pay too
much quiet false-positive cost on already clean sessions.

### Promote online decaying-peak threshold recovery

Rejected. It remained interesting as a possible future drift tracker, but it was
not the best production startup-repair policy and was later explicitly deferred.

### Use spread-only or chunk-mean gates

Rejected. They fail on realistic contamination modes, especially homogeneous or
weak-motion cases that the floor anchor and quiet-tail rescue handle better.

## Consequences

Benefits:

- `ClassicDetector` keeps the clean-session behavior of `max x 1.1` where that
  policy already works
- startup thresholding becomes robust to contaminated calibration windows
- the threshold policy is explicit and shared across Micro-ESPectre and C++

Trade-offs:

- startup calibration is more complex than a plain static threshold rule
- contaminated startup can extend calibration time by several extra chunks
- a small amount of weak sub-anchor motion may be traded away to avoid persistent
  quiet-room false positives

## Related

- `docs/adr/2026-07-07-use-core-6-as-the-production-ml-feature-set.md`
- `docs/adr/2026-07-08-promote-classic-detector-and-retire-legacy-baselines.md`
- git commits: `5b871159`, `76d86aa2`
