# ADR: use host-side validation gates for detector promotion

- Status: Accepted
- Date: 2026-03-08
- Recorded: 2026-07-09 (retrospective)
- Updated: 2026-10-05

## Context

A single aggregate metric, or grouped CV alone, can hide the failures users actually see: false alarms in one room, collapse on a weak link, trouble with a different packet rate. We need one promotion policy for detectors, features, and models, but running the full release suite on every training trial would make exploration far too slow.

## Decision

Promotion goes through layered host-side gates:

1. Compare candidates with grouped CV and selection replays. The holdout stays sealed.
2. Gate every recording on its own, paired and quiet alike. Chip averages are not enough.
3. Empty rooms are the only strict no-motion truth. High Accuracy must raise zero effective alarms on short empty recordings and at most two on catalog-marked long ones, both in normal replay and at 70% occupancy. The raw FP ceiling still applies and every alarm is reported. Lightweight may raise one effective alarm per empty recording (a single four-hit debounce burst); two on one short file is a defect. Static-presence captures are not no-motion truth, because a person sitting quietly still makes small movements, so they get a small explicit alarm budget.
4. Real low-RSSI recordings stay visible under their own stress policy instead of disappearing into averages.
5. The selected candidate, and any published performance claim, must also pass packet-rate, long-recording, generated-artifact, and Python/C++ parity checks.
6. Experiment commands never overwrite artifacts. Promotion is an explicit step.

Trainer ranking can use the narrower selection gates to iterate quickly, but passing them does not make a candidate production-ready.

## Decision history

| Date | Direction | Resolution |
| --- | --- | --- |
| 2026-03-08 | Promote detectors on host-side real-data gates | Accepted |
| 2026-07-17 | Long-recording gates inside every training trial | Moved to final promotion and published performance |
| 2026-07-23 | Static-presence captures as strict FP truth | Replaced by empty-room zero-alarm gates plus a static-presence budget |
| 2026-08-16 | Zero alarms for Lightweight on empty rooms | Relaxed to one per recording: at 70% occupancy a four-eval burst remained that no feature or coefficient change removed without losing Lightweight's resource advantage over High Accuracy |
| 2026-10-05 | Zero alarms for High Accuracy on long empty recordings | Relaxed to two per long recording to keep the recall gained on the larger corpus; short files still need zero, raw FP limits unchanged |

## Alternatives considered

- **Promote on grouped CV or one benchmark family.** Aggregate wins hide per-recording false positives, weak-link collapse, and runtime filtering bugs.
- **Run the full suite for every seed.** Too slow. It belongs to finalist promotion and releases.
- **Let each runtime define its own gates.** Detector behavior is a shared Python/C++ contract.

## Consequences

Candidate search stays fast, and production promotion still rests on evidence. Empty-room, weak-link, timing, long-recording, and parity failures each stay visible on their own. An exploratory result cannot be called production-ready until the full gates have run.

## Related

- [`2026-06-30-separate-ml-training-data-from-promotion-replays.md`](2026-06-30-separate-ml-training-data-from-promotion-replays.md)
- [`2026-07-08-define-the-lightweight-detector.md`](2026-07-08-define-the-lightweight-detector.md)
- [`2026-08-11-promote-channel-shape-trajectory-ml-features.md`](2026-08-11-promote-channel-shape-trajectory-ml-features.md)
- `2.6.0:CHANGELOG.md`
- Commits `a8b0dc4f` (long gates out of trials), `d6546146` (static-presence budget), `aee0b969` (Lightweight budget), `1ebbfb0a` (High Accuracy long-recording budget)
