# ADR: set the non-regression margin from measured seed noise

- Status: Accepted
- Date: 2026-07-27

## Context

`paired_result_non_regression` ratchets every reserved replay against the deployed baseline. Its margin was one changed evaluation, which on a `685`-evaluation recording means two extra false-positive evaluations are enough to block any candidate.

That margin was never measured. It was chosen as the smallest quantity that still tolerates rounding, on the assumption that a model which is genuinely no worse will reproduce the baseline almost exactly.

The assumption is wrong. Promoting Coherence-7 required `--force-promote` because one holdout recording went from `1` to `7` false-positive evaluations out of `685` with no effective alarm anywhere; see [2026-07-27-add-the-lag-ratio-to-the-production-ml-feature-set.md](2026-07-27-add-the-lag-ratio-to-the-production-ml-feature-set.md), which flagged the margin as worth revisiting on its own evidence rather than as part of the promotion it blocked.

## Decision

The per-recording margins are set from measured seed-to-seed dispersion:

| metric | margin | measured spread |
| --- | --- | --- |
| `fp_rate` | `5` evaluations | `4` evaluations |
| `recall` | `1` evaluation | `0` evaluations |
| `effective_alarms` | `0`, unchanged | `0` |

## Measurement

`tools/analyze_seed_dispersion.py` reads the per-seed runs a seed search persists and reports, per reserved replay, the range of a gate metric across seeds of the same feature set on the same recordings. Only the seed differs, so the spread is the noise floor of weight initialization.

Across fifteen seeds, false-positive evaluations on the hardest normal-link replay ranged from `3` to `7`. Recall did not move at all. Effective alarms stayed at `0` on every replay in every run.

So the old margin sat at roughly a quarter of the dispersion of the quantity it policed, while the quantity it policed never reached an alarm. `fp_rate` takes the measured `4` plus one evaluation of headroom, because a maximum over fifteen samples understates a range. Recall keeps its single evaluation: nothing measured asks for more.

## What is deliberately not changed

**Effective alarms keep a zero margin.** Seed `1538882188` produced aggregates strictly better than the baseline — `maxFP` `3.86%` against `4.43%`, equal worst recall — while a holdout S3 replay went from `0` to `1` effective alarm. The alarm ratchet caught what the aggregates hid. Any margin there would have promoted it.

**The `low_rssi` exemption stays.** It was expected to dissolve once the margin reflected real dispersion, since weak links show the same effect only larger. That cannot be established: every `low_rssi` pair is in `train`, `holdout`, or `exclude`, and none in `selection`, so its dispersion can only be measured by contaminating training or by burning the holdout.

## Alternatives Considered

### Keep one evaluation and rely on `--force-promote`

Rejected. The flag is a deliberate, documented override. Needing it routinely for candidates that improve the corpus turns it into the normal path, and an override that is always used stops being a decision.

### Derive the margin per replay from that replay's own dispersion

Rejected for now. It fits the data better, but it needs a dispersion estimate per recording, which means re-measuring whenever the corpus changes, and it would give the noisiest recording the widest licence. A single constant, stated in evaluations and traceable to a measurement, is easier to argue with.

## Consequences

Candidates whose false positives move within seed noise on a recording are no longer blocked.

The Coherence-7 blocker is verified arithmetically, in the unit test that pins `3` to `7` evaluations out of `685` as passing and `10` as failing. It cannot be re-verified end to end: the baseline it was measured against has since been replaced by the model it produced, so retraining seed `20260519` now compares that model with itself and would pass under any margin. What that run does confirm is that training is deterministic at a fixed seed — the re-export reproduced the committed weights bit for bit, changing only the timestamp.

Seed `1538882188` is the informative end-to-end case, because it was blocked by one of each kind: `3` versus `0` evaluations out of `666` on a C5 holdout pair, and a new effective alarm on an S3 holdout replay. Re-run after the change, it reports the alarm alone and the jitter blocker is gone, which exercises both halves of the decision on one candidate: the margin stopped rejecting seed noise, and the alarm ratchet still rejects a new alarm on a candidate whose aggregates read `maxFP` `3.86%` against a baseline `4.43%`.

The margin is now a claim about the corpus, not about arithmetic. It should be re-measured when the corpus changes materially, in particular once the C5 and S3 replacement captures land, and re-derived rather than inherited if the feature set changes.
