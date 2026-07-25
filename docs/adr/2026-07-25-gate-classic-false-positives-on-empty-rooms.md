# Gate Classic false positives on empty rooms

- Status: accepted
- Date: 2026-07-25

## Context

The Classic detector was held to `effective_alarms == 0` on the `static_presence`
baselines of every real low-RSSI pair, in both `test_low_rssi.cpp` and
`test_low_rssi_classic.py`. Six of the nine pairs failed it. The gap had been
open long enough to be recorded as a known limit, framed as weak-link noise and
tracked as feature-side work, with the dormant L1 noise-blend safeguard as the
leading suspect.

Measurement contradicted every part of that framing.

**The L1 term is not what moves.** Decomposing the logit into its two
contributions and comparing alarm evaluations against quiet ones on the same
recording:

| pair | L1 term shift | autocorrelation term shift |
| --- | --- | --- |
| C3 holdout | 0.04 | 4.20 |
| C5 holdout | 0.10 | 5.81 |
| C5 train | 0.16 | 6.71 |
| C5 train | 0.07 | 7.22 |
| C6 holdout | 0.15 | 7.93 |
| C6 train | 0.20 | 5.25 |
| S3 holdout | 0.21 | 5.73 |
| S3 train | 0.24 | 4.18 |

The safeguard sits at blend `0.000` on every pair that alarms: it activates above
an L1 floor of `0.0996` and those eight pairs measure `0.030` to `0.089`.
Activating it would not have helped, because it acts on the term that barely
moves. This says nothing about its value elsewhere; see the consequences below.

**It is not a weak-link phenomenon.** Extending the analysis from the nine
low-RSSI pairs to all `29` real pairs put the worst offenders on the strongest
links: `10` alarms at `-42.0 dBm`, `7` at `-41.1 dBm`, `6` at `-41.4 dBm`. The
weakest capture in the corpus, at `-84.3 dBm`, produced none. The correlation
with RSSI is absent. The gap looked weak-link-specific only because the test
that asserted it only ever loaded weak-link pairs.

**It is not detector noise.** The `12` empty-room recordings raise no alarm at
all, under the same calibration and replay protocol:

| stream | recordings | effective alarms | longest raw motion run |
| --- | --- | --- | --- |
| empty | 12 | 0 | 3 evaluations |
| static presence | 29 | 54 | 7 evaluations |

The empty recordings are shorter (`12k` packets against `17.8k`), but scaling for
length would still predict about `15` alarms where there are none.

`static_presence` means a stationary person is in the room. A stationary person
breathes, shifts, and moves a hand, and those are real channel perturbations.
The alarm windows are not separable from real motion by any window statistic
measured: turbulence autocorrelation reaches `0.31-0.65` against `0.46-0.80` for
motion, and on S3 holdout the turbulence standard deviation during alarms
(`0.0369`) exceeds the motion value (`0.0317`). They are not a different
phenomenon from motion. They are a weaker instance of it.

The only axis that separates them is persistence. Alarm runs reach `4-7`
evaluations; real motion runs have a median of `9` to `349`.

The project was also already inconsistent here: the ML detector's false-positive
gate uses the empty-room recordings, while Classic was gated on static presence,
the stricter and less defensible of the two.

## Decision

**Empty-room recordings are the false-positive ground truth for both detectors.**
`effective_alarms == 0` is asserted there, in `test_empty_rooms.cpp` and
`TestPerformanceMetrics::test_classic_empty_false_positive_rate`, which pass on
all `12` recordings with a raw per-evaluation rate under `6%`.

**Static-presence baselines carry a sanity bound, not a false-positive gate.**
The motion share on them is bounded at `12%` against a corpus maximum of `10.6%`.
The bound guards against drift; it does not encode a stationary occupant's
micro-motion as detector error.

`replay_idle_stream` in `tools/lib/performance_report.py` is the single replay
both detectors' empty-room gates run through, so the two cannot drift apart.

## Alternatives

**Raise `MOTION_ON_HITS` from `4` to `6`.** This works on the measurement: total
alarms across the corpus fall from `54` to `3`, because the alarm runs sit right
at the current four-hit requirement. It was rejected because it is a product
change wearing a test fix. Publication latency would go from `1 s` to `1.5 s` for
every deployment, and genuinely brief movement would be suppressed: on the
C3 holdout pair, whose motion is intermittent, debounced coverage drops from
`86.8%` to `79.9%`. The knob remains available for deployments that prefer
quietness to responsiveness.

**Activate the L1 noise-blend safeguard by lowering its threshold.** Rejected by
the measurement above: the L1 term contributes `0.04-0.24` logit units where the
autocorrelation term contributes `4.20-7.93`.

**Find a feature that separates micro-motion from gross motion.** This is the
"Presence vs Empty detection" roadmap item rather than a fix to this gate, and
the measurements here suggest it is hard: the two classes overlap on every
window statistic tried, so it needs an axis the current window does not carry,
such as cross-window persistence or a wider selected band.

**Keep asserting zero alarms on static presence.** Rejected because it asks the
detector to be blind to a real person while the roadmap separately asks for
exactly that sensitivity.

## Consequences

The C++ suite goes from `24/25` to `27/27`, and the Python low-RSSI Classic
suite from `2/9` to `9/9`. Three static-presence false-positive failures in
`test_validation_real_data.py` also clear, taking the real-data failures from
`13` to `10`. The remaining `10` are all recall shortfalls, a separate and
still-open gap.

The C++ test harness gained empty-room discovery and loading
(`csi_test_data.h`), so the C++ side can now be gated on the same ground truth
as the Python side. Both report identical numbers on the same recording, which
is the cross-check that the two replays agree.

The L1 noise-blend safeguard stays in the code, and removing it was tried and
reverted the same day. Being inert on the eight pairs that alarm does not make it
dead: the ninth weak-link pair has a quiet L1 floor of `0.2719`, saturates the
blend, and depends on it. Deleting the safeguard sent that pair's startup
threshold from `0.789` to `0.984` and its recall from `97.2%` to `77.5%`. It is a
rarely-taken branch that carries the pathological-link case, not an unused one,
and the earlier reading of it as dormant came from measuring only the sessions
where it does nothing.

The known-limits section of ALGORITHMS.md no longer describes weak-link false
alarms as an open Classic gap, because they were not false, not weak-link
specific, and not a gap.
