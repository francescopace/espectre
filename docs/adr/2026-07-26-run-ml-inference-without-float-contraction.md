# ADR: run ML inference without float contraction

- Status: Accepted
- Date: 2026-07-26

## Context

The performance report verifies that the Python and C++ runtimes decide the same
way before it publishes anything. Classic passed that gate cleanly while ML
drifted on four chips at once, by `0.06` to `0.72` points on recall, precision,
false-positive rate, and F1, plus one long-quiet alarm that C++ raised and
Python did not.

The deltas were small enough to look like accumulated rounding and stubborn
enough to survive every structural explanation. Three measurements ruled those
out.

**The evaluation cadence is in lockstep.** On the worst pair both runtimes count
exactly `700` baseline and `349` motion evaluations, at the same packet offsets,
with no contaminated resets. Nothing about windowing, warmup, or timing differs.

**The features are identical to float32 precision.** Dumped side by side, all
six ML features agree to a relative `1e-6`, which is the width of a `float`.

**The gap needed a shift `30x` larger than that.** Reaching the C++ result from
the Python probabilities takes about `+0.28` of logit. Feature noise at `1e-6`
moves nothing: a `+0.01` shift flips no decisions at all.

So the divergence was created after the features, inside inference.

The cause showed itself by accident. Adding a debug `fprintf` immediately after
`predict()` returned changed the results on ten of the twenty-eight paired
replays, and moved every one of them onto the Python value. A print statement
that reads a value cannot change how it was computed unless the computation was
never fixed in the first place.

`MLDetector::predict` accumulates in `float`:

```cpp
float val = biases[j];
for (int i = 0; i < in_size; i++) {
    val += current[i] * weights[i * out_size + j];
}
```

Compilers are allowed to contract each multiply-add into a single FMA, which
skips the rounding of the intermediate product. Building with
`-ffp-contract=off` reproduces the debug-print numbers exactly on all
twenty-eight pairs, which confirms contraction as the mechanism.

The effect survives into decisions because the MLP output feeds a threshold. On
recordings whose probabilities sit near `0.5` the rounding difference decides
whole evaluations: the worst pair moved `3.2` points of recall.

## Decision

ML inference runs without floating-point contraction, disabled at the source so
every consumer inherits it.

`predict()` carries `#pragma clang fp contract(off)` and, for GCC, a
`push_options`/`pop_options` pair around the function with
`optimize("fp-contract=off")`.

The alternative was a build flag. It was rejected because the ML core is
compiled by ESP-IDF, PlatformIO, and the host CMake tests, and a flag would have
to be added to each of them and kept there. A build that missed it would produce
a detector that decides differently from the one the corpus was measured on,
silently. Putting it next to the loop it protects means the arithmetic travels
with the code.

## Consequences

The report parity gate passes on ML. Both runtimes now decide identically on
every pair in the corpus.

**The published ML numbers move slightly down**, because contraction was
inflating them: ten pairs shift, and the shift is one-sided because every
divergent decision the contracted build made was a MOTION it would not otherwise
have reached. This is a correction, not a regression. The uncontracted result is
the one the Python reference produces, and it is the one that reproduces
independently of compiler and surrounding code.

**Contraction was also a portability hazard**, which is the more important half.
The host and the target need not agree on whether the FPU offers a fused
multiply-add or whether the optimizer chooses it, so before this the firmware
could have decided differently from every number the project had measured. That
class of divergence is now closed for ML.

Classic was never affected. Its metric is two features through a handful of
multiply-adds, too short a chain for contraction to move a decision, which is
why its side of the gate stayed clean throughout.

The cost is a small amount of inference throughput on hardware with a fused
multiply-add. The MLP is six inputs wide and runs once per evaluation tick, so
the cost does not register against the evaluation interval.
