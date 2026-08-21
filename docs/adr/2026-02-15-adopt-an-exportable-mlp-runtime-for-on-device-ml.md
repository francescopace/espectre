# ADR: adopt a portable shared MLP runtime for on-device ML

- Status: Accepted
- Date: 2026-02-15
- Recorded: 2026-07-09 (retrospective)
- Updated: 2026-08-12

## Context

The ML detector must run on constrained devices in both the C++ firmware and MicroPython paths without a heavyweight inference runtime. Training, export, inference, and parity must therefore form one portable contract rather than two independently maintained implementations.

Small temporal CNN and TCN candidates were later screened against the same false-positive-first validation policy. They did not justify a second runtime family, while the MLP remained cheap to export, inspect, and reproduce.

Python and C++ also produced small decision drift when compilers contracted floating-point expressions differently. Cross-runtime parity requires control over the arithmetic path as well as shared weights.

## Decision

Use a small feed-forward MLP as the supported on-device ML runtime form:

- train and export one model through the host-side pipeline;
- generate the Python and C++ weight artifacts from the same candidate;
- keep manual dense-layer inference in both runtimes, without a TFLite dependency;
- keep feature order, scaling, topology, biases, weights, and threshold metadata aligned across both generated artifacts;
- disable floating-point contraction for the ML inference translation unit so Python and C++ preserve the validated arithmetic order; and
- require the normal parity and generated-artifact gates before promotion.

The generated weight files own the exact deployed run metadata. This ADR owns the portable runtime and artifact-sharing contract, not one model seed or feature schema.

## Decision History

| Date | Direction | Resolution |
| --- | --- | --- |
| 2026-02-15 | Export a lightweight MLP and share its artifacts between Python and C++ | Adopted as one runtime contract |
| 2026-05-20 | Screen small temporal CNN and TCN replacements | Retained the MLP because the measured gains did not justify another runtime family |
| 2026-07-26 | Allow compiler-default floating-point contraction | Rejected after measurable Python/C++ decision drift; contraction is disabled for inference |

## Alternatives Considered

### Deploy TFLite

Rejected. It adds a runtime dependency and integration surface without improving the current constrained deployment contract.

### Maintain independent models per runtime

Rejected. Independently trained or hand-transcribed artifacts would make parity, review, and reproduction unreliable.

### Add a second temporal runtime family

Rejected for the measured candidates. A new family remains possible only after host-side evidence justifies its export, memory, implementation, and parity cost.

## Consequences

Benefits:

- one trained model drives both production runtimes;
- inference stays small, inspectable, and frontend-independent; and
- parity failures expose real contract drift instead of artifact provenance ambiguity.

Trade-offs:

- production model families must fit the manual export contract; and
- compiler settings for the inference unit are part of correctness.

## Related

- [`2026-07-02-use-pytorch-as-the-host-training-stack.md`](2026-07-02-use-pytorch-as-the-host-training-stack.md)
- [`2026-08-11-promote-channel-shape-trajectory-ml-features.md`](2026-08-11-promote-channel-shape-trajectory-ml-features.md)
- versioned changelog snapshot: `2.5.0:CHANGELOG.md`
- git commits: `3058c750`, `6e59e485`
