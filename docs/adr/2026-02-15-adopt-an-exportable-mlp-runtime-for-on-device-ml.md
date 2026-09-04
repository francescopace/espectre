# ADR: adopt a portable MLP runtime and PyTorch training stack

- Status: Accepted
- Date: 2026-02-15
- Recorded: 2026-07-09 (retrospective)
- Updated: 2026-08-26

## Context

The ML detector must run on constrained devices in both the C++ firmware and MicroPython paths without a heavyweight inference runtime. Training, export, inference, and parity must therefore form one portable contract rather than two independently maintained implementations.

The original host pipeline still produced TensorFlow, TFLite, and scaler artifacts after the deployed runtimes had standardized on manual exported-weight inference. The training framework no longer needed to determine the device runtime format.

Small temporal CNN and TCN candidates were later screened against the same false-positive-first validation policy. They did not justify a second runtime family, while the MLP remained cheap to export, inspect, and reproduce.

Python and C++ also produced small decision drift when compilers contracted floating-point expressions differently. Cross-runtime parity requires control over the arithmetic path as well as shared weights.

## Decision

Use a small feed-forward MLP as the supported on-device ML runtime form:

- train and export one model through the PyTorch host-side pipeline;
- generate the Python and C++ weight artifacts from the same candidate;
- keep manual dense-layer inference in both runtimes, without a TFLite dependency;
- keep feature order, scaling, topology, biases, weights, and threshold metadata aligned across both generated artifacts;
- disable floating-point contraction for the ML inference translation unit so Python and C++ preserve the validated arithmetic order; and
- require the normal parity and generated-artifact gates before promotion.

PyTorch is a host implementation choice subordinate to the export contract. Stop producing TensorFlow, TFLite, or standalone scaler artifacts for the active production path. A future trainer may replace PyTorch without changing device inference if it reproduces the same portable artifacts and validation contract.

The generated weight files own the exact deployed run metadata. This ADR owns the trainer/runtime boundary and artifact-sharing contract, not one model seed or feature schema.

## Decision History

| Date | Direction | Resolution |
| --- | --- | --- |
| 2026-02-15 | Export a lightweight MLP and share its artifacts between Python and C++ | Adopted as one runtime contract |
| 2026-05-20 | Screen small temporal CNN and TCN replacements | Retained the MLP because the measured gains did not justify another runtime family |
| 2026-07-02 | Replace TensorFlow/Keras training with PyTorch while preserving the exported-weight format | Accepted; the host framework remains subordinate to the runtime artifact contract |
| 2026-07-26 | Allow compiler-default floating-point contraction | Rejected after measurable Python/C++ decision drift; contraction is disabled for inference |

## Alternatives Considered

### Deploy TFLite

Rejected. It adds a runtime dependency and integration surface without improving the current constrained deployment contract.

### Maintain independent models per runtime

Rejected. Independently trained or hand-transcribed artifacts would make parity, review, and reproduction unreliable.

### Keep TensorFlow/Keras and TFLite artifacts in the active workflow

Rejected. The deployed runtimes do not consume them, so they add dependencies and artifact maintenance without protecting a production contract.

### Add a second temporal runtime family

Rejected for the measured candidates. A new family remains possible only after host-side evidence justifies its export, memory, implementation, and parity cost.

## Consequences

Benefits:

- one trained model drives both production runtimes;
- inference stays small, inspectable, and frontend-independent; and
- the host trainer can evolve without changing the device runtime format; and
- parity failures expose real contract drift instead of artifact provenance ambiguity.

Trade-offs:

- production model families must fit the manual export contract; and
- training reproducibility depends on the maintained PyTorch path until another trainer is deliberately adopted; and
- compiler settings for the inference unit are part of correctness.

## Related

- [`2026-08-11-promote-channel-shape-trajectory-ml-features.md`](2026-08-11-promote-channel-shape-trajectory-ml-features.md)
- versioned changelog snapshot: `2.5.0:CHANGELOG.md`
- git commits: `3058c750`, `6e59e485`
