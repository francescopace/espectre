# ADR: adopt a portable MLP runtime and PyTorch training stack

- Status: Accepted
- Date: 2026-02-15
- Recorded: 2026-07-09 (retrospective)
- Updated: 2026-08-26

## Context

The ML detector runs on small devices, in both the C++ firmware and MicroPython, with no room for a heavy inference runtime. Training, export, and inference have to share one format, or the two runtimes drift apart.

## Decision

The on-device model is a small feed-forward MLP with hand-written dense-layer inference in both runtimes. No TFLite.

- One PyTorch pipeline trains the model and generates both the Python and C++ weight files from the same candidate.
- Feature order, scaling, topology, weights, and threshold travel together in those files.
- The C++ inference code is compiled with `fp-contract=off`, so it does the arithmetic in the same order as Python.
- Promotion requires the usual parity and generated-artifact checks.

PyTorch is just the current trainer. It can be replaced as long as the exported files and validation stay the same. The weight files own the exact model metadata; this ADR only fixes the boundary between trainer and runtime.

## Decision history

| Date | Direction | Resolution |
| --- | --- | --- |
| 2026-02-15 | Lightweight MLP shared between Python and C++ | Adopted |
| 2026-05-20 | Tiny CNN and TCN candidates | Kept the MLP: gains too small for a second runtime family |
| 2026-07-02 | Move training from TensorFlow/Keras to PyTorch | Accepted; export format unchanged, TensorFlow, TFLite, and scaler artifacts dropped |
| 2026-07-27 | Compiler-default floating-point contraction | Rejected: it caused Python/C++ decision drift |

## Alternatives considered

- **TFLite on device.** Extra dependency and integration work for no gain.
- **Separate models per runtime.** Parity, review, and reproduction would become unreliable.
- **Keep TensorFlow and TFLite artifacts around.** Nothing on the device reads them.
- **A temporal model family.** Possible later, but only if host results justify the export, memory, and parity cost.

## Consequences

One trained model drives both runtimes, inference stays small and readable, and the trainer can change without touching the device. A parity failure means real drift, not a mix-up of artifacts. The limits: any production model must fit the manual export format, and the compiler flags for the inference code are part of correctness.

## Related

- [`2026-08-11-promote-channel-shape-trajectory-ml-features.md`](2026-08-11-promote-channel-shape-trajectory-ml-features.md)
- `2.5.0:CHANGELOG.md`
- Commits `6e59e485` and `3058c750` (first MLP), `d7c02559` (CNN/TCN sweep), `3889ef65` (PyTorch), `90fccc26` (`fp-contract=off`)
