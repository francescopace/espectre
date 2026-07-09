# ADR: use pytorch as the host training stack

- Status: Accepted
- Date: 2026-07-02
- Recorded: 2026-07-09 (retrospective)

## Context

The project initially shipped the ML detector with a training/export pipeline
that still produced TFLite artifacts even though the deployed runtime had
already standardized on manual exported-weight inference. Over time that made
the host-side training stack heavier than the runtime contract actually needed.

The `v3` changelog explicitly records a later cleanup: the ML training stack was
migrated from TensorFlow/Keras to PyTorch, while preserving the same runtime
weight export contract for Python and C++.

## Decision

Use PyTorch as the host-side training stack for the ML detector.

Concretely:

- train the supported MLP path with PyTorch
- preserve the same exported runtime weights contract used by Python and C++
- stop treating TFLite/scaler artifacts as part of the active production path
- keep the host-side framework choice subordinate to the existing runtime export
  format

## Alternatives Considered

### Keep TensorFlow/Keras as the training stack

Rejected. The project no longer needed the older stack once the deployed runtime
had converged on exported weights and manual inference.

### Keep producing TFLite artifacts as part of the normal workflow

Rejected. Those artifacts were no longer the active runtime contract and added
maintenance cost without corresponding deployment value.

## Consequences

Benefits:

- the host-side ML workflow is lighter and more aligned with the deployed model
- the project keeps the same runtime exports while simplifying the trainer
- framework evolution on the host side stays decoupled from runtime inference

Trade-offs:

- training reproducibility now depends on the PyTorch path rather than the older
  stack
- future trainer changes must still preserve the exported runtime contract

## Related

- `docs/adr/2026-02-15-share-ml-model-artifacts-between-python-and-cpp.md`
- `docs/adr/2026-07-07-use-core-6-as-the-production-ml-feature-set.md`
- git commit: `ef9df5bb`
