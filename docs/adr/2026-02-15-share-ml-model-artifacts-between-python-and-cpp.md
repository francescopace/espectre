# ADR: share ml model artifacts between python and c++

- Status: Accepted
- Date: 2026-02-15
- Recorded: 2026-07-09 (retrospective)

## Context

The `2.5.0` release marked the first explicit project commitment to an on-device
ML detector available in both the C++ firmware path and the Python
Micro-ESPectre path. That release did not frame ML only as an experiment; it
also introduced a training pipeline that exports artifacts for both runtimes.

Later changes refined the trainer, feature sets, and frameworks, but the core
contract remained: one host-side training flow produces runtime artifacts that
must stay aligned across Python and C++.

## Decision

Use a shared host-side training and export flow to produce runtime ML artifacts
for both Python and C++.

Concretely:

- train models on the host side
- export runtime weights for Python and C++
- keep the inference behavior aligned across both runtime implementations
- treat exported artifacts as the integration contract between training and
  deployment

## Alternatives Considered

### Keep ML only on one runtime path first

Rejected. The project wanted detector parity across its Python and C++ motion
detection stacks.

### Re-implement or hand-tune model weights independently per runtime

Rejected. That would make parity harder to maintain and would weaken confidence
in validation results across stacks.

## Consequences

Benefits:

- training and deployment gained a clear contract
- Python and C++ detectors could be validated against the same learned model
- later improvements to the trainer could preserve the same runtime-export model

Trade-offs:

- training-side changes now have to preserve runtime compatibility
- feature extraction and inference parity became an ongoing maintenance
  responsibility

## Related

- versioned changelog snapshot: `2.5.0:CHANGELOG.md`
- `docs/adr/2026-07-07-use-core-6-as-the-production-ml-feature-set.md`
- git commit: `6e59e485`
