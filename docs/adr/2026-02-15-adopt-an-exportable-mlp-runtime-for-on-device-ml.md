# ADR: adopt an exportable mlp runtime for on-device ml

- Status: Accepted
- Date: 2026-02-15
- Recorded: 2026-07-09 (retrospective)

## Context

When the ML detector became a real project direction, the repo needed a runtime
model form that could run on-device in both the ESPHome/C++ path and the
Micro-ESPectre/Python path without depending on a heavyweight inference
runtime.

The `2.5.0` release changelog makes that choice explicit: the ML detector ships
with manual MLP inference, no TFLite dependency, and a host-side workflow that
exports runtime weights. That is a durable model choice, not just an
implementation detail.

## Decision

Adopt a small exportable MLP as the supported on-device ML runtime form.

Concretely:

- use a feed-forward MLP that can be exported as plain weights
- keep runtime inference manual and lightweight in both Python and C++
- avoid a TFLite dependency in the deployed runtime path
- treat exportable dense layers as the runtime contract for the ML detector

## Alternatives Considered

### Depend on TFLite in the deployed runtime

Rejected. The project chose a lighter runtime path with fewer integration and
deployment constraints.

### Use a more complex runtime model family immediately

Rejected. The first production ML path favored a model that could be exported,
reviewed, and reproduced consistently across both runtimes.

## Consequences

Benefits:

- the runtime ML path stays lightweight and portable
- Python and C++ can share the same inference shape
- exported weights remain inspectable and easy to regenerate

Trade-offs:

- model architecture choices must stay compatible with the manual runtime path
- some model families are less attractive if they complicate export or runtime
  inference

## Related

- versioned changelog snapshot: `2.5.0:CHANGELOG.md`
- `docs/adr/2026-02-15-share-ml-model-artifacts-between-python-and-cpp.md`
- `docs/adr/2026-07-07-use-core-6-as-the-production-ml-feature-set.md`
- git commits: `3058c750`, `6e59e485`
