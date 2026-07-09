# ADR: retain mlp over small temporal models

- Status: Accepted
- Date: 2026-05-20
- Recorded: 2026-07-09 (retrospective)

## Context

Once the repo had a workable on-device MLP path, an obvious question was
whether small temporal models should replace it. The 2026-05-20 tiny CNN / TCN
screen compared the then-production `mlp-9` against lightweight temporal
candidates on the same FP-first long-recording ranking used for ML promotion.

That campaign did not show a clear enough win for the temporal candidates:

- `cnn-b` occasionally matched the MLP on total false positives, but it was less
  stable and materially worse on the weakest chip and seed combinations
- `tcn-a` was not competitive enough to displace the baseline during screening
- neither candidate justified adding model-family complexity to the runtime and
  export story

## Decision

Retain the exportable MLP as the supported production-oriented on-device ML
family, and do not promote the screened tiny CNN or TCN candidates.

Treat temporal models as research directions only unless a later candidate
produces a clear FP-first promotion win over the MLP line.

## Alternatives Considered

### Promote the tiny CNN

Rejected. Its occasional local wins did not outweigh its weaker worst-case
stability on the long-recording gate.

### Continue with the screened tiny TCN

Rejected. It did not show enough competitiveness even before the final
promotion comparison.

## Consequences

Benefits:

- the runtime ML contract stays aligned with the lightweight exported-weight MLP
- model-family choice remains easy to share across Python and C++
- future temporal work has a clear historical baseline to beat

Trade-offs:

- some potentially interesting temporal directions remain deferred
- the project accepts the simpler MLP path until a temporal model produces a
  clearer deployment-facing win

## Related

- `docs/adr/2026-02-15-adopt-an-exportable-mlp-runtime-for-on-device-ml.md`
- `docs/adr/2026-06-29-record-historical-ml-baseline-evolution-before-core-6.md`
- `docs/adr/2026-07-07-use-core-6-as-the-production-ml-feature-set.md`
