# ADR: reject detector-guided sample weighting as the default ml baseline policy

- Status: Accepted
- Date: 2026-07-07
- Recorded: 2026-07-09 (retrospective)

## Context

The repo tested detector-guided sample weighting twice as a way to improve the
ML baseline without changing the deployed runtime contract.

First, the 2026-07-03 MVS-guided weighting work showed that MVS could help as a
hard-negative mining signal, but it also showed the central risk: using a weak
detector as a broad training guide can import that detector's quiet-spike bias
into the ML decision boundary.

Later, the 2026-07-07 L1-delta-guided weighting re-ran the idea with a better
support detector. It produced the same strategic result: grouped-CV or
hard-chip recall could improve a little, but the long-quiet false-positive gate
did not improve safely enough to justify promotion.

Together these experiments established a durable rule. Detector-guided
weighting is useful analysis tooling, but it is not the default production
training policy when the promotion rule is long-quiet false-positive
non-regression.

## Decision

Keep unweighted training as the production ML baseline policy.

Concretely:

- do not use detector-guided sample weighting as the default training mode
- require any future guided-weighting candidate to beat the same paired
  promotion gates as an unweighted baseline

Follow-up (2026-07-17): `--sample-weight-mode` and the L1-guided / hard-negative
weighting paths were removed from `tools/train_ml_model.py`. The trainer now
always starts from uniform sample weights (optional `--positive-chip-boost`
remains).

## Alternatives Considered

### Use MVS-guided weighting as the default policy

Rejected. It could reduce some long-run counts in selected runs, but it also
carried the risk of importing MVS bias and was later superseded by the clean
AGC-active reset.

### Use L1-delta-guided weighting as the default policy

Rejected. It improved some offline metrics, but it still failed the repo's
false-positive-first promotion standard.

## Consequences

Benefits:

- the default baseline remains easier to interpret and compare across retrains
- detector guidance stays an opt-in experiment instead of hidden default bias
- long-quiet false-positive regressions remain the governing promotion blocker

Trade-offs:

- some small recall or grouped-CV gains are intentionally left on the table
- future weighting experiments must re-justify themselves from a strong
  unweighted control, not from historical momentum

## Related

- `docs/adr/2026-03-08-use-host-side-validation-gates-for-detector-promotion.md`
- `docs/adr/2026-07-04-keep-agc-active-and-standardize-cv-normalization.md`
- `docs/adr/2026-07-07-use-core-6-as-the-production-ml-feature-set.md`
