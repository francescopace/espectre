# ADR: use a fixed twelve-tone band selected from channel coherence

- Status: Accepted
- Date: 2026-07-25
- Updated: 2026-08-12

## Context

ESPectre originally selected subcarriers at runtime with NBVI. That reduced manual configuration but added startup calibration, memory, persistence, and cross-runtime complexity. The production path later moved to one fixed shared band, but its first twelve-tone layout was derived under an incorrect assumption that every chip delivered HT20 CSI with DC in bin 32.

Classic-MAC parts deliver Espressif's native `0~31, -32~-1` order, while Wi-Fi 6 parts deliver a centered order. After both layouts were normalized to the same centered grid, the replacement band could be selected from channel physics rather than detector scores on one corpus.

Quiet fluctuation is nearly independent tone to tone, while motion remains coherent over roughly ten subcarriers. Span therefore buys independent channel views; increasing tone density mostly adds correlated measurements.

## Decision

Use one fixed shared twelve-tone HT20 band in both production runtimes:

```text
DEFAULT_SUBCARRIERS = (4, 8, 13, 18, 23, 28, 36, 41, 46, 51, 56, 60)
```

These centered-grid indices represent subcarriers `+/-4`, `+/-9`, `+/-14`, `+/-19`, `+/-24`, and `+/-28`.

- Normalize recognized HT20 payload layouts before band extraction.
- Keep the band fixed across sessions, chips, frontends, Classic, and ML.
- Select for full usable-band span and independent looks, not per-dataset detector score.
- Keep twelve single-bin samples as the common amplitude path.
- Do not average adjacent subcarriers into every selected tone. Localized aggregation may exist only as an explicit feature-specific path, such as the production ML aggregated-IQR input.

## Decision History

| Date | Direction | Resolution |
| --- | --- | --- |
| 2025-12-03 | Select subcarriers at runtime with NBVI | Replaced by a fixed shared production band |
| 2026-06-09 | Remove runtime NBVI and use fixed shared subcarriers | Retained as the current operating model |
| 2026-07-20 | Keep the first twelve-tone layout | Replaced after correcting cross-chip HT20 bin ordering |
| 2026-07-25 | Select the full-span twelve-tone band from measured coherence | Accepted |
| 2026-08-05 | Average adjacent bins into the common selected-tone path | Rejected; only feature-local aggregation is allowed |

## Validation

Measured channel statistics showed the selected band spanning all 56 usable HT20 subcarriers and preserving about `3.55` independent looks, compared with about `2.44` for the previous layout. A later refit-based count sweep confirmed that denser bands added correlated information and degraded the high-rate stress capture even when chip aggregates appeared attractive.

The count therefore remains twelve on two independent grounds:

- channel coherence saturates the number of independent looks; and
- end-to-end replay does not justify the extra runtime cost or reduced high-rate dynamic range of denser bands.

Adjacent-bin averaging on the common path was also rejected after joint replay did not justify changing all Classic and ML amplitude inputs. The later ML aggregated-IQR feature is intentionally isolated and does not redefine this shared band.

## Alternatives Considered

### Restore runtime NBVI

Rejected. Its adaptive search does not justify the calibration, memory, persistence, and parity cost for the current product.

### Select the best detector score on the current corpus

Rejected. It would entangle the sensing geometry with one dataset and risk selecting a recording-specific coincidence.

### Increase the shared tone count

Rejected. Independent looks saturate, while tighter spacing raises correlation and hurts measured high-rate behavior.

### Aggregate adjacent bins everywhere

Rejected. It would silently redefine every consumer of the selected band. Feature-specific aggregation must be independently measured and explicitly named.

## Consequences

- Startup no longer performs environment-specific subcarrier search.
- Python and C++ share one normalized grid and one band definition.
- The band spans the usable HT20 range consistently across supported chip families.
- New band, count, or aggregation proposals require a refit, production replay, and cross-runtime parity evidence.

## Related

- [`2026-07-23-adopt-classifier-first-ht20-sensing-contract.md`](2026-07-23-adopt-classifier-first-ht20-sensing-contract.md)
- [`2026-07-04-keep-agc-active-and-standardize-cv-normalization.md`](2026-07-04-keep-agc-active-and-standardize-cv-normalization.md)
- [`../ALGORITHMS.md`](../ALGORITHMS.md)
- [`../FEATURES.md`](../FEATURES.md)
