# ADR: use a fixed twelve-tone band selected from channel coherence

- Status: Accepted
- Date: 2026-07-25
- Updated: 2026-08-12

## Context

ESPectre first picked subcarriers at runtime with NBVI. That saved manual setup but cost startup calibration, memory, persistence, and Python/C++ complexity. We then moved to one fixed band, but its first twelve-tone layout assumed every chip delivered HT20 CSI with DC in bin 32, which was wrong.

Classic-MAC chips deliver Espressif's native `0~31, -32~-1` order; Wi-Fi 6 chips deliver a centered order. Once both were normalized to the same centered grid, the band could be chosen from channel physics instead of detector scores on one corpus.

Quiet fluctuation is almost independent from tone to tone, while motion stays coherent over about ten subcarriers. So spreading tones wide buys independent views of the channel; packing them closer mostly adds correlated measurements.

## Decision

Both runtimes use one fixed twelve-tone HT20 band:

```text
DEFAULT_SUBCARRIERS = (4, 8, 13, 18, 23, 28, 36, 41, 46, 51, 56, 60)
```

On the centered grid these are subcarriers ±4, ±9, ±14, ±19, ±24, and ±28.

- HT20 layouts are normalized before the band is extracted.
- The band is the same for every session, chip, frontend, and detector.
- It is chosen for full span and independent looks, not for a detector score on some dataset.
- The common amplitude path uses twelve single bins. Adjacent-bin averaging is allowed only inside a named feature, like the aggregated-IQR input used by both profiles.

## Decision history

| Date | Direction | Resolution |
| --- | --- | --- |
| 2025-12-03 | Runtime subcarrier selection with NBVI | Replaced by a fixed shared band |
| 2026-06-09 | Remove runtime NBVI, use fixed subcarriers | Still the operating model |
| 2026-07-20 | First twelve-tone layout | Replaced after fixing cross-chip bin order |
| 2026-07-25 | Full-span twelve-tone band from measured coherence | Accepted |
| 2026-08-05 | Average adjacent bins in the common path | Rejected; only feature-local aggregation |

## Evidence

The new band covers all 56 usable HT20 subcarriers and gives about 3.55 independent looks, against 2.44 for the previous layout. A later refit sweep over tone counts showed that denser bands add correlated information and hurt the high-rate stress capture, even when chip averages looked better. Twelve stays because coherence caps the number of independent looks, and end-to-end replay does not justify the extra cost or lost dynamic range of more tones.

Averaging adjacent bins in the common path did not pay off in joint replay either. The aggregated-IQR feature has its own stream and does not change this band.

## Alternatives considered

- **Bring back runtime NBVI.** Its adaptive search is not worth the calibration, memory, persistence, and parity cost.
- **Pick the band with the best detector score.** Ties the geometry to one dataset and risks fitting a coincidence.
- **Use more tones.** Independent looks saturate, and tighter spacing hurts high-rate behavior.
- **Average adjacent bins everywhere.** Silently changes every consumer of the band. Feature-level aggregation must be measured and named on its own.

## Consequences

Startup does no environment-specific search, Python and C++ share one grid and one band, and the band covers the usable HT20 range on every supported chip. Any new band, count, or aggregation proposal needs a refit, production replay, and parity evidence.

## Related

- [`2026-07-23-adopt-classifier-first-ht20-sensing-contract.md`](2026-07-23-adopt-classifier-first-ht20-sensing-contract.md)
- [`2026-07-04-keep-agc-active-and-standardize-cv-normalization.md`](2026-07-04-keep-agc-active-and-standardize-cv-normalization.md)
- [Algorithms](../ALGORITHMS.md) and the [feature ledger](../FEATURES.md)
- Commits `bd3eeb48` (NBVI removed), `f3908919` (coherence band), `e8ae2420` (adjacent-bin averaging rejected)
