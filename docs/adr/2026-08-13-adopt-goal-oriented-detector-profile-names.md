# ADR: adopt goal-oriented detector profile names

- Status: Accepted
- Date: 2026-08-13
- Updated: 2026-08-13

## Context

The original `classic` and `ml` detector names described implementation lineage rather than the choice presented to users and SDK integrators. `Classic` did not communicate its lower resource demand or calibration trade-off, while `ML` exposed a current implementation technique without explaining its higher accuracy, broader generalization, or ability to operate without initial room calibration.

The same distinction must remain understandable across consumer-facing documentation, configuration, protocol values, MicroPython, and the published C++ SDK. Naming the profiles after relative power consumption was considered, but actual energy use depends on the chip, packet rate, frontend, and integration. It would also describe only the cost side of the decision and not the expected detection behavior.

## Decision

Use goal-oriented names for the two public detection profiles:

| Public profile | Configuration value | SDK class | Intended choice |
| --- | --- | --- | --- |
| Lightweight Detection | `lightweight` | `LightweightDetector` | Lower CPU and memory demand when initial calibration and a lower accuracy ceiling are acceptable |
| High-Accuracy Detection | `high_accuracy` | `HighAccuracyDetector` | Higher measured accuracy, better cross-environment transfer, and no initial room calibration when the platform can provide more CPU and memory |

These names describe stable product roles rather than current algorithms. The implementation behind either profile may evolve without another public rename as long as its role and documented trade-offs remain intact.

Keep `ML` terminology where it precisely identifies the implementation, including model training, model weights, ML features, and model artifacts. Historical dataset metadata, cached performance keys, and decision records may retain `classic` or `ml` when changing them would obscure provenance or break an internal data schema. Current user-facing text and callable SDK types use the profile names.

Do not provide `ClassicDetector`, `MLDetector`, or legacy configuration aliases. The rename is part of the unreleased breaking API transition, and `ClassicDetector` was not present in the preceding release. A compatibility layer would leave two vocabularies in the public surface without protecting an established external contract.

## Alternatives Considered

### Keep Classic and ML

Rejected. The names require knowledge of project history or implementation details, do not present a direct product choice, and couple the high-accuracy public surface to one model family.

### Use low- and high-consumption names

Rejected. CPU, memory, and energy costs are platform-dependent, and consumption names omit the accuracy, calibration, and generalization trade-offs that drive the selection.

### Use implementation-neutral low- and high-resource names

Rejected. These names improve the cost signal but still make the profiles sound like hardware tiers rather than detection strategies. `Lightweight` and `High Accuracy` communicate the primary benefit of each option while the comparison guide carries the complete trade-off.

### Retain compatibility aliases

Rejected. Aliases would make both naming systems part of the SDK and configuration contract, increase documentation ambiguity, and preserve technical names that the decision is intended to remove.

## Consequences

- Consumers and OEM integrators can select a detector from the intended outcome and available hardware budget.
- Configuration values, C++ and Python classes, examples, and current documentation share one vocabulary.
- The high-accuracy implementation may continue to use ML internally without making ML a permanent public product category.
- Documentation must present resource use as relative and platform-dependent rather than as a fixed power guarantee.
- Historical tools and performance data can contain `classic` and `ml` identifiers, so maintainers must distinguish provenance keys from current public names.
- This is an intentional breaking rename with no public compatibility path.

## Related

- [ALGORITHMS.md](../ALGORITHMS.md)
- [SDK.md](../SDK.md)
- [CHANGELOG.md](../CHANGELOG.md)
- [detectors.html](../web/content/guides/detectors.html)
- [2026-07-15-adopt-persisted-runtime-detector-selection.md](2026-07-15-adopt-persisted-runtime-detector-selection.md)
