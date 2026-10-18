# ADR: define the detection profiles

- Status: Accepted
- Date: 2026-07-15
- Updated: 2026-10-18

## Context

ESPectre ships two detectors and three frontends with different control surfaces: ESPHome has entities and Direct HTTP, Native has Direct HTTP and optional MQTT, and Matter has standard occupancy clusters plus Direct HTTP for ESPectre-specific settings. Implementing detector selection in each frontend would duplicate validation, persistence, threshold reset, and calibration.

The detectors were first called `classic` and `ml`. Those names described where they came from, not the choice a user or integrator is making: `Classic` said nothing about its lower resource use or its calibration trade-off, and `ML` named a technique without saying it is more accurate, generalizes better, and needs no room calibration at start. The names have to make sense in user docs, configuration, protocol values, MicroPython, and the C++ SDK.

## Decision

**Names.** The two public profiles are named after what they are for:

| Profile | Config value | SDK class | Pick it when |
| --- | --- | --- | --- |
| Lightweight Detection | `lightweight` | `LightweightDetector` | CPU and memory are tight, and initial calibration and a lower accuracy ceiling are acceptable |
| High-Accuracy Detection | `high_accuracy` | `HighAccuracyDetector` | The platform has more CPU and memory, and you want higher accuracy, better transfer across rooms, and no initial calibration |

The names describe product roles, not algorithms; what runs behind each profile can change without another rename, as long as the role and trade-offs hold. "ML" is still used where it is precise: training, weights, ML features, model artifacts. Historical dataset metadata, cached performance keys, and decision records may keep `classic` or `ml` when changing them would hide provenance or break an internal schema. There are no `ClassicDetector`, `MLDetector`, or old config aliases: the rename shipped as a breaking change in 3.0.0, and `ClassicDetector` was never in a release.

**Selection.** The active profile lives in the shared runtime and is exposed through each frontend's capabilities:

| Frontend | How to switch | Default and persistence |
| --- | --- | --- |
| ESPHome | ESPHome entities or Direct HTTP | Configured value, then the runtime choice is saved in the shared ESP-IDF store |
| Native | Direct HTTP or MQTT | Configured value, then saved in the shared store |
| Matter | Direct HTTP only | Published firmware starts on Lightweight, then saved in the shared store |

When the profile changes, the runtime validates the request, resets the threshold to the new detector's automatic default, starts calibration when switching to Lightweight (or cancels it when switching to High Accuracy), and emits the usual state change. Nothing is added to the CSI hot path. Matter's standard occupancy clusters have no equivalent control, so selection stays out of them; the Direct HTTP bridge accepts `detector` in `PATCH /sensing` like the other frontends.

## Decision history

| Date | Direction | Resolution |
| --- | --- | --- |
| 2026-07-15 | Shared persisted selection; ML as Matter's fixed default | Kept for writable frontends |
| 2026-07-24 | Matter's fixed default moves from ML to Classic | Kept, under the later Lightweight name |
| 2026-08-13 | Rename `classic` and `ml` to `lightweight` and `high_accuracy` | Accepted; shipped without aliases in 3.0.0 |
| 2026-08-26 | Persisted selection on Matter through Direct HTTP | Accepted, without a custom Matter cluster |

## Alternatives considered

- **Keep Classic and ML.** You need to know the project's history to choose, and the high-accuracy profile stays tied to one model family.
- **Low- and high-consumption names.** Costs depend on the platform, and the names hide the accuracy, calibration, and generalization trade-offs that actually drive the choice.
- **Low- and high-resource names.** A better cost signal, but they sound like hardware tiers rather than detection strategies. The comparison guide carries the full trade-off.
- **Compatibility aliases.** Both vocabularies would become part of the SDK and config contract.
- **Build-time selection only.** Comparing profiles on a deployed device would still need a reflash.
- **Per-frontend persistence.** Duplicates lifecycle and validation.
- **Selection in the Matter occupancy surface.** The standard clusters do not define it, and Direct HTTP covers it without a non-standard data model.

## Consequences

Users and OEM integrators choose by outcome and hardware budget, and config, C++ and Python classes, examples, and docs share one vocabulary. High Accuracy can keep using ML inside without ML becoming a product category, and docs must describe resource use as relative and platform-dependent, not as a power guarantee. All three frontends can switch profiles without reflashing, with persistence, threshold reset, and calibration in one place. Matter controllers still see a standard occupancy device, while Direct clients can pick the profile. Maintainers will still meet `classic` and `ml` in historical tools and performance data and must tell provenance keys from public names.

## Related

- [Algorithms](../ALGORITHMS.md), [SDK guide](../SDK.md), [changelog](../CHANGELOG.md), and the [detector comparison guide](../web/content/guides/detectors.html)
- [`2026-06-03-adopt-the-core-runtime-frontend-firmware-split.md`](2026-06-03-adopt-the-core-runtime-frontend-firmware-split.md)
- [`2026-07-02-use-one-resource-oriented-protocol-across-transports.md`](2026-07-02-use-one-resource-oriented-protocol-across-transports.md)
- [`2026-07-08-define-the-lightweight-detector.md`](2026-07-08-define-the-lightweight-detector.md)
- Commits `06803345` (persisted selection), `75081422` (Matter default to Classic), `b90d3e12` (profile names)
