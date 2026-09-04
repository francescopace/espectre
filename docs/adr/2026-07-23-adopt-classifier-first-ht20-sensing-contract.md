# ADR: adopt a classifier-first 20 MHz sensing contract

- Status: Accepted
- Date: 2026-07-23
- Updated: 2026-09-01

## Context

Parsing a CSI payload does not prove that it is valid detector input. Firmware and host tooling previously inferred too much from byte length, while band selection was initially forced to 2.4 GHz even on dual-band targets. The production contract must define PHY admission, layout normalization, and selected Wi-Fi band independently.

## Decision

Adopt one explicit classifier-first sensing contract:

- production sensing accepts only the named `lltf20`, `ht20`, and `vht20` capture profiles and a recognized layout that maps to the internal 64-subcarrier grid;
- classify every packet or dataset row before normalization;
- normalize only named 20 MHz layouts, including exact 64-subcarrier payloads and explicitly supported short or doubled estimates;
- drop unsupported or ambiguous formats with reason telemetry;
- require host training and validation to fail explicitly when filtering removes all valid sensing data;
- preserve historical captures without PHY metadata only when their stored layout itself proves the supported 20 MHz contract; and
- let the frontend or SDK integrator select `2g`, `5g`, or `auto`, while enforcing 20 MHz bandwidth, HT on 2.4 GHz, VHT on 5 GHz, and no HE capture.

The validated detector corpus remains 2.4 GHz. The 5 GHz and automatic band modes are available on supported dual-band targets, but their availability does not constitute detector-performance validation on a 5 GHz corpus. HE20, HT40, VHT40, and wider layouts require their own explicit promotion.

The runtime admission order is:

1. validate structure;
2. validate PHY, LTF, and width metadata;
3. recognize the payload layout;
4. normalize a named 20 MHz variant;
5. route to sensing or an explicit drop path.

## Decision History

| Date | Direction | Resolution |
| --- | --- | --- |
| 2026-07-23 | Make HT20 admission classifier-first | Accepted |
| 2026-08-05 | Force 2.4 GHz on every target | Replaced with explicit integrator-selected band mode while keeping HT20 on all selected bands |
| 2026-09-01 | Keep an 802.11n ceiling on 5 GHz | Replaced with VHT20 capture on VHT-capable 5 GHz targets while retaining the canonical 64-subcarrier detector view |

## Alternatives Considered

### Keep length-first normalization

Rejected. Structural compatibility is weaker than sensing compatibility and can silently reinterpret unsupported inputs.

### Accept every parseable PHY or width

Rejected. The repository lacks validated mappings, corpora, and parity gates for those sensing contracts.

### Force 2.4 GHz on dual-band devices

Rejected. It unnecessarily removes integrator choice. The validation status is documented separately from the supported radio configuration surface.

### Use automatic band selection everywhere

Rejected. Fixed-band deployments need deterministic policy, and integrators must be able to choose the intended band explicitly.

## Consequences

- Runtime and host tooling share one sensing admission boundary.
- Unsupported PHYs fail loudly instead of contaminating detectors or training data.
- The canonical 64-subcarrier detector view remains stable while the runtime selects LLTF20, HT20, or VHT20 capture from the chip and associated band.
- New PHY support requires layout mapping, representative data, detector validation, and Python/C++ parity.

## Related

- [`2026-07-03-unify-raw-csi-collection-over-http.md`](2026-07-03-unify-raw-csi-collection-over-http.md)
- [`2026-07-25-select-the-classic-band-from-channel-coherence.md`](2026-07-25-select-the-classic-band-from-channel-coherence.md)
- [`../ALGORITHMS.md`](../ALGORITHMS.md)
- [`../ARCHITECTURE.md`](../ARCHITECTURE.md)
