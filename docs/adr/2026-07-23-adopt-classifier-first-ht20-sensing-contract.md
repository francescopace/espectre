# ADR: adopt a classifier-first 20 MHz sensing contract

- Status: Accepted
- Date: 2026-07-23
- Updated: 2026-09-01

## Context

A CSI payload that parses is not necessarily valid detector input. Firmware and host tools used to guess too much from the byte length, and the band was forced to 2.4 GHz even on dual-band chips. PHY admission, layout normalization, and band choice need to be defined separately.

## Decision

Every packet, or dataset row, is classified before it is normalized. Only the `lltf20`, `ht20`, and `vht20` capture profiles are accepted, and only with a known layout that maps onto the internal 64-subcarrier grid. That includes exact 64-subcarrier payloads and the explicitly supported short or doubled estimates. Anything unsupported or ambiguous is dropped with a reason counter.

The runtime admits a packet in this order: check structure, check PHY, LTF, and width metadata, recognize the layout, normalize the named 20 MHz variant, then route it to sensing or to an explicit drop.

On the host, training and validation fail loudly if filtering leaves no valid data. Old captures without PHY metadata are kept only when their layout alone proves they are 20 MHz.

The frontend or SDK integrator picks the band: `2g`, `5g`, or `auto`. Bandwidth is always 20 MHz, with HT on 2.4 GHz, VHT on 5 GHz, and no HE capture.

The detector is validated on a 2.4 GHz corpus only. The 5 GHz and `auto` modes work on dual-band chips, but that is not a performance claim. HE20, HT40, VHT40, and wider layouts each need their own promotion.

## Decision history

| Date | Direction | Resolution |
| --- | --- | --- |
| 2026-07-23 | Classifier-first HT20 admission | Accepted |
| 2026-08-06 | 2.4 GHz forced on every target | Replaced by an integrator-selected band, HT20 on every band |
| 2026-09-01 | 802.11n ceiling on 5 GHz | Replaced by VHT20 capture on VHT-capable 5 GHz targets, same 64-subcarrier detector view |

## Alternatives considered

- **Normalize by length.** Structural fit is weaker than sensing fit and can silently misread unsupported input.
- **Accept every parseable PHY or width.** We have no validated mappings, corpora, or parity gates for them.
- **Force 2.4 GHz on dual-band devices.** Takes away the integrator's choice; validation status is documented separately.
- **Use `auto` everywhere.** Fixed-band deployments need a predictable policy and an explicit choice.

## Consequences

Runtime and host tools share one admission boundary, and unsupported PHYs fail loudly instead of polluting detectors or training data. The 64-subcarrier detector view stays stable while the runtime picks LLTF20, HT20, or VHT20 from the chip and the band. Any new PHY needs a layout mapping, representative data, detector validation, and Python/C++ parity.

## Related

- [`2026-07-03-unify-raw-csi-collection-over-http.md`](2026-07-03-unify-raw-csi-collection-over-http.md)
- [`2026-07-25-select-the-classic-band-from-channel-coherence.md`](2026-07-25-select-the-classic-band-from-channel-coherence.md)
- [Algorithms](../ALGORITHMS.md) and [architecture](../ARCHITECTURE.md)
- Commits `380bdf0c` (classifier-first admission), `7f02deec` (band selection), `9ae2fdf2` (automatic capture profiles)
