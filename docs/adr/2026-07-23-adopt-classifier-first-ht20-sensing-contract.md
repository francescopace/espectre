# ADR: adopt a classifier-first HT20 sensing contract

- Status: Accepted
- Date: 2026-07-23

## Context

ESPectre already treated HT20 as the intended production sensing baseline, but
that contract was still enforced indirectly in multiple places:

- firmware capture normalized payloads from byte length before making an
  explicit format decision
- detector paths assumed the normalized payload was already a valid HT20
  64-subcarrier view
- host loaders filtered mostly on `phy_mode=ht` and `channel_width=20`, without
  requiring `ltf_type=ht-ltf` or an approved stored layout
- historical datasets without PHY metadata were treated broadly as HT20, even
  when the stored layout itself did not prove that contract

That left an ambiguity gap between "payload that can be parsed" and "payload
that is valid production sensing input". The same gap also made future raw
collection work harder to reason about, because unsupported formats could be
normalized or admitted too early.

## Decision

Adopt one explicit classifier-first sensing contract:

- production sensing accepts only HT20 + HT-LTF + 64-subcarrier CSI
- every packet or dataset row is classified before normalization
- normalization is allowed only for known HT20 layouts that map into the same
  production grid
- unsupported or ambiguous formats are dropped from sensing with explicit reason
  telemetry
- host training and validation fail explicitly when format filtering removes all
  valid sensing data
- historical captures without per-record PHY metadata remain compatible only
  when the stored payload already matches the HT20 64-subcarrier sensing layout

Concretely, the runtime contract becomes:

1. validate structure
2. validate PHY, LTF, and width metadata
3. recognize the payload layout
4. normalize only when the layout is a named HT20 variant
5. route either to the detector or to an explicit drop path

The accepted normalization set is intentionally narrow:

- exact HT20 64-subcarrier payload: consume directly
- HT20 short estimate `57 -> 64`: consume only as a named normalization
- HT20 doubled estimate `256` or `228`: consume only as a named HT20
  normalization, not as a generic "same length implies same format" rule

This decision does not yet promote raw multi-layout collection into the
production sensing path. Unsupported formats may be preserved later in a
separate raw collection flow, but they do not enter Classic or ML sensing
implicitly.

## Alternatives Considered

### Keep length-first normalization and tighten only host filtering

Rejected. That would leave the firmware hot path able to reinterpret packets
before their format was decided, and it would keep runtime and host admission
rules divergent.

### Treat every parseable record as candidate sensing input

Rejected. Structural validity is weaker than sensing compatibility. Allowing
partial or ambiguous layouts to flow into detectors or training would create
silent behavior changes and make benchmark baselines harder to trust.

### Expand production sensing to wider or newer PHYs now

Rejected. The repository does not yet carry explicit, validated layout mapping,
training corpora, and regression gates for HT40, VHT, or HE sensing. Those
formats must be promoted individually, not inferred from payload length.

## Consequences

Benefits:

- the runtime and host toolchain now share one explicit sensing admission rule
- unsupported formats fail loudly instead of contaminating detector or training
  inputs
- telemetry can distinguish unsupported PHY, width, metadata, and layout causes
- future raw collection work has a clean boundary from the production sensing
  contract

Trade-offs:

- historical compatibility is narrower than before
- host tooling now rejects some previously tolerated captures unless they match
  the exact HT20 contract
- new PHY support requires explicit per-format implementation work instead of
  opportunistic reuse of the HT20 path

## Related

- [`2026-07-19-preserve-per-record-phy-provenance-in-streamer-datasets.md`](2026-07-19-preserve-per-record-phy-provenance-in-streamer-datasets.md)
- [`2026-07-20-keep-the-12-tone-ht20-classic-band.md`](2026-07-20-keep-the-12-tone-ht20-classic-band.md)
