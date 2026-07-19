# ADR: preserve per-record phy provenance in streamer datasets

- Status: Accepted
- Date: 2026-07-19

## Context

The original ESP32 can associate with an access point in 802.11b/g/n mode while
the access point selects legacy or HT rates independently for each frame. An
HT-LTF-only capture configuration can therefore appear to stall when rate
control temporarily leaves HT, even though the Wi-Fi link and UDP pacing remain
healthy.

Filtering legacy callbacks would keep datasets homogeneous but would also hide
the actual CSI supply available from the link. Treating every normalized
64-subcarrier record as HT20 would preserve packet count while losing the PHY
origin needed to separate legacy LLTF from HT-LTF during analysis.

The host also needs a stable pacing signal. A CSI callback deficit describes
sample supply, not necessarily uplink congestion, so using it to reduce pacing
can amplify a device-side callback deficit without relieving the TX path.

## Decision

Preserve the received PHY provenance on every streamer record and keep CSI
supply separate from transport congestion.

Concretely:

- stream protocol V7 carries `phy_mode`, `ltf_type`, and `channel_width` for
  every record
- the original ESP32 captures both legacy LLTF and HT-LTF; HT frames that
  contain both fields continue to export only the HT-LTF portion
- the collector stores the three normalized fields in each generated `.npz`
  dataset instead of filtering legacy records
- known historical datasets without these fields are interpreted as HT,
  HT-LTF, and 20 MHz because they were collected by the earlier HT20-only path
- adaptive host pacing uses sustained firmware TX backpressure as its control
  signal; receive rate and CSI freshness remain telemetry
- the wire enum reserves normalized wider channel widths, but the current
  capture, normalization, and ML pipeline remains HT20 until wideband payload
  support is implemented explicitly

The original ESP32 callback watchdog remains a target-specific recovery
mechanism. It does not change the chip-independent collector pacing policy.

## Alternatives Considered

### Filter non-HT records in the CSI callback

Rejected. This would discard valid CSI and could reproduce the low-supply
behavior that motivated the change. It would also prevent analysis of how AP
rate control affects collection.

### Store every normalized record as HT20

Rejected. A common 64-subcarrier representation does not imply a common PHY or
LTF. Removing that provenance would mix distinct measurements silently.

### Control pacing from callback freshness

Rejected. Low freshness is not evidence that the firmware TX path is
congested. Backing off in response can reduce the requested collection rate
without addressing the callback source.

## Consequences

Benefits:

- mixed-rate collections remain usable and can be stratified by PHY and LTF
- the collector does not reduce pacing solely because CSI callbacks fluctuate
- historical datasets keep an explicit, documented HT20 interpretation
- protocol fields are available for future HT40 or 5 GHz collection work

Trade-offs:

- analyses that require homogeneous input must select the desired PHY metadata
- streamer firmware and collector must move together when the record header
  changes
- wider-width enum values do not by themselves provide wideband collection
  support

## Related

- [`2026-07-03-adopt-a-dedicated-cpp-streamer-frontend-for-high-rate-csi-collection.md`](2026-07-03-adopt-a-dedicated-cpp-streamer-frontend-for-high-rate-csi-collection.md)
- [`2026-07-04-keep-agc-active-and-standardize-cv-normalization.md`](2026-07-04-keep-agc-active-and-standardize-cv-normalization.md)
