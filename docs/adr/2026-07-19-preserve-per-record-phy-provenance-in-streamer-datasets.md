# ADR: preserve per-record phy provenance in streamer datasets

- Status: Accepted
- Date: 2026-07-19

## Context

The streamer protocol exports normalized CSI records that the host later stores as datasets and reuses for offline validation, training, and regression tests. That data path benefits from keeping not only the normalized amplitudes, but also the PHY context that produced them.

Even when the current sensing contract is HT20-only, treating every normalized 64-subcarrier record as generic HT20 would erase information that can matter later:

- historical captures may need to be interpreted against the PHY/LTF that produced them
- future collection work may enable HT40, HE20, or other 5 GHz sensing modes
- training and test tooling may need to stratify, filter, or compare captures by PHY family, LTF type, or channel width
- the protocol should not need another record-format change just to preserve metadata that the firmware already knows at capture time

The host also needs a stable pacing signal. A CSI callback deficit describes sample supply, not necessarily uplink congestion, so using it to reduce pacing can amplify a device-side callback deficit without relieving the TX path.

## Decision

Preserve the received PHY provenance on every streamer record and keep CSI supply separate from transport congestion.

Concretely:

- stream protocol V7 carries `phy_mode`, `ltf_type`, and `channel_width` for every record
- the collector stores those normalized fields in each generated `.npz` dataset
- known historical datasets without these fields are interpreted as HT, HT-LTF, and 20 MHz because they were collected by the earlier HT20-only path
- adaptive host pacing uses sustained firmware TX backpressure as its control signal; receive rate and CSI freshness remain telemetry
- the wire enum reserves normalized wider channel widths, but the current capture, normalization, and ML pipeline remains HT20 until wideband payload support is implemented explicitly

The original ESP32 callback watchdog remains a target-specific recovery mechanism. It does not change the chip-independent collector pacing policy.

## Alternatives Considered

### Filter non-HT records in the CSI callback

Rejected. This would discard valid CSI and could reproduce the low-supply behavior that motivated the change. It would also prevent analysis of how AP rate control affects collection.

### Store every normalized record as HT20

Rejected. A common normalized representation does not imply a common PHY or LTF. Removing that provenance would silently discard metadata that can become useful for future HT40, HE20, or 5 GHz captures, and for training or test stratification.

### Control pacing from callback freshness

Rejected. Low freshness is not evidence that the firmware TX path is congested. Backing off in response can reduce the requested collection rate without addressing the callback source.

## Consequences

Benefits:

- datasets keep enough metadata to stratify or audit captures by PHY, LTF, and channel width when needed
- the collector does not reduce pacing solely because CSI callbacks fluctuate
- historical datasets keep an explicit, documented HT20 interpretation
- protocol fields are already in place for future HT40, HE20, or 5 GHz collection work without another schema change
- training and regression tooling can use the extra provenance when a future corpus needs per-PHY filtering, comparison, or diagnostics

Trade-offs:

- analyses that require homogeneous input must still select the desired PHY metadata explicitly
- streamer firmware and collector must move together when the record header changes
- wider-width enum values do not by themselves provide wideband collection support

## Related

- [`2026-07-03-adopt-a-dedicated-cpp-streamer-frontend-for-high-rate-csi-collection.md`](2026-07-03-adopt-a-dedicated-cpp-streamer-frontend-for-high-rate-csi-collection.md)
- [`2026-07-04-keep-agc-active-and-standardize-cv-normalization.md`](2026-07-04-keep-agc-active-and-standardize-cv-normalization.md)
