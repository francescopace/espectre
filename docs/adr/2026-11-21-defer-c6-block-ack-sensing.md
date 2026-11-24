# ADR: Defer ESP32-C6 sensing from access-point Block ACKs

- Status: Accepted
- Date: 2026-11-21

## Decision

Keep `wifi_raw` disabled on ESP32-C6. An access point's Block ACKs can supply CSI for the AP-to-C6 link. The diagnostic depended on private ESP-IDF driver interfaces, and integrated firmware has yet to reproduce the result on hardware.

To sense a second ESP directly, the demonstrated arrangement uses the C6 in `external`/HT20 mode and the other ESP as the sender of padded MCS0 Null Data frames.

## Evidence

On one associated 2.4 GHz HT20 AP, ordinary ACKs yielded no usable CSI. At MCS0, the AP returned Block ACKs at 6 Mbps. With MQTT and HTTP active, the ESP32-C6 diagnostic sustained 100 usable reports/s over 60 seconds. Filtering the measurement traffic by TID excluded application traffic and admitted zero reports with probes off. Promiscuous mode stayed off throughout.

## Reopening the decision

A production candidate needs a supported way to identify measurement traffic and deliver its Block ACKs while associated. The diagnostic used private ESP-IDF 6.1.1 transmit hooks and receive-slot controls.

Before enabling the source, reproduce zero admissions with probes off and approximately 100 fresh reports/s under concurrent MQTT and HTTP load on additional APs. Validate detector input in integrated firmware, including stop, restart, and roaming. Keep promiscuous mode off throughout.

## Related

- [Managed CSI traffic sources and the original short-frame investigation](2026-08-23-standardize-managed-csi-traffic-sources.md)
- [Current CSI compatibility limits](../CSI.md#compatibility-limits)
- [Temporal CSI admission](2026-08-15-use-fixed-temporal-csi-admission.md)
