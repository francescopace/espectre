# ADR: unify raw CSI collection over HTTP

- Status: Superseded in part
- Date: 2026-07-03
- Recorded: 2026-07-09 (retrospective)
- Updated: 2026-08-29

The binary framing, ordering, provenance, and fixed-ring behavior remain accepted. Explicit command-created sessions and bearer binding are superseded by automatic `GET /csi` lifetime ownership in [`2026-09-03-adopt-resource-oriented-device-api.md`](2026-09-03-adopt-resource-oriented-device-api.md).

## Context

ESPectre first moved high-rate collection from the Python prototype path into a dedicated C++ Streamer frontend with collector-paced UDP and explicit TX backpressure. That aligned collection with the C++ architecture, but the separate firmware later duplicated Wi-Fi lifecycle, discovery, builds, releases, and validation after the maintained sensing frontends gained a raw HTTP path.

Collection must retain the exact order and provenance of offered CSI records. Device-side pacing, adaptive sample replacement, and temporal admission can hide loss or bias a dataset. A full transport buffer must therefore produce an observable sequence gap and counter instead of silently choosing a newer sample.

Normalized amplitude does not erase radio context. Every record must retain the PHY family, LTF type, channel width, receive timestamps, sequence, and freshness metadata supplied by the capture path. The production sensing contract remains classifier-first HT20; carrying wider enum values does not promote wider sensing support.

## Decision

Remove the Streamer frontend and use raw HTTP as the only live collection transport across supported ESPectre frontends. The published framing uses protocol version `1`.

Raw collection does not change the configured traffic source, pace output, select a freshest sample, or apply temporal admission. The CSI callback first applies the bounded, fail-closed provenance classifier for the configured internal or external generator. Classified raw frames enter a preallocated 16-record atomic SPSC ring. A dedicated task-notified worker sends up to four ordered records per chunk. A full ring drops the newest record with an explicit counter. Each offered frame receives its 64-bit stream sequence before enqueue, so a drop creates an observable gap.

`./espectre collect` persistently selects `external`, opens the bearer-bound raw session, and imports `ExternalTrafficGenerator` from the standalone, standard-library-only `tools/espectre_traffic_generator.py`. Its `--pps` value controls only that UDP generator and dataset provenance. External datagrams carry the exact four-byte UTF-8 payload `"👻".encode("utf-8")` (`F0 9F 91 BB`) as the canonical marker. The web raw tool uses the device's existing internal or external configuration and does not expose a PPS control.

Raw HTTP prefixes every CSI V8 record with a 60-byte transport record. The V8 header preserves `phy_mode`, `ltf_type`, `channel_width`, receive timing, device sequence, chip, RSSI, channel, and CSI payload length. The HTTP prefix adds the session and stream sequence used to expose transport loss. The collector stores the normalized PHY fields in every generated `.npz` dataset. Historical datasets without these fields retain their documented HT, HT-LTF, and 20 MHz interpretation when their layout proves the earlier HT20 contract.

The published raw HTTP framing uses protocol version `1`. Host tooling retains read support for historical V7 records, but no maintained workflow emits Streamer UDP records.

## Decision History

| Date | Direction | Resolution |
| --- | --- | --- |
| 2026-07-03 | Use a dedicated C++ Streamer frontend with collector-paced UDP and TX-backpressure feedback | Replaced after raw collection became available in every maintained sensing frontend |
| 2026-07-19 | Preserve per-record PHY provenance in Streamer V7 datasets | Retained as a format invariant in CSI V8 and raw HTTP v2 |
| 2026-08-25 | Remove Streamer and collect through raw HTTP across supported ESPectre frontends | Accepted |
| 2026-08-25 | Pace or replace samples inside the HTTP data plane | Rejected because transport feedback would decide which records enter the dataset |
| 2026-08-29 | Publish the current raw HTTP framing as protocol version `1` instead of `2` | Accepted; CSI record version remains `8` |

## Alternatives Considered

### Retain Streamer as a collection fallback

Rejected. It preserves two firmware, discovery, protocol, and release surfaces and makes collection transport-dependent.

### Add credits or adaptive pacing to raw HTTP

Rejected. A fixed memory boundary with explicit drops keeps loss measurable without allowing transport feedback to select data.

### Store every normalized record as generic HT20

Rejected. A common normalized grid does not imply a common PHY or LTF, and erasing metadata would prevent later audit and stratification.

### Change traffic mode only for the collection session

Rejected. Collection is an intentional operator action. Persisting `external` keeps actual traffic ownership explicit after the session.

### Apply temporal admission before collection

Rejected. Raw research data must preserve classified input and timing so later detector views can reproduce the deployed admission contract.

## Consequences

- every maintained ESP-IDF frontend shares one raw collection contract and Direct discovery path;
- dataset capture observes all classified frames except explicit fixed-ring drops;
- after drain, `fresh_record_total + raw_drop_total == classified_frames_offered_to_raw`;
- per-record PHY and timing provenance survives the transport migration;
- traffic ownership remains a persisted device setting, with only `internal` and `external` as valid modes; and
- historical Streamer reviews, benchmarks, and datasets remain evidence, while Streamer code, builds, CI, release assets, and current documentation remain removed.

## Related

- [`2026-07-23-adopt-classifier-first-ht20-sensing-contract.md`](2026-07-23-adopt-classifier-first-ht20-sensing-contract.md)
- [`2026-08-17-adopt-improv-serial-and-direct-http-for-local-control.md`](2026-08-17-adopt-improv-serial-and-direct-http-for-local-control.md)
- [`2026-08-23-standardize-managed-csi-traffic-sources.md`](2026-08-23-standardize-managed-csi-traffic-sources.md)
- [`../CLI.md`](../CLI.md)
