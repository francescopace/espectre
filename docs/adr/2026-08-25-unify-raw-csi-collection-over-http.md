# ADR: unify raw CSI collection over HTTP

- Status: Accepted
- Date: 2026-08-25
- Supersedes: `2026-07-03-adopt-a-dedicated-cpp-streamer-frontend-for-high-rate-csi-collection.md`, `2026-07-19-preserve-per-record-phy-provenance-in-streamer-datasets.md`

## Context

The dedicated Streamer firmware duplicated Wi-Fi lifecycle, discovery, build, release, and validation surfaces while raw HTTP had already become available in the maintained sensing runtime. Experiments also showed that device-side HTTP pacing and sample replacement could hide capture loss and bias datasets. The external UDP traffic generator already provided the intended traffic source and was useful as a standalone script for Home Assistant and other hosts.

## Decision

Remove the Streamer frontend and make raw HTTP v2 the only live collection transport on Native, ESPHome, and Matter. Raw collection does not alter the configured traffic source, pace output, select a freshest sample, or apply temporal admission. The CSI callback first applies a bounded, fail-closed provenance classifier for the configured internal or external generator. Classified raw frames enter a preallocated 16-record atomic SPSC ring, a dedicated task-notified raw worker sends up to four ordered records per chunk, and a full ring drops the newest record with an explicit counter. Each offered frame receives its 64-bit stream sequence before enqueue, so a dropped record creates an observable gap.

`./espectre collect` persistently selects `external`, opens the bearer-bound raw session, and imports `ExternalTrafficGenerator` from the standalone, standard-library-only `tools/espectre_traffic_generator.py`. Its `--pps` value controls only that UDP generator and dataset provenance. External datagrams carry the exact one-byte payload `b'.'` (`0x2E`) as the canonical marker. The web raw tool uses the device's existing internal or external configuration and does not expose a PPS control.

Raw HTTP v2 uses a 60-byte prefix and CSI V8 records. It is intentionally incompatible with raw HTTP v1. Host tooling retains read support for historical V7 data, but no maintained workflow emits Streamer UDP records.

## Consequences

- every maintained ESP-IDF frontend shares one raw collection contract and one Direct discovery path;
- dataset capture observes all classified frames except bounded ring drops, with no hidden transport decimation;
- after drain, `fresh_record_total + raw_drop_total == classified_frames_offered_to_raw`;
- traffic ownership remains an intentional persistent device setting, and only `internal` and `external` remain valid modes; and
- historical Streamer ADRs, reviews, benchmarks, and datasets remain evidence, but Streamer code, builds, CI, release assets, and current documentation are removed.

## Alternatives considered

### Retain Streamer as a collection fallback

Rejected. A fallback would preserve two firmware, discovery, protocol, and release surfaces and make collection results transport-dependent.

### Add credits or adaptive pacing to raw HTTP

Rejected. Transport feedback would control which CSI records enter the dataset. A fixed memory boundary with explicit drops is simpler and makes loss measurable.

### Change traffic mode only for the collection session

Rejected. Running the collector is an intentional operator action. Persisting `external` makes the device's actual traffic ownership explicit after collection.
