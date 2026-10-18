# ADR: unify raw CSI collection over HTTP

- Status: Superseded in part
- Date: 2026-07-03
- Recorded: 2026-07-09 (retrospective)
- Updated: 2026-09-23

Framing, ordering, provenance, and the fixed ring still hold. Command-created sessions and bearer binding were replaced by `GET /csi` owning the session lifetime, see [`2026-07-02-use-one-resource-oriented-protocol-across-transports.md`](2026-07-02-use-one-resource-oriented-protocol-across-transports.md).

## Context

High-rate collection first moved from the Python prototype into a dedicated C++ Streamer frontend, with collector-paced UDP and TX backpressure. Once the regular sensing frontends could stream raw CSI over HTTP, Streamer was just a second copy of Wi-Fi lifecycle, discovery, builds, releases, and tests.

A dataset has to keep the exact order and provenance of the CSI it was offered. Device-side pacing, sample replacement, or temporal admission can hide loss or bias the data, so a full buffer must show up as a sequence gap and a counter, never as a silently swapped sample. Each record also keeps its radio context (PHY family, LTF type, channel width, timestamps, sequence) even though amplitudes are normalized. Carrying wider-bandwidth values does not mean sensing supports them: production stays classifier-first HT20.

## Decision

Streamer is removed. Raw HTTP is the only live collection transport on every supported frontend.

Collection does not touch the traffic source, pace output, pick the freshest sample, or apply temporal admission. The CSI callback runs the fail-closed provenance classifier for the configured generator, then pushes classified frames into a preallocated 16-record lock-free ring. A worker task sends them in order, up to four per chunk. When the ring is full the newest record is dropped and counted. Every offered frame gets its 64-bit stream sequence before it is queued, so a drop always leaves a visible gap.

Each frame on the wire is a 60-byte HTTP prefix (session and stream sequence), a CSI V8 header (PHY mode, LTF type, channel width, timing, device sequence, chip, RSSI, channel, length), and the payload. The stream framing is version `1`; firmware and tools support only CSI V8 records. The collector writes the PHY fields into every `.npz`. Older datasets without them are read as HT, HT-LTF, 20 MHz when their layout confirms it. Historical captures are read from the decoded NPZ arrays, so they need no binary parser.

`./espectre collect` switches the device to `external` traffic and keeps that setting. It drives the standalone, standard-library-only generator in `tools/ha_traffic_generator_addon/espectre_traffic_generator.py`, whose `--pps` sets only the UDP rate and the dataset provenance. External datagrams carry exactly `F0 9F 91 BB` (`"👻"` in UTF-8). The web raw tool uses whatever traffic configuration the device already has and has no PPS control.

## Decision history

| Date | Direction | Resolution |
| --- | --- | --- |
| 2026-07-03 | Dedicated C++ Streamer with collector-paced UDP and TX backpressure | Replaced once every sensing frontend had raw collection |
| 2026-07-19 | Keep per-record PHY provenance in Streamer V7 datasets | Kept as an invariant in CSI V8 |
| 2026-08-25 | Remove Streamer, collect through raw HTTP | Accepted |
| 2026-08-25 | Pace or replace samples inside the HTTP data plane | Rejected: transport feedback would choose which records reach the dataset |
| 2026-08-29 | Publish the raw HTTP framing as version `1` instead of `2` | Accepted; the record stays V8 |
| 2026-09-22 | Support only the latest CSI record version | Accepted; V8 unchanged, NPZ datasets still readable |

## Alternatives considered

- **Keep Streamer as a fallback.** Two firmware, discovery, protocol, and release surfaces, and collection that depends on the transport.
- **Credits or adaptive pacing on raw HTTP.** A fixed buffer with counted drops keeps loss measurable without letting the transport pick data.
- **Store everything as generic HT20.** A shared normalized grid is not a shared PHY or LTF; dropping the metadata blocks later audits.
- **Switch traffic mode only for the session.** Collection is a deliberate operator action, and keeping `external` makes traffic ownership explicit afterwards.
- **Apply temporal admission before collection.** Raw data must keep the classified input and timing so detector views can replay the deployed admission later.

## Consequences

All maintained frontends share one collection contract and one Direct discovery path. A dataset sees every classified frame except counted ring drops: after drain, `fresh_record_total + raw_drop_total == classified_frames_offered_to_raw`. PHY and timing provenance survived the move. Old Streamer reviews, benchmarks, and datasets remain as evidence; its code, builds, CI, and release assets are gone.

## Related

- [`2026-07-23-adopt-classifier-first-ht20-sensing-contract.md`](2026-07-23-adopt-classifier-first-ht20-sensing-contract.md)
- [`2026-08-17-adopt-improv-serial-and-direct-http-for-local-control.md`](2026-08-17-adopt-improv-serial-and-direct-http-for-local-control.md)
- [`2026-08-23-standardize-managed-csi-traffic-sources.md`](2026-08-23-standardize-managed-csi-traffic-sources.md)
- [CLI guide](../CLI.md)
- Commits `62128e07` (PHY provenance), `943d2c83` (collect over HTTP), `9b5ae041` (Streamer removal), `7a52b9e0` (framing v1), `22478b2f` (V8 only)
