# ESPectre Discovery

This document owns DNS-SD, mDNS, browser bootstrap discovery, and the `/devices` resource. The device API is specified in [API.md](API.md).

## DNS-SD and mDNS

Every networked frontend publishes `_espectre._tcp.local.` on TCP port `62587`. Its stable host name is `espectre-{device_id}.local`. The service instance may use the configured label, but consumers use `device_id` as identity.

The TXT record contains:

| Key | Value |
| --- | --- |
| `txtvers` | `1` |
| `protovers` | `1.0` |
| `device_id` | 16 lowercase hexadecimal characters |
| `name` | Effective display name |
| `frontend` | `native`, `esphome`, `matter`, or `micro` |
| `transport` | `http` |
| `path` | `/espectre/v1` |
| `firmware` | Running firmware version |
| `chip` | Active ESP-IDF target |
| `capabilities` | Bounded comma-separated coarse features |

There is no `events` TXT key. Clients derive `/events`, `/csi`, and resource URLs from `path` and the negotiated `capabilities` resource. Unknown TXT versions or protocol versions are incompatible.

`./espectre devices` performs a fresh browse and validates the ESPectre-owned record. It does not depend on `_esphomelib`, `_matterc`, or another upstream service schema. Discovery results are bounded, deduplicated by `device_id`, and sorted deterministically.

## Browser bootstrap

Ordinary web pages cannot enumerate mDNS services. The portal therefore resolves a nonce-scoped bootstrap name, `espectre-devices-{nonce}.local`, and requests `GET /espectre/v1/devices` from the selected responder. Native, ESPHome, and Matter can answer this one-shot request by performing a bounded DNS-SD browse. The response contains scan status and a `devices` array with validated device metadata, addresses, and the Direct base endpoint.

The responder does not retain a peer inventory. Concurrent discovery returns `409`, and a completed or timed-out scan releases its state. Results never flow through SSE or MQTT.

## Limits and fallback

At most eight devices and two local addresses per device are returned. Candidates with invalid identity, schema, transport, path, capability tokens, or non-local addresses are rejected. Conflicting records for one identity are excluded rather than guessed.

Micro publishes its own DNS-SD record but does not implement `/devices` or the bootstrap responder. If no Native, ESPHome, or Matter responder is reachable, connect with the device IP, the unique `espectre-{device_id}.local` host name, a remembered endpoint, or Improv Serial. Routed networks, multicast filtering, client isolation, and browser local-network permissions can prevent discovery without preventing direct connectivity.

## Validation

Validate discovery by checking the SRV target and port, every required TXT key, `path=/espectre/v1`, absence of the legacy `events` key, and agreement between the record and `GET /device` plus `GET /capabilities`. A browser bootstrap result must also pass the bounded identity, address, deduplication, and compatibility rules above.
