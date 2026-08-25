# ADR: replace local Direct WebSocket with HTTP

- Status: Accepted
- Date: 2026-08-25
- Supersedes: `2026-08-23-replace-native-ble-with-direct-websocket.md`, `2026-08-24-unify-frontend-discovery-and-direct-control.md`

## Context

The production portal is served over HTTPS, while first-party devices expose a trusted-LAN cleartext service. Browser mixed-content handling marks an HTTPS page that opens `ws://` as insecure even when Local Network Access permits the connection. That visible downgrade is not an acceptable production experience. A streaming `fetch()` to the same local device was validated without changing the page security indicator and can request local-network access through `targetAddressSpace: "local"`.

The Direct message envelopes and shared command engine are transport-neutral. Processed events are server-to-client, so SSE supplies the required direction without WebSocket framing. Native raw CSI also needs only a server-to-client binary response once an authenticated collection session has been started.

## Decision

Replace the local Direct WebSocket server and every first-party Direct WebSocket client with a shared HTTP service:

- `POST /espectre/v1/request` carries correlated JSON requests and responses;
- `GET /espectre/v1/events` carries versioned event envelopes as SSE with a 10-second heartbeat, fixed queues, at most two subscribers, telemetry coalescing, and slow-client eviction;
- Native C3 may advertise `GET /espectre/v1/csi`, a bearer-bound binary HTTP stream paced on the device at `target_pps`;
- Native, Matter, and Streamer listen on port `80`, while ESPHome listens independently on port `6054` and does not depend on ESPHome's port-80 web server or native API;
- every response uses `Cache-Control: no-store`, exact Origin allowlisting, CORS, and Private Network Access preflight handling;
- discovery TXT schema v2 declares `transport=http`, the request path, and the event path; and
- no local Direct WebSocket endpoint, subprotocol, alias, or automatic compatibility fallback remains.

The ESPectre v1 envelope, command engine, method catalog, and message families do not change. HTTP status reports transport-level parsing, policy, size, rate, and saturation failures. A valid request always receives a correlated response envelope.

Raw CSI start returns a random 128-bit session ID. The binary GET and stop request require the same bearer. A dedicated worker selects the freshest bounded sample at the requested cadence, emits a fresh V8 record or at most one no-sample heartbeat per second, and prefixes output with the 76-byte HTTP framing record. Credits and credit windows are removed. Abort, timeout, stop, network loss, reboot, or fault releases the session and restores the previous runtime state.

MQTT remains a separate user-owned integration channel. Browser MQTT is WSS-only and requires a browser-trusted certificate; device-to-broker TCP MQTT configuration is unaffected.

## Consequences

- the hosted portal can use local `fetch()` without the WebSocket mixed-content security downgrade;
- Configure, Monitor, Game, and Theremin share POST and SSE, while the raw CSI tool and CLI collector share the binary HTTP protocol;
- firmware keeps bounded queues and owner-task command dispatch without blocking CSI callbacks or runtime loops;
- older portals and firmware are intentionally incompatible at the Direct boundary, while manual IP, unique hostname, device ID, Improv Serial, MQTT, and Streamer UDP recovery paths remain; and
- a future optional relay must use outbound authenticated WSS from devices and WSS from browsers, remain protocol-documented and self-hostable, and never carry raw CSI. Local Direct HTTP remains the autonomous default.

## Alternatives considered

### Issue TLS certificates for `.local` devices

Rejected. A public CA cannot generally validate arbitrary private `.local` names, and installing a private trust root would make the browser workflow dependent on per-client administration.

### Keep local WebSocket and accept the warning

Rejected. The browser security downgrade is visible to every production user and can become a hard block as browser policy tightens.

### Require the managed relay

Rejected. Relay is useful for remote access and browser portability, but local setup and sensing must remain functional without an account, external service, or Internet connection.

### Use long polling

Rejected. SSE over streaming fetch provides bounded incremental delivery and heartbeats without repeated request overhead. Raw CSI uses a binary streaming response because SSE text encoding would add avoidable expansion and parsing cost.
