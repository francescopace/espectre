# ADR: adopt Improv Serial and Direct HTTP for local control

- Status: Superseded in part
- Date: 2026-08-17
- Updated: 2026-08-26

Transport, port, CORS boundary, and discovery still hold. The `/request` RPC and the bearer-bound CSI session were replaced by the resource API in [`2026-07-02-use-one-resource-oriented-protocol-across-transports.md`](2026-07-02-use-one-resource-oriented-protocol-across-transports.md).

## Context

Native used its own BLE service for provisioning, recovery, configuration, status, and live sensing. BLE and Wi-Fi CSI share the radio: with BLE active, CSI occupancy fell from about 80–90% to 35–45%, so sensing had to pause during setup. And since BLE could only stop once Wi-Fi and MQTT were up, a broker became mandatory just to start sensing.

Improv Serial already handles Wi-Fi provisioning right after a browser flash, with no extra radio load. Once on the LAN, ESPectre needs one discovery and control surface of its own with a stable identity; the ESPHome and Matter discovery records do not provide that.

The web portal is served over HTTPS and devices speak plain HTTP on the LAN. A local `ws://` connection showed a visible security downgrade in the browser. Streaming `fetch()` to local HTTP did not, and can request local-network access with `targetAddressSpace: "local"`. Events and raw CSI only flow from device to client, so they do not need WebSocket.

## Decision

The Native BLE service is removed. Native and ESPHome use standard Improv Serial, without private extensions, for first Wi-Fi setup and USB recovery. Matter keeps standard commissioning, and ESPHome keeps whatever its upstream integration owns.

Local control goes through one shared Direct HTTP service:

- TCP port `62587` (`0xF47B`, the low 16 bits of U+1F47B 👻) on Native, Matter, and ESPHome, separate from any port-80 server, captive portal, or native API;
- `GET /espectre/v1/events` streams processed events as SSE, with a 10 s heartbeat, fixed queues, at most two clients, telemetry coalescing, and eviction of slow clients;
- `GET /espectre/v1/csi` streams raw CSI as defined in the raw-collection ADR;
- every response sends `Cache-Control: no-store`, an exact Origin allowlist, CORS, and Private Network Access preflight;
- there is no local WebSocket endpoint, alias, or fallback.

MQTT stays optional on Native. Direct and MQTT carry the same messages through separate bounded queues. MQTT remains the path for Home Assistant discovery, automations, remote brokers, and multiple consumers, but local sensing does not need it.

Native stages remote Wi-Fi changes, checks association and addressing, and falls back to the last working network on failure. Improv Serial and the physical recovery action work without the portal or MQTT. Stored Wi-Fi and MQTT passwords are never returned over Direct.

Every frontend publishes `_espectre._tcp.local.` through its existing mDNS responder. The TXT record carries `device_id`, frontend, TXT and protocol versions, transport, endpoint paths, firmware, chip, and coarse capabilities. `./espectre devices` browses only this record and filters on `frontend`; it never reads `_esphomelib`, `_matterc`, or other upstream records. When multicast fails, explicit IPs, unique `.local` names, device IDs, remembered endpoints, and Improv Serial still work. Direct mutations go through the shared command engine; ESPHome republishes the affected entities, and Matter keeps commissioning and occupancy in Matter while using Direct for the controls Matter lacks.

The LAN is the trust boundary. Devices still restrict browser Origins, validate every request, limit clients, mutation rate, frame size, and queue depth, bind only to the station interface, and add no UPnP. A future optional relay would use outbound authenticated WSS from devices and WSS from browsers, be documented and self-hostable, and never carry raw CSI. Local Direct HTTP stays the default.

## Decision history

| Date | Direction | Resolution |
| --- | --- | --- |
| 2026-03-17 | First-party BLE for provisioning, control, and live sensing | Dropped: BLE coexistence cut CSI occupancy |
| 2026-08-17 | BLE only for setup and recovery, MQTT as the sensing plane | Replaced: still needed a broker to start sensing |
| 2026-08-23 | Remove BLE, provision with Improv Serial, Direct WebSocket on the LAN | BLE removal and Improv kept; WebSocket replaced after browser mixed-content tests |
| 2026-08-24 | One `_espectre._tcp` record and one Direct bridge for all C++ frontends | Kept, with HTTP endpoint metadata |
| 2026-08-25 | HTTP POST for commands, SSE for events, binary HTTP for raw CSI | Accepted |
| 2026-08-26 | Same messages over Direct HTTP and MQTT | Kept as the protocol contract |

## Alternatives considered

- **Keep BLE for setup and add Direct.** Keeps the radio cost and duplicates SDK, firmware, and portal surfaces.
- **Require MQTT after Improv.** A browser and a device on the same LAN can talk without a broker.
- **Keep local WebSocket and live with the warning.** Users see the downgrade, and browsers may block it outright later.
- **TLS certificates for `.local` devices.** Public CAs cannot validate private `.local` names, and a private root needs setup on every client.
- **Read ESPHome and Matter discovery records.** Ties the CLI to foreign schemas and gives no stable identity after Matter commissioning.
- **One service type per frontend.** Duplicates browsing and parsing, and every new frontend becomes a host-protocol change.
- **Rely on browser mDNS.** Web pages cannot enumerate DNS-SD, so manual and remembered endpoints must stay first-class.
- **Require a managed relay.** Setup and sensing must work with no account, external service, or Internet.

## Consequences

Native can be provisioned over USB, sense on Wi-Fi, and be monitored locally with no BLE and no broker. All C++ frontends share one Direct port, discovery record, command model, and event stream, and BLE no longer competes with CSI. Browser local-network permission and Origin policy become release-critical. Old BLE and WebSocket clients have to migrate, and recovery relies on Improv Serial, the physical reset, explicit addressing, or MQTT where configured.

## Related

- [API reference](../API.md), [discovery](../DISCOVERY.md), [architecture](../ARCHITECTURE.md), and [SDK guide](../SDK.md)
- [`2026-07-02-use-one-resource-oriented-protocol-across-transports.md`](2026-07-02-use-one-resource-oriented-protocol-across-transports.md)
- [`2026-07-03-unify-raw-csi-collection-over-http.md`](2026-07-03-unify-raw-csi-collection-over-http.md)
- Commits `cd9bec13` (BLE API), `a245dafd` (BLE for setup only), `8f0e81d3` (BLE removed), `781aa8f0` (unified discovery), `49af058c` (HTTP and SSE)
