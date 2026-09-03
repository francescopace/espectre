# ADR: adopt Improv Serial and Direct HTTP for local control

- Status: Superseded in part
- Date: 2026-08-17
- Updated: 2026-08-26

The transport choice, port, CORS boundary, and discovery service remain accepted. The `/request` RPC and explicit bearer-bound CSI session are superseded by the resource API in [`2026-09-03-adopt-resource-oriented-device-api.md`](2026-09-03-adopt-resource-oriented-device-api.md).

## Context

Native originally used first-party BLE for provisioning, recovery, configuration, status, and live sensing. BLE and Wi-Fi CSI share the ESP32 radio. Measurements showed that keeping BLE active reduced CSI occupancy from about 80–90% to 35–45%, so the runtime had to pause sensing during setup. Requiring both Wi-Fi and MQTT before BLE could stop then made a broker mandatory for the first supported sensing session.

Standard Improv Serial already provides Wi-Fi provisioning after a browser flash without adding another radio workload. Once the device joins the LAN, ESPectre needs one first-party discovery and control surface that carries the canonical ESPectre identity and command model. Upstream ESPHome and Matter discovery records do not provide that stable contract.

The production portal is served over HTTPS, while devices expose a trusted-LAN cleartext service. A local `ws://` connection caused a visible browser security downgrade. Streaming `fetch()` to local HTTP was validated without changing the page security indicator and can request local-network access through `targetAddressSpace: "local"`. Processed events and raw CSI are server-to-client streams, so neither requires WebSocket framing.

## Decision

Remove the first-party Native BLE surface. Use standard Improv Serial for initial Native and ESPHome Wi-Fi provisioning and USB recovery, without private protocol extensions. Matter retains its standard commissioning flow, and ESPHome retains any behavior owned by its upstream integration.

Expose local ESPectre control through one shared Direct HTTP service:

- `POST /espectre/v1/request` carries canonical correlated JSON requests and responses;
- `GET /espectre/v1/events` carries canonical processed events as SSE with a 10-second heartbeat, fixed queues, at most two subscribers, telemetry coalescing, and slow-client eviction;
- `GET /espectre/v1/csi` carries the capability-gated bearer-bound raw CSI stream defined by the raw-collection ADR;
- Native, Matter, and ESPHome listen on TCP port `62587` (`0xF47B`, the low 16 bits of `U+1F47B` GHOST), independent of frontend-owned port-80 servers, captive portals, and native APIs;
- every response uses `Cache-Control: no-store`, exact Origin allowlisting, CORS, and Private Network Access preflight handling; and
- no local Direct WebSocket endpoint, subprotocol, alias, or automatic compatibility fallback remains.

Native keeps MQTT optional. Direct and MQTT use the same application messages through independent bounded outbound queues. MQTT remains the Home Assistant Discovery, automation, remote-broker, and multi-consumer path; local sensing does not depend on it.

Native stages remote Wi-Fi changes, verifies association and address acquisition, and preserves or restores the last-known-good network on failure. Improv Serial and the documented physical recovery action remain available without the portal or MQTT. Stored Wi-Fi and MQTT passwords are never returned through Direct.

ESPectre frontends publish `_espectre._tcp.local.` through their existing mDNS lifecycle. The TXT record carries the canonical `device_id`, frontend, TXT and protocol versions, HTTP transport, endpoint paths, firmware, chip, and coarse capabilities. `./espectre devices` performs one fresh browse for this record and filters by the ESPectre-owned `frontend` field. It does not inspect `_esphomelib`, `_matterc`, or other upstream schemas. Explicit IP addresses, unique `.local` names, full device IDs, remembered endpoints, and Improv Serial remain deterministic fallbacks when multicast discovery is unavailable.

ESPHome and Matter add only the ESPectre service to the responder they already own. Direct mutations pass through the shared command engine; ESPHome republishes affected entity state, while Matter keeps commissioning and occupancy in Matter and uses Direct for the controls absent from standard occupancy clusters.

The local network is the access trust boundary. Devices still restrict browser Origins, validate every request before dispatch, limit clients, mutation rate, frame size, and queue depth, bind only to intended station interfaces, and do not add UPnP exposure. A future optional relay must use outbound authenticated WSS from devices and WSS from browsers, remain protocol-documented and self-hostable, and never carry raw CSI. Local Direct HTTP remains the autonomous default.

## Decision History

| Date | Direction | Resolution |
| --- | --- | --- |
| 2026-03-17 | Use first-party BLE for provisioning, runtime control, and live sensing | Rejected after BLE coexistence reduced CSI occupancy |
| 2026-08-17 | Restrict Native BLE to setup and recovery while keeping MQTT as the sensing plane | Replaced because it still made a broker mandatory for the first supported sensing session |
| 2026-08-23 | Remove first-party BLE, provision through Improv Serial, and use Direct WebSocket on the LAN | Retained the BLE removal and Improv boundary; replaced WebSocket after browser mixed-content validation |
| 2026-08-24 | Publish one `_espectre._tcp` record and one Direct bridge across C++ frontends | Retained with HTTP endpoint metadata instead of WebSocket metadata |
| 2026-08-25 | Use HTTP POST for commands, SSE for processed events, and binary HTTP for raw CSI | Accepted |
| 2026-08-26 | Carry the same canonical application messages over Direct HTTP and MQTT | Retained as the protocol contract |

## Alternatives Considered

### Keep BLE for setup and add Direct alongside it

Rejected. It preserves the radio coexistence cost and duplicate SDK, firmware, and portal surfaces.

### Require MQTT for all operation after Improv

Rejected. A browser and device on the same LAN can exchange local telemetry and commands without a broker.

### Keep local WebSocket and accept the warning

Rejected. The security downgrade is visible to production users and can become a hard block as browser policy tightens.

### Issue TLS certificates for `.local` devices

Rejected. Public certificate authorities cannot generally validate arbitrary private `.local` names, while a private trust root requires per-client administration.

### Discover upstream ESPHome and Matter records

Rejected. It couples the CLI to schemas outside ESPectre and does not provide a stable ESPectre identity after Matter commissioning.

### Publish one service type per frontend

Rejected. It duplicates browse, parsing, and lifecycle behavior and makes every frontend addition a host-protocol change.

### Depend on automatic browser mDNS enumeration

Rejected. Ordinary web pages cannot enumerate DNS-SD services, so manual and remembered endpoints remain first-class.

### Require a managed relay

Rejected. Local setup and sensing must work without an account, external service, or Internet connection.

## Consequences

- Native can provision over USB, sense on Wi-Fi, and expose local monitoring without BLE or an MQTT broker;
- every maintained C++ frontend uses one Direct HTTP port, discovery record, command model, and processed-event stream;
- BLE no longer competes with CSI or occupies the Native SDK and firmware surface;
- browser local-network permission and Origin policy remain release-critical compatibility boundaries;
- older BLE and Direct WebSocket clients require a deliberate migration; and
- recovery depends on Improv Serial, the physical network-reset action, explicit addressing, or MQTT where configured.

## Related

- [`../API.md`](../API.md)
- [`../ARCHITECTURE.md`](../ARCHITECTURE.md)
- [`../SDK.md`](../SDK.md)
- [`2026-07-02-use-one-message-model-and-command-engine-across-transports.md`](2026-07-02-use-one-message-model-and-command-engine-across-transports.md)
- [`2026-07-03-unify-raw-csi-collection-over-http.md`](2026-07-03-unify-raw-csi-collection-over-http.md)
