# ADR: unify frontend discovery and Direct control

- Status: Superseded
- Date: 2026-08-24
- Superseded by: `2026-08-25-replace-local-direct-websocket-with-http.md`

## Context

ESPectre frontends exposed different local discovery surfaces. Native published a first-party Direct WebSocket record, Streamer published a custom UDP record, ESPHome published its upstream native API record, and Matter exposed standard commissioning records only while a commissioning window was open. A generic host command therefore had to browse several service types and interpret schemas owned by other projects. Those schemas can change independently and do not consistently carry the canonical ESPectre `device_id`.

ESPHome and Matter also need a local detector-tuning surface. ESPHome must keep its Home Assistant entities aligned with runtime writes, while the standard Matter occupancy surface does not expose detector selection, threshold, debounce, or calibration controls. Streamer still needs a separate high-rate UDP data path, but status and diagnostics do not need a second protocol.

## Decision

Native, Streamer, ESPHome, and Matter publish one first-party DNS-SD service, `_espectre._tcp.local.`, for a Direct WebSocket endpoint. The TXT record carries `frontend`, the canonical `device_id`, TXT and protocol versions, endpoint path, firmware, chip, TLS mode, and coarse capabilities. Streamer also carries its UDP pacing port as TXT metadata; its SRV port always identifies Direct WebSocket.

`./espectre devices` performs one fresh PTR browse for this service and filters the normalized results by the `frontend` TXT field when requested. It does not maintain a cache and does not inspect `_esphomelib`, `_matterc`, or other upstream records. A populated result returns after a short quiet window; the configured timeout remains the upper bound when no complete record arrives.

The shared runtime Direct bridge supplies capability negotiation, identity, status, configuration, diagnostics, sensing controls, and supported runtime tuning methods to Streamer, ESPHome, and Matter. Native retains its broader Direct owner for Wi-Fi, MQTT, device labels, OTA, and the same sensing controls. ESPHome republishes entity state after accepted Direct mutations. Matter keeps commissioning and occupancy on Matter and uses Direct as the detector-tuning plane. Streamer keeps raw CSI and collector pacing on UDP; Direct never carries raw CSI.

ESPHome and Matter retain ownership of their existing mDNS responder. ESPectre adds and removes only `_espectre._tcp` so it does not reset the upstream responder, hostname, or standard services.

## Alternatives Considered

### Discover upstream ESPHome and Matter records

Rejected. It couples the CLI to schemas outside ESPectre, requires product heuristics, and cannot provide a stable ESPectre identity after Matter commissioning.

### Publish one custom service type per frontend

Rejected. It requires multiple PTR browses, duplicates parsing and lifecycle behavior, and makes every future frontend a CLI protocol change.

### Omit `_tcp` or `_udp` from the service name

Rejected. DNS-SD service types include an application label and a transport label. The discovered endpoint is a WebSocket over TCP for every frontend, even when a frontend also owns a separate UDP data plane.

### Carry Streamer raw CSI over Direct WebSocket

Rejected. The existing collector-paced UDP path owns high-rate raw CSI, batching, device-ID validation, and dataset capture. Direct is a bounded operational and tuning surface.

## Consequences

- one PTR browse discovers every current first-party C++ frontend;
- frontend filtering depends only on an ESPectre-owned TXT value;
- Matter remains discoverable after commissioning and gains a practical tuning plane;
- ESPHome runtime writes from Direct and Home Assistant converge on one state;
- Streamer discovery distinguishes its Direct SRV port from its UDP pacing metadata; and
- Micro-ESPectre remains undiscoverable until it implements the same Direct TCP contract.

## Related

- `docs/ESPECTRE_PROTOCOL.md`
- `docs/CLI.md`
- `docs/adr/2026-08-23-replace-native-ble-with-direct-websocket.md`
- `docs/adr/2026-07-03-adopt-a-dedicated-cpp-streamer-frontend-for-high-rate-csi-collection.md`
