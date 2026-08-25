# ADR: replace Native BLE with Improv Serial and Direct WebSocket

- Status: Superseded
- Date: 2026-08-23
- Supersedes: `2026-08-17-keep-native-ble-as-setup-recovery.md`
- Superseded by: `2026-08-25-replace-local-direct-websocket-with-http.md`

## Context

Native firmware currently uses first-party BLE for Wi-Fi and MQTT setup, device naming, status, and recovery. BLE and Wi-Fi CSI share the ESP32 radio, and measurements behind the previous decision showed that keeping BLE active reduced CSI occupancy from about 80–90% to 35–45%. The resulting policy pauses sensing while BLE setup is active and requires both Wi-Fi and MQTT configuration before BLE can stop.

That policy makes a broker mandatory for the first sensing session even though the device can already communicate over the local Wi-Fi network. It also duplicates configuration behavior across BLE and MQTT, publishes BLE types through the embeddable SDK, and ties the public Configure workflow to Web Bluetooth support.

ESP Web Tools already supports the standard Improv Serial handoff after flashing. A local WebSocket can then carry configuration, state, processed telemetry, diagnostics, runtime commands, and OTA operations without radio coexistence. MQTT remains useful for Home Assistant, automation, remote deployments, and multiple consumers, but it need not be a prerequisite for local sensing.

The production portal is served over HTTPS, while a device on a private network would normally expose plain `ws://`. Browser mixed-content and local-network policies are therefore an architectural gate, not an implementation detail. The supported origin and browser matrix must be proven before the Direct server and portal workflow are treated as releasable.

## Decision

Replace the first-party Native BLE surface with standard Improv Serial provisioning and a versioned local Direct WebSocket API. Ship the change as one cutover: no maintained Native build, release candidate, SDK package, or portal mode may contain both the removed BLE surface and the final Direct surface.

The cutover has these boundaries:

- remove the first-party BLE bindings, GATT protocol, Native BLE recovery service, BLE SDK types, build options, tests, portal client, and current Native BLE documentation;
- retain ESPHome-owned `esp32_improv` behavior and Matter-standard commissioning under their existing frontend owners;
- use standard Improv Serial for initial Wi-Fi provisioning and USB recovery, without private protocol extensions;
- expose processed telemetry and state over Direct WebSocket, never raw CSI;
- keep MQTT optional and allow Direct and MQTT to operate concurrently through independent bounded outbound queues;
- keep detector callbacks and evaluation paths free of serialization, allocation, socket writes, broker waits, and unbounded transport work;
- stage remote Wi-Fi changes, verify association and address acquisition, and preserve or restore the last-known-good network on failure;
- replace the BLE recovery button behavior with a documented physical action that makes Improv Serial usable or clears failed network configuration without requiring the portal or MQTT;
- advertise the Direct endpoint through a shared ESP-IDF mDNS lifecycle, while always supporting manual IP address, `.local` hostname, remembered-device, and credential-free QR or share-link fallbacks; and
- treat any browser discovery extension as optional. Direct setup and monitoring must remain usable without it.

Direct uses the shared ESPectre message semantics through a WebSocket-specific envelope. The endpoint, subprotocol, request correlation, error model, limits, capabilities, and compatibility window are owned by `docs/ESPECTRE_PROTOCOL.md`. MQTT topics are not encoded into Direct messages.

The local network is the access trust boundary. The device still restricts browser Origins, validates every frame and field before dispatch, limits clients, mutations, frame size, and queue depth, binds only to intended station interfaces, and never returns stored Wi-Fi or MQTT passwords. It does not add UPnP exposure or bind access to a short-lived Improv or USB token, because an operator must be able to reconnect later from another browser on the same trusted LAN.

The production portal may connect directly from its HTTPS origin only for browser and OS combinations that pass the recorded transport matrix without flags, disabled security, or certificate-warning workarounds. If the matrix does not support the central HTTPS-to-local-WebSocket workflow, implementation must stop before the server contract is committed and this decision must be updated to select a proven origin model, such as a device-hosted top-level local UI or an explicitly narrower browser support boundary.

## Compatibility And Release Policy

The BLE removal is an intentional breaking change to the published SDK and Native operator workflow. The active release notes must identify the last BLE release as a migration baseline, state that no overlap firmware or legacy BLE portal remains, and document OTA or USB reflashing, Improv recovery, physical network reset, and endpoint rediscovery.

Direct v1 supports setup, configuration, device and runtime status, processed telemetry, diagnostics, sensing control, and the OTA operations advertised by the device. Home Assistant MQTT Discovery remains MQTT-only. Clients must derive available operations from the advertised capability set and must reject an incompatible WebSocket subprotocol or envelope version before issuing mutations.

Firmware accepts the current Direct major version only. Additive fields within that version are ignored when unknown; unknown required message types, malformed fields, oversized frames, and unsupported commands return a correlated error or close the connection according to the protocol contract. A future incompatible envelope or semantic change requires a new WebSocket subprotocol version.

## Alternatives Considered

### Keep BLE for setup and add Direct alongside it

Rejected. An overlap release retains the radio coexistence cost, preserves the duplicate SDK and portal surfaces, and makes the supported recovery and sensing policy ambiguous.

### Require MQTT for all operation after Improv

Rejected. It preserves the current broker prerequisite even though a browser and device on the same LAN can exchange local processed telemetry and commands directly.

### Bind Direct access to an Improv-issued token

Rejected. A short-lived USB bootstrap token prevents later management from another authorized device and does not replace an explicit LAN trust and Origin policy.

### Depend on automatic browser mDNS enumeration

Rejected. Ordinary web pages cannot enumerate DNS-SD services, and a generally distributable Manifest V3 extension cannot be assumed to receive the required mDNS API. Manual and remembered endpoints remain first-class.

The extension feasibility gate was closed as a no-go on 2026-08-23. [Chromium's current permission metadata](https://chromium.googlesource.com/chromium/src/+/main/chrome/common/extensions/api/_permission_features.json) grants `mdns` to ordinary extensions only for four allowlisted IDs; the unrestricted alternative is limited to the discontinued Platform App type on desktop systems. Chrome's public [`chrome.mdns` reference](https://developer.chrome.com/docs/apps/reference/mdns) also remains in the Platform Apps documentation. A new ESPectre Manifest V3 or Web Store extension therefore cannot meet the distribution requirement, so no extension package, bridge, or extension test suite is created. Reconsidering a native companion, Isolated Web App, or another installable discovery helper requires a separate decision.

### Expose raw CSI over WebSocket

Rejected. Direct is an operational and configuration surface. Raw CSI remains owned by the Streamer UDP data path and host collection workflow.

## Consequences

Benefits:

- Native can sense and expose Monitor on the LAN without an MQTT broker;
- BLE no longer competes with CSI or consumes Native firmware and SDK surface;
- Direct and MQTT share command semantics and normalized device state;
- MQTT remains available for integrations without controlling device readiness; and
- recovery no longer depends on a browser-capable Bluetooth stack.

Trade-offs:

- the portal must handle browser local-network permission and secure-origin constraints explicitly;
- Wi-Fi recovery and last-known-good rollback become release-critical because BLE fallback is removed;
- Native gains an HTTP/WebSocket server, per-transport queues, connection policy, and additional resource measurements;
- users on unsupported browser and network combinations need a documented manual or alternate-origin path; and
- existing Native BLE users must migrate in a single release boundary.

Resource baseline policy: the last released BLE binary remains the valid static-size comparison, but the missing BLE-era dynamic heap, task stack, socket, queue-occupancy, and drop measurements are explicitly deferred. Those values were not preserved by the released protocol or validation artifacts, and instrumenting a rebuilt historical source would change the memory layout without recreating the released toolchain and runtime conditions. The deferral does not infer passing values or waive the accepted candidate's resource and timing gates. The frozen Direct candidate and its bounded, observable transport diagnostics become the dynamic baseline for subsequent releases; the review record owns the measured evidence, exact budgets, and closure rationale.

## Related

- `docs/review/2026-08-23-native-ble-to-local-websocket-migration.md`
- `docs/ESPECTRE_PROTOCOL.md`
- `docs/ARCHITECTURE.md`
- `docs/EMBEDDING.md`
- `docs/adr/2026-07-02-use-a-shared-espectre-protocol-across-esp-idf-frontends.md`
- `docs/adr/2026-08-17-keep-native-ble-as-setup-recovery.md`
