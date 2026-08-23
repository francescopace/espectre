# Native BLE to Local WebSocket Migration Review

## Review status

- Date: 2026-08-23
- Status: Proposed
- Scope: ESPectre Native firmware, complete removal of the first-party BLE surface from shared C++, the published SDK, tests, current documentation, and the public web portal, shared ESP-IDF runtime support, release tooling, and an optional Chrome discovery extension
- Out of scope: removing ESPHome `esp32_improv`, removing Matter-standard commissioning owned by the Matter stack, changing the Streamer UDP data protocol, exposing raw CSI through WebSocket, and making MQTT unavailable
- Related decision: `docs/adr/2026-08-17-keep-native-ble-as-setup-recovery.md`

This document records the implementation plan and release gates. Completing the review means that the plan is accepted; resolving the review means that every blocking task and exit criterion is complete.

## Proposed outcome

Native firmware should use standard Improv Serial for initial Wi-Fi provisioning and recovery, then expose a versioned local WebSocket API for configuration, status, telemetry, diagnostics, and runtime commands. MQTT remains an optional transport for Home Assistant, automation, remote deployments, and multi-device consumers, but it must no longer be required to configure the device, open Monitor, or start sensing from the web portal.

The migration is a single cutover, not an overlap release. The new Native firmware must not contain both the first-party BLE stack and Direct WebSocket support. Implementation may use intermediate local commits, but the released and tested result removes BLE before Direct firmware is distributed.

The portal should support the complete workflow without Web Bluetooth:

1. Flash firmware with ESP Web Tools.
2. Offer standard Improv Serial Wi-Fi provisioning while the device is connected over USB.
3. Open Direct mode using the device address returned or inferred after provisioning.
4. Read the device identity and Wi-Fi state, reconcile the selected BSSID with the device, and configure the device name and optional MQTT settings over WebSocket.
5. Monitor and control the device over the local WebSocket or, when selected, MQTT over WebSocket.

The device should advertise its Direct endpoint through mDNS/DNS-SD. Because a normal web page cannot enumerate mDNS services, the portal must always retain manual IP or `.local` entry, remembered devices, and a QR/share-link path. An optional Chrome extension may add automatic desktop discovery; the core workflow must not depend on that extension.

## Monitor UI and UX decision

Monitor should begin with a connection-method choice and converge on one dashboard after connection:

| Method | Positioning | Required input | Intended use |
| --- | --- | --- | --- |
| Direct WebSocket | Default and recommended for testing sensing | Remembered device, discovered device, IP address, `.local` hostname, or supported local URL | One device on the same LAN, no broker required |
| MQTT | Advanced integration path | Broker WebSocket URL, credentials when required, and the existing device or topic selection | Home Assistant, automation, remote access, and multi-device monitoring |

After connection, both methods must use the same normalized device model, charts, status, diagnostics, sensing controls, and capability-based OTA controls. The connection label in the site's device box must be exactly `WS` for a Direct WebSocket device, at the same hierarchy and visual weight as the existing `USB` and `MQTT` labels. The box should also show the active endpoint. Changing method must close the current socket or subscriptions before opening the new transport; it must not create two competing Monitor sessions accidentally.

The flash and Improv handoff should preselect Direct and populate the new device endpoint when available. Returning users may retain the preferred method and non-secret endpoints, but the portal must not persist Wi-Fi passwords, MQTT passwords, tokens, or telemetry payloads. Direct mode remains usable through manual addressing when discovery is unavailable.

## Architectural boundaries

- Remove the first-party BLE implementation completely from the shared C++ runtime, Native frontend, embeddable SDK facade, source lists, Kconfig, build defaults, tests, current documentation, and portal. Do not retain no-op bindings, compile-time BLE options, protocol commands, or a legacy web client.
- ESPHome keeps its independently owned upstream `esp32_improv` behavior, and Matter keeps commissioning owned by the Matter stack. Neither is implemented through the shared or Native ESPectre BLE surface being removed.
- Do not publish a Native overlap build containing both the removed BLE implementation and Direct WebSocket.
- Reuse the existing ESPectre message model across MQTT and Direct mode. Do not encode MQTT topic names as the WebSocket protocol.
- Use a versioned WebSocket subprotocol and envelope so incompatible clients fail explicitly.
- Serve processed telemetry and state, not raw CSI, over Direct mode.
- Keep CSI callbacks and detector loops independent of every network transport. WebSocket and MQTT publication must use separate bounded asynchronous queues, replace superseded telemetry with the latest sample, and preserve command results and state transitions.
- Require Wi-Fi for Direct mode. Treat MQTT as an optional configured capability.
- Keep Wi-Fi recovery independent of the portal and MQTT. Standard Improv Serial remains available whenever USB access is possible, and the existing physical recovery policy must be replaced with a documented non-BLE recovery path.
- Do not bind local access to an Improv or USB token. The device may be managed later from another browser or phone on the same trusted LAN.
- Treat the LAN as the trust boundary, while still validating browser origins, request sizes, schema, rate, and connection counts. Bind only to intended LAN interfaces, do not add UPnP exposure, and never return stored secrets.
- Reuse and generalize the Streamer mDNS implementation rather than creating a second lifecycle implementation.

## Current-state findings

| ID | Severity | Finding | Consequence |
| --- | --- | --- | --- |
| NWS-001 | Blocking | Native setup policy couples readiness to both Wi-Fi and MQTT, and starts BLE when either is missing. | Direct sensing cannot make MQTT optional until readiness and recovery policy are separated from transport availability. |
| NWS-002 | High | Native BLE owns both transport concerns and generic configuration commands. | Removing the transport directly would also remove configuration behavior unless command handling is made transport-neutral first. |
| NWS-003 | High | Wi-Fi configuration is persisted before the new connection is proven usable. | A failed remote update can strand a device that no longer has BLE recovery. |
| NWS-004 | High | Detector telemetry is produced on a latency-sensitive path, while a WebSocket client, MQTT broker, or MQTT acknowledgment path may be slow or unavailable. | Any synchronous publish or unbounded transport queue could block sensing or exhaust memory; the broker decouples MQTT consumers, but it does not remove device-to-broker backpressure. |
| NWS-005 | High | Browser clients can open local cleartext WebSockets, but arbitrary public pages must not gain an unrestricted configuration channel. | The server needs a narrow Origin policy and strict protocol validation even under a trusted-LAN model. |
| NWS-006 | Medium | Streamer already publishes mDNS, but Native does not, and standard browser pages cannot enumerate DNS-SD records. | Automatic discovery requires a browser-external helper; all clients still need deterministic fallback addressing. |
| NWS-007 | High | The portal Configure workflow and its tests are explicitly tied to Web Bluetooth. | BLE cannot be removed until Direct configuration, recovery guidance, analytics, and browser error states reach parity. |
| NWS-008 | High | BLE bindings are part of the published embeddable SDK and packaging scripts, not only the Native application. | A complete cutover is an intentional SDK break and must remove the facade include, source option, package metadata, Doxygen input, and embedding documentation in the same change. |
| NWS-009 | High | Chromium currently exposes `mdns` to ordinary extensions only through an allowlist, while the public `chrome.mdns` documentation remains under Platform Apps. | A generally installable Manifest V3 extension cannot be assumed to support mDNS; feasibility must be proven before committing to this distribution path. |
| NWS-010 | Blocking | The production portal is served over HTTPS, while a Native device would normally expose plain `ws://`; mixed-content and local-network policies differ by browser and are still evolving. | The central Direct workflow needs an early browser proof before the firmware protocol and portal are committed to plain WebSocket. |

## Work items

### Architecture and protocol

#### ARCH-01 — Record the durable decision

- Severity: Blocking
- Locations: `docs/adr/2026-08-17-keep-native-ble-as-setup-recovery.md`, new ADR under `docs/adr/`
- Work: Add an ADR for Improv Serial plus local WebSocket, including the trusted-LAN security model, optional MQTT role, discovery limitations, rejected alternatives, single-release cutover, SDK break, and recovery behavior. Mark the previous BLE ADR as superseded and cross-link both documents.
- Acceptance: The ADR explicitly rejects an overlap release, removes the published first-party BLE SDK surface, leaves ESPHome provisioning and Matter-standard commissioning under their existing frontend owners, and defines the release boundary and recovery requirements.

#### ARCH-02 — Freeze the Direct protocol contract

- Severity: Blocking
- Locations: `docs/ESPECTRE_PROTOCOL.md`, `src/cpp/runtime/espectre_protocol.h`, `src/cpp/runtime/espectre_protocol.cpp`
- Work: Define the WebSocket endpoint, versioned subprotocol, connection handshake, message envelope, request correlation, command results, error codes, maximum frame size, keepalive behavior, reconnect behavior, and capability negotiation. Map existing info, status, telemetry, diagnostics, configuration, runtime control, and OTA messages to transport-neutral message types.
- Acceptance: A client can determine compatibility before issuing a mutation; every request receives a correlated success or error result; MQTT and Direct mode share message semantics without requiring MQTT topic strings; unknown fields and versions have documented handling.

#### ARCH-03 — Define capability and compatibility policy

- Severity: High
- Locations: `docs/ESPECTRE_PROTOCOL.md`, `docs/ARCHITECTURE.md`, `docs/CHANGELOG.md`
- Work: Publish the Direct v1 capability set, the MQTT-only capability set, maximum supported clients, single-writer or multi-writer behavior, concurrent MQTT and Direct behavior, and the client/firmware compatibility window.
- Acceptance: Direct v1 covers setup, configuration, monitoring, sensing control, diagnostics, and supported OTA operations; Home Assistant discovery remains explicitly MQTT-only; the portal can hide unsupported controls from advertised capabilities.

#### ARCH-04 — Prove the browser-to-LAN transport

- Severity: Blocking
- Locations: disposable ESP-IDF or host WebSocket endpoint, production portal test page, supported browser matrix
- Work: From the HTTPS production origin, test plain WebSocket connections to a private IP literal and a `.local` hostname under the current stable versions of Chrome, Edge, Firefox, and Safari on claimed desktop and mobile platforms. Record mixed-content behavior, Chrome Local Network Access behavior, OS permission prompts, CSP requirements, and whether a user gesture is required.
- Acceptance: At least the declared support matrix connects without browser flags, disabled security, or certificate-warning workarounds. If it does not, select and record a viable architecture before DEV-02, such as a top-level device-hosted local UI, a support-scope restriction, or an accepted secure-local transport design.

### Device and shared runtime

#### DEV-01 — Make configuration handling transport-neutral

- Severity: Blocking
- Locations: `src/cpp/runtime/esp_idf/frontend_support/frontend_control_helpers.h`, `src/cpp/runtime/esp_idf/frontend_support/frontend_control_helpers.cpp`, `src/cpp/frontend/native/espectre/native_frontend.h`, `src/cpp/frontend/native/espectre/native_frontend.cpp`
- Work: Rename BLE-specific command result and helper types, move common validation and mutations behind a transport-neutral command dispatcher, and keep transport adapters responsible only for decoding, authorization context, and response delivery.
- Acceptance: Current MQTT behavior and the new Direct WebSocket invoke the same configuration and control implementation and return equivalent semantic results; no transport-neutral type retains a BLE-specific name after the cutover.

#### DEV-02 — Add a local WebSocket service

- Severity: Blocking
- Locations: new service under `src/cpp/runtime/esp_idf/`, `src/cpp/espectre_sources.cmake`, Native component build files
- Work: Implement the versioned endpoint using the ESP-IDF HTTP server stack, connection lifecycle, handshake, receive parsing, command dispatch, correlated responses, ping/pong or equivalent liveness, and clean shutdown on Wi-Fi loss or frontend stop.
- Acceptance: The service starts only after the station interface has a usable address, restarts after reconnection, rejects incompatible subprotocols and oversized or malformed frames, and releases all client and queue resources on disconnect.

#### DEV-03 — Protect the sensing path from transport backpressure

- Severity: Blocking
- Locations: new WebSocket service, `src/cpp/runtime/esp_idf/frontend_support/mqtt_transport_esp_idf.*`, `src/cpp/frontend/native/espectre/native_frontend.cpp`, shared runtime event/listener surfaces as needed
- Work: Fan out normalized runtime events into independent fixed-capacity outbound paths for Direct and MQTT. Coalesce or replace stale telemetry, preserve command results and state transitions, define slow-client and broker timeout behavior, bound QoS acknowledgment backlog, and avoid allocations, serialization, and network writes in CSI callbacks.
- Acceptance: A stalled browser, slow or unavailable broker, or delayed MQTT acknowledgment cannot block detector evaluation, grow memory without a bound, or stall the other transport; command responses are not silently replaced by telemetry; per-transport queue depth, drop count, reconnect count, and failure state are observable in diagnostics.

#### DEV-04 — Make MQTT optional

- Severity: Blocking
- Locations: `src/cpp/frontend/native/espectre/native_frontend.cpp`, `src/cpp/runtime/esp_idf/frontend_support/mqtt_transport_esp_idf.*`, Native configuration and status handling
- Work: Change readiness so a valid Wi-Fi connection can enter Direct sensing without MQTT credentials. Keep MQTT connection, Home Assistant discovery, and publication active when configured, allow MQTT configuration to be added, changed, or cleared through Direct mode, and define simultaneous fan-out behavior.
- Acceptance: A factory-reset device can provision Wi-Fi, open Monitor, and sense in Direct mode with no broker or MQTT configuration; adding MQTT does not restart sensing unnecessarily; clearing, losing, or slowing MQTT does not disable or degrade Direct mode.

#### DEV-05 — Make remote Wi-Fi changes recoverable

- Severity: Blocking
- Locations: `src/cpp/runtime/esp_idf/frontend_support/wifi_provisioning_service.h`, `src/cpp/runtime/esp_idf/frontend_support/wifi_provisioning_service.cpp`, Native recovery handling
- Work: Stage candidate credentials instead of immediately replacing the last-known-good configuration, verify association and address acquisition within a bounded window, commit on success, and roll back or expose a deterministic recovery mode on failure. Define how optional BSSID pinning falls back when the AP disappears or changes radios.
- Acceptance: Bad credentials, an unavailable BSSID, DHCP failure, and an interrupted apply operation do not permanently strand the device; secrets are write-only over Direct mode; the portal receives a final result or a documented reconnect outcome.

#### DEV-06 — Implement standard Improv Serial

- Severity: Blocking
- Locations: Native component manifest and CMake files, `src/cpp/frontend/native/app/main/app_main.cpp`, provisioning integration under `src/cpp/runtime/esp_idf/` or the Native frontend
- Work: Integrate the upstream ESP-IDF Improv Serial implementation without private protocol extensions. Expose standard Wi-Fi provisioning, provisioning state, device information, and the post-connect device URL expected by ESP Web Tools. Review and pin its permissive dependency and include it in license checks.
- Acceptance: ESP Web Tools can flash and provision a factory-reset Native device through the standard flow; an unmodified compatible Improv Serial client can provision it; custom MQTT, BSSID, and device-name settings remain Direct protocol operations.

#### DEV-07 — Replace BLE recovery and boot policy

- Severity: Blocking
- Locations: `src/cpp/frontend/native/espectre/ble_recovery_button_service.*`, `src/cpp/frontend/native/app/main/app_main.cpp`, `src/cpp/frontend/native/espectre/Kconfig.projbuild`
- Work: Define a physical recovery action that clears or temporarily bypasses failed Wi-Fi configuration and makes Improv Serial usable without starting BLE. Preserve safe button timing and accidental-reset protection, and report the recovery state through serial logs and status LEDs where available.
- Acceptance: Recovery works with MQTT unset, MQTT unavailable, Wi-Fi credentials invalid, and no browser session; the documented action is consistent across supported Native boards or explicitly lists board-specific differences.

#### DEV-08 — Generalize mDNS discovery for Native

- Severity: High
- Locations: `src/cpp/runtime/esp_idf/streamer_discovery_service.*`, new shared discovery owner if needed, Native component manifest and CMake files
- Work: Extract the common ESP-IDF mDNS lifecycle, then publish a Native TCP service such as `_espectre._tcp` with a stable `espectre-<device_id>.local` hostname. Include bounded TXT data for device ID, mutable display name, frontend, protocol version, endpoint path, firmware version, chip, TLS mode, and capabilities.
- Acceptance: Advertisement follows Wi-Fi address lifecycle, hostname identity does not change when the display name changes, TXT updates are atomic from a client perspective, and Streamer discovery retains its current UDP service contract.

#### DEV-09 — Enforce the local security boundary

- Severity: High
- Locations: WebSocket service, Native Kconfig and defaults, protocol documentation
- Work: Allow only configured production portal origins and documented local-development origins, define behavior for clients without an Origin header, validate every frame before dispatch, rate-limit mutations, cap clients and message sizes, bind only to intended interfaces, redact credentials, and avoid UPnP or WAN exposure.
- Acceptance: Cross-origin browser attempts from unapproved sites fail; non-browser integrations have an explicit supported policy; read APIs never return Wi-Fi or MQTT passwords; malformed and abusive clients are disconnected without destabilizing sensing.

#### DEV-10 — Remove BLE from every first-party surface

- Severity: Blocking
- Locations: `src/cpp/runtime/ble_bindings.h`, `src/cpp/runtime/ble_bindings_noop.h`, `src/cpp/runtime/ble_protocol.h`, `src/cpp/runtime/esp_idf/frontend_support/ble_bindings_nimble.*`, `src/cpp/frontend/native/espectre/ble_recovery_button_service.*`, `src/cpp/frontend/native/espectre/native_frontend.*`, `src/cpp/frontend/native/app/main/app_main.cpp`, `src/cpp/frontend/native/espectre/CMakeLists.txt`, `src/cpp/frontend/native/espectre/Kconfig.projbuild`, `src/cpp/frontend/native/app/sdkconfig.defaults*`, `src/cpp/espectre_sdk.h`, `src/cpp/espectre_sources.cmake`, `src/cpp/CMakeLists.txt`, `src/cpp/Kconfig.projbuild`, `src/cpp/Doxyfile`, `.github/scripts/build_sdk_package.py`, owning C++ protocol and helper files, C++ tests and mocks, portal assets and tests, and current documentation
- Work: Delete the BLE binding interface, no-op binding, protocol constants, NimBLE implementation, recovery-button BLE service, Native orchestration, `set_ble`, `STOP_BLE`, BLE status and capability fields, BLE sysinfo formatting, NimBLE and `bt` dependencies, `CONFIG_BT*` defaults, Kconfig choices, CMake source groups, SDK facade includes, Doxygen inputs, package options, mocks, fixtures, Web Bluetooth client and modes, analytics mappings, portal markup, and current documentation. Rename generic helpers that still contain BLE terminology. Do not ship a BLE-enabled Native variant or retain a hidden compile-time switch.
- Acceptance: Native binaries and SDK packages contain no first-party BLE or NimBLE code or dependency; `espectre_sdk.h` exposes no BLE type; build and package metadata offer no BLE option; protocol code contains no BLE command or status surface; C++ and web tests contain no BLE mock or supported behavior; the portal contains no Web Bluetooth call, asset, mode, command, or legacy page; current docs contain no Native BLE workflow. A repository-wide audit may retain only clearly historical ADR/changelog text, ESPHome-owned upstream Improv references, and Matter-owned commissioning references. This task closes only after TEST-03, TEST-06, WEB-07, DOC-01, DOC-02, DOC-03, and CI-01 close.

#### DEV-11 — Measure and lock resource impact

- Severity: High
- Locations: Native build matrix and release artifacts, performance documentation if a durable metric is adopted
- Work: Record firmware image size, partition headroom, static and peak heap, task stack high-water marks, socket count, per-transport queue depth and drops, and detector-loop occupancy before BLE removal, with Direct idle, Direct active, MQTT active, and Direct plus MQTT active.
- Acceptance: The final firmware is smaller than or materially no larger than the BLE baseline, fits every supported partition, stays within heap and stack budgets, and does not regress detector timing gates.

### Web portal

#### WEB-01 — Add a transport-neutral browser protocol layer

- Severity: Blocking
- Locations: `docs/web/assets/js/espectre-mqtt.js`, new Direct/protocol modules under `docs/web/assets/js/`, `docs/web/assets/js/app.js`
- Work: Extract reusable schema validation, request correlation, capability handling, state normalization, and error representation from transport-specific code. Add a Direct WebSocket client with bounded reconnect, request timeouts, liveness, and explicit close reasons.
- Acceptance: Configure and Monitor consume a common device model regardless of Direct or MQTT transport; the same Monitor components render both paths; connection failures are distinguishable from device command failures and protocol-version failures.

#### WEB-02 — Connect flash, Improv, and Direct setup

- Severity: Blocking
- Locations: `docs/web/assets/js/app.js`, ESP Web Tools manifest construction, setup content under `docs/web/content/`
- Work: Preserve the standard ESP Web Tools Improv prompt after flashing, consume or infer the provisioned device URL, and offer an immediate transition to Direct setup. Provide clear fallback instructions when the serial session, local hostname resolution, or automatic redirect is unavailable.
- Acceptance: A new user can flash, provision Wi-Fi, and reach Direct configuration without Bluetooth; canceling Improv does not block manual setup or reflashing.

#### WEB-03 — Replace BLE Configure with Direct Configure

- Severity: Blocking
- Locations: `docs/web/assets/js/app.js`, `docs/web/assets/js/espectre-ble.js`, Configure markup and styles
- Work: Move Wi-Fi scan and selection, BSSID reconciliation, device-name editing, MQTT add/change/clear, configuration status, and command results to Direct mode. Read the associated BSSID and update the portal selection when it differs from the originally selected AP. Remove Web Bluetooth feature detection, pairing flows, and BLE-only copy in the same cutover once the Direct owning tests exist.
- Acceptance: All supported Native configuration operations work over Direct mode; the UI clearly separates SSID from optional BSSID pinning; passwords and sensitive MQTT fields are never read back or stored in browser persistence.

#### WEB-04 — Add Direct mode to Monitor

- Severity: Blocking
- Locations: `docs/web/assets/js/app.js`, Monitor markup and styles, `docs/web/assets/js/espectre-mqtt.js`
- Work: Open Monitor with two explicit choices: `Direct WebSocket`, selected by default and described as the broker-free path for testing one LAN device, and `MQTT`, presented as the advanced path for integrations, remote access, and multiple devices. Show only the fields relevant to the selected method, reuse one dashboard after connection, and retain a deliberate method switch without duplicating monitoring state. Map the connected Direct mode to the exact device-box label `WS`, alongside `USB` and `MQTT`, and show its endpoint without exposing credentials.
- Acceptance: Direct mode can start and observe sensing without a broker; the Improv handoff opens Monitor with Direct and the device endpoint prefilled; a connected Direct device box shows `WS`, never `Direct`, `WebSocket`, or `BLE`, in the same label position used by `USB` and `MQTT`; MQTT retains its current broker workflow; transport switching cleans up old subscriptions, sockets, timers, and pending requests; simultaneous portal tabs obey the server's client and writer policy; no secret is persisted with the selected method.

#### WEB-05 — Implement discovery fallbacks and remembered devices

- Severity: High
- Locations: `docs/web/assets/js/app.js`, portal storage helpers, setup and Monitor UI
- Work: Accept an IP address, a `.local` hostname, or a full supported local URL; remember only non-secret device endpoints with an explicit forget action; generate and consume a QR/share link that contains no credentials; offer the optional extension when compatible.
- Acceptance: A new browser or mobile device can connect without USB or the extension when given the hostname, IP, or QR link; stale entries fail cleanly and can be edited or forgotten; links cannot inject arbitrary schemes, paths, or script content.

#### WEB-06 — Handle browser security and local-network UX

- Severity: High
- Locations: portal connection UI, CSP and security headers, `docs/web/content/security.html`, browser support documentation
- Work: Document HTTPS-page-to-local-`ws://` constraints and current browser local-network permission prompts, preserve strict CSP, show actionable errors for DNS, permission, Origin, mixed-content, timeout, and protocol failures, and test the supported desktop and mobile matrix.
- Acceptance: The portal does not suggest that mDNS enumeration works without the extension; unsupported combinations receive a usable manual fallback; no security header is weakened globally just to make Direct mode work.

Browser policy must be revalidated when implementation starts. Chrome's Local Network Access rollout and its WebSocket coverage are evolving, while general mixed-content guidance still treats HTTPS-to-`ws://` as unsafe: [Chrome Local Network Access](https://developer.chrome.com/blog/local-network-access), [MDN WebSocket security considerations](https://developer.mozilla.org/en-US/docs/Web/API/WebSockets_API/Writing_WebSocket_client_applications).

#### WEB-07 — Remove BLE assets and protect analytics privacy

- Severity: Blocking
- Locations: `docs/web/assets/js/espectre-ble.js`, `docs/web/assets/js/app.js`, `docs/web/assets/js/analytics.js`, `docs/web/assets/js/browser-support.js`, `docs/web/index.html`, `docs/web/package.json`, authored content under `docs/web/content/`, `test/web/test_espectre_ble.mjs`, `test/web/test_site_structure.mjs`, portal analytics calls and privacy text
- Work: Delete the BLE client and its test module, remove Web Bluetooth feature detection, connection modes, handlers, commands, labels, recovery UI, script tags, setup cards, analytics mappings, and stale copy, and regenerate derived pages from their authored fragments. Add the `ws` transport mode and the `WS` device-box label without retaining a compatibility alias for `ble`. Exclude SSID, BSSID, password, MQTT host, username, topic, device ID, IP, hostname, message payloads, and local URLs from analytics.
- Acceptance: The production portal makes no Web Bluetooth call, ships no BLE module, recognizes no `ble` connection mode, sends no `set_ble` or `STOP_BLE` command, presents no legacy BLE page, and emits no local-device or credential data through analytics. Site structure tests require the exact transport labels `WS`, `USB`, and `MQTT` and forbid BLE assets and UI hooks.

### Optional Chrome discovery extension, subject to feasibility

#### EXT-00 — Prove that a distributable extension can access mDNS

- Severity: Blocking for extension work, non-blocking for the BLE migration
- Locations: disposable extension spike, Chrome stable on macOS, Windows, and Linux, Chromium extension feature metadata
- Work: Build the smallest Manifest V3 package that requests `mdns`, registers a filtered `_espectre._tcp.local` listener, and runs as both an unpacked extension and a Web Store-equivalent package. Confirm whether a non-allowlisted extension ID receives the permission and API in current stable Chrome. Do not treat Chrome Apps behavior as evidence for Chrome Extensions.
- Acceptance: Proceed with EXT-01 only if a generally distributable extension can discover a real service without a Chromium allowlist. Otherwise, record the extension as rejected or open a separate decision for a native-messaging companion, an Isolated Web App, or another installable helper; none of those alternatives is implicitly authorized by this review.

#### EXT-01 — Define and scaffold the extension package

- Severity: Medium
- Locations: new repository-owned extension directory selected by ARCH-01, CI and release packaging
- Work: If EXT-00 passes, create a Manifest V3 extension with a background service worker, the required non-optional `mdns` permission, narrowly scoped `externally_connectable` matches for the production ESPectre portal, no remote executable code, and a stable extension identity strategy for portal integration.
- Acceptance: The unpacked extension installs in supported Chromium desktop browsers, passes manifest validation, and has no permissions beyond those required for ESPectre discovery and the portal bridge.

#### EXT-02 — Implement bounded DNS-SD discovery

- Severity: Medium
- Locations: extension service worker and shared discovery parser
- Work: Browse only `_espectre._tcp.local`, validate SRV and TXT records, normalize device ID, display name, host, addresses, port, path, protocol version, and capabilities, track TTL expiry and service removal, and accept only local hostnames or private/link-local addresses.
- Acceptance: Devices appear and disappear without stale indefinite entries; malformed or oversized records are ignored; Streamer UDP advertisements are not presented as Direct WebSocket endpoints.

#### EXT-03 — Add the portal-extension bridge

- Severity: Medium
- Locations: extension message handlers, `docs/web/assets/js/app.js`, portal discovery UI
- Work: Define a minimal versioned message API that returns discovery records to allowed portal origins. The extension must not proxy WebSocket traffic, telemetry, credentials, configuration commands, or analytics.
- Acceptance: The portal detects compatible and incompatible extension versions, requests discovery only after a user action, handles absence or denial, and keeps manual discovery usable at all times.

#### EXT-04 — Prepare extension distribution and support

- Severity: Medium
- Locations: extension build and test scripts, store metadata, privacy and support pages, release workflow
- Work: Produce a reproducible archive, icons and screenshots, a concise privacy disclosure, support and uninstall instructions, versioning, changelog linkage, and a Chrome Web Store submission checklist. Verify other Chromium browsers only before claiming support.
- Acceptance: The packaged artifact matches reviewed sources, collects no data, contains no remote code, and can be updated without breaking the portal bridge compatibility window.

The extension feasibility gate is based on current Chromium feature metadata, which restricts the `mdns` permission for extension types to specific allowlisted IDs, and on Chrome's placement of the public API under Platform Apps: [Chromium `_permission_features.json`](https://chromium.googlesource.com/chromium/src/+/main/chrome/common/extensions/api/_permission_features.json), [Chrome `mdns` API](https://developer.chrome.com/docs/apps/reference/mdns), [Chrome external messaging](https://developer.chrome.com/docs/extensions/develop/concepts/messaging), and [Manifest V3 overview](https://developer.chrome.com/docs/extensions/develop/migrate/what-is-mv3). Recheck these sources when EXT-00 starts because browser capabilities can change.

### Automated tests and validation

#### TEST-01 — Extend protocol contract tests

- Severity: Blocking
- Locations: `test/cpp/suites/runtime/test_espectre_protocol.cpp`, browser protocol tests under `test/web/`
- Work: Cover envelope versions, capabilities, correlation, every Direct request and event family, invalid JSON and types, unknown messages, size limits, error mapping, and C++/JavaScript fixtures generated from one canonical schema or fixture owner.
- Acceptance: Firmware and browser tests prove the same wire examples and reject the same invalid boundary cases without duplicating mutable protocol constants.

#### TEST-02 — Add WebSocket service tests

- Severity: Blocking
- Locations: owning new C++ runtime suite, ESP-IDF HTTP server mocks under `test/cpp/mocks/esp_idf/`, `test/cpp/cmake/EspectreTestLib.cmake`
- Work: Test start and stop lifecycle, Wi-Fi loss and reconnect, handshake and Origin policy, connection limits, fragmented or malformed frames if supported, command dispatch, response delivery, backpressure, slow-client eviction, and cleanup.
- Acceptance: Tests assert service state, dispatched commands, queue bounds, disconnects, and resource cleanup rather than incidental log text.

#### TEST-03 — Remove Native BLE tests and cover the Direct frontend

- Severity: Blocking
- Locations: `test/cpp/suites/frontend/test_native_frontend.cpp`, `test/cpp/support/ble_bindings_mock.*`, Native frontend test support
- Work: Delete `test/cpp/support/ble_bindings_mock.*` and every BLE expectation from `test/cpp/suites/frontend/test_native_frontend.cpp` and CMake test support. Introduce a Direct transport boundary mock where necessary, move generic command tests to their owning shared suite, and retain Native tests for readiness, lifecycle, concurrent MQTT and Direct behavior, recovery, reset, and frontend capabilities.
- Acceptance: C++ tests and test-support libraries contain no BLE mock, include, source, command, status field, setup assumption, or conditional build path, and cover sensing with Wi-Fi only, with MQTT, and with both Direct and MQTT connected.

#### TEST-04 — Expand Wi-Fi provisioning safety tests

- Severity: Blocking
- Locations: `test/cpp/suites/runtime/test_wifi_provisioning_service.cpp`
- Work: Cover candidate credentials, commit, timeout, rollback, reboot during apply, invalid BSSID, BSSID fallback, credential clearing, last-known-good preservation, and write-only secret responses.
- Acceptance: Every failure path leaves either a reachable previous network configuration or the documented physical and Improv recovery path.

#### TEST-05 — Test shared mDNS lifecycle and records

- Severity: High
- Locations: shared discovery owner tests, `test/cpp/mocks/esp_idf/mdns.h`, `test/cpp/mocks/esp_idf/mdns.cpp`
- Work: Cover initialization ownership, Native and Streamer service registration, address loss, reconnect, TXT updates, stable hostname generation, length limits, and shutdown.
- Acceptance: Native and Streamer can use the shared implementation without changing Streamer service identity or leaking mDNS resources.

#### TEST-06 — Replace and extend portal tests

- Severity: Blocking
- Locations: `test/web/test_espectre_ble.mjs`, `test/web/test_espectre_mqtt.mjs`, `test/web/test_site_structure.mjs`, new or renamed owning Direct protocol suite
- Work: Delete `test/web/test_espectre_ble.mjs` and replace its owned configuration contracts with Direct connection, protocol, Configure, Monitor, reconnect, transport switch, remembered-device, URL validation, QR/share-link, Improv handoff, extension bridge, capability, privacy, and error-state coverage in the appropriate owning suites. Update structural assertions so BLE assets, modes, commands, hooks, and copy are forbidden rather than expected, and assert the device-box transport map `ws: 'WS'`, `usb: 'USB'`, and `mqtt: 'MQTT'`.
- Acceptance: `npm test` covers Direct and retained MQTT paths, proves that a connected Direct device shows the `WS` label, and detects accidental Web Bluetooth code, BLE compatibility aliases, or sensitive analytics fields.

#### TEST-07 — Test the Chrome extension

- Severity: Medium
- Locations: extension test suite and package validation script
- Work: Stub `chrome.mdns`, cover service add/change/remove and TTL, TXT validation, address filtering, message origin and schema checks, incompatible bridge versions, absent permissions/API, and deterministic packaging.
- Acceptance: Tests run headlessly in CI where possible, and the manual checklist covers an unpacked build against a real Native device and the production portal origin.

#### TEST-08 — Run the hardware and browser matrix

- Severity: Blocking
- Locations: release checklist or owning frontend documentation
- Work: Validate flash, standard Improv, Direct configuration, sensing, MQTT add/clear, mDNS, reconnect, recovery, OTA, slow-browser behavior, and slow or unavailable broker behavior on every supported Native target: ESP32, ESP32-C3, ESP32-C5, ESP32-C6, and ESP32-S3. Test supported Chrome, Edge, Firefox, and Safari combinations on desktop and mobile, recording extension support separately from manual Direct mode.
- Acceptance: Each claimed target and browser has a recorded result, and unsupported browser restrictions are documented. ESP32-S2 re-enablement is covered separately by HW-01 after the BLE-free cutover.

#### TEST-09 — Run resource and latency benchmarks

- Severity: Blocking
- Locations: Native build artifacts and existing performance owners
- Work: Compare BLE baseline and final binaries, boot heap, steady-state heap, per-transport queue pressure, reconnect churn, Direct plus MQTT fan-out, and detector occupancy. During active sensing, exercise a slow or paused Direct browser, a broker that stops reading, an unavailable broker, delayed QoS acknowledgments, and recovery of each transport while the other remains active.
- Acceptance: DEV-11 budgets pass, no detector or calibration parity changes are introduced, neither transport can stall the other or the sensing path, and repeated connect/disconnect cycles do not leak memory or tasks.

### Documentation, CI, and release

#### DOC-01 — Update device and architecture documentation

- Severity: Blocking
- Locations: `README.md`, `docs/SETUP.md`, `docs/ARCHITECTURE.md`, `docs/ESPECTRE_PROTOCOL.md`, `src/cpp/frontend/native/README.md`, `docs/EMBEDDING.md`
- Work: Remove Native BLE instructions, protocol sections, commands, capability fields, SDK references, build options, troubleshooting, and architecture claims. Replace them with Improv and Direct workflows, optional MQTT, local security assumptions, discovery and fallback paths, recovery, supported boards and browsers, Direct API behavior, and the reduced SDK surface.
- Acceptance: No current-state document presents first-party BLE as implemented, optional, supported, or planned for Native or the shared SDK, and no current link targets a deleted BLE file. Historical ADR and changelog entries remain explicitly historical or superseded. ESPHome upstream Improv and Matter-standard commissioning documentation remain accurate and clearly separate.

#### DOC-02 — Update public portal content

- Severity: High
- Locations: `docs/web/README.md`, `docs/web/content/guides/setup.html`, `docs/web/content/security.html`, relevant privacy, support, and roadmap fragments under `docs/web/content/`, `docs/web/sitemap.xml` when routes change
- Work: Remove Native Bluetooth setup, recovery, compatibility, and browser-support copy from all authored portal fragments. Document flash-to-Improv handoff, Direct connection, the `WS` device-box label, manual IP and `.local` entry, QR links, extension installation, browser permission prompts, trusted-LAN limits, MQTT alternatives, troubleshooting, and extension privacy. Rebuild generated pages from shared fragments.
- Acceptance: Mobile and extension-free setup paths are first-class; generated pages contain no Native BLE workflow or stale deleted-asset link; the connection labels are documented as `WS`, `USB`, and `MQTT`; public security and compatibility claims match the tested matrix; generated pages and sitemap are current.

#### DOC-03 — Update release records

- Severity: High
- Locations: active unreleased section of `docs/CHANGELOG.md`, `docs/ROADMAP.md` only if product sequencing changes, release notes and firmware manifest metadata
- Work: Describe the final cumulative user-visible state, single-release BLE removal, retained MQTT support, recovery requirements, browser support, and the SDK compatibility break. Keep superseded experiments in the ADR rather than the changelog.
- Acceptance: Release notes state that no overlap firmware or legacy BLE portal remains, explain USB reflash or supported OTA migration for existing devices, and explain how users regain access after an address or Wi-Fi change.

#### CI-01 — Update build, packaging, and dependency gates

- Severity: Blocking
- Locations: Native CI and release workflows, `.github/scripts/build_native_firmware.sh`, `.github/scripts/build_sdk_package.py`, component manifests, license compliance checks
- Work: Remove Native and SDK NimBLE configuration, `bt` dependencies, BLE source groups, package toggles, and build-matrix variants in the cutover change. Add and pin required HTTP server, mDNS, and Improv components, validate licenses, package extension artifacts separately, and preserve all supported Native target builds.
- Acceptance: Every Native target and SDK package builds without BLE, NimBLE, `CONFIG_BT*`, or a BLE feature option; dependency, SDK-surface, generated-API, and license gates pass; extension packaging cannot modify firmware release artifacts.

#### REL-01 — Ship a single BLE-free cutover release

- Severity: Blocking
- Locations: release plan, firmware manifests, migration and recovery guidance
- Work: Publish Improv Serial, Direct WebSocket, mDNS, optional MQTT, the updated portal, and complete BLE removal in one release. Do not publish or maintain a Native image in which BLE and Direct coexist. Record the last BLE release as the comparison baseline, not as a supported alternative transport.
- Acceptance: Every new Native artifact is BLE-free; the portal contains only USB, WS, and MQTT connection paths; existing installations have a documented OTA or USB-reflash migration; factory-reset and recovery flows work without a BLE fallback.

#### REL-02 — Define rollback and support procedures

- Severity: High
- Locations: Native troubleshooting documentation, portal recovery guidance, release checklist
- Work: Document reflashing, Improv recovery, physical network reset, finding a device by hostname or router lease, clearing remembered endpoints, downgrading when supported, and diagnosing Origin or browser local-network restrictions.
- Acceptance: Support can recover a device without BLE, MQTT, the Chrome extension, or the original browser profile, provided USB or the documented physical recovery mechanism is available.

### Final hardware enablement

#### HW-01 — Re-enable ESP32-S2 for ESPHome, Native, and Streamer

- Severity: High
- Locations: ESPHome example configurations, Native and Streamer target mappings and build scripts, CI and release matrices, firmware manifest generation and verification, CLI target registries, owning tests, hardware documentation, and public setup content
- Work: After the first-party BLE cutover is complete, restore ESP32-S2 as a supported build, release, web-flashing, and documented hardware target for ESPHome, Native, and Streamer. Add the required ESPHome board configuration, Native and Streamer ESP-IDF target mappings, CI and release jobs, factory and applicable OTA artifacts, manifest metadata, CLI aliases, and hardware validation. Keep Matter excluded because ESP32-S2 has no BLE and the supported Matter commissioning flow depends on BLE; adding a non-BLE Matter commissioning path requires a separate accepted decision.
- Acceptance: ESPHome, Native, and Streamer build and publish ESP32-S2 artifacts through the same CI, compliance, manifest, CLI, and release gates as their other supported targets; the web installer offers only the applicable ESP32-S2 images; hardware validation covers flash, provisioning, sensing, recovery, reconnect, and frontend-specific transports; current documentation lists ESP32-S2 support accurately; Matter produces no ESP32-S2 artifact and makes no ESP32-S2 support claim.

### Final browser acceptance

#### E2E-01 — Run the complete browser workflow on ESP32-S2

- Severity: Blocking
- Locations: production web portal, physical ESP32-S2 running Native firmware, local `Ohana` Wi-Fi network, and the local Home Assistant MQTT broker
- Work: Start from a factory-reset ESP32-S2 and use a supported browser to complete the real user journey without CLI or BLE assistance: flash the published Native image through the web portal; provision the `Ohana` Wi-Fi network through the standard Improv Serial handoff using the locally supplied Wi-Fi password; connect to the device through Direct WebSocket; set the Wi-Fi BSSID pin to `E6:FA:C4:20:19:DE`; return to Monitor, start sensing over Direct WebSocket, and confirm that live status, diagnostics, motion score, threshold, movement state, and control results update normally. Return to Configure over Direct WebSocket, configure the local Home Assistant MQTT broker at `homeassistant.local` using username `mqtt` and the locally supplied MQTT password, then return to Monitor, select MQTT, discover the same device, and confirm that live sensing data and state arrive through MQTT. The Home Assistant broker must expose the browser-compatible WebSocket listener required by Monitor. Enter passwords manually or load them from an untracked local test profile; never commit them, include them in screenshots or logs, persist them in browser storage, or send them through analytics.
- Acceptance: One uninterrupted recorded test run proves web flash, Improv Serial provisioning, automatic or manual Direct reconnection, exact BSSID persistence and association, broker-free WS sensing, MQTT configuration over WS, and subsequent MQTT monitoring on the physical ESP32-S2. Monitor receives multiple consecutive telemetry updates through both WS and MQTT, command results are correlated and successful, switching transports cleans up the previous browser session, no credential is returned by the device or retained by the portal, and the release evidence records the firmware version, browser and OS, device target, redacted endpoint details, timestamps, screenshots or observations, and pass or fail for every step.

## Delivery sequence and gates

### Phase 1 — Contract and decoupling

Complete ARCH-01 through ARCH-04 and DEV-01. Capture the released BLE firmware's CSI, size, memory, and upgrade baseline before deleting it. Add Direct protocol fixtures and preserve MQTT behavior.

Gate: the Direct contract, browser transport, recovery model, SDK decision, and migration compatibility window are accepted before implementation spreads across firmware and portal code.

### Phase 2 — Direct implementation and BLE cutover

Complete DEV-02 through DEV-10, WEB-01 through WEB-07, TEST-01 through TEST-06, DOC-01, and CI-01 in the same unreleased change series. Direct and Improv may be brought up locally before deletion for development sequencing, but no maintained build, release candidate, portal mode, or compatibility switch may contain both the first-party BLE surface and the final Direct surface.

Gate: the only release candidate is BLE-free across C++, SDK packaging, tests, current documentation, and the portal; it can be flashed, provisioned, configured, monitored, recovered, and updated through USB, WS, or MQTT as applicable; a slow browser, broker, or MQTT acknowledgment path cannot affect sensing or the other transport.

### Phase 3 — Discovery and extension

Complete EXT-00 first. Complete EXT-01 through EXT-04 only if the feasibility gate passes. Treat any extension or replacement helper as an optional enhancement, and validate manual, remembered, `.local`, IP, and QR paths independently.

Gate: the discovery helper has a recorded go/no-go result, and Direct mode remains fully usable on mobile and non-Chromium browsers without automatic service enumeration.

### Phase 4 — Parity, hardware, and cutover release

Complete TEST-07 through TEST-09, DEV-11, DOC-02, DOC-03, REL-01, and REL-02. Run the repository-wide BLE audit and publish the cutover only after every blocking removal and recovery gate passes.

Gate: the supported firmware/browser matrix, resource budgets, recovery paths, migration guidance, and release upgrade path all pass; no Native binary, shared C++ or SDK surface, build definition, test, portal asset, current documentation, or supported workflow depends on the removed ESPectre BLE implementation; ESPHome-owned Improv behavior, Matter-owned commissioning, and Streamer discovery remain unchanged.

### Phase 5 — ESP32-S2 re-enablement

Complete HW-01 only after the BLE-free cutover is validated. Re-enable ESP32-S2 for ESPHome, Native, and Streamer without adding it to the Matter build or release matrix.

Gate: ESP32-S2 passes the same build, compliance, packaging, manifest, web-flashing, documentation, and applicable hardware gates as the other supported targets for ESPHome, Native, and Streamer; Matter remains excluded unless a separate decision introduces a supported non-BLE commissioning path.

### Phase 6 — Final browser acceptance

Complete E2E-01 last, against the production portal and published Native ESP32-S2 artifact, using the real local Wi-Fi and Home Assistant MQTT paths. Do not substitute unit tests, a host WebSocket server, simulated firmware, direct serial commands, or CLI configuration for any browser step.

Gate: the complete flash-to-WS-to-MQTT journey passes on physical ESP32-S2 hardware, the recorded evidence contains no credentials, and every failure is resolved and the full journey rerun from factory reset before the review closes.

## Required validation commands

Run the narrowest owning tests while implementing, then run the full gates before resolving the review:

```bash
cmake -S test/cpp -B test/cpp/build
cmake --build test/cpp/build
ctest --test-dir test/cpp/build --output-on-failure

cd docs/web
npm test

.venv/bin/pytest test/python/test_sdk_surface_invariants.py -v
python3 .github/scripts/generate_sdk_api.py
.venv/bin/pytest test/python/test_ci_pipeline.py test/python/test_license_compliance.py -v
python3 .github/scripts/build_static_pages.py
```

Run a case-insensitive repository audit for `BLE`, `Bluetooth`, `NimBLE`, `ble_bindings`, `ble_protocol`, `set_ble`, `STOP_BLE`, `ESPECTRE_SDK_ENABLE_BLE`, and `CONFIG_BT`. Every remaining match must be classified: historical ADR/changelog evidence, ESPHome-owned Improv, or Matter-owned commissioning is allowed; a Native/shared SDK implementation, test contract, web asset, current Native workflow, build option, or package option fails DEV-10.

Build the Native firmware for every supported target through `.github/scripts/build_native_firmware.sh`: `esp32`, `esp32c3`, `esp32c5`, `esp32c6`, and `esp32s3`. Run socket-binding tests outside the network sandbox when the sandbox rejects local sockets.

The hardware pass must additionally verify factory reset, flash, standard Improv Serial, automatic and manual Direct connection, Wi-Fi change success and rollback, BSSID fallback, device-name and MQTT updates, sensing without MQTT, simultaneous Direct and MQTT telemetry, mDNS re-advertisement, browser reconnect, physical recovery, OTA, a stalled WebSocket client, a slow or unavailable broker, delayed MQTT acknowledgments, power cycles, and repeated connection churn.

## Progress checklist

- [ ] ARCH-01 — Record the durable decision and supersede the BLE ADR.
- [ ] ARCH-02 — Freeze the Direct protocol contract.
- [ ] ARCH-03 — Define capability and compatibility policy.
- [ ] ARCH-04 — Prove the browser-to-LAN transport.
- [ ] DEV-01 — Make configuration handling transport-neutral.
- [ ] DEV-02 — Add the local WebSocket service.
- [ ] DEV-03 — Protect sensing from transport backpressure.
- [ ] DEV-04 — Make MQTT optional.
- [ ] DEV-05 — Make remote Wi-Fi changes recoverable.
- [ ] DEV-06 — Implement standard Improv Serial.
- [ ] DEV-07 — Replace BLE recovery and boot policy.
- [ ] DEV-08 — Generalize mDNS discovery for Native.
- [ ] DEV-09 — Enforce the local security boundary.
- [ ] DEV-10 — Remove BLE from every first-party surface.
- [ ] DEV-11 — Measure and lock resource impact.
- [ ] WEB-01 — Add a transport-neutral browser protocol layer.
- [ ] WEB-02 — Connect flash, Improv, and Direct setup.
- [ ] WEB-03 — Replace BLE Configure with Direct Configure.
- [ ] WEB-04 — Add Direct mode to Monitor.
- [ ] WEB-05 — Implement discovery fallbacks and remembered devices.
- [ ] WEB-06 — Handle browser security and local-network UX.
- [ ] WEB-07 — Remove BLE assets and protect analytics privacy.
- [ ] EXT-00 — Prove that a distributable Chrome extension can access mDNS.
- [ ] EXT-01 — Define and scaffold the Chrome extension package.
- [ ] EXT-02 — Implement bounded DNS-SD discovery.
- [ ] EXT-03 — Add the portal-extension bridge.
- [ ] EXT-04 — Prepare extension distribution and support.
- [ ] TEST-01 — Extend protocol contract tests.
- [ ] TEST-02 — Add WebSocket service tests.
- [ ] TEST-03 — Remove Native BLE tests and cover the Direct frontend.
- [ ] TEST-04 — Expand Wi-Fi provisioning safety tests.
- [ ] TEST-05 — Test shared mDNS lifecycle and records.
- [ ] TEST-06 — Replace and extend portal tests.
- [ ] TEST-07 — Test the Chrome extension.
- [ ] TEST-08 — Run the hardware and browser matrix.
- [ ] TEST-09 — Run resource and latency benchmarks.
- [ ] DOC-01 — Update device and architecture documentation.
- [ ] DOC-02 — Update public portal content.
- [ ] DOC-03 — Update release records.
- [ ] CI-01 — Update build, packaging, and dependency gates.
- [ ] REL-01 — Ship and validate a single BLE-free cutover release.
- [ ] REL-02 — Define rollback and support procedures.
- [ ] HW-01 — Re-enable ESP32-S2 for ESPHome, Native, and Streamer after the BLE-free cutover.
- [ ] E2E-01 — Run the complete browser workflow on a physical Native ESP32-S2.

## Resolution criteria

This review is resolved only when every Blocking item is complete, every High item is complete or explicitly deferred in the accepted ADR, required tests and hardware checks pass, resource measurements are recorded, and the single cutover release removes the first-party BLE surface from shared C++, Native, the SDK, tests, the portal, current documentation, build metadata, and package tooling without changing ESPHome-owned Improv behavior, Matter-owned commissioning, or the Streamer data protocol.
