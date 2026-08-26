# ESPectre Native Frontend

Native is the standalone ESP-IDF firmware for browser-based local setup, sensing over Direct HTTP, optional MQTT integration, Home Assistant MQTT Discovery, and HTTPS OTA. The shared message model is documented in [`ESPECTRE_PROTOCOL.md`](../../../../docs/ESPECTRE_PROTOCOL.md).

## Getting Started

The normal browser workflow is:

1. Open [Flash](https://espectre.dev/flash) in a supported Chromium browser and install the Native image for the detected chip.
2. Complete the standard Improv Serial prompt to provision Wi-Fi over USB.
3. Open Configure with the returned device URL, or enter the private IP, device name, full 16-character device ID, or last 6 ID characters.
4. Use Direct HTTP to inspect status, reconcile or pin the associated BSSID, edit the device label, and add optional MQTT settings.
5. Open Monitor and select Direct HTTP for broker-free sensing, or MQTT for Home Assistant, automation, remote brokers, and multiple devices.

Each `release`, `preview`, and `develop` channel publishes a full-flash image and an application-only OTA image for ESP32, ESP32-S2, ESP32-S3, ESP32-C3, ESP32-C5, and ESP32-C6. Matter does not publish an ESP32-S2 image because its supported commissioning flow requires a Bluetooth-capable target.

### Local ESP-IDF Workflow

Complete the shared [`Local Build Prerequisites`](../../../../docs/SETUP.md#local-build-prerequisites), then use the repository CLI:

```bash
./espectre native build --chip s2 --ota-channel develop --clean
./espectre native flash --chip s2 --port /dev/cu.usbmodemXXXX
./espectre monitor --port /dev/cu.usbmodemXXXX
```

The wrapper uses a local ESP-IDF installation when available and can use the pinned Docker image for builds. Flashing and serial monitoring require local tooling. `--ota-channel` selects the default release channel used when an OTA request omits one.

## Direct HTTP

Native starts `POST http://<device>:62587/espectre/v1/request` and `GET http://<device>:62587/espectre/v1/events` after Wi-Fi obtains an address. The production portal and `https://test.espectre.dev` validation origins are allowed by default; optional loopback development origins are controlled by Kconfig and remain disabled in published firmware. Requests use JSON, and events use SSE read through streaming `fetch` so the browser can request local-network access explicitly.

Direct mode provides:

- capability negotiation, device identity, status, configuration, diagnostics, and correlated command results
- Wi-Fi updates with optional BSSID and channel hints
- device-label and optional MQTT add, change, or clear operations
- sensing enable or pause through `set_sensing`, recalibration, detector selection, thresholds, hit counts, and traffic controls
- processed movement, state, calibration, diagnostics, and lifecycle events
- supported OTA status and control operations

The endpoints never return stored Wi-Fi or MQTT passwords. They cap request and response size, mutation rate, queued messages, and concurrent SSE subscribers. Telemetry may replace an older queued telemetry sample, while state transitions are preserved. MQTT uses its own 16-message frontend queue and bounded ESP-IDF outbox. The high-rate telemetry callback runs only while MQTT is connected or a Direct SSE client is present. Runtime callbacks only stage numeric sensing state; serialization and transport work run after detector evaluation returns.

Native images advertise bearer-bound raw CSI HTTP v2 collection. `start_raw_stream` on `/espectre/v1/request` accepts no rate parameter, returns a random 128-bit session ID, and one collector opens `/espectre/v1/csi` with that bearer. Classified frames bypass temporal sampling and enter a fixed 16-record ring; a dedicated worker sends up to four ordered V8 records per HTTP chunk with a 60-byte prefix and no pacing or replacement. Raw mode leaves the configured internal or external traffic source active and restores sensing on stop, abort, timeout, or network loss without changing persisted traffic configuration.

The device advertises `_espectre._tcp` through mDNS with a stable `espectre-<device_id>.local` hostname. [`ESPECTRE_PROTOCOL.md`](../../../../docs/ESPECTRE_PROTOCOL.md#mdnsdns-sd-discovery) owns the SRV and TXT record contract. Run `./espectre devices --frontend native` from the repository to enumerate advertised Native endpoints on an mDNS-visible LAN. Configure and Monitor present only private IP, device-ID, and device-name inputs: the portal maps a full ID to the unique hostname internally, while a name or 6-character suffix uses automatic discovery. One match connects directly, and multiple matches require explicit selection. Successful non-secret device references can be remembered or shared through a credential-free QR link.

Hosted HTTPS access to local cleartext HTTP depends on browser Private Network Access and mixed-content policy. The portal uses streaming `fetch` with `targetAddressSpace: "local"`, `Cache-Control: no-store`, and an explicit device Origin allowlist. Chrome 151 or later on macOS, Windows, and native Linux is the supported hosted path; compatibility with other browsers is not guaranteed. A local HTTP preview remains available for development as described in [`docs/web/README.md`](../../../../docs/web/README.md).

## Wi-Fi Provisioning and Recovery

Standard Improv Serial remains available through the primary serial console. It owns the Wi-Fi SSID and password and returns `https://espectre.dev/tools/configure/?target=<device-ip>`; Configure uses the target to prefill its Direct connection field. The same parameter also accepts a device name or ID when a browser link is shared. BSSID selection, Wi-Fi removal, device-label, MQTT, sensing, and OTA operations belong to Direct HTTP. Direct reports the current SSID and active band as read-only values but does not expose the Wi-Fi password or band selection.

Configure can scan asynchronously for access points that advertise the provisioned SSID. The station remains associated and Direct HTTP stays active during the scan, but off-channel radio work can briefly pause sensing and network traffic. Each protocol result contains the BSSID, channel, and RSSI; Configure displays the BSSID and signal strength, while retaining the channel only as an internal association hint. Choosing automatic selection clears both the BSSID pin and hint.

The Direct `clear_wifi_config` action removes the provisioned SSID and password, disconnects the station, and returns the device to Improv Serial provisioning. Configure asks for confirmation before sending it because the active Direct session normally closes before a response can be observed.

BSSID changes are staged. Native attempts the selected access point, commits it only after association and address acquisition, and rolls back to the last-known-good settings when the attempt fails or times out. After a successful commit or rollback, Native reboots once so the ESP32-C3 radio returns with a fresh CSI capture session; Direct and MQTT clients should reconnect after the device address becomes reachable again. If an optional BSSID is unavailable, the provisioning policy can retry the same SSID without the pin instead of permanently stranding the device.

Holding BOOT for `ESPECTRE_RECOVERY_BUTTON_HOLD_MS` clears saved Wi-Fi configuration and returns the device to Improv Serial provisioning. The default hold is 3 seconds. The default active-low GPIO is GPIO0 on ESP32, ESP32-S2, and ESP32-S3, GPIO9 on ESP32-C3 and ESP32-C6, and GPIO28 on ESP32-C5. Override or disable the input for boards that route BOOT differently.

Frontend-owned defaults in [`Kconfig.projbuild`](espectre/Kconfig.projbuild) are useful for reproducible lab images. Runtime provisioning stored in NVS takes precedence.

| Option | Purpose |
| --- | --- |
| `ESPECTRE_WIFI_SSID` | Initial Wi-Fi SSID |
| `ESPECTRE_WIFI_PASSWORD` | Initial Wi-Fi password |
| `ESPECTRE_WIFI_BSSID` | Optional AP-radio pin |
| `ESPECTRE_WIFI_BAND_2G`, `ESPECTRE_WIFI_BAND_5G`, `ESPECTRE_WIFI_BAND_AUTO` | Build-time band policy |
| `ESPECTRE_WIFI_CHANNEL` | Optional channel hint (`0` scans normally) |
| `ESPECTRE_RECOVERY_BUTTON_*` | Physical recovery GPIO and hold policy |

ESP32-C5 can use `5g` or `auto`; the other supported Native targets use `2g`. Sensing remains HT20.

## Automatic Discovery

Native, Matter, and ESPHome answer one-shot IPv4 bootstrap names in the form `espectre-devices-<24 hex>.local` and expose the capability-gated Direct method `discover_peers`. Configure and Monitor generate a new 96-bit lowercase nonce through Web Crypto for every attempt, contact one eligible responder, request a bounded fresh `_espectre._tcp.local.` browse, and then connect to the selected Native, ESPHome, or Matter endpoint on the shared Direct port. The selected device's exact Direct capability handshake controls which web configuration and runtime actions are shown; the frontend label and coarse DNS-SD capabilities are not authorization or UI feature gates. No nonce is retained or registered as device identity. Private IP, full ID, unique short ID, remembered device, Improv, QR, and share-link references remain available, but only a private IP endpoint is independent of mDNS.

The bootstrap responder uses the shared Direct port `62587` and remains IPv4-only; Native, ESPHome, and Matter advertise the same port for their selected endpoints. It accepts class-IN A and AAAA questions with an exact 24-character hexadecimal nonce. An A response returns the requested owner as a shared record with a 10-second TTL and no cache-flush bit, plus an NSEC record declaring that A exists and AAAA does not. A standalone AAAA question receives only that NSEC assertion, allowing dual-stack resolvers to continue without an IPv6 timeout while never advertising an IPv6 address. Pending responses are discarded on disconnect or address change. The responder does not answer the former static alias, register or announce nonce names, send nonce goodbyes, or provide a compatibility fallback. The browser bounds the bootstrap request to 10 seconds, and the responder bounds each peer query to 3 seconds. A result contains at most eight devices and two addresses per device; concurrent requests are rejected. Firmware accepts only canonical Native, ESPHome, and Matter records with validated on-link IPv4 endpoints and returns no credentials, configuration secrets, telemetry, CSI, or broker details.

Automatic discovery requires working local multicast, a host resolver that sends `.local` address queries over mDNS, client reachability, and at least one eligible Native, Matter, or ESPHome bootstrap responder. Multicast filtering, wireless client isolation, disabled mDNS, resolver restrictions, VLAN boundaries, or the absence of a reachable responder cause the portal to return to manual entry. The public [setup guide](https://espectre.dev/guides/setup/#setup-native-discovery) owns the browser and operating-system support and test-coverage matrix and the recovery procedure. The mDNS-free path is the current private IPv4 address returned by Improv Serial or shown in the router's DHCP lease table: a full device ID maps to `espectre-<device_id>.local`, while a device name or 6-character suffix invokes Auto-discovery. The repository `./espectre devices --frontend native` command performs its own fresh DNS-SD browse and may recover the IP when the host's `.local` resolver alone is disabled, but it still requires an mDNS-visible LAN.

## Optional MQTT and Home Assistant

MQTT is disabled until configured. Wi-Fi alone is sufficient for Native to start Direct HTTP and sense. Adding, losing, slowing, or clearing MQTT does not disable Direct mode.

When configured, MQTT runs concurrently with Direct HTTP and provides the canonical ESPectre MQTT topic surface, Home Assistant MQTT Discovery, retained availability, and integration with broker-based clients. Both transports invoke the same command engine; a query answers only its requester, while a mutation fans out the corresponding authoritative state event. Their outbound queues remain separate so broker backpressure cannot delay Direct sensing. Monitor connects to the broker only through `wss://`, so the broker must present a certificate trusted by the browser; device-to-broker MQTT TCP configuration is unchanged.

The Native `diagnostics` request returns uptime, current, minimum, and largest-block heap, CPU frequency, frontend-task stack high-water, bounded loop-load and detector-timing windows, and cached traffic, CSI, Wi-Fi, Direct, and MQTT diagnostics. Transport diagnostics include fixed client, queue, and MQTT outbox budgets alongside current occupancy and cumulative drops, send failures, and slow-client disconnects. Performance aggregation is unconditional production runtime state; it does not require a build option or periodic debug logger.

Home Assistant discovery is enabled in the published defaults and can be disabled with `CONFIG_ESPECTRE_HA_DISCOVERY_ENABLED`. It publishes the same primary sensing and tuning entities used by the ESPHome frontend:

| Entity | Behavior |
| --- | --- |
| Motion Detected | Filtered movement-state edges |
| Movement Score | Each detector evaluation |
| Threshold and hit counts | Retained state and writable control |
| Detection Profile | `lightweight` or `high_accuracy` |
| CSI Traffic Ownership and Source | Runtime traffic controls |
| Recalibrate | Configuration button that starts recalibration |
| Calibration Active | Diagnostic binary sensor that reports the authoritative runtime state |
| CSI and Wi-Fi diagnostics | Published on demand after Refresh Diagnostics |

Canonical topics under `espectre/v1/devices/{device_id}/...` remain available to standalone clients. See [`ESPECTRE_PROTOCOL.md`](../../../../docs/ESPECTRE_PROTOCOL.md) for the exact topic and payload contract.

## Detection and Traffic

Lightweight Detection uses less CPU and memory and learns a room-specific threshold from usable quiet-room coverage. High Accuracy uses the bundled model and skips threshold calibration, but still waits for CSI readiness and feature-window warmup. The selected profile persists across reboot and can be changed through Direct, MQTT, or Home Assistant.

`CONFIG_ESPECTRE_CSI_TARGET_PPS` sets the sensing cadence target. `CONFIG_ESPECTRE_CSI_TRAFFIC_MODE_*` selects internal or external traffic. External mode opens UDP port `5555`, joins `CONFIG_ESPECTRE_CSI_TRAFFIC_MULTICAST_GROUP`, `239.255.0.1` by default, and accepts only the exact four-byte UTF-8 marker `"👻".encode("utf-8")` (`F0 9F 91 BB`). Use [`espectre_traffic_generator.py`](../../../../tools/espectre_traffic_generator.py) standalone or through `./espectre collect`. A raw session uses the already configured traffic mode and does not change its rate.

## OTA

Native uses the shared ESP-IDF HTTPS OTA service. Direct and MQTT can call `ota_status`, `ota_check`, and `ota_start` when their advertised capabilities include OTA.

- `release`, `preview`, and `develop` select the corresponding publication channel.
- Clients cannot override the manifest host, image URL, chip, or target version.
- The HTTPS service downloads only a strictly newer release, prerelease, or rolling `git describe` identity; stale manifests cannot trigger a downgrade.
- A successful update schedules a reboot into the new OTA slot.
- Reconnection republishes device identity, online status, and OTA state.
- USB reflashing with the full factory image remains the recovery path when OTA cannot complete.

The per-chip manifest is named `espectre-native-ota-<chip>.json`. Its image URL points to the matching application-only `-ota.bin` artifact.

## Troubleshooting

### Direct mode cannot connect

1. Confirm that the device and browser are on the same LAN.
2. Try the current IP address if the `.local` hostname does not resolve.
3. Grant the browser's local-network permission when prompted.
4. Use a supported desktop Chromium browser if another browser does not implement hosted HTTPS-to-local-HTTP fetch with Private Network Access.
5. Confirm that the page origin is `https://espectre.dev`, `https://www.espectre.dev`, `https://test.espectre.dev`, or an explicitly enabled HTTP loopback origin. Development builds accept any port only for exact `localhost`, `127.0.0.1`, or `[::1]` hosts; published firmware disables that exception.

### The device does not join Wi-Fi

Reconnect over Improv Serial and provision the network again. If a BSSID pin is stale, configure the SSID without a pin. When remote configuration is unreachable, hold BOOT for the configured recovery interval and repeat Improv Serial provisioning.

### The device address changed or a saved endpoint is stale

Use the private IPv4 address returned by Improv Serial or shown in the router's DHCP lease table when Auto-discovery fails. A full device ID maps to a `.local` hostname, and a short ID or device name invokes Auto-discovery, so those references still depend on working mDNS. Use the portal's forget action to remove a stale remembered endpoint before entering the current IP. If the browser reports an Origin, mixed-content, or local-network permission error, use a claimed browser, grant access only for the ESPectre portal, and confirm that the device is still on the same trusted LAN.

### OTA failed or an older release is required

Reflash the full factory image over USB when OTA cannot complete. Downgrades are not a general compatibility promise: use an older factory image only when that release's migration notes explicitly allow it, and erase flash when its persisted configuration schema is incompatible. A full reflash and Improv Serial provisioning do not depend on MQTT, a remembered endpoint, or the original browser profile.

### MQTT data does not appear in Monitor

Confirm that the broker hostname resolves from both the ESP32 and browser, that the credentials are valid, and that the broker exposes MQTT over WebSockets. Direct Monitor should remain operational while broker issues are diagnosed.

## Implementation Map

- [`app/`](app/): standalone ESP-IDF entry point, Wi-Fi lifecycle, Improv Serial, mDNS, Direct service, and recovery wiring
- [`espectre/native_frontend.cpp`](espectre/native_frontend.cpp): transport-neutral command dispatch, event fan-out, MQTT integration, and Home Assistant adapter
- [`../../runtime/direct_http_protocol.cpp`](../../runtime/direct_http_protocol.cpp): canonical request parsing and Direct/MQTT protocol mapping
- [`../../runtime/esp_idf/direct_http_service_esp_idf.cpp`](../../runtime/esp_idf/direct_http_service_esp_idf.cpp): bounded ESP-IDF HTTP, SSE, and binary streaming server
- [`../../runtime/esp_idf/mdns_discovery_service.cpp`](../../runtime/esp_idf/mdns_discovery_service.cpp): shared Direct discovery lifecycle
- [`../../runtime/esp_idf/frontend_support/improv_serial_service.cpp`](../../runtime/esp_idf/frontend_support/improv_serial_service.cpp): standard Improv Serial adapter
- [`../../runtime/esp_idf/frontend_support/wifi_provisioning_service.cpp`](../../runtime/esp_idf/frontend_support/wifi_provisioning_service.cpp): staged Wi-Fi updates, commit, rollback, and BSSID fallback
- [`../../runtime/espectre_protocol.cpp`](../../runtime/espectre_protocol.cpp): shared command and application payload semantics
