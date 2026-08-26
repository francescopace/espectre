# ESPectre Native Frontend

Native is the standalone ESP-IDF firmware for browser-based local setup, sensing over Direct HTTP, optional MQTT integration, Home Assistant MQTT Discovery, and HTTPS OTA. The shared message model is documented in [`ESPECTRE_PROTOCOL.md`](../../../../docs/ESPECTRE_PROTOCOL.md).

## Getting Started

The normal browser workflow is:

1. Open [Flash](https://espectre.dev/tools/flash/) in a supported Chromium browser and install the Native image for the detected chip.
2. Complete the standard Improv Serial prompt to provision Wi-Fi over USB.
3. Open Configure with the returned device URL, or enter the private IP, device name, full 16-character device ID, or last 6 ID characters.
4. Use Direct HTTP to inspect status, reconcile or pin the associated BSSID, edit the device label, and add optional MQTT settings.
5. Open Monitor for broker-free sensing over Direct HTTP. Use MQTT for Home Assistant, automation, remote brokers, and other broker-based clients.

The `release`, `preview`, and `develop` channels publish a full-flash image and an application-only OTA image for ESP32, ESP32-S2, ESP32-S3, ESP32-C3, ESP32-C5, and ESP32-C6.

### Local ESP-IDF Workflow

Complete the shared [`Local Build Prerequisites`](../../../../docs/SETUP.md#local-build-prerequisites), then use the repository CLI:

```bash
./espectre native build --chip s2 --ota-channel develop --clean
./espectre native flash --chip s2 --port /dev/cu.usbmodemXXXX
./espectre monitor --port /dev/cu.usbmodemXXXX
```

Flashing and serial monitoring require local tooling. `--ota-channel` selects the default release channel used when an OTA request omits one.

## Direct HTTP

Native starts `POST http://<device>:62587/espectre/v1/request` and `GET http://<device>:62587/espectre/v1/events` after Wi-Fi obtains an address. The production portal and `https://test.espectre.dev` validation origins are allowed by default; optional loopback development origins are controlled by Kconfig and remain disabled in published firmware. Requests use JSON, and events use SSE read through streaming `fetch` so the browser can request local-network access explicitly.

Direct mode provides:

- capability negotiation, device identity, status, configuration, diagnostics, and correlated command results
- Wi-Fi updates with optional BSSID and channel hints
- device-label and optional MQTT add, change, or clear operations
- sensing enable or pause through `set_sensing`, recalibration, detector selection, thresholds, hit counts, and traffic controls
- processed movement, state, calibration, diagnostics, and lifecycle events
- supported OTA status and control operations

The endpoints never return stored Wi-Fi or MQTT passwords. They cap request and response size, mutation rate, queued messages, and concurrent SSE subscribers. Telemetry may replace an older queued sample, while state transitions are preserved. MQTT uses its own 16-message frontend queue and bounded ESP-IDF outbox.

The high-rate telemetry callback runs only while MQTT is connected or a Direct SSE client is present. Runtime callbacks stage numeric sensing state; serialization and transport work run after detector evaluation returns.

Native advertises the shared bearer-bound raw CSI HTTP v2 surface. A raw session keeps the configured traffic source active and restores sensing when the session ends without changing persisted traffic configuration. [`ESPECTRE_PROTOCOL.md`](../../../../docs/ESPECTRE_PROTOCOL.md#direct-raw-csi-v2) owns session authorization, framing, queue limits, and recovery behavior; [`CLI.md`](../../../../docs/CLI.md#collect) owns the `./espectre collect` workflow.

The device advertises `_espectre._tcp` through mDNS with a stable `espectre-<device_id>.local` hostname. Run `./espectre devices --frontend native` to enumerate Native endpoints on an mDNS-visible LAN. [`ESPECTRE_PROTOCOL.md`](../../../../docs/ESPECTRE_PROTOCOL.md#mdnsdns-sd-discovery) owns the SRV, TXT, and peer-discovery contract; [`SETUP.md`](../../../../docs/SETUP.md#direct-http-connectivity) owns browser permissions, supported connection identifiers, and recovery when discovery fails.

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

## Optional MQTT and Home Assistant

MQTT is disabled until configured. Wi-Fi alone is sufficient for Native to start Direct HTTP and sense. Adding, losing, slowing, or clearing MQTT does not disable Direct mode.

When configured, MQTT runs concurrently with Direct HTTP and provides the canonical ESPectre topic surface, Home Assistant MQTT Discovery, retained availability, and integration with broker-based clients. Both transports invoke the same command engine: a query answers only its requester, while a mutation fans out the corresponding authoritative state event. Their outbound queues remain separate, so broker backpressure cannot delay Direct sensing.

The browser Monitor uses Direct HTTP and does not connect to MQTT. Device-to-broker MQTT configuration is independent of the browser connection.

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

Native selects its build-time sensing defaults through the shared ESP-IDF `sdkconfig` menu and exposes supported runtime controls through Direct HTTP, optional MQTT, and Home Assistant MQTT Discovery. The accepted detector and traffic selections persist across reboot. [`SETUP.md`](../../../../docs/SETUP.md#shared-sensing-options) owns option names, defaults, traffic modes, and external-source configuration; [`TUNING.md`](../../../../docs/TUNING.md) owns profile selection, calibration, packet health, placement, and detector troubleshooting.

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

Use [`SETUP.md`](../../../../docs/SETUP.md#direct-http-connectivity) for Direct HTTP, browser permission, address, and discovery failures. Use [`TUNING.md`](../../../../docs/TUNING.md#troubleshooting) for missing motion, false positives, calibration, packet health, placement, or unstable detection.

### The device does not join Wi-Fi

Reconnect over Improv Serial and provision the network again. If a BSSID pin is stale, configure the SSID without a pin. When remote configuration is unreachable, hold BOOT for the configured recovery interval and repeat Improv Serial provisioning.

### OTA failed or an older release is required

Reflash the full factory image over USB when OTA cannot complete. Downgrades are not a general compatibility promise: use an older factory image only when that release's migration notes explicitly allow it, and erase flash when its persisted configuration schema is incompatible. A full reflash and Improv Serial provisioning do not depend on MQTT, a remembered endpoint, or the original browser profile.

### MQTT clients do not receive data

Confirm that the broker hostname resolves from the ESP32, that the credentials are valid, and that the intended broker client subscribes to the canonical topics. The browser Monitor uses Direct HTTP and should remain operational while broker issues are diagnosed.

## Implementation Map

- [`app/`](app/): standalone ESP-IDF entry point, Wi-Fi lifecycle, Improv Serial, mDNS, Direct service, and recovery wiring
- [`espectre/native_frontend.cpp`](espectre/native_frontend.cpp): transport-neutral command dispatch, event fan-out, MQTT integration, and Home Assistant adapter
- [`../../runtime/direct_http_protocol.cpp`](../../runtime/direct_http_protocol.cpp): canonical request parsing and Direct/MQTT protocol mapping
- [`../../runtime/esp_idf/direct_http_service_esp_idf.cpp`](../../runtime/esp_idf/direct_http_service_esp_idf.cpp): bounded ESP-IDF HTTP, SSE, and binary streaming server
- [`../../runtime/esp_idf/mdns_discovery_service.cpp`](../../runtime/esp_idf/mdns_discovery_service.cpp): shared Direct discovery lifecycle
- [`../../runtime/esp_idf/frontend_support/improv_serial_service.cpp`](../../runtime/esp_idf/frontend_support/improv_serial_service.cpp): standard Improv Serial adapter
- [`../../runtime/esp_idf/frontend_support/wifi_provisioning_service.cpp`](../../runtime/esp_idf/frontend_support/wifi_provisioning_service.cpp): staged Wi-Fi updates, commit, rollback, and BSSID fallback
- [`../../runtime/espectre_protocol.cpp`](../../runtime/espectre_protocol.cpp): shared command and application payload semantics
