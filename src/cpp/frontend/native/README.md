# ESPectre Native Frontend

Native is the standalone ESP-IDF firmware for browser-based local setup, sensing over Direct WebSocket, optional MQTT integration, Home Assistant MQTT Discovery, and HTTPS OTA. The shared message model is documented in [`ESPECTRE_PROTOCOL.md`](../../../../docs/ESPECTRE_PROTOCOL.md).

## Getting Started

The normal browser workflow is:

1. Open [Flash](https://espectre.dev/flash) in a supported Chromium browser and install the Native image for the detected chip.
2. Complete the standard Improv Serial prompt to provision Wi-Fi over USB.
3. Open Configure with the returned device URL, or enter the device IP or `espectre-<device_id>.local` hostname manually.
4. Use Direct WebSocket to inspect status, reconcile or pin the associated BSSID, edit the device label, and add optional MQTT settings.
5. Open Monitor and select Direct WebSocket for broker-free sensing, or MQTT for Home Assistant, automation, remote brokers, and multiple devices.

Each `release`, `preview`, and `develop` channel publishes a full-flash image and an application-only OTA image for ESP32, ESP32-S2, ESP32-S3, ESP32-C3, ESP32-C5, and ESP32-C6. Matter does not publish an ESP32-S2 image because its supported commissioning flow requires a Bluetooth-capable target.

### Local ESP-IDF Workflow

Complete the shared [`Local Build Prerequisites`](../../../../docs/SETUP.md#local-build-prerequisites), then use the repository CLI:

```bash
./espectre native build --chip s2 --ota-channel develop --clean
./espectre native flash --chip s2 --port /dev/cu.usbmodemXXXX
./espectre monitor --port /dev/cu.usbmodemXXXX
```

The wrapper uses a local ESP-IDF installation when available and can use the pinned Docker image for builds. Flashing and serial monitoring require local tooling. `--ota-channel` selects the default release channel used when an OTA request omits one.

## Direct WebSocket

Native starts `ws://<device>/espectre/v1/ws` after Wi-Fi obtains an address. Clients must negotiate the `espectre.v1` subprotocol. The production portal and `https://test.espectre.dev` validation origins are allowed by default; optional loopback development origins are controlled by Kconfig and remain disabled in published firmware.

Direct mode provides:

- capability negotiation, device identity, status, configuration, diagnostics, and correlated command results
- Wi-Fi updates with optional BSSID and channel hints
- device-label and optional MQTT add, change, or clear operations
- sensing start and stop, recalibration, detector selection, thresholds, hit counts, and traffic controls
- processed movement, state, calibration, diagnostics, and lifecycle events
- supported OTA status and control operations

The endpoint never returns stored Wi-Fi or MQTT passwords. It caps frame size, mutation rate, queued messages, and concurrent clients. Telemetry may replace an older queued telemetry sample, while command results and state transitions are preserved. Each Direct client has one asynchronous send in flight. MQTT uses its own 16-message frontend queue and bounded ESP-IDF outbox. Runtime callbacks only stage numeric sensing state; serialization and transport work run after detector evaluation returns.

The device advertises `_espectre._tcp` through mDNS with a stable `espectre-<device_id>.local` hostname. [`ESPECTRE_PROTOCOL.md`](../../../../docs/ESPECTRE_PROTOCOL.md#mdnsdns-sd-discovery) owns the SRV and TXT record contract. Run `./espectre devices --frontend native` from the repository to enumerate advertised Native endpoints on an mDNS-visible LAN. Browsers cannot enumerate DNS-SD directly, so Configure and Monitor always accept a manual IP address or `.local` hostname. Successful non-secret endpoints can be remembered or shared through a credential-free QR link.

Hosted HTTPS access to a local cleartext WebSocket depends on browser policy. The portal path supports Chrome 147 or later on desktop through Local Network Access; Chrome 151 on macOS passed the hosted HTTPS validation path against a physical ESP32-C3. Firefox and Safari block the hosted HTTPS-to-`ws://` workflow under their mixed-content policy; Edge and mobile Chrome remain unclaimed until their physical browser runs are recorded. If the hosted path is unavailable, serve the portal locally as described in [`docs/web/README.md`](../../../../docs/web/README.md).

## Wi-Fi Provisioning and Recovery

Standard Improv Serial remains available through the primary serial console. It owns initial Wi-Fi provisioning and returns the post-connect device URL. Custom BSSID, device-label, MQTT, sensing, and OTA operations belong to Direct WebSocket.

Remote Wi-Fi changes are staged. Native attempts the candidate network, commits it only after association and address acquisition, and rolls back to the last-known-good settings when the attempt fails or times out. After a successful commit or rollback, Native reboots once so the ESP32-C3 radio returns with a fresh CSI capture session; Direct and MQTT clients should reconnect after the device address becomes reachable again. If an optional BSSID is unavailable, the provisioning policy can retry the same SSID without the pin instead of permanently stranding the device.

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

MQTT is disabled until configured. Wi-Fi alone is sufficient for Native to start Direct WebSocket and sense. Adding, losing, slowing, or clearing MQTT does not disable Direct mode.

When configured, MQTT runs concurrently with Direct WebSocket and provides the canonical ESPectre MQTT topic surface, Home Assistant MQTT Discovery, retained availability, and integration with broker-based clients. Monitor connects to the broker through MQTT over WebSockets, so the broker must expose a browser-compatible `ws://` or `wss://` listener.

The Native `stats` request returns base uptime, current and minimum heap, frontend-task stack high-water, and loop fields plus cached traffic, CSI, Wi-Fi, Direct, and MQTT diagnostics. Transport diagnostics include fixed client, queue, and MQTT outbox budgets alongside current occupancy and cumulative drops. The cache is refreshed by the existing sensing status update; diagnostics do not add a separate sampling timer.

Home Assistant discovery is enabled in the published defaults and can be disabled with `CONFIG_ESPECTRE_HA_DISCOVERY_ENABLED`. It publishes the same primary sensing and tuning entities used by the ESPHome frontend:

| Entity | Behavior |
| --- | --- |
| Motion Detected | Filtered movement-state edges |
| Movement Score | Each detector evaluation |
| Threshold and hit counts | Retained state and writable control |
| Detection Profile | `lightweight` or `high_accuracy` |
| CSI Traffic Ownership and Source | Runtime traffic controls |
| Trigger Calibration | Starts recalibration |
| CSI and Wi-Fi diagnostics | Published on demand after Refresh Diagnostics |

Canonical topics under `espectre/v1/devices/{device_id}/...` remain available to standalone clients. See [`ESPECTRE_PROTOCOL.md`](../../../../docs/ESPECTRE_PROTOCOL.md) for the exact topic and payload contract.

## Detection and Traffic

Lightweight Detection uses less CPU and memory and learns a room-specific threshold from usable quiet-room coverage. High Accuracy uses the bundled model and skips threshold calibration, but still waits for CSI readiness and feature-window warmup. The selected profile persists across reboot and can be changed through Direct, MQTT, or Home Assistant.

`CONFIG_ESPECTRE_CSI_TARGET_PPS` sets the positive cadence target. `CONFIG_ESPECTRE_CSI_TRAFFIC_MODE_*` selects internal, external, or unmanaged traffic. External mode opens UDP port `5555` and joins `CONFIG_ESPECTRE_CSI_TRAFFIC_MULTICAST_GROUP`, `239.255.0.1` by default. Use [`espectre_traffic_generator.py`](../../../../tools/espectre_traffic_generator.py) with device IPs or the multicast address.

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
4. Use a supported Chromium browser if another browser blocks HTTPS-to-`ws://` mixed content.
5. Confirm that the page origin is `https://espectre.dev`, `https://www.espectre.dev`, `https://test.espectre.dev`, or an explicitly enabled HTTP loopback origin. Development builds accept any port only for exact `localhost`, `127.0.0.1`, or `[::1]` hosts; published firmware disables that exception.

### The device does not join Wi-Fi

Reconnect over Improv Serial and provision the network again. If a BSSID pin is stale, configure the SSID without a pin. When remote configuration is unreachable, hold BOOT for the configured recovery interval and repeat Improv Serial provisioning.

### The device address changed or a saved endpoint is stale

Try `espectre-<device_id>.local`, then check the router lease table for the current address. Use the portal's forget action to remove a stale remembered endpoint before entering the current IP address or `.local` hostname. If the browser reports an Origin, mixed-content, or local-network permission error, use a claimed browser, grant access only for the ESPectre portal, and confirm that the device is still on the same trusted LAN.

### OTA failed or an older release is required

Reflash the full factory image over USB when OTA cannot complete. Downgrades are not a general compatibility promise: use an older factory image only when that release's migration notes explicitly allow it, and erase flash when its persisted configuration schema is incompatible. A full reflash and Improv Serial provisioning do not depend on MQTT, a remembered endpoint, or the original browser profile.

### MQTT data does not appear in Monitor

Confirm that the broker hostname resolves from both the ESP32 and browser, that the credentials are valid, and that the broker exposes MQTT over WebSockets. Direct Monitor should remain operational while broker issues are diagnosed.

## Implementation Map

- [`app/`](app/): standalone ESP-IDF entry point, Wi-Fi lifecycle, Improv Serial, mDNS, Direct service, and recovery wiring
- [`espectre/native_frontend.cpp`](espectre/native_frontend.cpp): transport-neutral command dispatch, event fan-out, MQTT integration, and Home Assistant adapter
- [`../../runtime/direct_websocket_protocol.cpp`](../../runtime/direct_websocket_protocol.cpp): versioned Direct envelopes
- [`../../runtime/esp_idf/direct_websocket_service_esp_idf.cpp`](../../runtime/esp_idf/direct_websocket_service_esp_idf.cpp): bounded ESP-IDF WebSocket server
- [`../../runtime/esp_idf/mdns_discovery_service.cpp`](../../runtime/esp_idf/mdns_discovery_service.cpp): shared Native and Streamer discovery lifecycle
- [`../../runtime/esp_idf/frontend_support/improv_serial_service.cpp`](../../runtime/esp_idf/frontend_support/improv_serial_service.cpp): standard Improv Serial adapter
- [`../../runtime/esp_idf/frontend_support/wifi_provisioning_service.cpp`](../../runtime/esp_idf/frontend_support/wifi_provisioning_service.cpp): staged Wi-Fi updates, commit, rollback, and BSSID fallback
- [`../../runtime/espectre_protocol.cpp`](../../runtime/espectre_protocol.cpp): shared command and MQTT payload semantics
