# ESPectre Native Frontend

Native is the standalone ESP-IDF firmware for Direct HTTP, optional MQTT, Home Assistant MQTT Discovery, and HTTPS OTA. This guide covers its provisioning, integrations, build options, and recovery.

## Getting Started

Follow [SETUP.md](../../../../docs/SETUP.md) to select a supported board, flash Native, provision Wi-Fi, and check sensing. Wi-Fi alone is enough to use Device settings and Monitor. Add MQTT in Device settings when you need Home Assistant MQTT Discovery or broker-based clients.

### Local ESP-IDF Workflow

Complete the local build prerequisites in [CLI.md](../../../../docs/CLI.md#local-build-prerequisites), then run:

```bash
./espectre native build --chip s2 --ota-channel develop --clean
./espectre native flash --chip s2 --port /dev/cu.usbmodemXXXX
./espectre monitor --port /dev/cu.usbmodemXXXX
```

`--ota-channel` sets the default channel for OTA requests. See [CLI.md](../../../../docs/CLI.md#native-and-matter) for build, upload, and console options. Improv Serial uses the target's primary serial console, including TinyUSB CDC on maintained USB-OTG configurations that need it.

## Direct HTTP

After Wi-Fi connects, run `./espectre devices --frontend native` to find the Direct endpoint. Use Device settings for configuration and OTA, and Monitor for sensing controls and diagnostics. The endpoint never returns stored Wi-Fi or MQTT passwords.

See [API.md](../../../../docs/API.md) for resources, events, limits, and security; [DISCOVERY.md](../../../../docs/DISCOVERY.md) for endpoint discovery; and [CLI.md](../../../../docs/CLI.md#collect) for raw CSI collection. For a local browser-tool build, enable `CONFIG_ESPECTRE_DIRECT_DEV_ORIGINS_ENABLED` and follow [README.md](../../../../docs/web/README.md#local-preview); published firmware leaves loopback origins disabled.

## Wi-Fi Provisioning and Recovery

Use Improv Serial to set the Wi-Fi SSID and password over USB. It returns a Device settings link for the connected device. Direct HTTP can inspect the association, scan for access points on the provisioned network, select a BSSID, or remove saved Wi-Fi credentials. Band selection is a build-time option.

Scanning can briefly interrupt sensing and network traffic. A BSSID change disconnects clients while Native verifies the new association; failure restores the last-known-good settings. A power loss during the transaction leaves the pending candidate available for retry at boot. Automatic selection clears the pin and channel hint. See [API.md](../../../../docs/API.md#wi-fi-scan-and-bssid-selection) for requests and [CSI.md](../../../../docs/CSI.md#wi-fi-and-capture-lifecycle) for capture restart behavior.

Removing Wi-Fi credentials in Device settings disconnects the station and returns it to Improv Serial provisioning.

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

ESP32-C5 defaults to `auto` and can be pinned to `2g` or `5g`; the other supported Native targets use `2g`. See [CSI.md](../../../../docs/CSI.md#capture-profiles) for capture-profile selection and the limits of 5 GHz sensing.

## Optional MQTT and Home Assistant

MQTT is disabled until configured. It runs alongside Direct HTTP, and broker failures do not disable Direct sensing. The browser Monitor always uses Direct HTTP.

Device settings requires an explicit scheme, host, and port. Select `mqtt`, a bare local hostname such as `homeassistant.local`, and port `1883` for a typical trusted-LAN Home Assistant or Mosquitto broker. Select `mqtts` and the broker's TLS port, commonly `8883`, for a public-CA-secured broker; Native verifies both the certificate chain and broker hostname. Do not put `mqtt://`, `mqtts://`, credentials, a port, or a path in the host field. WebSocket MQTT and private certificate authorities are not supported in this configuration version.

An older saved endpoint without an explicit scheme is retained for recovery but remains disconnected and reports `configured: false`. Open Device settings over Direct HTTP and save the endpoint again with the intended scheme. Native never guesses whether an existing broker should use plaintext or TLS.

Home Assistant discovery is enabled in published firmware and can be disabled with `CONFIG_ESPECTRE_HA_DISCOVERY_ENABLED`. It exposes:

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

Standalone MQTT clients use the topics and payloads in [API.md](../../../../docs/API.md#mqtt). Production diagnostics are available through both Direct and MQTT, including transport queues, drops, and failures; see [API.md](../../../../docs/API.md#diagnostics).

## Detection and Traffic

Set build-time sensing defaults in the shared ESP-IDF menu using [SDK.md](../../../../docs/SDK.md#shared-sensing-options). Direct HTTP, MQTT, and Home Assistant expose runtime controls; accepted detector and traffic selections persist across reboot. Use [CSI.md](../../../../docs/CSI.md#traffic-sources) for traffic behavior and [TROUBLESHOOTING.md](../../../../docs/TROUBLESHOOTING.md#tuning-essentials) for practical tuning.

## OTA

Use Device settings, Direct HTTP, or MQTT to check for and install HTTPS OTA updates. [API.md](../../../../docs/API.md#ota-actions) defines the operations. Native pauses sensing and stops its transports during the download.

- `release`, `preview`, and `develop` select the corresponding publication channel.
- Clients cannot override the manifest host, image URL, chip, or target version.
- The HTTPS service downloads only a strictly newer release, prerelease, or rolling `git describe` identity; stale manifests cannot trigger a downgrade.
- A successful update schedules a reboot into the new OTA slot.
- A failed update restores Direct HTTP and MQTT; sensing resumes only if it was enabled before the update.
- USB reflashing with the full factory image remains the recovery path when OTA cannot complete.

OTA selects the application-only image for the device chip from the chosen channel's firmware manifest. Missing or ambiguous matches fail the check. The service is frontend code, outside the sensing SDK; see [ARCHITECTURE.md](../../../../docs/ARCHITECTURE.md) for layer ownership.

## Troubleshooting

Use [TROUBLESHOOTING.md](../../../../docs/TROUBLESHOOTING.md) for browser connectivity and sensing problems.

### The device does not join Wi-Fi

Reconnect over Improv Serial and provision the network again. If a BSSID pin is stale, configure the SSID without a pin. When remote configuration is unreachable, hold BOOT for the configured recovery interval and repeat Improv Serial provisioning.

### OTA failed or an older release is required

Reflash the full factory image over USB when OTA cannot complete. Downgrades are not a general compatibility promise: use an older factory image only when that release's migration notes explicitly allow it, and erase flash when its persisted configuration schema is incompatible. A full reflash and Improv Serial provisioning do not depend on MQTT, a remembered endpoint, or the original browser profile.

### MQTT clients do not receive data

Confirm that the endpoint reports `configured: true`, that the broker hostname resolves from the ESP32, that the selected scheme and port match the broker listener, that the credentials are valid, and that the intended broker client subscribes to the canonical topics. For `mqtts`, the certificate must chain to the ESP-IDF public bundle and identify the configured host. The browser Monitor uses Direct HTTP and should remain operational while broker issues are diagnosed.

## Implementation Map

- [`app/`](app/): standalone ESP-IDF entry point, Wi-Fi lifecycle, Improv Serial, mDNS, Direct service, and recovery wiring
- [native_frontend.cpp](espectre/native_frontend.cpp): lifecycle, runtime events, and OTA coordination
- [native_command_bindings.cpp](espectre/native_command_bindings.cpp): persistence, provisioning, and command bindings
- [native_direct_frontend.cpp](espectre/native_direct_frontend.cpp): Direct lifecycle, diagnostics, discovery, and collection
- [native_mqtt_frontend.cpp](espectre/native_mqtt_frontend.cpp): MQTT transport adapter
- [home_assistant_mqtt_frontend.cpp](espectre/home_assistant_mqtt_frontend.cpp): Home Assistant discovery and entity mapping

Shared runtime and transport components are mapped in [ARCHITECTURE.md](../../../../docs/ARCHITECTURE.md) and [SDK.md](../../../../docs/SDK.md).
