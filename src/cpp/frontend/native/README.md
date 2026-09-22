# ESPectre Native frontend

Native is the standalone ESP-IDF firmware for Direct HTTP, optional MQTT, Home Assistant MQTT Discovery, and HTTPS OTA. This guide covers its provisioning, integrations, build options, and recovery.

## Getting started

Follow the [setup guide](../../../../docs/SETUP.md) to select a supported board, flash Native, provision Wi-Fi, and check sensing. Wi-Fi alone is enough to use Device settings and Monitor. Add MQTT in Device settings when you need Home Assistant MQTT Discovery or broker-based clients.

### Local ESP-IDF workflow

Complete the local build prerequisites in [local build prerequisites](../../../../docs/CLI.md#local-build-prerequisites), then run:

```bash
./espectre native build --chip s2 --ota-channel develop --clean
./espectre native flash --chip s2 --port /dev/cu.usbmodemXXXX
./espectre monitor --port /dev/cu.usbmodemXXXX
```

`--ota-channel` sets the default OTA channel. See [`native` and `matter` commands](../../../../docs/CLI.md#native-and-matter) for all options.

Improv Serial uses the main serial console. On ESP32-S2 that is TinyUSB CDC, enabled by `ESPECTRE_TINYUSB_PRIMARY_CONSOLE` under **ESPectre Firmware** in menuconfig.

Per-chip settings live in `app/sdkconfig.defaults.<idf_target>`, which the CLI loads after the shared defaults. CPU frequency is explicit for every supported chip: 240 MHz on ESP32, ESP32-S2, ESP32-S3, and ESP32-C5, and 160 MHz on ESP32-C3 and ESP32-C6. Add future chip-specific overrides to these files.

## Direct HTTP

After Wi-Fi connects, run `./espectre devices --frontend native` to find the Direct endpoint. Use Device settings for configuration and OTA, and Monitor for sensing controls and diagnostics. The endpoint never returns stored Wi-Fi or MQTT passwords.

See the [API reference](../../../../docs/API.md) for resources, events, limits, and security; the [discovery reference](../../../../docs/DISCOVERY.md) for endpoint discovery; and [`collect` command](../../../../docs/CLI.md#collect) for raw CSI collection. For a local browser-tool build, enable `CONFIG_ESPECTRE_DIRECT_DEV_ORIGINS_ENABLED` and follow [local preview](../../../../docs/web/README.md#local-preview); published firmware leaves loopback origins disabled.

## Wi-Fi provisioning and recovery

Use Improv Serial to set the Wi-Fi SSID and password over USB. It returns a Device settings link for the connected device. Direct HTTP can inspect the association, scan for access points on the provisioned network, select a BSSID, or remove saved Wi-Fi credentials. Band selection is a build-time option.

A scan can briefly interrupt sensing and network traffic. Changing the BSSID disconnects clients while Native tests the new access point; if it fails, the previous settings come back, even after a power loss. Automatic selection clears the pin and channel hint. See [Wi-Fi scan and BSSID selection](../../../../docs/API.md#wi-fi-scan-and-bssid-selection) for requests and [Wi-Fi and capture lifecycle](../../../../docs/CSI.md#wi-fi-and-capture-lifecycle) for capture restart behavior.

Removing Wi-Fi credentials in Device settings disconnects the station and returns it to Improv Serial provisioning.

Holding BOOT for 3 seconds clears the saved Wi-Fi settings and returns to Improv Serial provisioning. BOOT is GPIO0 on ESP32, ESP32-S2, and ESP32-S3, GPIO9 on ESP32-C3 and ESP32-C6, and GPIO28 on ESP32-C5. Change the pin or hold time with the `ESPECTRE_RECOVERY_BUTTON_*` options if your board differs.

Frontend-owned defaults in [`Kconfig.projbuild`](espectre/Kconfig.projbuild) are useful for reproducible lab images. Runtime provisioning stored in NVS takes precedence.

| Option | Purpose |
| --- | --- |
| `ESPECTRE_WIFI_SSID` | Initial Wi-Fi SSID |
| `ESPECTRE_WIFI_PASSWORD` | Initial Wi-Fi password |
| `ESPECTRE_WIFI_BSSID` | Optional AP-radio pin |
| `ESPECTRE_WIFI_BAND_2G`, `ESPECTRE_WIFI_BAND_5G`, `ESPECTRE_WIFI_BAND_AUTO` | Build-time band policy |
| `ESPECTRE_WIFI_CHANNEL` | Optional channel hint (`0` scans normally) |
| `ESPECTRE_RECOVERY_BUTTON_*` | Physical recovery GPIO and hold policy |

Every target defaults to `auto`, which uses the bands the radio has: 2.4 GHz on single-band chips, and automatic selection on ESP32-C5. ESP32-C5 can also be pinned to `2g` or `5g`. See [capture profiles](../../../../docs/CSI.md#capture-profiles) for capture-profile selection and the limits of 5 GHz sensing.

## Optional MQTT and Home Assistant

MQTT is disabled until configured. It runs alongside Direct HTTP, and broker failures do not disable Direct sensing. The browser Monitor always uses Direct HTTP.

In Device settings, enter the scheme, host, and port separately:

- **Local broker** (Home Assistant or Mosquitto): `mqtt`, a host such as `homeassistant.local`, port `1883`.
- **TLS broker** with a public certificate: `mqtts`, the broker's TLS port (usually `8883`). Native checks the certificate and hostname.

Put only the hostname in the host field, without `mqtt://`, credentials, port, or path. MQTT over WebSocket and private certificate authorities are not supported.

A broker saved by an older firmware without a scheme stays disconnected and reports `configured: false`. Save it again in Device settings with the right scheme.

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
| Generator, network, CSI, and Wi-Fi diagnostics | Published on demand after Refresh Diagnostics |

Standalone MQTT clients use the [MQTT topics](../../../../docs/API.md#mqtt-topics). Production diagnostics are available through both Direct and MQTT, including transport queues, drops, and failures; see [API diagnostics](../../../../docs/API.md#diagnostics).

## Detection and traffic

Set build-time defaults in menuconfig; see [shared sensing options](../../../../docs/SDK.md#shared-sensing-options). Direct, MQTT, and Home Assistant can change the settings at runtime, and the device remembers detector and traffic choices. See [tuning essentials](../../../../docs/TROUBLESHOOTING.md#tuning-essentials).

## OTA

Use Device settings, Direct HTTP, or MQTT to check for and install HTTPS OTA updates. [OTA actions](../../../../docs/API.md#ota-actions) defines the operations. Native pauses sensing and stops its transports during the download.

- `release`, `preview`, and `develop` select the corresponding publication channel.
- Clients cannot override the manifest host, image URL, chip, or target version.
- The HTTPS service downloads only a strictly newer release, prerelease, or rolling `git describe` identity; stale manifests cannot trigger a downgrade.
- A successful update schedules a reboot into the new OTA slot.
- A failed update restores Direct HTTP and MQTT; sensing resumes only if it was enabled before the update.
- USB reflashing with the full factory image remains the recovery path when OTA cannot complete.

OTA selects the application-only image for the device chip from the chosen channel's firmware manifest. Missing or ambiguous matches fail the check. The service is frontend code, outside the sensing SDK; see the [architecture overview](../../../../docs/ARCHITECTURE.md) for layer ownership.

Device settings checks for updates when you connect. If the check fails, the device reports an OTA error and keeps running the installed firmware.

Official images accept only signed OTA updates. There is no automatic rollback yet: if a new image fails to start properly, reflash over USB. See [official images and personal builds](../../../../docs/SETUP.md#official-images-and-personal-builds) for USB versus OTA when switching between official and personal builds, and [firmware signing](../../../../docs/RELEASING.md#firmware-signing) for key custody.

## Troubleshooting

Use the [troubleshooting guide](../../../../docs/TROUBLESHOOTING.md) for browser connectivity and sensing problems.

### The device does not join Wi-Fi

Reconnect over Improv Serial and provision the network again. If a BSSID pin is stale, configure the SSID without a pin. When remote configuration is unreachable, hold BOOT for the configured recovery interval and repeat Improv Serial provisioning.

### OTA failed or an older release is required

Reflash the full factory image over USB. This works without MQTT or saved browser settings. Install an older release only if its migration notes allow it, and erase flash if its saved settings are incompatible.

### MQTT clients do not receive data

Check that:

- the MQTT settings report `configured: true`;
- the ESP32 can resolve the broker hostname;
- the scheme and port match the broker;
- the credentials are valid;
- your client subscribes to the [documented topics](../../../../docs/API.md#mqtt-topics).

For `mqtts`, the broker certificate must be signed by a public authority and match the hostname. Monitor keeps working over Direct in the meantime.

## Implementation map

The frontend uses public SDK headers. See [building against an SDK bundle](../../../../docs/CLI.md#building-against-an-sdk-bundle) for builds against an extracted SDK bundle and [frontend layer](../../../../docs/ARCHITECTURE.md#srccppfrontend) for source groups and ownership.

OTA comes from the shared frontend sources (`ESPECTRE_FRONTEND_OTA_SOURCES`), outside the SDK: `ota_service.h` defines the interface, `ota_service_https.h` downloads from the ESPectre release catalogs, and `ota_protocol.h` adds the OTA routes to the protocol.

Native and Matter pin `improv/improv` to `1.2.7` from the ESP Component Registry in their frontend manifests. The shared Improv Serial service uses this dependency, which the SDK excludes.

- [`app/`](app/): standalone ESP-IDF entry point, Wi-Fi lifecycle, Improv Serial, mDNS, Direct service, and recovery wiring
- [native_frontend.cpp](espectre/native_frontend.cpp): lifecycle, runtime events, and OTA coordination
- [native_command_bindings.cpp](espectre/native_command_bindings.cpp): persistence, provisioning, and command bindings
- [native_direct_frontend.cpp](espectre/native_direct_frontend.cpp): Direct lifecycle, diagnostics, discovery, and collection
- [native_mqtt_frontend.cpp](espectre/native_mqtt_frontend.cpp): MQTT transport adapter
- [home_assistant_mqtt_frontend.cpp](espectre/home_assistant_mqtt_frontend.cpp): Home Assistant discovery and entity mapping

Shared runtime and transport components are mapped in the [architecture overview](../../../../docs/ARCHITECTURE.md) and the [SDK guide](../../../../docs/SDK.md).
