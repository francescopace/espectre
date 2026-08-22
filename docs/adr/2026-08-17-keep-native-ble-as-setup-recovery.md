# ADR: keep Native BLE as setup and recovery

- Status: Accepted
- Date: 2026-08-17

## Context

Native firmware shares the ESP32 radio between Wi-Fi CSI sensing and Bluetooth Low Energy. With BLE up, measured CSI occupancy fell from about 80–90% to 35–45% even after advertising and connection intervals were slowed. Lowering the detector occupancy floor so BLE-era windows could still evaluate made detection possible, but it also made the first-look Native experience a starved radio rather than the supported MQTT path.

BLE had grown into a second operational plane: live movement notify, threshold and detector writes, motion-hit debounce, CSI traffic control, and recalibration. Configure still subscribed to that live surface. MQTT, Home Assistant Discovery, the Monitor, the Game, and the Theremin already carry the same sensing controls and telemetry once Wi-Fi and a broker are saved.

A nearby “see motion without MQTT” preview cannot provide the supported sensing path. With CSI paused, the meter has no data; with CSI active, the meter uses the degraded coexistence path. Recalibration over BLE would fit Lightweight to that starved stream.

## Decision

Native BLE is a proximity setup and recovery radio. It is not a detection plane.

Keep over BLE:

- advertising and connect
- Wi-Fi provision and clear
- MQTT broker provision and clear
- device identity (`device_id` read-only, `device_name`, `device_label`)
- read-only sysinfo status, including firmware, chip, Wi-Fi/MQTT connection, and the current detector settings as diagnostics
- `STOP_BLE`, MQTT `set_ble`, and a long BOOT-button press to re-enter setup

Remove from BLE:

- live movement, threshold, and motion-state notify
- `SET_THRESHOLD`, `SET_MOTION_HITS`, `SET_DETECTOR`, `SET_CSI_TRAFFIC_MODE`, `SET_TRAFFIC_GENERATOR_MODE`
- `RECALIBRATE`
- OTA status, check, and start; those remain on MQTT

MQTT, Home Assistant Discovery, and ESPHome remain the sensing-control family. Native starts BLE automatically when Wi-Fi SSID or MQTT host is missing, pauses CSI while BLE is up, keeps advertising across nearby client disconnects, stops BLE only when `STOP_BLE` or MQTT `set_ble` with `ble=off` explicitly closes setup, and does not lower the production occupancy floor to make BLE coexistence look ready. Compile-time Kconfig Wi-Fi and MQTT defaults count as configured, so lab images skip BLE at boot.

The web product presents Configure and Monitor as separate tools. Configure owns only Wi-Fi, MQTT, and the mutable device label. Start sensing opens Monitor, which connects and subscribes through the configured broker and reports sensing as active only after valid device telemetry arrives. Edit connectivity returns to Configure. Runtime sensing configuration and diagnostics remain on Monitor. If MQTT cannot reopen BLE, holding BOOT for 3 seconds invokes the same BLE-start intent and pauses sensing without erasing configuration.

The BLE telemetry characteristic UUID may remain in the GATT table so existing discovery still succeeds. Native does not notify on it, and `supports_live_telemetry` is false. BLE sysinfo capability flags for runtime sensing writes are false even when MQTT `info` still advertises those commands.

## Decision History

| Date | Direction | Resolution |
| --- | --- | --- |
| 2026-03-17 | BLE runtime control as a first-class standalone live surface | Replaced. Live notify and sensing writes assumed CSI and BLE could share the radio |
| 2026-08-16 | Setup-only BLE with CSI paused, while keeping sensing writes and optional live notify | Incomplete. The remaining live and sensing-control surface still invited BLE-during-detection |
| 2026-08-17 | Restrict BLE to Wi-Fi, MQTT, OTA, and identity/status | Superseded by the 2026-08-22 OTA transport decision |
| 2026-08-18 | Treat Kconfig Wi-Fi and MQTT defaults as configured so lab images skip BLE at boot | Clarified |
| 2026-08-22 | Keep OTA status, checks, and updates on MQTT instead of BLE | Accepted |

## Alternatives Considered

### Keep live BLE detection and train for 35–45% occupancy

Rejected. Occupancy that low is a coexistence failure, not a supported operating mode. Shipping it would make the BLE preview the worst Native detection the user sees first.

### Keep deferred sensing config over BLE, drop only live notify and recalibrate

Rejected. Detector, hits, traffic, and threshold already have MQTT and Home Assistant owners. Extra BLE writes duplicate that family and keep Configure looking like a live tuning tool.

### Replace BLE live detection with an on-device HTTP status or telemetry server

Rejected. HTTP would avoid the BLE/Wi-Fi radio conflict, but Native already has MQTT as the operational plane after Wi-Fi is saved. A third transport for brokerless live motion is a thin use case at a real RAM, auth, and maintenance cost. A later local status page can be reconsidered as recovery diagnostics, not as a detection plane.

## Consequences

Benefits:

- CSI occupancy on the supported Native path is no longer competing with BLE
- nearby BLE setup, MQTT live sensing, Game, and Theremin follow one flow: provision nearby, then detect and operate over MQTT
- the occupancy floor stays a sensing contract, not a BLE workaround

Trade-offs:

- first-boot Native cannot preview live motion until MQTT is up
- BLE clients that still send sensing writes or subscribe for movement must move those operations to MQTT
- BLE clients that send OTA commands must move those operations to MQTT
- the unused telemetry characteristic remains until a later GATT cleanup if binary size or service shape justifies removing it
- holding BOOT for the recovery interval changes the running mode, so boards that reuse that pin must override or disable the recovery-button Kconfig option

## Related

- `docs/ESPECTRE_PROTOCOL.md`
- `src/cpp/frontend/native/README.md`
- `docs/adr/2026-07-02-use-a-shared-espectre-protocol-across-esp-idf-frontends.md`
- `docs/adr/2026-08-15-use-fixed-temporal-csi-admission.md`
