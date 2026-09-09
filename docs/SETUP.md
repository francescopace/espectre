# Setup guide

Install a frontend, connect it to your network, and check that it detects movement. Use the published firmware for browser installation. Local builds are covered in [CLI.md](CLI.md#local-build-prerequisites); firmware integration is covered in [SDK.md](SDK.md).

## Choose your frontend

| Frontend | Use it for | Configuration and integration |
|----------|------------|-------------------------------|
| ESPHome | Home Assistant entities and YAML configuration | [README.md](../src/cpp/frontend/esphome/README.md) |
| Native | A standalone sensor, browser tools, and optional MQTT or Home Assistant MQTT Discovery | [README.md](../src/cpp/frontend/native/README.md) |
| Matter | A Matter occupancy sensor, with detector settings available through Direct HTTP | [README.md](../src/cpp/frontend/matter/README.md) |
| Micro-ESPectre | Research and MicroPython development | [README.md](../src/python/micro_espectre/README.md) |

## Check the hardware

You need a supported ESP32 board, a USB cable for flashing, and Wi-Fi. Every supported chip can use 2.4 GHz; ESP32-C5 also supports 5 GHz. Start with 2.4 GHz because detection quality on 5 GHz has not been characterized.

Use a LAN that permits communication between the browser and the device. The default internal ping source sends ICMP Echo Requests to the Wi-Fi gateway and uses its replies for sensing. Check that the network allows this traffic; [TROUBLESHOOTING.md](TROUBLESHOOTING.md#lan-traffic-blocked) covers isolation and filtering problems.

| Frontend | Supported chips | Delivery |
|----------|-----------------|----------|
| `ESPHome` | `ESP32-S3`, `ESP32-S2`, `ESP32-C6`, `ESP32-C5`, `ESP32-C3`, `ESP32` | Published web-flash images, Improv Serial, and fallback-AP provisioning |
| `Native` | `ESP32`, `ESP32-S3`, `ESP32-S2`, `ESP32-C3`, `ESP32-C5`, `ESP32-C6` | Published web-flash images and Improv Serial |
| `Matter` | `ESP32`, `ESP32-S3`, `ESP32-C3`, `ESP32-C5`, `ESP32-C6` | Published web-flash images and Matter commissioning |

## Web flash (no coding required)

Use desktop Chrome or Edge for Web Serial flashing. Firefox, Safari, and mobile browsers cannot use this installation path; use the local workflow in [CLI.md](CLI.md#local-build-prerequisites).

Choose `Release` for official firmware, `Preview` for the latest build from `main`, or `Development` for the latest build from `develop`.

1. Connect the board over USB
2. Open [espectre.dev/tools/flash](https://espectre.dev/tools/flash/), select **Connect USB device**, and choose the board from the browser's serial-port list
3. Wait for the installer to detect the chip and current firmware, then choose an update, reinstall, or another firmware type and channel
4. Review whether the installation preserves device data or erases the complete flash, then confirm
5. Keep the page open until the board restarts, and complete the displayed setup step

If the board does not enter download mode automatically, use its `BOOT` and `RESET` controls: hold `BOOT`, press and release `RESET`, release `BOOT`, and retry the flash. Board labels and automatic-reset behavior vary, so use the board documentation when those controls are named differently.

## Connect and configure

For Native and ESPHome, complete the installer's Wi-Fi setup over USB. For Matter, use the displayed QR or manual code to commission the device with your Matter controller. The selected frontend's README covers provisioning recovery and integration with Home Assistant or Matter.

Once the device joins your network:

1. Open [Device settings](https://espectre.dev/tools/device-settings/) on a computer on the same LAN.
2. Use the installer's device link, enter the current private IP, or select **Auto-discovery**. Grant the browser's local-network permission when prompted.
3. Set a device name and check its Wi-Fi connection. On Native, configure MQTT here if your integration needs it; enter your own endpoint and credentials.
4. Keep the default detector and traffic settings for the first test. Published C++ firmware starts with Lightweight Detection and internal ping traffic.

If the browser cannot reach the device, follow [TROUBLESHOOTING.md](TROUBLESHOOTING.md#device-not-reachable). The frontend READMEs describe their available controls and configuration syntax; [SDK.md](SDK.md#shared-sensing-options) contains the shared parameter reference.

## Sensor placement

Keep the device out of metal enclosures and behind as few heavy obstacles as practical. A distance of roughly `3-8 m` from the access point is a starting point. Walls, antenna orientation, access-point power, and furniture can matter more than distance.

At the chosen position, open [Monitor](https://espectre.dev/tools/monitor/) and check that valid CSI packets arrive steadily. Occupancy shows how much of the detector window contains valid input. The [sensor placement guide](https://espectre.dev/guides/placement/) covers room layouts, RSSI ranges, and a repeatable placement test.

## Check the first detection

If you moved the device after its initial calibration, restart it or use the advertised recalibration control at its final position. Keep the room quiet while Lightweight calibrates, and wait for calibration and detector readiness in Monitor before judging the result.

If you selected High Accuracy, it uses its trained threshold and skips quiet-room calibration. It still waits for valid CSI input and feature-window warmup before detection becomes ready. Use [TROUBLESHOOTING.md](TROUBLESHOOTING.md#detection-profile) when deciding whether to change the profile.

Walk through the monitored area and confirm that the movement score responds and the motion state changes. Stop moving and check that it returns to idle. Repeat the test from the positions you need to monitor.

Use [TROUBLESHOOTING.md](TROUBLESHOOTING.md) if CSI is missing, calibration stalls, or detection is unreliable. Its tuning section explains when to change the profile, threshold, or motion-hit settings.
