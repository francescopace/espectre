# Setup guide

Install ESPectre, connect it to your network, and check that it detects movement. This guide uses the published firmware and the browser installer. For local builds, see [local build prerequisites](CLI.md#local-build-prerequisites). To embed ESPectre in your own firmware, see the [SDK guide](SDK.md).

## Choose your frontend

| Frontend | Use it for | Guide |
|----------|------------|-------|
| ESPHome | Home Assistant entities and YAML configuration | [ESPHome guide](../src/cpp/frontend/esphome/README.md) |
| Native | A standalone sensor with browser tools and optional MQTT or Home Assistant MQTT Discovery | [Native guide](../src/cpp/frontend/native/README.md) |
| Matter | A Matter occupancy sensor; detector settings are available through Direct HTTP | [Matter guide](../src/cpp/frontend/matter/README.md) |

For sensing research in MicroPython, see the [Micro-ESPectre guide](../src/python/micro_espectre/README.md).

## Check the hardware

You need a supported ESP32 board, a USB cable, and a Wi-Fi network.

| Frontend | Supported chips |
|----------|-----------------|
| ESPHome | ESP32, ESP32-S2, ESP32-S3, ESP32-C3, ESP32-C5, ESP32-C6 |
| Native | ESP32, ESP32-S2, ESP32-S3, ESP32-C3, ESP32-C5, ESP32-C6 |
| Matter | ESP32, ESP32-S3, ESP32-C3, ESP32-C5, ESP32-C6 |

All chips work on 2.4 GHz. ESP32-C5 also supports 5 GHz, but detection quality on 5 GHz has not been measured yet, so start with 2.4 GHz.

By default, the device pings the Wi-Fi gateway and uses the replies for sensing. Your network must allow this traffic and let the browser reach the device. If it does not, see [LAN traffic blocked](TROUBLESHOOTING.md#lan-traffic-blocked).

## Web flash (no coding required)

Use desktop Chrome or Edge. Firefox, Safari, and mobile browsers cannot flash over USB; use the [local workflow](CLI.md#local-build-prerequisites) instead.

Pick a release channel: `Release` for official firmware, `Preview` for the latest build from `main`, or `Development` for the latest build from `develop`.

1. Connect the board over USB.
2. Open [espectre.dev/tools/flash](https://espectre.dev/tools/flash/), select **Connect USB device**, and choose the board.
3. Wait for the installer to detect the chip and firmware, then choose what to install.
4. Check whether the installation keeps device data or erases the whole flash, then confirm.
5. Keep the page open until the board restarts, then complete the setup step shown.

If the board does not enter download mode, hold `BOOT`, press and release `RESET`, release `BOOT`, and try again. Some boards label these buttons differently; check the board documentation.

## Connect and configure

For Native and ESPHome, complete the Wi-Fi setup in the installer. For Matter, commission the device with your Matter controller using the QR code or manual code shown. The frontend guide covers provisioning recovery and Home Assistant or Matter integration.

Once the device is on your network:

1. Open [Device settings](https://espectre.dev/tools/device-settings/) on a computer on the same network.
2. Use the installer's device link, enter the device IP, or select **Auto-discovery**. Allow local-network access if the browser asks.
3. Set a device name and check the Wi-Fi connection. On Native, you can also set up MQTT here with your own broker and credentials.
4. Keep the default detector and traffic settings for the first test.

If the browser cannot reach the device, see [Device not reachable](TROUBLESHOOTING.md#device-not-reachable). For every shared setting, see [shared sensing options](SDK.md#shared-sensing-options).

### Optional: external traffic from Home Assistant

On 64-bit Home Assistant OS, the **ESPectre Traffic Generator** add-on can send sensing traffic instead of each device's internal generator. Its panel switches ESPHome and Native MQTT devices to external traffic and shows CSI diagnostics. Configure Matter devices through Device settings instead.

Match the add-on's `rate_pps` to the device's `csi_target_pps`, then check the CSI rate and sensing readiness in Monitor. See the [add-on documentation](../tools/ha_traffic_generator_addon/DOCS.md) for installation and options, and [external sources](CSI.md#external-sources) for how external traffic works.

## Sensor placement

Keep the device out of metal enclosures and away from heavy obstacles. About 3–8 m from the access point is a good start, but walls, antennas, and furniture often matter more than distance.

Open [Monitor](https://espectre.dev/tools/monitor/) at the chosen spot and check that CSI packets arrive steadily and occupancy stays high. The [placement guide](https://espectre.dev/guides/placement/) covers room layouts, RSSI ranges, and a repeatable placement test.

## Check the first detection

The default Lightweight profile calibrates at startup. If you moved the device after the first boot, restart it or press recalibrate in its final position. Keep the room quiet until Monitor shows that calibration is complete and the detector is ready.

High Accuracy skips calibration but still needs a few seconds of valid CSI before it is ready. See [detection profile](TROUBLESHOOTING.md#detection-profile) to choose between the two.

Then:

1. Walk through the area and check that the movement score rises and the state changes to motion.
2. Stand still and check that it returns to idle.
3. Repeat from every spot you want to cover.

If CSI is missing, calibration stalls, or detection is unreliable, see [troubleshooting](TROUBLESHOOTING.md).

## Official images and personal builds

Official Native and ESPHome images accept only signed OTA updates. This affects how you switch between official and personal builds:

- **Official to personal build:** install the personal build over USB. OTA rejects unsigned images.
- **Personal build to official:** install the full official image over USB. This restores signed OTA updates.
- **ESPHome Device Builder:** after adopting an official image, install the first Builder image over USB. Later Builder updates can use the network.
- **Matter:** has no OTA. Always update over USB. Matter firmware does not verify image signatures.

The installer checks each published download before it writes flash. OTA details are in each frontend guide. Signing keys, rotation, and recovery are in [firmware signing](RELEASING.md#firmware-signing).
