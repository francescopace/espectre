# ESPectre × NM-CYD-C5 user guide

> Firmware: `src/cpp/frontend/esphome/examples/espectre-cyd-c5.yaml` (extends `espectre-c5.yaml`)
> Hardware: NM-CYD-C5 (external-antenna version recommended, nm-cyd-c5-ant; ESP32-C5, 2.8" 320×240 ST7789 touch display)

ESPectre detects motion from Wi-Fi signals. On the NM-CYD-C5, this firmware adds a touch screen: a live motion graph with the threshold line, buttons to adjust the threshold, and one-tap calibration. Home Assistant and the web tools work as usual.

---

## 1. Feature overview

| Feature | Access |
|---|---|
| Motion detection (Lightweight or High Accuracy) | Automatic; nothing to wear |
| Movement curve + Threshold line | On-device display (320×240) |
| Touch threshold adjustment (±0.05, range 0–1) | Screen buttons / HA |
| Calibration (automatic threshold) | Screen CALIBRATE button / HA Recalibrate button |
| Home Assistant integration | Native ESPHome API (auto-discovery) |
| Web tools (entity control, device settings) | https://espectre.dev/tools/ |

## 2. First-time provisioning

Build this board's configuration from the repository root:

```bash
./espectre esphome config --chip c5 --config src/cpp/frontend/esphome/examples/espectre-cyd-c5.yaml
./espectre esphome build --chip c5 --config src/cpp/frontend/esphome/examples/espectre-cyd-c5.yaml
```

The standard C5 image does not include the display. General setup and update instructions are in the [ESPHome guide](../../README.md).

After flashing, the device has no Wi-Fi settings and opens a setup hotspot:

1. On your phone or computer, connect to the **`ESPectre Fallback`** Wi-Fi network.
2. The setup page opens by itself (otherwise, go to `192.168.4.1`).
3. Choose your Wi-Fi network and enter its password.
4. The device restarts and connects.
5. **Its IP address appears at the top left of the screen.** Yellow `NO WIFI` means it is not connected.

> You can also provision over USB with Improv Serial, e.g. `./espectre provision --ssid MyNetwork` or the web flasher at https://espectre.dev/tools/flash/.

## 3. Adding to Home Assistant

Prerequisite: the device and Home Assistant are on the **same subnet**.

1. Install the **ESPHome** integration (Settings → Devices & Services → Add Integration → ESPHome).
2. Home Assistant finds the device by itself (as `espectre-` plus the last six characters of its MAC address). Click it and confirm.
3. If it does not appear, add it with the IP address shown on the screen.

Once added, the main entities are:

| Entity | Type | Description |
|---|---|---|
| Movement Score | sensor | Motion score (data source of the curve) |
| Motion Detected | binary_sensor | Motion state (usable as an automation trigger) |
| Threshold | number | Threshold, normalized range 0–1 (linked with the screen buttons) |
| Calibration Active | binary_sensor | ON while calibration is running |
| Recalibrate | button | Triggers re-calibration |
| WiFi Signal | sensor | Signal strength in dBm |

## 4. Screen layout and controls

```
┌──────────────────────────────────────────────┐
│ 192.168.1.100   MOTION/IDLE/CAL…  lightweight│ Header: IP / state / algorithm
├──────────────────────────────────────────────┤
│        ╭─╮        Movement curve (cyan)       │
│       ╱   ╲╭─╮    over-threshold segments red │
│  - - - - - - - -  Threshold dashed line (yellow)│
├──────────────────────────────────────────────┤
│  0.32 mv   0.25 thr                 -55dBm   │ Values row
├────────────┬──────────────────┬──────────────┤
│  THR −0.05 │    CALIBRATE     │   THR +0.05  │ Touch buttons
└────────────┴──────────────────┴──────────────┘
```

- **Graph**: about 68 seconds of history (about 4 points per second), with automatic scale. Parts of the curve above the yellow threshold line turn red.
- **THR −0.05 / +0.05**: lower or raise the threshold (0–1); the change is immediate.
- **CALIBRATE**: recalculates the threshold in about 10 seconds. Keep the room still meanwhile; `Calibration Active` is ON while it runs.
- Status text: `MOTION` (red) / `IDLE` (green) / `CALIBRATING...` (blue) / `BOOT` (yellow).

## 5. Web tools

Open https://espectre.dev/tools/device-settings/ and connect with the IP shown on the screen, or use **Auto-discovery** (see the [discovery reference](../../../../../../docs/DISCOVERY.md)).

To update the firmware, use the ESPHome dashboard or `esphome upload` over Wi-Fi.

## 6. Everyday tips

- **Placement:** avoid metal between the device and the router; see the [placement guide](https://espectre.dev/guides/placement/).
- **Recalibrate** after moving the device or rearranging the room, or when false alarms or missed movements increase. Keep the room empty and still. You can also use Recalibrate in Home Assistant.
- **Threshold:** too many false alarms → raise it; missed movements → lower it. For small movements, recalibrate before fine-tuning.
- **Network loss:** the device reconnects and resumes detection by itself.

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| Screen shows `NO WIFI` | Connect to the `ESPectre Fallback` network and set up Wi-Fi again. A 2.4 GHz network is recommended |
| Home Assistant cannot find the device | Check that both are on the same network or VLAN, or add the device with the IP on the screen |
| Touch buttons do not respond or are offset | Touch screens vary: adjust the four values in `touchscreen.calibration` in the YAML |
