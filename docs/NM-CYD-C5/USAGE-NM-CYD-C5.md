# ESPectre × NM-CYD-C5 User Guide

English | [简体中文](USAGE-NM-CYD-C5_zh.md)

> Firmware: `examples/espectre-cyd-c5.yaml` (nm-cyd-c5 branch)
> Hardware: NM-CYD-C5 (external-antenna version recommended: nm-cyd-c5-ant; ESP32-C5, 2.8" 320×240 ST7789 touch display)

ESPectre is a WiFi CSI (Channel State Information) based human motion detection system. In addition to the core motion detection, this firmware on the NM-CYD-C5 provides complete local interaction: a live motion curve, threshold line display, touch buttons for manual threshold adjustment, one-tap calibration, plus Home Assistant, web console and Bluetooth monitor support.

---

## 1. Feature Overview

| Feature | Access |
|---|---|
| Motion detection (MVS / ML algorithms) | Runs automatically — passive WiFi CSI sensing, nothing to wear |
| Movement curve + Threshold line | On-device display (320×240) |
| Touch threshold adjustment (±0.5, range 0–10) | Screen buttons / HA / Web / Bluetooth |
| NBVI auto-calibration (subcarrier selection + adaptive threshold) | Screen CALIBRATE button / HA Calibrate switch |
| Home Assistant integration | Native ESPHome API (auto-discovery) |
| Web console (entity view/control, logs, OTA) | Browser → device IP |
| Bluetooth live monitor (25 Hz curve) | `micro-espectre/espectre-monitor.html` |

![](c5-espectre.png)

## 2. First-Time Provisioning

The device ships with no saved WiFi credentials and automatically starts a provisioning hotspot on first boot after flashing:

1. On your phone or computer, scan for WiFi networks and connect to the **`ESPectre Fallback`** hotspot;
2. The captive portal opens automatically in the browser (or browse manually to `192.168.4.1`);
3. Select your WiFi network and enter its password — **a 2.4 GHz network is officially recommended** (ESPectre's CSI detection is forced onto the 2.4 GHz band; a 5 GHz network cannot be used for detection);
4. After a successful save, the device reboots and connects to your WiFi automatically;
5. Once connected, **the device IP address is shown directly in the top-left header of the screen** (yellow `NO WIFI` is displayed while disconnected).

> You can also provision over Bluetooth (BLE Improv) using the Home Assistant mobile app — the result is the same.

## 3. Adding to Home Assistant

Prerequisite: the device and Home Assistant are on the **same subnet**.

1. In Home Assistant, install the **ESPHome** integration (Settings → Devices & Services → Add Integration → ESPHome);
2. Open Home Assistant → **Settings** → find **ESPHome** → click **Add New Device**;
3. When the device and HA are on the same network it is **discovered automatically** (mDNS hostname `espectre`) — click it and confirm to finish adding; if it is not discovered, enter the IP address shown on the screen manually (port 6053; this firmware has no API encryption, so no encryption key is needed).

![](ESPHome-add.png)

Once added, 5 entities are available:

| Entity | Type | Description |
|---|---|---|
| Movement Score | sensor | Motion score (sliding variance, data source of the curve) |
| Motion Detected | binary_sensor | Motion state (usable as an automation trigger) |
| Threshold | number | Threshold, range 0–10 (linked with the screen buttons) |
| Calibrate | switch | Triggers re-calibration, turns OFF automatically when done |
| WiFi Signal | sensor | Signal strength in dBm |

![](ESPHome.png)

## 4. Screen Layout & Operation

```
┌──────────────────────────────────────────────┐
│ 192.168.1.100   MOTION/IDLE/CAL…       mvs   │ Header: IP / state / algorithm
├──────────────────────────────────────────────┤
│        ╭─╮        Movement curve (cyan)       │
│       ╱   ╲╭─╮    over-threshold segments red │
│  - - - - - - - -  Threshold dashed line (yellow)│
├──────────────────────────────────────────────┤
│  1.83 mv   1.10 thr                 -55dBm   │ Values row
├────────────┬──────────────────┬──────────────┤
│  THR −0.5  │    CALIBRATE     │   THR +0.5   │ Touch buttons
└────────────┴──────────────────┴──────────────┘
```

![](c5-no-wifi.png)

- **Graph area**: ~4.5 minutes of scrolling history (1 point/s), auto-scaled Y axis; curve segments above the yellow threshold line turn red, making the "decision" process intuitive.
- **THR −0.5 / +0.5**: step-adjust the threshold, range 0–10, takes effect immediately (session-level; after reboot it is recomputed from the calibration value / config).
- **CALIBRATE**: triggers NBVI re-calibration (~10 seconds — keep the environment still during it). Calibration automatically selects the optimal subcarriers and computes an adaptive threshold (P95×1.1). **A calibration result far above 10 (up to 10.0+) is normal**.
- Status text: `MOTION` (red) / `IDLE` (green) / `CALIBRATING...` (blue) / `BOOT` (yellow).

## 5. Web Console

Open `http://<device-ip>/` in a browser (the IP is shown in the screen header):

- View/control all entities (Movement Score, Motion, Threshold slider, Calibrate);
- A live log window at the bottom of the page;
- **OTA firmware upload**: future firmware updates can be uploaded directly as `firmware.ota.bin` from the web page — no USB needed.

> There is no access password by default — use only on a trusted LAN. To add authentication, enable `web_server.auth` in the YAML.

## 6. Bluetooth Live Monitor (Optional)

When you need a smoother real-time curve than the screen: open `micro-espectre/espectre-monitor.html` from this repository in Chrome/Edge and connect directly to the device over Web Bluetooth (25 Hz push of Movement + Threshold). You can watch the live curve and adjust the threshold with a slider. The device must be within Bluetooth range, and the browser must support Web Bluetooth (Chrome/Edge).

## 7. Daily Usage Tips

- **Placement**: keep the line of sight between the device and the router free of metal obstructions — CSI is sensitive to 2.4 GHz link quality;
- **When to calibrate**: after moving the device, changing the room layout, or when false positives/negatives become noticeable, press CALIBRATE again (keep the room empty and still during calibration); or trigger the Calibrate entity from the ESPHome web page or Home Assistant;
- **Threshold tuning**: more false positives → increase; more missed detections → decrease. For detecting small movements, re-calibrate first before fine-tuning;
- **Self-healing on network loss**: the device reconnects WiFi automatically, and CSI detection resumes by itself — no intervention needed.

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| Screen shows `NO WIFI` | Not connected: join the `ESPectre Fallback` hotspot and provision again, making sure to use a 2.4 GHz network |
| Browser cannot open the device IP | Confirm the firmware with `web_server` is flashed, and the device is on the same subnet as your computer |
| HA cannot find the device | Check the subnet/VLAN; or add it manually with IP:6053 |
| Touch buttons unresponsive or offset | Resistive touch varies per unit — fine-tune the four boundary values in `touchscreen.calibration` in the YAML |
| Calibration fails or is unsatisfactory | Keep the environment absolutely still during calibration; when the signal is too strong (<50 cm from the router) the gain lock skips, which is expected |
| Firmware update | Web OTA upload, or `esphome upload` (direct over WiFi, no USB) |
