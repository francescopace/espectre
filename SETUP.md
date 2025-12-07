# 🛜 ESPectre 👻 - Setup Guide

Complete guide to install and configure ESPectre with ESPHome.

---

## What You Need

---

**Hardware:**
- **ESP32 with CSI support** - ESP32-S3 or ESP32-C6 tested. Other variants (ESP32, S2, C3, C5) also supported experimentally.
- USB-C or Micro-USB cable (depending on board)
- Wi-Fi router (2.4 GHz, 802.11b|g|n|ax)

**Software:**
- Python 3.12 (⚠️ Python 3.14 has known issues with ESPHome)
- ESPHome 2024.x or newer
- Home Assistant (recommended, but optional)

---

## Quick Start

---

### 1. Install ESPHome

```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# venv\Scripts\activate   # On Windows

# Install ESPHome
pip install esphome
```

### 2. Download a configuration file

Download the example configuration for your hardware:

| Platform | Configuration File | Status |
|----------|-------------------|--------|
| **ESP32-C6** | [examples/espectre-c6.yaml](https://raw.githubusercontent.com/francescopace/espectre/main/examples/espectre-c6.yaml) | ✅ Tested |
| **ESP32-S3** | [examples/espectre-s3.yaml](https://raw.githubusercontent.com/francescopace/espectre/main/examples/espectre-s3.yaml) | ✅ Tested |
| **ESP32-C5** | [examples/espectre-c5.yaml](https://raw.githubusercontent.com/francescopace/espectre/main/examples/espectre-c5.yaml) | ⚠️ Experimental |
| **ESP32-C3** | [examples/espectre-c3.yaml](https://raw.githubusercontent.com/francescopace/espectre/main/examples/espectre-c3.yaml) | ⚠️ Experimental |
| **ESP32-S2** | [examples/espectre-s2.yaml](https://raw.githubusercontent.com/francescopace/espectre/main/examples/espectre-s2.yaml) | ⚠️ Experimental |
| **ESP32** | [examples/espectre-esp32.yaml](https://raw.githubusercontent.com/francescopace/espectre/main/examples/espectre-esp32.yaml) | ⚠️ Experimental |

These files are pre-configured to download the component automatically from GitHub.

> ⚠️ **Experimental platforms**: ESP32, ESP32-S2, ESP32-C3, and ESP32-C5 have CSI support but have not been extensively tested. Please report your results on [GitHub Discussions](https://github.com/francescopace/espectre/discussions)!

### 3. Edit WiFi credentials

Open the downloaded file and replace `YOUR_WIFI_SSID` and `YOUR_WIFI_PASSWORD` with your WiFi credentials.

### 4. Build and flash

```bash
esphome run espectre-c6.yaml  # or espectre-s3.yaml
```

That's it! 🎉 The device will be automatically discovered by Home Assistant.

---

## Development Setup

---

For development, contributions, or offline use, use the pre-configured development files.

### 1. Clone the repository

```bash
git clone https://github.com/francescopace/espectre.git
cd espectre
```

### 2. Install ESPHome

```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
pip install esphome
```

### 3. Create secrets file

```bash
cat > secrets.yaml << EOF
wifi_ssid: "YourWiFiName"
wifi_password: "YourWiFiPassword"
EOF
```

### 4. Build and flash

Use the development configuration files (with debug sensors and local component path):

| Platform | Development File |
|----------|-----------------|
| **ESP32-C6** | `examples/espectre-c6-dev.yaml` |
| **ESP32-S3** | `examples/espectre-s3-dev.yaml` |

```bash
# For ESP32-C6
esphome run examples/espectre-c6-dev.yaml

# For ESP32-S3
esphome run examples/espectre-s3-dev.yaml
```

### Development vs Production Files

| File | Component Source | WiFi | Logger | Debug Sensors |
|------|-----------------|------|--------|---------------|
| `espectre-c6.yaml` | GitHub | Placeholder | INFO | ❌ |
| `espectre-c6-dev.yaml` | Local | secrets.yaml | DEBUG | ✅ |
| `espectre-s3.yaml` | GitHub | Placeholder | INFO | ❌ |
| `espectre-s3-dev.yaml` | Local | secrets.yaml | DEBUG | ✅ |

---

## Docker / Home Assistant Add-on

---

If you run ESPHome in Docker or as a Home Assistant add-on, just download an example file to your config directory.

**Example for Docker with bind mount:**

```bash
# Your docker-compose.yml mounts /home/user/esphome/config:/config
cd /home/user/esphome/config

# Download the configuration file
curl -O https://raw.githubusercontent.com/francescopace/espectre/main/examples/espectre-c6.yaml

# Edit WiFi credentials in the file
vi espectre-c6.yaml  # Replace YOUR_WIFI_SSID and YOUR_WIFI_PASSWORD

# Run ESPHome
docker compose exec esphome esphome run espectre-c6.yaml
```

No need to copy any files manually - the component is downloaded automatically from GitHub!

---

## Configuration Parameters

---

### ESPectre Component

All parameters can be adjusted in the YAML file under the `espectre:` section:

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `traffic_generator_rate` | int | 100 | 0-1000 | Packets/sec for CSI generation (0=disabled) |
| `segmentation_threshold` | float | 1.0 | 0.5-10.0 | Motion sensitivity (lower=more sensitive) |
| `segmentation_window_size` | int | 50 | 10-200 | Moving variance window in packets |
| `selected_subcarriers` | list | auto | 0-63 | Fixed subcarriers (omit for auto-calibration) |
| `hampel_enabled` | bool | true | - | Enable Hampel outlier filter |
| `hampel_window` | int | 7 | 3-11 | Hampel filter window size |
| `hampel_threshold` | float | 4.0 | 1.0-10.0 | Hampel filter sensitivity (MAD multiplier) |
### Integrated Sensors (Created Automatically)

All sensors are created automatically when the `espectre` component is configured. You can optionally customize their names.

| Sensor Config | Type | Default Name | Description |
|---------------|------|--------------|-------------|
| `movement_sensor` | sensor | "Movement Score" | Current motion intensity value |
| `motion_sensor` | binary_sensor | "Motion Detected" | Motion state (on/off) |
| `threshold_number` | number | "Threshold" | Detection threshold (adjustable from HA) |

### Customizing Sensor Names

```yaml
espectre:
  movement_sensor:
    name: "Living Room Movement"
  motion_sensor:
    name: "Living Room Motion"
  threshold_number:
    name: "Living Room Threshold"
```

---

## Home Assistant Integration

---

ESPHome provides **automatic Home Assistant integration**. Once the device is flashed and connected to WiFi:

1. Home Assistant will automatically discover the device
2. Go to **Settings** → **Devices & Services** → **ESPHome**
3. Click **Configure** on the discovered device
4. All sensors will be automatically added

### Entities Created

Entity names are based on the device name in your YAML (default: `espectre`):

- **binary_sensor.espectre_motion_detected** - Motion state (on/off)
- **sensor.espectre_movement_score** - Movement intensity value
- **number.espectre_threshold** - Detection threshold (adjustable from Home Assistant)

> **Note:** If you change the device name, replace `espectre` with your device name in automations and dashboards.

### Automation Example

```yaml
automation:
  - alias: "Turn on light on motion"
    trigger:
      - platform: state
        entity_id: binary_sensor.espectre_motion_detected
        to: "on"
    action:
      - service: light.turn_on
        target:
          entity_id: light.living_room

  - alias: "Turn off light after no motion"
    trigger:
      - platform: state
        entity_id: binary_sensor.espectre_motion_detected
        to: "off"
        for:
          minutes: 5
    action:
      - service: light.turn_off
        target:
          entity_id: light.living_room
```

**Inactivity alert:**

```yaml
automation:
  - alias: "Inactivity Alert"
    trigger:
      - platform: state
        entity_id: binary_sensor.espectre_motion_detected
        to: "off"
        for:
          hours: 4
    condition:
      - condition: time
        after: "08:00:00"
        before: "22:00:00"
    action:
      - service: notify.mobile_app
        data:
          message: "No movement detected for 4 hours"
```

### Dashboard Examples

Two dashboard examples are available:

| Dashboard | Description |
|-----------|-------------|
| [home-assistant-dashboard.yaml](examples/home-assistant-dashboard.yaml) | Production dashboard with motion sensors |
| [home-assistant-dashboard-dev.yaml](examples/home-assistant-dashboard-dev.yaml) | Development dashboard with debug sensors (Free Heap, Loop Time, etc.) |

**How to use:**
1. Go to **Settings** → **Dashboards** → **Add Dashboard**
2. Click **Edit** on the new dashboard
3. Click the three dots menu → **Raw configuration editor**
4. Paste the YAML content from the file

> **Note:** If you changed the device name from `espectre`, replace all occurrences of `espectre_` with your device name (e.g., `espectre_living_room_`).

**Production dashboard includes:**
- **Gauge**: Visual representation of movement score with color-coded severity
- **Motion tile**: Current motion state with last changed time
- **Threshold control**: Adjustable detection threshold
- **History graph**: 24-hour view of movement and threshold

**Development dashboard adds:**
- **Free Heap**: Available memory (monitor for leaks)
- **Max Free Block**: Largest contiguous memory block
- **Loop Time**: Main loop execution time

---

## Traffic Generator

---

**⚠️ IMPORTANT:** The traffic generator is **ESSENTIAL** for CSI packet generation. Without it, the ESP32 receives zero CSI packets and detection will not work.

### What it does

- Generates UDP broadcast packets at configurable rate
- Each packet triggers a CSI callback from the WiFi driver
- Ensures continuous CSI data availability

### Recommended rates

| Rate | Use Case |
|------|----------|
| 50 pps | Basic presence detection, minimal overhead |
| 100 pps | **Recommended** - Activity recognition |
| 600-1000 pps | Fast motion detection, precision localization |
| 0 pps | Disabled (only if you have other continuous WiFi traffic) |

### Configuration

```yaml
espectre:
  traffic_generator_rate: 100  # packets per second
```

---

## NBVI Auto-Calibration

---

> ⚠️ **CRITICAL**: The room must be **still** during the first 10 seconds after boot. Movement during calibration will result in poor detection accuracy!

ESPectre automatically selects optimal subcarriers using the **NBVI algorithm**:

1. Collects 1000 CSI packets (~10 seconds)
2. Analyzes baseline characteristics (Room must be quiet, don't move)
3. Selects optimal 12 subcarriers
4. Saves configuration (persists across reboots)

To force recalibration: erase flash and re-flash.

---

## Custom Hardware Configuration

---

ESPectre now provides example configurations for all ESP32 variants with CSI support. If you need to customize further, use these guidelines:

### Required sdkconfig options

```yaml
esp32:
  variant: YOUR_VARIANT  # e.g., ESP32, ESP32C3
  framework:
    type: esp-idf
    version: 5.5.1
    sdkconfig_options:
      # Required for CSI (mandatory)
      CONFIG_ESP_WIFI_CSI_ENABLED: y
      
      # Power management (mandatory - disable for CSI reliability)
      CONFIG_PM_ENABLE: n
      CONFIG_ESP_WIFI_STA_DISCONNECTED_PM_ENABLE: n
      
      # WiFi 6 (optional - only if supported by your chip)
      # CONFIG_ESP_WIFI_11AX_SUPPORT: y
```

**Note:** For advanced sdkconfig tuning see official espressif doc: [ESP32 WiFi](https://docs.espressif.com/projects/esp-idf/en/stable/esp32/api-guides/wifi.html#how-to-configure-parameters).

---

## Troubleshooting

---

### No motion detection

1. **Verify traffic generator is enabled** (`traffic_generator_rate > 0`)
2. Check WiFi is connected (look for IP address in logs)
3. Wait for NBVI calibration to complete (~5-10 seconds after boot)
4. Adjust `segmentation_threshold` (try 0.5-2.0 for more sensitivity)

### False positives

1. Increase `segmentation_threshold` (try 2.0-5.0)
2. Check for interference sources (fans, AC, moving curtains)
3. Increase `segmentation_window_size` for more stable detection

### Calibration fails

1. Ensure room is quiet during calibration (first 5-10 seconds after boot)
2. Check traffic generator is running
3. Verify WiFi connection is stable

### Flash failed

1. Hold BOOT button on ESP32
2. Press RESET button
3. Release BOOT button
4. Run flash command again

### View logs

```bash
# Via USB
esphome logs <your-config>.yaml

# Via network (after first flash)
esphome logs <your-config>.yaml --device espectre.local
```

---

## Next Steps

---

- 📖 **Tuning Guide**: [TUNING.md](TUNING.md) - Optimize for your environment
- 📖 **Main Documentation**: [README.md](README.md) - Full project overview

---

## 📄 License

GPLv3 - See [LICENSE](LICENSE) for details.
