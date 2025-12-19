# Setup Guide

Complete guide to install and configure ESPectre with ESPHome.

---

## What You Need

---

**Hardware:**
- **ESP32 with CSI support** - ESP32-S3, ESP32-C6 or ESP32-C3 tested. Other variants (ESP32, S2, C5) also supported experimentally.
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

| Platform | Configuration File | CPU | WiFi | PSRAM | Status |
|----------|-------------------|-----|------|-------|--------|
| **ESP32-C6** | [espectre-c6.yaml](https://raw.githubusercontent.com/francescopace/espectre/main/examples/espectre-c6.yaml) | RISC-V @ 160MHz | WiFi 6 | ❌ | ✅ Tested |
| **ESP32-S3** | [espectre-s3.yaml](https://raw.githubusercontent.com/francescopace/espectre/main/examples/espectre-s3.yaml) | Xtensa @ 240MHz | WiFi 4 | ✅ 8MB | ✅ Tested |
| **ESP32-C3** | [espectre-c3.yaml](https://raw.githubusercontent.com/francescopace/espectre/main/examples/espectre-c3.yaml) | RISC-V @ 160MHz | WiFi 4 | ❌ | ✅ Tested |
| **ESP32-C5** | [espectre-c5.yaml](https://raw.githubusercontent.com/francescopace/espectre/main/examples/espectre-c5.yaml) | RISC-V @ 240MHz | WiFi 6 | ❌ | ⚠️ Experimental ¹ |
| **ESP32-S2** | [espectre-s2.yaml](https://raw.githubusercontent.com/francescopace/espectre/main/examples/espectre-s2.yaml) | Xtensa @ 240MHz | WiFi 4 | Optional | ⚠️ Experimental |
| **ESP32** | [espectre-esp32.yaml](https://raw.githubusercontent.com/francescopace/espectre/main/examples/espectre-esp32.yaml) | Xtensa @ 240MHz | WiFi 4 | Optional | ⚠️ Experimental |

**Recommendations**:
- **ESP32-C6**: Best for WiFi 6 environments, standard motion detection
- **ESP32-S3**: Best for advanced applications, future ML features (more memory)
- **ESP32-C3**: Budget-friendly option, compact form factor

These files are pre-configured to download the component automatically from GitHub.

> ⚠️ **Experimental platforms**: ESP32, ESP32-S2, and ESP32-C5 have CSI support but have not been extensively tested. Please report your results on [GitHub Discussions](https://github.com/francescopace/espectre/discussions)!
>
> ¹ ESP32-C5: `improv_serial` (USB provisioning) not yet supported by ESPHome. Use BLE or WiFi AP provisioning instead.

### 3. Build and flash

```bash
esphome run espectre-c6.yaml  # or espectre-s3.yaml
```

### 4. Configure WiFi

After flashing, configure WiFi using one of these methods:

| Method | How |
|--------|-----|
| **BLE** (easiest) | Use ESPHome app or Home Assistant Companion app |
| **USB** | Go to [web.esphome.io](https://web.esphome.io) → Connect → Configure WiFi |
| **Captive Portal** | Connect to "ESPectre Fallback" WiFi → Configure in browser |

That's it! The device will be automatically discovered by Home Assistant.

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
cat > examples/secrets.yaml << EOF
wifi_ssid: "YourWiFiName"
wifi_password: "YourWiFiPassword"
# Optional: lock to specific AP (useful for mesh networks)
# wifi_bssid: "AA:BB:CC:DD:EE:FF"
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
| `espectre-c6.yaml` | GitHub | Provisioning (BLE/USB/AP) | INFO | ❌ |
| `espectre-c6-dev.yaml` | Local | secrets.yaml | DEBUG | ✅ |
| `espectre-s3.yaml` | GitHub | Provisioning (BLE/USB/AP) | INFO | ❌ |
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

# Run ESPHome
docker compose exec esphome esphome run espectre-c6.yaml

# After flashing, configure WiFi via BLE, USB, or Captive Portal
```

No need to copy any files manually - the component is downloaded automatically from GitHub!

---

## Configuration Parameters

---

### ESPectre Component

All parameters can be adjusted in the YAML file under the `espectre:` section:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `traffic_generator_rate` | int | 100 | Packets/sec for CSI generation (0=disabled) |
| `segmentation_threshold` | float | 1.0 | Motion sensitivity (lower=more sensitive) |
| `segmentation_window_size` | int | 50 | Moving variance window in packets |
| `selected_subcarriers` | list | auto | Fixed subcarriers (omit for auto-calibration) |
| `lowpass_enabled` | bool | false | Enable low-pass filter for noise reduction |
| `lowpass_cutoff` | float | 11.0 | Low-pass filter cutoff frequency in Hz (5-20) |
| `hampel_enabled` | bool | false | Enable Hampel outlier filter |
| `hampel_window` | int | 7 | Hampel filter window size |
| `hampel_threshold` | float | 4.0 | Hampel filter sensitivity (MAD multiplier) |

For detailed parameter tuning (ranges, recommended values, troubleshooting), see [TUNING.md](TUNING.md).
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

> ⚠️ **Multiple devices?** If you uncommented `name_add_mac_suffix: true` in your YAML, entity names will include the MAC suffix (e.g., `sensor.espectre_a1b2c3_movement_score`). Update the dashboard entities accordingly.

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

The traffic generator creates UDP broadcast packets that trigger CSI callbacks from the WiFi driver. Default rate is **100 pps** (packets per second).

```yaml
espectre:
  traffic_generator_rate: 100  # packets per second (0-1000)
```

For detailed rate recommendations and Nyquist-Shannon sampling theory, see [TUNING.md](TUNING.md#traffic-generator-rate-0-1000-pps).

---

## NBVI Auto-Calibration

---

> ⚠️ **CRITICAL**: The room must be **still** during the first 10 seconds after boot. Movement during calibration will result in poor detection accuracy!

ESPectre automatically calibrates in two phases:

1. **Gain Lock** (~3 seconds, 300 packets): Stabilizes AGC/FFT for consistent amplitudes
2. **NBVI Calibration** (~7 seconds, 700 packets): Selects optimal 12 subcarriers
3. Saves configuration (persists across reboots)

Room must be quiet during the entire ~10 second calibration.

To force recalibration: erase flash and re-flash.

---

## Custom Hardware Configuration

---

ESPectre now provides example configurations for all ESP32 variants with CSI support. If you need to customize further, use these guidelines:

### Automatic sdkconfig options

ESPectre automatically sets all required and recommended sdkconfig options. You don't need to manually configure anything in most cases.

The component automatically configures:

| Option | Value | Purpose |
|--------|-------|---------|
| `CONFIG_ESP_WIFI_CSI_ENABLED` | `y` | Enable CSI (mandatory) |
| `CONFIG_PM_ENABLE` | `n` | Disable power management |
| `CONFIG_ESP_WIFI_STA_DISCONNECTED_PM_ENABLE` | `n` | Disable disconnected PM |
| `CONFIG_ESP_WIFI_AMPDU_TX_ENABLED` | `n` | More CSI callbacks |
| `CONFIG_ESP_WIFI_AMPDU_RX_ENABLED` | `n` | More CSI callbacks |
| `CONFIG_ESP_WIFI_DYNAMIC_RX_BUFFER_NUM` | `128` | Larger RX buffer |
| `CONFIG_FREERTOS_HZ` | `1000` | 1ms tick for precise timing |

### Platform-specific options (optional)

You only need to add sdkconfig options for platform-specific features:

```yaml
esp32:
  variant: ESP32C6  # or ESP32S3, etc.
  framework:
    type: esp-idf
    version: 5.5.1
    sdkconfig_options:
      # WiFi 6 (optional - C5, C6 only)
      CONFIG_ESP_WIFI_11AX_SUPPORT: y
      
      # CPU frequency (platform-dependent)
      CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ: "160"  # 160 for C6, 240 for S3
      
      # PSRAM (if available on your board)
      # CONFIG_ESP32S3_SPIRAM_SUPPORT: y
```

**Reference:** For advanced sdkconfig tuning see official Espressif documentation: [ESP32 WiFi](https://docs.espressif.com/projects/esp-idf/en/stable/esp32/api-guides/wifi.html#how-to-configure-parameters).

---

## Troubleshooting

---

### No motion detection

1. **Verify traffic generator is enabled** (`traffic_generator_rate > 0`)
2. Check WiFi is connected (look for IP address in logs)
3. Wait for NBVI calibration to complete (~10 seconds after boot)
4. Adjust `segmentation_threshold` (try 0.5-2.0 for more sensitivity)

### False positives

1. Increase `segmentation_threshold` (try 2.0-5.0)
2. Check for interference sources (fans, AC, moving curtains)
3. Increase `segmentation_window_size` for more stable detection

### Calibration fails

1. Ensure room is quiet during calibration (first 5-10 seconds after boot)
2. Check traffic generator is running
3. Verify WiFi connection is stable

### Unstable detection with mesh networks

If you have a mesh WiFi network, the sensor may roam between access points causing CSI inconsistencies. Lock it to a specific AP using the BSSID.

**For development files** (`espectre-*-dev.yaml`):
1. Add `wifi_bssid` to your `secrets.yaml`:
   ```yaml
   wifi_bssid: "AA:BB:CC:DD:EE:FF"
   ```
2. Uncomment the `bssid` line in your config file:
   ```yaml
   wifi:
     networks:
       - ssid: !secret wifi_ssid
         password: !secret wifi_password
         bssid: !secret wifi_bssid
   ```

**For production files** (`espectre-*.yaml` with provisioning):
Add the BSSID directly after configuring WiFi, or use the ESPHome dashboard to edit the configuration.

To find your AP's BSSID:
- Check your router's admin page
- Use a WiFi analyzer app on your phone
- Look in ESPectre logs after connection (shows connected BSSID)

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

- **Tuning Guide**: [TUNING.md](TUNING.md) - Optimize for your environment
- **Main Documentation**: [README.md](README.md) - Full project overview

---

## License

GPLv3 - See [LICENSE](LICENSE) for details.
