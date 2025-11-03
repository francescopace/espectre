# 🛜 ESPectre 👻 - Setup Guide

## What You Need

**Hardware:**
- ESP32-S3-DevKitC-1 N16R8 (16MB Flash, 8MB PSRAM)
- USB-C cable
- Wi-Fi router (2.4 GHz)

**Software:**
- ESP-IDF v5.x
- MQTT Broker (Mosquitto or Home Assistant)

---

## Installation

### 1. Install ESP-IDF

**Linux:**
```bash
sudo apt-get install git wget flex bison gperf python3 python3-pip \
  python3-venv cmake ninja-build ccache libffi-dev libssl-dev dfu-util libusb-1.0-0

mkdir -p ~/esp && cd ~/esp
git clone --recursive https://github.com/espressif/esp-idf.git
cd esp-idf && git checkout v5.3
./install.sh esp32s3
. ./export.sh
```

**macOS:**
```bash
brew install cmake ninja dfu-util python3

mkdir -p ~/esp && cd ~/esp
git clone --recursive https://github.com/espressif/esp-idf.git
cd esp-idf && git checkout v5.3
./install.sh esp32s3
. ./export.sh
```

**Windows:**
Download [ESP-IDF Windows Installer](https://dl.espressif.com/dl/esp-idf/)

### 2. Clone and Configure

```bash
git clone https://github.com/francescopace/espectre.git
cd espectre

# Set target (IMPORTANT)
idf.py set-target esp32s3

# Configure Wi-Fi and MQTT
idf.py menuconfig
```

In menuconfig:
- Go to **ESPectre Configuration**
- Set Wi-Fi SSID and password
- Set MQTT Broker URI (e.g., `mqtt://homeassistant.local:1883`)
- Set MQTT Topic (e.g., `home/espectre/node1`)
- Save with `S`, quit with `Q`

### 3. Build and Flash

```bash
idf.py build

# Linux
idf.py -p /dev/ttyUSB0 flash monitor

# macOS (find port with: ls /dev/cu.*)
idf.py -p /dev/cu.usbmodem* flash monitor
```

**If flash fails:** Hold BOOT button, press RESET, release BOOT, then run flash command.

**Exit monitor:** Ctrl+]

---

## Home Assistant

### MQTT Sensor

Add to `configuration.yaml`:

```yaml
mqtt:
  sensor:
    - name: "Movement Sensor"
      state_topic: "home/espectre/node1"
      unit_of_measurement: "intensity"
      icon: mdi:motion-sensor
      value_template: "{{ value_json.movement }}"
      json_attributes_topic: "home/espectre/node1"
      json_attributes_template: "{{ value_json | tojson }}"
```

### Automation Example

```yaml
automation:
  - alias: "Movement Alert"
    trigger:
      - platform: numeric_state
        entity_id: sensor.movement_sensor
        above: 0.6
    condition:
      - condition: template
        value_template: "{{ state_attr('sensor.movement_sensor', 'confidence') > 0.7 }}"
    action:
      - service: notify.mobile_app
        data:
          message: "Movement detected!"
```

---

## MQTT Messages

**Message format:**
```json
{
  "movement": 0.75,
  "confidence": 0.85,
  "state": "detected",
  "threshold": 0.40,
  "timestamp": 1730066405
}
```

**Fields:**
- `movement`: 0.0-1.0 (intensity)
- `confidence`: 0.0-1.0 (detection confidence)
- `state`: "idle" or "detected"
- `threshold`: current detection threshold
- `timestamp`: Unix timestamp

---

## Testing

### Verify MQTT

```bash
mosquitto_sub -h homeassistant.local -t "home/espectre/node1" -v
```

### Test Detection

1. Move around - `movement` should increase
2. Stand still - `state` should return to "idle"

---

## Calibration & Tuning

After installation, follow the **[Calibration & Tuning Guide](CALIBRATION.md)** to:
- Calibrate the sensor for your environment
- Optimize detection parameters
- Troubleshoot common issues
- Configure advanced features

### MQTT Commands Reference

All commands are sent to topic: `<your_topic>/cmd` (e.g., `home/espectre/kitchen/cmd`)  
Responses are published to: `<your_topic>/response`

**Command format:**
```json
{
  "cmd": "command_name",
  "value": value_or_parameter
}
```

**Available commands:**

| Command | Parameter | Description | Example |
|---------|-----------|-------------|---------|
| `threshold` | float (0.0-1.0) | Set detection threshold | `{"cmd": "threshold", "value": 0.40}` |
| `debounce` | int (1-10) | Set consecutive detections needed | `{"cmd": "debounce", "value": 3}` |
| `persistence` | int (1-30) | Timeout in seconds before downgrading state | `{"cmd": "persistence", "value": 3}` |
| `hysteresis` | float (0.1-1.0) | Ratio for threshold hysteresis | `{"cmd": "hysteresis", "value": 0.7}` |
| `variance_scale` | float (100-2000) | Variance normalization scale | `{"cmd": "variance_scale", "value": 400}` |
| `granular_states` | bool | Enable 4-state detection (IDLE/MICRO/DETECTED/INTENSE) | `{"cmd": "granular_states", "enabled": true}` |
| `hampel_filter` | bool | Enable/disable Hampel outlier filter | `{"cmd": "hampel_filter", "enabled": true}` |
| `hampel_threshold` | float (1.0-10.0) | Hampel filter sensitivity | `{"cmd": "hampel_threshold", "value": 2.0}` |
| `savgol_filter` | bool | Enable/disable Savitzky-Golay smoothing | `{"cmd": "savgol_filter", "enabled": true}` |
| `info` | none | Get current configuration | `{"cmd": "info"}` |
| `stats` | none | Get detection statistics | `{"cmd": "stats"}` |
| `analyze` | none | Analyze data and get recommended threshold | `{"cmd": "analyze"}` |
| `features` | none | Get current CSI feature values | `{"cmd": "features"}` |
| `weights` | none | Get current feature weights | `{"cmd": "weights"}` |
| `filters` | none | Get filter status and parameters | `{"cmd": "filters"}` |
| `logs` | bool | Enable/disable CSI logging | `{"cmd": "logs", "enabled": true}` |

**Advanced weight tuning:**
```json
{"cmd": "weight_variance", "value": 0.25}
{"cmd": "weight_spatial_gradient", "value": 0.25}
{"cmd": "weight_variance_short", "value": 0.35}
{"cmd": "weight_iqr", "value": 0.15}
```

**Example using mosquitto_pub:**
```bash
# Set threshold
mosquitto_pub -h homeassistant.local -t "home/espectre/kitchen/cmd" \
  -m '{"cmd": "threshold", "value": 0.35}'

# Get statistics
mosquitto_pub -h homeassistant.local -t "home/espectre/kitchen/cmd" \
  -m '{"cmd": "stats"}'

# Listen for response
mosquitto_sub -h homeassistant.local -t "home/espectre/kitchen/response"
```

---

## Troubleshooting

For detailed troubleshooting, see the **[Calibration & Tuning Guide](CALIBRATION.md#-troubleshooting-scenarios)**.

### Quick Setup Issues

#### Wi-Fi Connection Failed

**Check serial monitor:**
```bash
idf.py monitor
```

Look for connection errors and verify:
- ✅ Correct SSID and password in menuconfig
- ✅ Router is broadcasting 2.4GHz network
- ✅ ESP32 is within range of router

#### MQTT Not Publishing

**Verify MQTT broker is accessible:**
```bash
mosquitto_sub -h homeassistant.local -t "home/espectre/node1" -v
```

If no messages appear:
- ✅ Check MQTT broker URI in menuconfig
- ✅ Verify MQTT credentials (username/password)
- ✅ Ensure port 1883 is open (or 8883 for TLS)
- ✅ Check serial monitor for MQTT connection errors

#### Flash Failed

**Solution:**
1. Hold BOOT button on ESP32
2. Press RESET button
3. Release BOOT button
4. Run flash command again

### Getting Help

For detection issues, calibration problems, or advanced troubleshooting:
- 📖 **See**: [Calibration & Tuning Guide](CALIBRATION.md)
- 📝 **GitHub Issues**: [Report problems](https://github.com/francescopace/espectre/issues)
- 📧 **Email**: francesco.pace@gmail.com
