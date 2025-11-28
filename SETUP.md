# 🛜 ESPectre 👻 - Setup Guide

---

## What You Need

---

**Hardware:**
- **ESP32-S3**: Dual-core, 16MB Flash, 8MB PSRAM, better CPU performance
- **ESP32-C6**: Single-core, 4MB Flash, WiFi 6, higher CSI packet rate
- USB-C or Micro-USB cable (depending on board)
- Wi-Fi router (2.4 GHz, ESP32-C6 supports WiFi 6 on 2.4 GHz)

**Software:**
- ESP-IDF v6.1
- MQTT Broker (Mosquitto or Home Assistant)

---

## Installation

---

### 1. Install ESP-IDF

**macOS (tested):**

> **Note:** Tested on MacBook Air M2 with ESP-IDF v6.1-dev, but should also work with the latest stable version v5.5.1

```bash
brew install cmake ninja dfu-util python3

git clone --recursive https://github.com/espressif/esp-idf.git
cd esp-idf && git checkout v6.1-dev
./install.sh 
. ./export.sh
```

**Linux & Windows:**

For Linux and Windows installation instructions, please refer to the official Espressif documentation:
- 📖 [ESP-IDF Getting Started Guide for ESP32-S3](https://docs.espressif.com/projects/esp-idf/en/latest/esp32s3/get-started/index.html)
- 📖 [ESP-IDF Getting Started Guide for ESP32-C6](https://docs.espressif.com/projects/esp-idf/en/latest/esp32c6/get-started/index.html)

Make sure to:
- Install ESP-IDF v6.1-dev: `git checkout v6.1-dev`
- Run `./install.sh` (Linux) or `install.bat` (Windows)
- Source the environment: `. ./export.sh` (Linux) or `export.bat` (Windows)

### 2. Clone and Configure

```bash
git clone https://github.com/francescopace/espectre.git
cd espectre

# IMPORTANT: Copy the correct configuration for your ESP32 chip
# This ensures all critical settings are properly applied

# Clean any previous build and configuration
idf.py fullclean
rm -f sdkconfig

# For ESP32-S3:
cp sdkconfig.defaults.esp32s3 sdkconfig.defaults

# For ESP32-C6:
cp sdkconfig.defaults.esp32c6 sdkconfig.defaults

# Configure Wi-Fi and MQTT
idf.py menuconfig
```

**Note:** By copying the target-specific file to `sdkconfig.defaults`, you ensure that all critical configurations (like `CONFIG_ESP_WIFI_CSI_ENABLED` and `CONFIG_SPIRAM`) are properly applied during the build process. The target will be automatically detected from the configuration file.

In menuconfig:
- Go to **ESPectre Configuration**
- Set Wi-Fi SSID and password
- Set MQTT Broker URI (e.g., `mqtt://homeassistant.local:1883`)
- Set MQTT Topic (e.g., `home/espectre/node1`)
- Save with `S`, quit with `Q`

### 3. Build and Flash

```bash
# Build the project
idf.py build

# Flash to device (auto-detects port)
idf.py flash

# Or specify port manually:
# macOS: idf.py -p /dev/cu.usbmodem* flash
# Linux: idf.py -p /dev/ttyUSB0 flash (or /dev/ttyACM0)
# Windows: idf.py -p COM3 flash (check Device Manager for correct COM port)

# Monitor serial output (optional but recommended)
idf.py monitor
```

**If flash fails:** Hold BOOT button, press RESET, release BOOT, then run flash command.

**Exit monitor:** Ctrl+]

---

## Home Assistant

---

ESPectre can be integrated into Home Assistant in two ways, depending on your use case:

### Option 1: Binary Sensor (Motion Detection) - Recommended for Automations

Use this configuration if you want a simple **motion detected / no motion** sensor for automations.

Add to `configuration.yaml`:

```yaml
mqtt:
  binary_sensor:
    - name: "Movement Sensor"
      state_topic: "home/espectre/node1"
      value_template: "{{ value_json.state }}"
      payload_on: "motion"
      payload_off: "idle"
      device_class: motion
      json_attributes_topic: "home/espectre/node1"
      json_attributes_template: "{{ value_json | tojson }}"
```

**Displays:**
- 🟢 **ON** when motion is detected
- ⚫ **OFF** when idle
- All metrics available as attributes (movement, threshold, features)

**Best for:**
- ✅ Motion-triggered automations
- ✅ Security alerts
- ✅ Presence detection
- ✅ Simple on/off logic

**Automation Example:**

```yaml
automation:
  - alias: "Movement Detection Alert"
    trigger:
      - platform: state
        entity_id: binary_sensor.movement_sensor
        to: "on"
    action:
      - service: notify.mobile_app
        data:
          message: "Movement detected in monitored area"
          
  - alias: "Inactivity Alert"
    trigger:
      - platform: state
        entity_id: binary_sensor.movement_sensor
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

### Option 2: Numeric Sensor (Movement Intensity)

Use this configuration if you want to track the **movement intensity value** for analysis and graphing.

Add to `configuration.yaml`:

```yaml
mqtt:
  sensor:
    - name: "Movement Intensity"
      state_topic: "home/espectre/node1"
      unit_of_measurement: "intensity"
      icon: mdi:motion-sensor
      value_template: "{{ value_json.movement }}"
      json_attributes_topic: "home/espectre/node1"
      json_attributes_template: "{{ value_json | tojson }}"
```

**Displays:**
- Numeric value (e.g., 2.87) representing movement intensity
- All other metrics available as attributes

**Best for:**
- ✅ Historical graphs and trends
- ✅ Advanced analytics
- ✅ Threshold-based automations
- ✅ Comparing movement levels over time

**Automation Example:**

```yaml
automation:
  - alias: "High Movement Alert"
    trigger:
      - platform: numeric_state
        entity_id: sensor.movement_intensity
        above: 5.0
    action:
      - service: notify.mobile_app
        data:
          message: "High movement intensity detected: {{ states('sensor.movement_intensity') }}"
          
  - alias: "Movement Trend Analysis"
    trigger:
      - platform: time_pattern
        minutes: "/30"
    action:
      - service: notify.mobile_app
        data:
          message: "Average movement in last 30min: {{ state_attr('sensor.movement_intensity', 'movement') }}"
```

### Which Configuration Should I Use?

| Feature | Binary Sensor | Numeric Sensor |
|---------|--------------|----------------|
| Simple motion detection | ✅ Best | ⚠️ Requires threshold |
| Historical graphs | ⚠️ On/Off only | ✅ Best |
| Automations | ✅ Easy | ✅ Flexible |
| Visual clarity | ✅ Clear on/off | ⚠️ Needs interpretation |
| Use case | Security, presence | Analysis, tuning |

**💡 Tip:** You can configure **both** sensors simultaneously using different names (e.g., "Movement Sensor" and "Movement Intensity") to get the benefits of both approaches!

**Example of both configurations:**

```yaml
mqtt:
  binary_sensor:
    - name: "Movement Sensor"
      state_topic: "home/espectre/node1"
      value_template: "{{ value_json.state }}"
      payload_on: "motion"
      payload_off: "idle"
      device_class: motion
      json_attributes_topic: "home/espectre/node1"
      json_attributes_template: "{{ value_json | tojson }}"
  
  sensor:
    - name: "Movement Intensity"
      state_topic: "home/espectre/node1"
      unit_of_measurement: "intensity"
      icon: mdi:motion-sensor
      value_template: "{{ value_json.movement }}"
      json_attributes_topic: "home/espectre/node1"
      json_attributes_template: "{{ value_json | tojson }}"
```
---

## MQTT Messages

---

**Message format:**
```json
{
  "movement": 2.87,
  "threshold": 2.20,
  "state": "motion",
  "packets_processed": 15234,
  "packets_dropped": 0,
  "features": {
    "variance": 0.45,
    "skewness": 0.12,
    "kurtosis": 2.34,
    "entropy": 3.21,
    "iqr": 0.67,
    "spatial_variance": 0.89,
    "spatial_correlation": 0.76,
    "spatial_gradient": 1.23,
    "temporal_delta_mean": 0.34,
    "temporal_delta_variance": 0.56
  },
  "timestamp": 1730066405
}
```

**Fields:**
- `movement`: Moving variance value (float, typically 0.0-10.0) - indicates motion intensity
- `threshold`: Adaptive threshold value (float) - current detection threshold
- `state`: Current segmentation state - `"idle"` or `"motion"`
- `packets_processed`: CSI packets processed since last publish (integer)
- `features`: Object containing 10 extracted features (present when features extraction is enabled):
  - `variance`: Signal variance
  - `skewness`: Distribution asymmetry
  - `kurtosis`: Distribution tailedness
  - `entropy`: Signal randomness
  - `iqr`: Interquartile range
  - `spatial_variance`: Variance across subcarriers
  - `spatial_correlation`: Correlation between adjacent subcarriers
  - `spatial_gradient`: Rate of change across subcarriers
  - `temporal_delta_mean`: Average change from previous packet
  - `temporal_delta_variance`: Variance of changes from previous packet
- `timestamp`: Unix timestamp (seconds since epoch)

**Note:** The `features` object is included when feature extraction is enabled (default: enabled). Features are extracted continuously from filtered CSI data and published in every MQTT message.

---

## Monitoring & Configuration Tools

---

ESPectre provides two tools for monitoring and configuration:

### 🌐 Web-Based Monitor

**`espectre-monitor.html`** - Modern web interface with visual controls and real-time charts.

**Features:**
- ✅ Real-time visualization with interactive charts
- ✅ Live metrics dashboard (state, movement, threshold, segments)
- ✅ Visual configuration of all parameters (sliders, toggles)
- ✅ Device IP display
- ✅ Statistics viewer with modal popup
- ✅ Auto-sync configuration on connection
- ✅ No terminal required - works in any browser
- ✅ Cross-platform (desktop, tablet, mobile)

**How to use:**
1. Open `espectre-monitor.html` in your browser
2. Configure MQTT connection (broker address, port, topic, credentials)
3. Click "Connect"
4. Monitor real-time data and configure parameters visually

**Perfect for:**
- First-time users
- Visual parameter tuning
- Real-time monitoring
- Multi-device management

### 🖥️ CLI Tool

**`espectre-cli.py`** - Interactive command-line interface for advanced users and scripting.

**Features:**
- ✅ Interactive terminal session
- ✅ Real-time command feedback
- ✅ Scriptable for automation
- ✅ Lightweight and fast

**Quick start:**
```bash
./espectre-cli.py

---

## Calibration & Tuning

---

After installation, follow the **[CALIBRATION.md](CALIBRATION.md)** to:
- Calibrate the sensor for your environment
- Optimize detection parameters
- Troubleshoot common issues
- Configure advanced features

**Tip:** Use the web monitor (`espectre-monitor.html`) for easier visual tuning, or the CLI tool for advanced control.

---

### MQTT Commands Reference

---

Both the web monitor and CLI tool use MQTT commands under the hood. You can also send commands directly via MQTT for scripting and automation.

**Quick Start with CLI:**
```bash
# Launch interactive mode
./espectre-cli.py

# In the interactive session, type commands directly:
espectre> info          # Get current configuration
espectre> stats         # Show statistics
espectre> threshold 0.4 # Set detection threshold
espectre> help          # Show all commands
espectre> exit          # Exit CLI
```


#### Direct MQTT Commands (for scripting/automation)

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

| Area | Command | Parameter | Description | Example |
|------|---------|-----------|-------------|---------|
| **System** | `info` | none | Get current configuration (network, MQTT topics, filters, segmentation params, options) | `{"cmd": "info"}` |
| **System** | `stats` | none | Get runtime statistics (state, turbulence, variance, packets, segments, uptime) | `{"cmd": "stats"}` |
| **System** | `traffic_generator_rate` | int (0-1000) | Set WiFi traffic rate for continuous CSI (0=disabled, recommended: 100 pps) | `{"cmd": "traffic_generator_rate", "value": 100}` |
| **System** | `smart_publishing` | bool | Enable/disable smart publishing (reduces MQTT traffic) | `{"cmd": "smart_publishing", "enabled": true}` |
| **System** | `factory_reset` | none | Restore all settings to factory defaults | `{"cmd": "factory_reset"}` |
| **Segmentation** | `segmentation_threshold` | float (0.5-10.0) | Set segmentation threshold for motion detection | `{"cmd": "segmentation_threshold", "value": 2.2}` |
| **Segmentation** | `segmentation_window_size` | int (10-200) | Set moving variance window size in packets | `{"cmd": "segmentation_window_size", "value": 100}` |
| **Segmentation** | `subcarrier_selection` | array of int (0-63) | Set selected subcarriers for CSI processing (1-64 subcarriers) | `{"cmd": "subcarrier_selection", "indices": [47,48,49,50,51,52,53,54]}` |
| **Features** | `features_enable` | bool | Enable/disable feature extraction | `{"cmd": "features_enable", "enabled": true}` |
| **Features** | `butterworth_filter` | bool | Enable/disable Butterworth low-pass filter (8Hz cutoff) | `{"cmd": "butterworth_filter", "enabled": true}` |
| **Features** | `wavelet_filter` | bool | Enable/disable Wavelet db4 filter (low-freq noise) | `{"cmd": "wavelet_filter", "enabled": true}` |
| **Features** | `wavelet_level` | int (1-3) | Wavelet decomposition level (3=max denoising) | `{"cmd": "wavelet_level", "value": 3}` |
| **Features** | `wavelet_threshold` | float (0.5-2.0) | Wavelet noise threshold (1.0=balanced) | `{"cmd": "wavelet_threshold", "value": 1.0}` |
| **Features** | `hampel_filter` | bool | Enable/disable Hampel outlier filter | `{"cmd": "hampel_filter", "enabled": true}` |
| **Features** | `hampel_threshold` | float (1.0-10.0) | Hampel filter sensitivity | `{"cmd": "hampel_threshold", "value": 2.0}` |
| **Features** | `savgol_filter` | bool | Enable/disable Savitzky-Golay smoothing | `{"cmd": "savgol_filter", "enabled": true}` |

#### Info Command Response Structure

The `info` command returns **static configuration** organized into logical groups:

```json
{
  "network": {
    "ip_address": "192.168.1.100",
    "mac_address": "AA:BB:CC:DD:EE:FF",
    "traffic_generator_rate": 100,
    "channel": {
      "primary": 6,
      "secondary": 0
    },
    "bandwidth": "HT20",
    "protocol": "802.11b/g/n",
    "promiscuous_mode": false
  },
  "device": {
    "type": "ESP32-C6"
  },
  "mqtt": {
    "base_topic": "home/espectre/node1",
    "cmd_topic": "home/espectre/node1/cmd",
    "response_topic": "home/espectre/node1/response"
  },
  "segmentation": {
    "threshold": 2.2,
    "window_size": 100
  },
  "filters": {
    "butterworth_enabled": true,
    "wavelet_enabled": false,
    "wavelet_level": 3,
    "wavelet_threshold": 1.0,
    "hampel_enabled": false,
    "hampel_threshold": 2.0,
    "savgol_enabled": true,
    "savgol_window_size": 5
  },
  "options": {
    "features_enabled": true,
    "smart_publishing_enabled": false
  },
  "subcarriers": {
    "indices": [47, 48, 49, 50, 51, 52, 53, 54],
    "count": 8
  }
}
```

**Groups:**
- **`network`**: Network information (IP address, MAC address, WiFi channel, bandwidth, protocol, traffic generator rate)
- **`device`**: Device type (ESP32-C6, ESP32-S3, or ESP32)
- **`mqtt`**: MQTT topic configuration
- **`segmentation`**: Motion segmentation configuration parameters (threshold, window size)
- **`filters`**: Signal processing filters configuration
- **`options`**: General capabilities and features
- **`subcarriers`**: Selected subcarriers for CSI processing (configurable at runtime)

#### Stats Command Response Structure

The `stats` command returns **runtime metrics** for monitoring:

```json
{
  "timestamp": 1730066405,
  "uptime": "3h 24m 15s",
  "cpu_usage_percent": 5.4,
  "heap_usage_percent": 22.3,
  "state": "motion",
  "turbulence": 3.45,
  "movement": 2.87,
  "threshold": 2.20,
  "packets_processed": 15234,
  "packets_dropped": 0,
}
```

**Fields:**
- **`timestamp`**: Unix timestamp when stats were generated
- **`uptime`**: System uptime in human-readable format
- **`cpu_usage_percent`**: CPU usage percentage (0-100), calculated using FreeRTOS runtime statistics
- **`heap_usage_percent`**: Heap memory usage percentage (0-100), calculated as (used/total)*100
- **`state`**: Current segmentation state (idle/motion)
- **`turbulence`**: Last spatial turbulence value (for diagnostics)
- **`movement`**: Current moving variance (same as in periodic data)
- **`threshold`**: Current adaptive threshold (same as in periodic data)
- **`packets_processed`**: Total CSI packets processed

**System Resource Monitoring:**
- CPU and heap monitoring provide real-time visibility into system health
- Useful for detecting performance issues and memory leaks

### Factory Reset

Restore all settings to factory defaults and clear all saved data from NVS:

**Via MQTT:**
```bash
mosquitto_pub -h homeassistant.local -t "home/espectre/node1/cmd" \
  -m '{"cmd":"factory_reset"}'
```

**This will:**
- ⚠️ Clear all saved configuration parameters from NVS
- ⚠️ Restore all parameters to factory defaults (filters, segmentation threshold)
- ✅ Trigger NBVI automatic subcarrier calibration

**Note on Calibration:**
After factory reset, the system will automatically run NBVI (Normalized Baseline Variability Index) calibration at next boot to select optimal subcarriers. This process takes ~5 seconds and requires a quiet baseline period. For details on NBVI algorithm, see [micro-espectre/tools/README.md](micro-espectre/tools/README.md).

**When to use:**
- Configuration is corrupted or inconsistent
- Want to start fresh with default settings
- Testing different configurations
- Troubleshooting persistent issues

**Example using mosquitto_pub:**
```bash
# Set segmentation threshold
mosquitto_pub -h homeassistant.local -t "home/espectre/kitchen/cmd" \
  -m '{"cmd": "segmentation_threshold", "value": 0.35}'

# Get info
mosquitto_pub -h homeassistant.local -t "home/espectre/kitchen/cmd" \
  -m '{"cmd": "info"}'

# Listen for response
mosquitto_sub -h homeassistant.local -t "home/espectre/kitchen/response"
```

---

### Traffic Generator

---

**⚠️ IMPORTANT:** ESPectre requires continuous WiFi traffic to receive CSI packets. Without traffic, the ESP32 receives few/no CSI packets, resulting in poor detection.

**What it does:**
- Generates UDP broadcast packets at configurable rate (default: 15 packets/sec)
- Ensures continuous CSI data availability
- Essential for reliable detection

**Why it's needed:**
- ESP32 only receives CSI when there's WiFi traffic

**Configuration:**
```bash
# Via CLI
traffic_generator_rate 100  # Enable 100 pps

# Via MQTT
{"cmd":"traffic_generator_rate","value":100}
```

**Recommended rates:**
- **100 pps**: Activity recognition (default) - walking, sitting, gestures
- **600-1000 pps**: Fast motion detection - precision localization, rapid movements
- **50 pps**: Minimal overhead - basic presence detection
- **0 pps**: Disabled (only if you have other continuous WiFi traffic)

**Troubleshooting:**

Verify traffic generator is working:
```bash
# Or use tcpdump for detailed analysis
sudo tcpdump -i en0 -n udp port 12345
```

If no packets appear:
- ✅ Check ESP32 serial monitor for traffic generator errors
- ✅ Verify ESP32 and Mac are on same network
- ✅ Try increasing rate: `traffic_generator_rate 100`

**Enable via MQTT:**
```bash
# Enable wavelet filter
mosquitto_pub -h localhost -t "espectre/cmd" -m '{"cmd":"wavelet_filter","enabled":true}'

# Set decomposition level (1-3, recommended: 3)
mosquitto_pub -h localhost -t "espectre/cmd" -m '{"cmd":"wavelet_level","value":3}'

# Set threshold (0.5-2.0, recommended: 1.0)
mosquitto_pub -h localhost -t "espectre/cmd" -m '{"cmd":"wavelet_threshold","value":1.0}'
```

**Or use the interactive CLI:**
```bash
./espectre-cli.py
> wv on
> wvl 3
> wvt 1.0
```

**Performance impact:**
- CPU: ~5-8% additional load
- RAM: ~2 KB
- Latency: 320ms warm-up (32 samples)

---

### Smart Publishing

---

**How it works:**
- Publishes immediately when detection state changes (idle ↔ motion)
- Publishes when movement score changes significantly
- Publishes a heartbeat every 5 seconds even if nothing changed
- Skips redundant messages when values are stable

**Benefits:**
- 📉 Reduces MQTT bandwidth by 50-70% during stable periods
- 🔋 Lower network overhead and power consumption
- 📊 Still provides timely updates for state changes
- 💓 Regular heartbeat ensures system is alive

**Default:** Disabled (publishes every detection cycle)

**Enable/disable via MQTT:**
```bash
mosquitto_pub -h homeassistant.local -t "home/espectre/node1/cmd" \
  -m '{"cmd":"smart_publishing","enabled":true}'
```

**When to use:**
- ✅ High-traffic MQTT brokers
- ✅ Battery-powered or low-bandwidth scenarios
- ✅ Multiple ESPectre sensors on same network
- ✅ Home Assistant with many sensors

**When to disable:**
- ❌ Need every single data point for analysis
- ❌ Real-time graphing/monitoring
- ❌ Custom integrations expecting constant updates

---

## Troubleshooting

---

### Wi-Fi Connection Failed

**Check serial monitor:**
```bash
idf.py monitor
```

Look for connection errors and verify:
- ✅ Correct SSID and password in menuconfig
- ✅ Router is broadcasting 2.4GHz network
- ✅ ESP32 is within range of router

### MQTT Not Publishing

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

---

### Getting Help

---

For detection issues or parameter tuning:
- 📖 **See**: [CALIBRATION.md](CALIBRATION.md)
- � **GitHub Issues**: [Report problems](https://github.com/francescopace/espectre/issues)
- 📧 **Email**: francesco.pace@gmail.com
