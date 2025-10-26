# 🛜 ESPectre 👻 - Setup Guide

Complete setup guide for **ESPectre** on ESP32-S3-DevKitC-1.

---

## 📋 Prerequisites

### Hardware
- ✅ **ESP32-S3-DevKitC-1 N16R8** (16MB Flash, 8MB PSRAM)
- ✅ USB-C cable for programming and power
- ✅ Wi-Fi router (2.4 GHz)
- ✅ (Optional) External IPEX antenna for better reception
- ✅ (Optional) Home Assistant server

### Software
- ✅ **ESP-IDF v5.x** (or use pre-compiled binary)
- ✅ Python 3.8+ (for esptool.py)
- ✅ MQTT Broker (Mosquitto / Home Assistant MQTT)

---

## 🚀 Quick Start (Pre-compiled Binary)

### Option 1: Flash Pre-compiled Binary (Easiest) ⭐

**Step 1: Install esptool**
```bash
pip install esptool
```

**Step 2: Download firmware**
```bash
# Download latest release
wget https://github.com/francescopace/espectre/releases/latest/download/espectre.bin
```

**Step 3: Flash to ESP32-S3**
```bash
# Find your USB port
# Linux: usually /dev/ttyUSB0 or /dev/ttyACM0
# macOS: usually /dev/cu.usbserial-* or /dev/cu.SLAB_USBtoUART or /dev/cu.wchusbserial*
#        (use 'ls /dev/cu.*' to list available ports)
# Windows: usually COM3, COM4, etc.

# Erase flash (recommended for first install)
# Replace /dev/ttyUSB0 with your actual port
esptool.py --chip esp32s3 --port /dev/ttyUSB0 erase_flash

# Flash firmware
esptool.py --chip esp32s3 --port /dev/ttyUSB0 --baud 460800 \
  write_flash -z 0x0 espectre.bin
```

**Step 4: Configure via Serial**

Connect to serial console to configure Wi-Fi and MQTT:
```bash
# Linux
screen /dev/ttyUSB0 115200

# macOS (replace with your actual port from 'ls /dev/cu.*')
screen /dev/cu.usbserial-* 115200

# Or use idf.py monitor (if ESP-IDF installed)
# Linux
idf.py -p /dev/ttyUSB0 monitor

# macOS
idf.py -p /dev/cu.usbserial-* monitor

# Windows: use PuTTY or similar
```

On first boot, the device will prompt for configuration. Follow the on-screen instructions to set:
- Wi-Fi SSID and password
- MQTT broker IP/hostname
- MQTT topic
- MQTT credentials (optional)

**Step 5: Verify Operation**

After configuration, you should see:
```
I (1234) ESPectre: === ESPectre ESP32-S3 Starting ===
I (1456) WiFi: Connected, got IP: 192.168.1.100
I (1678) MQTT: Connected to broker
I (1890) ESPectre: Calibrating for 60 seconds...
I (62000) CSI: Calibration complete. Baseline: 0.123
```

Done! The sensor is now publishing to MQTT.

---

## 🔧 Option 2: Build from Source

### Step 1: Install ESP-IDF

#### Linux

```bash
# Install prerequisites
sudo apt-get install git wget flex bison gperf python3 python3-pip \
  python3-venv cmake ninja-build ccache libffi-dev libssl-dev \
  dfu-util libusb-1.0-0

# Clone ESP-IDF
mkdir -p ~/esp
cd ~/esp
git clone --recursive https://github.com/espressif/esp-idf.git
cd esp-idf
git checkout v5.3  # or latest stable version

# Install tools
./install.sh esp32s3

# Setup environment (add to ~/.bashrc for permanent)
. ./export.sh
```

#### macOS

```bash
# Install prerequisites using Homebrew
brew install cmake ninja dfu-util python3

# Clone ESP-IDF
mkdir -p ~/esp
cd ~/esp
git clone --recursive https://github.com/espressif/esp-idf.git
cd esp-idf
git checkout v5.3  # or latest stable version

# Install tools
./install.sh esp32s3

# Setup environment (add to ~/.zshrc or ~/.bash_profile for permanent)
. ./export.sh
```

#### Windows

Download and run [ESP-IDF Windows Installer](https://dl.espressif.com/dl/esp-idf/)

The installer will:
- Install ESP-IDF
- Install Python and required packages
- Install toolchain
- Configure environment variables

### Step 2: Clone ESPectre

```bash
git clone https://github.com/francescopace/espectre.git
cd espectre
```

### Step 3: Set Target to ESP32-S3

**IMPORTANT:** Before building, you must set the target to ESP32-S3:

```bash
idf.py set-target esp32s3
```

This command:
- Configures the project for ESP32-S3 (not the original ESP32)
- Regenerates the `sdkconfig` file with ESP32-S3 defaults
- Must be done before running `idf.py menuconfig`

**Note:** If you skip this step, you'll get a "Failed to connect to ESP32" error when flashing, because esptool will try to connect to the wrong chip type.

### Step 4: Configure

**IMPORTANT:** You MUST configure Wi-Fi and MQTT settings before building, otherwise the device won't be able to connect.

```bash
# Open menuconfig
idf.py menuconfig
```

Navigate to **ESPectre Configuration** and set:
- **Wi-Fi SSID**: Your Wi-Fi network name (required)
- **Wi-Fi Password**: Your Wi-Fi password (required)
- **MQTT Broker URI**: `mqtt://192.168.1.100:1883` (or your broker address) (required)
- **MQTT Topic**: `home/espectre/node1` (or your preferred topic) (required)
- **MQTT Username**: (optional) MQTT username
- **MQTT Password**: (optional) MQTT password

**How to navigate menuconfig:**
1. Use arrow keys to navigate
2. Press Enter to select a menu item
3. Type your values when prompted
4. Press `S` to save
5. Press `Q` to quit

**After saving, you'll see a confirmation that the configuration was written to `sdkconfig`.**

### Step 5: Build

```bash
# Build firmware
idf.py build
```

This will compile the firmware. First build takes ~5-10 minutes.

### Step 6: Flash

**IMPORTANT:** If the flash fails with "Failed to connect" or "Invalid head of packet", you need to put the ESP32-S3 in download mode manually:

1. **Hold down the BOOT button** (labeled BOOT or GPIO0)
2. **While holding BOOT, press and release the RESET button** (labeled RST or EN)
3. **Release the BOOT button**
4. **Now run the flash command immediately**

```bash
# Flash to ESP32-S3
# Linux
idf.py -p /dev/ttyUSB0 flash

# macOS (find your port with: ls /dev/cu.*)
idf.py -p /dev/cu.usbmodem1234561 flash

# Flash and monitor in one command
# Linux
idf.py -p /dev/ttyUSB0 flash monitor

# macOS
idf.py -p /dev/cu.usbmodem1234561 flash monitor
```

**Alternative method:** Start the flash command first, then when you see "Connecting........", immediately hold the BOOT button until you see "Writing at 0x00000000...".

### Step 7: Monitor

```bash
# View serial output
# Linux
idf.py -p /dev/ttyUSB0 monitor

# macOS (replace with your actual port from 'ls /dev/cu.*')
idf.py -p /dev/cu.usbmodem1234561 monitor

# Exit monitor: Ctrl+]
```

**Tip for macOS:** To find your USB port, run `ls /dev/cu.*` and look for devices like:
- `/dev/cu.usbmodem*` (ESP32-S3 native USB)
- `/dev/cu.usbserial-*` (CH340/CH341 chips)
- `/dev/cu.SLAB_USBtoUART` (CP2102 chips)

---

## 🏠 Step 3: Configure Home Assistant

### Install Mosquitto (if not present)

If you don't have an MQTT broker:

**Home Assistant Add-on:**
1. Go to **Supervisor** → **Add-on Store**
2. Search for **Mosquitto broker**
3. Click **Install**
4. Start the add-on
5. Enable **Start on boot**

**Standalone Mosquitto:**
```bash
# Linux
sudo apt-get install mosquitto mosquitto-clients
# Start service
sudo systemctl start mosquitto

# macOS
brew install mosquitto
# Start service
brew services start mosquitto
```

### Add MQTT Sensor

Add to your `configuration.yaml`:

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

### Add Automation

Example automation in `automations.yaml`:

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
          message: "Movement detected with high confidence!"
```

Restart Home Assistant after configuration.

---

## 📡 MQTT Message Format

ESPectre publishes JSON messages with the following structure:

### Message Fields

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `movement` | float | 0.0-1.0 | Normalized movement intensity |
| `confidence` | float | 0.0-1.0 | Detection confidence level |
| `state` | string | - | Current state: `"idle"`, `"detected"`, `"calibrating"` |
| `timestamp` | int | - | Unix timestamp |
| `baseline` | float | - | Current baseline (debug) |
| `threshold` | float | - | Detection threshold (debug) |

### Example Messages

**Idle state:**
```json
{
  "movement": 0.15,
  "confidence": 0.0,
  "state": "idle",
  "baseline": 0.12,
  "threshold": 0.35,
  "timestamp": 1730066400
}
```

**Movement detected:**
```json
{
  "movement": 0.75,
  "confidence": 0.85,
  "state": "detected",
  "baseline": 0.12,
  "threshold": 0.35,
  "timestamp": 1730066405
}
```

---

## 🧪 Testing

### Test Serial Output

```bash
# Linux
idf.py -p /dev/ttyUSB0 monitor

# macOS (replace with your actual port)
idf.py -p /dev/cu.usbserial-* monitor
```

**Expected output:**
```
I (1234) ESPectre: === ESP32-S3 Starting ===
I (1456) WiFi: Connected, got IP: 192.168.1.100
I (1678) MQTT: Connected to broker
I (1890) ESPectre: Calibrating for 60 seconds...
I (62000) CSI: Calibration complete. Baseline: 0.123
I (63000) CSI: Movement: 0.65, State: detected
```

### Test MQTT Connection

Use `mosquitto_sub` to verify messages:

```bash
# Subscribe to topic
mosquitto_sub -h 192.168.1.100 -t "home/espectre/node1" -v

# With authentication
mosquitto_sub -h 192.168.1.100 -t "home/espectre/node1" -u username -P password -v
```

You should see JSON messages every second.

### Test Movement Detection

1. Keep the environment static during calibration (60 seconds)
2. After calibration, move around the sensor
3. Observe `movement` value increase
4. Check `state` changes to `"detected"`
5. Stand still and verify `state` returns to `"idle"`

---

## 🎯 Parameter Optimization

### Runtime Configuration via CLI

The system comes with **optimized default values** from empirical testing. You can fine-tune parameters at runtime using the CLI tool without recompiling:

```bash
# View current configuration
./espectre-cli.sh info

# Analyze data and get recommended threshold
./espectre-cli.sh analyze

# Set detection threshold (0.0-1.0)
./espectre-cli.sh threshold 0.50

# Set debounce count (1-10)
./espectre-cli.sh debounce 2

# Set persistence timeout (1-30 seconds)
./espectre-cli.sh persistence 5

# Set hysteresis ratio (0.1-1.0)
./espectre-cli.sh hysteresis 0.7

# Enable/disable Savitzky-Golay smoothing filter
./espectre-cli.sh savgol_filter on

# Enable/disable Hampel outlier filter
./espectre-cli.sh hampel_filter on

# View all extracted features
./espectre-cli.sh features

# View current feature weights
./espectre-cli.sh weights

# Monitor real-time data
./espectre-cli.sh monitor
```

### Default Values (Optimized)

The system starts with these optimized defaults:

```c
// Detection Parameters
#define DEFAULT_THRESHOLD   0.50f     // Optimized from empirical analysis
#define DEBOUNCE_COUNT      2         // Requires 2 consecutive detections
#define PERSISTENCE_TIMEOUT 5         // Wait 5 seconds before downgrading state
#define HYSTERESIS_RATIO    0.7f      // Prevents state flickering
#define VARIANCE_SCALE      400.0f    // Sensitivity scale

// Granular State Thresholds
#define MICRO_MOVEMENT_THRESHOLD    0.10f  // IDLE → MICRO
#define INTENSE_MOVEMENT_THRESHOLD  0.70f  // DETECTED → INTENSE

// Filters (enabled by default)
Savitzky-Golay smoothing: ENABLED
Hampel outlier removal: DISABLED (can be enabled if needed)
Adaptive normalization: ALWAYS ACTIVE
```

### Calibration Workflow

1. **Deploy sensor** in target location
2. **Wait 1-2 minutes** for data collection (stay still)
3. **Run analysis**:
   ```bash
   ./espectre-cli.sh analyze
   ```
4. **Apply recommended threshold**:
   ```bash
   ./espectre-cli.sh threshold <recommended_value>
   ```
5. **Test movement detection** and fine-tune if needed

### Advanced: Modify Source Code Defaults

If you want to change compile-time defaults, edit `main/espectre.c`:

```c
// Signal Processing Parameters
#define DEBOUNCE_COUNT      2         // 1-10: Higher = fewer false positives
#define PERSISTENCE_TIMEOUT 5         // 1-30: Detection persistence (seconds)
#define DEFAULT_THRESHOLD   0.50f     // 0.0-1.0: Detection threshold
#define HYSTERESIS_RATIO    0.7f      // 0.1-1.0: Prevents flickering
#define VARIANCE_SCALE      400.0f    // 100-2000: Lower = higher sensitivity
```

After modifying, rebuild and flash:
```bash
idf.py build flash
```

---

## � Monitoring and Debugging

### View Logs

```bash
# Real-time logs
# Linux
idf.py -p /dev/ttyUSB0 monitor

# macOS (replace with your actual port)
idf.py -p /dev/cu.usbserial-* monitor

# Filter by log level
# Linux
idf.py -p /dev/ttyUSB0 monitor | grep "ESPectre"

# macOS
idf.py -p /dev/cu.usbserial-* monitor | grep "ESPectre"
```

### System Stats

Monitor in serial output:
- Packet reception rate
- Processing success rate
- Memory usage
- Wi-Fi signal strength

### Debug Mode

Enable verbose logging in menuconfig:
```bash
idf.py menuconfig
# Component config → Log output → Default log verbosity → Debug
```

---

## 🆘 Support

For issues or questions:
1. Check **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** for detailed solutions
2. Review serial logs for error messages
3. Consult [ESP-IDF documentation](https://docs.espressif.com/projects/esp-idf/en/latest/esp32s3/)
4. Open an [Issue on GitHub](https://github.com/francescopace/espectre/issues)

---

**Happy sensing! 🛜**
