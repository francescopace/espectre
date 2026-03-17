# Quick Start Guide - ESPectre Arduino

Get your CSI motion detector running in **5 minutes**

## Prerequisites

- ✅ Adafruit Feather ESP32-S3 Reverse TFT
- ✅ USB-C cable
- ✅ Arduino IDE 2.x installed
- ✅ 2.4 GHz WiFi network available

## Step 1: Install Arduino ESP32 (2 minutes)

1. Open Arduino IDE
2. Go to **File → Preferences**
3. Add this URL to "Additional Board Manager URLs":
   ```
   https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
   ```
4. Click **OK**
5. Go to **Tools → Board → Boards Manager**
6. Search for **"esp32"**
7. Install **"esp32 by Espressif Systems"** (version 2.0.14 or newer)

## Step 2: Install Libraries (1 minute)

1. Go to **Tools → Manage Libraries**
2. Install these 3 libraries:
   - Search **"Adafruit ST7789"** → Install
   - Search **"Adafruit GFX"** → Install
   - Search **"Adafruit NeoPixel"** → Install

## Step 3: Configure WiFi (30 seconds)

1. Open `config.h` in the Arduino IDE
2. Change these two lines:
   ```cpp
   #define WIFI_SSID "YourNetworkName"
   #define WIFI_PASSWORD "YourPassword"
   ```
3. Save the file

## Step 4: Select Board (30 seconds)

In Arduino IDE, go to **Tools** and set:

| Setting | Value |
|---------|-------|
| **Board** | "Adafruit Feather ESP32-S3 Reverse TFT" |
| **USB CDC On Boot** | "Enabled" |
| **PSRAM** | "OPI PSRAM" |
| **Flash Mode** | "QIO 80MHz" |
| **Upload Speed** | "921600" |
| **Port** | Select your USB port (e.g., /dev/cu.usbmodem14201) |

## Step 5: Upload (1 minute)

1. Connect your Feather to USB
2. Click **Upload** button (→)
3. Wait for "Done uploading"
4. Open **Serial Monitor** (magnifying glass icon)
5. Set baud rate to **115200**

## Step 6: Watch It Work!

You should see:

**Serial Monitor:**
```
ESPectre Arduino - Starting...
WiFi connected!
Gain locked: AGC=64, FFT=32
Calibration complete: 700 samples
Selected band: 11 14 18 22 26 30 34 38 42 46 50 54
Adaptive threshold: 0.987
Starting motion detection...

--- Idle | Var: 0.123 | Thr: 0.987 | Pkts: 52
```

**TFT Display:**
```
┌─────────────────┐
│  Idle           │ ← Green when idle
│                 │
│  Variance: 0.12 │
│  Threshold: 0.99│
│  Packets: 52    │
└─────────────────┘
```

**NeoPixel LED:** 🟢 Green (idle)

**Walk around the room** → Display shows **"MOTION"** in red! 🔴

## Troubleshooting

### "WiFi FAILED!" on display
- ❌ Wrong SSID or password
- ❌ 5 GHz network (ESP32 only supports 2.4 GHz)
- ✅ Fix: Check `config.h` and ensure 2.4 GHz network

### Blank display
- ❌ Board not selected correctly
- ✅ Fix: Verify board selection in Step 4
- ✅ Try: Open and run `HARDWARE_TEST/HARDWARE_TEST.ino` to diagnose

### "CSI FAILED!" error
- ❌ Old Arduino-ESP32 version
- ✅ Fix: Update to version 2.0.14 or newer
- ✅ Check: Tools → Boards Manager → esp32

### No motion detected
- ❌ Too far from router (>10m)
- ❌ Too close to router (<2m)
- ✅ Optimal range: **3-8 meters**
- ✅ Try: Move closer to WiFi router

### Compile errors
- ❌ Missing libraries
- ✅ Fix: Re-install libraries from Step 2
- ❌ Wrong board selected
- ✅ Fix: Double-check board selection

## Next Steps

✅ **Calibrate properly**: Keep room still during boot (first 15 seconds)

✅ **Experiment with placement**: Try different distances from router (3-8m optimal)

✅ **Check accuracy**: Walk around, verify motion is detected within 1-2 seconds

✅ **Monitor performance**: Watch Serial Monitor for variance and threshold values

✅ **Read full docs**: See `README.md` for advanced configuration

## Need Help?

- 📖 Full documentation: `README.md`
- 🔧 Hardware test: Open `HARDWARE_TEST/HARDWARE_TEST.ino`
- 💬 Open an issue: [GitHub Issues](https://github.com/paulhey/espectre/issues)
- 📚 ESPectre docs: [espectre.com](https://espectre.com)

---

**Enjoy your CSI motion detector 🎉**
