# Hardware Test Sketch

This is a diagnostic tool to verify all hardware components work correctly before running the main ESPectre sketch.

## How to Use

1. **Configure WiFi**:
   - Edit `config.h` in this folder
   - Set your WiFi SSID and password

2. **Open in Arduino IDE**:
   - Open `HARDWARE_TEST.ino` (this file)
   - **Important**: Do NOT open `arduino_espectre.ino` at the same time

3. **Select Board**:
   - Board: "Adafruit Feather ESP32-S3 Reverse TFT"
   - USB CDC On Boot: "Enabled"
   - Port: Select your USB port

4. **Upload**:
   - Click Upload button
   - Open Serial Monitor (115200 baud)

5. **Watch Tests Run**:
   - The sketch tests 5 components automatically
   - Results shown on TFT display and Serial Monitor

## Tests Performed

1. ✓ **Serial Communication** (115200 baud)
2. ✓ **TFT Display** (color test, text rendering)
3. ✓ **NeoPixel LED** (RGB color test)
4. ✓ **WiFi Connection** (SSID, IP, RSSI, channel)
5. ✓ **CSI Capability** (ESP-IDF functions)

## Expected Output

### Serial Monitor
```
========================================
ESPectre Arduino - Hardware Test
========================================

✓ Test 1: Serial communication OK
✓ Test 2: TFT Display OK
✓ Test 3: NeoPixel LED OK
✓ Test 4: WiFi Connected!
  - IP Address: 192.168.1.100
  - RSSI: -45 dBm
  - Channel: 6
✓ Test 5: CSI Enabled Successfully!

========================================
Test Summary:
========================================
Serial:    ✓ PASS
TFT:       ✓ PASS
NeoPixel:  ✓ PASS
WiFi:      ✓ PASS
CSI:       ✓ PASS
========================================

Result: 5/5 tests passed

✓ ALL TESTS PASSED!
Hardware is ready for ESPectre operation.
```

### TFT Display
```
┌────────────────┐
│   ALL OK!      │
│                │
│ 5/5 tests pass │
│ Ready for      │
│ ESPectre!      │
└────────────────┘
```

### LED
- 🔴 Red → 🟢 Green → 🔵 Blue → ⚪ White (during test)
- 🟢 Green blinking (if all pass)
- 🟡 Yellow (if some tests fail)

## Troubleshooting

### WiFi test fails
- Check SSID/password in `config.h`
- Ensure 2.4 GHz network (ESP32 doesn't support 5 GHz)
- Move closer to router

### CSI test fails
- Update Arduino-ESP32 to v2.0.14 or newer
- Check board selection
- Restart device and try again

### Display blank
- Check board selection (must be ESP32-S3 Feather)
- Verify TFT pins match your board
- Try different USB cable

## After Testing

If all tests pass:
1. ✅ Close this sketch
2. ✅ Open `arduino_espectre/arduino_espectre.ino`
3. ✅ Upload the main ESPectre sketch
4. ✅ Enjoy CSI motion detection!

---

**Note**: This is a separate sketch from the main `arduino_espectre.ino`. They cannot be opened simultaneously in Arduino IDE.
