# 🛜 ESPectre 👻 - Troubleshooting Guide

Comprehensive troubleshooting guide for common issues with ESPectre on ESP32-S3.

---

## 📑 Table of Contents

1. [Flash and Upload Issues](#flash-and-upload-issues)
2. [Wi-Fi Connection Issues](#wi-fi-connection-issues)
3. [MQTT Connection Issues](#mqtt-connection-issues)
4. [CSI Acquisition Issues](#csi-acquisition-issues)
5. [Detection Issues](#detection-issues)
6. [Performance Issues](#performance-issues)
7. [Memory Issues](#memory-issues)
8. [Hardware Issues](#hardware-issues)
9. [Build and Compilation Issues](#build-and-compilation-issues)

---

## Flash and Upload Issues

### Problem: "Failed to connect to ESP32" or "Invalid head of packet"

**Symptoms**: esptool or idf.py cannot connect to the device, or shows "Invalid head of packet (0x30): Possible serial noise or corruption"

**Solutions**:

1. **Put ESP32-S3 in download mode manually**
   ```bash
   # Method 1: Using BOOT and RESET buttons
   # 1. Hold down the BOOT button (labeled BOOT or GPIO0)
   # 2. While holding BOOT, press and release the RESET button (labeled RST or EN)
   # 3. Release the BOOT button
   # 4. Now run: idf.py -p /dev/cu.usbmodem1234561 flash
   
   # Method 2: Hold BOOT during flash command
   # 1. Start the flash command: idf.py -p /dev/cu.usbmodem1234561 flash
   # 2. When you see "Connecting........", immediately hold BOOT button
   # 3. Keep holding until you see "Writing at 0x00000000..."
   # 4. Release BOOT button
   ```

2. **For macOS users with ESP32-S3 native USB**
   - The ESP32-S3-DevKitC-1 uses native USB (shows as `/dev/cu.usbmodem*`)
   - May require manual boot mode entry more often than CH340-based boards
   - Try unplugging and replugging the USB cable while holding BOOT

2. **Check USB cable**
   - Must support data transfer (not just power)
   - Try a different cable
   - Avoid USB hubs, connect directly to computer

3. **Install/update USB drivers**
   - **CH340/CH341**: [Download drivers](http://www.wch.cn/downloads/CH341SER_ZIP.html)
   - **CP2102**: [Download drivers](https://www.silabs.com/developers/usb-to-uart-bridge-vcp-drivers)
   - Linux: Usually works out of the box
   - macOS: 
     - May need to allow in System Preferences → Security & Privacy
     - For CH340 chips: `brew install --cask wch-ch34x-usb-serial-driver`
     - Find your port with: `ls /dev/cu.*`

4. **Check device permissions (Linux)**
   ```bash
   # Add user to dialout group
   sudo usermod -a -G dialout $USER
   
   # Logout and login again
   # Or use sudo for flash commands
   ```

5. **Try different USB port**
   - USB 2.0 ports often work better than USB 3.0
   - Avoid front panel USB ports

6. **Verify device is detected**
   ```bash
   # Linux
   ls /dev/ttyUSB* /dev/ttyACM*
   dmesg | tail
   
   # macOS
   ls /dev/cu.*
   
   # Windows
   # Check Device Manager → Ports (COM & LPT)
   ```

### Problem: "A fatal error occurred: MD5 of file does not match"

**Symptoms**: Flash verification fails

**Solutions**:

1. **Erase flash completely**
   ```bash
   esptool.py --chip esp32s3 --port /dev/ttyUSB0 erase_flash
   ```

2. **Flash again with lower baud rate**
   ```bash
   idf.py -p /dev/ttyUSB0 -b 115200 flash
   ```

3. **Check power supply**
   - Use quality USB cable
   - Try powered USB hub
   - Use external 5V power supply

### Problem: "Timed out waiting for packet header"

**Symptoms**: Flash process hangs or times out

**Solutions**:

1. **Reset ESP32 manually**
   - Press and hold BOOT button
   - Press and release RESET button
   - Release BOOT button
   - Try flashing again

2. **Lower baud rate**
   ```bash
   idf.py -p /dev/ttyUSB0 -b 115200 flash
   ```

3. **Check for interference**
   - Move away from other electronic devices
   - Disconnect other USB devices

---

## Wi-Fi Connection Issues

### Problem: "WiFi: Disconnected, reconnecting..."

**Symptoms**: ESP32 cannot connect to Wi-Fi or keeps disconnecting

**Solutions**:

1. **Verify credentials**
   ```bash
   idf.py menuconfig
   # ESPectre Configuration → WiFi SSID/Password
   ```

2. **Check 2.4 GHz band**
   - ESP32-S3 only supports 2.4 GHz
   - Ensure router has 2.4 GHz enabled
   - Some routers disable 2.4 GHz by default

3. **Check Wi-Fi signal strength**
   - Move ESP32 closer to router
   - Check for obstacles (metal, concrete walls)
   - Monitor RSSI in serial output

4. **Router compatibility**
   - Ensure router supports 802.11n
   - Try disabling 802.11ax (Wi-Fi 6) if enabled
   - Check router security: WPA2 recommended

5. **Check SSID visibility**
   - Hidden SSIDs may cause issues
   - Make SSID visible temporarily for testing

6. **Disable power saving**
   ```c
   // In main/espectre.c, add after wifi_init():
   esp_wifi_set_ps(WIFI_PS_NONE);
   ```

### Problem: "WiFi: Authentication failed"

**Symptoms**: Wrong password or authentication error

**Solutions**:

1. **Verify password**
   - Check for typos
   - Passwords are case-sensitive
   - Special characters may need escaping

2. **Check security mode**
   - WPA2-PSK recommended
   - WPA3 may not be supported
   - Avoid WEP (deprecated)

3. **Router settings**
   - Disable MAC filtering temporarily
   - Check maximum client limit
   - Verify DHCP is enabled

---

## MQTT Connection Issues

### Problem: "MQTT: Disconnected from broker"

**Symptoms**: Cannot connect to MQTT broker

**Solutions**:

1. **Verify broker is running**
   ```bash
   # Check Mosquitto status
   sudo systemctl status mosquitto
   
   # Start if not running
   sudo systemctl start mosquitto
   ```

2. **Test broker connectivity**
   ```bash
   # From another machine
   mosquitto_pub -h <broker-ip> -t test -m "hello"
   
   # Subscribe to test
   mosquitto_sub -h <broker-ip> -t test
   ```

3. **Check broker IP/hostname**
   ```bash
   idf.py menuconfig
   # ESPectre Configuration → MQTT Broker URI
   # Format: mqtt://192.168.1.100:1883
   ```

4. **Verify network connectivity**
   ```bash
   # Ping broker from ESP32 network
   ping <broker-ip>
   ```

5. **Check firewall**
   ```bash
   # Linux: Allow MQTT port
   sudo ufw allow 1883/tcp
   
   # Check if port is open
   telnet <broker-ip> 1883
   ```

### Problem: "MQTT: Connection refused"

**Symptoms**: Broker refuses connection

**Solutions**:

1. **Check authentication**
   - Verify username/password in menuconfig
   - Test credentials with mosquitto_pub:
   ```bash
   mosquitto_pub -h <broker-ip> -t test -m "hello" -u username -P password
   ```

2. **Check broker configuration**
   ```bash
   # Edit mosquitto.conf
   sudo nano /etc/mosquitto/mosquitto.conf
   
   # Ensure these lines exist:
   listener 1883
   allow_anonymous true  # Or configure password file
   ```

3. **Restart broker**
   ```bash
   sudo systemctl restart mosquitto
   ```

### Problem: Messages not appearing in Home Assistant

**Symptoms**: MQTT publishes but HA doesn't show data

**Solutions**:

1. **Verify MQTT integration**
   - Settings → Devices & Services → MQTT
   - Check if integration is configured

2. **Check topic subscription**
   ```bash
   # Subscribe to all topics
   mosquitto_sub -h <broker-ip> -t "#" -v
   
   # Check if messages appear
   ```

3. **Verify sensor configuration**
   ```yaml
   # configuration.yaml
   mqtt:
     sensor:
       - name: "Movement Sensor"
         state_topic: "home/espectre/node1"  # Match ESP32 topic
         value_template: "{{ value_json.movement }}"
   ```

4. **Restart Home Assistant**
   - Developer Tools → YAML → Restart

---

## CSI Acquisition Issues

### Problem: No CSI packets received

**Symptoms**: `packets_received` stays at 0

**Solutions**:

1. **Verify CSI is enabled**
   ```bash
   # Check sdkconfig
   grep CONFIG_ESP_WIFI_CSI_ENABLED sdkconfig
   # Should show: CONFIG_ESP_WIFI_CSI_ENABLED=y
   ```

2. **Ensure Wi-Fi traffic exists**
   - Router must be transmitting packets
   - Other devices should be connected
   - Try streaming video or downloading files

3. **Check antenna**
   - If using external antenna, verify IPEX connection
   - Try built-in antenna
   - Check antenna orientation

4. **Move closer to router**
   - Optimal distance: 3-8 meters
   - Too far: weak signal
   - Too close: signal too stable

5. **Verify Wi-Fi channel**
   - ESP32 should be on same channel as router
   - Check router channel (1-11 for 2.4 GHz)

### Problem: Low CSI packet rate

**Symptoms**: Very few packets per second

**Solutions**:

1. **Increase Wi-Fi activity**
   - Connect more devices to router
   - Generate traffic (streaming, downloads)

2. **Check signal strength**
   - Move ESP32 to better location
   - Reduce obstacles between router and ESP32

3. **Verify router settings**
   - Ensure router is not in power-saving mode
   - Check beacon interval (should be ~100ms)

---

## Detection Issues

### Problem: No movement detected

**Symptoms**: `state` always shows "idle"

**Solutions**:

1. **Wait for calibration**
   - First 60 seconds are calibration
   - Keep environment static during calibration
   - Check serial output for "Calibration complete"

2. **Check baseline and threshold**
   ```
   # In serial output, look for:
   Calibration complete: baseline=0.123, threshold=0.350
   
   # If threshold is too high, detection won't trigger
   ```

3. **Adjust sensitivity**
   ```c
   // In main/espectre.c
   #define THRESHOLD_MULT 2.0f  // Lower = more sensitive
   ```

4. **Verify movement location**
   - Move in the area between router and ESP32
   - Avoid moving behind ESP32 or router
   - Try larger movements

5. **Check distance from router**
   - Optimal: 3-8 meters
   - Too close: signal too stable
   - Too far: signal too weak

### Problem: Too many false positives

**Symptoms**: Constant detections with no movement

**Solutions**:

1. **Increase threshold**
   ```c
   #define THRESHOLD_MULT 3.0f  // Higher = less sensitive
   ```

2. **Increase debouncing**
   ```c
   #define DEBOUNCE_COUNT 5  // Require more consecutive detections
   ```

3. **Increase filtering**
   ```c
   #define MEDIAN_WINDOW 7  // More noise filtering
   #define EMA_ALPHA 0.2f   // More smoothing
   ```

4. **Check for interference**
   - Move away from other electronic devices
   - Check for vibrations (fans, appliances)
   - Avoid areas with moving objects (curtains, plants)

5. **Recalibrate**
   - Reset ESP32 to recalibrate
   - Ensure environment is static during calibration

### Problem: Detection too slow

**Symptoms**: Delay between movement and detection

**Solutions**:

1. **Reduce filtering**
   ```c
   #define EMA_ALPHA 0.4f  // More responsive
   ```

2. **Reduce debouncing**
   ```c
   #define DEBOUNCE_COUNT 2  // Faster detection
   ```

3. **Reduce publish interval**
   ```c
   #define PUBLISH_INTERVAL 0.5f  // Publish twice per second
   ```

---

## Performance Issues

### Problem: High latency

**Symptoms**: Slow response, delayed MQTT messages

**Solutions**:

1. **Check CPU frequency**
   ```bash
   idf.py menuconfig
   # Component config → ESP32S3-Specific → CPU frequency → 240 MHz
   ```

2. **Optimize MQTT**
   - Reduce publish interval
   - Use QoS 0 (fire-and-forget)
   - Reduce payload size

3. **Check network**
   - Verify Wi-Fi signal strength
   - Check for network congestion
   - Test MQTT broker latency

### Problem: ESP32 resets randomly

**Symptoms**: Unexpected reboots, watchdog timeouts

**Solutions**:

1. **Check power supply**
   - Use quality USB cable
   - Ensure adequate current (500mA minimum)
   - Try external 5V power supply

2. **Check for stack overflow**
   ```bash
   # Increase stack size in menuconfig
   idf.py menuconfig
   # Component config → ESP32S3-Specific → Main task stack size → 8192
   ```

3. **Monitor watchdog**
   ```bash
   # Check serial output for:
   # "Task watchdog got triggered"
   ```

4. **Check brownout detector**
   ```bash
   idf.py menuconfig
   # Component config → ESP32S3-Specific → Brownout voltage → 2.7V
   ```

---

## Memory Issues

### Problem: "Out of memory" errors

**Symptoms**: Heap allocation failures, crashes

**Solutions**:

1. **Enable PSRAM**
   ```bash
   idf.py menuconfig
   # Component config → ESP32S3-Specific → Support for external PSRAM → Enable
   ```

2. **Check memory usage**
   ```c
   // Add to code for debugging
   ESP_LOGI(TAG, "Free heap: %d", esp_get_free_heap_size());
   ```

3. **Reduce buffer sizes**
   ```c
   #define BUFFER_SIZE 50  // Reduce from 100
   ```

4. **Check for memory leaks**
   - Review code for malloc without free
   - Check cJSON object cleanup

---

## Hardware Issues

### Problem: Antenna issues

**Symptoms**: Weak signal, poor CSI quality

**Solutions**:

1. **Check IPEX connection**
   - Ensure connector is fully seated
   - Check for damage

2. **Try different antenna**
   - Use external antenna for better reception
   - Position antenna vertically

3. **Test built-in antenna**
   - Disconnect external antenna
   - Compare performance

### Problem: Overheating

**Symptoms**: ESP32 gets very hot, throttling

**Solutions**:

1. **Add heatsink**
   - Small aluminum heatsink on ESP32 chip
   - Improve airflow

2. **Reduce CPU frequency**
   ```bash
   idf.py menuconfig
   # Component config → ESP32S3-Specific → CPU frequency → 160 MHz
   ```

3. **Check power supply**
   - Excessive current may indicate short circuit
   - Verify no solder bridges

---

## Build and Compilation Issues

### Problem: "CMake Error: Could not find ESP-IDF"

**Symptoms**: Build fails, ESP-IDF not found

**Solutions**:

1. **Source ESP-IDF environment**
   ```bash
   cd ~/esp/esp-idf
   . ./export.sh
   ```

2. **Add to shell profile**
   ```bash
   # Add to ~/.bashrc or ~/.zshrc
   alias get_idf='. $HOME/esp/esp-idf/export.sh'
   ```

### Problem: "undefined reference to esp_wifi_set_csi"

**Symptoms**: Linker error, CSI functions not found

**Solutions**:

1. **Enable CSI in sdkconfig**
   ```bash
   idf.py menuconfig
   # Component config → Wi-Fi → Enable CSI
   ```

2. **Clean and rebuild**
   ```bash
   idf.py fullclean
   idf.py build
   ```

### Problem: Build takes very long

**Symptoms**: Compilation is slow

**Solutions**:

1. **Enable ccache**
   ```bash
   idf.py menuconfig
   # Compiler options → Enable ccache
   ```

2. **Use ninja instead of make**
   ```bash
   # Already default in ESP-IDF v5.x
   ```

---

## 🆘 Getting Help

If you're still experiencing issues:

1. **Check serial logs**
   ```bash
   idf.py -p /dev/ttyUSB0 monitor
   ```

2. **Enable debug logging**
   ```bash
   idf.py menuconfig
   # Component config → Log output → Default log verbosity → Debug
   ```

3. **Collect information**
   - ESP-IDF version: `idf.py --version`
   - Board model: ESP32-S3-DevKitC-1 N16R8
   - Serial output with errors
   - Configuration (sdkconfig)

4. **Search existing issues**
   - [ESPectre GitHub Issues](https://github.com/francescopace/espectre/issues)
   - [ESP-IDF GitHub Issues](https://github.com/espressif/esp-idf/issues)

5. **Open new issue**
   - Provide detailed description
   - Include error messages
   - Attach serial logs
   - Describe steps to reproduce

---

## 📚 Additional Resources

- [ESP-IDF Documentation](https://docs.espressif.com/projects/esp-idf/en/latest/esp32s3/)
- [ESP32-S3 Datasheet](https://www.espressif.com/sites/default/files/documentation/esp32-s3_datasheet_en.pdf)
- [ESP32 Forum](https://esp32.com/)
- [Home Assistant MQTT](https://www.home-assistant.io/integrations/mqtt/)

---

**Remember**: Most issues can be resolved by checking logs, verifying configuration, and ensuring proper hardware connections. When in doubt, start with the basics: power, connections, and configuration.
