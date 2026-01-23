# Tuning Guide

Quick guide to tune ESPectre for reliable movement detection in your environment.

> **Note on Detection Algorithms**: This guide focuses on **MVS** (default algorithm). PCA uses fixed parameters and requires no tuning. See [SETUP.md](SETUP.md#choosing-detection-algorithm) for algorithm selection.

---

## Quick Start (5 minutes)

> **Note on Subcarrier Selection**: ESPectre automatically selects optimal subcarriers at first boot using NBVI (default) or P95 algorithm. No manual configuration needed. Only applies to MVS; PCA uses fixed subcarriers.

### 1. Flash and Boot

After flashing your device with ESPHome:

```bash
# View logs to monitor calibration
esphome logs <your-config>.yaml
```

### 2. Wait for Band Calibration

On first boot, keep the room **empty and still** for 10 seconds. The system will:
1. Collect ~700 CSI packets during baseline
2. Run calibration algorithm (NBVI or P95) to select optimal subcarriers
3. Select 12 optimal subcarriers for motion detection
4. Calculate adaptive threshold (P95 × 1.4)

Look for log messages like:
```
[I][Calibration]: ✓ Calibration successful: [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
```

### 3. Test Movement

Walk around the room while monitoring logs:

```bash
esphome logs <your-config>.yaml
```

Look for state changes:
- `state=MOTION` when moving
- `state=IDLE` when still

### 4. Adjust Threshold if Needed

By default, ESPectre uses an **adaptive threshold** calculated automatically during calibration (`P95 × 1.4`). This works well in most environments.

```yaml
espectre:
  segmentation_threshold: auto
```

| Value | Description |
|-------|-------------|
| `auto` | P95 × 1.4 - Minimizes false positives (default) |
| `min` | P100 × 1.0 - Maximum sensitivity (may have false positives) |
| `0.1-10.0` | Fixed manual threshold |

**Examples:**
```yaml
espectre:
  segmentation_threshold: auto  # Default, zero FP
  # segmentation_threshold: min  # Max sensitivity
  # segmentation_threshold: 1.5  # Fixed manual value
```

**Rule of thumb:**
- Too many false positives → use `auto` or increase threshold (try 2.0-5.0)
- Missing movements → use `min` or decrease threshold (try 0.5-0.8)

After changing, re-flash:
```bash
esphome run <your-config>.yaml
```

**Interactive tuning:** You can also adjust the threshold in real-time using [ESPectre - The Game](https://espectre.dev/game). Connect via USB, drag the threshold slider, and see immediate visual feedback. Note that runtime adjustments are temporary (session-only) - the adaptive threshold is recalculated on every boot.

---

## Understanding Parameters

> **MVS Only**: The following parameters apply to the MVS detection algorithm. PCA uses its own threshold calculation (`1 - min(correlation)`) and fixed subcarrier selection.

### Segmentation Threshold

**What it does:** Determines sensitivity for motion detection (MVS only).

**Default:** `auto` (P95 × 1.4, minimizes false positives)

| Value | Sensitivity | Use Case |
|-------|-------------|----------|
| 0.5-1.0 | High | Detect subtle movements |
| 1.5-3.0 | Medium | General purpose, most environments |
| 3.0-5.0 | Low | Noisy environments, reduce false positives |
| 5.0-10.0 | Very Low | Only detect significant movements |

**Configuration:**
```yaml
espectre:
  segmentation_threshold: auto  # or "min" or a number (0.1-10.0)
```

| Value | Formula | Effect |
|-------|---------|--------|
| `auto` | P95 × 1.4 | Minimizes false positives (default) |
| `min` | P100 × 1.0 | Maximum sensitivity |
| number | Fixed | Manual override |

**Note:** Runtime adjustments via Home Assistant slider are temporary (session-only). The adaptive threshold is recalculated on every boot.

### Window Size (10-200 packets)

**What it does:** Number of turbulence samples used to calculate moving variance.

**Default:** 50 packets

| Value | Response | Stability | Use Case |
|-------|----------|-----------|----------|
| 10-30 | Fast | Noisy | Quick response needed |
| 50-100 | Balanced | Good | **Recommended** |
| 100-200 | Slow | Very stable | Reduce flickering |

**Configuration:**
```yaml
espectre:
  segmentation_window_size: 50
```

<details>
<summary><b>Optimal Window Configuration Guide</b></summary>

To detect general movement (walking, arm movement, standing up), you need to balance **sensitivity** (capturing even minimal movements) and **robustness** (ignoring noise).

**Sampling Rate ($F_s$)**

Maintaining $F_s = 100 \text{ Hz}$ is an excellent compromise between accuracy and computational load for detecting most human activities.

**Moving Window Size ($N$)**

For general movement detection, a window is recommended that captures transient action while being long enough to dampen high-frequency noise.

| $T_{window}$ | $N$ (at 100 Hz) | Advantage for Presence Detection |
|--------------|-----------------|----------------------------------|
| $0.5$ seconds | $50$ packets | Extremely reactive, but too sensitive to noise. |
| $1$ second | $100$ packets | **Recommended**. Optimal balance, captures $1-2$ steps or a complete gesture. |
| $2$ seconds | $200$ packets | Slower to react, but very robust against false positives. |

**Recommendation:** Start with $N=50$ packets (corresponding to $0.5$ seconds). This is a good starting point for detecting activities like entering a room.

</details>

### Traffic Generator Rate (0-1000 pps)

**What it does:** Controls how many packets per second are sent for CSI measurement.

**Default:** 100 pps

| Rate | Use Case |
|------|----------|
| 50 pps | Basic presence detection, minimal overhead |
| 100 pps | **Recommended** - Activity recognition |
| 600-1000 pps | Fast motion detection, precision localization |
| 0 pps | Disabled - use external WiFi traffic (see [External Traffic Mode](SETUP.md#external-traffic-mode)) |

**Configuration:**
```yaml
espectre:
  traffic_generator_rate: 100
```

### Publish Interval (1-1000 packets)

**What it does:** Controls how many CSI packets are processed before updating Home Assistant sensors.

**Default:** Same as `traffic_generator_rate` (or 100 if traffic generator is disabled)

| Scenario | Configuration | Update Frequency |
|----------|---------------|------------------|
| Default | `traffic_generator_rate: 100` | ~1 update/sec |
| Faster updates | `publish_interval: 50` | ~2 updates/sec |
| External traffic | `traffic_generator_rate: 0`, `publish_interval: 100` | Depends on traffic |

**Configuration:**
```yaml
espectre:
  traffic_generator_rate: 100
  publish_interval: 50  # Optional: override publish rate
```

> **Note:** Lower `publish_interval` values increase CPU usage and Home Assistant traffic but provide more responsive detection.

<details>
<summary><b>Understanding Sampling Rates (Nyquist-Shannon Theorem)</b></summary>

The traffic generator rate determines the maximum frequency of motion that can be accurately detected. According to the **Nyquist-Shannon Sampling Theorem**, the sampling rate (Fs) must be at least twice the maximum frequency of the signal (Fmax):

$$F_s \geq 2 \times F_{max}$$

In Wi-Fi sensing, Fmax is the highest Doppler frequency generated by human movement reflected in the CSI signal.

**Application Scenarios:**

| Activity Type | Max Frequency (Fmax) | Minimum Sampling Rate (Fs) | Recommended Rate |
|---------------|---------------------|---------------------------|------------------|
| **Vital signs** (breathing, heartbeat) | < 5 Hz | ≥ 10 Hz | 10-30 pps |
| **Activity recognition** (walking, sitting, gestures) | ≈ 10-30 Hz | ≥ 60 Hz | 60-100 pps |
| **Fast motion** (rapid gestures, precision localization) | ≈ 300-400 Hz | ≥ 600 Hz | 600-1000 pps |

**Key Takeaways:**
- Higher rates enable detection of faster movements
- Lower rates are sufficient for slow movements (vital signs, presence)
- Choose the rate based on your application requirements
- Higher rates increase CPU usage and network traffic

</details>

---

## Adaptive Threshold

### Automatic Threshold Calibration

**What it does:** Automatically calculates the optimal detection threshold based on baseline noise characteristics. This minimizes false positives while maintaining high recall.

**Status:** Always enabled (automatic)

**How it works:**
1. During calibration, collects ~700 CSI packets during baseline (room quiet)
2. **Band selection** uses P95 (fixed algorithm) to rank candidate bands
3. **Threshold calculation** depends on `segmentation_threshold` setting

| Mode | Formula | Description |
|------|---------|-------------|
| `auto` (default) | P95 × 1.4 | Minimizes false positives |
| `min` | P100 × 1.0 | Maximum sensitivity (threshold = max baseline) |

**Note:** Band selection always uses P95 (the validated algorithm). Only the threshold calculation varies.

**Configuration:**
```yaml
espectre:
  segmentation_threshold: auto  # or "min" or a number
```

**Benefits:**
- **Minimizes false positives** with `auto` mode
- **High recall** (>98%) maintained across environments
- Same algorithm works on ESP32-S3, C6, C3, etc.
- No device-specific tuning required
- Fully automatic - no manual threshold adjustment needed

**When to override:**
- Too many false positives → increase threshold (try 2.0-5.0)
- Missing movements → decrease threshold (try 0.5-0.8)

---

## Hampel Filter

### Hampel Filter (Outlier Removal)

**What it does:** Removes statistical outliers from turbulence values using MAD (Median Absolute Deviation). This can help reduce false positives caused by sudden interference.

**Default:** Disabled

> ⚠️ **Note:** The Hampel filter is disabled by default because MVS already provides robust motion detection with 0% false positives in typical environments. Enabling it reduces detection sensitivity (Recall drops from 98.1% to 96.3%). Only enable in environments with high electromagnetic interference causing sudden spikes.

**Configuration:**
```yaml
espectre:
  hampel_enabled: false    # Enable/disable (default: false)
  hampel_window: 7         # Window size (3-11 packets)
  hampel_threshold: 4.0    # MAD multiplier (1.0-10.0, lower = more aggressive)
```

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `hampel_enabled` | false | - | Enable/disable Hampel filter |
| `hampel_window` | 7 | 3-11 | Number of samples in the sliding window |
| `hampel_threshold` | 4.0 | 1.0-10.0 | MAD multiplier (lower = more aggressive filtering) |

**When to enable:**
- Environments with high electromagnetic interference causing sudden spikes
- Industrial settings with heavy machinery
- Proximity to microwave ovens or multiple WiFi access points
- Experiencing unexplained false positives during baseline (room empty)

**When to keep disabled (default):**
- Normal home/office environment
- Maximum detection sensitivity needed

---

## Gain Lock

### AGC/FFT Gain Lock

**What it does:** Locks the AGC (Automatic Gain Control) and FFT gain values after initial calibration to eliminate amplitude variations caused by the WiFi hardware. This improves CSI stability and motion detection accuracy.

**Default:** auto

**Configuration:**
```yaml
espectre:
  gain_lock: auto  # auto (default), enabled, disabled
```

| Mode | Description |
|------|-------------|
| `auto` | Enable gain lock but skip if signal too strong (AGC < 30). **Recommended.** |
| `enabled` | Always force gain lock. May freeze if too close to AP. |
| `disabled` | Never lock gain. Less stable CSI but works at any distance. |

**How it works:**
1. During the first 300 packets (~3 seconds), ESPectre measures average AGC/FFT values
2. These values are then "locked" (forced) to eliminate hardware-induced variations
3. In `auto` mode, if AGC < 30 (signal too strong), gain lock is skipped with a warning

**When to change from `auto`:**

| Situation | Recommended Setting |
|-----------|---------------------|
| Normal operation (3-8m from AP) | `auto` (default) |
| Testing very close to AP (< 2m) | `disabled` |
| Debugging calibration issues | `disabled` |
| Maximum CSI stability needed | `enabled` (if RSSI < -40 dB) |

**Warning log when signal too strong:**
```
[W][GainController]: Signal too strong (AGC=14 < 30) - skipping gain lock
[W][GainController]: Move sensor 2-3 meters from AP for optimal performance
```

> **Note:** Gain lock is only available on ESP32-S3, ESP32-C3, ESP32-C5, and ESP32-C6. On ESP32 (original) and ESP32-S2, it's automatically skipped (not supported by hardware).

---

## Low-Pass Filter

### Low-Pass Filter (Noise Reduction)

**What it does:** Removes high-frequency noise from turbulence values using a 1st-order Butterworth IIR filter. This significantly reduces false positives in noisy RF environments.

**Default:** Disabled

> ℹ️ **Note:** The low-pass filter is disabled by default for maximum simplicity. Enable it if you experience false positives in noisy RF environments.

**Configuration:**
```yaml
espectre:
  lowpass_enabled: false    # Enable/disable (default: false)
  lowpass_cutoff: 11.0      # Cutoff frequency in Hz (5.0-20.0)
```

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `lowpass_enabled` | false | - | Enable/disable low-pass filter |
| `lowpass_cutoff` | 11.0 | 5.0-20.0 | Cutoff frequency in Hz (lower = more filtering) |

**Cutoff frequency guide:**
- **Lower (5-8 Hz)**: More aggressive filtering, reduces FP more but may miss fast movements
- **Default (11 Hz)**: Good balance (92% recall, <3% FP)
- **Higher (15-20 Hz)**: Less filtering, higher recall but more FP

**When to adjust:**
- Increase cutoff if detecting fast movements (sports, rapid gestures)
- Decrease cutoff in very noisy RF environments with persistent FP

---

## Sensor Placement

### Distance from Access Point

The distance between the ESP32 sensor and your WiFi access point (AP) significantly impacts CSI quality and system stability.

| Distance | RSSI | AGC | Status | Notes |
|----------|------|-----|--------|-------|
| < 0.5m | > -30 dB | 0-15 | System may freeze | Too close, signal saturated |
| 0.5-2m | -30 to -40 dB | 15-30 | Marginal | Works with `gain_lock: disabled` |
| **2-8m** | -40 to -70 dB | **30-60** | **Optimal** | Best CSI quality and stability |
| 8-15m | -70 to -80 dB | 60-80 | Good | Still reliable detection |
| > 15m | < -80 dB | > 80 | Reduced quality | Weaker signal, more noise |

**Why distance matters:**

When the sensor is too close to the AP, the received signal is extremely strong, causing:
1. **AGC saturation**: The automatic gain control cannot reduce amplification enough
2. **CSI distortion**: Signal clipping leads to unreliable CSI data
3. **Gain lock freeze**: When `phy_force_rx_gain()` is called with a very low AGC value, the WiFi driver may fail to decode frames, halting CSI reception entirely

**Symptoms of being too close:**
- System freezes after "Auto-Calibration Starting" log
- Repeated "ping_sock: send error=0" messages
- Low AGC values in logs (< 30)
- High RSSI (> -40 dB)

**Solution:**
1. Move the sensor 2-3 meters away from the AP
2. Or set `gain_lock: disabled` (less optimal but works at any distance)

**Checking your placement:**

Look at the gain lock log after WiFi connection:
```
[I][GainController]: Gain locked: AGC=51, FFT=234 (after 300 packets)
```

- **AGC > 30**: Good placement, gain lock works correctly
- **AGC < 30**: Consider moving the sensor further from the AP

---

## Troubleshooting

### Too Many False Positives

**Symptoms:** Detects motion when room is empty.

**Solutions (try in order):**

1. **Increase threshold:**
   ```yaml
   espectre:
     segmentation_threshold: 3.0  # Try 2.0-5.0
   ```

2. **Increase window size** (more stable):
   ```yaml
   espectre:
     segmentation_window_size: 100  # Try 100-150
   ```

3. **Enable low-pass filter** (removes RF noise):
   ```yaml
   espectre:
     lowpass_enabled: true
     lowpass_cutoff: 11.0
   ```

4. **Enable Hampel filter** (removes spikes from interference):
   ```yaml
   espectre:
     hampel_enabled: true
   ```

5. **Check for interference sources:**
   - Fans, AC units, moving curtains
   - Microwave ovens, other WiFi networks
   - Bluetooth devices, cordless phones
   - Pets moving in the room

6. **Re-calibrate:** Reset calibration (see below) in a quiet room

### Missing Movements

**Symptoms:** Doesn't detect when people move.

**Solutions (try in order):**

1. **Decrease threshold:**
   ```yaml
   espectre:
     segmentation_threshold: 0.5  # Try 0.5-0.8
   ```

2. **Decrease window size** (faster response):
   ```yaml
   espectre:
     segmentation_window_size: 30  # Try 25-40
   ```

3. **Check sensor position:**
   - Optimal: 3-8m from router
   - Avoid placing behind furniture or walls
   - Line of sight to monitored area helps

4. **Verify traffic generator is active:**
   ```yaml
   espectre:
     traffic_generator_rate: 100  # Must be > 0
   ```

### No CSI Packets

**Symptoms:** Logs show no CSI data or "CSI disabled".

**Solutions:**

1. **Verify WiFi connection:** Check logs for successful connection to AP

2. **Check traffic generator:**
   ```yaml
   espectre:
     traffic_generator_rate: 100  # Must be > 0
   ```

3. **Verify ESP-IDF configuration:** Ensure `CONFIG_ESP_WIFI_CSI_ENABLED: y` in sdkconfig

4. **Check router compatibility:** Some mesh routers or WiFi 6E may have issues

### System Freezes During Calibration

**Symptoms:** Device freezes after "Auto-Calibration Starting (file-based storage)" message. May show watchdog timeout or repeated "ping_sock: send error=0" messages.

**Cause:** Sensor is too close to the access point. When RSSI > -40 dB, the AGC value is very low (< 30), and forcing this gain causes the WiFi driver to fail decoding frames.

**Solutions:**

1. **Move the sensor further from the AP** (recommended):
   - Place at least 2-3 meters away
   - Optimal distance: 3-8 meters
   - Check logs for AGC value > 30 after gain lock

2. **Disable gain lock** (workaround):
   ```yaml
   espectre:
     gain_lock: disabled
   ```
   This allows operation at any distance but with slightly less stable CSI.

3. **Use `auto` mode** (default, v2.4.0+):
   ```yaml
   espectre:
     gain_lock: auto  # Default - skips gain lock if AGC < 30
   ```
   In `auto` mode, ESPectre automatically skips gain lock when the signal is too strong, logging a warning instead of freezing.

**Diagnosis:**

Check the gain lock log:
```
[I][GainController]: Gain locked: AGC=51, FFT=234  # Good - AGC > 30
```

vs.
```
[W][GainController]: Signal too strong (AGC=14 < 30) - skipping gain lock  # Auto mode protection
```

---

### Unstable Detection (Flickering)

**Symptoms:** Rapid flickering between IDLE and MOTION.

**Solutions:**

1. **Increase threshold:**
   ```yaml
   espectre:
     segmentation_threshold: 2.0
   ```

2. **Increase window size** (smooths transitions):
   ```yaml
   espectre:
     segmentation_window_size: 100
   ```

3. **Enable low-pass filter** (removes noise):
   ```yaml
   espectre:
     lowpass_enabled: true
   ```

### False Positives After WiFi Channel Change

**Symptoms:** Sudden MOTION detection when no one is moving, typically after router auto-channel switch.

**Automatic handling:** ESPectre v2.3.0+ automatically detects channel changes and resets the detection buffer. Look for this log message:

```
[W][CSIManager]: WiFi channel changed: 6 -> 11, resetting detection buffer
```

**If you see frequent channel changes:**

1. **Fix router channel:** Disable auto-channel and set a fixed channel in your router settings
2. **Avoid DFS channels:** Channels 52-144 (5GHz DFS) may switch unexpectedly due to radar detection
3. **Check for interference:** Nearby networks on the same channel can cause instability

### Runtime Recalibration

**When needed:** Recalibrate without reflashing (e.g., after moving furniture or changing room layout).

**How to recalibrate from Home Assistant:**

1. Go to your ESPectre device in Home Assistant
2. Find the **Calibrate** switch (`switch.espectre_calibrate`)
3. Turn it ON to start calibration
4. The switch will automatically turn OFF when calibration completes

**Important:**
- Keep the room quiet and empty during calibration (~10 seconds)
- The switch is disabled during calibration to prevent interruption
- You cannot cancel calibration once started

**Logs during recalibration:**
```
[I][espectre]: Manual recalibration triggered
[I][espectre]: Starting band calibration...
[I][espectre]: Calibration completed successfully
```

### Reset Calibration (Full Erase)

**When needed:** Start completely fresh with new subcarrier selection and clear all saved settings.

**How to reset:**

1. Erase flash completely:
   ```bash
   esphome run <your-config>.yaml --device /dev/ttyUSB0
   # Choose "Erase flash before uploading" if available
   ```

2. Or use ESPHome dashboard: **Clean Build Files** then re-install

**After reset:**
- Keep room quiet and empty for 10 seconds
- P95 band selection will automatically recalibrate
- Check logs for "Calibration successful"

---

## Monitoring

### View Real-Time Logs

```bash
# Via USB
esphome logs <your-config>.yaml

# Via network (after first flash)
esphome logs <your-config>.yaml --device espectre.local
```

### Home Assistant

After integration, monitor sensors in Home Assistant:
- **binary_sensor.espectre_motion_detected** - Motion state
- **sensor.espectre_movement_score** - Movement intensity
- **number.espectre_threshold** - Adjustable detection threshold

Use **History** graphs to visualize detection patterns over time.

**Tip:** You can adjust the threshold directly from Home Assistant without re-flashing. Changes are session-only - the adaptive threshold (P95 × 1.4) is recalculated on every boot.

---

## Quick Tips

1. **Start simple:** Tune only the segmentation threshold first
2. **One change at a time:** Adjust one parameter, re-flash, test for 5-10 minutes
3. **Document your settings:** Note what works for your environment
4. **Seasonal adjustments:** Retune when furniture changes or new interference sources appear
5. **Distance matters:** Keep sensor 2-8m from router (RSSI between -40 and -70 dB for best results)
6. **Check AGC value:** After boot, look for "Gain locked: AGC=XX" - values 30-60 are optimal
7. **Quiet calibration:** Ensure no movement during first 5-10 seconds after boot
8. **Try the game:** Use [ESPectre - The Game](https://espectre.dev/game) for interactive threshold tuning with real-time visual feedback

---

## Additional Resources

- **Main Documentation:** [README.md](README.md)
- **Setup Guide:** [SETUP.md](SETUP.md)

---

## License

GPLv3 - See [LICENSE](LICENSE) for details.
