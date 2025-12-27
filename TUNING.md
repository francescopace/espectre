# Tuning Guide

Quick guide to tune ESPectre for reliable movement detection in your environment.

---

## Quick Start (5 minutes)

> **Note on Subcarrier Selection**: ESPectre automatically selects optimal subcarriers using the NBVI (Normalized Baseline Variability Index) algorithm at first boot. No manual configuration needed.

### 1. Flash and Boot

After flashing your device with ESPHome:

```bash
# View logs to monitor calibration
esphome logs <your-config>.yaml
```

### 2. Wait for NBVI Calibration

On first boot, keep the room **empty and still** for 10 seconds. The system will:
1. Collect 1000 CSI packets
2. Analyze subcarrier characteristics
3. Select optimal 12 subcarriers
4. Save configuration (persists across reboots)

Look for log messages like:
```
[I][espectre:xxx]: NBVI calibration complete, selected 12 subcarriers
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

If detection isn't working well, modify `segmentation_threshold` in your YAML:

```yaml
espectre:
  segmentation_threshold: 1.0  # Default
```

**Rule of thumb:**
- Too many false positives → increase threshold (try 2.0-5.0)
- Missing movements → decrease threshold (try 0.5-0.8)

After changing, re-flash:
```bash
esphome run <your-config>.yaml
```

**Interactive tuning:** You can also adjust the threshold in real-time using [ESPectre - The Game](https://espectre.dev/game). Connect via USB, drag the threshold slider, and see immediate visual feedback. Changes are saved to flash and apply to Home Assistant as well.

---

## Understanding Parameters

### Segmentation Threshold (0.5-10.0)

**What it does:** Determines sensitivity for motion detection. With normalization enabled (default), this represents multiples of baseline noise.

**Default:** 1.0

| Value | Sensitivity | Use Case |
|-------|-------------|----------|
| 0.5-1.0 | High | **Default** - Detect subtle movements |
| 1.5-3.0 | Medium | General purpose, most environments |
| 3.0-5.0 | Low | Noisy environments, reduce false positives |
| 5.0-10.0 | Very Low | Only detect significant movements |

**Configuration:**
```yaml
espectre:
  segmentation_threshold: 1.0  # "1x baseline noise" (default)
```

**Note:** Normalization is always enabled, so threshold 1.0 means "4× baseline noise". This works consistently across all ESP32 variants.

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
| 0 pps | Disabled (only if you have other continuous WiFi traffic) |

**Configuration:**
```yaml
espectre:
  traffic_generator_rate: 100
```

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

## Normalization (Auto-Scaling)

### Baseline Variance Normalization

**What it does:** Automatically attenuates turbulence values when baseline is too high. This prevents extreme motion values while keeping signals in a consistent range.

**Status:** Always enabled (automatic)

**How it works:**
1. During calibration, calculates the baseline variance using selected subcarriers
2. If baseline > 0.25: attenuate with `scale = 0.25 / baseline_variance`
3. If baseline ≤ 0.25: no scaling needed (scale = 1.0)
4. Result: baseline is never amplified, only attenuated when necessary

**Fallback behavior:** If subcarrier selection fails (e.g., due to poor signal quality), normalization is still calculated using the default subcarriers. This ensures motion detection remains usable even when NBVI calibration cannot find optimal subcarriers.

**Benefits:**
- Prevents over-amplification of weak signals (avoids extreme motion values)
- Attenuates strong signals to keep them in usable range
- Same threshold works on ESP32-S3, C6, C3, etc.
- No device-specific tuning required

**Threshold guidelines:**

| Threshold | Baseline multiplier | Use Case |
|-----------|---------------------|----------|
| 0.5 | 2× baseline | Very sensitive, may have false positives |
| **1.0** | **4× baseline** | **Default** - Good balance |
| 2.0 | 8× baseline | Less sensitive, noisy environments |
| 3.0+ | 12×+ baseline | Only detect significant movements |

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

### Reset Calibration

**When needed:** Start fresh with new subcarrier selection.

**How to reset:**

1. Erase flash completely:
   ```bash
   esphome run <your-config>.yaml --device /dev/ttyUSB0
   # Choose "Erase flash before uploading" if available
   ```

2. Or use ESPHome dashboard: **Clean Build Files** then re-install

**After reset:**
- Keep room quiet and empty for 10 seconds
- NBVI will automatically recalibrate
- Check logs for "NBVI calibration complete"

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

**Tip:** You can adjust the threshold directly from Home Assistant without re-flashing. Changes are saved to device preferences and persist across reboots.

---

## Quick Tips

1. **Start simple:** Tune only the segmentation threshold first
2. **One change at a time:** Adjust one parameter, re-flash, test for 5-10 minutes
3. **Document your settings:** Note what works for your environment
4. **Seasonal adjustments:** Retune when furniture changes or new interference sources appear
5. **Distance matters:** Keep sensor 3-8m from router for optimal performance
6. **Quiet calibration:** Ensure no movement during first 5-10 seconds after boot
7. **Try the game:** Use [ESPectre - The Game](https://espectre.dev/game) for interactive threshold tuning with real-time visual feedback

---

## Additional Resources

- **Main Documentation:** [README.md](README.md)
- **Setup Guide:** [SETUP.md](SETUP.md)

---

## License

GPLv3 - See [LICENSE](LICENSE) for details.
