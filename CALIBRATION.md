# 🛜 ESPectre 👻 - Calibration Guide

Quick guide to calibrate ESPectre for reliable movement detection in your environment.

> **💡 Tip**: Use the CLI tool (`espectre-cli.sh`) or the web monitor (`espectre-monitor.html`) for real-time tuning.

![Web Monitor Configuration Panel](images/web_monitor_configurations.png)
*Web monitor configuration interface showing all tunable parameters*

---

## 🚀 Quick Start (5 minutes)

### 1. Launch Interactive CLI

```bash
./espectre-cli.sh
```

This starts an interactive session with real-time feedback.

### 2. Observe Baseline (Empty Room)

Keep the room empty and still for 1-2 minutes.

**Check runtime statistics:**
```
espectre> stats
```

Note the typical `movement` values when idle (usually 0.05-0.20).

### 3. Set Segmentation Threshold

**This is the critical parameter for motion detection.**

**Rule of thumb:**
- If baseline variance is 0.05-0.10 → set threshold to 0.30-0.40
- If baseline variance is 0.15-0.20 → set threshold to 0.45-0.55

```
espectre> segmentation_threshold 0.35
```

### 4. Test Movement

Walk around the room while monitoring:
```bash
idf.py monitor
```

Look for `state=MOTION` when moving, `state=IDLE` when still.

**Adjust if needed:**
- Too many false positives → increase threshold
- Missing movements → decrease threshold

---

## 🎯 Essential Parameters

### Segmentation Threshold

**What it does:** Determines sensitivity for motion detection.

**Default:** 0.40

**Quick reference:**
| Environment | Recommended Range |
|-------------|-------------------|
| Small room (< 20m²) | 0.25 - 0.35 |
| Medium room (20-50m²) | 0.35 - 0.45 |
| Large room (> 50m²) | 0.45 - 0.60 |
| High interference | 0.45 - 0.60 |

**Command:**
```
espectre> segmentation_threshold 0.35
```

---

## ⚙️ Optional Parameters

### Traffic Generator Rate

**What it does:** Controls how many packets per second are sent for CSI measurement.

**Default:** 20 pps

**When to adjust:**
- Lower (10-15 pps): Reduce network load, slower detection
- Higher (25-30 pps): Faster detection, more network traffic

**Command:**
```
espectre> traffic_rate 20
```

### Feature Extraction

**What it does:** Enables detailed feature analysis during motion (turbulence, PCA, etc.).

**Default:** Enabled

**When to disable:**
- Only need basic motion detection (IDLE/MOTION)
- Want to reduce CPU usage

**Commands:**
```
# Disable features
espectre> features_enable off

# Re-enable features
espectre> features_enable on
```

### Smart Publishing

**What it does:** Controls when data is published via MQTT.

**Options:**
- `always`: Publish every packet (high traffic)
- `motion_only`: Publish only during motion (recommended)
- `segments_only`: Publish only complete motion segments

**Default:** motion_only

**Command:**
```
espectre> smart_publishing motion_only
```

---

## 🔧 Optional Filters

**Note:** Filters are applied only to feature extraction, not to motion detection.

### Butterworth Low-Pass Filter

**What it does:** Removes high-frequency noise (>8Hz) from features.

**Default:** Enabled (recommended)

**Command:**
```
espectre> butterworth_filter on
```

### Hampel Filter (Outlier Removal)

**What it does:** Removes statistical outliers from features.

**Default:** Disabled

**When to enable:** High interference environments (fans, AC, microwave).

**Commands:**
```
espectre> hampel_filter on
espectre> hampel_threshold 2.5
```

### Savitzky-Golay Filter

**What it does:** Smooths feature values.

**Default:** Enabled

**Command:**
```
espectre> savgol_filter on
```

### Wavelet Filter (Advanced)

**What it does:** Removes persistent low-frequency noise using wavelet transform.

**Default:** Disabled

**When to enable:** Very high noise environments where other filters aren't enough.

**Commands:**
```
espectre> wavelet_filter on
espectre> wavelet_level 3
espectre> wavelet_threshold 1.0
```

**Performance impact:** ~5-8% CPU load, 320ms warm-up time.

---

## 🔍 Troubleshooting

### Too Many False Positives

**Symptoms:** Detects motion when room is empty.

**Solutions:**
1. Increase threshold:
   ```
   espectre> segmentation_threshold 0.50
   ```

2. Enable Hampel filter:
   ```
   espectre> hampel_filter on
   ```

3. Check for interference sources (fans, AC, moving curtains).

### Missing Movements

**Symptoms:** Doesn't detect when people move.

**Solutions:**
1. Decrease threshold:
   ```
   espectre> segmentation_threshold 0.30
   ```

2. Check sensor position (optimal: 3-8m from router).

3. Verify traffic generator is active:
   ```
   espectre> info
   ```

### Unstable Detection

**Symptoms:** Rapid flickering between IDLE and MOTION.

**Solutions:**
1. Adjust threshold:
   ```
   espectre> segmentation_threshold 0.40
   ```

2. Enable smoothing filters:
   ```
   espectre> savgol_filter on
   espectre> hampel_filter on
   ```

### Factory Reset

**When needed:** Start fresh with default settings.

**Command:**
```
espectre> factory_reset
```

This clears all NVS storage and restores factory defaults.

---

## 📊 Monitoring Commands

### Check Current Configuration
```
espectre> info
```

Shows all current settings (threshold, filters, network, etc.).

### Check Runtime Statistics
```
espectre> stats
```

Shows dynamic metrics (state, turbulence, variance, segments, uptime).

### Monitor Real-Time Detection
```bash
idf.py monitor
```

Shows live CSI data and detection logs.

---

## 🎓 Quick Tips

1. **Start simple:** Tune only the segmentation threshold first.
2. **One change at a time:** Adjust one parameter, test for 5-10 minutes.
3. **Document your settings:** Note what works for your environment.
4. **Seasonal adjustments:** Retune when furniture changes or new interference sources appear.
5. **Distance matters:** Keep sensor 3-8m from router for optimal performance.

---

## 📚 Additional Resources

- **Main Documentation:** [README.md](README.md)
- **Setup Guide:** [SETUP.md](SETUP.md)
- **GitHub Issues:** [Report problems](https://github.com/francescopace/espectre/issues)
- **Email:** francesco.pace@gmail.com

---

## 🔧 CLI Configuration

The CLI reads settings from environment variables:

```bash
export MQTT_BROKER="homeassistant.local"
export MQTT_PORT="1883"
export MQTT_TOPIC="home/espectre/node1"
export MQTT_USERNAME="mqtt"
export MQTT_PASSWORD="mqtt"
./espectre-cli.sh
```

Type `help` in the CLI for all available commands.
