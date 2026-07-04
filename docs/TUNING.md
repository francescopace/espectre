# Tuning Guide

Quick guide to tune ESPectre for reliable movement detection in your environment.

> **Note on Detection Algorithms**: This guide focuses on **MVS** (the default detection algorithm). Filters (low-pass, Hampel) apply to both MVS and ML detectors.
>
> **Frontend scope**: This guide is written as a shared tuning reference first. When a workflow differs by frontend, the text calls that out explicitly instead of treating one frontend as the universal path.
>
> **Syntax note**: YAML snippets in this guide are `ESPHome` examples only. For frontend-specific configuration syntax, provisioning, runtime controls, or workflow details, use the README of the frontend you are actually deploying.

---

## Quick Start (5 minutes)

> **Note on Subcarrier Selection**: ESPectre now uses one shared fixed 12-subcarrier set for both MVS and ML. Only the threshold bootstrap differs between detectors.

### 1. Flash and Boot

Flash the frontend you want to deploy, boot the device, and open whatever
runtime visibility surface that frontend provides:

- logs or serial monitor
- frontend-specific runtime telemetry
- controller-side device state, when available

For frontend-owned setup details, continue in the relevant README.

### 2. Wait for Startup Calibration

On first boot, keep the room **empty and still** for 10 seconds. The system will:
1. Start CSI capture with AGC active
2. Collect CSI packets during baseline (10 × `window_size`, default 1000 packets)
3. Keep the shared fixed 12-subcarrier set for motion detection
4. Calculate adaptive threshold based on baseline noise

Look for log messages like:
```
[I][espectre]: Starting MVS threshold calibration with fixed subcarriers
[I][espectre]: Calibration completed successfully
```

NOTE: In ML mode, the threshold remains fixed and startup is immediate once CSI capture is active.

### 3. Test Movement

Walk around the room while monitoring the runtime surface exposed by your
frontend.

Look for state changes:
- `state=MOTION` when moving
- `state=IDLE` when still

### 4. Adjust Threshold if Needed

By default, ESPectre uses an **adaptive threshold** calculated automatically during calibration based on baseline noise. This works well in most environments.

```yaml
espectre:
  segmentation_threshold: auto
```

| Value | Description |
|-------|-------------|
| `auto` | Adaptive threshold - Minimizes false positives (default) |
| `min` | Maximum sensitivity (may have false positives) |
| `0.0-10.0` | Fixed manual threshold |

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

Apply the change through the configuration or control surface of your frontend.
For example:

- `ESPHome`: edit YAML and re-flash or update the runtime threshold entity
- `Matter`: use the writable threshold attribute exposed by the Matter surface
- `Native`: use a BLE client such as `tools/web/espectre-ble.html`, which sends `SET_THRESHOLD:X.XX` when you release the threshold slider

**Interactive tuning:** You can also adjust the threshold in real-time using a BLE client built on the native frontend protocol, including [ESPectre - The Game](https://espectre.dev/game) as one example. Connect, drag the threshold slider, and see immediate visual feedback. Note that runtime adjustments are temporary (session-only) - the adaptive threshold is recalculated on every boot.

---

## Understanding Parameters

> The following parameters apply to the MVS detection algorithm.

### Segmentation Threshold

**What it does:** Determines sensitivity for motion detection (MVS only).

**Default:** `auto` (adaptive threshold, minimizes false positives)

| Value | Sensitivity | Use Case |
|-------|-------------|----------|
| 0.5-1.0 | High | Detect subtle movements |
| 1.5-3.0 | Medium | General purpose, most environments |
| 3.0-5.0 | Low | Noisy environments, reduce false positives |
| 5.0-10.0 | Very Low | Only detect significant movements |

**Example ESPHome syntax:**
```yaml
espectre:
  segmentation_threshold: auto  # or "min" or a number (0.0-10.0)
```

| Value | Formula | Effect |
|-------|---------|--------|
| `auto` | Adaptive | Minimizes false positives (default) |
| `min` | Maximum sensitivity | Catches faint motion |
| number | Fixed | Manual override |

**Runtime note:** Runtime adjustments are temporary unless your frontend also
persists the value. The adaptive threshold is recalculated on every boot.

### Detection Algorithm (mvs/ml)

**What it does:** Selects the motion detection algorithm.

**Default:** `mvs`

```yaml
espectre:
  detection_algorithm: mvs  # or ml
```

| Algorithm | Description | Threshold Range | Best For |
|-----------|-------------|-----------------|----------|
| `mvs` | Moving Variance Segmentation | 0.0 - 10.0 | General purpose, adaptive |
| `ml` | Neural network detector | 0.0 - 10.0 (scaled metric) | Calibration-free boot |

### Window Size (10-200 packets)

**What it does:** Number of turbulence samples used to calculate moving variance.

**Default:** 100 packets

| Value | Response | Stability | Use Case |
|-------|----------|-----------|----------|
| 10-30 | Fast | Noisy | Quick response needed |
| 50-100 | Balanced | Good | **Recommended** |
| 100-200 | Slow | Very stable | Reduce flickering |

**Example ESPHome syntax:**
```yaml
espectre:
  segmentation_window_size: 100  # default
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

**Recommendation:** Start with $N=100$ packets (corresponding to $1$ second at 100 pps). This is the default and a good starting point for detecting activities like entering a room.

</details>

### Traffic Generator Rate (0-1000 pps)

**What it does:** Controls how many packets per second are sent for CSI measurement.

This section applies to frontends that expose the shared internal traffic
generator, such as `ESPHome`. The standalone `streamer` frontend is
collector-driven and should be tuned through its own stimulus workflow and
frontend README.

**Default:** 100 pps

| Rate | Use Case |
|------|----------|
| 50 pps | Basic presence detection, minimal overhead |
| 100 pps | **Recommended** - Activity recognition |
| 600-1000 pps | Fast motion detection, precision localization |
| 0 pps | Disabled - use external Wi-Fi traffic through the workflow owned by your frontend |

**Example ESPHome syntax:**
```yaml
espectre:
  traffic_generator_rate: 100
```

### Publish Interval (1-1000 packets)

**What it does:** Controls how often ESPectre publishes the movement score and periodic logs.
The motion binary sensor is no longer tied to this cadence: it is published
immediately on `IDLE <-> MOTION` state changes.

**Default:** Same as `traffic_generator_rate` (or 100 if traffic generator is disabled)

| Scenario | Configuration | Update Frequency |
|----------|---------------|------------------|
| Default | `traffic_generator_rate: 100` | ~1 update/sec |
| Faster updates | `publish_interval: 50` | ~2 updates/sec |
| External traffic | `traffic_generator_rate: 0`, `publish_interval: 100` | Depends on traffic |

**Example ESPHome syntax:**
```yaml
espectre:
  traffic_generator_rate: 100
  publish_interval: 50  # Optional: override publish rate
```

> **Note:** Lower `publish_interval` values increase periodic frontend-side
> telemetry, MQTT publish frequency, and periodic logs, but the internal motion
> detection cadence is controlled separately by `evaluation_interval`. BLE live
> telemetry for nearby clients is a separate subscription-driven path.

### Evaluation Interval (1-1000 packets)

**What it does:** Controls how often the detector state machine is evaluated
internally. This cadence feeds the binary sensor edge detection and the
`motion_on_hits` / `motion_off_hits` counters.

**Default:** `25`

**Example ESPHome syntax:**
```yaml
espectre:
  evaluation_interval: 25
  motion_on_hits: 3
  motion_off_hits: 3
```

**How to think about it:**
- Lower values react faster but evaluate more often
- Higher values are cheaper but add latency
- `3` hits with `evaluation_interval: 25` means roughly `75` packets of
  consistent evidence before changing state

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
- for the standalone `streamer`, use the streamer frontend README for practical
  collector-driven transport profiles and measured C3 throughput

</details>

---

## Adaptive Threshold

### Automatic Threshold Calibration

**What it does:** Automatically calculates the optimal detection threshold based on baseline noise characteristics. This minimizes false positives while maintaining high recall while keeping the fixed default subcarriers.

**Status:** Always enabled (automatic)

**How it works:**
1. During startup calibration, collects 10 × `window_size` CSI packets during baseline (default 1000, room must be quiet)
2. Uses the shared fixed 12-subcarrier set
3. **Threshold calculation** depends on `segmentation_threshold` setting

| Mode | Formula | Description |
|------|---------|-------------|
| `auto` (default) | P95 × 1.1 | Minimizes false positives |
| `min` | P100 × 1.0 | Maximum sensitivity |

**Note:** The subcarrier set is fixed. Only the threshold calculation varies.

**Example ESPHome syntax:**
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

**Applies to:** Both MVS and ML detectors.

**Default:** Enabled (threshold: 5.0 MAD, window: 7)

> The Hampel filter removes outlier spikes in turbulence values. With threshold 5.0 it only replaces extreme outliers (>5 MAD from median), preserving motion sensitivity while eliminating false positives caused by transient interference.

**Example ESPHome syntax:**
```yaml
espectre:
  hampel_enabled: true     # default
  hampel_window: 7         # sliding window size (3-11)
  hampel_threshold: 5.0    # MAD multiplier, higher = less aggressive
```

See the README of your frontend for the concrete configuration surface. The
ESPHome YAML mapping lives in
[`../src/cpp/frontend/esphome/README.md`](../src/cpp/frontend/esphome/README.md).

**When to disable:**
- If you observe reduced sensitivity in very low-SNR environments
- When maximum detection sensitivity is needed and the environment has no transient interference

---

## Low-Pass Filter

### Low-Pass Filter (Noise Reduction)

**What it does:** Removes high-frequency noise from turbulence values using a 1st-order Butterworth IIR filter. This significantly reduces false positives in noisy RF environments.

**Applies to:** Both MVS and ML detectors.

**Default:** Disabled

> ℹ️ **Note:** The low-pass filter is disabled by default for maximum simplicity. Enable it if you experience false positives in noisy RF environments.

**Example ESPHome syntax:**
```yaml
espectre:
  lowpass_enabled: true
  lowpass_cutoff: 11.0
```

See the README of your frontend for the concrete configuration surface. The
ESPHome YAML mapping lives in
[`../src/cpp/frontend/esphome/README.md`](../src/cpp/frontend/esphome/README.md).

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
| < 0.5m | > -30 dB | 0-15 | Poor | Too close, signal saturated |
| 0.5-2m | -30 to -40 dB | 15-30 | Marginal | May clip or reduce CSI quality |
| **3-8m** | -40 to -70 dB | **30-60** | **Optimal** | Best CSI quality and stability |
| 8-15m | -70 to -80 dB | 60-80 | Good | Still reliable detection |
| > 15m | < -80 dB | > 80 | Reduced quality | Weaker signal, more noise |

**Why distance matters:**

When the sensor is too close to the AP, the received signal is extremely strong, causing:
1. **AGC saturation**: The automatic gain control cannot reduce amplification enough
2. **CSI distortion**: Signal clipping leads to unreliable CSI data
3. **Reduced detector quality**: heavily compressed amplitudes can make baseline and motion less separable

**Symptoms of being too close:**
- weak motion contrast despite packet flow
- unusually high RSSI (> -40 dB)
- low AGC values in logs (< 30)
- more clipped or low-dynamic-range CSI amplitudes in captures

**Solution:**
1. Move the sensor 2-3 meters away from the AP
2. Re-run startup calibration in a quiet room after repositioning

**Checking your placement:**

Look at RSSI, packet flow, and readiness/calibration stability after Wi-Fi
connection:

- **RSSI between -40 and -70 dB**: usually a good operating range
- **AGC comfortably above saturation**: better headroom for clean CSI
- **steady packet flow and stable ready/calibration state**: healthy setup

---

## Troubleshooting

The troubleshooting logic below is frontend-neutral. Inline parameter examples
use `ESPHome` YAML only as a concrete surface example; apply the equivalent
control surface in your frontend.

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

**Symptoms:** Runtime telemetry or logs show no CSI data, stalled packet flow,
or an explicit "CSI disabled" warning.

**Solutions:**

1. **Verify WiFi connection:** Check the connection state exposed by your
   frontend, logs, or serial monitor

2. **Check traffic generator:**
   ```yaml
   espectre:
     traffic_generator_rate: 100  # Must be > 0
   ```

3. **Verify ESP-IDF configuration:** Ensure `CONFIG_ESP_WIFI_CSI_ENABLED: y` in sdkconfig

4. **Check router compatibility:** Some mesh routers or WiFi 6E may have issues

5. **If protocol/bandwidth logs show `unavailable`:** this can happen on some target/band mode API paths and does not automatically mean CSI is broken. Focus on CSI packet flow (`pps`, dropped packets, calibration progress) to assess runtime health.

### Calibration Stalls or Poor Startup Quality

**Symptoms:** The device keeps recalibrating, readiness never becomes stable, or
runtime quality is poor immediately after startup.

**Cause:** Usually the room is not quiet enough during calibration, packet flow
is too sparse, or the sensor is placed too close to the access point.

**Solutions:**

1. **Move the sensor further from the AP**:
   - Place at least 2-3 meters away
   - Optimal distance: 3-8 meters
   - Prefer RSSI roughly between -40 and -70 dB

2. **Keep the room quiet during startup**:
   - avoid movement in the monitored area
   - wait for calibration/readiness to complete before evaluating motion quality

3. **Verify CSI packet flow and stimulus configuration**:
   - keep `traffic_generator_rate` above `0` on integrated frontends
   - verify the external stimulus path when using the streamer frontend

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

### Runtime Recalibration (MVS only)

**When needed:** Recompute the adaptive threshold without reflashing (e.g., after moving furniture or changing room layout). This applies only to MVS mode; ML keeps its fixed threshold.

Use the recalibration control exposed by your frontend, when available.

Examples:

- `ESPHome`: trigger the `calibrate_switch` entity
- `Matter`: use the writable recalibration attribute on the vendor cluster
- other frontends: use the runtime control surface they expose, if supported

**Important:**
- Keep the room quiet and empty during calibration (~13 seconds)
- The frontend control may be temporarily unavailable during calibration
- You cannot cancel calibration once started

**Logs during recalibration:**
```
[I][espectre]: Manual recalibration triggered
[I][espectre]: Starting MVS threshold calibration with fixed subcarriers
[I][espectre]: Calibration completed successfully
```

### Reset Calibration (MVS only)

**When needed:** Start completely fresh and recompute the adaptive threshold from a quiet baseline. This applies only to MVS mode.

**How to reset:**

This step is frontend-specific. In general, perform a clean reinstall or erase
the persisted frontend state before booting again in a quiet room.

`ESPHome` example:

1. Erase flash completely:
   ```bash
   esphome run <your-config>.yaml --device /dev/ttyUSB0
   # Choose "Erase flash before uploading" if available
   ```

2. Or use the ESPHome dashboard clean-build flow and then reinstall

**After reset:**
- Keep room quiet and empty for 10 seconds
- MVS threshold bootstrap will automatically rerun
- Check logs for "Calibration completed successfully"

---

## Monitoring

### Observe the Runtime

Whatever frontend you use, monitor at least:

- motion state
- movement score or equivalent live metric
- current threshold
- startup calibration progress, when exposed

`ESPHome` example:

```bash
# Via USB
esphome logs <your-config>.yaml

# Via network (after first flash)
esphome logs <your-config>.yaml --device espectre.local
```

One `ESPHome` example is the Home Assistant entity set:

- `binary_sensor.espectre_motion_detected`
- `sensor.espectre_movement_score`
- `number.espectre_threshold`

Other frontends expose the same concepts through a different transport or
controller model. Use the runtime surface of your frontend even if the exact
entity names, commands, or telemetry channel differ.

---

## Quick Tips

1. **Start simple:** Tune only the segmentation threshold first
2. **One change at a time:** Adjust one parameter, apply it through your frontend surface, then test for 5-10 minutes
3. **Document your settings:** Note what works for your environment
4. **Seasonal adjustments:** Retune when furniture changes or new interference sources appear
5. **Distance matters:** Keep sensor 3-8m from router (RSSI between -40 and -70 dB for best results)
6. **Check AGC headroom:** After boot, moderate AGC values and stable calibration are good signs
7. **Quiet calibration:** Ensure no movement during first ~13 seconds after boot
8. **Try a BLE client:** Use a client built on the native frontend protocol over BLE, such as [`tools/web/espectre-ble.html`](../tools/web/espectre-ble.html) or [ESPectre - The Game](https://espectre.dev/game), for interactive threshold tuning with real-time visual feedback

---

## Additional Resources

- **Main Documentation:** [README.md](../README.md)
- **Setup Guide:** [SETUP.md](SETUP.md) for the shared frontend chooser
- **ESPHome Frontend:** [`../src/cpp/frontend/esphome/README.md`](../src/cpp/frontend/esphome/README.md) for YAML syntax, Home Assistant entities, and troubleshooting
- **Architecture:** [ARCHITECTURE.md](ARCHITECTURE.md) for the shared runtime and frontend split
- **Protocol:** [ESPECTRE_PROTOCOL.md](ESPECTRE_PROTOCOL.md) for BLE/MQTT message semantics
- **Performance:** [PERFORMANCE.md](PERFORMANCE.md) for current validation metrics and caveats
