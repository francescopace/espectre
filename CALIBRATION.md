# 🛜 ESPectre 👻 - Calibration & Tuning Guide

This guide will help you calibrate and optimize ESPectre for your specific environment to achieve reliable movement detection.

---

## 📋 Table of Contents

- [Using the CLI Tool](#️-using-the-cli-tool)
- [Quick Start Calibration](#-quick-start-calibration-5-10-minutes)
- [Understanding Your Environment](#-understanding-your-environment)
- [Step-by-Step Calibration Process](#-step-by-step-calibration-process)
- [Advanced Tuning](#-advanced-tuning)
- [Troubleshooting Scenarios](#-troubleshooting-scenarios)
- [Use Case Examples](#-use-case-examples)

---

## 🛠️ Using the CLI Tool

The `espectre-cli.sh` script is a convenient wrapper around MQTT commands that simplifies calibration and tuning of your ESPectre sensor.

**ESPectre is designed for hot-tuning**: All parameters can be adjusted in real-time via MQTT commands without requiring a rebuild, reflash, or device restart. This allows you to fine-tune detection settings while the sensor is running and immediately see the results.

**How it works:**
- Sends commands to the MQTT topic `<your_topic>/cmd`
- Listens for responses on `<your_topic>/response`
- Formats JSON payloads automatically
- Handles timeouts and error messages

**Basic usage:**
```bash
# Get current configuration
./espectre-cli.sh info

# Analyze data and get recommended threshold
./espectre-cli.sh analyze

# Set detection threshold
./espectre-cli.sh threshold 0.35

# Listen to device responses
./espectre-cli.sh listen
```

**Behind the scenes**, the CLI uses `mosquitto_pub` and `mosquitto_sub` to communicate with your MQTT broker. For example:

```bash
./espectre-cli.sh threshold 0.35
```

Is equivalent to:
```bash
mosquitto_pub -h homeassistant.local -t "home/espectre/node1/cmd" \
  -m '{"cmd": "threshold", "value": 0.35}'
```

**Configuration:**

The CLI reads connection settings from environment variables:
- `MQTT_BROKER` - Broker hostname (default: homeassistant.local)
- `MQTT_PORT` - Broker port (default: 1883)
- `MQTT_TOPIC` - Base topic (default: home/espectre/node1)
- `MQTT_USERNAME` - Username (optional)
- `MQTT_PASSWORD` - Password (optional)

**Example with custom settings:**
```bash
export MQTT_BROKER="192.168.1.100"
export MQTT_TOPIC="home/espectre/bedroom"
./espectre-cli.sh analyze
```

**Listening to device responses:**

The `listen` command allows you to monitor all responses from the device in real-time. This is useful for debugging and seeing the results of commands you send.

```bash
# In a separate terminal, start listening to responses
./espectre-cli.sh listen
```

This will display all messages published by the device to the `<your_topic>/response` topic. Keep this running in a separate shell while you send commands from another terminal to see the device's responses immediately.

**Example workflow:**
```bash
# Terminal 1: Listen to responses
./espectre-cli.sh listen

# Terminal 2: Send commands
./espectre-cli.sh threshold 0.35
./espectre-cli.sh info
./espectre-cli.sh stats
```

Run `./espectre-cli.sh help` to see all available commands.

---

## 🚀 Quick Start Calibration (5-10 minutes)

This is the fastest way to get ESPectre working in your environment.

### Prerequisites

- ESPectre device installed and running
- MQTT broker accessible
- `espectre-cli.sh` script available

### Steps

1. **Position the sensor** at 3-8 meters from your Wi-Fi router at desk/table height (1-1.5m)

2. **Let it collect baseline data** for 1-2 minutes with the room empty and still

3. **Run analysis** to get recommended threshold:
   ```bash
   ./espectre-cli.sh analyze
   ```

4. **Apply the recommended threshold**:
   ```bash
   # Example: if analyze suggests 0.35
   ./espectre-cli.sh threshold 0.35
   ```

5. **Test detection** by walking around the room while observing the serial monitor:
   ```bash
   idf.py monitor
   ```
   You should see `🚶 MOVEMENT DETECTED` logs when moving and `✋ MOVEMENT STOPPED` when still.

6. **Fine-tune if needed**:
   - Too sensitive (false positives)? → Increase threshold by 0.05
   - Not sensitive enough (missed movements)? → Decrease threshold by 0.05

**Done!** Your sensor is now calibrated for basic operation.

---

## 🏠 Understanding Your Environment

ESPectre's performance depends heavily on your specific environment. Understanding these factors will help you optimize detection.

### Key Environmental Factors

#### 1. **Distance from Router**
| Distance | Signal Strength | Multipath | Sensitivity | Recommendation |
|----------|----------------|-----------|-------------|----------------|
| < 2m | Too strong | Minimal | Low | ❌ Too close |
| 3-8m | Strong | Good | High | ✅ **Optimal** |
| > 10-15m | Weak | Variable | Low | ❌ Too far |

#### 2. **Room Characteristics**

**Room Size:**
- Small rooms (< 20m²): Lower threshold (0.25-0.35)
- Medium rooms (20-50m²): Standard threshold (0.35-0.45)
- Large rooms (> 50m²): Higher threshold (0.45-0.60)

**Wall Materials:**
- Drywall: Good signal penetration
- Concrete: Reduced sensitivity, may need lower threshold
- Metal/reinforced concrete: Significantly reduced range

**Furniture & Objects:**
- More furniture = more multipath reflections = better detection
- Empty rooms may need lower threshold
- Metal objects can create interference

#### 3. **Interference Sources**

**High interference** (may cause false positives):
- Fans, air conditioners (moving air)
- Microwave ovens (2.4GHz interference)
- Other 2.4GHz devices (Bluetooth, wireless cameras)
- Moving curtains near windows
- Pets

**Solutions:**
- Increase `debounce` count (3-5)
- Increase `threshold` slightly
- Enable Hampel filter to remove outliers

#### 4. **Wi-Fi Traffic**

- Low traffic networks: More stable CSI data
- High traffic networks: More noise, may need filtering
- Mesh networks: Work fine, ensure 2.4GHz connection

---

## 📊 Step-by-Step Calibration Process

### Phase 1: Initial Setup (5 minutes)

1. **Position the sensor** following the distance guidelines above

2. **Verify Wi-Fi connection**:
   ```bash
   # Check serial output
   idf.py monitor
   # Look for: "WiFi connected, got IP: xxx.xxx.xxx.xxx"
   ```

3. **Verify MQTT connection**:
   ```bash
   mosquitto_sub -h homeassistant.local -t "home/espectre/node1" -v
   # You should see JSON messages every 1-5 seconds
   ```

### Phase 2: Baseline Collection (2-3 minutes)

1. **Ensure the room is empty and still**:
   - No people moving
   - No fans or air conditioning
   - Close doors/windows to prevent drafts

2. **Start serial monitor** to observe real-time CSI data:
   ```bash
   idf.py monitor
   ```
   This shows live detection logs with immediate feedback (no delays from MQTT optimizations).

3. **Observe the baseline values** for 1-2 minutes:
   - Look for `📊 CSI: movement=` logs
   - `movement` should be low (< 0.20)
   - `state` should be "idle"
   - Note any spikes or variations

### Phase 3: Analysis & Threshold Setting (2 minutes)

1. **Run statistical analysis**:
   ```bash
   ./espectre-cli.sh analyze
   ```

   **Example output:**
   ```json
   {
     "min": 0.02,
     "max": 0.18,
     "avg": 0.08,
     "stddev": 0.03,
     "p25": 0.06,
     "p50_median": 0.08,
     "p75": 0.11,
     "p95": 0.15,
     "recommended_threshold": 0.35,
     "current_threshold": 0.40
   }
   ```

2. **Interpret the results**:
   - `p50_median`: Typical idle value
   - `p95`: Maximum idle value (95th percentile)
   - `recommended_threshold`: Suggested value (median between p50 and p75)

3. **Apply the recommended threshold**:
   ```bash
   ./espectre-cli.sh threshold 0.35
   ```

### Phase 4: Movement Testing (5 minutes)

1. **Test with different movement types** while observing the serial monitor:

   Keep `idf.py monitor` running to see real-time detection logs.

   **Slow walking:**
   - Walk slowly across the room
   - Expected in logs: `movement=0.40-0.60`, `state=detected`

   **Normal walking:**
   - Walk at normal pace
   - Expected in logs: `movement=0.60-0.80`, `state=detected`

   **Fast movement:**
   - Walk quickly or wave arms
   - Expected in logs: `movement=0.80-1.00`, `state=detected`

2. **Test detection persistence**:
   - Move, then stand completely still
   - Detection should persist for ~3 seconds (default)
   - Then return to "idle"

3. **Adjust if needed**:

   **Too many false positives:**
   ```bash
   # Increase threshold
   ./espectre-cli.sh threshold 0.45
   
   # Or increase debounce
   ./espectre-cli.sh debounce 4
   ```

   **Missing movements:**
   ```bash
   # Decrease threshold
   ./espectre-cli.sh threshold 0.30
   
   # Or decrease debounce
   ./espectre-cli.sh debounce 2
   ```

### Phase 5: Verification (3 minutes)

1. **Run a complete test cycle**:
   - Leave room empty for 30 seconds → should be "idle"
   - Enter and move around → should detect within 1-2 seconds
   - Stand still → should return to "idle" after 3 seconds
   - Repeat 3-5 times

2. **Check statistics**:
   ```bash
   ./espectre-cli.sh stats
   ```

3. **Save your configuration** (note down the values):
   ```bash
   ./espectre-cli.sh info
   ```

---

## 🔧 Advanced Tuning

Once basic calibration is complete, you can fine-tune advanced parameters for optimal performance.

### Detection Parameters

#### Threshold
**What it does:** Minimum score needed to trigger detection

**Default:** 0.40

**When to adjust:**
- **Lower (0.25-0.35):** Small rooms, need high sensitivity
- **Higher (0.45-0.60):** Large rooms, reduce false positives

**Command:**
```bash
./espectre-cli.sh threshold 0.35
```

#### Debounce Count
**What it does:** Number of consecutive detections needed before triggering

**Default:** 3

**When to adjust:**
- **Lower (1-2):** Faster response, more false positives
- **Higher (4-5):** Slower response, fewer false positives

**Command:**
```bash
./espectre-cli.sh debounce 3
```

#### Persistence Timeout
**What it does:** Seconds to wait before returning to IDLE after last detection

**Default:** 3 seconds

**When to adjust:**
- **Lower (1-2s):** Quick response to absence
- **Higher (5-10s):** Avoid flickering in/out of detection

**Command:**
```bash
./espectre-cli.sh persistence 5
```

#### Hysteresis Ratio
**What it does:** Ratio for lower threshold (prevents rapid state flipping)

**Default:** 0.7 (lower threshold = 0.7 × upper threshold)

**When to adjust:**
- **Lower (0.5-0.6):** More stable, slower to return to idle
- **Higher (0.8-0.9):** Faster response, may flicker

**Command:**
```bash
./espectre-cli.sh hysteresis 0.7
```

### Signal Processing

#### Variance Scale
**What it does:** Normalization scale for variance feature (affects overall sensitivity)

**Default:** 400

**When to adjust:**
- **Lower (200-300):** Higher sensitivity (more responsive)
- **Higher (500-800):** Lower sensitivity (fewer false positives)

**Command:**
```bash
./espectre-cli.sh variance_scale 400
```

#### Hampel Filter (Outlier Removal)
**What it does:** Removes statistical outliers using Median Absolute Deviation

**Default:** Enabled, threshold 2.0

**When to use:**
- Enable in high-interference environments
- Helps reduce false positives from electrical noise

**Commands:**
```bash
# Enable/disable
./espectre-cli.sh hampel_filter on

# Adjust sensitivity (1.0-10.0, lower = more aggressive)
./espectre-cli.sh hampel_threshold 3.0
```

#### Savitzky-Golay Filter (Smoothing)
**What it does:** Polynomial smoothing to reduce noise

**Default:** Enabled

**When to use:**
- Keep enabled for most scenarios
- Disable only if you need raw, unfiltered data

**Command:**
```bash
./espectre-cli.sh savgol_filter on
```

#### Check Filter Status
```bash
./espectre-cli.sh filters
```

### Granular States (4-State Detection)

**Default mode:** 2 states (IDLE, DETECTED)

**Granular mode:** 4 states (IDLE, MICRO, DETECTED, INTENSE)

**When to use:**
- Distinguish between small movements (micro) and large movements (intense)
- Useful for activity classification
- May help distinguish pets from humans

**Thresholds:**
- IDLE: score < 0.10
- MICRO: score 0.10-0.50
- DETECTED: score 0.50-0.70
- INTENSE: score > 0.70

**Command:**
```bash
./espectre-cli.sh granular_states on
```

### Feature Weights (Expert Level)

ESPectre uses 4 key features for detection scoring. You can adjust their weights to optimize for your environment.

**Default weights:**
- Variance: 25%
- Spatial Gradient: 25%
- Variance Short: 35%
- IQR: 15%

**View current weights:**
```bash
./espectre-cli.sh weights
```

**Adjust individual weights:**
```bash
# Increase sensitivity to rapid changes
./espectre-cli.sh weight_variance_short 0.40

# Increase sensitivity to spatial patterns
./espectre-cli.sh weight_spatial_gradient 0.30

# Adjust other weights to maintain sum ≈ 1.0
./espectre-cli.sh weight_variance 0.20
./espectre-cli.sh weight_iqr 0.10
```

**Guidelines:**
- Weights should sum to approximately 1.0
- `variance_short` is most sensitive to movement
- `spatial_gradient` captures multipath changes
- Experiment in small increments (±0.05)

### View All Features

To see all 15 extracted features in real-time:
```bash
./espectre-cli.sh features
```

This shows:
- **Time-domain** (6): mean, variance, skewness, kurtosis, entropy, IQR
- **Spatial** (3): variance, correlation, gradient
- **Temporal** (3): autocorrelation, zero-crossing rate, peak rate
- **Multi-window** (3): variance at short/medium/long time scales

---

## 🔍 Troubleshooting Scenarios

### Scenario 1: Too Many False Positives

**Symptoms:**
- Detects movement when room is empty
- Frequent state changes without actual movement
- High `movement` values during idle periods

**Solutions:**

1. **Increase threshold:**
   ```bash
   ./espectre-cli.sh threshold 0.50
   ```

2. **Increase debounce:**
   ```bash
   ./espectre-cli.sh debounce 4
   ```

3. **Enable/tune Hampel filter:**
   ```bash
   ./espectre-cli.sh hampel_filter on
   ./espectre-cli.sh hampel_threshold 2.5
   ```

4. **Check for interference sources:**
   - Move sensor away from fans, AC units
   - Check for moving curtains or plants
   - Verify no pets in the room

5. **Increase variance scale (reduce sensitivity):**
   ```bash
   ./espectre-cli.sh variance_scale 600
   ```

### Scenario 2: Missing Movements

**Symptoms:**
- Doesn't detect when people move
- `movement` values stay low even during activity
- State remains "idle" during movement

**Solutions:**

1. **Decrease threshold:**
   ```bash
   ./espectre-cli.sh threshold 0.30
   ```

2. **Check sensor position:**
   - Verify distance from router (3-8m optimal)
   - Ensure external antenna is connected
   - Try different height (1-1.5m)

3. **Decrease variance scale (increase sensitivity):**
   ```bash
   ./espectre-cli.sh variance_scale 300
   ```

4. **Verify Wi-Fi signal strength:**
   ```bash
   idf.py monitor
   # Look for RSSI values > -70 dBm
   ```

5. **Reduce debounce for faster response:**
   ```bash
   ./espectre-cli.sh debounce 2
   ```

### Scenario 3: Unstable Detection

**Symptoms:**
- Rapid flickering between "idle" and "detected"
- Inconsistent detection of same movements
- High variance in `movement` values

**Solutions:**

1. **Adjust hysteresis:**
   ```bash
   ./espectre-cli.sh hysteresis 0.6
   ```

2. **Increase persistence timeout:**
   ```bash
   ./espectre-cli.sh persistence 5
   ```

3. **Enable smoothing filters:**
   ```bash
   ./espectre-cli.sh savgol_filter on
   ./espectre-cli.sh hampel_filter on
   ```

4. **Check Wi-Fi stability:**
   - Verify router isn't overloaded
   - Check for mesh network handoffs
   - Ensure stable 2.4GHz connection

### Scenario 4: Slow Response Time

**Symptoms:**
- Takes several seconds to detect movement
- Delayed return to idle state

**Solutions:**

1. **Reduce debounce count:**
   ```bash
   ./espectre-cli.sh debounce 2
   ```

2. **Reduce persistence timeout:**
   ```bash
   ./espectre-cli.sh persistence 2
   ```

3. **Lower threshold slightly:**
   ```bash
   ./espectre-cli.sh threshold 0.35
   ```

### Scenario 5: Can't Distinguish Pets from Humans

**Symptoms:**
- Pet movements trigger detection
- Want to ignore small animals

**Solutions:**

1. **Enable granular states:**
   ```bash
   ./espectre-cli.sh granular_states on
   ```

2. **Increase threshold to ignore MICRO movements:**
   ```bash
   ./espectre-cli.sh threshold 0.50
   ```

3. **Use Home Assistant automation to filter:**
   ```yaml
   # Only trigger on DETECTED or INTENSE states
   condition:
     - condition: template
       value_template: "{{ state_attr('sensor.movement_sensor', 'state') in ['detected', 'intense'] }}"
   ```

4. **Adjust spatial gradient weight** (pets create less spatial variation):
   ```bash
   ./espectre-cli.sh weight_spatial_gradient 0.35
   ./espectre-cli.sh weight_variance_short 0.30
   ```

---

## 📝 Use Case Examples

### Use Case 1: Small Bedroom (15m²)

**Environment:**
- Room size: 3m × 5m
- Distance from router: 5m
- Minimal furniture
- Need high sensitivity

**Optimal Configuration:**
```bash
./espectre-cli.sh threshold 0.30
./espectre-cli.sh debounce 2
./espectre-cli.sh persistence 3
./espectre-cli.sh variance_scale 300
./espectre-cli.sh hampel_filter on
```

**Expected Performance:**
- Detects person entering within 1 second
- Minimal false positives
- Good for presence detection

### Use Case 2: Large Living Room (50m²)

**Environment:**
- Room size: 7m × 7m
- Distance from router: 8m
- Lots of furniture (multipath)
- Some interference (TV, AC)

**Optimal Configuration:**
```bash
./espectre-cli.sh threshold 0.45
./espectre-cli.sh debounce 3
./espectre-cli.sh persistence 5
./espectre-cli.sh variance_scale 500
./espectre-cli.sh hampel_filter on
./espectre-cli.sh hampel_threshold 3.0
```

**Expected Performance:**
- Reliable detection across entire room
- Reduced false positives from interference
- Stable state transitions

### Use Case 3: Office with Multiple People

**Environment:**
- Open office space
- Multiple people moving
- Need to detect any activity
- High Wi-Fi traffic

**Optimal Configuration:**
```bash
./espectre-cli.sh threshold 0.35
./espectre-cli.sh debounce 2
./espectre-cli.sh persistence 10
./espectre-cli.sh granular_states on
./espectre-cli.sh hampel_filter on
./espectre-cli.sh savgol_filter on
```

**Expected Performance:**
- Detects any movement quickly
- Long persistence keeps "detected" during brief pauses
- Granular states show activity intensity

### Use Case 4: Hallway/Corridor

**Environment:**
- Narrow space (1.5m × 8m)
- People pass through quickly
- Need fast detection and release

**Optimal Configuration:**
```bash
./espectre-cli.sh threshold 0.40
./espectre-cli.sh debounce 2
./espectre-cli.sh persistence 2
./espectre-cli.sh hysteresis 0.8
./espectre-cli.sh variance_scale 400
```

**Expected Performance:**
- Quick detection when entering
- Fast return to idle after passing
- Minimal false positives

### Use Case 5: Elderly Care Monitoring

**Environment:**
- Bedroom or living area
- Need to detect falls or prolonged inactivity
- High reliability required

**Optimal Configuration:**
```bash
./espectre-cli.sh threshold 0.25
./espectre-cli.sh debounce 2
./espectre-cli.sh persistence 3
./espectre-cli.sh granular_states on
./espectre-cli.sh variance_scale 300
./espectre-cli.sh hampel_filter on
```

**Home Assistant Automation Example:**
```yaml
automation:
  - alias: "Fall Detection Alert"
    trigger:
      - platform: state
        entity_id: sensor.movement_sensor
        to: "intense"
    action:
      - service: notify.mobile_app
        data:
          message: "Sudden intense movement detected - possible fall"
          
  - alias: "Inactivity Alert"
    trigger:
      - platform: state
        entity_id: sensor.movement_sensor
        to: "idle"
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

**Expected Performance:**
- High sensitivity to all movements
- Granular states help distinguish normal vs. unusual activity
- Reliable for safety monitoring

---

## 🎓 Best Practices

### 1. Calibration Workflow

Always follow this sequence:
1. Position sensor optimally
2. Collect baseline (empty room)
3. Run `analyze` command
4. Apply recommended threshold
5. Test with movement
6. Fine-tune if needed
7. Document your settings

### 2. Iterative Tuning

Make changes incrementally:
- Adjust one parameter at a time
- Test for 5-10 minutes after each change
- Document what works and what doesn't
- Keep notes on your environment

### 3. Seasonal Adjustments

You may need to recalibrate when:
- Furniture is rearranged
- Seasonal changes (heating/cooling)
- New interference sources added
- Router position changes

### 4. Multiple Sensors

If using multiple sensors:
- Calibrate each independently
- Different rooms may need different settings
- Use descriptive MQTT topics (e.g., `home/espectre/bedroom`)
- Document settings per sensor

### 5. Monitoring & Maintenance

Regular checks:
```bash
# Weekly: Check statistics
./espectre-cli.sh stats

# Monthly: Re-run analysis
./espectre-cli.sh analyze

# As needed: View current config
./espectre-cli.sh info
```

---

## 📚 Additional Resources

- **Main Documentation:** [README.md](README.md)
- **Setup Guide:** [SETUP.md](SETUP.md)
- **GitHub Issues:** [Report problems or ask questions](https://github.com/francescopace/espectre/issues)
- **Email Support:** francesco.pace@gmail.com

---

## 🤝 Contributing

Found a configuration that works great for your environment? Share it!

Open a GitHub issue or pull request with:
- Your environment description
- Optimal configuration values
- Performance notes

Help others benefit from your experience!
