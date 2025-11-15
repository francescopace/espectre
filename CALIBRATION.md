# 🛜 ESPectre 👻 - Tuning Guide

This guide will help you tune and optimize ESPectre for your specific environment to achieve reliable movement detection.

---

## 📋 Table of Contents

- [Using the CLI Tool](#️-using-the-cli-tool)
- [Understanding Your Environment](#-understanding-your-environment)
- [Manual Tuning](#-manual-tuning)
- [Advanced Configuration](#-advanced-configuration)
- [Troubleshooting Scenarios](#-troubleshooting-scenarios)
- [Use Case Examples](#-use-case-examples)

---

## 🛠️ Using the CLI Tool

---

The `espectre-cli.sh` script provides an **interactive interface** for tuning your ESPectre sensor in real-time.

**ESPectre is designed for hot-tuning**: All parameters can be adjusted in real-time via MQTT commands without requiring a rebuild, reflash, or device restart. This allows you to fine-tune detection settings while the sensor is running and immediately see the results.

### Interactive Mode (Recommended)

**Launch the CLI:**
```bash
./espectre-cli.sh
```

This starts an **interactive session** with:
- ✅ **Continuous listening** to device responses in the background
- ✅ **Command prompt** for sending commands
- ✅ **Real-time feedback** - responses appear automatically
- ✅ **No need for multiple terminals** - everything in one place

**Example interactive session:**
```bash
$ ./espectre-cli.sh

╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║                   🛜  E S P e c t r e 👻                  ║
║                                                           ║
║                Wi-Fi motion detection system              ║
║          based on Channel State Information (CSI)         ║
║                                                           ║
╠═══════════════════════════════════════════════════════════╣
║                                                           ║
║                   Interactive CLI Mode                    ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝

Connected to: homeassistant.local:1883
Command topic: home/espectre/node1/cmd
Listening on: home/espectre/node1/response

Type 'help' for commands, 'exit' to quit

espectre> info
20:05:12 {"network": {"ip_address": "192.168.1.100"}, ...}

espectre> segmentation_threshold 0.35
20:05:18 {"response": "Segmentation threshold set to 0.35"}

espectre> help
[shows all available commands]

espectre> exit
Shutting down...
```

**Available commands in interactive mode:**
- Type `help` to see all commands
- Use shortcuts where available
- Type `exit` or `quit` to close the CLI

### Configuration

The CLI reads connection settings from environment variables:
- `MQTT_BROKER` - Broker hostname (default: homeassistant.local)
- `MQTT_PORT` - Broker port (default: 1883)
- `MQTT_TOPIC` - Base topic (default: home/espectre/node1)
- `MQTT_USERNAME` - Username (default: mqtt)
- `MQTT_PASSWORD` - Password (default: mqtt)

**Example with custom settings:**
```bash
export MQTT_BROKER="192.168.1.100"
export MQTT_TOPIC="home/espectre/bedroom"
./espectre-cli.sh
```

**Note:** For scripting or automation, you can also send commands directly via MQTT. See [SETUP.md](SETUP.md) for MQTT command reference.

---

## 🏠 Understanding Your Environment

---

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
- Increase `segmentation_threshold`
- Enable Hampel filter to remove outliers
- Adjust filter settings

#### 4. **Wi-Fi Traffic**

- Low traffic networks: More stable CSI data
- High traffic networks: More noise, may need filtering
- Mesh networks: Work fine, ensure 2.4GHz connection

---

## 📊 Manual Tuning

---

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

### Phase 2: Baseline Observation (2-3 minutes)

1. **Ensure the room is empty and still**:
   - No people moving
   - No fans or air conditioning
   - Close doors/windows to prevent drafts

2. **Start serial monitor** to observe real-time CSI data:
   ```bash
   idf.py monitor
   ```
   This shows live detection logs with immediate feedback.

3. **Observe the baseline values** for 1-2 minutes:
   - Look for `📊 CSI:` logs showing moving_variance and state
   - `state` should be "IDLE"
   - Note the typical `moving_variance` values
   - Note any spikes or variations

4. **Check runtime statistics** (in interactive CLI):
   ```
   espectre> stats
   ```
   This shows current turbulence, moving_variance, state, and segment information.

### Phase 3: Threshold Adjustment (2 minutes)

**Start the interactive CLI:**
```bash
./espectre-cli.sh
```

**In the interactive session:**

1. **Check current configuration**:
   ```
   espectre> info
   ```
   Look at the `segmentation` section to see current threshold and state.

2. **Adjust segmentation threshold based on observations**:
   - If baseline moving_variance is typically 0.05-0.10, set threshold to 0.30-0.40
   - If baseline is higher (0.15-0.20), increase threshold to 0.45-0.55
   
   ```
   espectre> segmentation_threshold 0.35
   ```

### Phase 4: Movement Testing (5 minutes)

1. **Test with different movement types** while observing the serial monitor:

   Keep `idf.py monitor` running to see real-time detection logs.

   **Slow walking:**
   - Walk slowly across the room
   - Expected: `state=MOTION`, moving_variance increases

   **Normal walking:**
   - Walk at normal pace
   - Expected: `state=MOTION`, higher moving_variance

   **Fast movement:**
   - Walk quickly or wave arms
   - Expected: `state=MOTION`, highest moving_variance

2. **Test state transitions**:
   - Move, then stand completely still
   - State should transition from MOTION → IDLE
   - Observe how quickly it transitions

3. **Adjust if needed** (in the interactive CLI session):

   **Too many false positives (MOTION when room is empty):**
   ```
   espectre> segmentation_threshold 0.45
   ```

   **Missing movements (stays IDLE during activity):**
   ```
   espectre> segmentation_threshold 0.30
   ```

### Phase 5: Verification (3 minutes)

1. **Run a complete test cycle**:
   - Leave room empty for 30 seconds → should be "IDLE"
   - Enter and move around → should detect within 1-2 seconds
   - Stand still → should return to "IDLE" after movement stops
   - Repeat 3-5 times

2. **Check final statistics** (in interactive CLI):
   ```
   espectre> stats
   ```
   Verify stable detection with good segment metrics.

3. **Save your configuration** (note down the values in interactive CLI):
   ```
   espectre> info
   ```

---

## 🔧 Advanced Configuration

---

Once basic tuning is complete, you can fine-tune advanced parameters for optimal performance.

### Segmentation Parameters

#### Segmentation Threshold
**What it does:** Adaptive threshold multiplier for motion detection (k-factor in MVS algorithm)

**Default:** 0.40

**How it works:** The system calculates `adaptive_threshold = k_factor × moving_variance`. Motion is detected when current variance exceeds this adaptive threshold.

**When to adjust:**
- **Lower (0.25-0.35):** Small rooms, need high sensitivity
- **Higher (0.45-0.60):** Large rooms, reduce false positives

**Interactive CLI:**
```
espectre> segmentation_threshold 0.35
```

### Feature Extraction

#### Features Enable/Disable
**What it does:** Toggle feature extraction during MOTION state

**Default:** Enabled

**When to disable:**
- Only need basic motion detection (IDLE/MOTION states)
- Want to reduce CPU usage
- Don't need detailed feature analysis

**Interactive CLI:**
```
# Disable features (motion detection only)
espectre> features_enable off

# Re-enable features
espectre> features_enable on
```

### Signal Processing Filters

**Important:** Filters are applied **only to feature extraction**, not to segmentation. Segmentation uses raw CSI data to preserve motion sensitivity.

#### Hampel Filter (Outlier Removal)
**What it does:** Removes statistical outliers using Median Absolute Deviation

**Default:** Disabled, threshold 2.0

**When to use:**
- Enable in high-interference environments
- Helps reduce noise in extracted features

**Interactive CLI:**
```
# Enable/disable
espectre> hampel_filter on

# Adjust sensitivity (1.0-10.0, lower = more aggressive)
espectre> hampel_threshold 3.0
```

#### Butterworth Low-Pass Filter
**What it does:** Removes high-frequency noise (>8Hz) from features - human movement is 0.5-8Hz

**Default:** Enabled (recommended to keep on)

**When to use:**
- **Always recommended** for feature extraction
- Filters out environmental noise in features
- Based on scientific papers (Lolla, CSI-HC, Dong et al.)

**Performance impact:**
- CPU: Minimal (~5-10 μs per packet)
- Benefit: Cleaner feature values

**Interactive CLI:**
```
espectre> butterworth_filter on
```

#### Wavelet Filter (Advanced - Daubechies db4)
**What it does:** Removes low-frequency persistent noise from features using wavelet transform

**Default:** Disabled (enable only for high-noise environments)

**When to enable:**
- ✅ High noise in feature values
- ✅ Persistent low-frequency noise that Butterworth can't remove
- ✅ Poor feature quality despite tuning

**Parameters:**
- **Level** (1-3): Decomposition depth (3 = most aggressive)
- **Threshold** (0.5-2.0): Noise removal intensity (1.0 = balanced)

**Interactive CLI:**
```
# Enable wavelet
espectre> wavelet_filter on

# Set maximum denoising (recommended for high noise)
espectre> wavelet_level 3

# Set balanced threshold
espectre> wavelet_threshold 1.0
```

**Performance impact:**
- CPU: ~5-8% additional load
- RAM: ~2 KB
- Latency: 320ms warm-up (32 samples)
- Benefit: Cleaner features in high-noise environments

**Based on research:**
- "Location Intelligence System" (2022) - ESP32 + wavelet db4
- "CSI-HC" (2020) - Butterworth + Sym8 combination

#### Savitzky-Golay Filter (Smoothing)
**What it does:** Polynomial smoothing to reduce noise in features

**Default:** Enabled

**When to use:**
- Keep enabled for most scenarios
- Provides smooth feature values

**Interactive CLI:**
```
espectre> savgol_filter on
```

#### Check Filter Status

**Interactive CLI:**
```
espectre> info
```

Look at the `filters` section in the JSON response.

---

## 🔍 Troubleshooting Scenarios

---

### Scenario 1: Complete Reset Needed

**Symptoms:**
- Configuration is corrupted or inconsistent
- Want to start fresh with default settings
- Testing different configurations

**Solution:**

Use the factory reset command to restore all defaults.

**Via Interactive CLI:**

Launch the CLI first:
```bash
./espectre-cli.sh
```

Then execute:
```
espectre> factory_reset
```

**Via MQTT:**
```bash
mosquitto_pub -h homeassistant.local -t "home/espectre/node1/cmd" \
  -m '{"cmd":"factory_reset"}'
```

**This will:**
- ✅ Clear all NVS storage (configuration)
- ✅ Restore all parameters to factory defaults
- ⚠️ You will need to retune after reset

### Scenario 2: Too Many False Positives

**Symptoms:**
- Detects movement when room is empty (state = MOTION)
- Frequent state changes without actual movement
- High `moving_variance` values during idle periods

**Diagnosis (in interactive CLI):**
```
espectre> stats
```
Check `turbulence` and `moving_variance` values during idle periods.

**Solutions (in interactive CLI):**

1. **Increase segmentation threshold:**
   ```
   espectre> segmentation_threshold 0.50
   ```

2. **Enable/tune Hampel filter (for features):**
   ```
   espectre> hampel_filter on
   espectre> hampel_threshold 2.5
   ```

3. **Check for interference sources:**
   - Move sensor away from fans, AC units
   - Check for moving curtains or plants
   - Verify no pets in the room

### Scenario 3: Missing Movements

**Symptoms:**
- Doesn't detect when people move (stays IDLE)
- `moving_variance` values stay low even during activity
- State remains "IDLE" during movement

**Diagnosis (in interactive CLI):**
```
espectre> stats
```
Check if `turbulence` and `moving_variance` increase during movement.

**Solutions (in interactive CLI):**

1. **Decrease segmentation threshold:**
   ```
   espectre> segmentation_threshold 0.30
   ```

2. **Check sensor position:**
   - Verify distance from router (3-8m optimal)
   - Ensure external antenna is connected
   - Try different height (1-1.5m)

3. **Verify Wi-Fi signal strength:**
   ```bash
   idf.py monitor
   # Look for RSSI values > -70 dBm
   ```

4. **Check traffic generator:**
   ```
   espectre> info
   ```
   Verify `traffic_generator_rate` is set (recommended: 15-20 pps)

### Scenario 4: Unstable Detection

**Symptoms:**
- Rapid flickering between "IDLE" and "MOTION"
- Inconsistent detection of same movements
- High variance in `moving_variance` values

**Diagnosis (in interactive CLI):**
```
espectre> stats
```
Monitor `state` transitions and `moving_variance` stability.

**Solutions (in interactive CLI):**

1. **Adjust segmentation threshold:**
   ```
   espectre> segmentation_threshold 0.40
   ```

2. **Enable smoothing filters (for features):**
   ```
   espectre> savgol_filter on
   espectre> hampel_filter on
   ```

3. **Check Wi-Fi stability:**
   - Verify router isn't overloaded
   - Check for mesh network handoffs
   - Ensure stable 2.4GHz connection

### Scenario 5: Feature Quality Issues

**Symptoms:**
- Noisy feature values (if features are enabled)
- Inconsistent feature readings

**Diagnosis (in interactive CLI):**
```
espectre> stats
```
Check segment quality metrics (`avg_turbulence`, `max_turbulence`).

**Solutions (in interactive CLI):**

1. **Enable Butterworth filter:**
   ```
   espectre> butterworth_filter on
   ```

2. **Enable Savitzky-Golay smoothing:**
   ```
   espectre> savgol_filter on
   ```

3. **For high noise, enable Wavelet filter:**
   ```
   espectre> wavelet_filter on
   espectre> wavelet_level 3
   espectre> wavelet_threshold 1.0
   ```

---

## 📝 Use Case Examples

---

**Note:** The configuration examples below show commands as they would be entered in the **interactive CLI session**. Launch the CLI with `./espectre-cli.sh` first, then enter the commands at the `espectre>` prompt.

### Use Case 1: Small Bedroom (15m²)

**Environment:**
- Room size: 3m × 5m
- Distance from router: 5m
- Minimal furniture
- Need high sensitivity

**Optimal Configuration (in interactive CLI):**
```
espectre> segmentation_threshold 0.30
espectre> features_enable on
espectre> hampel_filter on
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

**Optimal Configuration (in interactive CLI):**
```
espectre> segmentation_threshold 0.45
espectre> features_enable on
espectre> hampel_filter on
espectre> hampel_threshold 3.0
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

**Optimal Configuration (in interactive CLI):**
```
espectre> segmentation_threshold 0.35
espectre> features_enable on
espectre> hampel_filter on
espectre> savgol_filter on
```

**Expected Performance:**
- Detects any movement quickly
- Stable detection during activity

### Use Case 4: Hallway/Corridor

**Environment:**
- Narrow space (1.5m × 8m)
- People pass through quickly
- Need fast detection and release

**Optimal Configuration (in interactive CLI):**
```
espectre> segmentation_threshold 0.40
espectre> features_enable off
```

**Expected Performance:**
- Quick detection when entering
- Fast return to idle after passing
- Minimal CPU usage (features disabled)

### Use Case 5: Elderly Care Monitoring

**Environment:**
- Bedroom or living area
- Need to detect all movements
- High reliability required

**Optimal Configuration (in interactive CLI):**
```
espectre> segmentation_threshold 0.25
espectre> features_enable on
espectre> hampel_filter on
```

**Expected Performance:**
- High sensitivity to all movements
- Reliable for safety monitoring

---

## 🎓 Best Practices

---

### 1. Tuning Workflow

Always follow this sequence:
1. Position sensor optimally
2. Observe baseline (empty room)
3. Adjust segmentation threshold
4. Test with movement
5. Fine-tune if needed
6. Document your settings

### 2. Iterative Tuning

Make changes incrementally:
- Adjust one parameter at a time
- Test for 5-10 minutes after each change
- Document what works and what doesn't
- Keep notes on your environment

### 3. Seasonal Adjustments

You may need to retune when:
- Furniture is rearranged
- Seasonal changes (heating/cooling)
- New interference sources added
- Router position changes

### 4. Multiple Sensors

If using multiple sensors:
- Tune each independently
- Different rooms may need different settings
- Use descriptive MQTT topics (e.g., `home/espectre/bedroom`)
- Document settings per sensor

### 5. Monitoring & Maintenance

Regular checks (in interactive CLI):
```
# Check current configuration
espectre> info

# Monitor runtime statistics
espectre> stats

# Monitor via serial output
idf.py monitor
```

**Understanding the difference:**
- **`info`**: Static configuration (thresholds, filters, network settings)
- **`stats`**: Dynamic runtime metrics (state, turbulence, variance, segments, uptime)

---

## 📚 Additional Resources

---

- **Main Documentation:** [README.md](README.md)
- **Setup Guide:** [SETUP.md](SETUP.md)
- **GitHub Issues:** [Report problems or ask questions](https://github.com/francescopace/espectre/issues)
- **Email Support:** francesco.pace@gmail.com

---

## 🤝 Contributing

---

Found a configuration that works great for your environment? Share it!

Open a GitHub issue or pull request with:
- Your environment description
- Optimal configuration values
- Performance notes

Help others benefit from your experience!
