# 🛜 ESPectre 👻 - Calibration & Tuning Guide

This guide will help you calibrate and optimize ESPectre for your specific environment to achieve reliable movement detection.

---

## 📋 Table of Contents

- [Using the CLI Tool](#️-using-the-cli-tool)
- [Understanding Your Environment](#-understanding-your-environment)
- [Automatic Calibration](#-automatic-calibration)
- [Manual Calibration](#-manual-calibration)
- [Advanced Tuning](#-advanced-tuning)
- [Troubleshooting Scenarios](#-troubleshooting-scenarios)
- [Use Case Examples](#-use-case-examples)

---

## 🛠️ Using the CLI Tool

---

The `espectre-cli.sh` script provides an **interactive interface** for calibrating and tuning your ESPectre sensor in real-time.

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
║                  📡  Interactive CLI Mode                 ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝

Connected to: homeassistant.local:1883
Command topic: home/espectre/node1/cmd
Listening on: home/espectre/node1/response

Type 'help' for commands, 'exit' to quit

espectre> info
20:05:12 {"network": {"ip_address": "192.168.1.100"}, ...}

espectre> analyze
20:05:15 {"min": 0.02, "max": 0.18, "recommended_threshold": 0.35}

espectre> threshold 0.35
20:05:18 {"response": "Threshold set to 0.35"}

espectre> stats
20:05:20 {"detections": 42, "uptime": "3h 24m"}

espectre> help
[shows all available commands]

espectre> exit
Shutting down...
```

**Available commands in interactive mode:**
- Type `help` to see all commands
- Use shortcuts: `t` (threshold), `s` (stats), `i` (info), etc.
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
- Increase `debounce` count (3-5)
- Increase `threshold` slightly
- Enable Hampel filter to remove outliers

#### 4. **Wi-Fi Traffic**

- Low traffic networks: More stable CSI data
- High traffic networks: More noise, may need filtering
- Mesh networks: Work fine, ensure 2.4GHz connection

---

## 🤖 Automatic Calibration

---

ESPectre now includes an **automatic guided calibration system** that optimizes feature selection and weights for your specific environment.

### What is Auto-Calibration?

Instead of manually tuning weights and thresholds, the auto-calibration:

1. **Disables all filters** automatically before data collection
2. **Collects data** in two phases (baseline + movement)
3. **Analyzes** all CSI features using Fisher's criterion
4. **Selects** the top 4-6 most discriminant features
5. **Calculates** optimal weights automatically
6. **Analyzes signal characteristics** to determine optimal filter configuration
7. **Applies optimal filters** automatically with calculated parameters
8. **Reduces CPU usage** by 30-40% at runtime

### Benefits

- ⚡ **30-40% faster** - extracts only 4-6 features instead of 15
- 💾 **60% RAM savings** - smaller feature buffers
- 🎯 **Environment-specific** - optimized for YOUR room
- ✅ **Zero manual tuning** - fully automatic

### How to Use

#### Prerequisites

**Configure traffic generator rate first** (required for sample-based calibration):

```bash
./espectre-cli.sh
```

Then in the interactive session:
```
espectre> traffic_generator_rate 20
```

**Valid range:** 5-100 pps (packets per second)
- **Recommended:** 15 pps (balanced speed and accuracy)
- **Minimum:** 5 pps (slower but works)
- **Maximum:** 100 pps (very fast, may be too quick for movement phase)

**Why this is required:** Calibration now uses **sample-based collection** instead of time-based. The system calculates `target_samples = duration × traffic_rate` to ensure consistent data collection regardless of network conditions.

#### Via CLI (Recommended)

Launch the interactive CLI first:
```bash
./espectre-cli.sh
```

Then in the interactive session:

**Start calibration:**
```
espectre> calibrate start
```

**With custom sample count:**
```
espectre> calibrate start 2000
```

**Check progress:**
```
espectre> calibrate status
```

**Stop calibration:**
```
espectre> calibrate stop
```

#### Via MQTT

**Start:**
```bash
mosquitto_pub -h homeassistant.local -t "home/espectre/node1/cmd" \
  -m '{"cmd":"calibrate","action":"start","samples":1000}'
```

**Status:**
```bash
mosquitto_pub -h homeassistant.local -t "home/espectre/node1/cmd" \
  -m '{"cmd":"calibrate","action":"status"}'
```

### Calibration Process

The calibration uses **sample-based collection** for reliable results:

#### Phase 1: Baseline (target samples)
1. Start calibration
2. **LEAVE THE ROOM COMPLETELY EMPTY**
3. No people, no pets, no moving objects
4. System collects exactly the specified number of samples
5. **Example:** 1000 samples @ 20pps = ~50 seconds
6. Wait for automatic phase transition

#### Phase 2: Movement (target samples)
1. System automatically advances to Phase 2
2. **ENTER AND MOVE NORMALLY** in the room
3. Walk around naturally (not exaggerated)
4. Try different movement patterns:
   - Walking across the room
   - Different speeds and directions
   - Standing and moving arms
5. System collects exactly the specified number of samples
6. **Example:** 1000 samples @ 20pps = ~50 seconds

#### Phase 3: Analysis (Automatic)
1. System calculates Fisher scores for all 8 features
2. Selects top 4-6 most discriminant features
3. Calculates optimal weights
4. Returns to normal operation

### Example Output

```
🎯 Calibration configuration:
   Target samples: 1000 per phase
   Traffic rate: 15 pps
   Estimated duration: ~50 seconds per phase

🎯 Calibration started
🎯 Phase 1: BASELINE (target: 1000 samples)
📋 Please ensure the room is EMPTY and STATIC

... (collecting samples) ...

✅ Baseline phase complete (1000 samples collected)
🎯 Phase 2: MOVEMENT (target: 1000 samples)
📋 Please perform NORMAL MOVEMENT in the room

... (collecting samples) ...

✅ Movement phase complete (1000 samples collected)
🔬 Analyzing collected data...

✅ Calibration complete! Selected 4 features:
  1. variance (Fisher=15.23, weight=0.380)
  2. spatial_gradient (Fisher=10.45, weight=0.261)
  4. iqr (Fisher=5.67, weight=0.142)
  5. entropy (Fisher=2.34, weight=0.058)

🎯 Optimal threshold: 0.35 (baseline: 0.08, movement: 0.62)

🔧 Analyzing optimal filter configuration...
  Hampel filter: ON (threshold: 2.0) - detected 8.5% outliers
  Savitzky-Golay filter: ON - SNR=7.2 (noisy signal)
  Adaptive normalizer: ON (alpha: 0.01) - baseline drift=0.15
  Butterworth filter: ON (always recommended for noise reduction)

✅ Optimal filter configuration applied

💡 Expected CPU savings: ~67%
```

### Automatic Filter Optimization

The calibration system now **automatically analyzes your environment** and configures filters optimally:

#### 1. **Butterworth Filter**
- **Always enabled** - removes high-frequency noise (>8Hz)
- Human movement is typically 0.5-8Hz
- No configuration needed

#### 2. **Wavelet Filter** (Low-Frequency Noise Removal)
- **Enabled if:** Baseline variance > 500
- **Level:** 3 (maximum denoising)
- **Threshold:**
  - 1.0 (balanced) if variance 500-600
  - 1.5 (aggressive) if variance 600-800
  - 2.0 (very aggressive) if variance > 800
- **Purpose:** Removes persistent low-frequency noise that Butterworth can't handle
- **Performance:** Reduces variance by 70-84% in high-noise environments

#### 3. **Hampel Filter** (Outlier Detection)
- **Enabled if:** Outlier ratio > 5%
- **Threshold:** 
  - 2.0 (standard) if outlier ratio 5-15%
  - 3.0 (tolerant) if outlier ratio > 15%
- **Purpose:** Removes electrical spikes and interference

#### 4. **Savitzky-Golay Filter** (Smoothing)
- **Enabled if:** Signal-to-Noise Ratio (SNR) < 10.0
- **Purpose:** Smooths noisy signals while preserving shape
- **Disabled if:** Signal is already clean (SNR ≥ 10.0)

#### 5. **Adaptive Normalizer** (Baseline Tracking)
- **Enabled if:** Baseline drift > 0.1
- **Alpha (learning rate):**
  - 0.01 (standard) if drift 0.1-0.3
  - 0.02 (fast) if drift > 0.3
- **Purpose:** Adapts to slow environmental changes
- **Disabled if:** Baseline is stable (drift < 0.1)

### Filter Configuration in Status Response

When you check calibration status after completion, you'll receive the recommended filter configuration:

```bash
espectre> calibrate status
```

**Response:**
```json
{
  "phase": "ANALYZING",
  "progress": 1.0,
  "active": false,
  "num_selected": 5,
  "optimal_threshold": 0.35,
  "filter_config": {
    "butterworth_enabled": true,
    "hampel_enabled": true,
    "hampel_threshold": 2.0,
    "savgol_enabled": true,
    "adaptive_normalizer_enabled": true,
    "adaptive_normalizer_alpha": 0.01
  }
}
```

This configuration is **automatically applied** at the end of calibration - no manual intervention needed!

### Understanding Fisher Scores

Fisher's criterion measures how well a feature separates baseline from movement:

```
Fisher Score = (μ_movement - μ_baseline)² / (σ²_movement + σ²_baseline)
```

- **High score (>10)**: Excellent discriminator - large separation, low variance
- **Medium score (5-10)**: Good discriminator - useful for detection
- **Low score (<5)**: Poor discriminator - not selected

### When to Use Auto-Calibration

**Recommended for:**
- ✅ New installations
- ✅ Unfamiliar environments
- ✅ Maximizing performance
- ✅ Reducing CPU usage
- ✅ Environments with unique characteristics

**Manual tuning still better for:**
- ❌ Quick adjustments
- ❌ Fine-tuning specific parameters
- ❌ Troubleshooting specific issues
- ❌ When you know exactly what you need

### Auto-Calibration vs Manual Tuning

| Aspect | Auto-Calibration | Manual Tuning |
|--------|------------------|---------------|
| **Time Required** | 2-3 minutes | 10-30 minutes |
| **Expertise Needed** | None | Medium-High |
| **CPU Efficiency** | Optimized (4-6 features) | Standard (4 features) |
| **Accuracy** | Environment-specific | Generic |
| **Flexibility** | Limited | Full control |
| **Best For** | Initial setup | Fine-tuning |

**Recommendation:** Start with auto-calibration, then use manual tuning for fine adjustments if needed.

### Tips for Best Results

1. **Sample Count Selection:**
   - **1000 samples** (default): Recommended for most cases (~50s @ 20pps)
   - **500 samples**: Quick calibration (~25s @ 20pps)
   - **2000 samples**: More accurate (~100s @ 20pps)
   - **Range:** 100-10000 samples

2. **Baseline Phase:**
   - Truly empty the room
   - Close doors to prevent air currents
   - Turn off fans, AC, heaters
   - Wait outside the room

3. **Movement Phase:**
   - Move naturally, not exaggerated
   - Cover different areas of the room
   - Try various speeds (slow, normal, fast)
   - Include typical movements for your use case
   - **Note:** Duration = samples / traffic_rate

### Troubleshooting Auto-Calibration

**"Invalid sample count" error:**
- Valid range: 100-10000 samples
- Use: `calibrate start 1000`

**"Not enough samples" warning:**
- Increase sample count: `calibrate start 2000`
- Check Wi-Fi packet rate with `idf.py monitor`

**Calibration too fast/slow:**
- **Too fast:** Increase sample count: `calibrate start 2000`
- **Too slow:** Decrease sample count: `calibrate start 500`
- **Example:** 1000 samples @ 20pps = ~50 seconds

**All features have low Fisher scores:**
- Baseline phase likely had movement
- Recalibrate with truly static room

**Only 4 features selected:**
- Normal - other features didn't meet 20% threshold
- Still provides good performance

**Calibration doesn't improve detection:**
- Try more varied movement in Phase 2
- Increase sample count: `calibrate start 2000`
- Verify baseline phase was truly static
- Check traffic generator is running: `info`

---

## 📊 Manual Calibration

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

**Start the interactive CLI:**
```bash
./espectre-cli.sh
```

**In the interactive session:**

1. **Run statistical analysis**:
   ```
   espectre> analyze
   20:05:15 {"min": 0.02, "max": 0.18, "avg": 0.08, "stddev": 0.03, 
             "p25": 0.06, "p50_median": 0.08, "p75": 0.11, "p95": 0.15,
             "recommended_threshold": 0.35, "current_threshold": 0.40}
   ```

2. **Interpret the results**:
   - `p50_median`: Typical idle value
   - `p95`: Maximum idle value (95th percentile)
   - `recommended_threshold`: Suggested value (median between p50 and p75)

3. **Apply the recommended threshold**:
   ```
   espectre> threshold 0.35
   20:05:18 {"response": "Threshold set to 0.35"}
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

3. **Adjust if needed** (in the interactive CLI session):

   **Too many false positives:**
   ```
   espectre> threshold 0.45
   
   # Or increase debounce
   espectre> debounce 4
   ```

   **Missing movements:**
   ```
   espectre> threshold 0.30
   
   # Or decrease debounce
   espectre> debounce 2
   ```

### Phase 5: Verification (3 minutes)

1. **Run a complete test cycle**:
   - Leave room empty for 30 seconds → should be "idle"
   - Enter and move around → should detect within 1-2 seconds
   - Stand still → should return to "idle" after 3 seconds
   - Repeat 3-5 times

2. **Check statistics** (in interactive CLI):
   ```
   espectre> stats
   ```

3. **Save your configuration** (note down the values in interactive CLI):
   ```
   espectre> info
   ```

---

## 🔧 Advanced Tuning

---

Once basic calibration is complete, you can fine-tune advanced parameters for optimal performance.

### Detection Parameters

#### Threshold
**What it does:** Minimum score needed to trigger detection

**Default:** 0.40

**When to adjust:**
- **Lower (0.25-0.35):** Small rooms, need high sensitivity
- **Higher (0.45-0.60):** Large rooms, reduce false positives

**Interactive CLI:**
```
espectre> threshold 0.35
```

#### Debounce Count
**What it does:** Number of consecutive detections needed before triggering

**Default:** 3

**When to adjust:**
- **Lower (1-2):** Faster response, more false positives
- **Higher (4-5):** Slower response, fewer false positives

**Interactive CLI:**
```
espectre> debounce 3
```

#### Persistence Timeout
**What it does:** Seconds to wait before returning to IDLE after last detection

**Default:** 3 seconds

**When to adjust:**
- **Lower (1-2s):** Quick response to absence
- **Higher (5-10s):** Avoid flickering in/out of detection

**Interactive CLI:**
```
espectre> persistence 5
```

#### Hysteresis Ratio
**What it does:** Ratio for lower threshold (prevents rapid state flipping)

**Default:** 0.7 (lower threshold = 0.7 × upper threshold)

**When to adjust:**
- **Lower (0.5-0.6):** More stable, slower to return to idle
- **Higher (0.8-0.9):** Faster response, may flicker

**Interactive CLI:**
```
espectre> hysteresis 0.7
```

### Signal Processing

#### Variance Scale
**What it does:** Normalization scale for variance feature (affects overall sensitivity)

**Default:** 400

**When to adjust:**
- **Lower (200-300):** Higher sensitivity (more responsive)
- **Higher (500-800):** Lower sensitivity (fewer false positives)

**Interactive CLI:**
```
espectre> variance_scale 400
```

#### Hampel Filter (Outlier Removal)
**What it does:** Removes statistical outliers using Median Absolute Deviation

**Default:** Enabled, threshold 2.0

**When to use:**
- Enable in high-interference environments
- Helps reduce false positives from electrical noise

**Interactive CLI:**
```
# Enable/disable
espectre> hampel_filter on

# Adjust sensitivity (1.0-10.0, lower = more aggressive)
espectre> hampel_threshold 3.0
```

#### Butterworth Low-Pass Filter
**What it does:** Removes high-frequency noise (>8Hz) - human movement is 0.5-8Hz

**Default:** Enabled (recommended to keep on)

**When to use:**
- **Always recommended** - significantly reduces false positives
- Filters out environmental noise, vibrations, electrical interference
- Based on scientific papers (Lolla, CSI-HC, Dong et al.)

**Performance impact:**
- CPU: Minimal (~5-10 μs per packet)
- Benefit: +5-8% accuracy, -30-50% false positives

**Interactive CLI:**
```
espectre> butterworth_filter on
```

#### Wavelet Filter (Advanced - Daubechies db4)
**What it does:** Removes low-frequency persistent noise using wavelet transform

**Default:** Disabled (enable only for high-noise environments)

**When to enable:**
- ✅ High variance in static environment (>500)
- ✅ Persistent low-frequency noise that Butterworth can't remove
- ✅ Poor detection accuracy despite calibration
- ✅ Baseline drift issues

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

# For very high noise (variance >500), use aggressive threshold
espectre> wavelet_threshold 1.5
```

**Performance impact:**
- CPU: ~5-8% additional load
- RAM: ~2 KB
- Latency: 320ms warm-up (32 samples)
- Benefit: 70-80% variance reduction in high-noise environments

**Expected results:**
- Variance drops from >500 to <100 in static environment
- Spatial gradient reduces by ~60%
- Significantly improved baseline stability

**Based on research:**
- "Location Intelligence System" (2022) - ESP32 + wavelet db4
- "CSI-HC" (2020) - Butterworth + Sym8 combination

#### Savitzky-Golay Filter (Smoothing)
**What it does:** Polynomial smoothing to reduce noise

**Default:** Enabled

**When to use:**
- Keep enabled for most scenarios
- Disable only if you need raw, unfiltered data

**Interactive CLI:**
```
espectre> savgol_filter on
```

#### Check Filter Status

**Interactive CLI:**
```
espectre> info
```

Then filter the JSON output with jq if needed:
```bash
# From another terminal
mosquitto_sub -h homeassistant.local -t "home/espectre/node1/response" | jq '.filters'
```

**Example output:**
```json
{
  "butterworth_enabled": true,
  "hampel_enabled": true,
  "hampel_threshold": 2.0,
  "savgol_enabled": true,
  "savgol_window_size": 5,
  "adaptive_normalizer_enabled": true,
  "adaptive_normalizer_alpha": 0.01,
  "adaptive_normalizer_reset_timeout_sec": 30
}
```

#### Adaptive Normalizer (Signal Baseline Tracking)

**What it does:** Tracks the running mean and variance of the signal to adapt to environmental changes

**Default:** Enabled, alpha=0.01, auto-reset after 30 seconds of IDLE

**The Problem it Solves:**

In static conditions, you may observe a **progressive signal degradation** where the movement score decreases over time even without any changes:

```
Time:  0s    30s   60s   90s   120s
Score: 30% → 25% → 20% → 15% → 6%  (then stabilizes)
```

This happens because the adaptive normalizer "learns" the static noise as the new baseline, progressively reducing the normalized signal.

**The Solution:**

The system now includes **automatic reset** functionality:
- After N seconds of IDLE state (default: 30), the normalizer resets
- This prevents progressive degradation

**Configuration Parameters:**

1. **Enable/Disable:**
   ```
   # Enable (default)
   espectre> adaptive_normalizer on
   
   # Disable (not recommended - loses adaptive capability)
   espectre> adaptive_normalizer off
   ```

2. **Learning Rate (alpha):**
   Controls how quickly the normalizer adapts to changes
   
   ```
   # Slow adaptation (more stable, less responsive)
   espectre> adaptive_normalizer_alpha 0.001
   
   # Default (balanced)
   espectre> adaptive_normalizer_alpha 0.01
   
   # Fast adaptation (more responsive, less stable)
   espectre> adaptive_normalizer_alpha 0.05
   ```
   
   **Range:** 0.001-0.1
   - **Lower values (0.001-0.005):** Slower adaptation, more stable baseline
   - **Medium values (0.01-0.02):** Balanced (recommended)
   - **Higher values (0.05-0.1):** Faster adaptation, may degrade quicker

3. **Auto-Reset Timeout:**
   Seconds of IDLE before resetting the normalizer
   
   ```
   # Short timeout (frequent resets)
   espectre> adaptive_normalizer_reset_timeout 10
   
   # Default (balanced)
   espectre> adaptive_normalizer_reset_timeout 30
   
   # Long timeout (rare resets)
   espectre> adaptive_normalizer_reset_timeout 60
   
   # Disable auto-reset (not recommended)
   espectre> adaptive_normalizer_reset_timeout 0
   ```
   
   **Range:** 0-300 seconds (0 = disabled)

4. **View Statistics:**
   ```
   espectre> adaptive_normalizer_stats
   ```
   
   **Example output:**
   ```json
   {
     "enabled": true,
     "alpha": 0.01,
     "reset_timeout_sec": 30,
     "current_mean": 0.0234,
     "current_variance": 0.0012,
     "current_stddev": 0.0346
   }
   ```

**When to Adjust:**

- **Experiencing signal degradation:** Reduce alpha or decrease reset timeout
- **Too many resets in logs:** Increase reset timeout
- **Need faster environmental adaptation:** Increase alpha (but may degrade faster)
- **Very stable environment:** Can increase reset timeout to 60-120 seconds
- **High interference environment:** Decrease alpha for more stable baseline

**Recommended Settings by Environment:**

| Environment | Alpha | Reset Timeout | Reason |
|-------------|-------|---------------|--------|
| Stable home | 0.01 | 30s | Default, balanced |
| High interference | 0.005 | 20s | Slower adaptation, frequent resets |
| Office/dynamic | 0.02 | 45s | Faster adaptation, less frequent resets |
| Very stable lab | 0.001 | 60s | Very slow adaptation, rare resets |

### View Current Features and Weights

ESPectre automatically manages features and weights through the calibration system. The weights are calculated using Fisher's criterion to optimize detection for your specific environment.

**View all features with calibration info (in interactive CLI):**
```
espectre> features
```

**Example output:**
```json
{
  "time_domain_variance": {
    "value": 315.50,
    "selected": true,
    "weight": 0.380
  },
  "time_domain_entropy": {
    "value": 5.45,
    "selected": true,
    "weight": 0.261
  },
  "spatial_gradient": {
    "value": 17.26,
    "selected": true,
    "weight": 0.222
  },
  "time_domain_iqr": {
    "value": 18.5,
    "selected": true,
    "weight": 0.137
  },
  "time_domain_skewness": {
    "value": 0.0854
  },
  "spatial_correlation": {
    "value": -0.42
  }
}
```

**Understanding the output:**
- Features **with** `"selected": true` are used in detection (show their weight)
- Features **without** `"selected"` are calculated but not used
- Weights sum to 1.0 across all selected features
- After auto-calibration, the system automatically selects the 4-6 most discriminant features

**Available features (8 total):**
- **Time-domain** (5): variance, skewness, kurtosis, entropy, iqr
- **Spatial** (3): variance, correlation, gradient

---

## 🔍 Troubleshooting Scenarios

---

### Scenario 1: Complete Reset Needed

**Symptoms:**
- Configuration is corrupted or inconsistent
- Want to start fresh with default settings
- Testing different configurations
- Persistent issues after multiple tuning attempts

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
- ✅ Clear all NVS storage (calibration + configuration)
- ✅ Restore all parameters to factory defaults
- ✅ Reinitialize the calibration system
- ⚠️ You will need to recalibrate after reset

**After factory reset**, follow the [Automatic Calibration](#-automatic-calibration) process to set up the sensor again.

### Scenario 2: Too Many False Positives

**Symptoms:**
- Detects movement when room is empty
- Frequent state changes without actual movement
- High `movement` values during idle periods

**Solutions (in interactive CLI):**

1. **Increase threshold:**
   ```
   espectre> threshold 0.50
   ```

2. **Increase debounce:**
   ```
   espectre> debounce 4
   ```

3. **Enable/tune Hampel filter:**
   ```
   espectre> hampel_filter on
   espectre> hampel_threshold 2.5
   ```

4. **Check for interference sources:**
   - Move sensor away from fans, AC units
   - Check for moving curtains or plants
   - Verify no pets in the room

5. **Increase variance scale (reduce sensitivity):**
   ```
   espectre> variance_scale 600
   ```

### Scenario 3: Missing Movements

**Symptoms:**
- Doesn't detect when people move
- `movement` values stay low even during activity
- State remains "idle" during movement

**Solutions (in interactive CLI):**

1. **Decrease threshold:**
   ```
   espectre> threshold 0.30
   ```

2. **Check sensor position:**
   - Verify distance from router (3-8m optimal)
   - Ensure external antenna is connected
   - Try different height (1-1.5m)

3. **Decrease variance scale (increase sensitivity):**
   ```
   espectre> variance_scale 300
   ```

4. **Verify Wi-Fi signal strength:**
   ```bash
   idf.py monitor
   # Look for RSSI values > -70 dBm
   ```

5. **Reduce debounce for faster response:**
   ```
   espectre> debounce 2
   ```

### Scenario 4: Unstable Detection

**Symptoms:**
- Rapid flickering between "idle" and "detected"
- Inconsistent detection of same movements
- High variance in `movement` values

**Solutions (in interactive CLI):**

1. **Adjust hysteresis:**
   ```
   espectre> hysteresis 0.6
   ```

2. **Increase persistence timeout:**
   ```
   espectre> persistence 5
   ```

3. **Enable smoothing filters:**
   ```
   espectre> savgol_filter on
   espectre> hampel_filter on
   ```

4. **Check Wi-Fi stability:**
   - Verify router isn't overloaded
   - Check for mesh network handoffs
   - Ensure stable 2.4GHz connection

### Scenario 5: Slow Response Time

**Symptoms:**
- Takes several seconds to detect movement
- Delayed return to idle state

**Solutions (in interactive CLI):**

1. **Reduce debounce count:**
   ```
   espectre> debounce 2
   ```

2. **Reduce persistence timeout:**
   ```
   espectre> persistence 2
   ```

3. **Lower threshold slightly:**
   ```
   espectre> threshold 0.35
   ```

### Scenario 6: Progressive Signal Degradation

**Symptoms:**
- Movement score decreases progressively over time in static conditions
- Score drops from 30% → 25% → 20% → 15% → 6% and stabilizes
- Happens even when nothing changes in the environment
- Eventually stabilizes at a low value (6-15%)

**Cause:**
The adaptive normalizer is "learning" the static noise as the new baseline, progressively reducing the normalized signal.

**Solutions:**

1. **Reduce learning rate (slower adaptation):**
   ```
   espectre> adaptive_normalizer_alpha 0.005
   ```
   This makes the normalizer adapt more slowly, reducing degradation.

2. **Decrease reset timeout (more frequent resets):**
   ```
   espectre> adaptive_normalizer_reset_timeout 20
   ```
   Resets the normalizer more frequently before significant degradation occurs.

3. **Check current normalizer statistics:**
   ```
   espectre> adaptive_normalizer_stats
   ```
   Monitor `current_mean` and `current_variance` to see if they're drifting.

4. **Temporary workaround - disable normalizer:**
   ```
   espectre> adaptive_normalizer off
   ```
   Only use this as a last resort, as it removes adaptive capability.

**Recommended Fix:**
```
espectre> adaptive_normalizer_alpha 0.005
espectre> adaptive_normalizer_reset_timeout 30
```

This provides slower adaptation with regular resets to prevent degradation.

### Scenario 7: Can't Distinguish Pets from Humans

**Symptoms:**
- Pet movements trigger detection
- Want to ignore small animals

**Solutions (in interactive CLI):**

1. **Increase threshold to ignore MICRO movements:**
   ```
   espectre> threshold 0.50
   ```

2. **Use Home Assistant automation to filter:**
   ```yaml
   # Only trigger on DETECTED or INTENSE states
   condition:
     - condition: template
       value_template: "{{ state_attr('sensor.movement_sensor', 'state') in ['detected', 'intense'] }}"
   ```

3. **Run auto-calibration** to optimize feature selection for your environment (may help distinguish movement patterns):
   ```
   espectre> calibrate start
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
espectre> threshold 0.30
espectre> debounce 2
espectre> persistence 3
espectre> variance_scale 300
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
espectre> threshold 0.45
espectre> debounce 3
espectre> persistence 5
espectre> variance_scale 500
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
espectre> threshold 0.35
espectre> debounce 2
espectre> persistence 10
espectre> hampel_filter on
espectre> savgol_filter on
```

**Expected Performance:**
- Detects any movement quickly
- Long persistence keeps "detected" during brief pauses

### Use Case 4: Hallway/Corridor

**Environment:**
- Narrow space (1.5m × 8m)
- People pass through quickly
- Need fast detection and release

**Optimal Configuration (in interactive CLI):**
```
espectre> threshold 0.40
espectre> debounce 2
espectre> persistence 2
espectre> hysteresis 0.8
espectre> variance_scale 400
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

**Optimal Configuration (in interactive CLI):**
```
espectre> threshold 0.25
espectre> debounce 2
espectre> persistence 3
espectre> variance_scale 300
espectre> hampel_filter on
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
- Reliable for safety monitoring

---

## 🎓 Best Practices

---

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

Regular checks (in interactive CLI):
```
# Weekly: Check statistics
espectre> stats

# Monthly: Re-run analysis
espectre> analyze

# As needed: View current config
espectre> info
```

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
