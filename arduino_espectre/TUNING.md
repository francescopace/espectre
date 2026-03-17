# ESPectre Arduino - Tuning Guide

This guide helps you optimize motion detection for your specific environment.

## Current Default Settings

After testing and tuning, these are the optimal settings:

| Parameter | Default Value | Location |
|-----------|--------------|----------|
| **Threshold Multiplier** | P95 × 2.5 | `nbvi_calibrator.cpp` line 133 |
| **Window Size** | 50 packets | `config.h` line 9 |
| **Traffic Rate** | 100 pps | `config.h` line 13 |

## Understanding the Metrics

### Variance (Motion Metric)
The moving variance of CSI turbulence:
- **0.01 - 0.20**: Typical idle values (room still)
- **0.20 - 0.40**: Environmental noise (WiFi interference, HVAC)
- **0.40 - 1.0+**: Real motion detected

### Threshold
Calculated during calibration as **P95 × Multiplier**:
- **P95**: 95th percentile of variance during calibration
- **Multiplier**: Default 2.5 (balances sensitivity vs stability)

### Example from Real Use
```
P95 moving variance: 0.150
Adaptive threshold: 0.375  (0.150 × 2.5)

Idle variance: 0.012 - 0.168  ← Below threshold (no false positives)
Motion variance: 0.5 - 1.5+   ← Above threshold (detected)
```

## Sensitivity Tuning

### More Sensitive (Detects Smaller Movements)

**Use case**: Want to catch subtle movements (hand gestures, small motions)

**Edit**: `nbvi_calibrator.cpp` line 133
```cpp
// Change from 2.5 to 2.0
float threshold = p95 * 2.0f;
```

**Trade-off**: More false positives from environmental noise

**Expected behavior**:
- ✅ Detects smaller movements
- ⚠️ May trigger on WiFi interference, router traffic
- ⚠️ May trigger on HVAC, fans, opening doors

---

### Less Sensitive (Fewer False Positives)

**Use case**: Noisy WiFi environment, only want obvious motion

**Edit**: `nbvi_calibrator.cpp` line 133
```cpp
// Change from 2.5 to 3.0 or 3.5
float threshold = p95 * 3.0f;
```

**Trade-off**: May miss subtle movements

**Expected behavior**:
- ✅ Very stable, minimal false positives
- ✅ Good for crowded WiFi areas
- ⚠️ May miss small gestures or slow movements

---

### Balanced (Recommended - Current Default)

**Setting**: P95 × 2.5

**Best for**: Most environments with moderate WiFi activity

**Expected behavior**:
- ✅ Detects walking, arm movements
- ✅ Ignores most environmental noise
- ✅ ~1-2% false positive rate

## Smoothness Tuning

### Window Size

Controls how many CSI packets are used for variance calculation.

**Location**: `config.h` line 9

#### Faster Response (Smaller Window)

```cpp
#define WINDOW_SIZE 30  // Smaller = faster
```

**Trade-off**: More jittery, less stable

**Expected behavior**:
- ✅ Detects motion faster (~0.5s)
- ⚠️ More variance fluctuations
- ⚠️ More false positives

---

#### Smoother Detection (Larger Window)

```cpp
#define WINDOW_SIZE 75  // Larger = smoother
```

**Trade-off**: Slower response time

**Expected behavior**:
- ✅ Very stable, smooth transitions
- ✅ Fewer false positives
- ⚠️ Slower detection (~2-3s)

---

#### Balanced (Current Default)

```cpp
#define WINDOW_SIZE 50
```

**Best for**: Good balance between speed and stability

**Expected behavior**:
- ✅ Detects motion in 1-2 seconds
- ✅ Reasonably smooth
- ✅ Low false positive rate

## Traffic Rate Tuning

### Packet Generation Rate

Controls how many CSI packets are generated per second.

**Location**: `config.h` line 13

#### Higher Rate (More CSI Data)

```cpp
#define TRAFFIC_RATE_PPS 150  // More packets
```

**Trade-off**: Higher power consumption

**Expected behavior**:
- ✅ Faster detection updates
- ✅ More data for analysis
- ⚠️ ~20mA more current draw
- ⚠️ May overwhelm router

---

#### Lower Rate (Power Saving)

```cpp
#define TRAFFIC_RATE_PPS 50  // Fewer packets
```

**Trade-off**: Slower, less responsive

**Expected behavior**:
- ✅ Lower power consumption
- ✅ Less network traffic
- ⚠️ Slower detection (3-4s)
- ⚠️ Less stable variance

---

#### Balanced (Current Default)

```cpp
#define TRAFFIC_RATE_PPS 100
```

**Best for**: Good balance between performance and power

**Expected behavior**:
- ✅ ~138 pps actual rate (with HTTP overhead)
- ✅ Good detection speed
- ✅ Reasonable power consumption

## Environmental Optimization

### High WiFi Interference (Apartments, Offices)

**Symptoms**:
- Frequent false positives
- Variance spikes when idle

**Solution 1**: Increase threshold
```cpp
float threshold = p95 * 3.0f;  // or even 3.5f
```

**Solution 2**: Increase window size
```cpp
#define WINDOW_SIZE 75
```

**Solution 3**: Re-calibrate during quiet time
- Reset device at 3am when WiFi is quiet
- Or manually trigger calibration during still period

---

### Open Space (Large Room, Few Walls)

**Symptoms**:
- Works great, but want even better detection

**Solution**: Slightly increase sensitivity
```cpp
float threshold = p95 * 2.2f;
```

---

### Through Walls

**Symptoms**:
- Detection works but is less reliable

**Solution 1**: Increase window size for stability
```cpp
#define WINDOW_SIZE 75
```

**Solution 2**: Place sensor closer to router
- CSI signal degrades through walls
- Optimal: 3-8m from router with 1 wall or less

---

### Small Room

**Symptoms**:
- Too sensitive, triggers easily

**Solution**: Reduce sensitivity
```cpp
float threshold = p95 * 3.0f;
```

## Calibration Tips

### When to Re-calibrate

Calibrate when:
- ✅ First boot
- ✅ Environment changes (furniture moved)
- ✅ Router changes (new router, channel change)
- ✅ High false positive rate
- ✅ Time of day (morning vs evening WiFi patterns)

### How to Calibrate Properly

1. **Keep room completely still** for 15 seconds
2. **No movement** during calibration:
   - No walking
   - No opening doors
   - No fans, HVAC changes
3. **Verify in Serial Monitor**:
   ```
   Calibration complete: 700 samples collected ✓
   ```
4. **Check threshold is reasonable**:
   - Too low (<0.1): Re-calibrate, room might have moved
   - Normal (0.2-0.5): Good calibration
   - Too high (>1.0): Re-calibrate, something was moving

### Manual Threshold Override

If auto-calibration isn't working well, use manual threshold:

**Edit**: `arduino_espectre.ino` after calibration completes

```cpp
// After: detector.setThreshold(threshold);
// Add:
detector.setThreshold(0.5);  // Manual threshold (adjust as needed)
```

## Diagnostic Commands

### Check Current Performance

Watch Serial Monitor output:
```
--- Idle | Var: 0.059 | Thr: 0.375 | Pkts: 235
```

**What to look for**:
- **Idle variance < 0.7 × Threshold**: Good (stable idle)
- **Idle variance > 0.8 × Threshold**: Too close (increase threshold)
- **Motion variance > 2.0 × Threshold**: Good (clear detection)
- **Motion variance < 1.5 × Threshold**: Weak signal (reduce threshold or move closer)

### Measure False Positive Rate

1. Leave room for 10 minutes
2. Count false "MOTION" triggers
3. Calculate: `false_positives / 10 = FP per minute`

**Target**: <0.5 FP per minute (one false positive every 2+ minutes)

**If higher**: Increase threshold multiplier

### Measure True Positive Rate

1. Walk around room normally
2. Count how many times motion is detected
3. Ideal: Detects within 1-2 seconds every time

**If missing detections**: Decrease threshold multiplier

## Advanced Tuning

### Subcarrier Selection

By default, NBVI auto-selects 12 optimal subcarriers. To manually override:

**Edit**: `arduino_espectre.ino` around line 200

```cpp
// After NBVI calibration, override:
selected_band = {11, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54};
```

**Use case**: If you know specific subcarriers work better in your environment

### Hysteresis (Reduce Flicker)

Add state persistence to avoid rapid IDLE ↔ MOTION switching:

**Edit**: `mvs_detector.cpp` in `updateState()` method

```cpp
void MVSDetector::updateState() {
    if (!isReady()) return;

    motion_metric_ = calculateMovingVariance();

    // Add hysteresis
    static int motion_count = 0;

    if (motion_metric_ > threshold_) {
        motion_count++;
        if (motion_count >= 2) {  // Require 2 consecutive readings
            state_ = MOTION;
        }
    } else {
        motion_count = 0;
        state_ = IDLE;
    }
}
```

**Effect**: Motion must persist for 2+ updates before triggering

## Troubleshooting

### Still Getting False Positives

1. **Check calibration**: Was room still?
2. **Increase threshold**: Try 3.0× or 3.5×
3. **Check interference**:
   - Other WiFi devices nearby?
   - Microwave running?
   - Bluetooth devices?
4. **Re-calibrate at different time**: Try calibrating at night

### Missing Real Motion

1. **Check distance**: Are you 3-8m from router?
2. **Check walls**: More than 2 walls between sensor and router?
3. **Decrease threshold**: Try 2.0× or 1.8×
4. **Check CSI rate**: Should be 100+ pps in Serial Monitor
5. **Move sensor closer** to router

### Inconsistent Detection

1. **Check CSI packet rate**: Should be stable ~100-150 pps
2. **Increase window size**: Try 75 or 100
3. **Check router**: Is it under heavy load?
4. **Re-calibrate**: Fresh baseline may help

### Display Shows Wrong State

1. **Check Serial Monitor**: Does it match display?
2. **If mismatch**: Display update issue (rare)
3. **If match**: Tuning issue (see above)

## Performance Metrics

After tuning, aim for:

| Metric | Target | How to Measure |
|--------|--------|----------------|
| **True Positive Rate** | >95% | Walk around, count detections |
| **False Positive Rate** | <2% | Leave room, count false triggers |
| **Detection Latency** | 1-2s | Time from motion start to display update |
| **Stability** | No flicker | State shouldn't rapidly switch |

## Quick Tuning Cheat Sheet

| Problem | Solution | File | Line |
|---------|----------|------|------|
| Too many false positives | Increase threshold (3.0×) | `nbvi_calibrator.cpp` | 133 |
| Missing motion | Decrease threshold (2.0×) | `nbvi_calibrator.cpp` | 133 |
| Too jittery | Increase window (75) | `config.h` | 9 |
| Too slow | Decrease window (30) | `config.h` | 9 |
| High power use | Lower traffic (50 pps) | `config.h` | 13 |
| Slow detection | Higher traffic (150 pps) | `config.h` | 13 |

## Example Tuning Sessions

### Session 1: Office Environment (High WiFi)
```
Problem: False positives every 2 minutes
Initial: P95 × 2.5, threshold = 0.375
Action: Increased to P95 × 3.5
Result: threshold = 0.525, FP rate dropped to <1 per hour ✓
```

### Session 2: Home (Low Interference)
```
Problem: Missing small movements
Initial: P95 × 2.5, threshold = 0.310
Action: Decreased to P95 × 2.0, increased window to 75
Result: threshold = 0.248, catches all motion, very smooth ✓
```

### Session 3: Through-Wall Detection
```
Problem: Inconsistent detection through 2 walls
Initial: P95 × 2.5, 50 window
Action: Moved sensor closer (8m → 5m), window = 75
Result: Stable detection through 1 wall ✓
```

---

## Summary

**Start with defaults** (P95 × 2.5, window 50, 100 pps) and tune based on your environment:

1. **Test for 10 minutes** → count false positives
2. **If FP > 1/minute** → increase threshold to 3.0×
3. **If missing motion** → decrease threshold to 2.0×
4. **If jittery** → increase window to 75
5. **Re-calibrate** if major environment changes

Most users find **P95 × 2.5 to 3.0** works best with **window size 50-75**.

Happy tuning 🎯
