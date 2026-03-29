# Performance Metrics

This document provides detailed performance metrics for ESPectre's motion detection algorithms.

---

## Performance Targets

| Metric | Target (all chips) | Rationale |
|--------|--------------------|-----------|
| Recall | >95% | Minimize missed detections |
| FP Rate | <5% | Avoid false alarms |

--
### Test Configuration

Configuration used for all test results (unified across chips):

| Parameter | Value | Notes |
|-----------|-------|-------|
| Window Size | 75 | `DETECTOR_DEFAULT_WINDOW_SIZE` |
| Calibration | NBVI | Auto-selects 12 non-consecutive subcarriers |
| Hampel Filter | ON | Enabled for both MVS and ML (window=7, threshold=5.0 MAD) |
| Adaptive Threshold | Percentile-based | P95 × 1.1 (`DEFAULT_ADAPTIVE_FACTOR`) |
| CV Normalization | Per-file | Based on `gain_locked` metadata (`false` => apply CV norm) |

CV normalization is applied per-file based on whether data was collected with AGC gain lock enabled. See Test Data section for details.

---

## Test Data

| Chip | Baseline | Movement | Total | Gain Lock |
|------|----------|----------|-------|-----------|
| ESP32-C3 | 2684 | 2658 | 5342 | Yes |
| ESP32-C5 | 2609 | 2607 | 5216 | Yes |
| ESP32-C6 | 2697 | 2779 | 5476 | Yes |
| ESP32-S3 | 2655 | 2670 | 5325 | Yes |
| ESP32 | 2081 | 2189 | 4270 | No |

Data location: `micro-espectre/data/`

---

## Running Tests

```bash
source venv/bin/activate

# C++
cd test && pio test -f test_motion_detection -v

# Python
cd micro-espectre && pytest tests/test_validation_real_data.py -v
```

---

## Current Results

Results from C++ and Python tests follow the same trends (same algorithms, same data, same methodology), with small per-chip differences due to platform/runtime implementation details.

| Chip | Algorithm | Recall | Precision | FP Rate | F1-Score |
|------|-----------|--------|-----------|---------|----------|
| ESP32-C3 | MVS Default | 96.1% | 99.9% | 0.1% | 98.0% |
| ESP32-C3 | MVS + NBVI | 96.5% | 100.0% | 0.0% | 98.2% |
| ESP32-C3 | ML | 99.6% | 100.0% | 0.0% | 99.8% |
| ESP32-C5 | MVS Default | 99.7% | 99.2% | 1.1% | 99.5% |
| ESP32-C5 | MVS + NBVI | 99.1% | 100.0% | 0.0% | 99.6% |
| ESP32-C5 | ML | 100.0% | 100.0% | 0.0% | 100.0% |
| ESP32-C6 | MVS Default | 98.1% | 100.0% | 0.0% | 99.0% |
| ESP32-C6 | MVS + NBVI | 99.6% | 99.8% | 0.3% | 99.7% |
| ESP32-C6 | ML | 100.0% | 100.0% | 0.0% | 100.0% |
| ESP32-S3 | MVS Default | 99.8% | 98.0% | 2.8% | 98.9% |
| ESP32-S3 | MVS + NBVI | 96.7% | 100.0% | 0.0% | 98.3% |
| ESP32-S3 | ML | 99.8% | 100.0% | 0.0% | 99.9% |
| ESP32 | MVS Default | 99.8% | 100.0% | 0.0% | 99.9% |
| ESP32 | MVS + NBVI | 99.8% | 100.0% | 0.0% | 99.9% |
| ESP32 | ML | 99.6% | 100.0% | 0.0% | 99.8% |

**MVS Default**: Uses default subcarriers.
**MVS + NBVI**: Uses NBVI auto-calibration (production case).
**ML**: Neural network with chip-grouped CV, hard-positive mining, and Hampel filter.

---

## System Resources

Resource usage benchmarks for ESPectre with full ESPHome stack (WiFi, API, OTA, debug sensors).

Development YAML files (`-dev.yaml`) include ESPHome debug sensors for runtime monitoring of free heap, max block size, and loop time. 
These sensors are available in Home Assistant for continuous monitoring.

Additional performance logs are available at DEBUG level (`logger.level: DEBUG`):
- `[resources]` - Free heap at startup and post-calibration
- `[perf]` - Detection time per packet (logged every ~10 seconds)

---

### Flash Usage

| Chip | Firmware Size | Flash Used | Free App Slot |
|------|---------------|------------|---------------|
| ESP32-C3 | 1370 KB | 73.8% | 486 KB |
| ESP32-C5 | 1587 KB | 85.5% | 269 KB |
| ESP32-C6 | 1539 KB | 82.9% | 317 KB |
| ESP32-S3 | 1246 KB | 67.1% | 610 KB |

Partition layout uses two app slots (`app0`/`app1`, 1.81 MB each) plus a small `otadata` partition for OTA metadata.
 `Free App Slot` is the remaining space in one app slot after placing the firmware image.

---

### RAM Usage

| Chip | Phase | Free Heap | Notes |
|------|-------|-----------|-------|
| ESP32-C3 | Post-setup | 179 KB | After ESPectre init |
| ESP32-C3 | Post-calibration | 83 KB | After NBVI completes |
| ESP32-C5 | Post-setup | 162 KB | After ESPectre init |
| ESP32-C5 | Post-calibration | 71 KB | After NBVI completes |
| ESP32-C6 | Post-setup | 272 KB | After ESPectre init |
| ESP32-C6 | Post-calibration | 180 KB | After NBVI completes |
| ESP32-S3 | Post-setup | 8425 KB | After ESPectre init (includes PSRAM heap) |
| ESP32-S3 | Post-calibration | 8331 KB | After NBVI completes (includes PSRAM heap) |

---

### Detection Timing

Time to process one CSI packet (feature extraction + detection, measured on hardware).
At 100 pps, each packet has a 10 ms budget. 

| Chip | Algorithm | Detection Time | CPU @ 100 pps |
|------|-----------|----------------|---------------|
| ESP32-C3 | MVS | ~440 µs | ~4.4% |
| ESP32-C3 | ML | ~3400 µs | ~34% |
| ESP32-C5 | MVS | ~220 µs | ~2.2% |
| ESP32-C5 | ML | ~1500 µs | ~15% |
| ESP32-C6 | MVS | ~250 µs | ~2.5% |
| ESP32-C6 | ML | ~1900 µs | ~19% |
| ESP32-S3 | MVS | ~150 µs | ~1.5% |
| ESP32-S3 | ML | ~430 µs | ~4.3% |

The worst-case path is ML on ESP32-C3 (~3.5 ms peak, ~35% CPU), which still leaves substantial budget for WiFi, ESPHome, and Home Assistant communication.

**MVS**: Extracts a single feature (spatial turbulence) and its moving variance.

**ML**: Extracts 12 statistical features from sliding window, then runs MLP inference (12 → 16 → 8 → 1 = 328 MACs). 
The MLP itself is lightweight; most time is spent on feature extraction. 
For ML architecture details, see [ALGORITHMS.md](micro-espectre/ALGORITHMS.md#architecture).

---

## Result History (ESP32-C6)

| Date | Version | Dataset | Calibration | Algorithm | Recall | Precision | FP Rate | F1-Score |
|------|---------|---------|-------------|-----------|--------|-----------|---------|----------|
| 2026-03-29 | v2.8.0 | C6 |  -   | ML + Hampel | 100.0% | 100.0% | 0.0% | 100.0% |
| 2026-03-29 | v2.8.0 | C6 | NBVI | MVS + Hampel| 99.6% | 99.8% | 0.3% | 99.7% |
| 2026-03-11 | v2.6.1 | C6 |  -   | ML | 100.0% | 100.0% | 0.0% | 100.0% |
| 2026-03-11 | v2.6.1 | C6 | NBVI | MVS | 99.3% | 100.0% | 0.0% | 99.7% |
| 2026-03-08 | v2.6.0 | C6 |  -   | ML | 100.0% | 100.0% | 0.0% | 100.0% |
| 2026-03-08 | v2.6.0 | C6 | NBVI | MVS | 99.9% | 98.4% | 2.3% | 99.2% |
| 2026-02-15 | v2.5.0 | C6 |   -  | ML  | 99.9% | 100.0% | 0.0% | 99.9% |
| 2026-02-15 | v2.5.0 | C6 | NBVI | MVS | 99.9% | 99.9% | 0.1% | 99.9% |
| 2026-01-23 | v2.4.0 | C6 | NBVI | MVS | 99.8% | 96.5% | 3.6% | 98.1% |
| 2025-12-27 | v2.3.0 | C6 | NBVI | MVS | 96.4% | 100.0% | 0.0% | 98.2% |

---

## License

GPLv3 - See [LICENSE](LICENSE) for details.
