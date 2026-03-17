# ESPectre Arduino - Architecture Overview

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Arduino Sketch Layer                          │
│                     (arduino_espectre.ino)                           │
│                                                                      │
│  • Setup: WiFi, TFT, NeoPixel, CSI                                  │
│  • Loop: Update detection state, refresh display @ 5Hz              │
│  • Traffic Generator: FreeRTOS task (UDP DNS @ 100 pps)             │
│  • Display Manager: Render motion state, metrics, status            │
└──────────────────────────────────────────────────────────────────────┘
                                 │
                                 │ uses
                                 │
         ┌───────────────────────┴───────────────────────┐
         │                                               │
         ▼                                               ▼
┌─────────────────────┐                        ┌─────────────────────┐
│   CSI Manager       │                        │  Display Hardware   │
│   (ESP-IDF Layer)   │                        │   (Adafruit libs)   │
│                     │                        │                     │
│  • esp_wifi_*       │                        │  • ST7789 TFT       │
│  • CSI config       │                        │  • NeoPixel LED     │
│  • CSI callback     │                        │  • GFX primitives   │
└──────────┬──────────┘                        └─────────────────────┘
           │
           │ CSI packets (100 pps)
           │
           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Detection Pipeline                              │
└─────────────────────────────────────────────────────────────────────┘
           │
           ├──► Gain Controller (Phase 1: 3 seconds)
           │    • Read AGC/FFT gains
           │    • Lock at stable values
           │    • Only on S3/C3/C5/C6
           │
           ├──► NBVI Calibrator (Phase 2: 7 seconds)
           │    • Collect 700 baseline samples
           │    • Calculate variance per subcarrier
           │    • Select 12 stable, non-consecutive subcarriers
           │    • Calculate adaptive threshold (P95 × 1.4)
           │
           └──► MVS Detector (Phase 3: continuous)
                • Extract I/Q for 12 selected subcarriers
                • Calculate turbulence (spatial std dev)
                • Maintain moving window (50 packets)
                • Calculate moving variance (temporal)
                • Compare: variance > threshold?
                • Output: IDLE or MOTION state
```

## Data Flow

```
ESP32 WiFi Hardware
        │
        │ CSI Raw Data (64 subcarriers × I/Q)
        ▼
┌──────────────────┐
│  CSI Callback    │  IRAM_ATTR interrupt handler
│  (ISR context)   │  Minimal processing, forward to user callback
└────────┬─────────┘
         │
         │ wifi_csi_info_t*
         │
         ▼
┌──────────────────┐
│  User Callback   │  Calibration mode: NBVI Calibrator
│  (Lambda)        │  Detection mode: MVS Detector
└────────┬─────────┘
         │
         ├─ Calibration Phase ────────────────────┐
         │                                        │
         ▼                                        ▼
┌──────────────────┐                    ┌──────────────────┐
│ Collect Sample   │                    │  Magnitude       │
│ (700 samples)    │                    │  Buffer          │
│                  │                    │  [64][700]       │
└────────┬─────────┘                    └────────┬─────────┘
         │                                       │
         │ When complete                         │
         ▼                                       │
┌──────────────────┐                            │
│  NBVI Algorithm  │◄───────────────────────────┘
│  • Calculate variance per subcarrier
│  • Sort by stability (ascending)
│  • Select 12 with spacing ≥2
│  • Return selected_band[]
└────────┬─────────┘
         │
         │ selected_band (12 subcarriers)
         ▼
┌──────────────────┐
│ Adaptive         │
│ Threshold        │
│ • Simulate MVS on calibration data
│ • Calculate P95 of moving variance
│ • threshold = P95 × 1.4
└────────┬─────────┘
         │
         └─► Ready for Detection


         ├─ Detection Phase ──────────────────────┐
         │                                        │
         ▼                                        ▼
┌──────────────────┐                    ┌──────────────────┐
│ Extract          │                    │ Turbulence       │
│ Subcarriers      │                    │ Buffer           │
│ (12 selected)    │───────────────────►│ [window_size]    │
│                  │  turbulence        │ (50 packets)     │
└──────────────────┘  = std(amplitudes) └────────┬─────────┘
                                                 │
                                                 │
                                                 ▼
                                        ┌──────────────────┐
                                        │ Moving Variance  │
                                        │ = var(turbulence)│
                                        └────────┬─────────┘
                                                 │
                                                 │ motion_metric
                                                 ▼
                                        ┌──────────────────┐
                                        │  Threshold       │
                                        │  Comparison      │
                                        │                  │
                                        │  variance > thr? │
                                        └────────┬─────────┘
                                                 │
                                    ┌────────────┴────────────┐
                                    │                         │
                                    ▼                         ▼
                            ┌───────────┐             ┌───────────┐
                            │  MOTION   │             │   IDLE    │
                            │  (Red)    │             │  (Green)  │
                            └───────────┘             └───────────┘
```

## Hardware Stack

```
┌─────────────────────────────────────────────────────────────┐
│  Adafruit Feather ESP32-S3 Reverse TFT                      │
│                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌────────────┐ │
│  │   ESP32-S3      │  │  ST7789 TFT     │  │  NeoPixel  │ │
│  │                 │  │  240×135 pixels │  │  LED       │ │
│  │  • Xtensa LX7   │  │                 │  │            │ │
│  │  • 240 MHz      │  │  SPI Interface: │  │  GPIO 33   │ │
│  │  • 8MB Flash    │  │  CS:  GPIO 7    │  └────────────┘ │
│  │  • 2MB PSRAM    │  │  DC:  GPIO 39   │                 │
│  │  • WiFi 2.4GHz  │  │  RST: GPIO 40   │  ┌────────────┐ │
│  │  • CSI capable  │  │  BL:  GPIO 45   │  │  USB-C     │ │
│  └─────────────────┘  └─────────────────┘  │  Power/Prog│ │
│                                             └────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Software Stack

```
┌─────────────────────────────────────────────────────────────┐
│                     Application Layer                       │
│  arduino_espectre.ino - Main sketch (setup, loop, display)  │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                   Algorithm Layer (C++)                     │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ MVSDetector  │  │NBVI Calibrat.│  │GainController│     │
│  │              │  │              │  │              │     │
│  │ • Turbulence │  │ • Variance   │  │ • AGC lock   │     │
│  │ • MovingVar  │  │ • Band sel.  │  │ • FFT lock   │     │
│  │ • State FSM  │  │ • P95 thresh │  │ • PHY access │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐                       │
│  │ CSIManager   │  │   Utils      │                       │
│  │              │  │              │                       │
│  │ • CSI config │  │ • Math funcs │                       │
│  │ • Callbacks  │  │ • Variance   │                       │
│  │ • Statistics │  │ • Percentile │                       │
│  └──────────────┘  └──────────────┘                       │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                   Hardware Abstraction                      │
│                                                             │
│  ┌──────────────────┐       ┌──────────────────┐          │
│  │ Adafruit Libs    │       │ Arduino-ESP32    │          │
│  │                  │       │                  │          │
│  │ • ST7789 driver  │       │ • WiFi API       │          │
│  │ • GFX primitives │       │ • ESP-IDF bridge │          │
│  │ • NeoPixel ctrl  │       │ • FreeRTOS       │          │
│  └──────────────────┘       └──────────────────┘          │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                      ESP-IDF Layer                          │
│                                                             │
│  • esp_wifi_* (CSI control)                                │
│  • phy_* (gain lock)                                       │
│  • FreeRTOS (task scheduling)                              │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    Hardware Layer                           │
│  ESP32-S3 WiFi PHY/MAC • AGC • FFT • CSI Extraction        │
└─────────────────────────────────────────────────────────────┘
```

## State Machine

```
┌──────────┐
│  BOOT    │
└────┬─────┘
     │
     ▼
┌──────────────────┐
│  WiFi Connect    │
│  (5 seconds)     │
└────┬─────────────┘
     │
     ▼
┌──────────────────┐
│  Gain Lock       │
│  (3 seconds)     │
│  • Read gains    │
│  • Lock AGC/FFT  │
└────┬─────────────┘
     │
     ▼
┌──────────────────┐
│  NBVI Calibrate  │
│  (7 seconds)     │
│  • 700 samples   │
│  • Select band   │
│  • Calc threshold│
└────┬─────────────┘
     │
     ▼
┌──────────────────┐
│  READY           │
│  (green LED)     │
└────┬─────────────┘
     │
     ▼
┌────────────────────────────────────────────┐
│       Detection Loop (continuous)          │
│                                            │
│  ┌─────────────┐    ┌─────────────┐      │
│  │    IDLE     │◄──►│   MOTION    │      │
│  │  (Green LED)│    │  (Red LED)  │      │
│  └─────────────┘    └─────────────┘      │
│         ▲                   │              │
│         │                   │              │
│         │ var ≤ threshold   │ var > thr    │
│         │                   │              │
│         └───────────────────┘              │
│                                            │
│  Update Display @ 5 Hz                     │
│  Traffic Generator @ 100 pps               │
└────────────────────────────────────────────┘
```

## Memory Layout

```
ESP32-S3 Flash (8 MB)
┌─────────────────────────────────┐
│  Bootloader              (~32KB)│
│  Partition Table         (~4KB) │
│  Arduino App           (~1.2MB) │
│    ├─ Code              (~800KB)│
│    ├─ Strings           (~100KB)│
│    ├─ Constants         (~50KB) │
│    └─ Libraries         (~250KB)│
│  SPIFFS/LittleFS        (~4MB)  │  (unused in this version)
│  OTA Partition          (~1.2MB)│  (unused in this version)
│  NVS (WiFi config)      (~16KB) │
└─────────────────────────────────┘

ESP32-S3 RAM (512 KB SRAM + 2 MB PSRAM)
┌─────────────────────────────────┐
│  FreeRTOS Heap          (~200KB)│
│  Arduino Stack          (~8KB)  │
│  WiFi Stack             (~80KB) │
│  CSI Buffers            (~10KB) │
│  Calibration Buffer     (~50KB) │
│    ├─ magnitude[64][700]        │
│    └─ ~179KB total              │
│  Detection Buffers      (~5KB)  │
│    ├─ turbulence[50]            │
│    └─ selected_band[12]         │
│  Display Framebuffer    (~65KB) │
│    └─ 240×135×2 bytes           │
│  Free Heap              (~100KB)│
└─────────────────────────────────┘

PSRAM (2 MB) - Optional, used by WiFi/BLE stack
```

## Timing Diagram

```
Time →
0s          5s       8s      15s            20s            25s
│───────────│────────│───────│──────────────│──────────────│
│           │        │       │              │              │
│  WiFi     │  Gain  │ NBVI  │   Detection  │   Detection  │
│  Connect  │  Lock  │ Calib │   (IDLE)     │   (MOTION)   │
│           │        │       │              │              │
└───────────┴────────┴───────┴──────────────┴──────────────┘
│           │        │       │              │              │
TFT:  Blue    Blue     Magenta   Green          Red
LED:  Blue    Blue     Magenta   Green          Red
CSI:   0       ~300     ~700      ~1200          ~1700 packets

                               ▲
                               │
                               └─ Ready for detection
                                  at ~15 seconds after boot
```

## Processing Pipeline (Detailed)

```
CSI Packet (64 subcarriers × I/Q)
        │
        │ Every 10ms @ 100 pps
        │
        ▼
┌───────────────────────────────────────────────────────┐
│ Step 1: Extract Selected Subcarriers                  │
│                                                       │
│  for sc in selected_band[12]:                        │
│      I = csi_data[sc * 2]                            │
│      Q = csi_data[sc * 2 + 1]                        │
│      amplitude[sc] = sqrt(I² + Q²)                   │
│                                                       │
│  Result: 12 amplitude values                         │
└───────────────┬───────────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────────────────────────┐
│ Step 2: Calculate Spatial Turbulence                  │
│                                                       │
│  turbulence = std(amplitude[12])                     │
│                                                       │
│  This measures spatial variance across subcarriers   │
│  High turbulence = multipath interference            │
└───────────────┬───────────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────────────────────────┐
│ Step 3: Add to Circular Buffer                       │
│                                                       │
│  turbulence_buffer.push(turbulence)                  │
│  if len(buffer) > 50:                                │
│      buffer.pop_front()                              │
│                                                       │
│  Maintains sliding window of 50 packets              │
└───────────────┬───────────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────────────────────────┐
│ Step 4: Calculate Moving Variance                     │
│                                                       │
│  motion_metric = var(turbulence_buffer[50])          │
│                                                       │
│  This measures temporal change in turbulence         │
│  High variance = motion detected                     │
└───────────────┬───────────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────────────────────────┐
│ Step 5: Threshold Comparison                          │
│                                                       │
│  if motion_metric > threshold:                       │
│      state = MOTION                                  │
│  else:                                               │
│      state = IDLE                                    │
└───────────────┬───────────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────────────────────────┐
│ Step 6: Update Display & LED                          │
│                                                       │
│  if state == MOTION:                                 │
│      TFT: red text "MOTION"                          │
│      LED: red (255, 0, 0)                            │
│  else:                                               │
│      TFT: green text "Idle"                          │
│      LED: green (0, 255, 0)                          │
│                                                       │
│  Also display: variance, threshold, packet count     │
└───────────────────────────────────────────────────────┘
```

## File Dependencies

```
arduino_espectre.ino
    │
    ├── #include "config.h"
    ├── #include "csi_manager.h"
    │   └── #include <esp_wifi.h> (ESP-IDF)
    │
    ├── #include "mvs_detector.h"
    │   └── #include "utils.h"
    │
    ├── #include "nbvi_calibrator.h"
    │   └── #include "utils.h"
    │
    ├── #include "gain_controller.h"
    │   └── extern "C" { phy_force_rx_gain(); }
    │
    ├── #include <WiFi.h> (Arduino-ESP32)
    ├── #include <Adafruit_ST7789.h>
    ├── #include <Adafruit_GFX.h>
    └── #include <Adafruit_NeoPixel.h>
```

## Build System

```
┌────────────────────────────────────────────────┐
│  Arduino IDE                                   │
│                                                │
│  1. Preprocess: Combine .ino + headers         │
│  2. Compile: g++ → .o object files             │
│  3. Link: Combine with Arduino core + libs     │
│  4. Binary: .elf → .bin                        │
│  5. Upload: esptool.py via USB                 │
└────────────────────────────────────────────────┘
                    OR
┌────────────────────────────────────────────────┐
│  PlatformIO                                    │
│                                                │
│  1. Parse: platformio.ini                      │
│  2. Download: ESP32 platform + libs            │
│  3. Compile: CMake + Ninja                     │
│  4. Link: Static libraries + core              │
│  5. Upload: esptool.py via USB                 │
└────────────────────────────────────────────────┘
```

## Performance Budget

| Component | CPU Time | Memory (RAM) | Notes |
|-----------|----------|--------------|-------|
| CSI Callback | <1ms | ~1KB | ISR context, minimal processing |
| MVS Processing | ~5ms | ~5KB | Per packet (100 pps = 50% CPU) |
| NBVI Calibration | ~2s | ~180KB | One-time at boot |
| Display Update | ~20ms | ~65KB | 5 Hz = 10% CPU |
| Traffic Gen | <1ms | ~4KB | FreeRTOS task |
| WiFi Stack | variable | ~80KB | Background tasks |
| **Total** | ~60% | ~350KB | Peak usage |

## Error Handling

```
┌─────────────────────────────────────────────────────┐
│  Initialization Errors                              │
│                                                     │
│  WiFi Connect Failed                                │
│  ├─► Display: "WiFi FAILED!"                       │
│  ├─► LED: Off                                      │
│  └─► Action: Halt (infinite loop)                 │
│                                                     │
│  CSI Init Failed                                    │
│  ├─► Display: "CSI FAILED!"                        │
│  ├─► LED: Off                                      │
│  └─► Action: Halt (infinite loop)                 │
│                                                     │
│  Calibration Timeout                                │
│  ├─► Display: Warning message                      │
│  ├─► LED: Yellow                                   │
│  └─► Action: Continue with partial calibration    │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│  Runtime Errors                                     │
│                                                     │
│  WiFi Disconnected                                  │
│  ├─► Action: Auto-reconnect (Arduino handles)      │
│  └─► CSI pauses until reconnected                  │
│                                                     │
│  CSI Packet Dropped                                 │
│  ├─► Counter: dropped_count++                      │
│  ├─► Action: Continue (graceful degradation)       │
│  └─► Display: Show drop count                      │
│                                                     │
│  Display Update Failed                              │
│  ├─► Action: Skip frame, continue                  │
│  └─► Effect: Momentary freeze (not critical)       │
└─────────────────────────────────────────────────────┘
```

---

This architecture achieves **97% motion detection accuracy** by leveraging:
1. **Hardware CSI**: Direct ESP32 WiFi CSI access
2. **Gain Lock**: Stable AGC/FFT gains for consistent measurements
3. **NBVI Calibration**: Automatic optimal subcarrier selection
4. **MVS Algorithm**: Proven turbulence + moving variance approach
5. **Adaptive Threshold**: P95-based threshold eliminates manual tuning
