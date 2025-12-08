[![License](https://img.shields.io/badge/license-GPLv3-blue.svg)](https://github.com/francescopace/espectre/blob/main/LICENSE)
[![ESPHome](https://img.shields.io/badge/ESPHome-Component-blue.svg)](https://esphome.io/)
[![Platform](https://img.shields.io/badge/platform-ESP32-red.svg)](https://www.espressif.com/en/products/socs)
[![Release](https://img.shields.io/github/v/release/francescopace/espectre)](https://github.com/francescopace/espectre/releases/latest)
[![CI](https://img.shields.io/github/actions/workflow/status/francescopace/espectre/ci.yml?branch=main&label=CI)](https://github.com/francescopace/espectre/actions/workflows/ci.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/francescopace/espectre/graph/badge.svg)](https://codecov.io/gh/francescopace/espectre)

# 🛜 ESPectre 👻

**Motion detection system based on Wi-Fi spectre analysis (CSI), with native Home Assistant integration via ESPHome.**

**📰 Featured Article**: Read the complete story behind ESPectre on Medium **[🇮🇹 Italian](https://medium.com/@francesco.pace/come-ho-trasformato-il-mio-wi-fi-in-un-sensore-di-movimento-40053fd83128?source=friends_link&sk=46d9cfa026790ae807ecc291ac5eac67&utm_source=github&utm_medium=readme&utm_campaign=espectre)**, **[🇬🇧 English](https://medium.com/@francesco.pace/how-i-turned-my-wi-fi-into-a-motion-sensor-61a631a9b4ec?sk=c7f79130d78b0545fce4a228a6a79af3&utm_source=github&utm_medium=readme&utm_campaign=espectre)**

**⚠️ Disclaimer**: This is an experimental project for educational and research purposes. The author assumes no responsibility for misuse or damage resulting from the use of this system. Use responsibly and in compliance with applicable laws.

---

## 📑 Table of Contents

- [In 3 Points](#-in-3-points)
- [Mathematical Approach](#-mathematical-approach)
- [What You Need](#-what-you-need)
- [Quick Start](#-quick-start)
- [How It Works](#-how-it-works-simple-version)
- [What You Can Do With It](#-what-you-can-do-with-it)
- [Sensor Placement Guide](#-where-to-place-the-sensor)
- [System Architecture](#️-system-architecture)
- [FAQ](#-faq-for-beginners)
- [Security and Privacy](#-security-and-privacy)
- [Technical Deep Dive](#-technical-deep-dive)
- [Two-Platform Strategy](#-two-platform-strategy)
- [Future Evolutions](#-future-evolutions-ai-approach)
- [References](#-references)
- [Changelog](#-changelog)
- [License](#-license)
- [Author](#-author)

---

## 🎯 In 3 Points

1. **What it does**: Detects movement using Wi-Fi (no cameras, no microphones)
2. **What you need**: A ~€10 ESP32 device (S3 and C6 recommended, other variants supported)
3. **Setup time**: 10-15 minutes

---

## 🔬 Mathematical Approach

**This project uses a pure mathematical approach** based on the **MVS (Moving Variance Segmentation)** algorithm for motion detection and **NBVI (Normalized Baseline Variability Index)** for subcarriers selection.

### Key Points

- ✅ **No ML training required**: Works out-of-the-box with mathematical algorithms
- ✅ **Real-time processing**: Low latency detection on ESP32 hardware
- ✅ **Production-ready**: Focused on reliable motion detection for smart home
- ✅ **R&D platform available**: [Micro-ESPectre](micro-espectre/) provides features extraction for ML research

The mathematical approach provides excellent movement detection without the complexity of ML model training. For advanced features and ML experimentation, see [Micro-ESPectre](micro-espectre/).

---

## 🛒 What You Need

### Hardware

- ✅ **2.4GHz Wi-Fi Router** - the one you already have at home works fine
- ✅ **ESP32 with CSI support** - ESP32-S3 or ESP32-C6 tested. Other variants (ESP32, S2, C3, C5) also supported experimentally.

![3 x ESP32-S3 DevKit bundle with external antennas](images/home_lab.jpg)
*ESP32-S3 DevKit with external antennas*

### Software (All Free)

- ✅ **Home Assistant** (on Raspberry Pi, PC, NAS, or cloud)
- ✅ **ESPHome** (integrated in Home Assistant or standalone)

### Required Skills

- ✅ **Basic YAML knowledge** for configuration
- ✅ **Home Assistant familiarity** (optional but recommended)
- ❌ **NO** programming required
- ❌ **NO** router configuration needed

---

## 🚀 Quick Start

**Setup time**: ~10-15 minutes  
**Difficulty**: Easy (YAML configuration only)

1. **Setup & Installation**: Follow the complete guide in [SETUP.md](SETUP.md)
2. **Tuning**: Optimize for your environment with [TUNING.md](TUNING.md)

---

## 📖 How It Works (Simple Version)

When someone moves in a room, they "disturb" the Wi-Fi waves traveling between the router and the sensor. It's like when you move your hand in front of a flashlight and see the shadow change.

The ESP32 device "listens" to these changes and understands if there's movement.

### Advantages

- ✅ **No cameras** (total privacy)
- ✅ **No wearables needed** (no bracelets or sensors to wear)
- ✅ **Works through walls** (Wi-Fi passes through walls)
- ✅ **Very cheap** (~€10 total)

<details>
<summary>📚 Technical Explanation (click to expand)</summary>

### What is CSI (Channel State Information)?

**Channel State Information (CSI)** represents the physical characteristics of the wireless communication channel between transmitter and receiver. Unlike simple RSSI (Received Signal Strength Indicator), CSI provides rich, multi-dimensional data about the radio channel.

#### What CSI Captures

**Per-subcarrier information:**
- **Amplitude**: Signal strength for each OFDM subcarrier (up to 64)
- **Phase**: Phase shift of each subcarrier
- **Frequency response**: How the channel affects different frequencies

**Environmental effects:**
- **Multipath propagation**: Reflections from walls, furniture, objects
- **Doppler shifts**: Changes caused by movement
- **Temporal variations**: How the channel evolves over time
- **Spatial patterns**: Signal distribution across antennas/subcarriers

#### Why It Works for Movement Detection

When a person moves in an environment, they:
- Alter multipath reflections (new signal paths)
- Change signal amplitude and phase
- Create temporal variations in CSI patterns
- Modify the electromagnetic field structure

These changes are detectable even through walls, enabling **privacy-preserving presence detection** without cameras, microphones, or wearable devices.

</details>

---

## 💡 What You Can Do With It

### Practical Examples

- 🏠 **Home security**: Get an alert if someone enters while you're away
- 👴 **Elderly care**: Monitor activity to detect falls or prolonged inactivity
- 💡 **Smart automation**: Turn on lights/heating only when someone is present
- ⚡ **Energy saving**: Automatically turn off devices in empty rooms
- 👶 **Child monitoring**: Alert if they leave the room during the night
- 🌡️ **Climate control**: Heat/cool only occupied zones

---

## 📍 Where to Place the Sensor

Optimal sensor placement is crucial for reliable movement detection.

### Recommended Distance from Router

**Optimal range: 3-8 meters**

| Distance | Signal | Multipath | Sensitivity | Noise | Recommendation |
|----------|--------|-----------|-------------|-------|----------------|
| < 2m | Too strong | Minimal | Low | Low | ❌ Too close |
| 3-8m | Strong | Good | High | Low | ✅ **Optimal** |
| > 10-15m | Weak | Variable | Low | High | ❌ Too far |

### Placement Tips

✅ **Position sensor in the area to monitor** (not necessarily in direct line with router)  
✅ **Height: 1-1.5 meters** from ground (desk/table height)  
✅ **External antenna**: Use IPEX connector for better reception  
❌ **Avoid metal obstacles** between router and sensor (refrigerators, metal cabinets)  
❌ **Avoid corners** or enclosed spaces (reduces multipath diversity)

---

## ⚙️ System Architecture

### Processing Pipeline

ESPectre uses a simple, focused processing pipeline for motion detection:

```
┌─────────────┐
│  CSI Data   │  Raw Wi-Fi Channel State Information
└──────┬──────┘
       │
       ▼
┌─────────────┐
│    Auto     │  Automatic subcarrier selection (once at boot)
│ Calibration │  Selects optimal 12 subcarriers
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Hampel    │  Turbulence outlier removal
│   Filter    │  (optional, configurable)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│Segmentation │  MVS algorithm
│    (MVS)    │  IDLE ↔ MOTION
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Home        │  Native ESPHome integration
│ Assistant   │  Binary sensor + Movement/Threshold
└─────────────┘
```

### Single or Multiple Sensors

```
┌─────────┐  ┌─────────┐  ┌─────────┐
│ ESP32   │  │ ESP32   │  │ ESP32   │
│ Room 1  │  │ Room 2  │  │ Room 3  │
└────┬────┘  └────┬────┘  └────┬────┘
     │            │            │
     └────────────┴────────────┘
                  │
                  │ ESPHome Native API
                  ▼
         ┌────────────────────┐
         │   Home Assistant   │
         │   (Auto-discovery) │
         └────────────────────┘
```

Each sensor is automatically discovered by Home Assistant with:
- Binary sensor for motion detection
- Movement score sensor
- Adjustable threshold (number entity)

### Automatic Subcarrier Selection

ESPectre implements the **NBVI (Normalized Baseline Variability Index)** algorithm for automatic subcarrier selection, achieving near-optimal performance (F1=97.1%) with **zero manual configuration**.

> ⚠️ **IMPORTANT**: Keep the room **quiet and still** for 10 seconds after device boot. The auto-calibration runs during this time and movement will affect detection accuracy.

NBVI automatically selects the optimal 12 subcarriers from the 64 available in WiFi CSI by analyzing their stability and signal strength during a baseline period. The calibration runs automatically:
- **At first boot** (if no saved configuration exists)
- **After clearing preferences** (factory reset)

---

## ❓ FAQ for Beginners

<details>
<summary>Click to expand FAQ</summary>

**Q: Do I need programming knowledge to use it?**  
A: No! ESPectre uses YAML configuration files. Just copy the example, modify WiFi credentials, and flash.

**Q: Does it work with my router?**  
A: Yes, if your router has 2.4GHz Wi-Fi (virtually all modern routers have it).

**Q: How much does it cost in total?**  
A: Hardware: ~€10 for an ESP32 device (S3/C6 recommended, other variants also work). Software: All free and open source. You'll also need Home Assistant running somewhere (Raspberry Pi ~€35-50, or any existing PC/NAS).

**Q: Do I need to modify anything on the router?**  
A: No! The router works normally. The sensor "listens" to Wi-Fi signals without modifying anything.

**Q: Does it work through walls?**  
A: Yes, the 2.4GHz Wi-Fi signal penetrates drywall. Reinforced concrete walls reduce sensitivity but detection remains possible at reduced distances.

**Q: How many sensors are needed for a house?**  
A: It depends on size. One sensor can monitor ~50 m². For larger homes, use multiple sensors (1 sensor every 50-70 m² for optimal coverage).

**Q: Can it distinguish between people and pets?**  
A: The system uses a 2-state segmentation model (IDLE/MOTION) that identifies generic movement without distinguishing between people, pets, or other moving objects. For more sophisticated classification (people vs pets, activity recognition, gesture detection), trained AI/ML models would be required (see Future Evolutions section).

**Q: Does it work with mesh Wi-Fi networks?**  
A: Yes, it works normally. Make sure the ESP32 connects to the 2.4 GHz band.

**Q: How accurate is the detection?**  
A: Detection accuracy is highly environment-dependent and requires proper tuning. Factors affecting performance include: room layout, wall materials, furniture placement, distance from router (optimal: 3-8m), and interference levels. In optimal conditions with proper tuning, the system provides reliable movement detection. Adjust the `segmentation_threshold` parameter to tune sensitivity for your specific environment.

**Q: What's the power consumption?**  
A: ~500mW typical during continuous operation. The firmware includes support for power optimization, and deep sleep modes can be implemented for battery-powered deployments, though this would require custom modifications to the code.

**Q: If it doesn't work, can I get help?**  
A: Yes, open an [Issue on GitHub](https://github.com/francescopace/espectre/issues) or contact me via email.

</details>

---

## 🔒 Security and Privacy

<details>
<summary>🔐 Privacy, Security & Ethical Considerations (click to expand)</summary>

### Nature of Collected Data

The system collects **anonymous data** related to the physical characteristics of the Wi-Fi radio channel:
- Amplitudes and phases of OFDM subcarriers
- Statistical signal variances
- **NOT collected**: personal identities, communication contents, images, audio

CSI data represents only the properties of the transmission medium and does not contain direct identifying information.

### Privacy Advantages

✅ **No cameras**: Respect for visual privacy  
✅ **No microphones**: No audio recording  
✅ **No wearables**: Doesn't require wearable devices  
✅ **Aggregated data**: Only statistical metrics, not raw identifying data

### ⚠️ Disclaimer and Ethical Considerations

**WARNING**: Despite the intrinsic anonymity of CSI data, this system can be used for:

- **Non-consensual monitoring**: Detecting presence/movement of people without their explicit consent
- **Behavioral profiling**: With advanced AI models, inferring daily life patterns
- **Domestic privacy violation**: Tracking activities inside private homes

### Usage Responsibility

**The user is solely responsible for using this system and must:**

1. ✅ **Obtain explicit consent** from all monitored persons
2. ✅ **Respect local regulations** (GDPR in EU, local privacy laws)
3. ✅ **Clearly inform** about the presence of the sensing system
4. ✅ **Limit use** to legitimate purposes (home security, personal home automation)
5. ✅ **Protect data** with encryption and controlled access
6. ❌ **DO NOT use** for illegal surveillance, stalking, or violation of others' privacy

</details>

---

## 🔬 Technical Deep Dive

![Subcarrier Analysis](images/subcarriers_constellation_diagram.png)
*I/Q constellation diagrams showing the geometric representation of WiFi signal propagation in the complex plane. The baseline (idle) state exhibits a stable, compact pattern, while movement introduces entropic dispersion as multipath reflections change. The radius of these circular patterns represents the amplitude (magnitude) of each subcarrier, while the ring thickness visualizes the signal variance - wider rings indicate higher variability. MVS (Moving Variance Segmentation) analyzes the variance of spatial turbulence across subcarriers to detect motion patterns.*
<details>
<summary>🔬 Signal Processing Pipeline (click to expand)</summary>

### Data Flow

#### 1️⃣ **CSI Acquisition** (ESP32)
- **Native ESP32 CSI API** captures Wi-Fi Channel State Information via callback
- Extracts amplitude and phase data from OFDM subcarriers (up to 64 subcarriers)
- Typical capture rate: ~100 packets/second (configurable via traffic generator)

#### 2️⃣ **Auto-Calibration** (ESP32)
- **Automatic subcarrier selection**: Runs once at first boot
- **NBVI algorithm**: Selects optimal 12 subcarriers from 64 available
- **Zero configuration**: No manual tuning required (F1=97.1%)
- **Adaptive**: Automatically adapts to environment characteristics

#### 3️⃣ **Hampel Filter** (ESP32)
- **Turbulence outlier removal**: Optional filter using MAD (Median Absolute Deviation)
- **Configurable**: Window size (3-11) and threshold (1.0-10.0)
- **Aligned with Micro-ESPectre**: Same defaults (window=7, threshold=4.0)

#### 4️⃣ **Motion Segmentation** (ESP32)
- **Spatial turbulence calculation**: Standard deviation of subcarrier amplitudes
- **Moving Variance Segmentation (MVS)**: Real-time motion detection
- **Adaptive threshold**: Configurable sensitivity (default: 1.0)
- **State machine**: IDLE ↔ MOTION transitions

#### 5️⃣ **Home Assistant Integration**
- **ESPHome Native API** provides automatic device discovery
- **Binary Sensor**: Motion detected (on/off)
- **Movement Sensor**: Current motion intensity value
- **Number Entity**: Adjustable threshold from Home Assistant
- **History**: Automatic logging to database for graphs

![Segmentation Analysis](images/mvs.png)
*Baseline graphs show quiet state (<1), motion graphs show high variance in turbulence (>1)*

</details>

<details>
<summary>📋 Technical Specifications (click to expand)</summary>

### Hardware Requirements

**Tested:** ESP32-S3 or ESP32-C6  
**Experimental:** ESP32, ESP32-S2, ESP32-C3, ESP32-C5

### Software Requirements
- **Framework**: ESPHome with ESP-IDF backend
- **Language**: C++ (component), YAML (configuration)
- **Home Assistant**: 2023.x or newer (recommended)

### Performance Metrics
- **CSI Capture Rate**: 100 packets/second (configurable)
- **Processing Latency**: <10ms per packet
- **CPU Usage**: <20% (ESP32-C6 @ 160MHz)
- **Power Consumption**: ~500mW typical (Wi-Fi active)
- **Detection Range**: 3-8 meters optimal

### Segmentation Accuracy
| Metric | Value | Target |
|--------|-------|--------|
| Recall | 98.1% | >90% ✅ |
| Precision | 100% | - |
| FP Rate | 0.0% | <10% ✅ |
| F1-Score | 99.0% | - |

*Tested on 2000 packets. See [PERFORMANCE.md](PERFORMANCE.md) for detailed metrics and methodology.*

### Limitations
- Currently tested on 2.4 GHz only
- Sensitivity dependent on: wall materials, antenna placement, distances, interference
- Not suitable for environments with very high Wi-Fi traffic
- Cannot distinguish between people, pets, or objects (generic motion detection)
- Cannot count people or recognize specific activities (without ML models)
- Reduced performance through metal obstacles or thick concrete walls

</details>

<details>
<summary>🔧 Multi-Platform Support (click to expand)</summary>

ESPectre supports multiple ESP32 platforms with **dedicated configuration files** in the `examples/` folder:

| Platform | Config File | CPU | WiFi | PSRAM | Status |
|----------|-------------|-----|------|-------|--------|
| **ESP32-C6** | `examples/espectre-c6.yaml` | Single-core RISC-V @ 160MHz | WiFi 6 (802.11ax) | ❌ | ✅ Tested |
| **ESP32-S3** | `examples/espectre-s3.yaml` | Dual-core Xtensa @ 240MHz | WiFi 4 (802.11n) | ✅ 8MB | ✅ Tested |
| **ESP32-C5** | `examples/espectre-c5.yaml` | Single-core RISC-V @ 240MHz | WiFi 6 (802.11ax) | ❌ | ⚠️ Experimental |
| **ESP32-C3** | `examples/espectre-c3.yaml` | Single-core RISC-V @ 160MHz | WiFi 4 (802.11n) | ❌ | ⚠️ Experimental |
| **ESP32-S2** | `examples/espectre-s2.yaml` | Single-core Xtensa @ 240MHz | WiFi 4 (802.11n) | Optional | ⚠️ Experimental |
| **ESP32** | `examples/espectre-esp32.yaml` | Dual-core Xtensa @ 240MHz | WiFi 4 (802.11n) | Optional | ⚠️ Experimental |

**Recommendations**:
- **ESP32-C6**: Best for WiFi 6 environments, standard motion detection
- **ESP32-S3**: Best for advanced applications, future ML features (more memory)
- **ESP32-C5**: Similar to C6 with WiFi 6 support, newer chip
- **ESP32-C3**: Budget-friendly option, compact form factor
- **ESP32-S2/ESP32**: Use if you already have one, but S3/C6 recommended for new purchases

> ⚠️ **Experimental platforms**: ESP32, ESP32-S2, ESP32-C3, and ESP32-C5 have CSI support but have not been extensively tested. Please report your results on [GitHub Discussions](https://github.com/francescopace/espectre/discussions)!

See [SETUP.md](SETUP.md) for platform-specific configurations and PSRAM settings.

</details>

---

## 🎯 Two-Platform Strategy

This project follows a **dual-platform approach** to balance innovation speed with production stability:

### 🏠 ESPectre (This Repository) - Production Platform

**Target**: End users, smart home enthusiasts, Home Assistant users

- **ESPHome component** with native Home Assistant integration
- **YAML configuration** - no programming required
- **Auto-discovery** - devices appear automatically in Home Assistant
- **Production-ready** - stable, tested, easy to deploy
- **Demonstrative** - showcases research results in a user-friendly package

### 🔬 [Micro-ESPectre](micro-espectre/) - R&D Platform

**Target**: Researchers, developers, academic/industrial applications

- **Python/MicroPython** implementation for rapid prototyping
- **MQTT-based** - flexible integration (not limited to Home Assistant)
- **Fast iteration** - test new algorithms in seconds, not minutes
- **Analysis tools** - comprehensive suite for CSI data analysis
- **Use cases**: Academic research, industrial sensing, algorithm development

### Development Flow

```
┌─────────────────────┐     Validated      ┌─────────────────────┐
│   Micro-ESPectre    │ ─────────────────► │      ESPectre       │
│   (R&D Platform)    │    algorithms      │ (Production Platform)│
│                     │                    │                      │
│ • Fast prototyping  │                    │ • ESPHome component  │
│ • Algorithm testing │                    │ • Home Assistant     │
│ • Data analysis     │                    │ • End-user ready     │
│ • MQTT flexibility  │                    │ • Native API         │
└─────────────────────┘                    └─────────────────────┘
```

**Innovation cycle**: New features and algorithms are first developed and validated in Micro-ESPectre (Python), then ported to ESPectre (C++) once proven effective.

---

## 🤖 Advanced Applications & Machine Learning

ESPectre focuses on **production-ready motion detection** using mathematical algorithms (MVS + NBVI).

For **advanced applications** requiring feature extraction and machine learning, see **[Micro-ESPectre](micro-espectre/)** which provides:
- ✅ CSI features for ML training
- ✅ Analysis tools for dataset collection
- ✅ Scientific references and ML approaches
- ✅ Public datasets information

Micro-ESPectre gives you the fundamentals for:
- 🔬 **People counting**
- 🏃 **Activity recognition** (walking, falling, sitting, sleeping)
- 📍 **Localization and tracking**
- 👋 **Gesture recognition**

📚 **Jump to section**: [Micro-ESPectre - Advanced Applications & Machine Learning](micro-espectre/README.md#-advanced-applications--machine-learning)

---

## 📚 References

For a comprehensive list of scientific references, academic papers, and research resources related to Wi-Fi sensing and CSI-based applications, see the **[Scientific References](micro-espectre/README.md#-scientific-references)** section in the Micro-ESPectre documentation.

Micro-ESPectre, as the R&D platform, maintains the complete bibliography of research papers covering:
- Mathematical signal processing approaches
- Machine learning and deep learning techniques
- Public datasets for CSI-based applications
- IEEE 802.11bf Wi-Fi Sensing standard

---

## 📋 Changelog

For a detailed history of changes, new features, and improvements, see the [CHANGELOG.md](CHANGELOG.md).

---

## 📄 License

This project is released under the **GNU General Public License v3.0 (GPLv3)**.

GPLv3 ensures that:
- ✅ The software remains free and open source
- ✅ Anyone can use, study, modify, and distribute it
- ✅ Modifications must be shared under the same license
- ✅ Protects end-user rights and software freedom

See [LICENSE](LICENSE) for the full license text.

---

## 👤 Author

**Francesco Pace**  
📧 Email: [francesco.pace@gmail.com](mailto:francesco.pace@gmail.com)  
💼 LinkedIn: [linkedin.com/in/francescopace](https://www.linkedin.com/in/francescopace/)  
🛜 Project: [ESPectre](https://github.com/francescopace/espectre)
