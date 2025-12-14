[![License](https://img.shields.io/badge/license-GPLv3-blue.svg)](https://github.com/francescopace/espectre/blob/main/LICENSE)
[![ESPHome](https://img.shields.io/badge/ESPHome-Component-blue.svg)](https://esphome.io/)
[![Platform](https://img.shields.io/badge/platform-ESP32-red.svg)](https://www.espressif.com/en/products/socs)
[![Release](https://img.shields.io/github/v/release/francescopace/espectre)](https://github.com/francescopace/espectre/releases/latest)
[![CI](https://img.shields.io/github/actions/workflow/status/francescopace/espectre/ci.yml?branch=main&label=CI)](https://github.com/francescopace/espectre/actions/workflows/ci.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/francescopace/espectre/graph/badge.svg)](https://codecov.io/gh/francescopace/espectre)

# 🛜 ESPectre 👻

**Motion detection system based on Wi-Fi spectre analysis (CSI), with native Home Assistant integration via ESPHome.**

**📰 Featured Article**: Read the complete story behind ESPectre on Medium **[🇮🇹 Italian](https://medium.com/@francesco.pace/come-ho-trasformato-il-mio-wi-fi-in-un-sensore-di-movimento-40053fd83128?source=friends_link&sk=46d9cfa026790ae807ecc291ac5eac67&utm_source=github&utm_medium=readme&utm_campaign=espectre)**, **[🇬🇧 English](https://medium.com/@francesco.pace/how-i-turned-my-wi-fi-into-a-motion-sensor-61a631a9b4ec?sk=c7f79130d78b0545fce4a228a6a79af3&utm_source=github&utm_medium=readme&utm_campaign=espectre)**


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
- [Documentation](#-documentation)
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

- ✅ **No ML training required**: Works out-of-the-box with mathematical algorithms
- ✅ **Real-time processing**: Low latency detection on ESP32 hardware
- ✅ **Production-ready**: Focused on reliable motion detection for smart home
- ✅ **R&D platform available**: [Micro-ESPectre](micro-espectre/) provides features extraction for ML research

📚 **For algorithm details** (MVS, NBVI, Hampel filter), see [ALGORITHMS.md](micro-espectre/ALGORITHMS.md).

---

## 🛒 What You Need

### Hardware

- ✅ **2.4GHz Wi-Fi Router** - the one you already have at home works fine
- ✅ **ESP32 with CSI support** - ESP32-C6, ESP32-S3, ESP32-C3 or other variants. See [SETUP.md](SETUP.md) for the complete platform comparison table.

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

📚 **Want to understand the technical details?** See [ALGORITHMS.md](micro-espectre/ALGORITHMS.md) for CSI explanation and signal processing documentation.

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

ESPectre implements the **NBVI (Normalized Baseline Variability Index)** algorithm for automatic subcarrier selection, achieving near-optimal performance (F1=97.6%) with **zero manual configuration**.

> ⚠️ **IMPORTANT**: Keep the room **quiet and still** for 10 seconds after device boot. The auto-calibration runs during this time and movement will affect detection accuracy.

📚 **For NBVI algorithm details**, see [ALGORITHMS.md](micro-espectre/ALGORITHMS.md#nbvi-automatic-subcarrier-selection).

---

## ❓ FAQ for Beginners

<details>
<summary>Click to expand FAQ</summary>

**Q: Do I need programming knowledge to use it?**  
A: No! ESPectre uses YAML configuration files. Just download the example, flash it, and configure WiFi via the ESPHome app or web interface.

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

📚 **For algorithm details** (MVS, NBVI, Hampel filter), see [ALGORITHMS.md](micro-espectre/ALGORITHMS.md).

📊 **For performance metrics** (confusion matrix, F1-score, benchmarks), see [PERFORMANCE.md](PERFORMANCE.md).

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

Micro-ESPectre gives you the fundamentals for:
- 🔬 **People counting**
- 🏃 **Activity recognition** (walking, falling, sitting, sleeping)
- 📍 **Localization and tracking**
- 👋 **Gesture recognition**

### Development Flow

```
┌─────────────────────┐     Validated      ┌──────────────────────┐
│   Micro-ESPectre    │ ─────────────────► │      ESPectre        │
│   (R&D Platform)    │    algorithms      │ (Production Platform)│
│                     │                    │                      │
│ • Fast prototyping  │                    │ • ESPHome component  │
│ • Algorithm testing │                    │ • Home Assistant     │
│ • Data analysis     │                    │ • End-user ready     │
│ • MQTT flexibility  │                    │ • Native API         │
└─────────────────────┘                    └──────────────────────┘
```

**Innovation cycle**: New features and algorithms are first developed and validated in Micro-ESPectre (Python), then ported to ESPectre (C++) once proven effective.

---

## 📚 Documentation

### ESPectre (Production)

| Document | Description |
|----------|-------------|
| [Intro](README.md) | Project overview, quick start, FAQ |
| [Setup Guide](SETUP.md) | Installation and configuration with ESPHome |
| [Tuning Guide](TUNING.md) | Parameter tuning for optimal detection |
| [Performance](PERFORMANCE.md) | Benchmarks, confusion matrix, F1-score |
| [Test Suite](test/README.md) | PlatformIO Unity test documentation |

### Micro-ESPectre (R&D)

| Document | Description |
|----------|-------------|
| [Intro](micro-espectre/README.md) | R&D platform overview, CLI, MQTT, Web Monitor |
| [Algorithms](micro-espectre/ALGORITHMS.md) | Scientific documentation of MVS, NBVI, Hampel filter |
| [Analysis Tools](micro-espectre/tools/README.md) | CSI analysis and optimization scripts |
| [ML Data Collection](micro-espectre/ML_DATA_COLLECTION.md) | Building labeled datasets for machine learning |

📋 **[Changelog](CHANGELOG.md)** - Version history and release notes

📚 **[Scientific References](micro-espectre/README.md#-scientific-references)** - Comprehensive list of scientific references, academic papers, and research resources

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
