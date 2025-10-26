![License](https://img.shields.io/badge/license-GPLv3-blue.svg)
![C](https://img.shields.io/badge/C-ESP--IDF-orange.svg)
![Platform](https://img.shields.io/badge/platform-ESP32--S3-red.svg)
![Status](https://img.shields.io/badge/status-experimental-orange.svg)

# 🛜 ESPectre 👻

Presence detection system based on Wi-Fi spectre analysis (CSI).  
Uses **ESP32-S3** as sensor and **Home Assistant** (or MQTT broker) as collection server.

---

## 📖 What is CSI (Channel State Information)?

**Channel State Information (CSI)** represents the physical characteristics of the wireless communication channel between transmitter and receiver. Unlike simple RSSI (Received Signal Strength Indicator), CSI provides detailed information about:

- **Amplitude and phase** of each OFDM subcarrier
- **Frequency response** of the radio channel
- **Multipath effects** caused by reflections, diffractions, and scattering

When a person moves in an environment, they modify the electromagnetic field causing measurable variations in CSI. This allows detecting presence and movement without wearable devices or cameras, ensuring privacy and coverage through walls.

**Important**: ESPectre captures CSI from **all Wi-Fi traffic** in the environment, not just from a specific network. This means movement detection works based on any Wi-Fi activity in the area (routers, devices, neighbors' networks), making the system more robust and reliable.

---

## 💡 Practical Use Cases

- **Home security**: Intrusion detection when you're away
- **Elderly care**: Activity monitoring to detect falls or prolonged inactivity
- **Smart automation**: Lights/heating activation based on real presence
- **Energy saving**: Automatic device shutdown in empty rooms
- **Child monitoring**: Alerts if they leave the room during the night
- **HVAC optimization**: Climate control only in occupied zones

---

## 🚀 Quick Start

**What you need**: ESP32-S3 DevKit, USB-C cable, Wi-Fi router, Home Assistant  
**Setup time**: ~15-20 minutes  
**Difficulty**: Beginner (pre-compiled binary available)

![ESP32-S3 DevKit bundle with external antennas](images/DUBEUYEW%203pcs%20ESP32-S3%20DevKitC-1%20N16R8.jpg)
*ESP32-S3 DevKit bundle with external antennas (recommended for better reception)*

**[Complete Setup Guide →](SETUP.md)**

---

### Wi-Fi Router Requirements

For CSI capture, the router must support at least:

📡 **Minimum standard**: **802.11n (Wi-Fi 4)** or higher  
📡 **Band**: **2.4 GHz** (ESP32-S3 limitation)  
📡 **Mode**: Any standard consumer router is compatible

**Important note**: CSI is extracted **receiver-side** (ESP32-S3), not from the router. The router simply acts as a transmitter of standard Wi-Fi packets. No special configurations or modified firmware are needed on the router.

---

## 📍 Sensor Placement

Optimal sensor placement is crucial for reliable movement detection. The position relative to the Wi-Fi router significantly affects detection performance.

### Recommended Distance from Router

**Optimal range: 3-8 meters**

| Distance | Signal Quality | Detection Performance | Recommendation |
|----------|---------------|----------------------|----------------|
| < 2m | Too strong, stable | ❌ Low sensitivity | Too close |
| 3-8m | Strong with multipath | ✅ Optimal detection | **Recommended** |
| > 10-15m | Weak, noisy | ❌ Too many false positives | Too far |

### Why Distance Matters

**Too Close (< 2m):**
- Signal is too strong and stable
- Minimal multipath reflections
- Movement causes small CSI variations
- Result: Poor detection sensitivity

**Too Far (> 10-15m):**
- Signal is weak and unstable
- High noise levels
- Low Signal-to-Noise Ratio (SNR)
- Result: Difficult to distinguish movement from noise

**Optimal Distance (3-8m):**
- Strong signal with multipath reflections
- Movement significantly alters signal path
- Good Signal-to-Noise Ratio
- Result: Reliable and sensitive detection

### Placement Tips

✅ **Position sensor in the area to monitor** (not necessarily in direct line with router)  
✅ **Height: 1-1.5 meters** from ground (desk/table height)  
✅ **Avoid metal obstacles** between router and sensor (refrigerators, metal cabinets)  
✅ **External antenna**: Use IPEX connector for better reception  
❌ **Avoid corners** or enclosed spaces (reduces multipath diversity)

### Testing and Fine-Tuning

After installation:
1. Start with sensor ~5 meters from router
2. Move around and observe detection values
3. Adjust position based on results:
   - Too sensitive (false positives) → Move slightly farther
   - Not sensitive enough → Move slightly closer
4. Fine-tune parameters via menuconfig if needed

---

## ⚙️ Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         ESP32-S3 SENSOR                                 │
│  ┌──────────────┐    ┌─────────────────┐    ┌──────────────────────┐    │
│  │   ESP32      │───▶│  CSI Processing │───▶│   MQTT Publisher     │    │
│  │  CSI API     │    │   Pipeline      │    │  (esp-mqtt)          │    │
│  └──────────────┘    └─────────────────┘    └──────────────────────┘    │
│         │                     │                        │                │
│    Native API          ┌──────┴──────┐          JSON Payload            │
│    Callback            │  Pipeline:  │          movement: 0.0-1.0       │
│                        │  • Filters  │          confidence: 0.0-1.0     │
│                        │  • 17 Feat. │          state: idle/micro/      │
│                        │  • Weighted │                detected/intense  │
│                        │  • 5 States │          + 17 features           │
│                        └─────────────┘                                  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ MQTT (TCP:1883)
                                    │ Topic: home/espectre/node1
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         MQTT BROKER                                     │
│                      (Mosquitto / HA Add-on)                            │
│                                                                         │
│  • Message routing and queuing                                          │
│  • Authentication & authorization                                       │
│  • TLS/SSL encryption (optional)                                        │
│  • Retained messages for last state                                     │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ MQTT Subscribe
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       HOME ASSISTANT                                    │
│  ┌──────────────┐    ┌─────────────────┐    ┌──────────────────────┐    │
│  │ MQTT Sensor  │───▶│   Automations   │───▶│   Actions            │    │
│  │ Integration  │    │   & Scripts     │    │   • Notifications    │    │
│  └──────────────┘    └─────────────────┘    │   • Lights control   │    │
│         │                     │             │   • Climate control  │    │
│    Entity:                Triggers:         │   • Security alerts  │    │
│    sensor.movement       • Numeric state    └──────────────────────┘    │
│    State: 0.65           • Template                     │               │
│    Attributes:           • Time pattern                 │               │
│    • confidence                                         ▼               │
│    • baseline                                  ┌──────────────────┐     │
│    • threshold                                 │   Dashboard      │     │
│                                                │   • History      │     │
│                                                │   • Graphs       │     │
│                                                │   • Cards        │     │
│                                                └──────────────────┘     │
└─────────────────────────────────────────────────────────────────────────┘
```

### Data Flow Pipeline

#### 1️⃣ **CSI Acquisition** (ESP32-S3)
- **Native ESP32 CSI API** captures Wi-Fi Channel State Information via callback
- Extracts amplitude and phase data from OFDM subcarriers (up to 64 subcarriers)
- Typical capture rate: ~10-100 packets/second depending on Wi-Fi traffic

#### 2️⃣ **Signal Processing** (ESP32-S3)
The `espectre.c` firmware applies an advanced multi-stage processing pipeline:

**Stage 1: Advanced Filters** (Optional, configurable)
- **Hampel Filter**: Outlier removal using MAD (Median Absolute Deviation)
- **Savitzky-Golay Filter**: Polynomial smoothing (enabled by default)
- **Adaptive Normalization**: Running statistics with Welford's algorithm

**Stage 2: Feature Extraction** (17 mathematical features)
- **Time-domain** (6): Mean, Variance, Skewness, Kurtosis, Entropy, IQR
- **Spatial** (3): Spatial variance, correlation, gradient across subcarriers
- **Temporal** (3): Autocorrelation, zero-crossing rate, peak rate
- **Multi-window** (3): Variance on short/medium/long time windows
- **Derivative** (2): First and second derivatives

**Stage 3: Multi-Criteria Detection**
- Weighted scoring from 4 most discriminant features
- Configurable weights (Variance 35%, Spatial Gradient 30%, Variance Short 25%, IQR 10%)
- Optimized ranges based on empirical analysis

**Stage 4: Granular State Machine** (5 states)
- **IDLE**: No movement (score < 0.10)
- **MICRO**: Minimal movement (score 0.10-0.50)
- **DETECTED**: Clear movement (score 0.50-0.70)
- **INTENSE**: Strong movement (score > 0.70)
- **CALIBRATING**: Initial calibration phase
- Debouncing: Requires 2 consecutive detections (default)
- Persistence: 5 seconds timeout before downgrading state
- Hysteresis: Prevents state flickering

#### 3️⃣ **MQTT Publishing** (ESP32-S3 → Broker)
- Publishes JSON payload every 1 second (configurable)
- QoS level 0 (fire-and-forget) for low latency
- Retained message option for last known state
- Automatic reconnection on connection loss

**Example Payload:**
```json
{
  "movement": 0.65,
  "confidence": 0.82,
  "state": "detected",
  "baseline": 0.12,
  "threshold": 0.35,
  "timestamp": 1730066400
}
```

#### 4️⃣ **Home Assistant Integration**
- **MQTT Sensor** subscribes to topic and creates entity
- **State**: Primary `movement` value (0.0-1.0)
- **Attributes**: All other metrics available for conditions
- **History**: Automatic logging to database for graphs

#### 5️⃣ **Automation & Actions**
Trigger automations based on:
- **Numeric state**: `movement > 0.6` (active movement)
- **Confidence level**: `confidence > 0.7` (high certainty)
- **State changes**: `idle` → `detected` transitions
- **Time patterns**: Only during specific hours
- **Template conditions**: Complex logic combining multiple sensors

**Example Use Cases:**
- 🚨 Security alert when movement detected while away
- 💡 Turn on lights when entering room
- 🌡️ Adjust thermostat based on occupancy
- 📱 Push notification for unusual activity patterns
- ⏰ Disable alarm if movement detected (person woke up)

### Multi-Sensor Deployment

For larger spaces, deploy multiple sensors:

```
┌─────────┐  ┌─────────┐  ┌─────────┐
│ Sensor1 │  │ Sensor2 │  │ Sensor3 │
│ Kitchen │  │ Bedroom │  │  Living │
└────┬────┘  └────┬────┘  └────┬────┘
     │            │            │
     └────────────┴────────────┘
                  │
            MQTT Broker
                  │
          Home Assistant
     (Aggregates all sensors)
```

Each sensor publishes to its own topic:
- `home/espectre/kitchen`
- `home/espectre/bedroom`
- `home/espectre/living`

Home Assistant can then:
- Monitor each room independently
- Create group sensors for whole-house occupancy
- Implement zone-based automations
- Track movement patterns across rooms

---

## ❓ FAQ

**Q: Does it work through walls?**  
A: Yes, the 2.4GHz Wi-Fi signal penetrates drywall. Reinforced concrete walls reduce sensitivity but detection remains possible at reduced distances.

**Q: How many sensors are needed for a house?**  
A: It depends on size and layout. One sensor can monitor ~50 m². For larger homes, deploy multiple sensors (1 sensor every 50-70 m² for optimal coverage).

**Q: Can it distinguish between people and pets?**  
A: The basic implementation no, it only detects generic movement. With trained AI models, it's possible to distinguish movement patterns (see Roadmap).

**Q: Does it consume a lot of Wi-Fi bandwidth?**  
A: No, MQTT traffic is minimal (~1 KB/s per sensor). Network impact is negligible.

**Q: Does it work with mesh Wi-Fi networks?**  
A: Yes, it works normally. Make sure the ESP32 connects to the 2.4 GHz band.

**Q: Is a dedicated server necessary?**  
A: No, Home Assistant can run on Raspberry Pi, NAS, or cloud. Alternatively, just an MQTT broker (Mosquitto) on any device is sufficient.

**Q: How accurate is the detection?**  
A: It depends on the environment. In open spaces: excellent. With many metal obstacles or thick walls: reduced. Parameter calibration is fundamental.

**Q: What's the power consumption?**  
A: ~500mW typical. Can be reduced with deep sleep modes for battery operation.

---

## 🔒 Security and Privacy

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

### Security Recommendations

- **Encrypt MQTT communications** (TLS/SSL)
- **Authenticate MQTT clients** with username/password
- **Isolate the network** of sensors from the main network
- **Limit access** to MQTT broker only from authorized devices
- **Document installation** and inform residents/visitors
- **Implement data retention policies** (automatic deletion of old data)

### Regulatory Compliance

In Europe (GDPR), this system may require:
- Privacy notice for residents and visitors
- Data processing register
- Privacy impact assessment (DPIA) for extended uses
- Explicit consent for monitoring third parties

**Consult a privacy specialist before deployment in shared or public environments.**

---

## 📋 Technical Specifications

### Hardware Requirements
- **Board**: ESP32-S3-DevKitC-1 N16R8
- **Flash**: 16MB
- **PSRAM**: 8MB
- **Wi-Fi**: 802.11 b/g/n (2.4 GHz only)
- **Antenna**: Built-in PCB antenna + IPEX connector for external
- **Power**: USB-C 5V or 3.3V via pins

### Software Requirements
- **Framework**: ESP-IDF v5.x
- **Language**: C
- **Build System**: CMake
- **Flash Tool**: esptool.py

### Limitations
- Works only on 2.4 GHz band (ESP32-S3 hardware limitation)
- Sensitivity dependent on: wall materials, antenna placement, distances, interference
- Not suitable for environments with very high Wi-Fi traffic

---

## 🗺️ Roadmap

### Phase 1: Foundation ✅ (Completed)
- [x] ESP32-S3 implementation with movement detection
- [x] MQTT and Home Assistant integration
- [x] Complete documentation and setup guide
- [x] **Advanced Feature Extraction** (17 mathematical features)
  - [x] Time-domain features (mean, variance, skewness, kurtosis, entropy, IQR)
  - [x] Spatial features (variance, correlation, gradient across subcarriers)
  - [x] Temporal features (autocorrelation, zero-crossing rate, peak rate)
  - [x] Multi-window analysis (short/medium/long variance)
- [x] **Advanced Filters** (Hampel, Savitzky-Golay, Adaptive Normalization)
  - [x] Hampel filter for outlier removal (MAD method)
  - [x] Savitzky-Golay polynomial smoothing (enabled by default)
  - [x] Adaptive normalization with Welford's algorithm
- [x] **Multi-Criteria Detection** with weighted scoring
  - [x] 4 most discriminant features with optimized weights
  - [x] Configurable weights runtime via MQTT
  - [x] Empirically optimized ranges
- [x] **Granular State Machine** (5 states: IDLE, MICRO, DETECTED, INTENSE, CALIBRATING)
- [x] **Automatic Calibration** with `analyze` command
  - [x] Statistical analysis (percentiles, recommended threshold)
  - [x] Runtime parameter optimization
- [x] Anti-false positive system (debouncing, hysteresis, persistence)
- [x] CLI tool for configuration and monitoring

### Phase 2: Core Improvements 🚧
- [ ] **Web Dashboard**: Real-time CSI data visualization with interactive charts
  - Live amplitude heatmaps per subcarrier
  - Historical graphs and statistics
  - System status monitoring
  - Configuration interface
- [ ] **Automatic Calibration**: Intelligent parameter tuning
  - Environment detection and auto-configuration
  - Optimal threshold calculation
  - Performance metrics and recommendations
- [ ] **Performance Optimizations**: Enhanced efficiency
  - Parallel processing for CSI analysis
  - Intelligent caching mechanisms
  - Reduced MQTT overhead with batching
  - Memory leak prevention and optimization
- [ ] **Modular Architecture**: Plugin-based system
  - Configurable processing pipeline
  - Custom analyzer plugins
  - Separation of CSI core from analysis modules
  - Hot-reload configuration support

### Phase 3: Data Collection & ML Integration 🤖
- [ ] **Labeled Data Collection Mode**: Dataset generation for ML training
  - Multi-class scenario support (0, 1, 2+ people)
  - Automatic annotation with timestamps
  - Export to standard formats (HDF5, CSV, TFRecord, NPY)
  - Data validation and quality checks
- [ ] **People Counting Model**: ML-based occupancy estimation
  - CNN-LSTM architecture for temporal patterns
  - Multi-class classification (0, 1, 2, 3+ people)
  - Proof of concept with public datasets
  - Accuracy benchmarking and validation
- [ ] **Local Inference Engine**: Edge AI deployment
  - TensorFlow Lite integration for Raspberry Pi 4
  - ONNX Runtime support
  - Model quantization for performance
  - Preprocessing pipeline for ML
- [ ] **Activity Recognition**: Human activity classification
  - Walking, sitting, standing, falling detection
  - Temporal sequence analysis
  - Pre-trained models for common scenarios
  - Transfer learning support
- [ ] **Continuous Learning**: Model improvement over time
  - Online learning capabilities
  - Model versioning and A/B testing
  - Performance monitoring and drift detection

### Phase 4: Multi-Sensor & Scalability 🌐
- [ ] **Multi-Sensor Fusion**: Coordinated sensing network
  - Data aggregation from multiple sensors
  - Consensus algorithms for reliability
  - Distributed processing architecture
  - Sensor health monitoring
- [ ] **Localization & Tracking**: Position estimation
  - Triangulation algorithms
  - Person trajectory tracking
  - Zone-based presence mapping
  - Privacy-preserving location data
- [ ] **Mobile App**: Remote configuration and monitoring
  - Sensor discovery and setup
  - Real-time alerts and notifications
  - Historical data visualization
  - Multi-sensor management
- [ ] **Advanced Event System**: Complex pattern recognition
  - Temporal event sequences (enter → sit → exit)
  - Anomaly detection (falls, unusual patterns)
  - Intelligent alerting with priority levels
  - Integration with other IoT sensors

### Phase 5: Standardization & Future 🔮
- [ ] **IEEE 802.11bf Preparation**: Next-gen Wi-Fi Sensing
  - Abstraction layer for CSI acquisition
  - Support for standardized CSI formats
  - Compatibility with 802.11bf hardware
  - Migration path from Nexmon to native sensing
- [ ] **Multi-Band Support**: Extended frequency coverage
  - 5 GHz band support (when hardware available)
  - 6 GHz Wi-Fi 6E integration
  - Frequency diversity for improved accuracy
- [ ] **Gesture Recognition**: Touchless control
  - Hand gesture classification
  - Sign language recognition
  - Smart home control via gestures
  - Real-time inference with low latency
- [ ] **Vital Signs Monitoring**: Health sensing capabilities
  - Breathing rate detection
  - Heart rate estimation (with 802.11bf)
  - Sleep quality monitoring
  - Elderly care applications

---

## 🤖 Future Evolutions: AI-Based approach

The current implementation uses an **advanced mathematical approach** with 17 features and multi-criteria detection to identify movement patterns. While this provides excellent results without requiring ML training, scientific research has shown that **Machine Learning** and **Deep Learning** techniques can extract even richer information from CSI data for more complex tasks like people counting, activity recognition, and gesture detection.

### Advanced Applications

#### 1. **People Counting**
Classification or regression models can estimate the number of people present in an environment by analyzing complex patterns in CSI.

**References:**
- *Wang et al.* (2017) - "Device-Free Crowd Counting Using WiFi Channel State Information" - IEEE INFOCOM
- *Xi et al.* (2016) - "Electronic Frog Eye: Counting Crowd Using WiFi" - IEEE INFOCOM

#### 2. **Activity Recognition**
Neural networks (CNN, LSTM, Transformer) can classify human activities like walking, falling, sitting, sleeping.

**References:**
- *Wang et al.* (2015) - "Understanding and Modeling of WiFi Signal Based Human Activity Recognition" - ACM MobiCom
- *Yousefi et al.* (2017) - "A Survey on Behavior Recognition Using WiFi Channel State Information" - IEEE Communications Magazine
- *Zhang et al.* (2019) - "WiFi-Based Indoor Robot Positioning Using Deep Neural Networks" - IEEE Access

#### 3. **Localization and Tracking**
Deep learning algorithms can estimate position and trajectory of moving people.

**References:**
- *Wang et al.* (2016) - "CSI-Based Fingerprinting for Indoor Localization: A Deep Learning Approach" - IEEE Transactions on Vehicular Technology
- *Chen et al.* (2018) - "WiFi CSI Based Passive Human Activity Recognition Using Attention Based BLSTM" - IEEE Transactions on Mobile Computing

#### 4. **Gesture Recognition**
Models trained on CSI temporal sequences can recognize hand gestures for touchless control.

**References:**
- *Abdelnasser et al.* (2015) - "WiGest: A Ubiquitous WiFi-based Gesture Recognition System" - IEEE INFOCOM
- *Jiang et al.* (2020) - "Towards Environment Independent Device Free Human Activity Recognition" - ACM MobiCom

### Available Public Datasets

- **UT-HAR**: Human Activity Recognition dataset (University of Texas)
- **Widar 3.0**: Gesture recognition dataset with CSI
- **SignFi**: Sign language recognition dataset
- **FallDeFi**: Fall detection dataset

## 🛜 Future Evolutions: Standardized Wi-Fi Sensing

Currently, CSI extraction requires firmware modifications (like Nexmon) because it's not a standard feature. However, the **IEEE 802.11bf (Wi-Fi Sensing)** standard will radically change this scenario.

### IEEE 802.11bf - Wi-Fi Sensing

The **802.11bf** standard was **[officially published on September 26, 2025](https://standards.ieee.org/ieee/802.11bf/11574/)**, introducing **Wi-Fi Sensing** as a native feature of the Wi-Fi protocol. Main characteristics:

🔹 **Native sensing**: Detection of movements, gestures, presence, and vital signs  
🔹 **Interoperability**: Standardized support across different vendors  
🔹 **Optimizations**: Specific protocols to reduce overhead and power consumption  
🔹 **Privacy by design**: Privacy protection mechanisms integrated into the standard  
🔹 **Greater precision**: Improvements in temporal and spatial granularity  
🔹 **Existing infrastructure**: Works with already present Wi-Fi infrastructure

### Adoption Status (2025)

**Market**: The Wi-Fi Sensing market is in its early stages and is expected to experience significant growth in the coming years as the 802.11bf standard enables native sensing capabilities in consumer devices.

**Hardware availability**: 
- ⚠️ **Consumer routers**: Currently **there are no widely available consumer routers** with native 802.11bf support
- 🏢 **Commercial/industrial**: Experimental devices and integrated solutions already in use
- 🔧 **Hardware requirements**: Requires multiple antennas, Wi-Fi 6/6E/7 support, and AI algorithms for signal processing

**Expected timeline**:
- **2025-2026**: First implementations in enterprise and premium smart home devices
- **2027-2028**: Diffusion in high-end consumer routers
- **2029+**: Mainstream adoption in consumer devices

### Future Benefits for Wi-Fi Sensing

When 802.11bf is widely adopted, applications like this project will become:
- **More accessible**: No need for specialized hardware or modified firmware
- **More reliable**: Standardization ensures predictable behavior
- **More efficient**: Protocols optimized for continuous sensing
- **More secure**: Privacy mechanisms integrated at the standard level
- **More powerful**: Ability to detect even vital signs (breathing, heartbeat)

**Perspective**: In the next 3-5 years, routers and consumer devices will natively support Wi-Fi Sensing, making projects like this implementable without specialized hardware or firmware modifications. This will open new possibilities for smart home, elderly care, home security, health monitoring, and advanced IoT applications.

**For now**: Solutions like this project based on **[Nexmon CSI](https://github.com/seemoo-lab/nexmon_csi)** remain the most accessible and economical way to experiment with Wi-Fi Sensing.

---

## 📚 References

### Standards and Specifications
- [IEEE 802.11bf - Wi-Fi Sensing Standard](https://standards.ieee.org/ieee/802.11bf/11574/)

### ESP32 Documentation
- [ESP-IDF Programming Guide](https://docs.espressif.com/projects/esp-idf/en/latest/esp32s3/)
- [ESP32 CSI Documentation](https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-reference/network/esp_wifi.html)

### Tools
- [Home Assistant MQTT Integration](https://www.home-assistant.io/integrations/mqtt/)
- [ESP-IDF](https://github.com/espressif/esp-idf)

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

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a branch for your feature (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 👤 Author

**Francesco Pace**  
📧 Email: [francesco.pace@gmail.com](mailto:francesco.pace@gmail.com)  
💼 LinkedIn: [linkedin.com/in/francescopace](https://www.linkedin.com/in/francescopace/)  
🐙 GitHub: [@francescopace](https://github.com/francescopace)  
🛜 Project: [ESPectre](https://github.com/francescopace/espectre)

---

## 📧 Contact and Support

For questions, issues, or suggestions, open an [Issue](../../issues) on GitHub.

---

**⚠️ Final Disclaimer**: This is an experimental project for educational and research purposes. The author assumes no responsibility for misuse or damage resulting from the use of this system. Use responsibly and in compliance with applicable laws.
