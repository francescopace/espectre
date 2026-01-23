# Roadmap

**Vision**: ESPectre aims to democratize Wi-Fi sensing by providing an open-source, privacy-first motion detection system with a path toward machine learning-powered gesture recognition, Human Activity Recognition (HAR), and 3D indoor localization.

This roadmap outlines the evolution from the current mathematical approach (just IDLE/MOTION) toward ML-enhanced capabilities (Gesture detection, Human Activity Recognition) and advanced spatial sensing (3D localization via phase-coherent multi-antenna arrays), while maintaining the project's core principles: community-friendly, vendor-neutral, and privacy-first.

---

## Table of Contents

- [Market Opportunity](#market-opportunity)
- [Current State](#current-state)
- [Timeline Overview](#timeline-overview)
- [Short-Term (3-6 months)](#short-term-3-6-months)
- [Mid-Term (6-12 months)](#mid-term-6-12-months)
- [Long-Term (12-24 months)](#long-term-12-24-months)
- [Architecture Evolution](#architecture-evolution)
- [Principles & Governance](#principles--governance)
- [How to Propose Changes](#how-to-propose-changes)

---

## Market Opportunity

The global Wi-Fi sensing market is experiencing rapid growth, driven by demand for non-intrusive, privacy-preserving sensing solutions.

| Metric | Value | Source |
|--------|-------|--------|
| **Market Size (2024)** | $2.1B | Allied Market Research |
| **Projected Size (2030)** | $12.5B | Allied Market Research |
| **CAGR** | 34.2% | 2024-2030 |

### Key Drivers

- **Privacy concerns**: Camera-free sensing for elderly care, healthcare, and smart homes
- **Cost efficiency**: Leverages existing WiFi infrastructure (no additional hardware)
- **Regulatory push**: IEEE 802.11bf (Wi-Fi Sensing) standardization in progress

### Target Applications

| Application | Market Segment | ESPectre Capability |
|-------------|----------------|---------------------|
| **Smart Home** | Consumer IoT | Motion detection, presence sensing |
| **Elderly Care** | Healthcare | Fall detection, activity monitoring |
| **Security** | Commercial | Intrusion detection, occupancy |
| **Retail Analytics** | Enterprise | People counting, traffic flow |
| **Gesture Control** | Consumer Electronics | Hands-free device interaction |
| **Indoor Localization** | Logistics/Retail | Asset tracking, navigation (30-50cm accuracy) |

### Competitive Positioning

| Competitor | Approach | ESPectre Advantage |
|------------|----------|-------------------|
| **Origin Wireless** | Proprietary, cloud-dependent | Open-source, edge-first, no subscription |
| **Cognitive Systems** | Enterprise-only, high cost | Affordable ($5 hardware), DIY-friendly |
| **Espressif esp-radar** | SDK only, no integration | Full Home Assistant integration, production-ready |

ESPectre is uniquely positioned as the **only open-source, production-ready WiFi sensing platform** with native smart home integration.

---

## Current State

ESPectre v2.x provides a motion detection system using mathematical algorithms:

| Component | Status | Description |
|-----------|--------|-------------|
| **MVS Algorithm** | Production | Moving Variance Segmentation for motion detection (F1=99.0%) |
| **Band Calibration** | Production | Automatic subcarrier selection (F1=98.2%) |
| **ESPHome Integration** | Production | Native Home Assistant integration with auto-discovery |
| **Micro-ESPectre** | Production | Python R&D platform for rapid prototyping |
| **ML Data Collection** | Ready | Infrastructure for labeled CSI dataset creation |
| **Analysis Tools** | Ready | Comprehensive suite for CSI analysis and validation |

---

## Timeline Overview

```
     Q1 2026              Q2-Q3 2026              Q4 2026 - Q4 2027
        │                     │                          │
        ▼                     ▼                          ▼
┌───────────────┐     ┌───────────────┐     ┌─────────────────────┐
│  SHORT-TERM   │────▶│   MID-TERM    │────▶│     LONG-TERM       │
│   3-6 months  │     │   6-12 months │     │    12-24 months     │
├───────────────┤     ├───────────────┤     ├─────────────────────┤
│ Data & Docs   │     │ ML Models     │     │ 3D Localization     │
│ Dataset infra │     │ Training      │     │ Advanced Apps       │
│ Tooling       │     │ Edge Inference│     │ Multi-sensor Fusion │
└───────────────┘     └───────────────┘     └─────────────────────┘
```

---

## Short-Term (3-6 months)

**Focus**: Data collection, documentation, and ML groundwork.

### Data & Datasets

| Task | Priority | Status |
|------|----------|--------|
| Expand labeled CSI dataset (gestures, activities) | High | Planned |
| Community data contribution guidelines | High | Planned |
| Dataset versioning and reproducibility | Medium | Planned |
| Multi-environment data collection (offices, homes, industrial) | Medium | Planned |

### Documentation & Tooling

| Task | Priority | Status |
|------|----------|--------|
| Feature extraction pipeline documentation | High | Planned |
| Data labeling best practices guide | Medium | Planned |
| Jupyter notebooks for CSI exploration | Medium | Planned |
| Automated data quality validation | Low | Planned |

### Infrastructure

| Task | Priority | Status |
|------|----------|--------|
| Standardized dataset format (HDF5 or extended NPZ) | Medium | Planned |
| Dataset registry and metadata management | Low | Planned |

---

## Mid-Term (6-12 months)

**Focus**: ML model development, training infrastructure, and initial inference capabilities.

### Model Development

| Task | Priority | Status |
|------|----------|--------|
| Gesture recognition models (RF, CNN, LSTM) | High | Planned |
| Human Activity Recognition (HAR) models | High | Planned |
| People counting / presence estimation | Medium | Planned |
| Fall detection | Medium | Planned |

### Training Infrastructure

| Task | Priority | Status |
|------|----------|--------|
| Centralized training experiments (local) | High | Planned |
| Model versioning and experiment tracking | High | Planned |
| Hyperparameter optimization pipelines | Medium | Planned |
| Cross-validation with diverse environments | Medium | Planned |

### Inference (Exploratory)

| Task | Priority | Status |
|------|----------|--------|
| Edge inference on ESP32 (TensorFlow Lite Micro) | Medium | Exploratory |
| Model optimization (quantization, pruning) | Medium | Exploratory |
| Latency and memory profiling | Medium | Exploratory |

---

## Long-Term (12-24 months)

**Focus**: 3D indoor localization and advanced applications.

### 3D Localization

**Goal**: Transform motion detection into real-time 3D indoor localization with 30-50 cm accuracy.

This capability represents a significant leap from binary motion detection to precise spatial tracking, enabling applications like indoor navigation, asset tracking, and advanced gesture recognition.

| Capability | Description |
|------------|-------------|
| **Technology** | Phase-coherent multi-antenna array (4× ESP32-C5) |
| **Frequency** | 5GHz WiFi 6 for improved accuracy |
| **Algorithm** | MUSIC (Multiple Signal Classification) for AoA triangulation |
| **Target Accuracy** | 30-50 cm in 3D space |
| **Hardware Cost** | ~€75 per unit (custom PCB) |

| Task | Priority | Status |
|------|----------|--------|
| Phase coherence validation (2-device prototype) | High | Research |
| AoA estimation proof-of-concept | High | Research |
| Custom PCB design (4× ESP32-C5 + shared clock) | Medium | Research |
| MUSIC algorithm implementation | Medium | Research |
| 5GHz CSI extraction validation | Medium | Research |

### Advanced Applications

| Task | Priority | Status |
|------|----------|--------|
| Multi-sensor fusion (multiple ESP32 devices) | Medium | Exploratory |
| Room-level activity tracking | Medium | Exploratory |
| Vital signs monitoring (breathing, heartbeat) | Low | Research |
| Integration with IEEE 802.11bf (Wi-Fi Sensing standard) | Low | Research |

---

## Architecture Evolution

ESPectre's architecture evolves through three major versions, each adding capabilities while maintaining backward compatibility.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ARCHITECTURE EVOLUTION                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  v2.x (Current)          v3.x (ML-Enhanced)         v4.x (3D Spatial)       │
│  ───────────────         ─────────────────         ────────────────         │
│                                                                             │
│  ┌───────────┐           ┌───────────┐             ┌───────────────┐        │
│  │  ESP32    │           │  ESP32    │             │ 4× ESP32-C5   │        │
│  │  ┌─────┐  │           │  ┌─────┐  │             │ Phase-Coherent│        │
│  │  │ CSI │  │           │  │ CSI │  │             │   ┌─────┐     │        │
│  │  └──┬──┘  │           │  └──┬──┘  │             │   │ CSI │     │        │
│  │     │     │           │     │     │             │   └──┬──┘     │        │
│  │  ┌──▼──┐  │           │  ┌──▼──┐  │             └──────┼────────┘        │
│  │  │ MVS │  │           │  │ MVS │  │                    │                 │
│  │  └──┬──┘  │           │  └──┬──┘  │             ┌──────▼────────┐        │
│  └─────┼─────┘           │  ┌──▼──┐  │             │  Local/Cloud  │        │
│        │                 │  │ ML  │  │             │  ┌─────────┐  │        │
│        │                 │  │Edge │  │             │  │ MUSIC   │  │        │
│        │                 │  └──┬──┘  │             │  │Algorithm│  │        │
│        │                 └─────┼─────┘             │  └────┬────┘  │        │
│        │                       │                   │  ┌────▼────┐  │        │
│        │                       │                   │  │ 3D Pos  │  │        │
│        ▼                       ▼                   │  │ (X,Y,Z) │  │        │
│  ┌──────────┐            ┌──────────┐              │  └────┬────┘  │        │
│  │   Home   │            │   Home   │              └───────┼───────┘        │
│  │Assistant │            │Assistant │                      │                │
│  └──────────┘            └──────────┘                      ▼                │
│                                                      ┌──────────┐           │
│  Output:                 Output:                     │   Home   │           │
│  IDLE/MOTION             Gesture, HAR,               │Assistant │           │
│                          Fall Detection              └──────────┘           │
│                                                                             │
│                                                      Output:                │
│                                                      3D Position            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

| Version | Capability | Processing | Key Innovation |
|---------|------------|------------|----------------|
| **v2.x** | Motion detection (IDLE/MOTION) | 100% Edge | MVS algorithm, auto-calibration |
| **v3.x** | Gesture, HAR, fall detection | 100% Edge | TFLite Micro inference |
| **v4.x** | 3D indoor localization | Edge + Local/Cloud | Phase-coherent antenna array |

---

## Principles & Governance

ESPectre is committed to open-source principles and community-driven development.

### Core Principles

| Principle | Description |
|-----------|-------------|
| **Edge-First** | All processing happens locally on ESP32 - no cloud dependency |
| **Privacy-Preserving** | CSI data never leaves the device; no cameras, no recordings |
| **Hardware-Agnostic** | Supports ESP32, ESP32-S2/S3, ESP32-C3/C5/C6 variants |
| **Open Development** | All development happens in the open on GitHub |
| **Reproducibility** | Experiments and results must be reproducible |

### Governance

| Aspect | Approach |
|--------|----------|
| **License** | GPLv3 - ensures software remains free and open source |
| **Decision Making** | Maintainer-led with community input via GitHub Discussions |
| **Roadmap Updates** | Quarterly reviews based on community feedback and resources |

### Contributing

We welcome contributions! See **[CONTRIBUTING.md](CONTRIBUTING.md)** for:
- Code contribution guidelines
- Data contribution guidelines
- Development setup
- Code style and commit conventions

---

## How to Propose Changes

This roadmap evolves with community input. Here's how you can contribute:

| Method | Use Case |
|--------|----------|
| **GitHub Issues** | Propose new features or report blockers for existing items |
| **GitHub Discussions** | Discuss priorities, trade-offs, and architectural decisions |
| **Pull Request** | Submit changes to this file with your proposal |

### Process

1. **Check existing items** - Review current roadmap and open issues
2. **Open an Issue** - Describe your proposal with use case and rationale
3. **Discuss** - Engage with maintainers and community in the issue/discussion
4. **Submit PR** - Once there's consensus, update this file via Pull Request

---

## Roadmap Updates

This roadmap is reviewed and updated quarterly. Last update: **January 2026**

For the latest status and discussion:
- [GitHub Issues](https://github.com/francescopace/espectre/issues?q=is%3Aissue+label%3Aroadmap)
- [GitHub Discussions](https://github.com/francescopace/espectre/discussions)

---

## License

GPLv3 - See [LICENSE](LICENSE) for details.

