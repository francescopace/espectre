# Roadmap

**Vision**: ESPectre aims to democratize Wi-Fi sensing by providing an open-source, privacy-first motion detection system with a path toward machine learning-powered gesture recognition and Human Activity Recognition (HAR).

This roadmap outlines the evolution from the current mathematical approach (just IDLE/MOTION) toward ML-enhanced capabilities (Gesture detection, Human Activity Recognition), while maintaining the project's core principles: community-friendly, vendor-neutral, and privacy-first.

---

## Table of Contents

- [Current State](#current-state)
- [Timeline Overview](#timeline-overview)
- [Short-Term (3-6 months)](#short-term-3-6-months)
- [Mid-Term (6-12 months)](#mid-term-6-12-months)
- [Long-Term (12-24 months)](#long-term-12-24-months)
- [Architecture Overview](#architecture-overview)
- [Principles & Governance](#principles--governance)
- [How to Propose Changes](#how-to-propose-changes)

---

## Current State

ESPectre v2.x provides a motion detection system using mathematical algorithms:

| Component | Status | Description |
|-----------|--------|-------------|
| **MVS Algorithm** | Production | Moving Variance Segmentation for motion detection (F1=99.0%) |
| **NBVI Calibration** | Production | Automatic subcarrier selection (F1=98.2%) |
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
│ Data & Docs   │     │ ML Models     │     │ MLOps Platform      │
│ Dataset infra │     │ Training      │     │ Edge-MLOps Hybrid   │
│ Tooling       │     │ Edge Inference│     │ Advanced Apps       │
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

**Focus**: MLOps inference services, scalable deployment, and advanced applications.

### MLOps Inference Services

| Task | Priority | Status |
|------|----------|--------|
| MLOps inference API design | High | Exploratory |
| Privacy-preserving inference architecture | High | Exploratory |
| Reference deployment (on-premises or cloud) | Medium | Exploratory |
| Home Assistant cloud add-on (optional) | Medium | Exploratory |

### Edge-MLOps Hybrid

| Task | Priority | Status |
|------|----------|--------|
| Lightweight edge models + MLOps fallback | Medium | Exploratory |
| Federated learning exploration | Low | Research |
| On-device model updates | Low | Research |

### Advanced Applications

| Task | Priority | Status |
|------|----------|--------|
| Multi-sensor fusion (multiple ESP32 devices) | Medium | Exploratory |
| Room-level activity tracking | Medium | Exploratory |
| Vital signs monitoring (breathing, heartbeat) | Low | Research |
| Integration with IEEE 802.11bf (Wi-Fi Sensing standard) | Low | Research |

---

## Architecture Overview

### Current Architecture (v2.x)

```
┌─────────────────────────────────────────────────────────────────────┐
│                         EDGE (ESP32)                                │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐   │
│  │   CSI Data   │───▶│  Calibration │───▶│ Motion Segmentation  │   │
│  │  (Raw I/Q)   │    │  (Auto-tune) │    │   (IDLE/MOTION)      │   │
│  └──────────────┘    └──────────────┘    └───────────┬──────────┘   │
└──────────────────────────────────────────────────────┼──────────────┘
                                                       │
                                           ┌───────────▼───────────┐
                                           │   Home Assistant      │
                                           │   (Auto-discovery)    │
                                           └───────────────────────┘
```

### Future Architecture (v3.x+)

```
┌─────────────────────────────────────────────────────────────────────┐
│                         EDGE (ESP32)                                │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐   │
│  │   CSI Data   │───▶│  Calibration │───▶│  Motion Segmentation │   │
│  │  (Raw I/Q)   │    │  (Auto-tune) │    │ + Feature Extraction │   │
│  └──────────────┘    └──────────────┘    └───────────┬──────────┘   │
│                                                      │              │
│                                        ┌─────────────▼───────────┐  │
│                                        │     Edge Inference      │  │
│                                        │     (TFLite Micro)      │  │
│                                        └─┬───▲───────────────┬───┘  │
└──────────────────────────────────────────┼───│───────────────┼──────┘
                                           │   │               │
                                (Optional) │   │               │
                                           │   │               │
           ┌───────────────────────────────┼───│────┐          │
           │ Onprem/Cloud MLOps Platform   │   │    │          │
           ├───────────────────────────────┼───│────┤          │
           │  ┌────────────┐  ┌────────────▼───┴─┐  │          │
           │  │  Training  │  │  Model Serving   │  │          │
           │  │  Pipeline  │  │  (Gesture/HAR)   │  │          │
           │  └────────────┘  └──────────────────┘  │          │
           └────────────────────────────────────────┘          │
                                                               │
                                         ┌─────────────────────▼─┐
                                         │   Home Assistant      │
                                         │   (Auto-discovery)    │
                                         └───────────────────────┘
```

---

## Principles & Governance

ESPectre is committed to open-source principles and community-driven development.

### Core Principles

| Principle | Description |
|-----------|-------------|
| **Edge-First** | Core motion detection always works locally on ESP32 |
| **Privacy-Preserving** | CSI data is anonymous; only features (not raw CSI) sent externally if enabled |
| **Platform-Agnostic** | External Inference Platform works with any Kubernetes-based MLOps platform (cloud or on-premises) |
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

This roadmap is reviewed and updated quarterly. Last update: **December 2025**

For the latest status and discussion:
- [GitHub Issues](https://github.com/francescopace/espectre/issues?q=is%3Aissue+label%3Aroadmap)
- [GitHub Discussions](https://github.com/francescopace/espectre/discussions)

---

## License

GPLv3 - See [LICENSE](LICENSE) for details.

