# Roadmap

**Vision**: ESPectre aims to democratize Wi-Fi sensing by providing an open-source, privacy-first sensing platform that starts from motion detection and evolves toward interoperable frontends, practical presence and occupancy intelligence, multi-room home awareness, and long-term 3D localization research.

This roadmap outlines the evolution from the current mathematical approach (`IDLE` / `MOTION`) toward a modular platform with a clear separation between core logic, runtime, and frontend, support for multiple integration paths (`ESPHome`, `Matter`, and custom firmware), and a future orchestration layer that can aggregate signals from multiple devices across the home. Advanced 3D localization remains part of the long-term vision, but it is treated as a stage-gated research track rather than the default next product step.

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
| **Smart Home** | Consumer IoT | Motion detection, presence sensing, interoperable automation |
| **Elderly Care** | Healthcare | Fall detection, activity monitoring |
| **Security** | Commercial | Intrusion detection, occupancy |
| **Retail Analytics** | Enterprise | People counting, traffic flow |
| **Home Orchestration** | Smart Home Platforms | Multi-room state, sensor fleet visibility, room-to-room tracking |
| **Indoor Localization** | Logistics/Retail | Research track for asset tracking and navigation (30-50 cm target) |

### Competitive Positioning

| Competitor | Approach | ESPectre Advantage |
|------------|----------|-------------------|
| **Origin Wireless** | Proprietary, cloud-dependent | Open-source, edge-first, no subscription |
| **Cognitive Systems** | Enterprise-only, high cost | Affordable ($5 hardware), DIY-friendly |

ESPectre is uniquely positioned as an **open-source Wi-Fi sensing platform** that combines edge-first sensing, smart home integration, and an emerging multi-device orchestration path.

---

## Current State

ESPectre v2.x provides a production motion detection system and the foundations for a broader sensing platform:

| Component | Status | Description |
|-----------|--------|-------------|
| **MVS Algorithm** | Production | Moving Variance Segmentation for motion detection |
| **Band Calibration** | Production | Automatic subcarrier selection (NBVI) |
| **ESPHome Integration** | Production | Native Home Assistant integration with auto-discovery |
| **Core / Runtime / Frontend Split** | In Progress | Decouple sensing logic from integration/runtime details |
| **Custom Firmware Path** | In Progress | Enable alternate runtimes and frontends on top of the same core |
| **Matter Frontend** | In Progress | Extend compatibility beyond Home Assistant toward major ecosystems |
| **Micro-ESPectre** | Production | Python R&D platform for rapid prototyping |
| **ML Data Collection** | Ready | Infrastructure for labeled CSI dataset creation |
| **Analysis Tools** | Ready | Comprehensive suite for CSI analysis and validation |

---

## Timeline Overview

```
     Q1 2026               Q2-Q4 2026                2027+
        │                      │                        │
        ▼                      ▼                        ▼
┌────────────────┐    ┌────────────────────┐   ┌──────────────────────┐
│   SHORT-TERM   │───▶│      MID-TERM      │──▶│      LONG-TERM       │
│    3-6 months  │    │     6-12 months    │   │      12-24 months    │
├────────────────┤    ├────────────────────┤   ├──────────────────────┤
│ Platform split │    │ Multi-frontend     │   │ Web orchestration    │
│ Tooling        │    │ Matter             │   │ Multi-room fusion    │
│ Data pipeline  │    │ Presence/occupancy │   │ Sensor management    │
└────────────────┘    └────────────────────┘   └──────────────────────┘
```

Parallel to the product roadmap, ESPectre keeps a separate research track for
phase-coherent multi-node localization. That work remains active, but it is
gated by synchronization, hardware, and phase-quality milestones before it can
be treated as a mainline product direction.

---

## Short-Term (3-6 months)

**Focus**: Platform modularization, tooling, and replayable sensing workflows.

### Architecture Platform

| Task | Priority | Status |
|------|----------|--------|
| Finalize separation between core, runtime, and frontend | High | In Progress |
| Define stable interfaces for alternate runtimes and frontends | High | In Progress |
| Enable custom firmware assembly from shared core components | High | Planned |
| Document platform boundaries and integration contracts | Medium | Planned |

### Data & Datasets

| Task | Priority | Status |
|------|----------|--------|
| Expand labeled CSI dataset for occupancy, presence, and room-level scenarios | High | Planned |
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

**Focus**: Multi-frontend support, practical inference, and interoperable deployments.

### Frontends and Runtimes

| Task | Priority | Status |
|------|----------|--------|
| Ship `Matter` frontend for major ecosystem compatibility | High | In Progress |
| Keep `ESPHome` frontend aligned with the modular platform | High | Planned |
| Validate custom firmware builds on alternate runtime/frontend combinations | High | Planned |
| Harden runtime abstraction for future deployment targets | Medium | Planned |

### Practical Sensing Capabilities

| Task | Priority | Status |
|------|----------|--------|
| Presence and occupancy baselines across multiple environments | High | In Progress |
| People counting / room-level estimation | Medium | Planned |
| Multi-device event fusion without precise localization | Medium | Planned |
| Fall detection | Medium | Planned |
| Gesture detection research backlog | Low | Deferred |
| Human Activity Recognition (HAR) research backlog | Low | Deferred |

### Training Infrastructure

| Task | Priority | Status |
|------|----------|--------|
| Centralized training experiments (local) | High | Planned |
| Model versioning and experiment tracking | High | Planned |
| Hyperparameter optimization pipelines | Medium | Planned |
| Cross-validation with diverse environments | Medium | Planned |

### Inference

| Task | Priority | Status |
|------|----------|--------|
| Edge inference on ESP32 (manual MLP) | High | Done |
| TensorFlow Lite Micro integration | Medium | Exploratory |
| Model optimization (quantization, pruning) | Medium | Exploratory |
| Latency and memory profiling | Medium | Planned |

---

## Long-Term (12-24 months)

**Focus**: Product orchestration for the home, with research-stage spatial sensing in parallel.

### Product / Platform Track

**Goal**: Build a service layer that makes multiple ESPectre devices feel like one coherent home sensing system.

This track emphasizes practical value before precise localization: device visibility,
sensor management, updates, realtime state, and room-to-room movement
understanding driven by multi-device event fusion.

| Capability | Description |
|------------|-------------|
| **Service Model** | Local-first or self-hosted web service for home sensing orchestration |
| **Primary Value** | Sensor visibility, fleet management, realtime home state |
| **Spatial Output** | Room-to-room movement and multi-room awareness without precise coordinates |
| **Device Scope** | Multiple ESPectre nodes, multiple frontends, shared home view |
| **Upgrade Path** | Product-grade orchestration first, finer spatial semantics later |

| Task | Priority | Status |
|------|----------|--------|
| Web service for sensor inventory and status inspection | High | Planned |
| Device and firmware update management workflows | High | Planned |
| Realtime multi-room home state visualization | High | Planned |
| Multi-device event fusion for room-to-room movement tracking | High | Planned |
| Unified view across `ESPHome`, `Matter`, and custom firmware nodes | Medium | Planned |

### Research Track: 3D Localization

**Goal**: Preserve a path toward real-time 3D indoor localization with a
30-50 cm target, but only after the required synchronization and hardware gates
are met.

This remains a research milestone, not the default next product milestone. The
current practical strength of the project is multi-device semantic fusion and
occupancy-style inference, while phase-coherent localization still depends on
open work around reference frames, hardware discipline, thermal stability, and
phase observability under realistic occupied conditions.

| Capability | Description |
|------------|-------------|
| **Technology** | Wireless phase-coherent multi-node architecture |
| **Frequency** | Prefer 5 GHz capable hardware when the capture path is validated |
| **Algorithm** | AoA / MUSIC only after phase-quality gates are passed |
| **Target Accuracy** | 30-50 cm in 3D space |
| **Decision Policy** | Stage-gated: observability first, localization later |

| Task | Priority | Status |
|------|----------|--------|
| Reference-frame compensation and cadence validation | High | Research |
| Hardware tier comparison (`XO` vs `TCXO` / `VCTCXO`) | High | Research |
| Thermal and drift characterization under realistic conditions | High | Research |
| Go / no-go decision for phase-coherent path on current hardware | High | Research |
| AoA estimation proof-of-concept after phase gates | Medium | Research |
| Multi-node geometry scaling policy (`3` -> `4`) | Medium | Research |
| Custom carrier/backplane only after validation gates | Low | Research |

### Later Advanced Applications

| Task | Priority | Status |
|------|----------|--------|
| Gesture detection | Low | Deferred |
| Human Activity Recognition (HAR) | Low | Deferred |
| Vital signs monitoring (breathing, heartbeat) | Low | Research |
| Integration with IEEE 802.11bf (Wi-Fi Sensing standard) | Low | Research |

---

## Architecture Evolution

ESPectre's architecture evolves through major versions that widen the platform
surface area while keeping the sensing core reusable across runtimes and
frontends.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ARCHITECTURE EVOLUTION                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  v2.x (Current)          v3.x (Platform)            v4.x (Orchestration)    │
│  ───────────────         ───────────────            ─────────────────────    │
│                                                                             │
│  ┌───────────┐           ┌───────────┐             ┌───────────────┐        │
│  │  ESP32    │           │ SharedCore │             │ WebService │          │
│  │  ┌─────┐  │           │ Runtime(s) │             │ MultiRoom  │          │
│  │  │ CSI │  │           │ Frontend(s)│             │ Fusion     │          │
│  │  └──┬──┘  │           │  ┌───────┐ │             │  ┌───────┐ │          │
│  │     │     │           │  │Matter │ │             │  │Rooms  │ │          │
│  │  ┌──▼──┐  │           │  │ESPHome│ │             │  │Events │ │          │
│  │  │ MVS │  │           │  │Custom │ │             │  │Devices│ │          │
│  │  └──┬──┘  │           │  └───┬───┘ │             │  └───┬───┘ │          │
│  └─────┼─────┘           └──────┼──────┘             └──────┼──────┘         │
│        │                        │                             │               │
│        ▼                        ▼                             ▼               │
│  ┌──────────┐            ┌──────────────┐              ┌──────────────┐      │
│  │Assistant │            │ MultiEcosystem│             │ HomeDashboard │      │
│  └──────────┘            │ Integration   │             └──────────────┘      │
│                                                                             │
│  Output:                 Output:                      Output:                │
│  IDLE/MOTION             Presence, occupancy,         Realtime home state,  │
│                          custom firmware,             room-to-room movement  │
│                          Matter support               understanding          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

| Version | Capability | Processing | Key Innovation |
|---------|------------|------------|----------------|
| **v2.x** | Motion detection (IDLE/MOTION) | 100% Edge | MVS algorithm, auto-calibration |
| **v3.x** | Modular sensing platform | Edge + host tooling | Core/runtime/frontend split, `Matter`, custom firmware |
| **v4.x** | Home sensing orchestration | Edge + local service | Web service, multi-room fusion, device management |

3D localization remains a separate research track that can influence future
versions only after its validation gates are met.

---

## Principles & Governance

ESPectre is committed to open-source principles and community-driven development.

### Core Principles

| Principle | Description |
|-----------|-------------|
| **Edge-First** | Sensing stays local-first on devices and local services, with no cloud dependency required |
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

We welcome contributions! See **[CONTRIBUTING.md](../CONTRIBUTING.md)** for:
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

This roadmap is reviewed and updated quarterly. Last update: **June 2026**

For the latest status and discussion:
- [GitHub Issues](https://github.com/francescopace/espectre/issues?q=is%3Aissue+label%3Aroadmap)
- [GitHub Discussions](https://github.com/francescopace/espectre/discussions)

---

## License

GPLv3 - See [LICENSE](../LICENSE) for details.

