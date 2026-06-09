# Roadmap

`v2.x` remains the production baseline for motion detection. The roadmap below
focuses on what each next version contains and how far it has progressed.

- `v3.x`: modular sensing platform
- `v4.x`: home orchestration layer

## Summary

| Version | Purpose | Status | Progress |
|---------|---------|--------|----------|
| **v3.x** | Turn ESPectre into a reusable platform across frontends and runtimes | In Progress | Core architecture is landed; frontend expansion and practical sensing work are still ongoing |
| **v4.x** | Build a local orchestration layer across multiple ESPectre nodes | Planned | Direction is defined, but implementation has not started yet |

---

## v3.x - Modular Sensing Platform

**Goal**: move from a single integration-focused firmware to a reusable platform
with shared sensing logic, a stable runtime contract, and multiple frontend
paths.

### Contains

| Area | Scope |
|------|-------|
| **Architecture** | Shared `core`, `runtime`, and `frontend` layers |
| **Runtime contract** | Stable frontend-oriented APIs such as `IEspectreRuntime`, snapshots, events, and capabilities |
| **ESPHome path** | Production frontend kept on top of the shared platform |
| **Matter path** | Second frontend proving that the same runtime/core can target another ecosystem |
| **Custom firmware path** | Ability to assemble alternate firmware targets from shared platform layers |
| **Practical sensing** | Presence and occupancy baselines, plus reusable inference/tooling foundations |
| **Host-side tooling** | Analysis tools, notebooks, datasets, and training workflows that support the platform direction |

### Implementation Checklist

- [x] Core / runtime / frontend split
- [x] Runtime contract and platform boundaries
- [x] ESPHome frontend stabilized without symlink-dependent packaging
- [ ] Matter frontend completed beyond the current experimental stage
- [ ] Custom firmware assembly productized from shared platform layers
- [ ] Presence / occupancy baselines validated for broader use
- [x] Edge ML inference on ESP32
- [ ] Training / dataset infrastructure completed for broader reproducibility
- [x] Notebooks / exploration tooling available

---

## v4.x - Home Orchestration Layer

**Goal**: make multiple ESPectre devices behave like one coherent home sensing
system through a local-first service layer.

### Contains

| Area | Scope |
|------|-------|
| **Local service** | Web or self-hosted service for orchestration across devices |
| **Device visibility** | Sensor inventory, runtime status, and fleet inspection |
| **Management** | Device lifecycle and firmware update workflows |
| **Realtime state** | Unified live view of the home across multiple nodes |
| **Multi-room fusion** | Room-to-room movement and multi-device event fusion |
| **Cross-frontend view** | Unified view across `ESPHome`, `Matter`, and custom firmware nodes |

### Implementation Checklist

- [ ] Local service for orchestration across devices
- [ ] Sensor inventory and runtime status view
- [ ] Device and firmware management workflows
- [ ] Realtime home state visualization
- [ ] Multi-device event fusion
- [ ] Unified view across `ESPHome`, `Matter`, and custom firmware nodes

---

## Roadmap Updates

Last update: **June 2026**

For discussion and proposed changes:

- [GitHub Issues](https://github.com/francescopace/espectre/issues?q=is%3Aissue+label%3Aroadmap)
- [GitHub Discussions](https://github.com/francescopace/espectre/discussions)

---

## License

GPLv3 - See [LICENSE](../LICENSE) for details.

