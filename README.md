[![License](https://img.shields.io/badge/license-GPLv3-blue.svg)](https://github.com/francescopace/espectre/blob/main/LICENSE)
[![Release](https://img.shields.io/github/v/release/francescopace/espectre)](https://github.com/francescopace/espectre/releases/latest)
[![CI](https://img.shields.io/github/actions/workflow/status/francescopace/espectre/ci.yml?branch=main&label=CI)](https://github.com/francescopace/espectre/actions/workflows/ci.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/francescopace/espectre/graph/badge.svg)](https://codecov.io/gh/francescopace/espectre)

<h1>ESPectre <img src="docs/web/assets/images/brand/espectre-logo.svg" alt="ESPectre logo" width="40" align="absmiddle" /></h1>

**ESPectre** is an open-source Wi-Fi sensing platform for ESP32 devices.

It detects motion from ordinary Wi-Fi signals, without cameras, microphones, wearables, or radar hardware. It integrates directly with Home Assistant through ESPHome or Native MQTT Discovery and offers a standards-based Matter occupancy-sensor path. It can also connect over BLE or custom integrations.

## How It Works

Wi-Fi signals bounce around a room. When a person moves, those reflections change. ESPectre reads channel state information (CSI), a measurement of how the radio channel changes across Wi-Fi frequencies, and turns those variations into motion and movement-score signals.

ESPectre needs just one device to work, but you can put one in every room to build a room-level detection mesh.

With ESPectre, ordinary Wi-Fi smart devices can double as ambient sensing nodes. Lights, switches, HVAC devices, appliances, and custom ESP32 products can add motion or occupancy awareness without cameras or dedicated sensors.

## Quick Start

If you want the fastest path, use the browser flasher:

1. Open [espectre.dev/flash](https://espectre.dev/flash/) with a Chromium-based browser
2. Pick the ESPHome, Native, or Matter base firmware and ESP32 target
3. Flash the board
4. Configure Wi-Fi and the remaining parameters by following the on-screen instructions

The browser tools share one site:

- [Configure](https://espectre.dev/configure/) provisions and tunes Native over BLE
- [MQTT Monitor](https://espectre.dev/monitor/) displays telemetry and device controls
- [The Game](https://espectre.dev/game/) and [Theremin](https://espectre.dev/theremin/) provide interactive sensing demos

GitHub Releases also provide Native OTA payloads; ESPHome updates are compiled and installed through ESPHome Device Builder after the device is adopted.

Supported hardware:

- ESP32-C6, ESP32-C5, ESP32-C3, ESP32-S3, and classic ESP32
- a normal Wi-Fi network; 2.4 GHz on every board, plus 5 GHz on the ESP32-C5

ESPectre includes two on-device detection profiles because deployments have different accuracy and resource budgets:

| Detection profile | Choose it when | Startup |
|---|---|---|
| `lightweight` | CPU time and working memory matter more than maximum accuracy, such as on smaller chips or firmware that must reserve resources for other features | Adapts to the room from about 10 seconds of clean, ready quiet-room coverage after temporal warmup |
| `high_accuracy` | Higher accuracy and better generalization justify additional feature state, memory, and inference work | Uses its trained threshold and skips quiet-room threshold calibration; it starts after CSI is ready and its feature window has filled |

## Build Your Own Path

| Path | Best for | Start here |
| ---- | -------- | ---------- |
| **ESPHome** | Home Assistant users who want the most polished production path | [ESPHome frontend](src/cpp/frontend/esphome/README.md) |
| **Matter** | Controllers with Matter occupancy-sensor support; validation is still limited | [Matter frontend](src/cpp/frontend/matter/README.md) |
| **Native BLE/MQTT** | Standalone devices, Home Assistant MQTT Discovery, web clients, and custom apps | [Native frontend](src/cpp/frontend/native/README.md) |
| **Streamer** | CSI data capture, dataset collection, live experiments, and ML workflows | [Streamer frontend](src/cpp/frontend/streamer/README.md) |
| **Micro-ESPectre** | MicroPython prototyping and optional Home Assistant MQTT Discovery | [Micro-ESPectre README](src/python/micro_espectre/README.md) |
| **SDK** | Custom firmware, smart-device makers, and OEM exploration | [EMBEDDING.md](docs/EMBEDDING.md) |

## Responsible Use

ESPectre does not use cameras, microphones, or wearables. It works with derived radio-channel measurements, and the project is designed around a local-first privacy boundary. Motion and occupancy signals can still reveal sensitive patterns such as presence, routines, sleep, and absence from home. Use ESPectre only in spaces where you have the right to deploy it, inform affected people, protect retained data, and follow local privacy laws.

## Documentation

- **Install and operate:** [SETUP.md](docs/SETUP.md), [CLI.md](docs/CLI.md), and [TUNING.md](docs/TUNING.md)
- **Understand and integrate:** [ARCHITECTURE.md](docs/ARCHITECTURE.md), [EMBEDDING.md](docs/EMBEDDING.md), [ESPECTRE_PROTOCOL.md](docs/ESPECTRE_PROTOCOL.md), and [ALGORITHMS.md](docs/ALGORITHMS.md)
- **Collect and train:** [ML_DATA_COLLECTION.md](docs/ML_DATA_COLLECTION.md), [ML_TRAINING.md](docs/ML_TRAINING.md), [FEATURES.md](docs/FEATURES.md), and the generated [performance report](docs/performance/README.md)
- **Research and direction:** [LITERATURE.md](docs/LITERATURE.md), [ROADMAP.md](docs/ROADMAP.md), the [ADR index](docs/adr/README.md), and [CHANGELOG.md](docs/CHANGELOG.md)
- **Frontend reference:** [ESPHome](src/cpp/frontend/esphome/README.md), [Native](src/cpp/frontend/native/README.md), [Matter](src/cpp/frontend/matter/README.md), and [Streamer](src/cpp/frontend/streamer/README.md)
- **Contributing:** [CONTRIBUTING.md](CONTRIBUTING.md) and [GitHub Discussions](https://github.com/francescopace/espectre/discussions)

## Related Projects

- [radio-presence-scanner](https://github.com/francescopace/radio-presence-scanner): complementary BLE radio presence sensing from host devices, with an optional HTTP dashboard.
- [micropython-esp32-csi](https://github.com/francescopace/micropython-esp32-csi): MicroPython firmware distribution used by the Micro-ESPectre workflow.

## Acknowledgments

- Thanks to [Espressif](https://www.espressif.com/) for making CSI accessible in ESP-IDF and for recognizing ESPectre as a [community project](https://github.com/espressif/esp-csi#6-related-resources) in [esp-csi](https://github.com/espressif/esp-csi).
- Thanks to the TOMMY team for the constructive public discussion around Wi-Fi sensing approaches, including their [TOMMY vs ESPectre](https://www.tommysense.com/docs/comparisons/espectre-comparison) comparison page.
- Thanks to the [MicroPython](https://github.com/micropython/micropython) maintainers for reviewing, testing, and merging our [PR](https://github.com/micropython/micropython/pull/18460), which extended the ESP32 `network.WLAN` implementation in mainline MicroPython with direct Wi-Fi CSI access methods. That merge matters well beyond ESPectre: it opened public MicroPython access to ESP32 CSI data for the wider community, where that support did not previously exist, and turned a key part of our sensing stack into upstream open-source infrastructure.

## License

ESPectre is dual-licensed:

- **GPLv3** for open-source use: see [LICENSE](LICENSE).
- **Commercial licenses** for embedding ESPectre into proprietary firmware: see [LICENSING.md](LICENSING.md).

Third-party terms and build-specific compliance artifacts are described in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
