[![License](https://img.shields.io/badge/license-GPLv3-blue.svg)](https://github.com/francescopace/espectre/blob/main/LICENSE)
[![SoC](https://img.shields.io/badge/SoC-ESP32-orange.svg)](https://www.espressif.com/en/products/socs)
[![ESP-IDF component](https://img.shields.io/badge/ESP--IDF-component-E7352C?logo=espressif)](https://components.espressif.com/components/francescopace/espectre)
[![ESP-IDF staging](https://img.shields.io/badge/ESP--IDF-staging-E7352C?logo=espressif)](https://components-staging.espressif.com/components/francescopace/espectre)
[![C++ coverage](https://badgen.net/https/github.com/francescopace/espectre/releases/download/snapshot/coverage-cpp-runtime.json?cache=300)](https://github.com/francescopace/espectre/actions/workflows/ci.yml?query=branch%3Amain)
[![Python coverage](https://badgen.net/https/github.com/francescopace/espectre/releases/download/snapshot/coverage-python.json?cache=300)](https://github.com/francescopace/espectre/actions/workflows/ci.yml?query=branch%3Amain)
[![Web coverage](https://badgen.net/https/github.com/francescopace/espectre/releases/download/snapshot/coverage-web.json?cache=300)](https://github.com/francescopace/espectre/actions/workflows/ci.yml?query=branch%3Amain)

<h1>ESPectre <img src="docs/web/assets/images/brand/espectre-logo.svg" alt="ESPectre logo" width="40" align="absmiddle" /></h1>

**Turn an ESP32 into a private, local Wi-Fi motion sensor.**

When someone moves through a room, they change the way Wi-Fi signals travel through it. ESPectre reads those changes and reports motion in real time. No camera, no microphone, no wearable, and no dedicated radar hardware: just a supported ESP32 and the Wi-Fi network already in the room.

**ESPectre** is an open-source platform that brings together ready-to-flash firmware, an embeddable C++ SDK, a MicroPython implementation, browser tools, a host CLI, an open dataset, open model weights, and the research workflow used to build and validate the detectors.

[Flash from your browser](https://espectre.dev/tools/flash/) · [See the live tools](https://espectre.dev/tools/) · [Read the documentation](https://espectre.dev/guides/) · [Explore the SDK](https://espectre.dev/sdk/)

## Applications and integrations

Use ESPectre to turn on lights or a display when someone walks in, adjust heating when a room is in use, or send an alert when there is movement where nobody should be. It works with Home Assistant (through ESPHome or MQTT), appears as a standard Matter occupancy sensor, offers a local HTTP API, and can be built into your own ESP32 firmware with the C++ SDK.

Everything runs on the device: it measures the Wi-Fi channel (channel state information, or CSI) and reports motion and a movement score. Nothing needs to go to the cloud. One board covers one area, so plan on one board per room.

ESPectre detects changes in the radio environment. It does not identify people, count them, prove that a room is empty, or replace a safety-certified security, medical, or emergency system.

## Supported hardware

- ESP32-C6, ESP32-C5, ESP32-C3, ESP32-S3, ESP32-S2, and classic ESP32
- a normal Wi-Fi 4 (802.11n) network on 2.4 GHz

## Choose a firmware path

| Path | Best for | Start here |
|---|---|---|
| **Native** | Standalone sensors, MQTT integrations, including Home Assistant MQTT Discovery, and custom applications | [Native frontend](src/cpp/frontend/native/README.md) |
| **ESPHome** | Home Assistant users who want native entities, ESPHome provisioning, and Device Builder updates | [ESPHome frontend](src/cpp/frontend/esphome/README.md) |
| **Matter** | Matter controllers with occupancy-sensor support; controller validation is still limited | [Matter frontend](src/cpp/frontend/matter/README.md) |
| **Micro-ESPectre** | Lightweight sensing in MicroPython with local, read-only Direct HTTP monitoring | [Micro-ESPectre README](src/python/micro_espectre/README.md) |

## Quick start

The quickest way needs only a browser. Use desktop Chrome 151 or later; Edge can flash, but Device settings and Monitor may not work in it.

1. Open [Flash](https://espectre.dev/tools/flash/) in desktop Chrome or Edge.
2. Connect a [supported ESP32](#supported-hardware) over USB, then choose a firmware and release channel.
3. Complete on-screen Wi-Fi provisioning, or commission Matter with a supported controller.
4. Optionally, open [Device settings](https://espectre.dev/tools/device-settings/) to pin a preferred access point or set up MQTT.
5. Open [Monitor](https://espectre.dev/tools/monitor/) to watch motion, tune detection, and inspect the device.

To find your devices on the network, use **Find devices** in the browser tools or `./espectre devices`. See the [discovery reference](docs/DISCOVERY.md) for requirements and limits.

![ESPectre Monitor](docs/web/assets/images/guides/sensing-dashboard.png)

> **Matter status:** Matter support is still being tested with the different controller ecosystems. A controller that supports occupancy sensors may not have been tested with ESPectre yet.
> See [Matter controller compatibility](src/cpp/frontend/matter/README.md#matter-controller-compatibility) for the current matrix.

## Local setup

To build and flash from this repository, start with the [setup guide](docs/SETUP.md). To see all available commands, run:

```bash
./espectre --help
```

## Documentation

| Topic | What it covers | Guides |
|---|---|---|
| **Install and operate** | Device setup, CLI workflows, and troubleshooting | [Setup guide](docs/SETUP.md), [CLI reference](docs/CLI.md), [Troubleshooting guide](docs/TROUBLESHOOTING.md) |
| **Understand and integrate** | Runtime architecture, CSI acquisition, API, discovery, algorithms, and the C++ SDK | [Architecture overview](docs/ARCHITECTURE.md), [CSI guide](docs/CSI.md), [API reference](docs/API.md), [Discovery reference](docs/DISCOVERY.md), [Algorithms reference](docs/ALGORITHMS.md), [SDK guide](docs/SDK.md) |
| **Collect and train** | CSI collection, model training, feature history, performance, and literature | [Data collection guide](docs/ML_DATA_COLLECTION.md), [ML training guide](docs/ML_TRAINING.md), [Feature ledger](docs/FEATURES.md), [Performance report](docs/performance/README.md), [Literature review](docs/LITERATURE.md) |
| **Research and direction** | Roadmap, architecture decisions, and release history | [Roadmap](docs/ROADMAP.md), [ADR index](docs/adr/README.md), [Changelog](docs/CHANGELOG.md) |
| **Frontend reference** | Guides for each firmware: ESPHome, Native, Matter, and Micro-ESPectre | [ESPHome](src/cpp/frontend/esphome/README.md), [Native](src/cpp/frontend/native/README.md), [Matter](src/cpp/frontend/matter/README.md), [Micro](src/python/micro_espectre/README.md) |
| **Contributing** | Contributions, release procedures, and project discussions | [Contributing guide](CONTRIBUTING.md), [Release guide](docs/RELEASING.md), [GitHub Discussions](https://github.com/francescopace/espectre/discussions) |

## Datasets, models, and validation

ESPectre publishes the research assets and validation evidence behind its detectors:

| Asset | What it gives you | Start here |
|---|---|---|
| **CSI dataset** | Real recordings for empty rooms, static presence, and motion, with catalog and provenance in [dataset_info.json](data/dataset_info.json) | [Data/](data/) |
| **Model weights** | Trained weights in [C++](src/cpp/core/ml_weights.h) and [Python](tools/lib/ml_weights.py), plus the training, export, and validation workflow | [ML training guide](docs/ML_TRAINING.md) |
| **Feature ledger** | Features that were tested, promoted, or rejected, including unsuccessful experiments | [Feature ledger](docs/FEATURES.md) |
| **Algorithms and reports** | Detector behavior, the generated [performance report](docs/performance/README.md), and [dataset quality report](data/auto_generated/DATASET_QUALITY_CHECK.md) | [Algorithms reference](docs/ALGORITHMS.md) |
| **Literature and direction** | External research, [architecture decision records](docs/adr/README.md), and the public [roadmap](docs/ROADMAP.md) | [Literature review](docs/LITERATURE.md) |

## Security, privacy, and transparency

A sensor that can reveal presence should not be a black box. Wi-Fi sensing records no images or audio, but motion data can still reveal routines, sleep, or when nobody is home. ESPectre treats that risk as part of the design:

- motion detection is local, and cloud connectivity is not required;
- raw CSI collection is optional and meant for research and debugging;
- the [security and responsible-use guide](https://espectre.dev/security/) documents deployment boundaries, consent, data minimization, abuse reporting, and private vulnerability reporting;
- firmware releases include build-specific SBOMs, notices, and license archives so their contents can be inspected;
- the website and all browser tools are in this repository under [docs/web](docs/web/README.md), with their privacy rules and analytics policy;
- protocols, algorithms, test results, limitations, and plans are all public.

Use ESPectre only where you have the right to. Tell the people affected, get consent where required, protect access to the device and its data, and follow privacy laws.

## Acknowledgments

- Thanks to [Espressif](https://www.espressif.com/) for making CSI accessible in ESP-IDF and for recognizing ESPectre as a [community project](https://github.com/espressif/esp-csi#6-related-resources) in [esp-csi](https://github.com/espressif/esp-csi).
- Thanks to the [MicroPython](https://github.com/micropython/micropython) maintainers for reviewing, testing, and merging [ESPectre's CSI contribution](https://github.com/micropython/micropython/pull/18460), which added direct CSI methods to mainline `network.WLAN`.

## License

ESPectre first-party code is available under **GPLv3**, and eligible parts also under a separate commercial license:

- **GPLv3** fits if you can publish the source of your firmware or application. See [LICENSE](LICENSE).
- **Commercial license**, for **closed-source firmware**: see the [licensing terms](LICENSING.md). It covers only eligible material, does not replace third-party licenses, and does not cover the ESPHome frontend, which stays GPLv3-only.

Third-party terms and build-specific compliance artifacts are described in the [third-party notices](THIRD_PARTY_NOTICES.md).
