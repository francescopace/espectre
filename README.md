[![License](https://img.shields.io/badge/license-GPLv3-blue.svg)](https://github.com/francescopace/espectre/blob/main/LICENSE)
[![Release](https://img.shields.io/github/v/release/francescopace/espectre)](https://github.com/francescopace/espectre/releases/latest)
[![CI](https://img.shields.io/github/actions/workflow/status/francescopace/espectre/ci.yml?branch=main&label=CI)](https://github.com/francescopace/espectre/actions/workflows/ci.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/francescopace/espectre/graph/badge.svg)](https://codecov.io/gh/francescopace/espectre)

<h1>ESPectre <img src="docs/web/assets/images/brand/espectre-logo.svg" alt="ESPectre logo" width="40" align="absmiddle" /></h1>

**Turn an ESP32 into a private, local Wi-Fi motion sensor.**

When someone moves through a room, they change the way Wi-Fi signals travel through it. ESPectre reads those changes and reports motion in real time. No camera, no microphone, no wearable, and no dedicated radar hardware: just a supported ESP32 and the Wi-Fi network already in the room.

**ESPectre** is an open-source platform that brings together ready-to-flash firmware, an embeddable C++ SDK, a MicroPython implementation, browser tools, a host CLI, an open dataset, open model weights, and the research workflow used to build and validate the detectors.

[**Flash from your browser**](https://espectre.dev/tools/flash/) · [See the live tools](https://espectre.dev/tools/) · [Read the documentation](https://espectre.dev/guides/) · [Explore the SDK](https://espectre.dev/sdk/)

## What You Can Build

Use ESPectre to add motion-aware lighting, heating, cooling, notifications, or room automations to a home, workspace, or prototype. Connect it to Home Assistant through ESPHome or MQTT, expose it as a standard Matter occupancy sensor, consume its local Direct HTTP API, or embed the sensing engine in your own ESP32 firmware.

Detection runs on the device. ESPectre reports a motion state and a movement score, so an application can react without sending raw sensing data to a cloud service. One board covers one sensing area; room-level coverage normally requires one board in each room.

ESPectre detects changes in the radio environment. It does not identify people, count them, prove that a room is empty, or replace a safety-certified security, medical, or emergency system.

## Start in Four Steps

The quickest path uses the browser and requires no local build environment:

1. Open [Flash](https://espectre.dev/tools/flash/) in a Chromium-based browser.
2. Connect a supported ESP32 over USB, then choose a firmware and release channel.
3. Complete on-screen Wi-Fi provisioning, or commission Matter with a supported controller.
4. Open [Monitor](https://espectre.dev/tools/monitor/) to watch motion, tune detection, and inspect the device.

![ESPectre Monitor](docs/web/assets/images/guides/sensing-dashboard.png)

> **Matter status:** The Matter frontend is still being validated across controller ecosystems. A controller may support standard Matter occupancy sensors without having been tested with current firmware.
> See [Matter controller compatibility](src/cpp/frontend/matter/README.md#matter-controller-compatibility) for the current matrix.

## Using the CLI

The repository wrapper exposes the available workflows through:

```bash
./espectre --help
```

## Supported hardware

- ESP32-C6, ESP32-C5, ESP32-C3, ESP32-S3, ESP32-S2, and classic ESP32
- a normal Wi-Fi 4 (802.11n) network on 2.4 GHz

## One Platform, Several Ways to Use It

| Part | What it gives you | Start here |
|---|---|---|
| **Firmware** | ESPHome, Native, Matter, and MicroPython paths for different products and integrations | [Choose a firmware path](#choose-a-firmware-path) |
| **C++ SDK** | The same sensing core and runtime for custom ESP-IDF firmware | [SDK.md](docs/SDK.md) |
| **Browser tools** | Flash, configure, discover, monitor, tune, inspect raw CSI, and run interactive demos | [espectre.dev/tools](https://espectre.dev/tools/) |
| **Host CLI and research tools** | Build firmware, provision devices, issue commands, collect CSI, train models, and validate results | [CLI.md](docs/CLI.md) |
| **Open research assets** | Raw datasets, model weights, feature history, training code, and reproducible performance reports | [Research You Can Inspect](#research-you-can-inspect) |
| **Documentation and website** | Guides, protocol references, architecture decisions, security guidance, and the complete website source | [Documentation](#documentation) |

## Choose a Firmware Path

| Path | Best for | Start here |
|---|---|---|
| **Native Direct/MQTT** | Standalone sensors, browser-local sensing, Home Assistant MQTT Discovery, and custom applications | [Native frontend](src/cpp/frontend/native/README.md) |
| **ESPHome** | Home Assistant users who want native entities, ESPHome provisioning, and Device Builder updates | [ESPHome frontend](src/cpp/frontend/esphome/README.md) |
| **Matter** | Matter controllers with occupancy-sensor support; controller validation is still limited | [Matter frontend](src/cpp/frontend/matter/README.md) |
| **Micro-ESPectre** | Lightweight sensing in MicroPython with local, read-only Direct HTTP monitoring | [Micro-ESPectre README](src/python/micro_espectre/README.md) |

ESPHome, Native, and Matter can choose between a `lightweight` detector, which learns a room-specific threshold at startup and leaves more resources to the rest of the application, and a `high_accuracy` detector, which runs the trained model included in the repository. Their behavior and measured trade-offs are documented in [SETUP.md](docs/SETUP.md#detection-profiles-and-startup), [ALGORITHMS.md](docs/ALGORITHMS.md), and the [performance report](docs/performance/README.md).

## Research You Can Inspect

ESPectre publishes the research assets and validation evidence behind its detectors:

- The [open CSI dataset](data/) contains real recordings for empty rooms, static presence, and motion, with its catalog and provenance in [dataset_info.json](data/dataset_info.json).
- The trained model weights are committed in both [C++](src/cpp/core/ml_weights.h) and [Python](tools/lib/ml_weights.py), and [ML_TRAINING.md](docs/ML_TRAINING.md) documents how training, export, and validation work.
- [FEATURES.md](docs/FEATURES.md) records features that were tested, promoted, or rejected, including unsuccessful experiments.
- [ALGORITHMS.md](docs/ALGORITHMS.md), the generated [performance report](docs/performance/README.md), and the [dataset quality report](data/auto_generated/DATASET_QUALITY_CHECK.md) make detector behavior and current evidence reviewable.
- [LITERATURE.md](docs/LITERATURE.md), the [architecture decision records](docs/adr/README.md), and the public [roadmap](docs/ROADMAP.md) separate what is shipped, what has been measured, and what remains a direction for future work.

The project has also contributed ESP32 CSI support upstream to MicroPython. The merged [MicroPython PR #18460](https://github.com/micropython/micropython/pull/18460) added direct CSI methods to the mainline `network.WLAN` implementation, so Micro-ESPectre no longer depends on a project-specific MicroPython fork.

## Security, Privacy, and Transparency

A sensor that can reveal presence should not be a black box. Wi-Fi sensing avoids images and audio, but motion and occupancy data can still reveal routines, sleep, or absence from home. ESPectre treats that risk as part of the engineering work:

- motion detection is local, and cloud connectivity is not required;
- raw CSI collection is optional and intended for defined research and debugging needs;
- the [security and responsible-use guide](https://espectre.dev/security/) documents deployment boundaries, consent, data minimization, abuse reporting, and private vulnerability reporting;
- firmware releases include build-specific SBOMs, notices, and license archives so their contents can be inspected;
- the public website and every browser tool are part of this repository under [docs/web](docs/web/README.md), including their source, privacy rules, analytics contract, and pinned browser dependencies;
- protocols, algorithms, performance evidence, limitations, roadmap decisions, and known validation boundaries are documented in public.

Use ESPectre only in spaces and networks where you have the right to deploy it. Inform affected people, obtain consent where required, protect access to the device and its data, and follow applicable privacy laws.

## Documentation

- **Install and operate**
  - [SETUP.md](docs/SETUP.md)
  - [CLI.md](docs/CLI.md)
  - [TUNING.md](docs/TUNING.md)
- **Understand and integrate**
  - [ARCHITECTURE.md](docs/ARCHITECTURE.md)
  - [ESPECTRE_PROTOCOL.md](docs/ESPECTRE_PROTOCOL.md)
  - [ALGORITHMS.md](docs/ALGORITHMS.md)
  - [SDK.md](docs/SDK.md)
- **Collect and train**
  - [ML_DATA_COLLECTION.md](docs/ML_DATA_COLLECTION.md)
  - [ML_TRAINING.md](docs/ML_TRAINING.md)
  - [FEATURES.md](docs/FEATURES.md)
  - [performance report](docs/performance/README.md)
  - [LITERATURE.md](docs/LITERATURE.md)
- **Research and direction**
  - [ROADMAP.md](docs/ROADMAP.md)
  - [ADR index](docs/adr/README.md)
  - [CHANGELOG.md](docs/CHANGELOG.md)
- **Frontend reference**
  - [ESPHome](src/cpp/frontend/esphome/README.md)
  - [Native](src/cpp/frontend/native/README.md)
  - [Matter](src/cpp/frontend/matter/README.md)
  - [Micro](src/python/micro_espectre/README.md)
- **Contributing**
  - [CONTRIBUTING.md](CONTRIBUTING.md)
  - [GitHub Discussions](https://github.com/francescopace/espectre/discussions)

## Acknowledgments

- Thanks to [Espressif](https://www.espressif.com/) for making CSI accessible in ESP-IDF and for recognizing ESPectre as a [community project](https://github.com/espressif/esp-csi#6-related-resources) in [esp-csi](https://github.com/espressif/esp-csi).
- Thanks to the [MicroPython](https://github.com/micropython/micropython) maintainers for reviewing, testing, and merging [ESPectre's upstream CSI contribution](https://github.com/micropython/micropython/pull/18460).
- Thanks to the TOMMY team for the constructive public discussion around Wi-Fi sensing approaches, including their [TOMMY vs ESPectre](https://www.tommysense.com/docs/comparisons/espectre-comparison) comparison page.

## License

ESPectre first-party code is available under GPLv3, and eligible parts are also available under a separate commercial agreement:

- Choose **GPLv3** when your firmware or application can comply with GPLv3, including making the corresponding source available. See [LICENSE](LICENSE).
- If you want to embed ESPectre in **proprietary or closed-source firmware** without GPLv3 source-disclosure obligations, see [LICENSING.md](LICENSING.md) for commercial licensing. The commercial license covers eligible material and does not replace third-party terms; the ESPHome C++ frontend remains GPLv3-only.

Third-party terms and build-specific compliance artifacts are described in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
