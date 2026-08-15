# Repository CLI

Use this reference when you already know which ESPectre workflow you need and want the repository command for it. Start with [SETUP.md](SETUP.md) if you have not chosen a frontend yet. Frontend READMEs own configuration, prerequisites, and device-specific troubleshooting.

The command tables below are summaries; `./espectre --help` and `./espectre <namespace> --help` are authoritative for current flags.

## Launchers

| Host | Launcher |
|------|----------|
| macOS/Linux | `./espectre` |
| Windows PowerShell/CMD | `.\espectre.cmd` |

Run the CLI from the repository root.

## Command Map

| Namespace | Purpose |
|-----------|---------|
| `esphome` | Build, flash, validate, or monitor the ESPHome frontend |
| `native` | Build or flash the native ESP-IDF frontend |
| `matter` | Build, flash, or read onboarding data from the Matter ESP-IDF frontend |
| `streamer` | Build or flash the streamer ESP-IDF frontend |
| `micro` | Flash, deploy, run, and verify the MicroPython workflow |
| `monitor` | Attach to serial logs with auto-reconnect support |
| `collect` | Run live CSI inspection and dataset collection flows |
| `doctor` | Validate the local ESP-IDF environment used by the wrapper |
| `mqtt` | Open the interactive MQTT shell |
| `ui` | Open local browser tools |
| `version` | Show the CLI version label |
| `about` | Show project and CLI information |

## Common Patterns

- Use `./espectre --help` for the current top-level command list.
- Use `./espectre <namespace> --help` for namespace-specific flags.
- The wrapper prefers repository defaults and shared host autodetection over long manual setup steps.
- `Native`, `Matter`, and `Streamer` prefer the local ESP-IDF environment detected by the wrapper, including the native toolchain managed by the pinned ESPHome installation, and fall back to Docker for builds when no local installation is available. Use `./espectre doctor` to inspect the local ESP-IDF path.

## Frontend Workflow Commands

### `esphome`

The `esphome` namespace exposes:

| Command | Purpose |
|---------|---------|
| `build` | Build the selected ESPHome firmware |
| `flash` | Flash the selected ESPHome firmware |
| `config` | Validate and render the selected config |
| `monitor` | Open logs for the selected config |

Common flags include `--chip`, `--dev`, `--config`, and `--device`.

The wrapper explicitly selects ESPHome's native `esp-idf` toolchain for every command. It does not use the legacy PlatformIO build backend.

For `build`, cleanup flags are:

- `--clean`: run `esphome clean` for the selected config before compiling.
- `--clean-all`: run `esphome clean-all` for the config root before compiling.

### `native`, `matter`, and `streamer`

The three ESP-IDF namespaces expose `build` and `flash`:

| Command | Purpose |
|---------|---------|
| `build` | Configure the chip target and build the firmware |
| `flash` | Flash the frontend with the detected ESP-IDF environment |

For `build`, cleanup flags are:

- `--clean`: remove only the resolved build directory for the selected chip, such as `build-esp32c3`.
- `--clean-all`: remove all frontend build directories plus shared artifacts such as `sdkconfig`, `sdkconfig.old`, and `dependencies.lock`.

Build environment flags are:

- `--backend auto`: prefer local ESP-IDF and use Docker only when no local installation is detected; this is the default.
- `--backend local`: require local ESP-IDF and do not consider Docker.
- `--backend docker`: require the pinned ESP-IDF Docker image.
- `--pull ask|missing|never`: ask before downloading a missing Docker image, download it automatically, or require it to be cached. The default is `ask`; non-interactive jobs should use `missing` or `never` explicitly.

Docker builds use a separate directory such as `build-esp32c3-docker`, which prevents host and container CMake caches from sharing incompatible absolute paths. Docker is a build backend only; `flash` continues to use the detected local ESP-IDF environment and host serial port.

For `flash`, the wrapper selects the serial port first, then prefers the build directory that matches the connected chip detected on that port. Without a match, it falls back to the local configured target or the legacy `build/` layout.

`flash` still delegates to `idf.py flash`, so ESP-IDF may configure CMake or complete a missing build inside that selected directory before writing the firmware. The important guarantee is that the wrapper now prefers the chip-matched build directory first.

Matter additionally exposes:

| Command | Purpose |
|---------|---------|
| `qr` | Reset the connected device and print its persisted QR payload and manual pairing code |

Examples:

```bash
./espectre native build --chip c3
./espectre native build --chip c3 --backend docker
./espectre native build --chip c3 --clean
./espectre native build --chip c3 --clean-all
./espectre esphome build --chip c3 --clean
./espectre esphome build --chip c3 --clean-all
./espectre matter build --chip c6
./espectre matter flash --port /dev/cu.usbmodemXXXX
./espectre matter qr --port /dev/cu.usbmodemXXXX
./espectre streamer flash --port /dev/cu.usbmodemXXXX
```

## Device And Host Commands

### `micro`

The `micro` namespace owns MicroPython device lifecycle commands:

| Command | Purpose |
|---------|---------|
| `./espectre micro flash --erase` | Flash the CSI-enabled MicroPython firmware |
| `./espectre micro deploy` | Copy Micro-ESPectre Python sources to the device |
| `./espectre micro run` | Start the device application |
| `./espectre micro verify` | Check firmware and device readiness |

Notes:

- `--port` is optional; the CLI tries to auto-detect a serial device when possible.
- `micro flash` also supports `--chip` and `--firmware`.

### `monitor`

`monitor` attaches to a serial port and streams logs.

Common flags:

- `--port`
- `--baud`
- `--raw`
- `--reset`

By default, `monitor` attaches without resetting the device. Add `--reset` when you want a hard reset on open, for example to capture boot-time logs from the beginning.

Example:

```bash
./espectre monitor --port /dev/cu.usbmodemXXXX
```

Reset on open:

```bash
./espectre monitor --port /dev/cu.usbmodemXXXX --reset
```

### `collect`

`collect` is the unified host-side CSI collection entry point.

It uses one runtime collection path:

- live inspection when `--label` is omitted
- live recording when `--label` is set
- read-only dataset inventory when `--info` is used

Common flags:

| Flag | Purpose |
|------|---------|
| `--list-devices` | Browse Streamer devices via mDNS, print the resolved targets, and exit |
| `--target` | IPv4 target destination, or comma-separated destinations |
| `--duration` | Stop after N seconds |
| `--label` | Dataset label for saved collections; omit for live inspection without saving |
| `--start-delay` | Wait N seconds before starting collection; requires `--duration` |
| `--pps` | Target delivered record rate; adaptive pacing may send above it to compensate path loss |
| `--fixed` | Keep `--pps` as a constant send rate instead of the default adaptive pacing |
| `--detector` | Detector used by the ready gate: `lightweight` or `high_accuracy`; a comma-separated list is available only for live comparison |
| `--ready-stable-seconds` | Seconds below threshold before saved collection starts; set `0` to disable the ready gate |

When `--target` is omitted, `collect` performs one mDNS/DNS-SD browse for `_espectre-streamer._udp.local.` at startup:

- `0` devices: fail explicitly and suggest `--target`
- `1` device: auto-select it
- `N` devices: prompt for an interactive choice

`--target` remains the deterministic bypass, and keeps the existing single-target, multi-unicast, broadcast, and multicast workflows unchanged.

`--list-devices` uses the same one-shot browse, prints the resolved Streamer targets (`device_id`, chip, IP, and target port), and exits without starting UDP pacing, the CSI receiver, or dataset capture.

`--info` is also read-only: it uses `dataset_info.json` as the source of truth and prints one table per `environment`, with label rows and one column per chip.

In live streamer mode, `collect` sends UDP pacing traffic to the device. The device learns the collector address, creates one CSI record for each valid pacing packet, and batches records into return datagrams. Without `--label`, the collector only inspects the stream; with `--label`, it saves a dataset.

Pacing terms:

- **Delivered rate:** CSI records received by the collector, measured in packets per second (`pps`).
- **Backpressure:** firmware reports that it cannot transmit records as quickly as they are produced.
- **Freshness:** the share of pacing packets that produce new CSI rather than stale or missing records.

The default adaptive policy backs off on sustained backpressure and can compensate broadcast or multicast path loss by sending above the requested delivered rate. Unicast never boosts above `--pps` because Wi-Fi already retries unicast delivery. Use `--fixed` when an experiment requires a constant send rate. Transport thresholds and control-loop behavior are implementation details owned by the Streamer [README.md](../src/cpp/frontend/streamer/README.md).

`--detector` always selects the production detector used for collection readiness. The detector window is configured in milliseconds; `--pps` supplies only its initial packet estimate, then `collect` uses the same measured-timing and resize policy as Micro-ESPectre to resolve the window independently for each device. `lightweight` performs its normal startup calibration before it can become ready. `high_accuracy` does not use startup calibration, but still needs its feature window to fill. Live inspection can compare `lightweight,high_accuracy` in parallel.

When `--label` is set, saved collection waits for the detector to stay below threshold for `--ready-stable-seconds` before packets are recorded. Set `--ready-stable-seconds 0` to bypass that gate explicitly.

When `--start-delay` is set, `--duration` is required. The collector waits first, then starts the ordinary live pacing and capture flow.

For discovery-selected unicast targets, the collector also validates that the first CSI packets carry the same `device_id` announced over mDNS. If the IP was reused by a different Streamer, collection aborts instead of saving mixed data under the wrong identity.

Examples:

```bash
./espectre collect --target 192.168.1.50
./espectre collect --target 192.168.1.50 --pps 120
./espectre collect --target 192.168.1.50 --pps 120 --fixed
./espectre collect --label wave --duration 45 --target 192.168.1.50
./espectre collect --label wave --duration 45 --start-delay 15 --target 192.168.1.50
./espectre collect --info
```

### `mqtt`

`mqtt` opens the interactive MQTT shell for ESPectre Protocol devices.

When `--device-id` is provided, the shell targets that device directly.

When `--device-id` is not provided, the shell briefly subscribes to:

```text
espectre/v1/devices/+/info
espectre/v1/devices/+/status
```

It then:

1. collects device identities seen at runtime
2. shows an interactive selection list
3. falls back to manual device-id entry if nothing is discovered

After selection, the shell binds to the chosen device command topics:

```text
espectre/v1/devices/{device_id}/commands/request
espectre/v1/devices/{device_id}/commands/+
```

This behavior is transport-level and is not specific to the MicroPython frontend; it also applies to other ESPectre devices that expose the same MQTT topic surface.

Common MQTT flags:

| Flag | Default |
|------|---------|
| `--broker` | `homeassistant.local` or `MQTT_BROKER` |
| `--port-mqtt` | `1883` or `MQTT_PORT` |
| `--topic-prefix` | `espectre/v1/devices` or `MQTT_TOPIC_PREFIX` |
| `--device-id` | explicit argument or `MQTT_CLIENT_ID`; otherwise runtime discovery |
| `--username` | `mqtt` or `MQTT_USERNAME` |
| `--password` | `mqtt` or `MQTT_PASSWORD` |

Examples:

```bash
./espectre mqtt
./espectre mqtt --device-id 0x00007c2c6742bbac
./espectre mqtt --broker 192.168.1.20 --device-id native-lab
```

The interactive shell also exposes the Native OTA commands:

```text
ota_status
ota_check
ota_start
```

Stable Native firmware always uses its built-in latest-release GitHub manifest, while snapshot firmware always uses the rolling snapshot manifest. The command surface does not accept server, manifest, image, or version overrides. Frontends that report `supports_ota: false`, including Micro-ESPectre, reject these commands.

### `ui`

`ui` serves the unified website from an ephemeral localhost port and opens the selected browser application. Keep the command running while using the page, and press `Ctrl+C` to stop the local server. This mode supports local MQTT WebSocket endpoints that use `ws://`; hosted HTTPS pages should normally use `wss://`.

Supported interfaces:

- `mqtt`
- `ble`
- `theremin`

Examples:

```bash
./espectre ui
./espectre ui ble
./espectre ui theremin
```

## Utility Commands

| Command | Purpose |
|---------|---------|
| `./espectre doctor` | Validate the ESP-IDF environment used by the wrapper |
| `./espectre version` | Show the current CLI version label |
| `./espectre about` | Show project and CLI information |

## Related Documents

- [`SETUP.md`](SETUP.md) for shared setup, frontend selection, and entry points
- [`README.md` (micro_espectre)](../src/python/micro_espectre/README.md) for the MicroPython runtime workflow
- frontend READMEs under `src/cpp/frontend/` for frontend-specific build, flash, provisioning, and protocol details
