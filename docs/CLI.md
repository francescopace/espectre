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
- `micro deploy --config <path>` deploys an alternate local override as device `config_local.py`; the firmware benchmark uses this to keep laboratory settings isolated from the developer's normal config.

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
| `--target` | Unicast device IP, comma-separated unicast IPs, or joined multicast group `239.255.0.1`; LAN broadcast does not produce CSI |
| `--duration` | Stop after N seconds |
| `--label` | Dataset label for saved collections; omit for live inspection without saving |
| `--start-delay` | Wait N seconds before starting collection; requires `--duration` |
| `--pps` | Collector temporal target and detector slot cadence |
| `--fixed` | Keep `--pps` as a constant send rate and ignore TX backpressure slowdowns |
| `--detector` | Detector used by the ready gate: `lightweight` or `high_accuracy`; a comma-separated list is available only for live comparison |
| `--ready-stable-seconds` | Seconds below threshold before saved collection starts; set `0` to disable the ready gate |

When `--target` is omitted, `collect` performs one mDNS/DNS-SD browse for `_espectre-streamer._udp.local.` at startup:

- `0` devices: fail explicitly and suggest `--target`
- `1` device: auto-select it
- `N` devices: prompt for an interactive choice

`--target` remains the deterministic bypass. Use a unicast device IP, comma-separated unicast IPs, or the firmware multicast group (`239.255.0.1` by default). Subnet and limited broadcast targets are accepted by the CLI but do not produce a usable CSI stream.

`--list-devices` uses the same one-shot browse, prints the resolved Streamer targets (`device_id`, chip, IP, and target port), and exits without starting UDP pacing, the CSI receiver, or dataset capture.

`--info` is also read-only: it uses `dataset_info.json` as the source of truth and prints one table per `environment`, with label rows and one column per chip.

In live streamer mode, `collect` sends UDP pacing traffic to the device. The device learns the collector address, creates one CSI record for each valid pacing packet, and batches records into return datagrams. Without `--label`, the collector only inspects the stream; with `--label`, it saves a dataset.

Pacing terms:

- **Delivered rate:** CSI records received by the collector, measured in packets per second (`pps`).
- **Admitted rate:** records that occupy a detector slot after temporal admission.
- **Excess:** extra same-slot records that do not improve occupancy.
- **Backpressure:** firmware reports that it cannot transmit records as quickly as they are produced.
- **Freshness:** the share of pacing packets that produce new CSI rather than stale or missing records.

The default collect policy backs off on sustained TX backpressure, spaces reductions across three control windows, does not fall below 70% of the requested target, and recovers toward `--pps` when backpressure clears. Occupancy remains telemetry and never changes the send rate. `--pps` stays the detector grid. Use `--fixed` when an experiment requires a constant send rate. Transport thresholds and control-loop behavior are implementation details owned by the Streamer [README.md](../src/cpp/frontend/streamer/README.md).

`--detector` always selects the production detector used for collection readiness. `--pps` is the collector's temporal target: the live detector and derived sensing view admit at most one packet per slot through the production Micro-ESPectre sampler, while Streamer firmware still transports the raw timestamped stream. `lightweight` performs its normal startup calibration before it can become ready. `high_accuracy` does not use startup calibration, but still needs its feature window to fill. Live inspection can compare `lightweight,high_accuracy` in parallel.

When `--label` is set, saved collection waits for the detector to stay below threshold for `--ready-stable-seconds` before packets are recorded. Set `--ready-stable-seconds 0` to bypass that gate explicitly.

When `--start-delay` is set, `--duration` is required. The collector waits first, then starts the ordinary live pacing and capture flow.

For discovery-selected unicast targets, the collector also validates that the first CSI packets carry the same `device_id` announced over mDNS. If the IP was reused by a different Streamer, collection aborts instead of saving mixed data under the wrong identity.

Examples:

```bash
./espectre collect --target 192.168.1.50
./espectre collect --target 192.168.1.50,192.168.1.51
./espectre collect --target 239.255.0.1
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

1. collects device identities from retained `info` and `status` plus any live publishes during the scan
2. shows an interactive selection list
3. falls back to manual device-id entry if nothing is discovered

After selection, the shell publishes commands to `commands/request` and subscribes to the matching response and payload topics:

```text
espectre/v1/devices/{device_id}/commands/request
espectre/v1/devices/{device_id}/commands/accepted
espectre/v1/devices/{device_id}/commands/rejected
espectre/v1/devices/{device_id}/commands/catalog
espectre/v1/devices/{device_id}/info
espectre/v1/devices/{device_id}/stats
espectre/v1/devices/{device_id}/ota/state
```

After selection the shell requests MQTT `commands` to populate help and tab completion from `commands/catalog`. `info`, `stats`, `ota_status`, and `commands` publish their payloads on those dedicated topics. Command ACKs annotate the typed prompt line with `✓` or `✗ reason` when the terminal allows it. Otherwise they appear on the next line. Payload topics are still dumped as YAML.

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

MQTT commands are forwarded to the selected device. The shell keeps only local utilities (`help`, `about`, `clear`, and `exit`) plus a few aliases (`i`, `st`, `ble`, …). Help and tab completion use the device `commands` catalog when the device publishes one. Unknown or unsupported commands are rejected by the device with `✗ command: reason`. Write values after the command name (`ble on`, `set_threshold 0.35`). Multi-field writes use named tokens after the command (`set_motion_hits motion_on_hits=4 motion_off_hits=3`).

`ble on` publishes MQTT `set_ble` with `ble=on` so a provisioned Native device advertises again. `ble off` stops BLE only when both Wi-Fi SSID and MQTT host are already present from Kconfig defaults or NVS. `ota_check` and `ota_start` accept an optional channel (`release`, `preview`, or `develop`), for example `ota_check preview` or `ota_start channel=develop`. Omitting the channel keeps the firmware's build-time default. OTA payloads containing server, manifest, image, or version overrides are rejected by the device. Frontends that report `supports_ota: false`, including Micro-ESPectre, reject the OTA commands. Frontends without Native BLE lifecycle control reject `set_ble`.

Browser tools such as Flash, Configure, Monitor, and Theremin live on [espectre.dev](https://espectre.dev). Serial logs remain `./espectre monitor`.

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
