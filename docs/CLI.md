# Repository CLI

This reference lists the repository commands. If you have not chosen a frontend yet, start with the [setup guide](SETUP.md). Configuration and device-specific troubleshooting are in each frontend guide.

The tables below are summaries. For the exact current flags, run `./espectre --help` or `./espectre <namespace> --help`.

## Launchers

| Host | Launcher |
|------|----------|
| macOS/Linux | `./espectre` |
| Windows PowerShell/CMD | `.\espectre.cmd` |

Run the CLI from the repository root. The examples below use `./espectre`; on Windows, use `.\espectre.cmd` instead.

## Local build prerequisites

Create the repository environment with Python `3.14`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, use `py -3 -m venv .venv` and `.\.venv\Scripts\Activate.ps1`, then run the same install command.

Native, Matter, and Micro-ESPectre builds need ESP-IDF. The CLI looks for it in this order:

1. An active `IDF_PATH` environment.
2. A standard local ESP-IDF installation.
3. The ESP-IDF toolchain already installed by ESPHome.
4. The pinned ESP-IDF Docker image, if Docker is running.

```bash
./espectre native build --chip c3
```

About Docker:

- A cached image is used without asking. A missing image is downloaded only after you confirm; in scripts, pass `--pull missing`.
- If Docker is installed but stopped, the CLI asks you to start it.
- Docker only compiles. Flashing always uses the serial port on your computer.

Use `--backend local` or `--backend docker` to force one option, and `./espectre doctor` to check the local ESP-IDF setup. If you have neither, install Docker, build any ESPHome configuration once (which installs a toolchain), or install ESP-IDF `5.5.5` following the official [ESP-IDF Get Started](https://docs.espressif.com/projects/esp-idf/en/stable/esp32/get-started/index.html) guide.

ESPHome commands always use ESPHome's own ESP-IDF toolchain, never PlatformIO.

### Building against an SDK bundle

By default, builds use the SDK in your checkout. To build against an extracted SDK bundle instead, set `ESPECTRE_SDK_ROOT` to the absolute path of its `src/cpp` directory:

```bash
ESPECTRE_SDK_ROOT=/path/to/extracted-sdk/src/cpp ./espectre native build --chip c3
ESPECTRE_SDK_ROOT=/path/to/extracted-sdk/src/cpp ./espectre matter build --chip c3
ESPECTRE_SDK_ROOT=/path/to/extracted-sdk/src/cpp ./espectre esphome build --chip c3
ESPECTRE_SDK_ROOT=/path/to/extracted-sdk/src/cpp ./espectre micro build --chip c3
```

This checks that the firmware builds with the published SDK alone. Changing or unsetting the variable reconfigures the build. Docker builds mount the bundle read-only. The firmware version still comes from your checkout. See the [frontend layer](ARCHITECTURE.md#srccppfrontend) for the source boundaries.

### Optional compiler cache

`ccache` speeds up repeated builds, especially Matter. Local builds use it automatically when it is on `PATH`; Docker builds always have a cache.

Install it for local builds:

- macOS: `brew install ccache`
- Debian or Ubuntu: `sudo apt update && sudo apt install ccache` (other distributions: the `ccache` package)
- Windows: included with the official ESP-IDF Tools installer. Otherwise, use the [official release](https://ccache.dev/download.html) or `choco install ccache`.

Check it with `ccache --version`. When active, `build` and `doctor` print `Compiler cache: ccache`. To turn it off for one shell, set `IDF_CCACHE_ENABLE=0` (PowerShell: `$env:IDF_CCACHE_ENABLE = "0"`).

### Generated files and caches

Generated files live under `.cache/`:

| Directory | Contents |
|-----------|----------|
| `firmware/`, `sdk/` | Distribution files |
| `reports/` | Audit and coverage reports |
| `build/` | Docker toolchain homes |
| `npz/`, `ruff/`, `pytest/` | Reusable caches |

The virtual environment is `.venv/`, and each frontend keeps its own build directories. Set `ESPECTRE_NPZ_CACHE_DIR` to keep the NPZ cache on another disk. An old `.npz_cache/` can be moved to `.cache/npz/`; do not move caches while a build, test, or training run is using them.

## Command map

| Namespace | Purpose |
|-----------|---------|
| `esphome` | Build, flash, validate, or monitor the ESPHome frontend |
| `native` | Build or flash the Native frontend |
| `matter` | Build, flash, or read the pairing codes of the Matter frontend |
| [`micro`](../src/python/micro_espectre/README.md#commands) | Build, flash, deploy, run, and verify Micro-ESPectre |
| `monitor` | Show serial logs, reconnecting automatically |
| `devices` | Find ESPectre devices on the local network |
| `provision` | Set up Wi-Fi on Native or ESPHome over USB (Improv Serial) |
| `direct` | Send one Direct HTTP request to a device |
| `collect` | Watch live CSI or record a dataset |
| `doctor` | Check the local ESP-IDF environment |
| `mqtt` | Open the interactive MQTT shell |
| `version` | Show the CLI version |
| `about` | Show project and CLI information |

## Serial port selection

Commands that use a serial port (`flash`, `monitor`, `provision`, `matter qr`) pick it the same way:

- `--chip` keeps only ports that match the chip. Flash commands require it, and esptool also checks the chip while flashing.
- If a board exposes both a native USB port and a UART bridge, the native USB port is preferred.
- If several ports still match, the CLI asks you to choose. It never opens or resets a port just to identify it.
- `--port` selects one port explicitly; an incompatible port is rejected.

ESP32-S2 boards using USB CDC must already be in download mode to flash.

## Frontend workflow commands

### `esphome`

| Command | Purpose |
|---------|---------|
| `build` | Build the firmware |
| `flash` | Flash the firmware |
| `config` | Validate and show the configuration |
| `monitor` | Show logs |

Common flags are `--chip`, `--config`, and `--device`. Each chip has one example configuration; the CLI uses it with the ESPectre component from your local checkout.

- `build --clean` runs `esphome clean` first; `--clean-all` runs `esphome clean-all`.
- `flash --erase` erases all flash before a serial upload. It does not work over the network.
- `flash --firmware <path>` uploads a prebuilt image instead of the last build: a factory image over serial, or an OTA image to a hostname or IP.
- `build --json` prints a final JSON object with the frontend, chip, image path, size, and SHA-256.

```bash
./espectre esphome flash --chip c6 --device espectre-<mac-suffix>.local --firmware espectre-esphome-3.0.0-esp32c6-ota.bin
```

### `native` and `matter`

| Command | Purpose |
|---------|---------|
| `build` | Build the firmware for one chip |
| `flash` | Flash the last successful build over serial |

`build` flags:

- `--clean`: delete the build directory for this chip (for example `build-esp32c3`).
- `--clean-all`: delete all build directories, flash images, `sdkconfig`, and `dependencies.lock`.
- `--backend auto|local|docker`: pick where to build. `auto` (default) uses local ESP-IDF when available, otherwise Docker.
- `--pull ask|missing|never`: what to do when the Docker image is missing. The default `ask` prompts; scripts should use `missing` or `never`.
- `--json`: print the same final JSON object as ESPHome.

Each chip has its own build directory and `sdkconfig`, so building one chip never touches another. Docker builds use separate directories (such as `build-esp32c3-docker`).

A successful build copies its flash files to `build-flash-<chip>`. `flash --chip <chip>` always flashes the last successful build from there, local or Docker, and a failed build never replaces it. `flash` does not need ESP-IDF or Docker. If there is nothing to flash, it tells you to run `build` first. `ESPECTRE_IDF_BUILD_DIR` overrides the build directory.

`flash --erase` erases the whole flash first. On Matter this also deletes the pairing identity, so the device creates new pairing codes on the next boot.

Matter also has:

| Command | Purpose |
|---------|---------|
| `qr` | Restart the device and print its QR payload and manual pairing code |

`qr` requires `--chip`; add `--no-reset` to read without restarting. `matter qr --json` and `matter flash --json` print the port, chip, QR payload, and pairing code for other tools. Treat these codes as secrets.

Examples:

```bash
./espectre native build --chip c3
./espectre native build --chip c3 --backend docker
./espectre native build --chip c3 --clean
./espectre native build --chip c3 --clean-all
./espectre native flash --chip c5
./espectre esphome build --chip c3 --clean
./espectre esphome build --chip c3 --clean-all
./espectre matter build --chip c6
./espectre matter flash --chip c6 --port /dev/cu.usbmodemXXXX
./espectre matter qr --chip c6 --port /dev/cu.usbmodemXXXX
```

## Device and host commands

### `monitor`

`monitor` shows the serial logs. Flags: `--port`, `--chip`, `--frontend`, `--baud`, `--raw`, and `--reset`. Ports are chosen as described in [serial port selection](#serial-port-selection).

By default it attaches without restarting the device. `--reset` restarts it first and requires `--chip`. On USB CDC consoles, such as ESP32-S2, `--reset` does not work: reset the board by hand instead.

```bash
./espectre monitor --chip c3 --frontend native --port /dev/cu.usbmodemXXXX
./espectre monitor --chip c3 --frontend native --port /dev/cu.usbmodemXXXX --reset
```

### `devices`

`devices` searches the network for `_espectre._tcp.local.` and lists each device's frontend, ID, name, chip, IP, and Direct address. It ignores other services such as ESPHome's or Matter's. See [DNS-SD and mDNS](DISCOVERY.md#dns-sd-and-mdns) for the record format.

| Flag | Purpose |
|------|---------|
| `--frontend native\|esphome\|matter\|micro` | Only list one frontend |
| `--chip esp32\|c3\|s2\|s3\|c5\|c6` | Only list one chip family |
| `--timeout <seconds>` | How long to search; the default is 6 seconds |
| `--json` | Print JSON for scripts |

```bash
./espectre devices
./espectre devices --frontend native
./espectre devices --frontend matter --timeout 5
./espectre devices --frontend matter --chip s3 --json
```

The search always runs for the full timeout, so slower devices have time to answer. A shorter timeout may miss devices. Your computer and the devices must be on the same mDNS network: VLANs, client isolation, and multicast filtering can hide them. If a device does not show up, use its IP, its `.local` name, or Improv Serial; see [client validation and fallback](DISCOVERY.md#client-validation-and-fallback).

### `provision`

`provision` sets up Wi-Fi on a new Native or ESPHome device over USB, using Improv Serial. It accepts `--chip`, `--frontend`, and `--port` like `monitor`.

```bash
ESPECTRE_WIFI_PASSWORD='secret' ./espectre provision --chip c3 --frontend native --port /dev/cu.usbmodemXXXX --ssid MyNetwork
```

The password comes from `ESPECTRE_WIFI_PASSWORD`, from the variable named by `--password-env`, or from a hidden prompt. It is never accepted on the command line or printed. `--timeout` limits the whole exchange, and `--json` prints the port, device address, and result for other tools.

### `direct`

`direct` sends one request to a device. Give an HTTP verb and a resource, then either `--endpoint` with the device URL or `--frontend` (optionally with `--chip`) to find it. If several devices match, the CLI asks which one.

```bash
./espectre direct get health --frontend native
./espectre direct get diagnostics --endpoint http://espectre-0123456789abcdef.local
./espectre direct get diagnostics --data '{"fields":["traffic_tx_pps","csi_hw_error_total"]}' --endpoint http://espectre-0123456789abcdef.local
./espectre direct get diagnostics --data '{"fields":[]}' --endpoint http://espectre-0123456789abcdef.local
./espectre direct patch sensing --frontend esphome --data '{"detector":"high_accuracy"}'
./espectre direct post sensing/calibrations --frontend matter --chip s3
```

The CLI sends `Origin: https://test.espectre.dev`, which firmware allows by default. Use `--origin` only for another origin the firmware already accepts. Requests are limited to 2,048 bytes and responses to 8,192 bytes, and responses are checked against the [API reference](API.md).

#### Access-point selection

```bash
./espectre direct post wifi/scans  # start an access-point scan
./espectre direct get wifi/access-points  # list BSSID, channel, and RSSI
./espectre direct put wifi/bssid --data '{"bssid":"AA:BB:CC:DD:EE:FF"}'  # pin one AP
```

Wait a few seconds after starting the scan before reading the results. The device reconnects after you pin or clear an access point.

To go back to automatic selection, keeping the SSID and password, run the command below or choose automatic selection in Device settings. Do this also after replacing or removing a pinned access point.

```bash
./espectre direct delete wifi/bssid
```

See [Wi-Fi scan and BSSID selection](API.md#wi-fi-scan-and-bssid-selection) for the requests and responses.

### `collect`

`collect` receives raw CSI from a device over HTTP. It has three modes:

- **Live view:** without `--label`.
- **Recording:** with `--label`, saves a dataset.
- **Inventory:** `--info` prints the recorded datasets from `dataset_info.json`, one table per environment, without touching any device.

| Flag | Purpose |
|------|---------|
| `--target` | Device IP, hostname, Direct address, or device ID; omit it to search the network |
| `--frontend` | Only consider `native`, `esphome`, or `matter` devices when searching |
| `--source-ip` | Local IPv4 address to send from, on computers with several interfaces |
| `--duration` | Stop after N seconds |
| `--label` | Dataset label: 1–64 ASCII letters, digits, `_`, or `-`, starting with a letter or digit |
| `--start-delay` | Wait N seconds before starting; requires `--duration` |
| `--pps` | Packets per second to send, also stored as the dataset rate |
| `--detector` | `lightweight` or `high_accuracy` for the readiness check; a comma-separated list compares both in live view |
| `--ready-stable-seconds` | How long the detector must stay idle before recording starts; `0` disables the check |

```bash
./espectre collect --target 192.168.1.51 --pps 100
```

**Choosing the device.** Without `--target`, the CLI searches for 6 seconds like `devices`. With no device it stops and suggests `--target`; with one it uses it; with several it asks. `--target` skips the search. If a device was found by searching, its CSI records must carry the same `device_id`; if another device has taken that IP address, collection stops instead of mixing data.

**What happens during collection:**

1. The device is switched to `external` traffic mode. It stays in that mode afterwards.
2. The CLI starts sending UDP markers at `--pps` (the same generator as the Home Assistant add-on), then opens `GET /csi`.
3. The device sends every valid CSI record, without pacing or skipping.
4. When recording, the CLI waits until the selected detector is ready and has stayed idle for `--ready-stable-seconds`. Lightweight calibrates first; High Accuracy fills its feature window.
5. Closing the connection ends collection, then the generator stops.

`--duration` counts from the first packet in live view, or from the start of recording. It is checked even if packets stop arriving, with up to one second of delay. With `--start-delay`, the CLI waits before starting everything.

**After saving,** each file is checked for integrity, signal quality, occupancy, and continuity. Occupancy is measured on full detector windows: below 85% gives a warning, below 70% fails. A failed file is kept for diagnosis, but `collect` exits with an error. Each file records the device, firmware, endpoint, requested and measured rate, and format versions.

Terms in the output:

- **Delivered rate:** CSI records received per second.
- **Admitted rate:** records that fill a detector slot.
- **Excess:** extra records in an already filled slot; they do not improve occupancy.
- **Backpressure:** the device cannot send records as fast as it produces them.
- **Queue drop:** a record lost because the device's 16-record queue was full.

The sender alone sets the rate. After draining, `fresh_record_total + raw_drop_total` must equal `classified_frames_offered_to_raw`, so any loss before the network send is visible. The saved stream keeps every record; only the live detector view applies the detector's slot sampling.

```bash
./espectre devices --frontend native
./espectre collect --target 192.168.1.50 --pps 120
./espectre collect --frontend esphome --pps 120
./espectre collect --label wave --duration 45 --target espectre-0123456789abcdef.local
./espectre collect --label wave --duration 45 --start-delay 15 --target http://192.168.1.50
./espectre collect --info
```

### `mqtt`

`mqtt` opens an interactive shell for ESPectre devices on an MQTT broker.

| Flag | Default |
|------|---------|
| `--broker` | `homeassistant.local` or `MQTT_BROKER` |
| `--port-mqtt` | `1883` or `MQTT_PORT` |
| `--topic-prefix` | `espectre/v1/devices` or `MQTT_TOPIC_PREFIX` |
| `--device-id` | The device to use; without it, the shell searches |
| `--username` | `mqtt` or `MQTT_USERNAME` |
| `--password` | `mqtt` or `MQTT_PASSWORD` |

```bash
./espectre mqtt
./espectre mqtt --device-id 3cf79180d3a0aca4
./espectre mqtt --broker 192.168.1.20 --device-id native-lab
```

Without `--device-id`, the shell listens briefly on `espectre/v1/devices/+/device` and `+/health`, lists the devices it sees, and lets you pick one or type an ID.

Commands go to the device's `commands/request` topic and results come back on `commands/result`. The shell also shows the retained `capabilities`, `device`, `health`, `sensing`, `wifi`, and `ota` topics as YAML. Help and tab completion come from the device's `capabilities`. Besides device commands, the shell has only `help`, `about`, `clear`, and `exit`.

- Change settings with `update_sensing`, for example `update_sensing threshold=0.35 motion_on_hits=4 motion_off_hits=3`.
- `read_diagnostics` returns all values; use `read_diagnostics fields=traffic_tx_pps,csi_hw_error_total` for some, or `fields=[]` for the list.
- `check_ota` and `start_ota` accept `channel=release|preview|develop`. Without it, the device uses its built-in default.
- Results are marked `✓` or `✗ code: reason`.

The built-in OTA default comes from `native build --ota-channel release|preview|develop`, or the `NATIVE_OTA_CHANNEL` variable; it is `release` otherwise.

Browser tools such as Flash, Device settings, Monitor, and Theremin are on [espectre.dev](https://espectre.dev).

## Utility commands

| Command | Purpose |
|---------|---------|
| `./espectre doctor` | Check the ESP-IDF environment used by the CLI |
| `./espectre version` | Show the CLI version |
| `./espectre about` | Show project and CLI information |

## Related documents

- [Setup guide](SETUP.md): installation and choosing a frontend
- Frontend guides: [ESPHome](../src/cpp/frontend/esphome/README.md), [Native](../src/cpp/frontend/native/README.md), [Matter](../src/cpp/frontend/matter/README.md), and [Micro-ESPectre](../src/python/micro_espectre/README.md)
