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
| [`micro`](../src/python/micro_espectre/README.md#commands) | Build, flash, deploy, run, and verify the research frontend |
| `monitor` | Attach to serial logs with auto-reconnect support |
| `devices` | Discover advertised ESPectre devices on the local network |
| `provision` | Provision Native or ESPHome Wi-Fi through Improv Serial |
| `direct` | Send one Direct HTTP protocol request to a device |
| `collect` | Run live CSI inspection and dataset collection flows |
| `doctor` | Validate the local ESP-IDF environment used by the wrapper |
| `mqtt` | Open the interactive MQTT shell |
| `version` | Show the CLI version label |
| `about` | Show project and CLI information |

## Common Patterns

- Use `./espectre --help` for the current top-level command list.
- Use `./espectre <namespace> --help` for namespace-specific flags.
- The wrapper prefers repository defaults and shared host autodetection over long manual setup steps.
- `Native` and `Matter` prefer the local ESP-IDF environment detected by the wrapper, including the native toolchain managed by the pinned ESPHome installation, and fall back to Docker for builds when no local installation is available. Use `./espectre doctor` to inspect the local ESP-IDF path.
- Serial selection is shared across published frontend flash, monitor, provision, and onboarding operations. The resolver first waits through a bounded USB re-enumeration window, then keeps ports compatible with the frontend and action. When `--chip` is supplied and more than one candidate remains, it identifies the connected chips. A single best match is selected automatically; multiple equally suitable matches produce a prompt. Identification uses esptool and resets every probed board. The identification table and the selection prompt list each port with its chip and physical console (`uart`, `usb_cdc`, or `usb_serial_jtag`). Pass `--port` to require that exact compatible device; explicit ports use the same re-enumeration and compatibility checks.

## Frontend Workflow Commands

### `esphome`

The `esphome` namespace exposes:

| Command | Purpose |
|---------|---------|
| `build` | Build the selected ESPHome firmware |
| `flash` | Flash the selected ESPHome firmware |
| `config` | Validate and render the selected config |
| `monitor` | Open logs for the selected config |

Common flags include `--chip`, `--config`, and `--device`. Serial `flash` and `monitor` follow the shared `--chip` selection rule when `--device` is omitted or names a serial port. `esphome flash --firmware <path>` uploads a prebuilt image instead of the most recent local build: serial flashing expects an ESPHome factory image written at offset `0x0`, while a hostname or IP address expects an ESPHome OTA image.

```bash
./espectre esphome flash --chip c6 --device espectre.local --firmware espectre-esphome-3.0.0-esp32c6-ota.bin
```

`esphome flash --erase` clears all flash data before a serial upload. It resolves or requires a serial device and cannot be combined with an OTA hostname.

Each chip uses one canonical example. The repository CLI keeps that device configuration and switches the ESPectre component source from GitHub to the local checkout.

`esphome build --json` emits one final JSON object after the normal build log. It identifies the frontend, chip, exact application artifact, byte size, and SHA-256 digest for machine consumers. Omitting `--config` keeps canonical config selection inside the CLI.

The wrapper explicitly selects ESPHome's native `esp-idf` toolchain for every command. It does not use the legacy PlatformIO build backend.

For `build`, cleanup flags are:

- `--clean`: run `esphome clean` for the selected config before compiling.
- `--clean-all`: run `esphome clean-all` for the config root before compiling.

### `native` and `matter`

The Native and Matter namespaces expose `build` and `flash`:

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

`native build --json` and `matter build --json` emit the same final build-metadata object as ESPHome, including the exact artifact selected from the resolved chip build directory.

Local builds enable `ccache` automatically when the binary is on `PATH`. Docker builds already keep a persistent compiler cache. Set `IDF_CCACHE_ENABLE=0` to disable the local cache.

Docker builds use a separate directory such as `build-esp32c3-docker`, which prevents host and container CMake caches from sharing incompatible absolute paths. Docker is a build backend only; `flash` continues to use the detected local ESP-IDF environment and host serial port.

For `flash`, `--chip` selects that chip's build directory, such as `build-esp32c5` for `--chip c5`, and verifies that the selected serial device contains the requested chip before erasing or writing flash. Serial selection follows the shared `--chip` rule above. Without `--chip`, the wrapper selects the serial port first, then prefers the build directory that matches the connected chip detected on that port. Without a match, it falls back to the local configured target or the legacy `build/` layout. `--erase` clears all flash data before writing the selected Native or Matter image. On Matter, this also removes the persisted onboarding identity, so the next boot generates a new QR code.

When the current `sdkconfig` already matches the selected chip, `flash` delegates to `idf.py flash`, so ESP-IDF may configure CMake or complete a missing build inside that directory before writing the firmware. When `sdkconfig` belongs to a different chip, `flash` writes the already-built image from the selected directory and does not rebuild. Rebuilds still share one `sdkconfig`, so `native build --chip c5` after an S3 build overwrites that file.

Matter also exposes:

| Command | Purpose |
|---------|---------|
| `qr` | Reset the connected device and print its persisted QR payload and manual pairing code |

Use `matter qr --json` or `matter flash --json` when another tool must consume onboarding data. The final JSON object contains the selected port, chip, QR payload, and manual code; treat that output as a commissioning secret.

`qr` uses the shared serial selection when `--chip` is supplied.

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
./espectre matter qr --port /dev/cu.usbmodemXXXX
```

## Device And Host Commands

### `monitor`

`monitor` attaches to a serial port and streams logs.

Common flags:

- `--port`
- `--chip`
- `--frontend`
- `--baud`
- `--raw`
- `--reset`

When `--chip` is supplied, serial selection follows the shared rule above. Native `monitor` and `provision` first keep ports whose USB console matches the chip, then identify connected chips if more than one candidate remains. An explicit incompatible `--port` is rejected. Without `--chip`, the same selection flow uses all ports compatible with the requested action. By default, `monitor` attaches without resetting the device after the port is chosen. Add `--reset` on UART and USB Serial/JTAG consoles when you want a hard reset on open, for example to capture boot-time logs from the beginning. USB CDC consoles such as the ESP32-S2 TinyUSB console do not expose a generic hard-reset channel; reset those boards manually and run `monitor` without `--reset`.

Example:

```bash
./espectre monitor --chip c3 --frontend native --port /dev/cu.usbmodemXXXX
```

Reset on open:

```bash
./espectre monitor --chip c3 --frontend native --port /dev/cu.usbmodemXXXX --reset
```

### `devices`

`devices` performs a fresh host-side browse for `_espectre._tcp.local.` and lists compatible firmware through one first-party record contract. It does not inspect `_esphomelib`, `_matterc`, or other upstream service types. The normalized result includes the frontend, device identity, display name, chip, IP address, and Direct HTTP endpoint. [`ESPECTRE_PROTOCOL.md`](ESPECTRE_PROTOCOL.md#mdnsdns-sd-discovery) defines the record-level contract.

| Flag | Purpose |
|------|---------|
| `--frontend native|esphome|matter|micro` | Limit discovery to one frontend; omit it to browse every supported service |
| `--chip esp32|c3|s2|s3|c5|c6` | Limit normalized records to one chip family |
| `--timeout <seconds>` | Set the maximum one-shot browse duration; the default is 2.5 seconds |
| `--json` | Emit machine-readable normalized records for scripts and tooling |

Examples:

```bash
./espectre devices
./espectre devices --frontend native
./espectre devices --frontend matter --timeout 5
./espectre devices --frontend matter --chip s3 --json
./espectre devices --frontend esphome
./espectre devices --frontend matter
./espectre devices --json
```

The command uses the repository `zeroconf` dependency and requires the host and device to share an mDNS-visible network. Each invocation starts a new PTR browse; there is no discovery cache. The timeout is an upper bound rather than an unconditional delay: after the first complete record, discovery returns when no record has been added, changed, or removed for 350 ms. If no device responds, it waits for the full timeout. VLAN boundaries, client isolation, and multicast filtering may hide otherwise reachable devices; explicit IP addresses, Native `.local` names, remembered endpoints, and Improv Serial remain the deterministic fallbacks.

### `provision`

`provision` uses the shared Improv Serial v1 client to configure a clean Native or ESPHome device over USB. It accepts the same optional `--chip`, `--frontend`, and `--port` capability-aware selection used by `monitor`. The Wi-Fi password is read from `ESPECTRE_WIFI_PASSWORD` by default, or from the variable named by `--password-env`; when the variable is unset, the CLI prompts without echoing the password. The command validates framing, checksums, state transitions, correlated RPC results, UTF-8 strings, and the returned device URL. Add `--json` to return the selected port, endpoint, and provisioning evidence to another tool.

```bash
ESPECTRE_WIFI_PASSWORD='secret' ./espectre provision --chip c3 --frontend native --port /dev/cu.usbmodemXXXX --ssid MyNetwork
```

The password is never accepted as a command-line value, printed, or included in the returned endpoint. `--timeout` bounds the complete state, device-info, and Wi-Fi provisioning exchange.

### `direct`

`direct` sends one correlated ESPectre protocol `1.0` request through HTTP POST. Supply `--endpoint` with an HTTP(S) device URL, or use `--frontend` to discover a device. Add `--chip` to narrow frontend discovery before selection. When discovery returns multiple matching records, the CLI prompts for an explicit selection.

```bash
./espectre direct status --frontend native
./espectre direct status --frontend matter --chip s3
./espectre direct diagnostics --endpoint http://espectre-0123456789abcdef.local
./espectre direct set_detector --frontend esphome --params '{"detector":"high_accuracy"}'
```

The client sends the exact allowed `https://test.espectre.dev` Origin by default, limits the JSON request to 4,096 bytes, accepts a response up to 8,192 bytes, validates the canonical correlated result, and closes cleanly. The POST body and result use the same message shapes as MQTT `commands/request` and `commands/result`. Use `--origin` only for another exact Origin already allowed by the firmware; the CLI does not weaken device Origin policy.

### `collect`

`collect` is the HTTP-only host-side CSI collection entry point. One runtime path supports three modes:

- live inspection when `--label` is omitted
- live recording when `--label` is set
- read-only dataset inventory when `--info` is used

Common flags:

| Flag | Purpose |
|------|---------|
| `--target` | Device IP, hostname, full Direct endpoint, or device ID; omit it to discover a raw-capable device |
| `--frontend` | Optional `native`, `esphome`, or `matter` discovery filter |
| `--source-ip` | Optional local IPv4 source for hosts with multiple interfaces |
| `--duration` | Stop after N seconds |
| `--label` | Dataset label for saved collections; use 1-64 ASCII letters, digits, underscores, or hyphens, starting with a letter or digit; omit for live inspection without saving |
| `--start-delay` | Wait N seconds before starting collection; requires `--duration` |
| `--pps` | Intentional external UDP generator rate and nominal dataset rate |
| `--detector` | Detector used by the ready gate: `lightweight` or `high_accuracy`; a comma-separated list is available only for live comparison |
| `--ready-stable-seconds` | Seconds below threshold before saved collection starts; set `0` to disable the ready gate |

When `--target` is omitted, `collect` performs one fresh browse for `_espectre._tcp.local.` at startup and keeps raw-capable ESPectre records at their advertised Direct port:

- `0` devices: fail explicitly and suggest `--target`
- `1` device: auto-select it
- `N` devices: prompt for an interactive choice

The collector uses the same event-driven completion as `devices`: once a complete record arrives, 350 ms without a changed record completes discovery. If no record arrives, the 2.5-second default timeout is consumed in full.

`--target` remains the deterministic bypass. The collector resolves an IP, hostname, full Direct endpoint, or full device ID through the same Direct resolver. Native, Matter, and ESPHome use port `62587`; a full manually entered endpoint may specify another explicit port, but the resolver does not probe legacy ports.

`--info` is also read-only: it uses `dataset_info.json` as the source of truth and prints one table per `environment`, with label rows and one column per chip.

Live collection negotiates raw HTTP, persistently sets `csi_traffic_mode` to `external`, verifies the resulting configuration, opens one bearer-bound binary response stream, and starts `ExternalTrafficGenerator` from `tools/espectre_traffic_generator.py`. The generator sends the exact four-byte UTF-8 UDP marker `"👻".encode("utf-8")` (`F0 9F 91 BB`) at `--pps`; the device forwards every classified CSI frame without HTTP pacing or temporal decimation. The generator stops before the raw session, and the collector intentionally does not restore the previous traffic mode.

Example:

```bash
./espectre collect --target 192.168.1.51 --pps 100
```

Saved files and catalog entries record `transport=http`, the Direct endpoint, requested and observed PPS, raw protocol version, CSI record version, frontend, chip, firmware, and device ID.

Collection terms:

- **Delivered rate:** CSI records received by the collector, measured in packets per second (`pps`).
- **Admitted rate:** records that occupy a detector slot after temporal admission.
- **Excess:** extra same-slot records that do not improve occupancy.
- **Backpressure:** firmware reports that it cannot transmit records as quickly as they are produced.
- **Queue drop:** a classified record rejected because the fixed 16-record raw ring is full.

The external generator is the sole rate owner. HTTP applies no credit window, adaptive rate, sample replacement, or device-side timer. After drain, the invariant `fresh_record_total + raw_drop_total == classified_frames_offered_to_raw` exposes any hidden loss before the network send.

`--detector` selects the production detector used for collection readiness. The derived live sensing view applies the production temporal sampler to raw records using the nominal `--pps`; the saved raw stream remains un-decimated. `lightweight` performs its normal startup calibration before it can become ready. `high_accuracy` does not use startup calibration, but still needs its feature window to fill. Live inspection can compare `lightweight,high_accuracy` in parallel.

When `--label` is set, saved collection waits for the detector to stay below threshold for `--ready-stable-seconds` before packets are recorded. Set `--ready-stable-seconds 0` to bypass that gate explicitly.

After saving each capture, the collector runs the validator's canonical per-file integrity, signal-quality, temporal-occupancy, and stream-continuity checks. Temporal occupancy is measured on complete production detector windows, warns below 85%, and fails below the shared 70% admission floor. The post-collect summary does not use average packet rate as a quality proxy because excess same-slot records do not improve detector occupancy. A failed capture remains saved for diagnosis, but `collect` exits unsuccessfully.

When `--start-delay` is set, `--duration` is required. The collector waits first, then starts the ordinary generator and capture flow.

For discovery-selected targets, the collector also validates that CSI records carry the same `device_id` announced over mDNS. If an address was reused by another device, collection aborts instead of saving mixed data under the wrong identity.

Examples:

```bash
./espectre devices --frontend native
./espectre collect --target 192.168.1.50 --pps 120
./espectre collect --frontend esphome --pps 120
./espectre collect --label wave --duration 45 --target espectre-0123456789abcdef.local
./espectre collect --label wave --duration 45 --start-delay 15 --target http://192.168.1.50
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
espectre/v1/devices/{device_id}/commands/result
espectre/v1/devices/{device_id}/capabilities
espectre/v1/devices/{device_id}/info
espectre/v1/devices/{device_id}/status
espectre/v1/devices/{device_id}/config
espectre/v1/devices/{device_id}/ota_status
```

After selection the shell consumes the retained `capabilities` schema to populate help and tab completion. Every query, mutation, and action returns through `commands/result`; query payloads are nested in `data`. Command results annotate the typed prompt line with `✓` or `✗ code: reason` when the terminal allows it. Otherwise they appear on the next line. Retained state topics are still dumped as YAML.

This behavior belongs to the MQTT transport and applies to ESPectre devices that advertise the MQTT topic surface.

Common MQTT flags:

| Flag | Default |
|------|---------|
| `--broker` | `homeassistant.local` or `MQTT_BROKER` |
| `--port-mqtt` | `1883` or `MQTT_PORT` |
| `--topic-prefix` | `espectre/v1/devices` or `MQTT_TOPIC_PREFIX` |
| `--device-id` | Explicit device identifier; otherwise runtime discovery |
| `--username` | `mqtt` or `MQTT_USERNAME` |
| `--password` | `mqtt` or `MQTT_PASSWORD` |

Examples:

```bash
./espectre mqtt
./espectre mqtt --device-id 3cf79180d3a0aca4
./espectre mqtt --broker 192.168.1.20 --device-id native-lab
```

MQTT commands are forwarded to the selected device. The shell keeps only local utilities (`help`, `about`, `clear`, and `exit`) plus short read aliases such as `i` and `d`. Help, tab completion, and argument discovery use the device `capabilities` schema. Unknown or unsupported commands are rejected by the device with a stable result code. Write values after the command name (`set_threshold 0.35`). Multi-field writes use named tokens after the command (`set_motion_hits motion_on_hits=4 motion_off_hits=3`).

`ota_check` and `ota_start` accept an optional channel (`release`, `preview`, or `develop`), for example `ota_check preview` or `ota_start channel=develop`. Omitting the channel keeps the firmware's build-time default. OTA payloads containing server, manifest, image, or version overrides are rejected by the device. Frontends omit unsupported OTA commands from `capabilities`.

Native builds accept `--ota-channel release|preview|develop`. The selected value is compiled into the firmware and is used whenever an MQTT OTA command omits `channel`; it is propagated through both local and Docker build backends. The default is `release`, or `NATIVE_OTA_CHANNEL` when that environment variable is set.

Browser tools such as Flash, Configure, Monitor, and Theremin live on [espectre.dev](https://espectre.dev). Serial logs remain `./espectre monitor`.

## Utility Commands

| Command | Purpose |
|---------|---------|
| `./espectre doctor` | Validate the ESP-IDF environment used by the wrapper |
| `./espectre version` | Show the current CLI version label |
| `./espectre about` | Show project and CLI information |

## Related Documents

- [`SETUP.md`](SETUP.md) for shared setup, frontend selection, and entry points
- frontend READMEs under `src/cpp/frontend/` for frontend-specific build, flash, provisioning, and protocol details
