# ESPectre SDK <img src="https://espectre.dev/assets/images/brand/espectre-logo.svg" alt="ESPectre logo" width="40" align="absmiddle" />

[Home](https://espectre.dev/) · [Tools](https://espectre.dev/tools/) · [Guides](https://espectre.dev/guides/) · [SDK](https://espectre.dev/sdk/) · [Roadmap](https://espectre.dev/roadmap/) · [Media](https://espectre.dev/media/) · [GitHub](https://github.com/francescopace/espectre) · [Contacts](https://espectre.dev/contact/) · [Commercial License](https://espectre.dev/licensing/)

ESPectre adds Wi-Fi motion detection to ESP-IDF firmware. It reads Channel State Information (CSI) from the chip's own radio, so no extra hardware is needed. The SDK captures CSI, calibrates, and reports motion through a C++ API. Your firmware keeps control of Wi-Fi, tasks, and what to do with the result. MQTT, Direct HTTP, discovery, and provisioning are optional.

## Requirements

| Requirement | Supported configuration |
|-------------|-------------------------|
| ESP-IDF | `>=5.5.3`; release builds use 5.5.6 |
| C++ | C++17 or later; CI builds use C++17, C++20, and C++23 |
| Target | ESP32, ESP32-S2, ESP32-S3, ESP32-C3, ESP32-C5, or ESP32-C6 |
| License | `GPL-3.0-only`, or a separate commercial license; see [Licensing](#licensing) |

ESP-IDF 5.5.0–5.5.2 are not supported because their ESP32-C5 CSI configuration lacks the `lltf_bit_mode` field. The minimum applies to every target.

### ESP-IDF compatibility validation

On 2026-10-21 the `wifi_motion_detection` example, with all optional services enabled, built and linked with these ESP-IDF versions:

| ESP-IDF image | ESP32 | ESP32-S2 | ESP32-S3 | ESP32-C3 | ESP32-C5 | ESP32-C6 |
|---------------|-------|----------|----------|----------|----------|----------|
| `v5.5.3` | Pass | Pass | Pass | Pass | Pass | Pass |
| `v5.5.4` | Pass | Pass | Pass | Pass | Pass | Pass |
| `v5.5.5` | Pass | Pass | Pass | Pass | Pass | Pass |
| `v5.5.6` | Pass | Pass | Pass | Pass | Pass | Pass |
| `v6.0.3` | Pass | Pass | Pass | Pass | Pass | Pass |
| `v6.1` | Pass | Pass | Pass | Pass | Pass | Pass |

These are build checks only; ESP-IDF 6.x has not been tested on hardware yet. See [ESP-IDF compatibility checks](RELEASING.md#esp-idf-compatibility-checks) for what remains.

## Installation

The SDK is published on the ESP Component Registry as `francescopace/espectre`. Component Manager downloads the sources and builds them with your application.

| Registry | Contents |
|----------|----------|
| [Production](https://components.espressif.com/components/francescopace/espectre) | Tagged releases and release candidates |
| [Staging](https://components-staging.espressif.com/components/francescopace/espectre) | Snapshots of `main` (versions ending in `.main`) and `develop` (ending in `.develop`) |

Release candidates and snapshots are for evaluation, and their API can still change. Old snapshots are removed from staging over time.

### Add to an existing project

From your project directory, add the latest stable release:

```sh
idf.py add-dependency "francescopace/espectre"
```

To pin a version, or to use staging, copy the version from the registry page and set the registry URL. The copy of this guide inside a registry package already has both values filled in.

```sh
ESPECTRE_VERSION="VERSION_FROM_REGISTRY"
ESPECTRE_REGISTRY_URL="https://components.espressif.com"
idf.py add-dependency --registry-url "$ESPECTRE_REGISTRY_URL" "francescopace/espectre=$ESPECTRE_VERSION"
```

The dependency is saved in `main/idf_component.yml` and the resolved version in `dependencies.lock`. Run `idf.py update-dependencies` to move to a newer compatible version.

### Try the example

The registry package includes a complete project that connects to Wi-Fi and logs motion. With the same two values set:

```sh
idf.py create-project-from-example --registry-url "$ESPECTRE_REGISTRY_URL" "francescopace/espectre=$ESPECTRE_VERSION:wifi_motion_detection"
cd wifi_motion_detection
idf.py set-target esp32c3
idf.py menuconfig
idf.py build
idf.py -p YOUR_PORT flash monitor
```

Set your SSID and password under **ESPectre example** in menuconfig. The [example guide](../src/cpp/examples/wifi_motion_detection/README.md) explains how it works and what to check on the board.

## Add sensing to your firmware

1. Add the [recommended project configuration](#recommended-project-configuration) to `sdkconfig.defaults`. Only `CONFIG_ESP_WIFI_CSI_ENABLED=y` is required.
2. Register a [log sink](#logging) if you want SDK messages. The SDK has no default logger.
3. Initialize NVS, ESP-NETIF, and the default event loop, and create the station interface. Set up the runtime before you start the station, so it can apply its radio settings.
4. Build a `RuntimeConfig`, call `setup()`, and keep calling `loop()` from the same task.
5. Publish motion only while `ready_to_publish` is true. It is false during calibration and while Wi-Fi recovers.

```cpp
#include "espectre_sdk.h"

class MotionSensor : public espectre::IRuntimeListener {
 public:
  bool setup() {
    auto config = espectre::make_runtime_sensing_config_from_kconfig();
    config.device_id = espectre::derive_runtime_device_id();  // Needed for the protocol and CSI streaming.
    return runtime_.set_config(config) && runtime_.setup(this);
  }

  void loop() { runtime_.loop(); }
  void shutdown() { runtime_.shutdown(); }

  void on_motion_state_changed(const espectre::RuntimeSnapshot &snapshot) override {
    if (snapshot.ready_to_publish) {
      publish_motion(snapshot.motion_state == espectre::MotionState::MOTION);
    }
  }

  void on_sensing_readiness_changed(const espectre::RuntimeSnapshot &snapshot) override {
    if (!snapshot.ready_to_publish) {
      mark_unavailable();  // Otherwise the last published value stays in place.
    }
  }

 private:
  void publish_motion(bool motion);  // Hand off to another task; do not block.
  void mark_unavailable();
  espectre::RuntimeFrontendController runtime_;
};
```

`RuntimeFrontendController` runs the sensing and calls you back through `IRuntimeListener`. Keep the listener alive until `shutdown()` returns. Call `runtime_.snapshot()` whenever you need the current state. In SDK names, "frontend" means your integration code.

### Wi-Fi

If your firmware already manages Wi-Fi, keep it, and check `traffic_allows_radio_work()` before reconfiguring the station and `wifi_scan_allowed()` before scanning. While such work waits, call `hold_pending_traffic_restart(true)`, and call it with `false` once the work has reached the driver. Set `wifi_scan_results_managed_externally` if your code consumes every scan result itself.

If you have no Wi-Fi code yet, use `StandaloneWifiService` from `espectre_services_sdk.h`, as the example does. It needs no optional group.

- Initialize NVS first. The service creates the netif and event loop and starts the driver.
- Call `setup()` on the service, then on the runtime, then `start()` on the service. Call both `loop()` methods from one task.
- Connect `set_radio_work_ready_callback()` to `traffic_allows_radio_work()`, and before each runtime `loop()` call `hold_pending_traffic_restart(wifi.has_deferred_radio_work())`.
- Set `StandaloneWifiConfig::band_policy` from `RuntimeConfig::wifi_band_policy`: the service defaults to 2.4 GHz.
- The `ssid`, `password`, and `bssid` strings must stay valid until shutdown.
- After `max_retry` quick reconnects (default 8), it waits 30 seconds and tries again, until it connects or shuts down.
- Shut down the runtime first, then the service.

### Recommended project configuration

The Native, Matter, and ESPHome firmware use these options:

| Option | Value | Why |
| --- | --- | --- |
| `CONFIG_ESP_WIFI_CSI_ENABLED` | `y` | Required. Enables CSI in the Wi-Fi driver. |
| `CONFIG_ESP_WIFI_AMPDU_RX_ENABLED` | `n` | The radio reports one CSI sample per transmission, so unaggregated frames give more samples. |
| `CONFIG_ESP_WIFI_AMPDU_TX_ENABLED` | `n` | Needed for a fixed [transmit rate](CSI.md#transmit-rate). With it on, the runtime uses automatic rates. |
| `CONFIG_PM_ENABLE` | `n` | Keeps frequency scaling and light sleep off. This is the ESP-IDF default. |
| `CONFIG_ESP_WIFI_STA_DISCONNECTED_PM_ENABLE` | `n` | Keeps Wi-Fi power saving off while reconnecting. |
| `CONFIG_FREERTOS_HZ` | `1000` | Millisecond timing for traffic pacing. |
| `CONFIG_LWIP_MULTICAST_PING` | `y` | Answers ICMP Echo Requests sent to the [external traffic](CSI.md#external-sources) multicast group. |

Turning off A-MPDU lowers Wi-Fi throughput for the whole firmware, so OTA updates and other network services get slower. The Native firmware also raises Wi-Fi buffer counts and lwIP mailbox sizes; see its [sdkconfig.defaults](../src/cpp/frontend/native/app/sdkconfig.defaults).

### Logging

Register a `LogSink` with `set_log_sink()` before setup, and check the result. It has two callbacks: `enabled(context, level, tag)` decides whether a message is formatted, and `write(context, level, tag, line, format, args)` writes it. The arguments are valid only during the call.

Both callbacks must be thread-safe and must not block: they can run on your task, on service tasks, or in the CSI capture path. Do not log through ESPectre from inside a sink. Keep the sink's context alive, and replace or clear the sink only after the runtime has stopped. The example forwards to ESP-IDF logging through `esp_log_va`.

### Reference firmware

The repository has three complete firmware built on the public SDK. They are not part of the SDK package.

| Firmware | What it shows | Guide |
|----------|---------------|-------|
| Native | Standalone ESP-IDF application with Direct HTTP, optional MQTT, and Home Assistant MQTT Discovery | [Native guide](../src/cpp/frontend/native/README.md) |
| ESPHome | ESPHome component that maps YAML configuration and Home Assistant entities to the runtime | [ESPHome guide](../src/cpp/frontend/esphome/README.md) |
| Matter | Matter occupancy sensor with commissioning and a Direct HTTP bridge for sensing controls | [Matter guide](../src/cpp/frontend/matter/README.md) |

For Arduino-ESP32 sketches, the [Arduino library](../src/cpp/frontend/arduino/README.md) wraps the runtime; it is in development.

## Public headers

| Header | Use it for |
|--------|------------|
| `espectre_sdk.h` | Sensing runtime, configuration, snapshots, and listener callbacks |
| `espectre_core_sdk.h` | Detectors and temporal sampling when your application captures CSI itself |
| `espectre_protocol_sdk.h` | ESPectre Protocol messages, JSON, diagnostic fields, and the Direct HTTP and MQTT contracts |
| `espectre_services_sdk.h` | Optional Direct HTTP, discovery, provisioning, and application services; includes the protocol header |
| `espectre_mqtt_sdk.h` | The ESP-IDF MQTT transport |

Only what these headers expose is public. Optional services also need their [capability group](#optional-capability-groups).

## Optional capability groups

All groups are off by default, so the base SDK pulls in no network stack. Enable what you need under **ESPectre SDK** in menuconfig:

| Menuconfig option | Adds | Pulls in |
|-------------------|------|----------|
| `ESPECTRE_SDK_ENABLE_MQTT` | `EspIdfMqttTransport` | `mqtt` on ESP-IDF 5.5; `espressif/mqtt` 1.0.0 on ESP-IDF 6 |
| `ESPECTRE_SDK_ENABLE_DIRECT` | Direct HTTP, SSE, raw CSI streaming, peer discovery, and mDNS | `esp_http_server` and `espressif/mdns` 1.12 to 1.14 |
| `ESPECTRE_SDK_ENABLE_PROVISIONING` | Device configuration store and Wi-Fi provisioning | Nothing extra |
| `ESPECTRE_SDK_ENABLE_FRONTEND_SUPPORT` | Shared bootstrap, control, sysinfo, and MQTT payload helpers | Turns on provisioning |

The mDNS version is pinned because discovery uses private mDNS functions. Component Manager reads these options from Kconfig (ESP-IDF 5.5 and Component Manager 2.2 or newer), so:

- Declare ESPectre as a direct dependency of your project.
- Run `idf.py update-dependencies` after turning MQTT or Direct on or off.
- If you use `COMPONENTS` or `MINIMAL_BUILD`, add `mqtt` (ESP-IDF 5) and `esp_http_server` yourself.

The registry page lists every optional dependency, even those a minimal build does not use.

A few things the groups do not do:

- The SDK does not set up a console or USB.
- Provisioning stores the credentials you give it, but the onboarding flow is yours; Improv Serial is not included.
- `DirectHttpServiceConfig` allows no browser origins by default. `for_first_party_portals()` allows the official ESPectre web tools.
- `IMqttTransport` and `IDirectHttpService` are header-only, so you can implement your own without enabling a group.

## Shared sensing options

Set these `RuntimeConfig` fields before setup. The defaults are those of a plain `RuntimeConfig{}`. To start from menuconfig, call `make_runtime_sensing_config_from_kconfig()` and override what you need. The options are under **ESPectre Sensing**. Kconfig choices match the C++ values, such as `CONFIG_ESPECTRE_CSI_CAPTURE_PROFILE_HT_VHT=y` for `CsiCapturePolicy::HT_VHT`. The one exception is `CONFIG_ESPECTRE_TRAFFIC_GENERATOR_MODE_EXTERNAL`, which means `TrafficGeneratorMode::EXTERNAL_HOST`.

| `RuntimeConfig` field | Type and values | Default | Notes |
|-----------------------|-----------------|---------|-------|
| `wifi_band_policy` | `WifiBandPolicy`: `BAND_2G`, `BAND_5G`, or `AUTO` | `AUTO` | `AUTO` uses every band the chip has. `BAND_5G` needs ESP32-C5 and fails setup elsewhere |
| `detection_algorithm` | `DetectionAlgorithm`: `LIGHTWEIGHT` or `HIGH_ACCURACY` | `LIGHTWEIGHT` | See [choosing a detection profile](#choosing-a-detection-profile) |
| `threshold` | `float` | `RUNTIME_THRESHOLD_DETECTOR_DEFAULT` | `0-1`. The default follows the detector: Lightweight calibrates it, High Accuracy uses a fixed value |
| `runtime_detector_selection_enabled` | `bool` | `false` | Allows `set_detection_algorithm()` at runtime; both detectors are then built into flash |
| `window_size_ms` | `uint32_t` | `1000` | `1000-2000` ms detector window |
| `csi_target_pps` | `uint32_t` | `100` | `1-500` packets per second the detector and managed traffic aim for |
| `csi_sample_spacing` | `TemporalCsiSpacing`: `HALF_SLOT` or `DISTINCT` | `HALF_SLOT` | `DISTINCT` helps on congested channels. See [temporal admission](ALGORITHMS.md#temporal-admission) |
| `csi_capture_policy` | `CsiCapturePolicy`: `AUTO`, `LLTF`, or `HT_VHT` | `AUTO` | Fixed after setup. Other packets are dropped and counted. See [capture profiles](CSI.md#capture-profiles) |
| `traffic_generator_mode` | `TrafficGeneratorMode`: `PING`, `DNS`, `DNS_TCP`, `WIFI_RAW`, or `EXTERNAL_HOST` | `PING` | What generates the packets the device measures. `WIFI_RAW` is not available on ESP32-C6 and needs `AUTO` or `LLTF` capture. `EXTERNAL_HOST` sends nothing and listens for [external CSI traffic](API.md#external-csi-traffic) |
| `traffic_generator_target_ip` | `std::string` | empty | See [traffic destination](#traffic-destination) |
| `traffic_generator_target_mac` | `std::string` | `auto` | See [traffic destination](#traffic-destination) |
| `csi_traffic_udp_port` | `uint16_t` | `5555` | UDP port for `EXTERNAL_HOST` traffic |
| `csi_traffic_multicast_group` | `std::string` | `"239.255.0.1"` | Multicast group for `EXTERNAL_HOST` traffic; empty disables it. Unicast to the device still works |
| `evaluation_interval_ms` | `uint32_t` | `250` | `10-10000` ms between detector evaluations |
| `motion_on_hits` | `uint8_t` | `4` | `1-20` consecutive hits to switch to motion |
| `motion_off_hits` | `uint8_t` | `3` | `1-20` consecutive hits to switch back to idle |
| `lowpass_enabled` | `bool` | `false` | Low-pass filter |
| `lowpass_cutoff` | `float` | `11.0` | `5.0-20.0` Hz, tuned for 100 pps |
| `hampel_enabled` | `bool` | `true` | Hampel outlier filter |
| `hampel_window` | `uint8_t` | `7` | `3-11` samples |
| `hampel_threshold` | `float` | `5.0` | `1.0-10.0` MAD units |
| `persist_runtime_overrides` | `bool` | `false` | Saves runtime changes in NVS; see below |
| `wifi_scan_results_managed_externally` | `bool` | `false` | Set it when your Wi-Fi code consumes every scan result |
| `device_id` | `uint64_t` | `0` | Needed for the ESPectre Protocol and CSI streaming; use `derive_runtime_device_id()` |

Defaults and range checks are in [runtime_sensing_schema.h](../src/cpp/runtime/runtime_sensing_schema.h) and [runtime_config_utils.cpp](../src/cpp/runtime/runtime_config_utils.cpp). Motion hits are explained in [motion-hit filtering](ALGORITHMS.md#motion-hit-filtering).

The transmit rate is a build option, not a field: `CONFIG_ESPECTRE_WIFI_TX_RATE_MBPS` is `"0"` (automatic), `"6"`, or `"6.5"`, with the quotes. It defaults to `"6.5"` on ESP32 and `"0"` on other chips. See [transmit rate](CSI.md#transmit-rate).

Some settings can also change at runtime through the controller setters or the [sensing operations](API.md#sensing-update-and-calibration). Check `capabilities()` first. [Tuning essentials](TROUBLESHOOTING.md#tuning-essentials) explains when a change helps.

By default nothing is saved, and your `RuntimeConfig` is applied on every boot. With `persist_runtime_overrides`, setter changes to the traffic mode, raw destination MAC, sample spacing, motion hits, and selectable detector are saved in the `espectre` NVS namespace. Saved values win over your configuration until `clear_persisted_overrides()` erases them. The threshold is never saved.

### Choosing a detection profile

Lightweight leaves more CPU and memory for your application and calibrates its threshold in a quiet room. High Accuracy detects better at a higher cost. Change `detection_algorithm` alone and the default threshold follows. See [why two detection profiles](ALGORITHMS.md#why-two-detection-profiles).

### Traffic destination

`traffic_generator_target_ip` is where `PING`, `DNS`, and `DNS_TCP` send packets. Leave it empty to use the Wi-Fi gateway. Otherwise use a unicast IPv4 address, such as `192.168.1.10`, of a host that answers that protocol; DNS needs a resolver on port 53. Hostnames, loopback, multicast, and reserved addresses are rejected. It can only be set before setup, or with `CONFIG_ESPECTRE_TRAFFIC_GENERATOR_TARGET_IP`.

`traffic_generator_target_mac` is where `WIFI_RAW` sends Null Data frames. `auto` follows the access point the device is connected to, including after roaming. A unicast MAC sends to another ESPectre device on the same channel running in `EXTERNAL_HOST` mode, so you can place both ends of the sensing link where you want; see [two-device sensing](CSI.md#two-device-sensing). Set it with `CONFIG_ESPECTRE_TRAFFIC_GENERATOR_TARGET_MAC`, or change it at runtime with `set_traffic_generator_target_mac()`. Group, broadcast, and zero MACs are rejected. The device's own MAC is rejected at runtime; set before setup, it stops raw traffic from starting.

## Runtime contract

These are the rules every integration needs. The [integration reference](../src/cpp/sdk_integration.dox) has the full details.

### Lifecycle

Call `set_config()`, `setup(listener)`, `loop()` repeatedly, and `shutdown()`. Check every `bool` result. `set_config()` fails while the runtime is running, so shut down first. A failed setup can be retried, and the configuration survives `shutdown()`.

After setup, `config()` returns the values in use, including the resolved threshold. Writing to it only affects the next setup; use the setters for live changes.

### Threading

Make every lifecycle and control call from one task. Listener callbacks run on that task, never in the Wi-Fi capture context. Keep them short and hand network or storage work to another task: a slow callback delays `loop()` and can overflow the CSI queue. Commands that arrive on network callbacks should also be queued and applied from the owner task.

Raw CSI packet callbacks are the exception. They run in the Wi-Fi capture context, so they must not block or allocate; copy samples into a preallocated queue. Each raw view holds 128 bytes (`HT20_CSI_LEN`).

### Errors and capabilities

Check `capabilities()` after setup before you offer a control. A control method returns `false` when the value is out of range, the capability is missing, or the backend refuses, and the runtime stays unchanged. Later faults arrive through `on_runtime_fault()`. A failed calibration is not fatal: sensing continues with the current threshold.

### Diagnostics

`diagnostics_sample()` returns a one-second sample of traffic and CSI rates and the current link. `diagnostics()` returns the cumulative counters behind it. [API diagnostics](API.md#diagnostics) describes the fields.

### Versioning

The public headers follow Semantic Versioning for source compatibility: patch releases keep documented behavior, minor releases add compatible APIs, and major releases can break them. There is no binary ABI promise, so rebuild the SDK with your firmware.

- Create configuration and snapshot structs with their defaults and set fields by name, because minor releases can add fields.
- `CsiCaptureProfile` and `CsiCapturePolicy` can gain values, so handle unknown ones.
- The core-only interface in `espectre_core_sdk.h` can change in a minor release until the stationary presence detector ships.
- Detector coefficients can change in fixes, so telemetry values are not bit-exact across releases.

Use `ESPECTRE_SDK_VERSION_STRING` to identify the SDK and `ESPECTRE_SDK_VERSION_AT_LEAST(major, minor, patch)` for compile-time checks. Your firmware version (`EspectreDeviceInfo::firmware_version`) and `ESPECTRE_PROTOCOL_VERSION` are separate. To override the packaged version, define all four `ESPECTRE_SDK_VERSION_*` macros.

## Advanced integrations

### Core-only

If your application captures CSI itself, include `espectre_core_sdk.h` and feed the detectors through `TemporalCsiSampler`. Check `detector.is_valid()` and the result of `sampler.configure(...)`, since both allocate buffers without throwing. You then drive the sampler, the evaluation cadence, and motion-hit filtering. Re-read `get_threshold()` after each `update_state()`, because Lightweight can lower it without a callback.

Normalize each payload to the centered HT20 layout. For LLTF captures, call `zero_ht20_lltf_missing_bins()` on the raw view, copy it into your own detector buffer, and call `prepare_ht20_detector_input()` on the copy. The [integration reference](../src/cpp/sdk_integration.dox) gives the full sequence, and [csi_pipeline.cpp](../src/cpp/runtime/esp_idf/csi_pipeline.cpp) shows how the runtime does it.

### Build integration

Component Manager configures everything for you. To vendor the SDK, use the directory that contains `espectre_sources.cmake` as the SDK root: the registry package root, or `src/cpp/` in a source bundle.

- As an ESP-IDF component: copy the SDK root to `components/espectre/`, add `espectre` to your `REQUIRES`, and enable groups in menuconfig.
- As a core-only library: compile `ESPECTRE_CORE_SOURCES` with `ESPECTRE_SHARED_INCLUDE_DIRS`.
- As a full runtime from source lists: also compile `ESPECTRE_RUNTIME_ESP_IDF_SOURCES`, link `ESPECTRE_RUNTIME_ESP_IDF_TRAFFIC_LINK_OPTIONS`, require `esp_wifi`, `esp_event`, `esp_netif`, `esp_timer`, `nvs_flash`, `lwip`, `freertos`, `heap`, and `mbedtls`, and add the groups you need:

| Group | Source list | Also needs |
|-------|-------------|------------|
| MQTT | `ESPECTRE_RUNTIME_ESP_IDF_MQTT_SOURCES` | `mqtt` |
| Direct | `ESPECTRE_RUNTIME_ESP_IDF_DIRECT_SOURCES` | `esp_http_server`, `mdns` and its `private_include` directory, and `ESPECTRE_RUNTIME_ESP_IDF_DIRECT_LINK_OPTIONS` |
| Provisioning | `ESPECTRE_RUNTIME_ESP_IDF_PROVISIONING_SOURCES` | Nothing extra |
| Frontend support | `ESPECTRE_RUNTIME_FRONTEND_SUPPORT_SOURCES` | The provisioning sources |

For core-only firmware that still wants managed traffic, add `ESPECTRE_RUNTIME_ESP_IDF_TRAFFIC_SOURCES` and its link options; it needs `esp_netif`, `esp_timer`, `esp_wifi`, `freertos`, and `lwip`.

A core-only target from a source bundle extracted into `espectre/`:

```cmake
set(ESPECTRE_CPP_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/espectre/src/cpp")
include("${ESPECTRE_CPP_ROOT}/espectre_sources.cmake")
add_library(espectre_core STATIC ${ESPECTRE_CORE_SOURCES})
target_compile_features(espectre_core PUBLIC cxx_std_17)
target_include_directories(espectre_core PUBLIC ${ESPECTRE_SHARED_INCLUDE_DIRS})
```

Link your target to `espectre_core` to inherit the includes and C++ standard. The SDK root is the only include directory, so include other headers by path, such as `runtime/runtime_config.h`.

The traffic link options wrap `esp_netif_new`, `esp_netif_destroy`, `esp_netif_receive`, and `esp_netif_transmit_wrap` to count station traffic, so your application cannot wrap them too. The counters follow the default `WIFI_STA_DEF` interface. The Direct options also wrap the private mDNS function `mdns_priv_receive_action`.

### Transport and protocol extensions

Route parsed requests through `FrontendCommandEngine`, and describe your own routes with an `EspectreProtocolExtension` checked by `validate_protocol_extension()`. Use the same catalog for capabilities, Direct HTTP, and command parsing. [Contract principles](API.md#contract-principles) describes the shared message format. The integration reference covers handlers and `quiesce()` for firmware updates you run yourself.

### Task priorities

You choose the priority of the task that calls `loop()`. The SDK's own service tasks are set at build time under **Advanced task scheduling**, from `1` to `10`:

| Kconfig option | Default | Task |
|----------------|---------|------|
| `CONFIG_ESPECTRE_DIRECT_HTTPD_TASK_PRIORITY` | `1` | Direct HTTP server |
| `CONFIG_ESPECTRE_DIRECT_WORKER_TASK_PRIORITY` | `2` | Direct responses and SSE |
| `CONFIG_ESPECTRE_RAW_WORKER_TASK_PRIORITY` | `3` | Raw CSI over HTTP |
| `CONFIG_ESPECTRE_TRAFFIC_TASK_PRIORITY` | `1` | Managed PING or DNS traffic |

Test any change under your real workload: a higher-priority task can starve sensing or networking.

## Source bundles

To vendor the sources without Component Manager, download the `.tar.gz` or `.zip` SDK assets from [GitHub Releases](https://github.com/francescopace/espectre/releases). Tagged releases are named `espectre-sdk-<version>`. The rolling [snapshot](https://github.com/francescopace/espectre/releases/tag/snapshot) release carries `espectre-sdk-preview` from `main`, and [snapshot-dev](https://github.com/francescopace/espectre/releases/tag/snapshot-dev) carries `espectre-sdk-develop` from `develop`. Each comes with an `sdk-manifest-*.json` that records the source commit and SHA-256 digests.

Bundles place the SDK under `src/cpp/`; registry packages place it at the root and add the example. Both are built with your application and contain no precompiled libraries.

Registry packages also include a generated `API.md` with the full C++ reference for that version. Bundles include the header comments, `src/cpp/sdk_integration.dox`, and a stamped `src/cpp/Doxyfile`: run `doxygen src/cpp/Doxyfile` with Doxygen 1.18.0 from the bundle root to generate the reference XML in `output/xml/`.

## Licensing

The SDK is `GPL-3.0-only`, which allows commercial use under the GPLv3 terms, including sharing the corresponding source. A separate commercial license is available for closed-source firmware. See the [licensing terms](../LICENSING.md).
