# ESPectre SDK <img src="https://espectre.dev/assets/images/brand/espectre-logo.svg" alt="ESPectre logo" width="40" align="absmiddle" />

[Home](https://espectre.dev/) · [Tools](https://espectre.dev/tools/) · [Guides](https://espectre.dev/guides/) · [SDK](https://espectre.dev/sdk/) · [Roadmap](https://espectre.dev/roadmap/) · [Media](https://espectre.dev/media/) · [GitHub](https://github.com/francescopace/espectre) · [Contacts](https://espectre.dev/contact/) · [Commercial License](https://espectre.dev/licensing/)

ESPectre adds motion detection to ESP-IDF firmware using Wi-Fi Channel State Information (CSI). It provides detectors, calibration, and motion events through a C++ API. The runtime owns CSI capture and sensing; your application owns networking, task scheduling, and how it uses the results. Optional services support MQTT, Direct HTTP, discovery, and provisioning.

## Requirements

| Requirement | Supported configuration |
|-------------|-------------------------|
| ESP-IDF | `>=5.5.3`; release builds use 5.5.5 |
| C++ | C++17 or later; CI builds use C++17, C++20, and C++23 |
| Target | ESP32, ESP32-S2, ESP32-S3, ESP32-C3, ESP32-C5, or ESP32-C6 |
| License | `GPL-3.0-only`, with a separate commercial agreement available; see [Licensing](#licensing) |

### ESP-IDF compatibility validation

On 2026-09-20 the `wifi_motion_detection` example, with MQTT, provisioning, Direct, and frontend support enabled, built and linked with each ESP-IDF version below:

| ESP-IDF image | ESP32 | ESP32-C3 | ESP32-C5 | ESP32-C6 |
|---------------|-------|----------|----------|----------|
| `v5.5.3` | Pass | Pass | Pass | Pass |
| `v5.5.4` | Pass | Pass | Pass | Pass |
| `v5.5.5` | Pass | Pass | Pass | Pass |
| `v6.0.3` | Pass | Pass | Pass | Pass |
| `v6.1.0` | Pass | Pass | Pass | Pass |

ESP-IDF 5.5.0–5.5.2 are not supported: their ESP32-C5 CSI configuration lacks the `lltf_bit_mode` field the runtime uses for 8-bit LLTF samples ([5.5.2](https://github.com/espressif/esp-idf/blob/v5.5.2/components/esp_wifi/include/esp_wifi_he_types.h), [5.5.3](https://github.com/espressif/esp-idf/blob/v5.5.3/components/esp_wifi/include/esp_wifi_he_types.h)). The minimum applies to every target.

These are build checks only; hardware validation on ESP-IDF 6.x is still pending. See [ESP-IDF compatibility checks](RELEASING.md#esp-idf-compatibility-checks) for the details and the remaining release checks.

## ESP Component Registry

Install `francescopace/espectre` through ESP-IDF Component Manager. It downloads the SDK sources and builds them with your application; no manual archive download is required.

| Registry | Available versions | Website channel |
|----------|--------------------|-----------------|
| [Production](https://components.espressif.com/components/francescopace/espectre) | Tagged releases, including prereleases | Release |
| [Staging](https://components-staging.espressif.com/components/francescopace/espectre) | Branch snapshots | Preview (`main`) and Develop (`develop`) |

Tagged prereleases, such as release candidates, are published to production for evaluation; their API can still change before the final release. Older prereleases remain on staging.

To pin a version or create the example project, pick an exact version on the registry page and set it below. Use the production URL for tagged releases and the staging URL for branch snapshots.

```sh
ESPECTRE_VERSION="VERSION_FROM_REGISTRY"
ESPECTRE_REGISTRY_URL="https://components.espressif.com"
```

Registry packages fill in these values for you. Snapshot versions end in `.main` or `.develop`. Old snapshots are removed from staging over time, so pick a recent one.

### Add to an existing project

From your ESP-IDF project directory, add the latest compatible stable release from the default production registry:

```sh
idf.py add-dependency "francescopace/espectre"
```

The resolved version is saved in `dependencies.lock`. Run `idf.py update-dependencies` to move to a newer compatible version.

For a specific version from staging or production, use the settings above:

```sh
idf.py add-dependency --registry-url "$ESPECTRE_REGISTRY_URL" "francescopace/espectre=$ESPECTRE_VERSION"
```

This saves the version and registry for the SDK in `main/idf_component.yml`.

### Create the example project

To try sensing on a board, create the complete Wi-Fi motion detection project from the same registry:

```sh
idf.py create-project-from-example --registry-url "$ESPECTRE_REGISTRY_URL" "francescopace/espectre=$ESPECTRE_VERSION:wifi_motion_detection"
cd wifi_motion_detection
```

The example pins the SDK to the same version and registry. If an older staging example lists only a version in `main/idf_component.yml`, add `registry_url: https://components-staging.espressif.com` under `francescopace/espectre`.

Build and run on your target board:

```sh
idf.py set-target esp32c3
idf.py menuconfig
idf.py build
idf.py -p YOUR_PORT flash monitor
```

Set your SSID and password (and optionally a BSSID) under **ESPectre example** in menuconfig. Keep them in your local configuration. The example connects to Wi-Fi and logs motion once calibration is done. See the [example guide](../src/cpp/examples/wifi_motion_detection/README.md) for details.

## Minimal configuration

When adding the SDK to an existing application:

1. Apply the [recommended project configuration](#recommended-project-configuration). Only `CONFIG_ESP_WIFI_CSI_ENABLED=y` is required. The standalone example already sets it in `sdkconfig.defaults`.
2. Configure sensing through `idf.py menuconfig`. Pass `make_runtime_sensing_config_from_kconfig()` to the controller to use those settings. Optional service groups are disabled by default; enable the groups you need under **ESPectre SDK**.
3. Initialize NVS, ESP-NETIF, and the default event loop, then create the station interface and initialize Wi-Fi before runtime setup. Your application must configure and start Wi-Fi; the example shows this with `StandaloneWifiService`. Prefer runtime setup before station start so the runtime can apply its CSI radio policy.
4. Check the result of `setup()`, and call `loop()` regularly from the same task. Run `shutdown()` on that task before releasing application services. Publish movement only when `ready_to_publish` is true; calibration and Wi-Fi recovery can temporarily make sensing unavailable.

The SDK has no default logger. Register a [log sink](#logging) before setup if your application needs SDK messages.

If your application has no Wi-Fi code yet, `StandaloneWifiService` can handle the station for you:

- Initialize NVS first. The service creates the netif and default event loop and starts the driver.
- Call its methods from one task and keep calling `loop()` for connection events and retries.
- Shut down the sensing controller before the Wi-Fi service. `shutdown()` releases the driver, netif, and handlers; you can call `setup()` again afterwards.
- It borrows the `ssid`, `password`, and `bssid` strings until shutdown. SSIDs over 32 bytes or passwords over 64 bytes return `ESP_ERR_INVALID_ARG`.
- `max_retry` (default 8) sets quick reconnect attempts per round. After a failed round it waits 30 seconds and tries again, until connected or shut down.

### Recommended project configuration

The Native, Matter, and ESPHome frontends build with these options. Set them in your `sdkconfig.defaults`:

| Option | Value | Why |
| --- | --- | --- |
| `CONFIG_ESP_WIFI_CSI_ENABLED` | `y` | Required. Enables CSI in the Wi-Fi driver. |
| `CONFIG_ESP_WIFI_AMPDU_RX_ENABLED` | `n` | The radio reports one CSI sample per received transmission, so without aggregation more frames carry CSI. With aggregation, an access point retry resends several frames, and the runtime drops only a repeat of the last one. |
| `CONFIG_ESP_WIFI_AMPDU_TX_ENABLED` | `n` | Required for a fixed [transmit rate](CSI.md#transmit-rate). With TX A-MPDU on, the runtime falls back to automatic rates. |
| `CONFIG_PM_ENABLE` | `n` | Keeps CPU frequency scaling and automatic light sleep off. This is the ESP-IDF default. |
| `CONFIG_ESP_WIFI_STA_DISCONNECTED_PM_ENABLE` | `n` | Keeps Wi-Fi power saving off while the station reconnects. |
| `CONFIG_FREERTOS_HZ` | `1000` | Gives millisecond task timing for traffic pacing. |

Disabling A-MPDU applies to the whole firmware and lowers Wi-Fi throughput, so OTA updates and other network services on the same device may be slower. The Native frontend also raises Wi-Fi buffer counts and lwIP mailbox sizes; see its [sdkconfig.defaults](../src/cpp/frontend/native/app/sdkconfig.defaults).

## Public headers

Include the header for the integration you need. Optional services also need their [capability group](#optional-capability-groups).

| Header | Use it for |
|--------|------------|
| `espectre_sdk.h` | Sensing runtime, configuration, snapshots, and listener callbacks |
| `espectre_core_sdk.h` | Detectors and temporal sampling when your application owns CSI capture |
| `espectre_protocol_sdk.h` | The ESPectre Protocol: messages, JSON, diagnostic fields, and the Direct HTTP and MQTT transport contracts |
| `espectre_services_sdk.h` | Optional Direct HTTP, discovery, provisioning, and application services. Includes the protocol header |
| `espectre_mqtt_sdk.h` | The ESP-IDF MQTT implementation |

The main types are `RuntimeConfig` (startup settings), `RuntimeFrontendController` (lifecycle and controls), `RuntimeSnapshot` (sensing state), `RuntimeCapabilities` (optional controls), and `IRuntimeListener` (events). Only what these headers expose is public; other headers in the package are internal.

## Integration paths

### Full runtime (recommended)

Use `espectre_sdk.h` and `RuntimeFrontendController`. The controller calls your application through `IRuntimeListener`. In SDK names, "frontend" means your integration code.

```cpp
#include "espectre_sdk.h"

class ProductFrontend : public espectre::IRuntimeListener {
 public:
  bool setup() {
    runtime_.set_config(
        espectre::make_runtime_sensing_config_from_kconfig());
    return runtime_.setup(this);
  }

  void loop() { runtime_.loop(); }
  void shutdown() { runtime_.shutdown(); }

  void on_motion_state_changed(const espectre::RuntimeSnapshot &snapshot) override {
    if (!snapshot.ready_to_publish) {
      return;
    }
    publish_motion(snapshot.motion_state == espectre::MotionState::MOTION);
  }

 private:
  void publish_motion(bool motion);  // Queue work for the application.
  espectre::RuntimeFrontendController runtime_;
};
```

Keep the adapter alive until `shutdown()` returns, check the setup result, and make `publish_motion()` a quick, non-blocking handoff. For a complete project, use the registry example.

- Publish motion only when `snapshot.ready_to_publish` is true. Before that, the runtime is still calibrating.
- Override `on_sensing_readiness_changed()` to mark your output unavailable when readiness clears; otherwise the last published value stays in place.
- Call `runtime_.snapshot()` whenever you need the current state; there is no need for your own cache.
- Run `setup()`, `loop()`, and `shutdown()` on one task.
- Check `capabilities()` before offering a control.

Your firmware handles boot, provisioning, networking, OTA, and the product itself. ESPectre handles CSI capture, calibration, and detection. You work with two types:

- `RuntimeFrontendController` (`runtime/esp_idf/runtime_frontend_controller.h`): `setup()`, `loop()`, runtime threshold/detector control, recalibration, and snapshot access. It owns the sensing backend.
- `IRuntimeListener` (`runtime/runtime_events.h`): callbacks for sensing readiness, motion-state changes, periodic updates, threshold/detector changes (including Lightweight settled-level recovery), calibration lifecycle, live telemetry, and runtime faults. Every sensing callback receives a `RuntimeSnapshot`, so gate each one on `ready_to_publish`. The controller emits `on_sensing_readiness_changed()` once per availability transition from its loop, including detector warm-up and input expiry. If you publish a writable threshold control, override `on_threshold_changed()` rather than inferring the live value from telemetry.

After `setup()`, `config()` returns the configuration in use, including the resolved threshold. `shutdown()` returns a default threshold to the sentinel, so the next setup again follows the detector; a value set with `set_threshold()` is kept. Writing to `config()` after setup only affects the next setup; use the runtime setters for live changes.

By default the runtime saves nothing, and your configuration is the only source of truth. Set `persist_runtime_overrides` to true to keep detector, motion-hit, and traffic changes made through the setters across reboots. Saved values then take precedence over your configuration until `clear_persisted_overrides()` erases them. Persistence needs initialized NVS and uses the `espectre` namespace.

Use the [traffic destination](#traffic-destination) setting to select the internal generator's IP destination before setup.

If you use the ESPectre Protocol or CSI streaming, set `RuntimeConfig::device_id` to `derive_runtime_device_id()` before setup. It returns a stable pseudonym derived from the station MAC. The controller does not fill in a zero ID for you.

### Reference integrations

The repository has three complete firmware integrations (frontends) built on the public SDK:

| Frontend | Integration example | Reference |
|----------|---------------------|-----------|
| Native | Standalone ESP-IDF application with Direct HTTP, optional MQTT, and Home Assistant MQTT Discovery | [Native guide](../src/cpp/frontend/native/README.md) |
| ESPHome | ESPHome component that maps YAML configuration and Home Assistant entities to the runtime | [ESPHome guide](../src/cpp/frontend/esphome/README.md) |
| Matter | Matter occupancy sensor with network commissioning and a Direct HTTP bridge for sensing controls | [Matter guide](../src/cpp/frontend/matter/README.md) |

They are not part of the SDK package. For a minimal project, start from the [Wi-Fi motion detection example](../src/cpp/examples/wifi_motion_detection/README.md).

### Logging

The SDK installs no default logger and does not fall back to `stdio`. Register a `LogSink` with both callbacks through `set_log_sink()` before runtime setup, and check its result. An invalid sink leaves the existing registration unchanged. ESPectre copies the sink but does not own its context; keep the context alive until `clear_log_sink()`, and replace or clear the sink only after every runtime and callback source using it has stopped.

The `enabled(context, level, tag)` callback decides whether to format a message. The `write(context, level, tag, line, format, args)` callback consumes its format string and `va_list`; the arguments remain valid only for that call. Both callbacks must be thread-safe, bounded, and non-blocking because calls may come from the runtime owner task, service tasks, or CSI capture paths. Do not call the ESPectre logger recursively from a sink.

The included example connects the sink to ESP-IDF Log v2 through `esp_log_va` and declares the application's `log` dependency. The SDK itself requires no ESP-IDF logging backend. Without a complete sink, it skips message formatting and argument evaluation.

### Optional capability groups

| Menuconfig option | `espectre_sources.cmake` variable | Adds | Additional source-list requirements |
|-------------------|-----------------------------------|------|------------------------------------|
| `ESPECTRE_SDK_ENABLE_FRONTEND_SUPPORT` | `ESPECTRE_RUNTIME_FRONTEND_SUPPORT_SOURCES` | Shared bootstrap, control, sysinfo, and MQTT payload helpers | `ESPECTRE_RUNTIME_ESP_IDF_PROVISIONING_SOURCES` for bootstrap and persisted config |
| `ESPECTRE_SDK_ENABLE_MQTT` | `ESPECTRE_RUNTIME_ESP_IDF_MQTT_SOURCES` | `EspIdfMqttTransport` over `esp-mqtt` | `mqtt` |
| `ESPECTRE_SDK_ENABLE_PROVISIONING` | `ESPECTRE_RUNTIME_ESP_IDF_PROVISIONING_SOURCES` | Device config store and Wi-Fi provisioning | None beyond the base runtime |
| `ESPECTRE_SDK_ENABLE_DIRECT` | `ESPECTRE_RUNTIME_ESP_IDF_DIRECT_SOURCES` | Direct HTTP, SSE, raw CSI streaming, peer discovery, and mDNS | `esp_http_server` and `mdns`, the mDNS `private_include` directory, and `ESPECTRE_RUNTIME_ESP_IDF_DIRECT_LINK_OPTIONS` |

All groups are off by default, and the minimal SDK pulls in no networking stacks. `ESPECTRE_SDK_ENABLE_FRONTEND_SUPPORT` also turns on provisioning.

- **MQTT** uses the bundled MQTT component on ESP-IDF 5.5, and `espressif/mqtt` `1.0.0` (the version ESPHome uses) on ESP-IDF 6.
- **Direct** uses `esp_http_server` and downloads `espressif/mdns` `1.12.0`. That version is pinned because the discovery responder uses private mDNS functions.

Component Manager reads these options through [Kconfig dependency conditions](https://docs.espressif.com/projects/idf-component-manager/en/latest/reference/manifest_file.html#kconfig-options) (ESP-IDF 5.5 and Component Manager 2.2 or newer). For this to work:

- Declare ESPectre as a direct dependency of your project.
- After turning MQTT or Direct on or off, run `idf.py update-dependencies` before building. Otherwise the lockfile may keep dependencies you no longer need.
- If you limit components with `COMPONENTS` or `MINIMAL_BUILD`, add `mqtt` (on ESP-IDF 5) and `esp_http_server` yourself.

The registry lists all optional dependencies even though a minimal build does not use them. Source-list integrations declare their own dependencies. `espectre_mqtt_sdk.h` needs the MQTT component; `espectre_services_sdk.h` does not need the HTTP server headers.

The SDK does not set up a console or USB; your application does.

Other notes:

- For a core-only integration with managed traffic, link `ESPECTRE_CORE_SOURCES` and `ESPECTRE_RUNTIME_ESP_IDF_TRAFFIC_SOURCES`. The traffic group needs `esp_netif`, `esp_timer`, `esp_wifi`, `freertos`, and `lwip`, and links with `ESPECTRE_RUNTIME_ESP_IDF_TRAFFIC_LINK_OPTIONS`. The full runtime already includes it.
- The provisioning service stores credentials your application supplies. The onboarding protocol is up to you; the SDK does not include Improv Serial.
- `IMqttTransport` and `IDirectHttpService` are header-only, so you can implement them without enabling a group.
- `DirectHttpServiceConfig` allows no browser origins by default. `for_first_party_portals()` allows the official ESPectre portals.

## Supported hardware

The targets in [Requirements](#requirements) need no extra hardware: sensing uses the built-in single-antenna Wi-Fi radio at 20 MHz with AGC on. ESP32-C5 supports 2.4 and 5 GHz; the others use 2.4 GHz. ESP32-C6 does not support `TrafficGeneratorMode::WIFI_RAW`.

`wifi_band_policy` defaults to `AUTO` in both `RuntimeConfig` and Kconfig. `AUTO` uses every band the radio has, so it means 2.4 GHz on single-band chips. `BAND_5G` fails setup on chips without 5 GHz.

Set `RuntimeConfig::csi_capture_policy` before setup; it cannot change at runtime. Packets outside the selected profile are dropped and counted. See [capture profiles](CSI.md#capture-profiles) for what each value selects.

## Choosing a detection profile

Choose Lightweight to leave more CPU and memory for the rest of your application. Choose High Accuracy for better detection at a higher cost. Set `detection_algorithm` alone: the default threshold follows the selected detector. To let users switch at runtime, set `runtime_detector_selection_enabled`; the controller then advertises `supports_runtime_detector_selection` and accepts `set_detection_algorithm()`. A build that can switch at runtime includes both detectors in flash. See [why two detection profiles](ALGORITHMS.md#why-two-detection-profiles).

## Shared sensing options

Configure sensing with `RuntimeConfig` before setup. The table shows the defaults of a plain `RuntimeConfig`. To start from menuconfig instead, call `make_runtime_sensing_config_from_kconfig()` and then apply your own overrides. Kconfig choices map directly to C++ values: for example, `CONFIG_ESPECTRE_CSI_CAPTURE_PROFILE_HT_VHT=y` means `CsiCapturePolicy::HT_VHT`.

Defaults and checks are defined in [runtime_sensing_schema.h](../src/cpp/runtime/runtime_sensing_schema.h) and [runtime_config_utils.cpp](../src/cpp/runtime/runtime_config_utils.cpp).

| `RuntimeConfig` field | C++ type / values | Default | Range / notes |
|-----------------------|-------------------|---------|---------------|
| `wifi_band_policy` | `WifiBandPolicy`: `BAND_2G`, `BAND_5G`, or `AUTO` | `AUTO` | `BAND_5G` requires ESP32-C5 among the supported targets |
| `detection_algorithm` | `DetectionAlgorithm`: `LIGHTWEIGHT` or `HIGH_ACCURACY` | `LIGHTWEIGHT` | Lightweight uses less detector CPU and working memory; High Accuracy skips quiet-room threshold calibration |
| `threshold` | `float` | `RUNTIME_THRESHOLD_DETECTOR_DEFAULT` | `0-1`, or the default sentinel, which applies `runtime_default_threshold()` of the selected detector at setup. Lightweight replaces it during calibration, while High Accuracy keeps the configured value. Use `set_threshold()` for session changes when supported |
| `runtime_detector_selection_enabled` | `bool` | `false` | Advertises runtime detector switching and, with persistence, restores the saved detector |
| `window_size_ms` | `uint32_t` | `1000` | `1000-2000` milliseconds; combined with `csi_target_pps` to define a fixed temporal slot window |
| `csi_target_pps` | `uint32_t` | `100` | `1-500`; defines detector slot cadence and the managed-traffic target, but never enables or disables traffic |
| `csi_capture_policy` | `CsiCapturePolicy`: `AUTO`, `LLTF`, or `HT_VHT` | `AUTO` | Set before setup; no runtime setter. `HT_VHT` resolves HT20 or VHT20 from chip and band; `WIFI_RAW` requires `AUTO` or `LLTF` |
| `csi_traffic_udp_port` | `uint16_t` | `5555` | UDP port the listener binds with `EXTERNAL_HOST` traffic |
| `csi_traffic_multicast_group` | `std::string`: IPv4 multicast address, or empty | `"239.255.0.1"` | Joined by the UDP listener with `EXTERNAL_HOST` traffic. Empty disables the join. Unicast to the device IP still works |
| `traffic_generator_mode` | `TrafficGeneratorMode`: `PING`, `DNS`, `DNS_TCP`, `WIFI_RAW`, or `EXTERNAL_HOST` | `PING` | `DNS` uses UDP, `DNS_TCP` uses persistent TCP, and experimental `WIFI_RAW` sends Null Data to the AP. `EXTERNAL_HOST` sends nothing and listens for UDP markers and ICMP Echo Requests from another host, independently from `csi_target_pps` |
| `traffic_generator_target_ip` | `std::string`: unicast IPv4 address, or empty | empty | Destination for internal `PING`, `DNS`, and `DNS_TCP`; empty uses the Wi-Fi default gateway. Ignored by `WIFI_RAW` and external traffic |
| `evaluation_interval_ms` | `uint32_t` | `250` | `10-10000` milliseconds between detector evaluations |
| `motion_on_hits` | `uint8_t` | `4` | `1-20` consecutive evaluation hits for `IDLE -> MOTION` |
| `motion_off_hits` | `uint8_t` | `3` | `1-20` consecutive evaluation hits for `MOTION -> IDLE` |
| `lowpass_enabled` | `bool` | `false` | Enables low-pass filtering |
| `lowpass_cutoff` | `float` | `11.0` | `5.0-20.0` Hz against a nominal regular `100 pps` cadence; other targets or substantial missing-slot patterns require filter revalidation |
| `hampel_enabled` | `bool` | `true` | Enables Hampel outlier filtering |
| `hampel_window` | `uint8_t` | `7` | `3-11` samples |
| `hampel_threshold` | `float` | `5.0` | `1.0-10.0` MAD units |
| `persist_runtime_overrides` | `bool` | `false` | `true` restores and saves the traffic generator mode, motion hits, and selectable detector across reboots in NVS |
| `wifi_scan_results_managed_externally` | `bool` | `false` | Set it when your Wi-Fi stack scans on its own and consumes every scan result; the runtime then leaves the driver result list to it |
| `device_id` | `uint64_t` | `0` | Identity for the ESPectre Protocol and CSI streaming; assign `derive_runtime_device_id()`. Zero means unresolved |

The transmit rate is a build-time Kconfig string, not a `RuntimeConfig` field: `CONFIG_ESPECTRE_WIFI_TX_RATE_MBPS="0"` (automatic), `"6"`, or `"6.5"`. Keep the quotes. See [transmit rate](CSI.md#transmit-rate) for defaults and requirements.

Only some settings can change at runtime. Check the advertised capabilities, then use the runtime setters or the [sensing operations](API.md#sensing-update-and-calibration). See [tuning essentials](TROUBLESHOOTING.md#tuning-essentials) for when to change a setting.

### Traffic destination

`traffic_generator_target_ip` sets where `ping`, `dns`, and `dns_tcp` send their packets. Leave it empty to use the Wi-Fi gateway. It is ignored by `wifi_raw` and external traffic, and can be set only before setup (or with `CONFIG_ESPECTRE_TRAFFIC_GENERATOR_TARGET_IP` in menuconfig).

Use a plain unicast IPv4 address, such as `192.168.1.10`, of a host that answers the selected protocol. DNS modes need a resolver on port `53`. Hostnames, loopback, multicast, and reserved addresses are rejected.

Motion-hit timing is explained in [motion-hit filtering](ALGORITHMS.md#motion-hit-filtering).

## Runtime contract

The [integration reference](../src/cpp/sdk_integration.dox) owns the complete runtime contract. These rules cover what every integration needs.

### Lifecycle

Call `set_config()`, `setup(listener)`, `loop()` repeatedly, and then `shutdown()`. Create the Wi-Fi station interface and default event loop before setup, preferably before station start; setup after association is also supported. Keep the listener valid until `shutdown()` returns, or until the controller is destroyed when you rely on its destructor.

Check every `bool` result. `set_config()` returns `false` while the runtime is set up, so call `shutdown()` first. A failed setup leaves the controller available for retry, and `shutdown()` keeps the configuration for the next setup. Publish sensing results only while `RuntimeSnapshot::ready_to_publish` is true: Wi-Fi recovery, calibration, and detector warm-up can clear it.

If your firmware owns the Wi-Fi station, check `traffic_allows_radio_work()` before reconfiguring it and `wifi_scan_allowed()` before starting a scan, so your radio work does not race the runtime's traffic stops or recovery scan. The integration reference describes the full sequence.

### Threading

Use one owner task for lifecycle and control calls. Listener callbacks run on that task, from `loop()` or inline in a control call, and never in Wi-Fi capture context. Keep them bounded and non-blocking, and queue network or storage work for another task: slow callbacks delay `loop()` and can overflow the CSI mailbox. Queue commands received by network callbacks and apply them from the owner task.

Raw CSI packet callbacks are the exception: they run synchronously in Wi-Fi capture context and must not block or allocate. Copy accepted samples into a preallocated queue for later processing.

### Errors and capabilities

Check `controller.capabilities()` after setup before exposing optional controls. Control methods return `false` when a value is out of range, the capability is unavailable, or the backend refuses the request, and a rejected control leaves the runtime unchanged. Allocation failures make `setup()` return `false`. Later faults arrive through `IRuntimeListener::on_runtime_fault()`. A failed calibration is not fatal: sensing continues with the threshold in force.

### Diagnostics

Use `controller.diagnostics_sample()` for the shared one-second sample of traffic and CSI rates plus the current link. `controller.diagnostics()` returns the cumulative counters behind it, for totals or a custom interval. [API diagnostics](API.md#diagnostics) defines the wire fields, and the integration reference explains how to read them.

### Versioning

The public facades provide source compatibility under Semantic Versioning: patch releases preserve documented behavior, minor releases add compatible APIs, and major releases may break compatibility. Prereleases may change before the final release. Rebuild the SDK with your firmware; the source package does not promise binary ABI compatibility.

Construct public configuration and snapshot structs with their defaults, then assign named fields, because minor releases may append fields. `CsiCaptureProfile` and `CsiCapturePolicy` are open enums, so handle values you do not know. The core-only detector interface (`espectre_core_sdk.h`) may change in a minor release until the stationary presence detector ships. Detector coefficients may change in compatible fixes, so exact floating-point telemetry is not guaranteed across releases.

Use `ESPECTRE_SDK_VERSION_STRING` to identify the SDK and `ESPECTRE_SDK_VERSION_AT_LEAST(major, minor, patch)` for compile-time feature guards. Your application version, supplied through `EspectreDeviceInfo::firmware_version`, and `ESPECTRE_PROTOCOL_VERSION` are separate. To override the packaged identity, define all four `ESPECTRE_SDK_VERSION_*` macros consistently; the integration reference has the rules.

## Advanced integrations

### Core-only

If your application owns CSI capture, include `espectre_core_sdk.h` and use the detectors with `TemporalCsiSampler`. Check `detector.is_valid()` after construction and the result of `sampler.configure(...)`: both allocate bounded working buffers without throwing. Your application then drives the sampler, evaluation cadence, and motion-hit filtering, and re-reads `get_threshold()` after each `update_state()`, because Lightweight can lower it without a callback.

Keep raw samples separate from detector input. Normalize each payload into the centered HT20 convention and, for LLTF captures, zero the missing bins with `zero_ht20_lltf_missing_bins()` on the raw view. Then copy it into a private detector buffer and call `prepare_ht20_detector_input()` on that copy with the capture profile and source metadata. The [integration reference](../src/cpp/sdk_integration.dox) gives the sampler sequence, and [csi_pipeline.cpp](../src/cpp/runtime/esp_idf/csi_pipeline.cpp) shows the production pipeline.

### Build integration

Component Manager configures the registered component automatically. For vendoring or custom CMake targets, use the directory containing `espectre_sources.cmake` as the SDK root: the registry package root, or `src/cpp/` in a source bundle.

- For a vendored ESP-IDF component, copy the SDK root to `components/espectre/` and add `espectre` to your application's `REQUIRES`. Enable optional groups under **ESPectre SDK** in menuconfig.
- For a core-only target, compile `ESPECTRE_CORE_SOURCES` and add `ESPECTRE_SHARED_INCLUDE_DIRS`.
- For a full-runtime source-list target, also compile `ESPECTRE_RUNTIME_ESP_IDF_SOURCES`, plus the optional source groups and stack dependencies your application needs from [Optional capability groups](#optional-capability-groups).

For a source bundle extracted into `espectre/`, a core-only target is:

```cmake
set(ESPECTRE_CPP_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/espectre/src/cpp")
include("${ESPECTRE_CPP_ROOT}/espectre_sources.cmake")
add_library(espectre_core STATIC ${ESPECTRE_CORE_SOURCES})
target_compile_features(espectre_core PUBLIC cxx_std_17)
target_include_directories(espectre_core PUBLIC ${ESPECTRE_SHARED_INCLUDE_DIRS})
```

Link your application target to `espectre_core` to inherit the includes and C++ standard. If overriding SDK identity, apply all four version macros through `target_compile_definitions(espectre_core PUBLIC ...)`. The SDK root is the only include directory: include other SDK headers by their layer-prefixed path, such as `runtime/runtime_config.h`.

The full runtime links with `ESPECTRE_RUNTIME_ESP_IDF_TRAFFIC_LINK_OPTIONS`. These options wrap `esp_netif_new`, `esp_netif_destroy`, `esp_netif_receive`, and `esp_netif_transmit_wrap` for the whole application to count station traffic, so your application cannot wrap the same functions. The counters follow the default `WIFI_STA_DEF` station interface. The Direct group also wraps the private mDNS function `mdns_priv_receive_action`. Component Manager applies these options automatically.

### Transport and protocol extensions

Pass parsed SDK requests through `FrontendCommandEngine`, and describe application routes with an `EspectreProtocolExtension` checked by `validate_protocol_extension()`. Use the same catalog for capability output, Direct HTTP, and command parsing. [Contract principles](API.md#contract-principles) define the shared message contract. The integration reference covers validators, handlers, and `RuntimeFrontendController::quiesce()` for application-owned firmware updates.

### Task priorities and raw CSI

Your application owns the task that calls `RuntimeFrontendController::loop()`. The SDK's **Advanced task scheduling** menu separately configures its service tasks, using priorities from `1` to `10`:

| Kconfig option | Default | Service |
|----------------|---------|---------|
| `CONFIG_ESPECTRE_DIRECT_HTTPD_TASK_PRIORITY` | `1` | Direct HTTP server |
| `CONFIG_ESPECTRE_DIRECT_WORKER_TASK_PRIORITY` | `2` | Direct responses and SSE delivery |
| `CONFIG_ESPECTRE_RAW_WORKER_TASK_PRIORITY` | `3` | Raw CSI HTTP delivery |
| `CONFIG_ESPECTRE_TRAFFIC_TASK_PRIORITY` | `1` | Managed PING or DNS traffic |

These are compile-time settings. Validate priority changes under the application's workload: higher-priority tasks preempt lower-priority work and can starve sensing or networking. ESP-IDF owns the internal Wi-Fi and lwIP priorities.

Raw CSI views carry the normalized `HT20_CSI_LEN` payload of 128 bytes, so size capture queues for it. The integration reference documents the Direct raw queue and its drop counters.

## Source bundles

For source vendoring, download SDK `.tar.gz` or `.zip` assets from [GitHub Releases](https://github.com/francescopace/espectre/releases). Tagged releases use `espectre-sdk-<version>` filenames. The rolling [snapshot](https://github.com/francescopace/espectre/releases/tag/snapshot) release carries `espectre-sdk-preview` from `main`; [snapshot-dev](https://github.com/francescopace/espectre/releases/tag/snapshot-dev) carries `espectre-sdk-develop` from `develop`. Use rolling and prerelease builds for evaluation. The accompanying `sdk-manifest-<version>.json`, `sdk-manifest-preview.json`, or `sdk-manifest-develop.json` records the source identity and archive SHA-256 digests.

Source bundles place the component under `src/cpp/`; registry packages place it at the package root and include an example project. Both distributions compile with your application and contain no chip-specific precompiled libraries.

Registry packages include a generated `API.md` with the complete C++ reference and integration contracts for their version and source commit. Source bundles include the public header comments, `src/cpp/sdk_integration.dox`, and a stamped `src/cpp/Doxyfile`. To regenerate the reference XML locally, install Doxygen 1.17.0 and run `doxygen src/cpp/Doxyfile` from the extracted bundle root; the XML is written under `output/xml/`.

## Licensing

The public SDK is `GPL-3.0-only`, which permits commercial use subject to GPLv3's terms and applicable corresponding-source obligations. A separate commercial license is available for proprietary firmware. See the [licensing terms](../LICENSING.md).
