# ESPectre SDK <img src="https://espectre.dev/assets/images/brand/espectre-logo.svg" alt="ESPectre logo" width="40" align="absmiddle" />

[Home](https://espectre.dev/) · [Tools](https://espectre.dev/tools/) · [Guides](https://espectre.dev/guides/) · [SDK](https://espectre.dev/sdk/) · [Roadmap](https://espectre.dev/roadmap/) · [Media](https://espectre.dev/media/) · [GitHub](https://github.com/francescopace/espectre) · [Contacts](https://espectre.dev/contact/) · [Commercial License](https://espectre.dev/licensing/)

ESPectre adds motion detection to ESP-IDF firmware using Wi-Fi Channel State Information (CSI). It provides detectors, calibration, and motion events through a C++ API. The runtime owns CSI capture and sensing; your application owns networking, task scheduling, and how it uses the results. Optional services support MQTT, Direct HTTP, discovery, and provisioning.

## Requirements

| Requirement | Supported configuration |
|-------------|-------------------------|
| ESP-IDF | `>=5.5.5,<5.6.0`; release builds use 5.5.5 |
| C++ | C++17 |
| Target | ESP32, ESP32-S2, ESP32-S3, ESP32-C3, ESP32-C5, or ESP32-C6 |
| License | `GPL-3.0-only`, with a separate commercial agreement available; see [Licensing](#licensing) |

## ESP Component Registry

Install `francescopace/espectre` through ESP-IDF Component Manager. It downloads the SDK sources and builds them with your application; no manual archive download is required.

| Registry | Available versions | Website channel |
|----------|--------------------|-----------------|
| [Production](https://components.espressif.com/components/francescopace/espectre) | Stable releases only | Release, when the tagged version is stable |
| [Staging](https://components-staging.espressif.com/components/francescopace/espectre) | Tagged prereleases and branch snapshots | Release candidates, Preview (`main`), and Develop (`develop`) |

If the production component page is unavailable, the first stable SDK release has not been published there yet. Use staging to evaluate the SDK. A website **Release** download can be a release candidate; it does not imply production-registry availability.

For a specific version or the example project, select an exact version from the registry page. In an activated ESP-IDF environment, replace `VERSION_FROM_REGISTRY` below with that version. Use `https://components-staging.espressif.com` for snapshots and prereleases, or `https://components.espressif.com` for stable releases.

```sh
ESPECTRE_VERSION="VERSION_FROM_REGISTRY"
ESPECTRE_REGISTRY_URL="https://components-staging.espressif.com"
```

Registry packages fill in these settings with their own version and registry. Snapshot versions end in `.main` or `.develop` and differ from the website archive labels. Older snapshots are periodically removed from staging; use a retained version when resolving a new project.

### Add to an existing project

From your ESP-IDF project directory, add the latest compatible stable release from the default production registry:

```sh
idf.py add-dependency "francescopace/espectre"
```

Component Manager records the resolved version in `dependencies.lock` when configuring the project. Subsequent builds reuse that version; run `idf.py update-dependencies` to resolve newer compatible versions.

For a specific version from staging or production, use the settings above:

```sh
idf.py add-dependency --registry-url "$ESPECTRE_REGISTRY_URL" "francescopace/espectre=$ESPECTRE_VERSION"
```

This records the version and registry on the SDK dependency in `main/idf_component.yml`. Other dependencies keep their own registry settings.

### Create the example project

To try sensing on a board, create the complete Wi-Fi motion detection project from the same registry:

```sh
idf.py create-project-from-example --registry-url "$ESPECTRE_REGISTRY_URL" "francescopace/espectre=$ESPECTRE_VERSION:wifi_motion_detection"
cd wifi_motion_detection
```

The packaged example pins its SDK dependency to the same version and registry. For an older staging example whose `main/idf_component.yml` has only a version, add `registry_url: https://components-staging.espressif.com` under `francescopace/espectre` before configuring or building it.

Build and run on your target board:

```sh
idf.py set-target esp32c3
idf.py menuconfig
idf.py build
idf.py -p YOUR_PORT flash monitor
```

Set your SSID and password under **ESPectre example** in menuconfig; you can also pin an access point with its BSSID. The example enables CSI, connects to Wi-Fi, and logs motion after calibration is ready. Its [README.md](../src/cpp/examples/wifi_motion_detection/README.md) covers configuration, lifecycle, and hardware checks. Keep credentials in the local project configuration.

## Minimal configuration

When adding the SDK to an existing application:

1. Enable `CONFIG_ESP_WIFI_CSI_ENABLED=y` in the project configuration. The standalone example already sets this in `sdkconfig.defaults`.
2. Configure sensing through `idf.py menuconfig`. Pass `make_runtime_sensing_config_from_kconfig()` to the controller to use those settings. Optional service groups are disabled by default; enable the groups you need under **ESPectre SDK**.
3. Initialize NVS, ESP-NETIF, and the default event loop, then create the station interface and initialize Wi-Fi before runtime setup. Your application must configure and start Wi-Fi; the example shows this with `StandaloneWifiService`. Prefer runtime setup before station start so the runtime can apply its CSI radio policy.
4. Check the result of `setup()`, and call `loop()` regularly from the same task. Run `shutdown()` on that task before releasing application services. Publish movement only when `ready_to_publish` is true; calibration and Wi-Fi recovery can temporarily make sensing unavailable.

The SDK has no default logger. Register a [log sink](#logging) before setup if your application needs SDK messages.

`StandaloneWifiService` can own station setup for an application without an existing Wi-Fi driver or station interface. Initialize NVS first; the service creates the netif and default event loop, and initializes the driver. Its `shutdown()` releases the driver, netif, and handlers, and allows another `setup()`; failed setup also releases acquired resources. The default event loop remains available. Shut down the sensing controller before the Wi-Fi service. Call service methods from one task, and keep calling `loop()` to process connection events and retries.

The service borrows the null-terminated `ssid`, `password`, and `bssid` strings until shutdown or replacement. SSIDs up to 32 bytes and passwords up to 64 bytes are preserved; longer values return `ESP_ERR_INVALID_ARG`. `max_retry` sets the number of immediate reconnect attempts per burst, with a default of eight. After an exhausted burst, the service waits 30 seconds before starting another; it continues until connected or shut down. Zero disables immediate retries but keeps the delayed attempts.

## Public headers

Include the facade for the integration you need. Component Manager builds the sensing runtime automatically; optional services require the groups described below.

| Header | Use it for |
|--------|------------|
| `espectre_sdk.h` | Sensing runtime, configuration, snapshots, and listener callbacks |
| `espectre_core_sdk.h` | Detectors and temporal sampling when your application owns CSI capture |
| `espectre_services_sdk.h` | Optional Direct HTTP, discovery, provisioning, and application services |
| `espectre_mqtt_sdk.h` | The ESP-IDF MQTT implementation |

Use `RuntimeConfig` for startup settings, `RuntimeFrontendController` for lifecycle and controls, `RuntimeSnapshot` for sensing state, and `RuntimeCapabilities` to discover optional controls. Implement `IRuntimeListener` to receive events. The facade headers expose these types and their documented public methods. Headers included only as implementation dependencies are internal; their presence in the package does not make them extension points.

## Integration paths

### Full runtime (recommended)

Use `espectre_sdk.h` and `RuntimeFrontendController` to integrate sensing with an existing application. The controller connects the runtime to your application through `IRuntimeListener` callbacks. In SDK names, "frontend" refers to this application integration code.

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

This adapter assumes the Wi-Fi lifecycle described above. Keep the adapter alive while the runtime is active, check its setup result, and implement `publish_motion()` as a bounded, non-blocking handoff to your application. For a complete project, use the registry example.

The runtime reports state through `RuntimeSnapshot` and sends events to `IRuntimeListener`. A snapshot is one immutable view of runtime state; capabilities report which optional controls the backend supports. Follow these rules when adding application behavior:

- Gate sensing output on `snapshot.ready_to_publish`. The runtime emits snapshots while it calibrates, and motion state is not meaningful before that flag is true.
- Read `runtime_.snapshot()` for on-demand state. The controller refreshes it before forwarding each listener callback, so your application can use it without maintaining a second cache.
- Run `setup()`, `loop()`, and `shutdown()` on one task.
- Ask `capabilities()` before exposing a control, rather than assuming the active runtime supports it.

Your firmware owns boot, provisioning, networking policy, OTA, and the product surface; the ESPectre runtime owns CSI capture, calibration, detection, and eventing behind two contracts:

- `IEspectreRuntime` (`runtime/runtime_interface.h`): `setup()`, `loop()`, runtime threshold/detector control, recalibration, and snapshot access.
- `IRuntimeListener` (`runtime/runtime_events.h`): callbacks for sensing readiness, motion-state changes, periodic updates, threshold/detector changes (including Lightweight settled-level recovery), calibration lifecycle, live telemetry, and runtime faults. The controller emits `on_sensing_readiness_changed()` once per availability transition from its loop, including detector warm-up and input expiry. If you publish a writable threshold control, override `on_threshold_changed()` rather than inferring the live value from telemetry.

`RuntimeFrontendController` wires configuration, runtime-control persistence, and the runtime backend together. After `setup()`, `config()` reflects the backend's effective configuration, including persisted detector, motion-hit, and traffic overrides; direct writes to `config()` after setup only stage the next setup, while live changes use the capability-gated runtime setters.

Use the [traffic destination](#traffic-destination) setting to select the internal generator's IP destination before setup.

Set `RuntimeConfig::device_id` to `derive_runtime_device_id()` before setup when the integration uses the ESPectre Protocol or CSI streaming. The helper returns a cached pseudonym derived from the station MAC; zero remains an unresolved sentinel and is not replaced by `RuntimeFrontendController`.

### Reference integrations

The ESPectre repository includes three C++ firmware integrations, called frontends, that use the public SDK. Their source code shows how to connect sensing to an application framework or protocol:

| Frontend | Integration example | Reference |
|----------|---------------------|-----------|
| Native | Standalone ESP-IDF application with Direct HTTP, optional MQTT, and Home Assistant MQTT Discovery | [README.md](../src/cpp/frontend/native/README.md) |
| ESPHome | ESPHome component that maps YAML configuration and Home Assistant entities to the runtime | [README.md](../src/cpp/frontend/esphome/README.md) |
| Matter | Matter occupancy sensor with network commissioning and a Direct HTTP bridge for sensing controls | [README.md](../src/cpp/frontend/matter/README.md) |

These applications live outside the SDK package and have their own build instructions. For a minimal ESP-IDF project, use the included Wi-Fi motion detection example and its [README.md](../src/cpp/examples/wifi_motion_detection/README.md).

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
| `ESPECTRE_SDK_ENABLE_DIRECT` | `ESPECTRE_RUNTIME_ESP_IDF_DIRECT_SOURCES` | Direct HTTP, SSE, raw CSI streaming, peer discovery, and mDNS | `esp_http_server` and `mdns` |

Each group is off by default. The minimal SDK uses only components bundled with ESP-IDF and downloads no additional stack dependencies. `ESPECTRE_SDK_ENABLE_FRONTEND_SUPPORT` also selects provisioning because its bootstrap helpers call that service.

The manifest uses [Kconfig dependency conditions](https://docs.espressif.com/projects/idf-component-manager/en/latest/reference/manifest_file.html#kconfig-options), supported by ESP-IDF 5.5.5 and Component Manager 2.2 or newer. Declare ESPectre as a direct project dependency so Component Manager can read its Kconfig options. Enabling `ESPECTRE_SDK_ENABLE_DIRECT` downloads `espressif/mdns`, pinned to `1.12.0` because the bootstrap responder uses its private API. Direct is off by default. External components retain their own licenses. Source-list integrations declare their selected stack dependencies themselves.

After changing Direct in an existing project through menuconfig, run `idf.py update-dependencies` before `idf.py build`. This refreshes the lockfile for the selected configuration; a normal reconfigure can retain previously resolved optional dependencies after their switches are disabled.

Your application owns console and USB configuration, including any TinyUSB dependency. The SDK does not initialize a console. Integrations migrating from the prerelease SDK must replace calls to `initialize_primary_console()` with their own console setup and declare any required USB dependencies in the application.

For a core-only integration with managed traffic, link `ESPECTRE_CORE_SOURCES` and `ESPECTRE_RUNTIME_ESP_IDF_TRAFFIC_SOURCES`. The traffic group requires ESP-IDF's `esp_netif`, `esp_timer`, `esp_wifi`, `freertos`, and `lwip` components. The full ESP-IDF runtime already includes this group; add it separately only when using the focused integration.

The provisioning service accepts credentials supplied by your application. Choose and implement the onboarding protocol in your firmware; the SDK does not include Improv Serial.

You can implement `IMqttTransport` or `IDirectHttpService` without enabling a group because the interfaces are header-only. `DirectHttpServiceConfig` keeps its generic Origin allowlist empty; `for_first_party_portals()` explicitly selects the official production and validation portals.

## Supported hardware

The targets in [Requirements](#requirements) use single-antenna Wi-Fi CSI with AGC active and 20 MHz bandwidth. No extra radio hardware is required. ESP32-C5 supports both 2.4 GHz and 5 GHz; the other listed targets use 2.4 GHz. ESP32-C6 rejects `RuntimeTrafficMode::WIFI_RAW`.

A directly constructed `RuntimeConfig` defaults to `WifiBandPolicy::BAND_2G`. The Kconfig default is `AUTO` on ESP32-C5 and `BAND_2G` on the other targets. The runtime applies the selected band policy and 20 MHz bandwidth; unsupported policies fail setup.

Set `RuntimeConfig::csi_capture_profile` before setup. `CsiCapturePolicy::AUTO` selects LLTF20 for internal `WIFI_RAW`, VHT20 on a supported 5 GHz link, and HT20 otherwise. `LLTF` always selects LLTF20; `HT_VHT` selects VHT20 on a supported 5 GHz link and HT20 otherwise. `WIFI_RAW` requires `AUTO` or `LLTF`. Packets outside the selected profile are dropped and counted. There is no runtime profile setter or persisted profile override; the protocol's read-only `csi_profile` field reports the effective profile. [CSI.md](CSI.md#capture-profiles) describes the sample layouts and normalization.

## Choosing a detection profile

Choose Lightweight when sensing must leave more CPU time and working memory for the rest of the application. High Accuracy uses additional feature state and neural inference to improve detection quality. A build with runtime switching may contain both detectors and ML weights even while Lightweight is active. See [ALGORITHMS.md](ALGORITHMS.md#why-two-detection-profiles) for detector behavior and resource tradeoffs.

## Shared sensing options

Configure sensing with `RuntimeConfig` before setup. The defaults and validators are defined in [runtime_sensing_schema.h](../src/cpp/runtime/runtime_sensing_schema.h) and [runtime_config_utils.cpp](../src/cpp/runtime/runtime_config_utils.cpp). `runtime_traffic_mode_supported()` also checks the target: ESP32-C6 rejects `RuntimeTrafficMode::WIFI_RAW`, including runtime changes and persisted selections. See [CSI.md](CSI.md#compatibility-limits) for radio limits.

The table lists C++ fields and defaults for a directly constructed `RuntimeConfig`. To use menuconfig settings, start with `make_runtime_sensing_config_from_kconfig()`, then assign any application overrides before passing the configuration to the controller. That helper uses the Kconfig band policy described above and selects the initial threshold with `runtime_default_threshold(config.detection_algorithm)`.

Kconfig uses choice symbols such as `CONFIG_ESPECTRE_CSI_CAPTURE_PROFILE_HT_VHT=y`; the equivalent C++ assignment uses `CsiCapturePolicy::HT_VHT` for `RuntimeConfig::csi_capture_profile`.

| `RuntimeConfig` field | C++ type / values | Default | Range / notes |
|-----------------------|-------------------|---------|---------------|
| `wifi_band_policy` | `WifiBandPolicy`: `BAND_2G`, `BAND_5G`, or `AUTO` | `BAND_2G` | `BAND_5G` and `AUTO` require ESP32-C5 among the supported targets |
| `detection_algorithm` | `DetectionAlgorithm`: `LIGHTWEIGHT` or `HIGH_ACCURACY` | `LIGHTWEIGHT` | Lightweight uses less detector CPU and working memory; High Accuracy skips quiet-room threshold calibration |
| `segmentation_threshold` | `float` | `RUNTIME_SEGMENTATION_THRESHOLD_DEFAULT` | `0-1`; Lightweight replaces it during calibration, while High Accuracy keeps the configured value. Use `set_threshold_runtime()` for session changes when supported |
| `segmentation_window_size_ms` | `uint32_t` | `1000` | `1000-2000` milliseconds; combined with `csi_target_pps` to define a fixed temporal slot window |
| `csi_target_pps` | `uint32_t` | `100` | `1-500`; defines detector slot cadence and the managed-traffic target, but never enables or disables traffic |
| `csi_capture_profile` | `CsiCapturePolicy`: `AUTO`, `LLTF`, or `HT_VHT` | `AUTO` | Set before setup; no runtime setter. `HT_VHT` resolves HT20 or VHT20 from chip and band; `WIFI_RAW` requires `AUTO` or `LLTF` |
| `csi_traffic_mode` | `CsiTrafficMode`: `INTERNAL` or `EXTERNAL` | `INTERNAL` | Selects device-generated traffic or externally supplied UDP markers and ICMP Echo Requests independently from `csi_target_pps` |
| `csi_traffic_multicast_group` | `std::string`: IPv4 multicast address, or empty | `"239.255.0.1"` | Joined by the UDP listener in `EXTERNAL`. Empty disables the join. Unicast to the device IP still works |
| `traffic_generator_mode` | `RuntimeTrafficMode`: `PING`, `DNS`, `DNS_TCP`, or `WIFI_RAW` | `PING` | `DNS` uses UDP, `DNS_TCP` uses persistent TCP, and experimental `WIFI_RAW` sends Null Data to the AP |
| `traffic_generator_target_ip` | `std::string`: unicast IPv4 address, or empty | empty | Destination for internal `PING`, `DNS`, and `DNS_TCP`; empty uses the Wi-Fi default gateway. Ignored by `WIFI_RAW` and external traffic |
| `evaluation_interval_ms` | `uint32_t` | `250` | `10-10000` milliseconds between detector evaluations |
| `motion_on_hits` | `uint8_t` | `4` | `1-20` consecutive evaluation hits for `IDLE -> MOTION` |
| `motion_off_hits` | `uint8_t` | `3` | `1-20` consecutive evaluation hits for `MOTION -> IDLE` |
| `lowpass_enabled` | `bool` | `false` | Enables low-pass filtering |
| `lowpass_cutoff` | `float` | `11.0` | `5.0-20.0` Hz against a nominal regular `100 pps` cadence; other targets or substantial missing-slot patterns require filter revalidation |
| `hampel_enabled` | `bool` | `true` | Enables Hampel outlier filtering |
| `hampel_window` | `uint8_t` | `7` | `3-11` samples |
| `hampel_threshold` | `float` | `5.0` | `1.0-10.0` MAD units |

The shared string setting `CONFIG_ESPECTRE_WIFI_TX_RATE_MBPS` accepts `"0"` for Auto and `"6"` or `"6.5"` for a fixed rate in Mbps. It defaults to `"6.5"` on classic ESP32 and `"0"` on every other supported target, including builds without this Kconfig symbol. The 6.5-Mbps choice selects HT20 MCS0 with long GI; the 6-Mbps choice selects legacy OFDM. Firmware compilation rejects any other value. Use quoted values in sdkconfig, for example `CONFIG_ESPECTRE_WIFI_TX_RATE_MBPS="6.5"`; the string type allows the fractional HT rate. When migrating an existing integer override, update it to the quoted form or remove it to adopt the new target default. The shared Wi-Fi lifecycle owns station-rate configuration across connections, independently of sensing or traffic generator state. This is a build-time radio policy, not a `RuntimeConfig` field or runtime-writable setting. [CSI.md](CSI.md#internal-generators) describes raw injection, station-rate scope, and compatibility limits.

When migrating from earlier v3 snapshots, assign the former `traffic_generator_rate` value to `RuntimeConfig::csi_target_pps` and set `csi_traffic_mode` to `CsiTrafficMode::INTERNAL`. Persisted `pacing` and `disabled` values migrate once to `internal`. [API.md](API.md#sensing-update-and-calibration) defines the string values used by protocol requests.

Runtime-writable controls are a subset of startup configuration. Inspect the advertised capabilities and use the corresponding runtime setters or the operations in [API.md](API.md#sensing-update-and-calibration). [TROUBLESHOOTING.md](TROUBLESHOOTING.md#tuning-essentials) explains when to adjust a setting; [CSI.md](CSI.md) describes traffic and capture behavior.

### Traffic destination

Set `RuntimeConfig::traffic_generator_target_ip` before setup to override the destination for internal `ping`, `dns`, and `dns_tcp`. Empty uses the current Wi-Fi gateway. `wifi_raw` and external traffic ignore this setting. It has no runtime API mutation.

The value must be dotted-decimal unicast IPv4 without leading zeros. Hostnames, loopback, unspecified, multicast, and reserved addresses are rejected. Choose a reachable host that replies to the selected protocol; DNS modes require a resolver on port `53`, with TCP query support for `dns_tcp`. The runtime applies the same resolved address to traffic generation and CSI response filtering after each connection. Wi-Fi configuration continues to own association and the default route.

For menuconfig configuration, set `CONFIG_ESPECTRE_TRAFFIC_GENERATOR_TARGET_IP` and use `make_runtime_sensing_config_from_kconfig()` to load it.

The hit-filter timing model is described in [ALGORITHMS.md](ALGORITHMS.md#motion-hit-filtering).

## Runtime contract

### Lifecycle

Call `set_config()`, `setup(listener)`, `loop()` repeatedly, and then `shutdown()`. Keep the listener alive until shutdown completes. Create the Wi-Fi station interface and default event loop before setup, preferably before station start. Setup after association is also supported.

Check every `bool` result. A failed setup leaves the controller available for retry; `shutdown()` retains its configuration for the next setup. Only publish sensing results while `RuntimeSnapshot::ready_to_publish` is true. Wi-Fi recovery, calibration, and detector warm-up can temporarily clear that flag.

### Threading

Use one owner task for lifecycle and control calls. Listener callbacks run from that task's `loop()` or inline in a control call. Keep them bounded and non-blocking, and queue network or storage work for another task. Internal mailboxes do not make the control API thread-safe.

Queue commands received by network callbacks and apply them from the owner task. Listener callbacks never run in Wi-Fi capture context: the runtime defers capture events through a bounded mailbox. Slow callbacks delay the next `loop()` call and can cause that mailbox to overflow.

Raw CSI packet callbacks are the exception: they run synchronously in Wi-Fi capture context and must also avoid allocation. Copy accepted samples into a preallocated queue for later processing. Returning `false` reports a consumer drop or backpressure; it does not stop collection. Do not call runtime controls or stop collection from a raw callback. A successful `stop_raw_collection()` waits for any in-progress packet callback before releasing its context, so the owner can then reclaim that context.

### Errors and capabilities

Check `controller.capabilities()` after setup before exposing optional controls. Capability flags default to `false`, and the controller rejects controls the active backend does not advertise. Control methods return `false` when a value is outside the supported range, the capability is unavailable, or the backend refuses the request; a rejected control leaves the runtime unchanged.

The control surface reports failure through return values and listener callbacks. Runtime-backend, sampler, and detector storage allocations are non-throwing: allocation failure makes `setup()` return `false` and reports the fault synchronously to the listener. Fix the configuration or resource shortage before retrying setup.

Asynchronous faults arrive through `IRuntimeListener::on_runtime_fault()`. `on_calibration_finished(snapshot, success)` reports calibration outcome separately. A failed calibration is not fatal: sensing continues with the configured threshold.

### Diagnostics

Use `controller.diagnostics()` for cumulative capture and link counters, and `controller.diagnostics_sample()` for the shared one-second sample. Read the same sample across transport adapters so their observation windows agree. For a custom interval, initialize `RuntimeDiagnosticsSampler` with `reset(totals, now_ms)`, then call `sample(totals, now_ms)` from an existing periodic callback.

`csi_accepted_pps` measures identity-accepted input; `csi_admitted_pps` measures the input retained for the detector after temporal admission. Compare admitted PPS with `RuntimeConfig::csi_target_pps` together with `csi_occupancy_ratio`, callback-queue overflow, same-slot excess, missing-slot, stale, and out-of-order rates. The cumulative snapshot also exposes the callback queue's occupancy and capacity. These measurements do not change the device's traffic rate.

The ESP-IDF runtime always collects diagnostics and bounded 10-second performance windows. A snapshot combines the latest complete window with heap measurements and configured CPU frequency. `runtime_load_percent` measures wall time inside the ESPectre runtime loop, including detector processing and listener delivery; it does not measure whole-system CPU utilization or transport work on other tasks. Detector timing is sampled on an evaluation tick after approximately 1,000 detector packets. [API.md](API.md#diagnostics) defines the wire fields, units, and optionality.

### Versioning

The public facades provide source compatibility under Semantic Versioning. Patch releases preserve documented behavior; minor releases add compatible APIs; major releases may break compatibility. Prereleases may change before the final release. Rebuild the SDK with your firmware: the source package does not promise binary ABI compatibility.

Construct public configuration and snapshot structs with their defaults, then assign named fields. Positional aggregate initialization is outside the compatibility contract because minor releases may append fields. Minor releases may also add callbacks with default implementations, types, functions, and overloads; existing calls retain their meaning. Detector coefficients and generated weights may change in compatible fixes, so exact floating-point telemetry is not guaranteed across releases.

Use `ESPECTRE_SDK_VERSION_STRING` to identify the SDK and `ESPECTRE_SDK_VERSION_AT_LEAST(major, minor, patch)` for compile-time feature guards. The legacy packed `ESPECTRE_SDK_VERSION_NUMBER` is not suitable for version ordering. Supply your application version separately through `EspectreDeviceInfo::firmware_version` and discovery or provisioning configuration; `ESPECTRE_PROTOCOL_VERSION` separately identifies the wire format.

Published packages stamp `runtime/espectre_sdk_version.h`. The SDK does not inspect Git or infer its identity from your application. If overriding the packaged version, define all four macros consistently for the SDK and its consumers: `ESPECTRE_SDK_VERSION_STRING`, `ESPECTRE_SDK_VERSION_MAJOR`, `ESPECTRE_SDK_VERSION_MINOR`, and `ESPECTRE_SDK_VERSION_PATCH`. A complete compiler override takes precedence over the header. Missing or incomplete identity falls back to `"0.0.0"` and zero numeric components, so version guards for newer releases evaluate to false.

## Advanced integrations

### Core-only

If your application owns CSI capture, include `espectre_core_sdk.h` and use the detectors with `TemporalCsiSampler`. Check `detector.is_valid()` after construction and `sampler.configure(...)` before processing packets: both use non-throwing allocation for bounded working buffers. Detectors and samplers are movable but non-copyable because they own live temporal state.

The sampler selects temporal slots; your application stores the selected normalized CSI payload. Process each incoming sample in this order:

1. Call `admit()` before replacing the stored payload.
2. If it returns `true`, consume the previously stored payload: clear detector history when `reset_required()` is true, call `advance_missing_slots(missing_slots_before())`, and then call `process_packet()`.
3. If `gap_reset_required()` is true, clear detector history before processing post-gap data.
4. If `selected_current()` is true, replace the stored payload with the current normalized CSI.

At the end of a finite stream, call `flush()` and consume the stored payload if it returns `true`. Your application also owns evaluation cadence and motion-hit filtering. Re-read `get_threshold()` after each `update_state()`: Lightweight can lower the threshold without a setter call, and this path has no `on_threshold_changed()` callback.

Keep raw samples separate from detector preparation. For LLTF input, normalize into the centered HT20 convention, zero missing bins with `zero_ht20_lltf_missing_bins()`, copy the raw view into a detector buffer, and call `impute_ht20_lltf_detector_bins()` only on that copy. [csi_pipeline.cpp](../src/cpp/runtime/esp_idf/csi_pipeline.cpp) shows the production normalization, admission, evaluation, and hit-filter sequence.

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

Link your application target to `espectre_core` to inherit the includes and C++ standard. If overriding SDK identity, apply all four version macros through `target_compile_definitions(espectre_core PUBLIC ...)`. Prefer layer-prefixed includes such as `runtime/runtime_interface.h` to avoid collisions with generic header names in your application.

### Transport and protocol extensions

Pass parsed SDK requests through `FrontendCommandEngine`. Keep requester-scoped query results separate from state changes published to active transports. An `EspectreCommandValidator` checks parameters and fills the command without changing device state; `FrontendCommandEngine::execute()` expects validation to have succeeded already.

For application routes, create an immutable `EspectreProtocolExtension` and check it with `validate_protocol_extension()`. Use that same catalog for capability output, `DirectHttpServiceConfig::protocol_extension`, `direct_http_request_to_command()`, and `parse_espectre_command()`, and keep it alive while the adapters use it. Your application implements the handlers and enforces each route's transport availability. [API.md](API.md#contract-principles) defines the shared message contract.

`RuntimeDirectHttpBridgeConfig::loop_time_ms_getter` can supply the latest complete application loop duration in milliseconds. Measure only the loop body, excluding other tasks and time between calls; without this callback, `loop_time_ms` is `null`. This measurement is separate from the runtime's `loop_avg_us` performance-window average.

Firmware updates remain application-owned. `RuntimeFrontendController::quiesce()` suspends telemetry, sensing services, and raw collection while retaining the configured backend. Restore the desired service and telemetry gates when resuming.

### Task priorities and raw CSI

Your application owns the task that calls `RuntimeFrontendController::loop()`. The SDK's **Advanced task scheduling** menu separately configures its service tasks, using priorities from `1` to `10`:

| Kconfig option | Default | Service |
|----------------|---------|---------|
| `CONFIG_ESPECTRE_DIRECT_HTTPD_TASK_PRIORITY` | `1` | Direct HTTP server |
| `CONFIG_ESPECTRE_DIRECT_WORKER_TASK_PRIORITY` | `2` | Direct responses and SSE delivery |
| `CONFIG_ESPECTRE_RAW_WORKER_TASK_PRIORITY` | `3` | Raw CSI HTTP delivery |
| `CONFIG_ESPECTRE_TRAFFIC_TASK_PRIORITY` | `1` | Managed PING or DNS traffic |

These are compile-time settings. Validate priority changes under the application's workload: higher-priority tasks preempt lower-priority work and can starve sensing or networking. ESP-IDF owns the internal Wi-Fi and lwIP priorities.

The built-in capture pipeline normalizes LLTF, HT, and VHT samples to `HT20_CSI_LEN`: 128 bytes containing 64 complex subcarriers. `RawCsiPacketView` exposes this view, so size capture queues for the normalized payload. The separate `RAW_CSI_MAX_PAYLOAD_BYTES` limit for stored and transmitted records is 512 bytes.

`EspIdfDirectHttpService` allocates a 16-slot raw queue when collection starts, with 128 payload bytes per slot, and releases it when collection stops. It sends batches of up to four records. Stalled sends can overflow the queue; inspect `raw_csi.raw_drop_total` separately from capture-quality rejections. Follow the callback and shutdown rules in [Threading](#threading).

## Source bundles

For source vendoring, download SDK `.tar.gz` or `.zip` assets from [GitHub Releases](https://github.com/francescopace/espectre/releases). Tagged releases use `espectre-sdk-<version>` filenames. The rolling [snapshot](https://github.com/francescopace/espectre/releases/tag/snapshot) release carries `espectre-sdk-preview` from `main`; [snapshot-dev](https://github.com/francescopace/espectre/releases/tag/snapshot-dev) carries `espectre-sdk-develop` from `develop`. Use rolling and prerelease builds for evaluation. The accompanying `sdk-manifest-<version>.json`, `sdk-manifest-preview.json`, or `sdk-manifest-develop.json` records the source identity and archive SHA-256 digests.

Source bundles place the component under `src/cpp/`; registry packages place it at the package root and include an example project. Both distributions compile with your application and contain no chip-specific precompiled libraries.

Registry packages include a generated `API.md` with the complete C++ reference and integration contracts for their version and source commit. Source bundles include the public header comments, `src/cpp/sdk_integration.dox`, and a stamped `src/cpp/Doxyfile`. To regenerate the reference XML locally, install Doxygen 1.17.0 and run `doxygen src/cpp/Doxyfile` from the extracted bundle root; the XML is written under `output/xml/`.

## Licensing

The public SDK is `GPL-3.0-only`, which permits commercial use subject to GPLv3's terms and applicable corresponding-source obligations. A separate commercial license is available for proprietary firmware. See [LICENSING.md](../LICENSING.md).
