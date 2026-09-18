# Wi-Fi motion detection with ESPectre

This ESP-IDF project connects to Wi-Fi and logs motion events through the public SDK. It owns the Wi-Fi station and default event loop; integrate the sensing controller with your existing Wi-Fi lifecycle when embedding it in another application.

## Create the project

Use ESP-IDF 5.5.5 with its environment activated. Choose an exact version from the [staging registry](https://components-staging.espressif.com/components/francescopace/espectre) for a snapshot or prerelease, or the [production registry](https://components.espressif.com/components/francescopace/espectre) for a stable release. If production is not available yet, use staging for evaluation.

Replace `VERSION_FROM_REGISTRY` below with the selected version. For a stable release, change the registry URL to `https://components.espressif.com`. Packaged copies of this README already select their own version and registry.

```sh
ESPECTRE_VERSION="VERSION_FROM_REGISTRY"
ESPECTRE_REGISTRY_URL="https://components-staging.espressif.com"
idf.py create-project-from-example --registry-url "$ESPECTRE_REGISTRY_URL" "francescopace/espectre=$ESPECTRE_VERSION:wifi_motion_detection"
cd wifi_motion_detection
```

The packaged example's `main/idf_component.yml` pins the SDK version and registry. For an older staging example with only a version, add `registry_url: https://components-staging.espressif.com` under `francescopace/espectre` before building. Preview snapshots come from `main`; Develop snapshots come from `develop`. Both use staging, as do tagged prereleases.

Component Manager downloads the example and resolves its SDK dependency when configuring the project. The default configuration needs only ESP-IDF and ESPectre. Enabling Direct in menuconfig adds mDNS. After changing that option in an existing project, run `idf.py update-dependencies` to refresh the lockfile before building. The application owns its console configuration; select the appropriate ESP-IDF console for your board, or add and initialize TinyUSB in your application if needed.

## Build and run

From the created project directory, select a supported target: ESP32, ESP32-S2, ESP32-S3, ESP32-C3, ESP32-C5, or ESP32-C6. For ESP32-C3:

```sh
idf.py set-target esp32c3
idf.py menuconfig
idf.py build
idf.py -p YOUR_PORT flash monitor
```

Set your SSID and password under **ESPectre example**, and optionally pin an access point with its BSSID. Credentials are stored in the local `sdkconfig`, which must not be committed or redistributed. The defaults enable Wi-Fi CSI and the Lightweight detector. Use the ESPectre sensing menu to select another detector or change sensing parameters.

The example registers an SDK log sink, initializes NVS and Wi-Fi, sets up the sensing controller, and starts the station. Wi-Fi and runtime events are processed from the same task. The sensing runtime owns CSI acquisition; the Wi-Fi service owns connectivity and reconnection.

After eight immediate reconnect attempts, the Wi-Fi service waits 30 seconds and starts another burst, so an access point can return after a long outage. Keep calling both services' `loop()` methods during recovery. When stopping the application, shut down the sensing controller before the Wi-Fi service; the latter releases its driver and station interface and supports setup again. Credential strings must remain valid until Wi-Fi shutdown or replacement; this example uses static configuration strings.

Motion is reported only while `ready_to_publish` is true. Startup and Wi-Fi recovery may require calibration before movement is meaningful. If credentials are missing, the example logs an actionable error and stops.

## Validate on hardware

Confirm that the startup log reports the expected SDK version, Wi-Fi connects, sensing becomes ready, movement and idle transitions are reported, and reconnecting the access point restores sensing readiness. Capture a bounded log for each check; do not treat a successful build as runtime validation.

## Optional service build checks

The repository's SDK verifier generates CI configuration for individual SDK service groups or all groups together. `optional_services.cpp` references their implementations to catch missing link dependencies, but does not start MQTT or Direct servers. The default example only uses sensing and Wi-Fi.

The public package uses GPL-3.0-only. A separate commercial agreement is available for proprietary integration. See the component's licensing files and <https://espectre.dev/sdk/> for the complete integration guide.
