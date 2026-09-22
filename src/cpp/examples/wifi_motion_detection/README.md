# Wi-Fi motion detection with ESPectre

This ESP-IDF project connects to Wi-Fi and logs motion using the ESPectre SDK. It sets up Wi-Fi and the event loop itself; in your own application, connect the sensing controller to the Wi-Fi code you already have.

## Create the project

You need ESP-IDF 5.5.3 or newer, with its environment activated. Pick a version from the [production registry](https://components.espressif.com/components/francescopace/espectre) (tagged releases and release candidates) or the [staging registry](https://components-staging.espressif.com/components/francescopace/espectre) (snapshots of `main` and `develop`, and older prereleases), and put it in place of `VERSION_FROM_REGISTRY`. For staging, also change the registry URL. The copy of this README inside a package already has both set.

```sh
ESPECTRE_VERSION="VERSION_FROM_REGISTRY"
ESPECTRE_REGISTRY_URL="https://components.espressif.com"
idf.py create-project-from-example --registry-url "$ESPECTRE_REGISTRY_URL" "francescopace/espectre=$ESPECTRE_VERSION:wifi_motion_detection"
cd wifi_motion_detection
```

The example's `main/idf_component.yml` pins the SDK version and registry. If an older staging example lists only a version, add `registry_url: https://components-staging.espressif.com` under `francescopace/espectre`.

The example needs neither MQTT nor an HTTP server. For those optional services, see the [SDK guide](https://espectre.dev/sdk/#optional-capability-groups). Console setup is up to you: pick the right ESP-IDF console for your board, or add TinyUSB yourself.

## Build and run

In the project folder, pick your chip (ESP32, ESP32-S2, ESP32-S3, ESP32-C3, ESP32-C5, or ESP32-C6). For ESP32-C3:

```sh
idf.py set-target esp32c3
idf.py menuconfig
idf.py build
idf.py -p YOUR_PORT flash monitor
```

In menuconfig, set your SSID and password under **ESPectre example** (a BSSID is optional). They are saved in the local `sdkconfig`: never commit or share it. CSI and the Lightweight detector are on by default; change the detector and other settings in the ESPectre sensing menu.

How it works:

- It registers a log sink, starts NVS and Wi-Fi, sets up the sensing controller, and connects. One task handles both Wi-Fi and sensing events.
- If the connection drops, it retries eight times in a row, then waits 30 seconds and tries again, so it recovers even after a long outage. Keep calling both `loop()` methods meanwhile.
- To stop, shut down the sensing controller first, then Wi-Fi. The Wi-Fi strings must stay valid until then (the example uses static strings).
- Motion is reported only when `ready_to_publish` is true, which needs a calibration after startup and after Wi-Fi recovers.
- Without credentials, it logs what to set and stops.

## Validate on hardware

A successful build does not prove it works. On the board, check that:

- the log shows the expected SDK version;
- Wi-Fi connects and sensing becomes ready;
- moving and standing still switch between motion and idle;
- after restarting the access point, sensing becomes ready again.

Save a short log for each check.

## Optional service build checks

CI builds this example with each optional SDK service group, and with all of them. `optional_services.cpp` references those services only to catch missing link dependencies; it never starts MQTT or Direct.

## License

The package is GPL-3.0-only; a separate commercial license is available for closed-source products. See the licensing files and the [SDK guide](https://espectre.dev/sdk/).
