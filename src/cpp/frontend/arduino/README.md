# ESPectre Arduino library

The ESPectre Arduino library detects motion from Wi-Fi channel state information (CSI) in an Arduino-ESP32 sketch. Your sketch keeps control of Wi-Fi and of everything else it does.

Arduino support is in development for v3.2.0 and is not yet a supported release.

## Requirements

- Arduino-ESP32 core 3.3 or later.
- ESP32, ESP32-S2, ESP32-S3, ESP32-C3, ESP32-C5, or ESP32-C6.
- A 2.4 GHz or 5 GHz access point that the board stays connected to.

## Installation

Releases do not include the library yet. CI builds `espectre-arduino-<version>.zip` as the `arduino-library` workflow artifact. In the Arduino IDE, choose **Sketch > Include Library > Add .ZIP Library**. With Arduino CLI, run:

```bash
arduino-cli config set library.enable_unsafe_install true
```

```bash
arduino-cli lib install --zip-path espectre-arduino-<version>.zip
```

## Getting started

Open **File > Examples > ESPectre > MotionDetection**, set your network in `arduino_secrets.h`, and upload it. Open the serial monitor at 115200 baud.

```cpp
#include <WiFi.h>
#include <ESPectre.h>

ESPectre sensor;

void setup() {
  Serial.begin(115200);
  WiFi.mode(WIFI_STA);
  WiFi.begin("your-network", "your-password");
  sensor.onMotion([](bool motion) { Serial.println(motion ? "Motion" : "Idle"); });
  sensor.begin();
}

void loop() {
  sensor.loop();
  delay(10);
}
```

Call `begin()` after `WiFi.begin()`. The connection does not have to be up yet. Call `loop()` from every Arduino loop: callbacks run there, so keep them short.

The sensor calibrates after each connection. `ready()` is false until then, and `motion()` stays false while sensing is not ready.

## API

| Member | Purpose |
|--------|---------|
| `config()` | Configuration for the next `begin()`, such as `threshold` or `csi_target_pps` |
| `begin()` / `end()` | Start or stop sensing. `begin()` returns false when setup fails |
| `loop()` | Advance sensing and deliver callbacks |
| `ready()`, `motion()` | Whether results can be trusted, and whether motion is detected |
| `movement()`, `threshold()` | Latest movement score and the threshold it is compared against |
| `onMotion()`, `onReady()`, `onFault()` | Callbacks for motion changes, readiness changes, and runtime faults |
| `runtime()` | The full SDK controller for thresholds, recalibration, and diagnostics |

The configuration starts from the SDK defaults. Settings changed at runtime are not saved, so the values in your sketch always apply after a reboot. Set `config().persist_runtime_overrides` to true to save them.

See the [SDK guide](https://espectre.dev/sdk/) for the full runtime contract.

## Wi-Fi

ESPectre turns Wi-Fi power save off at every connection, because it reduces the CSI rate. Do not call `WiFi.setSleep(true)` while sensing.

ESPectre sends a small stream of ping packets to the gateway to keep CSI flowing. See [traffic destination](https://espectre.dev/sdk/#traffic-destination) to choose another target.

## Logging

ESPectre logs follow the **Core Debug Level** setting, like the core's own messages. They are off by default.

## Limitations

- The library includes sensing only. MQTT, Direct HTTP, and provisioning are not included.
- The `traffic.tx_packets_total` and `traffic.rx_packets_total` diagnostics stay at zero.
- On ESP32, the fixed transmit rate does not apply, because Arduino builds ESP-IDF with A-MPDU enabled.
- The example uses 70–88% of the default 1.2 MB app partition, the most on ESP32-C5 and ESP32-C6. Choose a larger **Partition Scheme** when your sketch adds other libraries.

## Licensing

ESPectre is available under GPLv3 and a separate commercial license. See `LICENSING.md`.
