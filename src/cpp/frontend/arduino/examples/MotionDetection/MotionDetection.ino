// SPDX-License-Identifier: GPL-3.0-only
// Commercial licensing available under separate agreement; see LICENSING.md.
//
// Prints motion detected from Wi-Fi CSI. Set your network in arduino_secrets.h.

#include <WiFi.h>
#include <ESPectre.h>

#include "arduino_secrets.h"

ESPectre sensor;

void setup() {
  Serial.begin(115200);

  // The sketch owns Wi-Fi. ESPectre needs the station, not the connection.
  WiFi.mode(WIFI_STA);
  WiFi.begin(SECRET_SSID, SECRET_PASSWORD);

  sensor.onReady([](bool ready) {
    Serial.println(ready ? "Sensing ready" : "Waiting for Wi-Fi or calibration");
  });
  sensor.onMotion([](bool motion) {
    Serial.println(motion ? "Motion detected" : "Idle");
  });
  sensor.onFault([](const char *message) {
    Serial.printf("ESPectre fault: %s\n", message);
  });

  if (!sensor.begin()) {
    Serial.println("Unable to start ESPectre");
  }
}

void loop() {
  sensor.loop();
  delay(10);
}
