// SPDX-License-Identifier: GPL-3.0-only
// Commercial licensing available under separate agreement; see LICENSING.md.
//
// Shows motion detected from Wi-Fi CSI on the display and NeoPixel of the
// Adafruit Feather ESP32-S3 Reverse TFT. Set your network in arduino_secrets.h
// and install the "Adafruit ST7735 and ST7789 Library" from the Library Manager.
//
// Inspired by John Edgar Park's ESPectre demo for the same board.

#include <WiFi.h>
#include <Adafruit_GFX.h>
#include <Adafruit_ST7789.h>
#include <ESPectre.h>

#include "arduino_secrets.h"

// The board variant defines the display and NeoPixel pins.
#if !defined(TFT_CS) || !defined(TFT_DC) || !defined(TFT_RST) || !defined(TFT_BACKLITE) || !defined(PIN_NEOPIXEL)
#error "Select the Adafruit Feather ESP32-S3 Reverse TFT in Tools > Board."
#endif

const uint32_t REFRESH_INTERVAL_MS = 200;
const uint32_t FAULT_BLINK_MS = 400;
const uint32_t FAULT_HOLD_MS = 5000;
const uint16_t TRACK_COLOR = 0x4208;  // dark grey
const int16_t BAR_X = 6;
const int16_t BAR_Y = 124;
const int16_t BAR_WIDTH = 228;
const int16_t BAR_HEIGHT = 8;

Adafruit_ST7789 tft(TFT_CS, TFT_DC, TFT_RST);
ESPectre sensor;

// What the display reports.
enum class Screen : uint8_t { Connecting, Calibrating, Waiting, Idle, Motion, Fault };

// Headline and NeoPixel color of each screen, in Screen order. The NeoPixel is
// blue while waiting, green when idle, red on motion, and blinks yellow on a fault.
struct Look {
  const char *headline;
  uint16_t color;
  uint8_t red;
  uint8_t green;
  uint8_t blue;
};

const Look LOOKS[] = {
  {"Connecting", ST77XX_YELLOW, 0, 0, 40},
  {"Calibrating", ST77XX_MAGENTA, 0, 0, 40},
  {"Waiting", ST77XX_YELLOW, 0, 0, 40},
  {"Idle", ST77XX_GREEN, 0, 40, 0},
  {"Motion", ST77XX_RED, 40, 0, 0},
  {"Error", ST77XX_ORANGE, 40, 28, 0},
};

// A fixed-width line of text that redraws only the characters that changed,
// so a refresh stays short.
struct Field {
  int16_t x;
  int16_t y;
  uint8_t size;
  uint8_t width;  // characters, at most 39
  uint16_t color = ST77XX_BLACK;
  char shown[40] = {};

  void show(const char *text, uint16_t newColor) {
    char next[sizeof(shown)];
    snprintf(next, sizeof(next), "%-*.*s", width, width, text);
    for (uint8_t i = 0; i < width; i++) {
      const bool recolored = newColor != color && next[i] != ' ';
      if (next[i] != shown[i] || recolored) {
        tft.drawChar(x + i * 6 * size, y, next[i], newColor, ST77XX_BLACK, size);
      }
    }
    memcpy(shown, next, sizeof(shown));
    color = newColor;
  }
};

Field wifiField{120, 10, 1, 19};
Field headlineField{3, 36, 3, 13};
Field scoreField{6, 68, 2, 19};
Field thresholdField{6, 90, 2, 19};
Field hintField{6, 112, 1, 38};

char faultMessage[80] = "";
bool faulted = false;
uint32_t faultMs = 0;
bool refreshNow = true;
uint32_t lastRefreshMs = 0;

// Callbacks run inside sensor.loop(), so they record what happened and leave
// the drawing to the next refresh. The display shortens a long fault message.
void reportFault(const char *message) {
  Serial.printf("ESPectre fault: %s\n", message);
  strlcpy(faultMessage, message, sizeof(faultMessage));
  faulted = true;
  faultMs = millis();
  refreshNow = true;
}

void showLed(uint8_t red, uint8_t green, uint8_t blue) {
  static uint32_t shown = UINT32_MAX;
  const uint32_t color = (uint32_t(red) << 16) | (uint32_t(green) << 8) | blue;
  if (color != shown) {
    rgbLedWrite(PIN_NEOPIXEL, red, green, blue);
    shown = color;
  }
}

// Draws the calibration bar, or removes it when percent is negative.
void showProgress(int percent) {
  static int16_t shown = -1;
  const int16_t fill = percent < 0 ? -1 : BAR_WIDTH * percent / 100;
  if (fill == shown) {
    return;
  }
  if (fill < 0) {
    tft.fillRect(BAR_X, BAR_Y, BAR_WIDTH, BAR_HEIGHT, ST77XX_BLACK);
  } else if (shown < 0) {
    tft.fillRect(BAR_X, BAR_Y, fill, BAR_HEIGHT, ST77XX_MAGENTA);
    tft.fillRect(BAR_X + fill, BAR_Y, BAR_WIDTH - fill, BAR_HEIGHT, TRACK_COLOR);
  } else if (fill > shown) {
    tft.fillRect(BAR_X + shown, BAR_Y, fill - shown, BAR_HEIGHT, ST77XX_MAGENTA);
  } else {
    tft.fillRect(BAR_X + fill, BAR_Y, shown - fill, BAR_HEIGHT, TRACK_COLOR);
  }
  shown = fill;
}

void refreshDisplay() {
  static Screen shown = Screen::Fault;
  const bool connected = WiFi.status() == WL_CONNECTED;
  const auto &snapshot = sensor.runtime().snapshot();

  // A fault stays up long enough to read, and until results can be trusted again.
  if (faulted && sensor.ready() && millis() - faultMs >= FAULT_HOLD_MS) {
    faulted = false;
  }

  Screen screen = Screen::Waiting;
  if (faulted) {
    screen = Screen::Fault;
  } else if (!connected) {
    screen = Screen::Connecting;
  } else if (sensor.ready()) {
    screen = sensor.motion() ? Screen::Motion : Screen::Idle;
  } else if (snapshot.calibrating) {
    screen = Screen::Calibrating;
  }

  // Score, threshold, and progress stay empty unless this screen fills them,
  // so a lost connection or readiness removes the last result.
  char score[20] = "";
  char threshold[20] = "";
  const char *hint = "";
  int progress = -1;
  switch (screen) {
    case Screen::Idle:
    case Screen::Motion:
      snprintf(score, sizeof(score), "Score     %.3f", sensor.movement());
      snprintf(threshold, sizeof(threshold), "Threshold %.3f", sensor.threshold());
      break;
    case Screen::Calibrating:
      hint = "Keep the room quiet";
      // The runtime reports the packet budget, which can grow while it calibrates.
      if (snapshot.calibration_target_packets > 0) {
        const uint32_t target = snapshot.calibration_target_packets;
        const uint32_t packets = snapshot.calibration_packets < target ? snapshot.calibration_packets : target;
        progress = packets * 100 / target;
        snprintf(score, sizeof(score), "Progress  %3d%%", progress);
      }
      break;
    case Screen::Connecting:
      hint = sizeof(SECRET_SSID) > 1 ? "Network: " SECRET_SSID : "Set your network in arduino_secrets.h";
      break;
    case Screen::Waiting:
      hint = "Waiting for CSI packets";
      break;
    case Screen::Fault:
      hint = faultMessage;
      break;
  }

  const Look &look = LOOKS[static_cast<uint8_t>(screen)];
  wifiField.show(connected ? "Wi-Fi connected" : "Wi-Fi connecting...", connected ? ST77XX_GREEN : ST77XX_YELLOW);
  headlineField.show(look.headline, look.color);
  scoreField.show(score, ST77XX_WHITE);
  thresholdField.show(threshold, ST77XX_WHITE);
  hintField.show(hint, ST77XX_WHITE);
  showProgress(progress);

  const bool blinkOff = screen == Screen::Fault && (millis() / FAULT_BLINK_MS) % 2 == 1;
  showLed(blinkOff ? 0 : look.red, blinkOff ? 0 : look.green, blinkOff ? 0 : look.blue);

  if (screen != shown) {
    shown = screen;
    if (screen != Screen::Fault) {
      Serial.println(look.headline);
    }
  }
}

void setup() {
  Serial.begin(115200);

  tft.init(135, 240);
  tft.setRotation(3);
  tft.setTextWrap(false);
  tft.fillScreen(ST77XX_BLACK);
  tft.setTextSize(2);
  tft.setTextColor(ST77XX_CYAN);
  tft.setCursor(6, 6);
  tft.print("ESPectre");
  pinMode(TFT_BACKLITE, OUTPUT);
  digitalWrite(TFT_BACKLITE, HIGH);

  // The sketch owns Wi-Fi. ESPectre needs the station, not the connection.
  WiFi.mode(WIFI_STA);
  WiFi.begin(SECRET_SSID, SECRET_PASSWORD);

  sensor.onFault(reportFault);
  sensor.onReady([](bool) { refreshNow = true; });
  sensor.onMotion([](bool) { refreshNow = true; });

  if (!sensor.begin() && !faulted) {
    reportFault("Unable to start ESPectre");
  }
}

void loop() {
  sensor.loop();

  // Refresh at about 5 Hz, and at once when a callback reports a change.
  const uint32_t now = millis();
  if (refreshNow || now - lastRefreshMs >= REFRESH_INTERVAL_MS) {
    refreshNow = false;
    lastRefreshMs = now;
    refreshDisplay();
  }
  delay(10);
}
