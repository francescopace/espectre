/**
 * Minimal display test using exact same code as working demo
 */

#include <Adafruit_ST7789.h>

// Use board's built-in pin definitions (no manual defines needed)
Adafruit_ST7789 display = Adafruit_ST7789(TFT_CS, TFT_DC, TFT_RST);

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("Display Test Starting...");

  // Initialize display (exact same as working demo)
  display.init(135, 240);
  display.setRotation(3);

  // Enable backlight FIRST so we can see the colors!
  pinMode(TFT_BACKLITE, OUTPUT);
  digitalWrite(TFT_BACKLITE, HIGH);
  Serial.println("Backlight ON");

  Serial.println("Drawing colors...");

  // Test colors
  display.fillScreen(ST77XX_RED);
  Serial.println("RED");
  delay(1000);

  display.fillScreen(ST77XX_GREEN);
  Serial.println("GREEN");
  delay(1000);

  display.fillScreen(ST77XX_BLUE);
  Serial.println("BLUE");
  delay(1000);

  display.fillScreen(ST77XX_BLACK);
  Serial.println("BLACK");

  // Draw text
  display.setTextColor(ST77XX_WHITE);
  display.setTextSize(3);
  display.setCursor(20, 30);
  display.println("WORKS!");
  Serial.println("Text drawn");

  Serial.println("Done! Do you see colors and text?");
}

void loop() {
  delay(1000);
}
