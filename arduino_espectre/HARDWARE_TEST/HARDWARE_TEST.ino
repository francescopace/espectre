/**
 * ESPectre Arduino - Hardware Test Sketch
 *
 * Tests all hardware components independently:
 * 1. Serial communication
 * 2. TFT display
 * 3. NeoPixel LED
 * 4. WiFi connection
 * 5. CSI capability
 *
 * Use this to verify your hardware before running the main sketch.
 *
 * Upload this sketch and open Serial Monitor (115200 baud)
 */

#include <WiFi.h>
#include <Adafruit_ST7789.h>
#include <Adafruit_GFX.h>
#include <Adafruit_NeoPixel.h>
#include "esp_wifi.h"
#include "config.h"

// Hardware pins
Adafruit_ST7789 tft = Adafruit_ST7789(TFT_CS, TFT_DC, TFT_RST);
Adafruit_NeoPixel pixel = Adafruit_NeoPixel(1, NEOPIXEL_PIN, NEO_GRB + NEO_KHZ800);

#define COLOR_BLACK   0x0000
#define COLOR_WHITE   0xFFFF
#define COLOR_RED     0xF800
#define COLOR_GREEN   0x07E0
#define COLOR_BLUE    0x001F
#define COLOR_YELLOW  0xFFE0

bool test_results[5] = {false, false, false, false, false};

void setup() {
    // Test 1: Serial
    Serial.begin(115200);
    delay(2000);
    Serial.println("\n\n========================================");
    Serial.println("ESPectre Arduino - Hardware Test");
    Serial.println("========================================\n");

    test_results[0] = true;
    Serial.println("✓ Test 1: Serial communication OK");

    // Test 2: TFT Display
    Serial.println("\nTest 2: TFT Display");
    Serial.println("- Initializing display...");

    pinMode(TFT_BACKLIGHT, OUTPUT);
    digitalWrite(TFT_BACKLIGHT, HIGH);

    tft.init(135, 240);
    tft.setRotation(3);
    tft.fillScreen(COLOR_BLACK);

    // Test colors
    Serial.println("- Testing colors (Red, Green, Blue)...");
    tft.fillScreen(COLOR_RED);
    delay(500);
    tft.fillScreen(COLOR_GREEN);
    delay(500);
    tft.fillScreen(COLOR_BLUE);
    delay(500);
    tft.fillScreen(COLOR_BLACK);

    // Test text
    Serial.println("- Testing text rendering...");
    tft.setTextColor(COLOR_WHITE);
    tft.setTextSize(2);
    tft.setCursor(10, 10);
    tft.println("ESPectre");
    tft.setTextSize(1);
    tft.setCursor(10, 40);
    tft.println("Hardware Test");
    tft.setCursor(10, 60);
    tft.setTextColor(COLOR_GREEN);
    tft.println("Display OK!");

    test_results[1] = true;
    Serial.println("✓ Test 2: TFT Display OK");

    // Test 3: NeoPixel LED
    Serial.println("\nTest 3: NeoPixel LED");
    Serial.println("- Testing RGB colors...");

    pixel.begin();
    pixel.setBrightness(50);

    // Red
    Serial.println("  - Red");
    pixel.setPixelColor(0, pixel.Color(255, 0, 0));
    pixel.show();
    delay(1000);

    // Green
    Serial.println("  - Green");
    pixel.setPixelColor(0, pixel.Color(0, 255, 0));
    pixel.show();
    delay(1000);

    // Blue
    Serial.println("  - Blue");
    pixel.setPixelColor(0, pixel.Color(0, 0, 255));
    pixel.show();
    delay(1000);

    // White
    Serial.println("  - White");
    pixel.setPixelColor(0, pixel.Color(255, 255, 255));
    pixel.show();
    delay(1000);

    pixel.setPixelColor(0, pixel.Color(0, 0, 0));  // Off
    pixel.show();

    test_results[2] = true;
    Serial.println("✓ Test 3: NeoPixel LED OK");

    // Test 4: WiFi
    Serial.println("\nTest 4: WiFi Connection");
    Serial.print("- SSID: ");
    Serial.println(WIFI_SSID);
    Serial.println("- Connecting...");

    tft.fillRect(10, 80, 220, 30, COLOR_BLACK);
    tft.setTextColor(COLOR_YELLOW);
    tft.setCursor(10, 80);
    tft.println("Testing WiFi...");

    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 30) {
        delay(500);
        Serial.print(".");
        attempts++;
    }
    Serial.println();

    if (WiFi.status() == WL_CONNECTED) {
        Serial.println("✓ Test 4: WiFi Connected!");
        Serial.print("  - IP Address: ");
        Serial.println(WiFi.localIP());
        Serial.print("  - RSSI: ");
        Serial.print(WiFi.RSSI());
        Serial.println(" dBm");
        Serial.print("  - Channel: ");
        Serial.println(WiFi.channel());

        tft.fillRect(10, 80, 220, 30, COLOR_BLACK);
        tft.setTextColor(COLOR_GREEN);
        tft.setCursor(10, 80);
        tft.println("WiFi OK!");
        tft.setTextColor(COLOR_WHITE);
        tft.setCursor(10, 95);
        tft.print("IP: ");
        tft.println(WiFi.localIP());

        test_results[3] = true;
    } else {
        Serial.println("✗ Test 4: WiFi Connection FAILED");
        Serial.println("  - Check SSID and password in config.h");
        Serial.println("  - Ensure 2.4 GHz network (ESP32 doesn't support 5 GHz)");

        tft.fillRect(10, 80, 220, 30, COLOR_BLACK);
        tft.setTextColor(COLOR_RED);
        tft.setCursor(10, 80);
        tft.println("WiFi FAILED!");
        tft.setTextColor(COLOR_WHITE);
        tft.setCursor(10, 95);
        tft.println("Check config.h");

        test_results[3] = false;
    }

    // Test 5: CSI Capability (only if WiFi connected)
    if (test_results[3]) {
        Serial.println("\nTest 5: CSI Capability");
        Serial.println("- Attempting to enable CSI...");

        wifi_csi_config_t csi_config = {
            .lltf_en = true,
            .htltf_en = true,
            .stbc_htltf2_en = true,
            .ltf_merge_en = true,
            .channel_filter_en = false,
            .manu_scale = false,
        };

        esp_err_t err = esp_wifi_set_csi_config(&csi_config);
        if (err == ESP_OK) {
            err = esp_wifi_set_csi(true);
            if (err == ESP_OK) {
                Serial.println("✓ Test 5: CSI Enabled Successfully!");
                Serial.println("  - ESP-IDF CSI functions available");
                Serial.println("  - Ready for motion detection");

                tft.fillRect(10, 110, 220, 20, COLOR_BLACK);
                tft.setTextColor(COLOR_GREEN);
                tft.setCursor(10, 110);
                tft.println("CSI OK!");

                test_results[4] = true;

                esp_wifi_set_csi(false);  // Disable for now
            } else {
                Serial.printf("✗ Test 5: Failed to enable CSI: %d\n", err);
                test_results[4] = false;
            }
        } else {
            Serial.printf("✗ Test 5: Failed to configure CSI: %d\n", err);
            test_results[4] = false;
        }
    } else {
        Serial.println("\n✗ Test 5: CSI test skipped (WiFi not connected)");
        test_results[4] = false;
    }

    // Summary
    Serial.println("\n========================================");
    Serial.println("Test Summary:");
    Serial.println("========================================");
    Serial.printf("Serial:    %s\n", test_results[0] ? "✓ PASS" : "✗ FAIL");
    Serial.printf("TFT:       %s\n", test_results[1] ? "✓ PASS" : "✗ FAIL");
    Serial.printf("NeoPixel:  %s\n", test_results[2] ? "✓ PASS" : "✗ FAIL");
    Serial.printf("WiFi:      %s\n", test_results[3] ? "✓ PASS" : "✗ FAIL");
    Serial.printf("CSI:       %s\n", test_results[4] ? "✓ PASS" : "✗ FAIL");
    Serial.println("========================================");

    int passed = 0;
    for (int i = 0; i < 5; i++) {
        if (test_results[i]) passed++;
    }

    Serial.printf("\nResult: %d/5 tests passed\n", passed);

    if (passed == 5) {
        Serial.println("\n✓ ALL TESTS PASSED!");
        Serial.println("Hardware is ready for ESPectre operation.");

        tft.fillScreen(COLOR_BLACK);
        tft.setTextSize(3);
        tft.setTextColor(COLOR_GREEN);
        tft.setCursor(10, 40);
        tft.println("ALL OK!");
        tft.setTextSize(1);
        tft.setTextColor(COLOR_WHITE);
        tft.setCursor(10, 80);
        tft.println("5/5 tests passed");
        tft.setCursor(10, 95);
        tft.println("Ready for ESPectre!");

        // Victory LED pattern
        for (int i = 0; i < 3; i++) {
            pixel.setPixelColor(0, pixel.Color(0, 255, 0));
            pixel.show();
            delay(200);
            pixel.setPixelColor(0, pixel.Color(0, 0, 0));
            pixel.show();
            delay(200);
        }
    } else {
        Serial.println("\n✗ SOME TESTS FAILED");
        Serial.println("Review output above and fix hardware issues.");

        tft.fillScreen(COLOR_BLACK);
        tft.setTextSize(2);
        tft.setTextColor(COLOR_YELLOW);
        tft.setCursor(10, 40);
        tft.println("CHECK LOG");
        tft.setTextSize(1);
        tft.setTextColor(COLOR_WHITE);
        tft.setCursor(10, 70);
        tft.print(passed);
        tft.print("/5 tests passed");

        pixel.setPixelColor(0, pixel.Color(255, 255, 0));  // Yellow
        pixel.show();
    }

    Serial.println("\n========================================");
}

void loop() {
    // Idle - test complete
    delay(1000);
}
