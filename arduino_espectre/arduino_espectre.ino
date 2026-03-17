/**
 * ESPectre Arduino - WiFi CSI Motion Detector
 *
 * Standalone motion detector for Adafruit Feather ESP32-S3 Reverse TFT
 * Uses WiFi Channel State Information (CSI) to detect motion through walls
 *
 * Hardware:
 * - Adafruit Feather ESP32-S3 Reverse TFT (240x135 ST7789)
 * - Built-in NeoPixel LED
 * - External WiFi antenna (recommended)
 *
 * Required Libraries (install via Library Manager):
 * - Adafruit ST7789 (v1.10+)
 * - Adafruit GFX (v1.11+)
 * - Adafruit NeoPixel (v1.12+)
 * - Arduino-ESP32 (v2.0.14+)
 *
 * Algorithm: MVS (Moving Variance Segmentation) + NBVI calibration
 * Accuracy: ~97% in optimal conditions (3-8m from router)
 *
 * Adapted from ESPectre (github.com/paulhey/espectre)
 */

#include <WiFi.h>
#include <Adafruit_ST7789.h>
#include <Adafruit_GFX.h>
#include <Adafruit_NeoPixel.h>
#include "csi_manager.h"
#include "mvs_detector.h"
#include "nbvi_calibrator.h"
#include "gain_controller.h"
#include "config.h"

// Global objects
Adafruit_ST7789 tft = Adafruit_ST7789(TFT_CS, TFT_DC, TFT_RST);
Adafruit_NeoPixel pixel = Adafruit_NeoPixel(1, NEOPIXEL_PIN, NEO_GRB + NEO_KHZ800);
CSIManager csiManager;
MVSDetector detector(WINDOW_SIZE);
NBVICalibrator calibrator;
GainController gainController;

// State variables
std::vector<uint8_t> selected_band;
bool calibration_complete = false;
uint32_t last_display_update = 0;
TaskHandle_t trafficGenTask = NULL;

// Color definitions (using ST77XX standard colors)
#define COLOR_BLACK   ST77XX_BLACK
#define COLOR_WHITE   ST77XX_WHITE
#define COLOR_RED     ST77XX_RED
#define COLOR_GREEN   ST77XX_GREEN
#define COLOR_BLUE    ST77XX_BLUE
#define COLOR_YELLOW  ST77XX_YELLOW
#define COLOR_CYAN    ST77XX_CYAN
#define COLOR_MAGENTA ST77XX_MAGENTA

void setup() {
    Serial.begin(115200);
    delay(2000);  // Longer delay for Serial to stabilize
    Serial.println("\n\n=================================");
    Serial.println("ESPectre Arduino - Starting...");
    Serial.println("=================================");
    Serial.flush();

    // Initialize TFT display
    Serial.println("Initializing TFT display...");

    Serial.println("Initializing ST7789...");
    tft.init(135, 240);  // Init ST7789 240x135 (matches working demo)
    Serial.println("✓ ST7789 initialized");

    tft.setRotation(3);   // Landscape mode (240x135)
    Serial.println("✓ Rotation set to landscape");

    tft.setTextWrap(false);
    Serial.println("✓ TFT configuration complete");

    // Enable backlight early so we can see everything
    pinMode(TFT_BACKLITE, OUTPUT);
    digitalWrite(TFT_BACKLITE, HIGH);
    Serial.println("✓ Backlight enabled");

    // Welcome screen - test pattern first
    Serial.println("Drawing test pattern...");

    // Test: Fill screen with bright colors to verify display works
    tft.fillScreen(ST77XX_RED);
    delay(500);
    tft.fillScreen(ST77XX_GREEN);
    delay(500);
    tft.fillScreen(ST77XX_BLUE);
    delay(500);
    tft.fillScreen(ST77XX_BLACK);

    Serial.println("Drawing welcome screen...");
    tft.setTextColor(ST77XX_CYAN);
    tft.setTextSize(3);
    tft.setCursor(20, 30);
    tft.println("ESPectre");

    tft.setTextColor(ST77XX_WHITE);
    tft.setTextSize(1);
    tft.setCursor(20, 70);
    tft.println("CSI Motion Detector");
    tft.setCursor(20, 85);
    tft.println("Arduino Edition");
    Serial.println("✓ Welcome screen drawn");

    // Initialize NeoPixel
    Serial.println("Initializing NeoPixel...");
    pixel.begin();
    pixel.setBrightness(50);
    pixel.setPixelColor(0, pixel.Color(0, 0, 255));  // Blue = initializing
    pixel.show();
    Serial.println("✓ NeoPixel initialized");

    delay(2000);

    // Connect to WiFi
    Serial.println("\nConnecting to WiFi...");
    tft.fillScreen(COLOR_BLACK);
    tft.setTextSize(2);
    tft.setTextColor(COLOR_YELLOW);
    tft.setCursor(10, 10);
    tft.println("Connecting WiFi");

    tft.setTextSize(1);
    tft.setTextColor(COLOR_WHITE);
    tft.setCursor(10, 40);
    tft.print("SSID: ");
    tft.println(WIFI_SSID);

    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

    uint8_t connect_attempts = 0;
    while (WiFi.status() != WL_CONNECTED && connect_attempts < 30) {
        delay(500);
        Serial.print(".");
        connect_attempts++;
    }

    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("\nFailed to connect to WiFi!");
        tft.fillScreen(COLOR_BLACK);
        tft.setTextColor(COLOR_RED);
        tft.setCursor(10, 60);
        tft.println("WiFi FAILED!");
        tft.println("Check config.h");
        while (true) delay(1000);
    }

    Serial.println("\nWiFi connected!");
    Serial.print("IP Address: ");
    Serial.println(WiFi.localIP());
    Serial.print("RSSI: ");
    Serial.print(WiFi.RSSI());
    Serial.println(" dBm");

    tft.setCursor(10, 60);
    tft.setTextColor(COLOR_GREEN);
    tft.print("IP: ");
    tft.println(WiFi.localIP());
    tft.setCursor(10, 75);
    tft.print("RSSI: ");
    tft.print(WiFi.RSSI());
    tft.println(" dBm");

    delay(1000);

    // Initialize CSI Manager
    Serial.println("\nInitializing CSI...");
    if (!csiManager.begin()) {
        Serial.println("Failed to initialize CSI!");
        tft.fillScreen(COLOR_BLACK);
        tft.setTextColor(COLOR_RED);
        tft.setCursor(10, 60);
        tft.println("CSI FAILED!");
        while (true) delay(1000);
    }

    // Set CSI callback
    csiManager.setCallback([](const wifi_csi_info_t* data) {
        if (!calibration_complete) {
            // Calibration phase
            calibrator.collectSample(data->buf);
        } else {
            // Detection phase
            detector.processPacket(data->buf, selected_band);
        }
    });

    // Start traffic generator
    Serial.println("\nStarting traffic generator...");
    startTrafficGenerator();

    // Wait for traffic to stabilize
    Serial.println("Waiting 2 seconds for traffic to stabilize...");
    delay(2000);

    // Verify CSI packets are flowing
    uint32_t initial_count = csiManager.getTotalCount();
    Serial.printf("Initial CSI packet count: %u\n", initial_count);
    delay(1000);
    uint32_t after_count = csiManager.getTotalCount();
    Serial.printf("CSI packets after 1 second: %u (rate: %u pps)\n",
                  after_count, after_count - initial_count);

    if (after_count - initial_count < 10) {
        Serial.println("\n⚠️  WARNING: CSI packet rate is very low!");
        Serial.println("This may indicate:");
        Serial.println("  - Traffic generator not working");
        Serial.println("  - Gateway IP incorrect");
        Serial.println("  - WiFi connection issue");
        Serial.printf("  - Gateway IP: %s\n", WiFi.gatewayIP().toString().c_str());
        Serial.println("Continuing anyway, but calibration may fail...\n");
    } else {
        Serial.printf("✓ CSI packets flowing at ~%u pps\n\n", after_count - initial_count);
    }

    // Phase 1: Gain Lock (3 seconds)
    Serial.println("\n--- Phase 1: Gain Lock ---");
    tft.fillScreen(COLOR_BLACK);
    tft.setTextSize(2);
    tft.setTextColor(COLOR_BLUE);
    tft.setCursor(10, 30);
    tft.println("Gain Lock");

    tft.setTextSize(1);
    tft.setTextColor(COLOR_WHITE);
    tft.setCursor(10, 60);
    tft.println("Stabilizing AGC/FFT...");
    tft.setCursor(10, 75);
    tft.println("3 seconds");

    pixel.setPixelColor(0, pixel.Color(0, 0, 255));  // Blue
    pixel.show();

    delay(3000);

    // Attempt gain lock (gracefully continues if not available)
    bool gain_locked = gainController.lockGain();
    if (gain_locked && gainController.isLocked()) {
        Serial.printf("✓ Gain locked: AGC=%d, FFT=%d\n",
                     gainController.getAgcGain(), gainController.getFftGain());
    } else if (gainController.isSupported()) {
        Serial.println("⚠ Gain lock attempted but failed - continuing anyway");
    } else {
        Serial.println("⚠ Gain lock not available - CSI will still work");
    }

    // Phase 2: NBVI Calibration (7-10 seconds)
    Serial.println("\n--- Phase 2: NBVI Calibration ---");
    Serial.println("Keep room STILL for accurate calibration!");

    tft.fillScreen(COLOR_BLACK);
    tft.setTextSize(2);
    tft.setTextColor(COLOR_MAGENTA);
    tft.setCursor(10, 20);
    tft.println("Calibrating");

    tft.setTextSize(1);
    tft.setTextColor(COLOR_YELLOW);
    tft.setCursor(10, 50);
    tft.println("Keep room STILL!");

    tft.setTextColor(COLOR_WHITE);
    tft.setCursor(10, 75);
    tft.println("Collecting samples...");

    pixel.setPixelColor(0, pixel.Color(255, 0, 255));  // Magenta
    pixel.show();

    uint32_t cal_start = millis();
    uint32_t last_progress_update = 0;

    while (!calibrator.isComplete() && (millis() - cal_start < 12000)) {
        // Update progress every 200ms
        if (millis() - last_progress_update > 200) {
            size_t samples = calibrator.getSampleCount();
            float progress = (samples * 100.0f) / 700.0f;

            tft.fillRect(10, 95, 220, 20, COLOR_BLACK);
            tft.setCursor(10, 95);
            tft.print("Progress: ");
            tft.print((int)progress);
            tft.print("% (");
            tft.print(samples);
            tft.println("/700)");

            Serial.printf("Calibration progress: %d/700 (%.1f%%)\n", samples, progress);

            last_progress_update = millis();
        }

        delay(10);
    }

    if (!calibrator.isComplete()) {
        Serial.println("Warning: Calibration timeout! May have fewer samples.");
    }

    Serial.printf("Calibration complete: %d samples collected\n", calibrator.getSampleCount());

    // Select optimal band using NBVI
    Serial.println("\nSelecting optimal subcarriers...");
    selected_band = calibrator.selectBand();

    if (selected_band.size() != BAND_SIZE) {
        Serial.printf("Warning: Only %d subcarriers selected (expected %d)\n",
                     selected_band.size(), BAND_SIZE);
    }

    // Calculate adaptive threshold
    float threshold = calibrator.calculateAdaptiveThreshold(selected_band);
    detector.setThreshold(threshold);
    calibration_complete = true;

    Serial.println("\n=================================");
    Serial.println("Calibration Results:");
    Serial.print("Selected band: ");
    for (uint8_t sc : selected_band) {
        Serial.print(sc);
        Serial.print(" ");
    }
    Serial.println();
    Serial.printf("Adaptive threshold: %.3f\n", threshold);
    Serial.println("=================================\n");

    // Ready screen
    tft.fillScreen(COLOR_BLACK);
    tft.setTextSize(3);
    tft.setTextColor(COLOR_GREEN);
    tft.setCursor(40, 50);
    tft.println("READY!");

    pixel.setPixelColor(0, pixel.Color(0, 255, 0));  // Green = ready
    pixel.show();

    delay(1500);

    Serial.println("Starting motion detection...\n");
}

void loop() {
    // Update motion state
    detector.updateState();

    // Update display at 5 Hz
    uint32_t now = millis();
    if (now - last_display_update > 200) {
        updateDisplay();
        last_display_update = now;
    }

    delay(1);  // Yield to WiFi/FreeRTOS tasks
}

void updateDisplay() {
    if (!detector.isReady()) {
        // Warming up
        tft.fillScreen(COLOR_BLACK);
        tft.setTextSize(2);
        tft.setTextColor(COLOR_YELLOW);
        tft.setCursor(10, 50);
        tft.println("Warming up...");

        tft.setTextSize(1);
        tft.setCursor(10, 80);
        tft.print("Packets: ");
        tft.print(detector.getTotalPackets());
        tft.print("/");
        tft.println(detector.getWindowSize());
        return;
    }

    MotionState state = detector.getState();
    float metric = detector.getMotionMetric();
    float threshold = detector.getThreshold();

    // Clear screen
    tft.fillScreen(COLOR_BLACK);

    // Display motion state (large text)
    tft.setTextSize(4);
    tft.setCursor(10, 15);

    if (state == MOTION) {
        tft.setTextColor(COLOR_RED);
        tft.println("MOTION");
        pixel.setPixelColor(0, pixel.Color(255, 0, 0));  // Red LED

        Serial.print(">>> MOTION DETECTED | ");
    } else {
        tft.setTextColor(COLOR_GREEN);
        tft.println("Idle");
        pixel.setPixelColor(0, pixel.Color(0, 255, 0));  // Green LED

        Serial.print("--- Idle | ");
    }
    pixel.show();

    // Display metrics
    tft.setTextSize(1);
    tft.setTextColor(COLOR_WHITE);

    // Variance (motion metric)
    tft.setCursor(10, 70);
    tft.print("Variance: ");
    tft.setTextColor(COLOR_CYAN);
    tft.println(metric, 3);

    // Threshold
    tft.setTextColor(COLOR_WHITE);
    tft.setCursor(10, 85);
    tft.print("Threshold: ");
    tft.setTextColor(COLOR_YELLOW);
    tft.println(threshold, 3);

    // Packet count
    tft.setTextColor(COLOR_WHITE);
    tft.setCursor(10, 100);
    tft.print("Packets: ");
    tft.println(detector.getTotalPackets());

    // CSI stats
    tft.setCursor(10, 115);
    tft.print("CSI Total: ");
    tft.print(csiManager.getTotalCount());
    tft.print(" Drop: ");
    tft.println(csiManager.getDroppedCount());

    // Serial debug output
    Serial.printf("Var: %.3f | Thr: %.3f | Pkts: %u\n",
                 metric, threshold, detector.getTotalPackets());
}

void startTrafficGenerator() {
    xTaskCreate([](void* param) {
        // Wait a bit for system to stabilize
        vTaskDelay(1000 / portTICK_PERIOD_MS);

        IPAddress gateway = WiFi.gatewayIP();

        Serial.println("\n=== Traffic Generator Started ===");
        Serial.printf("Gateway: %s\n", gateway.toString().c_str());
        Serial.printf("Target rate: %d pps\n", TRAFFIC_RATE_PPS);
        Serial.println("Method: HTTP HEAD requests");
        Serial.println("================================\n");

        uint32_t packet_interval_ms = 1000 / TRAFFIC_RATE_PPS;
        uint32_t requests_sent = 0;
        uint32_t last_report = millis();

        while (true) {
            WiFiClient client;

            // Quick HTTP HEAD request to gateway (most routers have web interface)
            if (client.connect(gateway, 80, 100)) {  // 100ms timeout
                client.print("HEAD / HTTP/1.1\r\n");
                client.print("Host: ");
                client.print(gateway.toString());
                client.print("\r\n");
                client.print("Connection: close\r\n\r\n");
                client.flush();

                // Brief wait for response (generates CSI on RX)
                vTaskDelay(5 / portTICK_PERIOD_MS);

                client.stop();
                requests_sent++;
            } else {
                // If HTTP fails, fall back to UDP
                WiFiUDP udp;
                uint8_t dummy[] = {0x00, 0x01, 0x02, 0x03};
                udp.beginPacket(gateway, 53);
                udp.write(dummy, sizeof(dummy));
                udp.endPacket();
                requests_sent++;
            }

            // Report every 10 seconds
            if (millis() - last_report > 10000) {
                Serial.printf("Traffic gen: %u requests sent\n", requests_sent);
                last_report = millis();
            }

            vTaskDelay(packet_interval_ms / portTICK_PERIOD_MS);
        }
    }, "TrafficGen", 8192, NULL, 1, &trafficGenTask);  // Larger stack for WiFiClient
}
