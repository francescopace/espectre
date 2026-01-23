/*
 * ESPectre - NBVI Calibrator Unit Tests
 *
 * Tests the NBVICalibrator class for automatic subcarrier selection.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include <unity.h>
#include <cstdint>
#include <cstring>
#include <cmath>
#include "nbvi_calibrator.h"
#include "csi_manager.h"
#include "mvs_detector.h"
#include "wifi_csi_interface.h"
#include "esphome/core/log.h"

// Include CSI data loader
#include "csi_test_data.h"

#define baseline_packets csi_test_data::baseline_packets()
#define num_baseline csi_test_data::num_baseline()

using namespace esphome::espectre;

static const char *TAG = "test_nbvi_calibrator";

// Default subcarrier selection for testing
static const uint8_t DEFAULT_BAND[12] = {11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22};

/**
 * Mock WiFi CSI for testing
 */
class WiFiCSIMock : public IWiFiCSI {
 public:
  esp_err_t set_csi_config(const wifi_csi_config_t* config) override {
    (void)config;
    return ESP_OK;
  }
  esp_err_t set_csi_rx_cb(wifi_csi_cb_t cb, void* ctx) override {
    (void)cb; (void)ctx;
    return ESP_OK;
  }
  esp_err_t set_csi(bool enable) override {
    (void)enable;
    return ESP_OK;
  }
};

static WiFiCSIMock g_wifi_mock;
static MVSDetector g_detector(50, 1.0f);
static CSIManager g_csi_manager;

void setUp(void) {
    g_csi_manager.init(&g_detector, DEFAULT_BAND, 100, GainLockMode::DISABLED, &g_wifi_mock);
}

void tearDown(void) {
}

// ============================================================================
// INITIALIZATION TESTS
// ============================================================================

void test_nbvi_calibrator_init(void) {
    NBVICalibrator calibrator;
    
    calibrator.init(&g_csi_manager);
    
    TEST_ASSERT_FALSE(calibrator.is_calibrating());
}

void test_nbvi_calibrator_init_custom_path(void) {
    NBVICalibrator calibrator;
    
    calibrator.init(&g_csi_manager, "/tmp/test_buffer.bin");
    
    TEST_ASSERT_FALSE(calibrator.is_calibrating());
}

// ============================================================================
// CONFIGURATION TESTS
// ============================================================================

void test_nbvi_calibrator_set_buffer_size(void) {
    NBVICalibrator calibrator;
    calibrator.init(&g_csi_manager);
    
    calibrator.set_buffer_size(1000);
    
    TEST_ASSERT_EQUAL(1000, calibrator.get_buffer_size());
}

void test_nbvi_calibrator_set_window_size(void) {
    NBVICalibrator calibrator;
    calibrator.init(&g_csi_manager);
    
    calibrator.set_window_size(300);
    
    TEST_PASS();  // No getter, just verify no crash
}

void test_nbvi_calibrator_set_window_step(void) {
    NBVICalibrator calibrator;
    calibrator.init(&g_csi_manager);
    
    calibrator.set_window_step(100);
    
    TEST_PASS();
}

void test_nbvi_calibrator_set_percentile(void) {
    NBVICalibrator calibrator;
    calibrator.init(&g_csi_manager);
    
    calibrator.set_percentile(15);
    
    TEST_PASS();
}

void test_nbvi_calibrator_set_alpha(void) {
    NBVICalibrator calibrator;
    calibrator.init(&g_csi_manager);
    
    calibrator.set_alpha(0.7f);
    
    TEST_PASS();
}

void test_nbvi_calibrator_set_min_spacing(void) {
    NBVICalibrator calibrator;
    calibrator.init(&g_csi_manager);
    
    calibrator.set_min_spacing(2);
    
    TEST_PASS();
}

void test_nbvi_calibrator_set_noise_gate_percentile(void) {
    NBVICalibrator calibrator;
    calibrator.init(&g_csi_manager);
    
    calibrator.set_noise_gate_percentile(30);
    
    TEST_PASS();
}

// ============================================================================
// CALIBRATION STATE TESTS
// ============================================================================

void test_nbvi_calibrator_is_calibrating_false_initially(void) {
    NBVICalibrator calibrator;
    calibrator.init(&g_csi_manager);
    
    TEST_ASSERT_FALSE(calibrator.is_calibrating());
}

void test_nbvi_calibrator_start_calibration(void) {
    NBVICalibrator calibrator;
    calibrator.init(&g_csi_manager);
    
    bool callback_called = false;
    auto callback = [&callback_called](const uint8_t*, uint8_t, const std::vector<float>&, bool) {
        callback_called = true;
    };
    
    esp_err_t result = calibrator.start_calibration(DEFAULT_BAND, 12, callback);
    
    // On native platform, SPIFFS may not be available (ESP_ERR_NOT_SUPPORTED = 0x101)
    // Accept either success or not supported
    TEST_ASSERT_TRUE(result == ESP_OK || result == 0x101);
}

void test_nbvi_calibrator_start_calibration_while_calibrating(void) {
    NBVICalibrator calibrator;
    calibrator.init(&g_csi_manager);
    
    auto callback = [](const uint8_t*, uint8_t, const std::vector<float>&, bool) {};
    
    esp_err_t first_result = calibrator.start_calibration(DEFAULT_BAND, 12, callback);
    
    // Skip test if SPIFFS not available
    if (first_result == 0x101) {
        TEST_IGNORE_MESSAGE("SPIFFS not available on native platform");
        return;
    }
    
    esp_err_t result = calibrator.start_calibration(DEFAULT_BAND, 12, callback);
    TEST_ASSERT_EQUAL(ESP_ERR_INVALID_STATE, result);
}

// ============================================================================
// ADD PACKET TESTS
// ============================================================================

void test_nbvi_calibrator_add_packet_when_not_calibrating(void) {
    NBVICalibrator calibrator;
    calibrator.init(&g_csi_manager);
    
    int8_t csi_buf[128] = {0};
    bool result = calibrator.add_packet(csi_buf, 128);
    
    TEST_ASSERT_FALSE(result);
}

void test_nbvi_calibrator_add_packet_during_calibration(void) {
    NBVICalibrator calibrator;
    calibrator.init(&g_csi_manager);
    calibrator.set_buffer_size(10);  // Small buffer for testing
    
    auto callback = [](const uint8_t*, uint8_t, const std::vector<float>&, bool) {};
    esp_err_t result = calibrator.start_calibration(DEFAULT_BAND, 12, callback);
    
    // Skip test if SPIFFS not available
    if (result == 0x101) {
        TEST_IGNORE_MESSAGE("SPIFFS not available on native platform");
        return;
    }
    
    int8_t csi_buf[128] = {0};
    bool buffer_full = false;
    
    for (int i = 0; i < 15; i++) {
        buffer_full = calibrator.add_packet(csi_buf, 128);
        if (buffer_full) break;
    }
    
    TEST_ASSERT_TRUE(buffer_full);
}

// ============================================================================
// CALLBACK TESTS
// ============================================================================

void test_nbvi_calibrator_set_collection_complete_callback(void) {
    NBVICalibrator calibrator;
    calibrator.init(&g_csi_manager);
    
    bool callback_called = false;
    calibrator.set_collection_complete_callback([&callback_called]() {
        callback_called = true;
    });
    
    TEST_PASS();  // Just verify it doesn't crash
}

// ============================================================================
// REAL DATA TESTS
// ============================================================================

void test_nbvi_calibrator_add_real_packets(void) {
    if (!csi_test_data::load()) {
        TEST_IGNORE_MESSAGE("Failed to load test data");
        return;
    }
    
    NBVICalibrator calibrator;
    calibrator.init(&g_csi_manager);
    calibrator.set_buffer_size(50);  // Small buffer for faster test
    
    auto callback = [](const uint8_t*, uint8_t, const std::vector<float>&, bool) {};
    esp_err_t result = calibrator.start_calibration(DEFAULT_BAND, 12, callback);
    
    // Skip test if SPIFFS not available
    if (result == 0x101) {
        TEST_IGNORE_MESSAGE("SPIFFS not available on native platform");
        return;
    }
    
    bool buffer_full = false;
    int packets_added = 0;
    
    for (int i = 0; i < 60 && i < num_baseline; i++) {
        buffer_full = calibrator.add_packet(baseline_packets[i], 128);
        packets_added++;
        if (buffer_full) break;
    }
    
    ESP_LOGI(TAG, "Added %d packets, buffer_full=%d", packets_added, buffer_full);
    TEST_ASSERT_TRUE(buffer_full);
}

// ============================================================================
// ENTRY POINT
// ============================================================================

int process(void) {
    UNITY_BEGIN();
    
    // Initialization tests
    RUN_TEST(test_nbvi_calibrator_init);
    RUN_TEST(test_nbvi_calibrator_init_custom_path);
    
    // Configuration tests
    RUN_TEST(test_nbvi_calibrator_set_buffer_size);
    RUN_TEST(test_nbvi_calibrator_set_window_size);
    RUN_TEST(test_nbvi_calibrator_set_window_step);
    RUN_TEST(test_nbvi_calibrator_set_percentile);
    RUN_TEST(test_nbvi_calibrator_set_alpha);
    RUN_TEST(test_nbvi_calibrator_set_min_spacing);
    RUN_TEST(test_nbvi_calibrator_set_noise_gate_percentile);
    
    // Calibration state tests
    RUN_TEST(test_nbvi_calibrator_is_calibrating_false_initially);
    RUN_TEST(test_nbvi_calibrator_start_calibration);
    RUN_TEST(test_nbvi_calibrator_start_calibration_while_calibrating);
    
    // Add packet tests
    RUN_TEST(test_nbvi_calibrator_add_packet_when_not_calibrating);
    RUN_TEST(test_nbvi_calibrator_add_packet_during_calibration);
    
    // Callback tests
    RUN_TEST(test_nbvi_calibrator_set_collection_complete_callback);
    
    // Real data tests
    RUN_TEST(test_nbvi_calibrator_add_real_packets);
    
    return UNITY_END();
}

#if defined(ESP_PLATFORM)
extern "C" void app_main(void) { process(); }
#else
int main(int argc, char **argv) { return process(); }
#endif
