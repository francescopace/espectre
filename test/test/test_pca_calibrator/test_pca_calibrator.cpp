/*
 * ESPectre - PCA Calibrator Unit Tests
 * 
 * Tests for the PCACalibrator class that collects baseline correlation
 * values for PCA-based motion detection threshold calculation.
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include <unity.h>
#include <cstring>
#include <vector>
#include "pca_calibrator.h"
#include "csi_manager.h"

// Test data from csi_test_data.h
#include "csi_test_data.h"

// Mocks
#include "esphome/core/log.h"
#include "esp_wifi.h"
#include "wifi_csi_interface.h"

using namespace esphome::espectre;

// ============================================================================
// Macros for test data access
// ============================================================================
#define baseline_packets csi_test_data::baseline_packets()
#define num_baseline csi_test_data::num_baseline()

// ============================================================================
// Mock WiFi interface for CSI manager
// ============================================================================
class WiFiCSIMock : public IWiFiCSI {
 public:
  esp_err_t set_csi_config(const wifi_csi_config_t* config) override { return ESP_OK; }
  esp_err_t set_csi_rx_cb(wifi_csi_cb_t cb, void* ctx) override { return ESP_OK; }
  esp_err_t set_csi(bool enable) override { return ESP_OK; }
};

static WiFiCSIMock g_wifi_mock;

// Default subcarrier selection
static constexpr uint8_t NUM_SELECTED_SUBCARRIERS = 12;
static const uint8_t* get_optimal_subcarriers() {
    static const uint8_t sc[12] = {11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22};
    return sc;
}

// ============================================================================
// Test Helpers
// ============================================================================

class MockDetector : public IDetector {
public:
    void process_packet(const int8_t* csi_data, size_t csi_len,
                        const uint8_t* selected_subcarriers = nullptr,
                        uint8_t num_subcarriers = 0) override {
        packet_count_++;
    }
    void update_state() override {}
    MotionState get_state() const override { return MotionState::IDLE; }
    float get_motion_metric() const override { return 0.0f; }
    bool set_threshold(float threshold) override { return true; }
    float get_threshold() const override { return 1.0f; }
    void reset() override { packet_count_ = 0; }
    bool is_ready() const override { return true; }
    uint32_t get_total_packets() const override { return packet_count_; }
    const char* get_name() const override { return "Mock"; }
    
private:
    uint32_t packet_count_{0};
};

// ============================================================================
// Basic Initialization Tests
// ============================================================================

void test_pca_calibrator_constructor(void) {
    PCACalibrator calibrator;
    TEST_ASSERT_FALSE(calibrator.is_calibrating());
    TEST_ASSERT_EQUAL(700, calibrator.get_buffer_size());  // Same as NBVI/P95
}

void test_pca_calibrator_init(void) {
    MockDetector detector;
    CSIManager csi_manager;
    csi_manager.init(&detector, get_optimal_subcarriers(), 100, GainLockMode::DISABLED, &g_wifi_mock);
    
    PCACalibrator calibrator;
    calibrator.init(&csi_manager);
    
    TEST_ASSERT_FALSE(calibrator.is_calibrating());
}

void test_pca_calibrator_set_buffer_size(void) {
    PCACalibrator calibrator;
    calibrator.set_buffer_size(500);
    TEST_ASSERT_EQUAL(500, calibrator.get_buffer_size());
}

// ============================================================================
// Calibration Start Tests
// ============================================================================

void test_pca_calibrator_start_without_init(void) {
    PCACalibrator calibrator;
    
    std::vector<float> cal_values;
    bool callback_called = false;
    
    esp_err_t err = calibrator.start_calibration(nullptr, 0,
        [&](const uint8_t* band, uint8_t size, const std::vector<float>& values, bool success) {
            callback_called = true;
        });
    
    TEST_ASSERT_EQUAL(ESP_ERR_INVALID_STATE, err);
    TEST_ASSERT_FALSE(callback_called);
}

void test_pca_calibrator_start_success(void) {
    MockDetector detector;
    CSIManager csi_manager;
    csi_manager.init(&detector, get_optimal_subcarriers(), 100, GainLockMode::DISABLED, &g_wifi_mock);
    
    PCACalibrator calibrator;
    calibrator.init(&csi_manager);
    
    bool callback_called = false;
    
    esp_err_t err = calibrator.start_calibration(nullptr, 0,
        [&](const uint8_t* band, uint8_t size, const std::vector<float>& values, bool success) {
            callback_called = true;
        });
    
    TEST_ASSERT_EQUAL(ESP_OK, err);
    TEST_ASSERT_TRUE(calibrator.is_calibrating());
}

void test_pca_calibrator_start_while_calibrating(void) {
    MockDetector detector;
    CSIManager csi_manager;
    csi_manager.init(&detector, get_optimal_subcarriers(), 100, GainLockMode::DISABLED, &g_wifi_mock);
    
    PCACalibrator calibrator;
    calibrator.init(&csi_manager);
    
    // Start first calibration
    esp_err_t err1 = calibrator.start_calibration(nullptr, 0,
        [](const uint8_t*, uint8_t, const std::vector<float>&, bool) {});
    TEST_ASSERT_EQUAL(ESP_OK, err1);
    
    // Try to start second calibration
    esp_err_t err2 = calibrator.start_calibration(nullptr, 0,
        [](const uint8_t*, uint8_t, const std::vector<float>&, bool) {});
    TEST_ASSERT_EQUAL(ESP_ERR_INVALID_STATE, err2);
}

// ============================================================================
// Packet Processing Tests
// ============================================================================

void test_pca_calibrator_add_packet_not_calibrating(void) {
    MockDetector detector;
    CSIManager csi_manager;
    csi_manager.init(&detector, get_optimal_subcarriers(), 100, GainLockMode::DISABLED, &g_wifi_mock);
    
    PCACalibrator calibrator;
    calibrator.init(&csi_manager);
    
    // Not in calibration mode - should return false
    int8_t dummy_data[128] = {0};
    bool result = calibrator.add_packet(dummy_data, 128);
    TEST_ASSERT_FALSE(result);
}

void test_pca_calibrator_add_packet_increments(void) {
    if (!csi_test_data::load()) {
        TEST_IGNORE_MESSAGE("Failed to load test data");
        return;
    }
    
    MockDetector detector;
    CSIManager csi_manager;
    csi_manager.init(&detector, get_optimal_subcarriers(), 100, GainLockMode::DISABLED, &g_wifi_mock);
    
    PCACalibrator calibrator;
    calibrator.init(&csi_manager);
    calibrator.set_buffer_size(100);  // Small buffer for test
    
    bool calibration_complete = false;
    
    esp_err_t err = calibrator.start_calibration(nullptr, 0,
        [&](const uint8_t*, uint8_t, const std::vector<float>&, bool) {
            calibration_complete = true;
        });
    TEST_ASSERT_EQUAL(ESP_OK, err);
    
    // Add some packets
    const int pkt_size = csi_test_data::packet_size();
    for (int i = 0; i < 50 && i < num_baseline; i++) {
        calibrator.add_packet(baseline_packets[i], pkt_size);
    }
    
    // Should still be calibrating (not enough packets)
    TEST_ASSERT_TRUE(calibrator.is_calibrating());
}

// ============================================================================
// Full Calibration Tests
// ============================================================================

void test_pca_calibrator_full_calibration(void) {
    if (!csi_test_data::load()) {
        TEST_IGNORE_MESSAGE("Failed to load test data");
        return;
    }
    
    MockDetector detector;
    CSIManager csi_manager;
    csi_manager.init(&detector, get_optimal_subcarriers(), 100, GainLockMode::DISABLED, &g_wifi_mock);
    
    PCACalibrator calibrator;
    calibrator.init(&csi_manager);
    calibrator.set_buffer_size(100);  // Smaller for faster test
    
    bool calibration_success = false;
    std::vector<float> received_cal_values;
    uint8_t received_band_size = 0;
    
    esp_err_t err = calibrator.start_calibration(nullptr, 0,
        [&](const uint8_t* band, uint8_t size, const std::vector<float>& values, bool success) {
            calibration_success = success;
            received_cal_values = values;
            received_band_size = size;
        });
    TEST_ASSERT_EQUAL(ESP_OK, err);
    
    // Feed all baseline packets
    const int pkt_size = csi_test_data::packet_size();
    for (int i = 0; i < 100 && i < num_baseline; i++) {
        calibrator.add_packet(baseline_packets[i], pkt_size);
    }
    
    // Verify calibration completed
    TEST_ASSERT_TRUE(calibration_success);
    TEST_ASSERT_EQUAL(16, received_band_size);  // PCA uses 16 subcarriers (every 4th)
    TEST_ASSERT_TRUE(received_cal_values.size() > 0);
    
    printf("PCA Calibration results:\n");
    printf("  Band size: %d subcarriers\n", received_band_size);
    printf("  Correlation values: %zu\n", received_cal_values.size());
    
    // All correlation values should be between 0 and 1
    for (float val : received_cal_values) {
        TEST_ASSERT_TRUE(val >= 0.0f && val <= 1.0f);
    }
}

void test_pca_calibrator_threshold_calculation(void) {
    if (!csi_test_data::load()) {
        TEST_IGNORE_MESSAGE("Failed to load test data");
        return;
    }
    
    MockDetector detector;
    CSIManager csi_manager;
    csi_manager.init(&detector, get_optimal_subcarriers(), 100, GainLockMode::DISABLED, &g_wifi_mock);
    
    PCACalibrator calibrator;
    calibrator.init(&csi_manager);
    calibrator.set_buffer_size(200);
    
    bool calibration_success = false;
    std::vector<float> received_cal_values;
    
    esp_err_t err = calibrator.start_calibration(nullptr, 0,
        [&](const uint8_t* band, uint8_t size, const std::vector<float>& values, bool success) {
            calibration_success = success;
            received_cal_values = values;
        });
    TEST_ASSERT_EQUAL(ESP_OK, err);
    
    // Feed baseline packets
    const int pkt_size = csi_test_data::packet_size();
    for (int i = 0; i < 200 && i < num_baseline; i++) {
        calibrator.add_packet(baseline_packets[i], pkt_size);
    }
    
    TEST_ASSERT_TRUE(calibration_success);
    TEST_ASSERT_TRUE(received_cal_values.size() > 0);
    
    // Calculate threshold using the formula from threshold.h
    // PCA: threshold = (1 - min(correlation)) * PCA_SCALE
    float min_corr = *std::min_element(received_cal_values.begin(), received_cal_values.end());
    float threshold = (1.0f - min_corr) * PCA_SCALE;
    
    printf("PCA Threshold calculation:\n");
    printf("  Min correlation: %.4f\n", min_corr);
    printf("  Threshold ((1 - min_corr) * %.0f): %.4f\n", PCA_SCALE, threshold);
    
    // Threshold should be reasonable (scaled values, typically 0.5-5.0 for stable baseline)
    TEST_ASSERT_TRUE(threshold > 0.0f);
    TEST_ASSERT_TRUE(threshold < 10.0f);
}

// ============================================================================
// Collection Complete Callback Test
// ============================================================================

void test_pca_calibrator_collection_complete_callback(void) {
    MockDetector detector;
    CSIManager csi_manager;
    csi_manager.init(&detector, get_optimal_subcarriers(), 100, GainLockMode::DISABLED, &g_wifi_mock);
    
    PCACalibrator calibrator;
    calibrator.init(&csi_manager);
    calibrator.set_buffer_size(50);  // Very small for quick test
    
    bool collection_complete_called = false;
    calibrator.set_collection_complete_callback([&]() {
        collection_complete_called = true;
    });
    
    if (!csi_test_data::load()) {
        TEST_IGNORE_MESSAGE("Failed to load test data");
        return;
    }
    
    esp_err_t err = calibrator.start_calibration(nullptr, 0,
        [](const uint8_t*, uint8_t, const std::vector<float>&, bool) {});
    TEST_ASSERT_EQUAL(ESP_OK, err);
    
    // Feed packets until calibration completes
    const int pkt_size = csi_test_data::packet_size();
    for (int i = 0; i < 50 && i < num_baseline; i++) {
        calibrator.add_packet(baseline_packets[i], pkt_size);
    }
    
    TEST_ASSERT_TRUE(collection_complete_called);
}

// ============================================================================
// Test Runner
// ============================================================================

void setUp(void) {}
void tearDown(void) {}

int main(int argc, char **argv) {
    UNITY_BEGIN();
    
    // Basic initialization
    RUN_TEST(test_pca_calibrator_constructor);
    RUN_TEST(test_pca_calibrator_init);
    RUN_TEST(test_pca_calibrator_set_buffer_size);
    
    // Calibration start
    RUN_TEST(test_pca_calibrator_start_without_init);
    RUN_TEST(test_pca_calibrator_start_success);
    RUN_TEST(test_pca_calibrator_start_while_calibrating);
    
    // Packet processing
    RUN_TEST(test_pca_calibrator_add_packet_not_calibrating);
    RUN_TEST(test_pca_calibrator_add_packet_increments);
    
    // Full calibration
    RUN_TEST(test_pca_calibrator_full_calibration);
    RUN_TEST(test_pca_calibrator_threshold_calculation);
    
    // Callbacks
    RUN_TEST(test_pca_calibrator_collection_complete_callback);
    
    return UNITY_END();
}
