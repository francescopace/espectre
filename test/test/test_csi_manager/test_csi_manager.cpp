/*
 * ESPectre - CSIManager Unit Tests
 *
 * Tests the CSIManager class functionality with IDetector interface.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include <unity.h>
#include <cstdint>
#include <cstring>
#include "csi_manager.h"
#include "mvs_detector.h"
#include "wifi_csi_interface.h"
#include "esphome/core/log.h"
#include "esp_wifi.h"

using namespace esphome::espectre;

static const char *TAG = "test_csi_manager";

// Test subcarrier selection
static const uint8_t TEST_SUBCARRIERS[12] = {11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22};

/**
 * Mock WiFi CSI for testing
 */
class WiFiCSIMock : public IWiFiCSI {
 public:
  esp_err_t set_csi_config(const wifi_csi_config_t* config) override {
    (void)config;
    return config_error_;
  }
  esp_err_t set_csi_rx_cb(wifi_csi_cb_t cb, void* ctx) override {
    callback_ = cb;
    callback_ctx_ = ctx;
    return callback_error_;
  }
  esp_err_t set_csi(bool enable) override {
    if (csi_error_ != ESP_OK) return csi_error_;
    enabled_ = enable;
    return ESP_OK;
  }
  bool is_enabled() const { return enabled_; }
  
  void set_config_error(esp_err_t err) { config_error_ = err; }
  void set_callback_error(esp_err_t err) { callback_error_ = err; }
  void set_csi_error(esp_err_t err) { csi_error_ = err; }
  void reset_errors() { config_error_ = ESP_OK; callback_error_ = ESP_OK; csi_error_ = ESP_OK; }
  
  void trigger_callback(wifi_csi_info_t* data) {
    if (callback_ && callback_ctx_) {
      callback_(callback_ctx_, data);
    }
  }
  
 private:
  bool enabled_{false};
  esp_err_t config_error_{ESP_OK};
  esp_err_t callback_error_{ESP_OK};
  esp_err_t csi_error_{ESP_OK};
  wifi_csi_cb_t callback_{nullptr};
  void* callback_ctx_{nullptr};
};

static WiFiCSIMock g_wifi_mock;

void setUp(void) {
    g_wifi_mock.reset_errors();
}

void tearDown(void) {
}

// ============================================================================
// INITIALIZATION TESTS
// ============================================================================

void test_csi_manager_init(void) {
    MVSDetector detector(50, 1.0f);
    CSIManager manager;
    
    manager.init(&detector, TEST_SUBCARRIERS, 100, GainLockMode::DISABLED, &g_wifi_mock);
    
    TEST_ASSERT_FALSE(manager.is_enabled());
    TEST_ASSERT_NOT_NULL(manager.get_detector());
}

// ============================================================================
// ENABLE/DISABLE TESTS
// ============================================================================

void test_csi_manager_enable(void) {
    MVSDetector detector(50, 1.0f);
    CSIManager manager;
    manager.init(&detector, TEST_SUBCARRIERS, 100, GainLockMode::DISABLED, &g_wifi_mock);
    
    esp_err_t err = manager.enable();
    
    TEST_ASSERT_EQUAL(ESP_OK, err);
    TEST_ASSERT_TRUE(manager.is_enabled());
    TEST_ASSERT_TRUE(g_wifi_mock.is_enabled());
}

void test_csi_manager_enable_twice_returns_ok(void) {
    MVSDetector detector(50, 1.0f);
    CSIManager manager;
    manager.init(&detector, TEST_SUBCARRIERS, 100, GainLockMode::DISABLED, &g_wifi_mock);
    
    manager.enable();
    esp_err_t err = manager.enable();
    
    TEST_ASSERT_EQUAL(ESP_OK, err);
    TEST_ASSERT_TRUE(manager.is_enabled());
}

void test_csi_manager_disable(void) {
    MVSDetector detector(50, 1.0f);
    CSIManager manager;
    manager.init(&detector, TEST_SUBCARRIERS, 100, GainLockMode::DISABLED, &g_wifi_mock);
    
    manager.enable();
    esp_err_t err = manager.disable();
    
    TEST_ASSERT_EQUAL(ESP_OK, err);
    TEST_ASSERT_FALSE(manager.is_enabled());
}

void test_csi_manager_disable_when_not_enabled(void) {
    MVSDetector detector(50, 1.0f);
    CSIManager manager;
    manager.init(&detector, TEST_SUBCARRIERS, 100, GainLockMode::DISABLED, &g_wifi_mock);
    
    esp_err_t err = manager.disable();
    
    TEST_ASSERT_EQUAL(ESP_OK, err);
    TEST_ASSERT_FALSE(manager.is_enabled());
}

// ============================================================================
// THRESHOLD TESTS
// ============================================================================

void test_csi_manager_set_threshold(void) {
    MVSDetector detector(50, 1.0f);
    CSIManager manager;
    manager.init(&detector, TEST_SUBCARRIERS, 100, GainLockMode::DISABLED, &g_wifi_mock);
    
    manager.set_threshold(2.5f);
    
    TEST_ASSERT_EQUAL_FLOAT(2.5f, detector.get_threshold());
}

// ============================================================================
// SUBCARRIER SELECTION TESTS
// ============================================================================

void test_csi_manager_update_subcarrier_selection(void) {
    MVSDetector detector(50, 1.0f);
    CSIManager manager;
    manager.init(&detector, TEST_SUBCARRIERS, 100, GainLockMode::DISABLED, &g_wifi_mock);
    
    uint8_t new_subcarriers[12] = {20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};
    manager.update_subcarrier_selection(new_subcarriers);
    
    TEST_PASS();
}

// ============================================================================
// PROCESS PACKET TESTS
// ============================================================================

void test_csi_manager_process_packet_null_data(void) {
    MVSDetector detector(50, 1.0f);
    CSIManager manager;
    manager.init(&detector, TEST_SUBCARRIERS, 100, GainLockMode::DISABLED, &g_wifi_mock);
    
    manager.process_packet(nullptr);
    
    TEST_ASSERT_EQUAL(MotionState::IDLE, detector.get_state());
}

void test_csi_manager_process_packet_short_data(void) {
    MVSDetector detector(50, 1.0f);
    CSIManager manager;
    manager.init(&detector, TEST_SUBCARRIERS, 100, GainLockMode::DISABLED, &g_wifi_mock);
    
    wifi_csi_info_t csi_info = {};
    int8_t short_buf[5] = {0};
    csi_info.buf = short_buf;
    csi_info.len = 5;
    
    manager.process_packet(&csi_info);
    
    TEST_ASSERT_EQUAL(MotionState::IDLE, detector.get_state());
}

void test_csi_manager_process_packet_valid_data(void) {
    MVSDetector detector(50, 1.0f);
    CSIManager manager;
    manager.init(&detector, TEST_SUBCARRIERS, 100, GainLockMode::DISABLED, &g_wifi_mock);
    
    // Create valid CSI data (128 bytes for HT20)
    int8_t csi_buf[128];
    for (int i = 0; i < 128; i++) {
        csi_buf[i] = (int8_t)(i % 64 - 32);
    }
    
    wifi_csi_info_t csi_info = {};
    csi_info.buf = csi_buf;
    csi_info.len = 128;
    csi_info.rx_ctrl.channel = 6;
    
    manager.process_packet(&csi_info);
    
    TEST_ASSERT_EQUAL(1, detector.get_total_packets());
}

// ============================================================================
// ERROR PATH TESTS
// ============================================================================

void test_csi_manager_enable_config_error(void) {
    MVSDetector detector(50, 1.0f);
    CSIManager manager;
    manager.init(&detector, TEST_SUBCARRIERS, 100, GainLockMode::DISABLED, &g_wifi_mock);
    
    g_wifi_mock.set_config_error(ESP_ERR_INVALID_ARG);
    
    esp_err_t result = manager.enable(nullptr);
    
    TEST_ASSERT_EQUAL(ESP_ERR_INVALID_ARG, result);
    TEST_ASSERT_FALSE(manager.is_enabled());
}

void test_csi_manager_enable_callback_error(void) {
    MVSDetector detector(50, 1.0f);
    CSIManager manager;
    manager.init(&detector, TEST_SUBCARRIERS, 100, GainLockMode::DISABLED, &g_wifi_mock);
    
    g_wifi_mock.set_callback_error(ESP_ERR_NO_MEM);
    
    esp_err_t result = manager.enable(nullptr);
    
    TEST_ASSERT_EQUAL(ESP_ERR_NO_MEM, result);
    TEST_ASSERT_FALSE(manager.is_enabled());
}

void test_csi_manager_enable_csi_error(void) {
    MVSDetector detector(50, 1.0f);
    CSIManager manager;
    manager.init(&detector, TEST_SUBCARRIERS, 100, GainLockMode::DISABLED, &g_wifi_mock);
    
    g_wifi_mock.set_csi_error(ESP_FAIL);
    
    esp_err_t result = manager.enable(nullptr);
    
    TEST_ASSERT_EQUAL(ESP_FAIL, result);
    TEST_ASSERT_FALSE(manager.is_enabled());
}

void test_csi_manager_disable_error(void) {
    MVSDetector detector(50, 1.0f);
    CSIManager manager;
    manager.init(&detector, TEST_SUBCARRIERS, 100, GainLockMode::DISABLED, &g_wifi_mock);
    
    manager.enable(nullptr);
    g_wifi_mock.set_csi_error(ESP_FAIL);
    
    esp_err_t result = manager.disable();
    
    TEST_ASSERT_EQUAL(ESP_FAIL, result);
    TEST_ASSERT_TRUE(manager.is_enabled());
}

// ============================================================================
// CALLBACK WRAPPER TESTS
// ============================================================================

void test_csi_manager_callback_wrapper_triggered(void) {
    MVSDetector detector(50, 1.0f);
    CSIManager manager;
    manager.init(&detector, TEST_SUBCARRIERS, 100, GainLockMode::DISABLED, &g_wifi_mock);
    
    manager.enable(nullptr);
    
    int8_t csi_buf[128] = {0};
    wifi_csi_info_t csi_info = {};
    csi_info.buf = csi_buf;
    csi_info.len = 128;
    csi_info.rx_ctrl.channel = 6;
    
    g_wifi_mock.trigger_callback(&csi_info);
    
    TEST_ASSERT_TRUE(detector.get_total_packets() > 0);
}

void test_csi_manager_callback_wrapper_null_data(void) {
    MVSDetector detector(50, 1.0f);
    CSIManager manager;
    manager.init(&detector, TEST_SUBCARRIERS, 100, GainLockMode::DISABLED, &g_wifi_mock);
    
    manager.enable(nullptr);
    
    uint32_t packets_before = detector.get_total_packets();
    
    g_wifi_mock.trigger_callback(nullptr);
    
    TEST_ASSERT_EQUAL(packets_before, detector.get_total_packets());
}

// ============================================================================
// CLEAR DETECTOR BUFFER TEST
// ============================================================================

void test_csi_manager_clear_detector_buffer(void) {
    MVSDetector detector(50, 1.0f);
    CSIManager manager;
    manager.init(&detector, TEST_SUBCARRIERS, 100, GainLockMode::DISABLED, &g_wifi_mock);
    
    // Process some packets
    int8_t csi_buf[128] = {0};
    wifi_csi_info_t csi_info = {};
    csi_info.buf = csi_buf;
    csi_info.len = 128;
    csi_info.rx_ctrl.channel = 6;
    
    for (int i = 0; i < 10; i++) {
        manager.process_packet(&csi_info);
    }
    
    // Clear buffer
    manager.clear_detector_buffer();
    
    // Detector should be reset
    TEST_ASSERT_EQUAL_FLOAT(0.0f, detector.get_motion_metric());
}

// ============================================================================
// GAIN LOCK TESTS
// ============================================================================

void test_csi_manager_gain_lock_disabled(void) {
    MVSDetector detector(50, 1.0f);
    CSIManager manager;
    manager.init(&detector, TEST_SUBCARRIERS, 100, GainLockMode::DISABLED, &g_wifi_mock);
    
    // With DISABLED, gain is immediately locked
    TEST_ASSERT_TRUE(manager.is_gain_locked());
}

void test_csi_manager_get_gain_controller(void) {
    MVSDetector detector(50, 1.0f);
    CSIManager manager;
    manager.init(&detector, TEST_SUBCARRIERS, 100, GainLockMode::DISABLED, &g_wifi_mock);
    
    const GainController& gc = manager.get_gain_controller();
    TEST_ASSERT_TRUE(gc.is_locked());
}

// ============================================================================
// ENTRY POINT
// ============================================================================

int process(void) {
    UNITY_BEGIN();
    
    // Initialization tests
    RUN_TEST(test_csi_manager_init);
    
    // Enable/Disable tests
    RUN_TEST(test_csi_manager_enable);
    RUN_TEST(test_csi_manager_enable_twice_returns_ok);
    RUN_TEST(test_csi_manager_disable);
    RUN_TEST(test_csi_manager_disable_when_not_enabled);
    
    // Threshold tests
    RUN_TEST(test_csi_manager_set_threshold);
    
    // Subcarrier selection tests
    RUN_TEST(test_csi_manager_update_subcarrier_selection);
    
    // Process packet tests
    RUN_TEST(test_csi_manager_process_packet_null_data);
    RUN_TEST(test_csi_manager_process_packet_short_data);
    RUN_TEST(test_csi_manager_process_packet_valid_data);
    
    // Error path tests
    RUN_TEST(test_csi_manager_enable_config_error);
    RUN_TEST(test_csi_manager_enable_callback_error);
    RUN_TEST(test_csi_manager_enable_csi_error);
    RUN_TEST(test_csi_manager_disable_error);
    
    // Callback wrapper tests
    RUN_TEST(test_csi_manager_callback_wrapper_triggered);
    RUN_TEST(test_csi_manager_callback_wrapper_null_data);
    
    // Clear buffer test
    RUN_TEST(test_csi_manager_clear_detector_buffer);
    
    // Gain lock tests
    RUN_TEST(test_csi_manager_gain_lock_disabled);
    RUN_TEST(test_csi_manager_get_gain_controller);
    
    return UNITY_END();
}

#if defined(ESP_PLATFORM)
extern "C" void app_main(void) { process(); }
#else
int main(int argc, char **argv) { return process(); }
#endif
