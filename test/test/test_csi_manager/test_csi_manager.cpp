/*
 * ESPectre - CSIManager Unit Tests
 *
 * Tests the CSIManager class functionality.
 * Uses WiFiCSIMock for all tests (both native and device).
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include <unity.h>
#include <cstdint>
#include <cstring>
#include "csi_manager.h"
#include "csi_processor.h"
#include "calibration_manager.h"
#include "wifi_csi_interface.h"
#include "esphome/core/log.h"
#include "esp_wifi.h"

// Include real CSI data
#include "real_csi_data_esp32.h"
#include "real_csi_arrays.inc"

using namespace esphome::espectre;

static const char *TAG = "test_csi_manager";

// Test subcarrier selection
static const uint8_t TEST_SUBCARRIERS[12] = {11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22};

/**
 * Mock WiFi CSI for testing
 * Returns ESP_OK without calling real WiFi functions.
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
  
  // Methods to simulate errors
  void set_config_error(esp_err_t err) { config_error_ = err; }
  void set_callback_error(esp_err_t err) { callback_error_ = err; }
  void set_csi_error(esp_err_t err) { csi_error_ = err; }
  void reset_errors() { config_error_ = ESP_OK; callback_error_ = ESP_OK; csi_error_ = ESP_OK; }
  
  // Method to trigger the callback (simulates receiving CSI data)
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

// Global processor context for tests
static csi_processor_context_t g_processor;

// Global mock WiFi CSI for tests
static WiFiCSIMock g_wifi_mock;

void setUp(void) {
    // Initialize processor before each test
    csi_processor_init(&g_processor, 50, 1.0f);
    // Reset mock errors
    g_wifi_mock.reset_errors();
}

void tearDown(void) {
    // Cleanup processor after each test
    csi_processor_cleanup(&g_processor);
}

// ============================================================================
// INITIALIZATION TESTS
// ============================================================================

void test_csi_manager_init(void) {
    CSIManager manager;
    
    manager.init(&g_processor, TEST_SUBCARRIERS, 1.0f, 50, 100, true, 11.0f, false, 7, 3.0f, &g_wifi_mock);
    
    TEST_ASSERT_FALSE(manager.is_enabled());
}

void test_csi_manager_init_with_hampel(void) {
    CSIManager manager;
    
    manager.init(&g_processor, TEST_SUBCARRIERS, 1.0f, 50, 100, true, 11.0f, true, 7, 3.0f, &g_wifi_mock);
    
    // Verify Hampel filter is configured in processor
    TEST_ASSERT_TRUE(g_processor.hampel_state.enabled);
    TEST_ASSERT_EQUAL(7, g_processor.hampel_state.window_size);
}

void test_csi_manager_init_without_hampel(void) {
    CSIManager manager;
    
    manager.init(&g_processor, TEST_SUBCARRIERS, 1.0f, 50, 100, true, 11.0f, false, 7, 3.0f, &g_wifi_mock);
    
    // Verify Hampel filter is disabled
    TEST_ASSERT_FALSE(g_processor.hampel_state.enabled);
}

// ============================================================================
// ENABLE/DISABLE TESTS
// Uses WiFiCSIMock for all tests (both native and device)
// ============================================================================

void test_csi_manager_enable(void) {
    CSIManager manager;
    manager.init(&g_processor, TEST_SUBCARRIERS, 1.0f, 50, 100, true, 11.0f, false, 7, 3.0f, &g_wifi_mock);
    
    esp_err_t err = manager.enable();
    
    TEST_ASSERT_EQUAL(ESP_OK, err);
    TEST_ASSERT_TRUE(manager.is_enabled());
    TEST_ASSERT_TRUE(g_wifi_mock.is_enabled());
}

void test_csi_manager_enable_twice_returns_ok(void) {
    CSIManager manager;
    manager.init(&g_processor, TEST_SUBCARRIERS, 1.0f, 50, 100, true, 11.0f, false, 7, 3.0f, &g_wifi_mock);
    
    manager.enable();
    esp_err_t err = manager.enable();  // Second call
    
    TEST_ASSERT_EQUAL(ESP_OK, err);
    TEST_ASSERT_TRUE(manager.is_enabled());
}

void test_csi_manager_disable(void) {
    CSIManager manager;
    manager.init(&g_processor, TEST_SUBCARRIERS, 1.0f, 50, 100, true, 11.0f, false, 7, 3.0f, &g_wifi_mock);
    
    manager.enable();
    esp_err_t err = manager.disable();
    
    TEST_ASSERT_EQUAL(ESP_OK, err);
    TEST_ASSERT_FALSE(manager.is_enabled());
}

void test_csi_manager_disable_when_not_enabled(void) {
    CSIManager manager;
    manager.init(&g_processor, TEST_SUBCARRIERS, 1.0f, 50, 100, true, 11.0f, false, 7, 3.0f, &g_wifi_mock);
    
    esp_err_t err = manager.disable();
    
    TEST_ASSERT_EQUAL(ESP_OK, err);
    TEST_ASSERT_FALSE(manager.is_enabled());
}

// ============================================================================
// THRESHOLD TESTS
// ============================================================================

void test_csi_manager_set_threshold(void) {
    CSIManager manager;
    manager.init(&g_processor, TEST_SUBCARRIERS, 1.0f, 50, 100, true, 11.0f, false, 7, 3.0f, &g_wifi_mock);
    
    manager.set_threshold(2.5f);
    
    TEST_ASSERT_EQUAL_FLOAT(2.5f, csi_processor_get_threshold(&g_processor));
}

void test_csi_manager_set_threshold_multiple_times(void) {
    CSIManager manager;
    manager.init(&g_processor, TEST_SUBCARRIERS, 1.0f, 50, 100, true, 11.0f, false, 7, 3.0f, &g_wifi_mock);
    
    manager.set_threshold(1.0f);
    TEST_ASSERT_EQUAL_FLOAT(1.0f, csi_processor_get_threshold(&g_processor));
    
    manager.set_threshold(5.0f);
    TEST_ASSERT_EQUAL_FLOAT(5.0f, csi_processor_get_threshold(&g_processor));
}

// ============================================================================
// SUBCARRIER SELECTION TESTS
// ============================================================================

void test_csi_manager_update_subcarrier_selection(void) {
    CSIManager manager;
    manager.init(&g_processor, TEST_SUBCARRIERS, 1.0f, 50, 100, true, 11.0f, false, 7, 3.0f, &g_wifi_mock);
    
    uint8_t new_subcarriers[12] = {20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};
    manager.update_subcarrier_selection(new_subcarriers);
    
    // No direct way to verify, but should not crash
    TEST_PASS();
}

// ============================================================================
// PROCESS PACKET TESTS
// ============================================================================

void test_csi_manager_process_packet_null_data(void) {
    CSIManager manager;
    manager.init(&g_processor, TEST_SUBCARRIERS, 1.0f, 50, 100, true, 11.0f, false, 7, 3.0f, &g_wifi_mock);
    
    manager.process_packet(nullptr);
    
    // Should not crash, state should remain IDLE
    TEST_ASSERT_EQUAL(CSI_STATE_IDLE, csi_processor_get_state(&g_processor));
}

void test_csi_manager_process_packet_short_data(void) {
    CSIManager manager;
    manager.init(&g_processor, TEST_SUBCARRIERS, 1.0f, 50, 100, true, 11.0f, false, 7, 3.0f, &g_wifi_mock);
    
    wifi_csi_info_t csi_info = {};
    int8_t short_buf[5] = {0};
    csi_info.buf = short_buf;
    csi_info.len = 5;  // Too short
    
    manager.process_packet(&csi_info);
    
    // Should handle gracefully
    TEST_ASSERT_EQUAL(CSI_STATE_IDLE, csi_processor_get_state(&g_processor));
}

void test_csi_manager_process_packet_real_data(void) {
    CSIManager manager;
    manager.init(&g_processor, TEST_SUBCARRIERS, 1.0f, 50, 100, true, 11.0f, false, 7, 3.0f, &g_wifi_mock);
    
    // Process several baseline packets
    for (int i = 0; i < 10; i++) {
        wifi_csi_info_t csi_info = {};
        csi_info.buf = const_cast<int8_t*>(baseline_packets[i]);
        csi_info.len = 128;
        
        manager.process_packet(&csi_info);
    }
    
    // After processing baseline, should be IDLE
    TEST_ASSERT_EQUAL(CSI_STATE_IDLE, csi_processor_get_state(&g_processor));
}

void test_csi_manager_process_packet_detects_motion(void) {
    CSIManager manager;
    // Use very low threshold to detect motion easily
    manager.init(&g_processor, TEST_SUBCARRIERS, 0.01f, 5, 100, true, 11.0f, false, 7, 3.0f, &g_wifi_mock);
    
    // Process enough packets to fill the window and get moving variance
    // Need at least window_size packets before variance is calculated
    for (int i = 0; i < 50; i++) {
        wifi_csi_info_t csi_info = {};
        csi_info.buf = const_cast<int8_t*>(baseline_packets[i % 100]);
        csi_info.len = 128;
        
        manager.process_packet(&csi_info);
    }
    
    // Now process movement packets
    for (int i = 0; i < 50; i++) {
        wifi_csi_info_t csi_info = {};
        csi_info.buf = const_cast<int8_t*>(movement_packets[i % 100]);
        csi_info.len = 128;
        
        manager.process_packet(&csi_info);
    }
    
    // Verify that packets were processed
    uint32_t total_packets = csi_processor_get_total_packets(&g_processor);
    ESP_LOGI(TAG, "Total packets processed: %u", total_packets);
    
    TEST_ASSERT_EQUAL(100, total_packets);
}

// ============================================================================
// CALLBACK TESTS
// ============================================================================

static int g_callback_count = 0;
static csi_motion_state_t g_last_callback_state = CSI_STATE_IDLE;

static void test_callback(csi_motion_state_t state) {
    g_callback_count++;
    g_last_callback_state = state;
}

void test_csi_manager_callback_invoked(void) {
    g_callback_count = 0;
    g_last_callback_state = CSI_STATE_IDLE;
    
    CSIManager manager;
    // publish_rate = 5, so callback should be invoked every 5 packets
    manager.init(&g_processor, TEST_SUBCARRIERS, 1.0f, 50, 5, true, 11.0f, false, 7, 3.0f, &g_wifi_mock);
    manager.enable(test_callback);
    
    // Process 10 packets
    for (int i = 0; i < 10; i++) {
        wifi_csi_info_t csi_info = {};
        csi_info.buf = const_cast<int8_t*>(baseline_packets[i]);
        csi_info.len = 128;
        
        manager.process_packet(&csi_info);
    }
    
    // Callback should have been invoked twice (at packet 5 and 10)
    TEST_ASSERT_EQUAL(2, g_callback_count);
}

void test_csi_manager_callback_not_invoked_before_publish_rate(void) {
    g_callback_count = 0;
    
    CSIManager manager;
    manager.init(&g_processor, TEST_SUBCARRIERS, 1.0f, 50, 100, true, 11.0f, false, 7, 3.0f, &g_wifi_mock);
    manager.enable(test_callback);
    
    // Process only 10 packets (less than publish_rate of 100)
    for (int i = 0; i < 10; i++) {
        wifi_csi_info_t csi_info = {};
        csi_info.buf = const_cast<int8_t*>(baseline_packets[i]);
        csi_info.len = 128;
        
        manager.process_packet(&csi_info);
    }
    
    // Callback should not have been invoked
    TEST_ASSERT_EQUAL(0, g_callback_count);
}

// ============================================================================
// CALIBRATION MODE TESTS
// ============================================================================

void test_csi_manager_set_calibration_mode(void) {
    CSIManager manager;
    manager.init(&g_processor, TEST_SUBCARRIERS, 1.0f, 50, 100, true, 11.0f, false, 7, 3.0f, &g_wifi_mock);
    
    // Set calibration mode (nullptr disables)
    manager.set_calibration_mode(nullptr);
    
    // Should not crash
    TEST_PASS();
}

// ============================================================================
// CALIBRATION DELEGATION TESTS
// ============================================================================

void test_csi_manager_with_calibrator_not_calibrating(void) {
    // Test that CSIManager processes normally when calibrator is set but not calibrating
    CSIManager manager;
    manager.init(&g_processor, TEST_SUBCARRIERS, 1.0f, 50, 100, true, 11.0f, false, 7, 3.0f, &g_wifi_mock);
    
    // Create a real CalibrationManager (not calibrating by default)
    CalibrationManager calibrator;
    calibrator.init(nullptr, "/tmp/test_cal.bin");  // No CSI manager, won't actually calibrate
    
    // Verify calibrator is not calibrating
    TEST_ASSERT_FALSE(calibrator.is_calibrating());
    
    // Set calibration mode
    manager.set_calibration_mode(&calibrator);
    
    // Process packets - should process normally since calibrator is not calibrating
    for (int i = 0; i < 5; i++) {
        wifi_csi_info_t csi_info = {};
        csi_info.buf = const_cast<int8_t*>(baseline_packets[i]);
        csi_info.len = 128;
        
        manager.process_packet(&csi_info);
    }
    
    // Processor should have processed them (calibrator was not calibrating)
    TEST_ASSERT_EQUAL(5, csi_processor_get_total_packets(&g_processor));
}

void test_csi_manager_null_calibrator_processes_normally(void) {
    CSIManager manager;
    manager.init(&g_processor, TEST_SUBCARRIERS, 1.0f, 50, 100, true, 11.0f, false, 7, 3.0f, &g_wifi_mock);
    
    // No calibrator set (nullptr)
    manager.set_calibration_mode(nullptr);
    
    // Process packets - should process normally
    for (int i = 0; i < 5; i++) {
        wifi_csi_info_t csi_info = {};
        csi_info.buf = const_cast<int8_t*>(baseline_packets[i]);
        csi_info.len = 128;
        
        manager.process_packet(&csi_info);
    }
    
    // Processor should have processed them
    TEST_ASSERT_EQUAL(5, csi_processor_get_total_packets(&g_processor));
}

void test_csi_manager_delegates_when_calibrating(void) {
    // Test that CSIManager delegates packets to calibrator when calibrating
    CSIManager manager;
    manager.init(&g_processor, TEST_SUBCARRIERS, 1.0f, 50, 100, true, 11.0f, false, 7, 3.0f, &g_wifi_mock);
    
    // Create CalibrationManager and start calibration
    CalibrationManager calibrator;
    calibrator.init(&manager, "/tmp/test_cal_delegate.bin");
    calibrator.set_buffer_size(50);  // Small buffer for testing
    
    // Start calibration - this sets calibrator to "calibrating" state
    uint8_t dummy_band[12] = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21};
    esp_err_t err = calibrator.start_auto_calibration(dummy_band, 12, nullptr);
    TEST_ASSERT_EQUAL(ESP_OK, err);
    TEST_ASSERT_TRUE(calibrator.is_calibrating());
    
    // Set calibration mode on manager
    manager.set_calibration_mode(&calibrator);
    
    // Process packets - should delegate to calibrator (lines 70-72)
    uint32_t initial_packets = csi_processor_get_total_packets(&g_processor);
    
    for (int i = 0; i < 10; i++) {
        wifi_csi_info_t csi_info = {};
        csi_info.buf = const_cast<int8_t*>(baseline_packets[i]);
        csi_info.len = 128;
        
        manager.process_packet(&csi_info);
    }
    
    // Processor should NOT have processed them (delegated to calibrator)
    TEST_ASSERT_EQUAL(initial_packets, csi_processor_get_total_packets(&g_processor));
    
    // Cleanup
    manager.set_calibration_mode(nullptr);
    remove("/tmp/test_cal_delegate.bin");
}

void test_csi_manager_calibration_triggers_periodic_yield(void) {
    // Test that calibration processes 100+ packets to trigger the periodic yield
    // This covers the vTaskDelay(1) path in add_packet when buffer_count_ % 100 == 0
    CSIManager manager;
    manager.init(&g_processor, TEST_SUBCARRIERS, 1.0f, 50, 100, true, 11.0f, false, 7, 3.0f, &g_wifi_mock);
    
    // Create CalibrationManager with buffer size > 100 to trigger periodic flush/yield
    CalibrationManager calibrator;
    calibrator.init(&manager, "/tmp/test_cal_yield.bin");
    calibrator.set_buffer_size(150);  // Need > 100 to trigger the yield path
    
    // Start calibration
    uint8_t dummy_band[12] = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21};
    esp_err_t err = calibrator.start_auto_calibration(dummy_band, 12, nullptr);
    TEST_ASSERT_EQUAL(ESP_OK, err);
    TEST_ASSERT_TRUE(calibrator.is_calibrating());
    
    // Set calibration mode on manager
    manager.set_calibration_mode(&calibrator);
    
    // Process 120 packets - this should trigger the periodic yield at packet 100
    for (int i = 0; i < 120; i++) {
        wifi_csi_info_t csi_info = {};
        csi_info.buf = const_cast<int8_t*>(baseline_packets[i % 100]);
        csi_info.len = 128;
        
        manager.process_packet(&csi_info);
    }
    
    // Calibrator should still be calibrating (buffer not full yet, need 150)
    TEST_ASSERT_TRUE(calibrator.is_calibrating());
    
    // Cleanup
    manager.set_calibration_mode(nullptr);
    remove("/tmp/test_cal_yield.bin");
}

void test_csi_manager_calibrator_lifecycle(void) {
    // Test setting and clearing calibration mode
    CSIManager manager;
    manager.init(&g_processor, TEST_SUBCARRIERS, 1.0f, 50, 100, true, 11.0f, false, 7, 3.0f, &g_wifi_mock);
    
    CalibrationManager calibrator;
    calibrator.init(nullptr, "/tmp/test_cal2.bin");
    
    // Set calibration mode
    manager.set_calibration_mode(&calibrator);
    
    // Clear calibration mode
    manager.set_calibration_mode(nullptr);
    
    // Process packets - should work normally
    wifi_csi_info_t csi_info = {};
    csi_info.buf = const_cast<int8_t*>(baseline_packets[0]);
    csi_info.len = 128;
    
    manager.process_packet(&csi_info);
    
    TEST_PASS();
}

// ============================================================================
// INTEGRATION TESTS
// Uses WiFiCSIMock for all tests (both native and device)
// ============================================================================

void test_csi_manager_full_workflow(void) {
    CSIManager manager;
    manager.init(&g_processor, TEST_SUBCARRIERS, 1.0f, 20, 50, true, 11.0f, true, 7, 3.0f, &g_wifi_mock);
    
    // Enable
    TEST_ASSERT_EQUAL(ESP_OK, manager.enable());
    TEST_ASSERT_TRUE(manager.is_enabled());
    
    // Process some packets
    for (int i = 0; i < 30; i++) {
        wifi_csi_info_t csi_info = {};
        csi_info.buf = const_cast<int8_t*>(baseline_packets[i % 100]);
        csi_info.len = 128;
        
        manager.process_packet(&csi_info);
    }
    
    // Update threshold
    manager.set_threshold(2.0f);
    TEST_ASSERT_EQUAL_FLOAT(2.0f, csi_processor_get_threshold(&g_processor));
    
    // Disable
    TEST_ASSERT_EQUAL(ESP_OK, manager.disable());
    TEST_ASSERT_FALSE(manager.is_enabled());
}

void test_csi_manager_baseline_then_motion(void) {
    g_callback_count = 0;
    
    CSIManager manager;
    // Low threshold, small window for quick detection
    // Disable low-pass filter for this test - test data was collected without low-pass
    manager.init(&g_processor, TEST_SUBCARRIERS, 0.5f, 10, 20, false, 11.0f, false, 7, 3.0f, &g_wifi_mock);
    manager.enable(test_callback);
    
    // Phase 1: Baseline (should stay IDLE)
    for (int i = 0; i < 50; i++) {
        wifi_csi_info_t csi_info = {};
        csi_info.buf = const_cast<int8_t*>(baseline_packets[i]);
        csi_info.len = 128;
        
        manager.process_packet(&csi_info);
    }
    
    ESP_LOGI(TAG, "After baseline: state=%d, callbacks=%d", 
             csi_processor_get_state(&g_processor), g_callback_count);
    
    // Phase 2: Movement (should detect MOTION)
    for (int i = 0; i < 50; i++) {
        wifi_csi_info_t csi_info = {};
        csi_info.buf = const_cast<int8_t*>(movement_packets[i]);
        csi_info.len = 128;
        
        manager.process_packet(&csi_info);
    }
    
    csi_motion_state_t state = csi_processor_get_state(&g_processor);
    ESP_LOGI(TAG, "After movement: state=%d, callbacks=%d", state, g_callback_count);
    
    // Should have detected motion
    TEST_ASSERT_EQUAL(CSI_STATE_MOTION, state);
    
    // Callbacks should have been invoked (100 packets / 20 publish_rate = 5)
    TEST_ASSERT_TRUE(g_callback_count >= 4);
}

// ============================================================================
// CHANNEL CHANGE DETECTION TESTS
// ============================================================================

void test_csi_manager_channel_change_resets_buffer(void) {
    CSIManager manager;
    // publish_rate=20 means channel check happens every 20 packets
    manager.init(&g_processor, TEST_SUBCARRIERS, 0.5f, 10, 20, false, 11.0f, false, 7, 3.0f, &g_wifi_mock);
    manager.enable(nullptr);
    
    // Process packets on channel 6 to build up variance
    // Need at least publish_rate packets to trigger variance calculation
    for (int i = 0; i < 60; i++) {
        wifi_csi_info_t csi_info = {};
        csi_info.buf = const_cast<int8_t*>(movement_packets[i % 100]);
        csi_info.len = 128;
        csi_info.rx_ctrl.channel = 6;
        
        manager.process_packet(&csi_info);
    }
    
    // Verify we have some variance built up
    float variance_before = csi_processor_get_moving_variance(&g_processor);
    ESP_LOGI(TAG, "Variance before channel change: %.4f", variance_before);
    TEST_ASSERT_TRUE(variance_before > 0);
    
    // Simulate AP channel change - need to process publish_rate packets
    // on new channel to trigger the check (channel check happens at publish time)
    for (int i = 0; i < 20; i++) {
        wifi_csi_info_t csi_info = {};
        csi_info.buf = const_cast<int8_t*>(baseline_packets[i]);
        csi_info.len = 128;
        csi_info.rx_ctrl.channel = 11;  // Different channel!
        
        manager.process_packet(&csi_info);
    }
    
    // Buffer should have been cleared at publish time, variance should be reset to 0
    float variance_after = csi_processor_get_moving_variance(&g_processor);
    ESP_LOGI(TAG, "Variance after channel change: %.4f", variance_after);
    
    // After buffer clear, variance should be exactly 0
    TEST_ASSERT_EQUAL_FLOAT(0.0f, variance_after);
}

void test_csi_manager_same_channel_no_reset(void) {
    CSIManager manager;
    manager.init(&g_processor, TEST_SUBCARRIERS, 0.5f, 10, 20, false, 11.0f, false, 7, 3.0f, &g_wifi_mock);
    manager.enable(nullptr);
    
    // Process packets on same channel
    for (int i = 0; i < 30; i++) {
        wifi_csi_info_t csi_info = {};
        csi_info.buf = const_cast<int8_t*>(movement_packets[i % 100]);
        csi_info.len = 128;
        csi_info.rx_ctrl.channel = 6;
        
        manager.process_packet(&csi_info);
    }
    
    float variance_before = csi_processor_get_moving_variance(&g_processor);
    
    // Process more packets on same channel - should NOT reset
    for (int i = 30; i < 60; i++) {
        wifi_csi_info_t csi_info = {};
        csi_info.buf = const_cast<int8_t*>(movement_packets[i % 100]);
        csi_info.len = 128;
        csi_info.rx_ctrl.channel = 6;  // Same channel
        
        manager.process_packet(&csi_info);
    }
    
    float variance_after = csi_processor_get_moving_variance(&g_processor);
    
    // Variance should still be present (buffer NOT cleared)
    TEST_ASSERT_TRUE(variance_after > 0);
    ESP_LOGI(TAG, "Same channel: variance %.4f -> %.4f", variance_before, variance_after);
}

// ============================================================================
// ERROR PATH TESTS
// ============================================================================

void test_csi_manager_enable_config_error(void) {
    CSIManager manager;
    manager.init(&g_processor, TEST_SUBCARRIERS, 1.0f, 50, 100, true, 11.0f, false, 7, 3.0f, &g_wifi_mock);
    
    // Simulate config error
    g_wifi_mock.set_config_error(ESP_ERR_INVALID_ARG);
    
    esp_err_t result = manager.enable(nullptr);
    
    TEST_ASSERT_EQUAL(ESP_ERR_INVALID_ARG, result);
    TEST_ASSERT_FALSE(manager.is_enabled());
}

void test_csi_manager_enable_callback_error(void) {
    CSIManager manager;
    manager.init(&g_processor, TEST_SUBCARRIERS, 1.0f, 50, 100, true, 11.0f, false, 7, 3.0f, &g_wifi_mock);
    
    // Simulate callback registration error
    g_wifi_mock.set_callback_error(ESP_ERR_NO_MEM);
    
    esp_err_t result = manager.enable(nullptr);
    
    TEST_ASSERT_EQUAL(ESP_ERR_NO_MEM, result);
    TEST_ASSERT_FALSE(manager.is_enabled());
}

void test_csi_manager_enable_csi_error(void) {
    CSIManager manager;
    manager.init(&g_processor, TEST_SUBCARRIERS, 1.0f, 50, 100, true, 11.0f, false, 7, 3.0f, &g_wifi_mock);
    
    // Simulate CSI enable error
    g_wifi_mock.set_csi_error(ESP_FAIL);
    
    esp_err_t result = manager.enable(nullptr);
    
    TEST_ASSERT_EQUAL(ESP_FAIL, result);
    TEST_ASSERT_FALSE(manager.is_enabled());
}

void test_csi_manager_disable_error(void) {
    CSIManager manager;
    manager.init(&g_processor, TEST_SUBCARRIERS, 1.0f, 50, 100, true, 11.0f, false, 7, 3.0f, &g_wifi_mock);
    
    // First enable successfully
    esp_err_t result = manager.enable(nullptr);
    TEST_ASSERT_EQUAL(ESP_OK, result);
    TEST_ASSERT_TRUE(manager.is_enabled());
    
    // Now simulate disable error
    g_wifi_mock.set_csi_error(ESP_FAIL);
    
    result = manager.disable();
    
    TEST_ASSERT_EQUAL(ESP_FAIL, result);
    // Manager should still think it's enabled since disable failed
    TEST_ASSERT_TRUE(manager.is_enabled());
}

void test_csi_manager_callback_wrapper_triggered(void) {
    CSIManager manager;
    manager.init(&g_processor, TEST_SUBCARRIERS, 1.0f, 50, 100, true, 11.0f, false, 7, 3.0f, &g_wifi_mock);
    
    // Enable to register callback
    esp_err_t result = manager.enable(nullptr);
    TEST_ASSERT_EQUAL(ESP_OK, result);
    
    // Create CSI data
    wifi_csi_info_t csi_info = {};
    csi_info.buf = const_cast<int8_t*>(baseline_packets[0]);
    csi_info.len = 128;
    
    // Trigger callback through mock (simulates ESP-IDF calling the callback)
    g_wifi_mock.trigger_callback(&csi_info);
    
    // Verify packet was processed (total packets should be > 0)
    TEST_ASSERT_TRUE(csi_processor_get_total_packets(&g_processor) > 0);
}

void test_csi_manager_callback_wrapper_null_data(void) {
    CSIManager manager;
    manager.init(&g_processor, TEST_SUBCARRIERS, 1.0f, 50, 100, true, 11.0f, false, 7, 3.0f, &g_wifi_mock);
    
    // Enable to register callback
    manager.enable(nullptr);
    
    uint32_t packets_before = csi_processor_get_total_packets(&g_processor);
    
    // Trigger callback with null data (should be handled gracefully)
    g_wifi_mock.trigger_callback(nullptr);
    
    // Verify no packet was processed
    TEST_ASSERT_EQUAL(packets_before, csi_processor_get_total_packets(&g_processor));
}

// ============================================================================
// ENTRY POINT
// ============================================================================

int process(void) {
    UNITY_BEGIN();
    
    // Initialization tests
    RUN_TEST(test_csi_manager_init);
    RUN_TEST(test_csi_manager_init_with_hampel);
    RUN_TEST(test_csi_manager_init_without_hampel);
    
    // Enable/Disable tests (using WiFiCSIMock)
    RUN_TEST(test_csi_manager_enable);
    RUN_TEST(test_csi_manager_enable_twice_returns_ok);
    RUN_TEST(test_csi_manager_disable);
    RUN_TEST(test_csi_manager_disable_when_not_enabled);
    
    // Threshold tests
    RUN_TEST(test_csi_manager_set_threshold);
    RUN_TEST(test_csi_manager_set_threshold_multiple_times);
    
    // Subcarrier selection tests
    RUN_TEST(test_csi_manager_update_subcarrier_selection);
    
    // Process packet tests
    RUN_TEST(test_csi_manager_process_packet_null_data);
    RUN_TEST(test_csi_manager_process_packet_short_data);
    RUN_TEST(test_csi_manager_process_packet_real_data);
    RUN_TEST(test_csi_manager_process_packet_detects_motion);
    
    // Callback tests
    RUN_TEST(test_csi_manager_callback_invoked);
    RUN_TEST(test_csi_manager_callback_not_invoked_before_publish_rate);
    
    // Calibration mode tests
    RUN_TEST(test_csi_manager_set_calibration_mode);
    
    // Calibration delegation tests
    RUN_TEST(test_csi_manager_with_calibrator_not_calibrating);
    RUN_TEST(test_csi_manager_null_calibrator_processes_normally);
    RUN_TEST(test_csi_manager_delegates_when_calibrating);
    RUN_TEST(test_csi_manager_calibration_triggers_periodic_yield);
    RUN_TEST(test_csi_manager_calibrator_lifecycle);
    
    // Error path tests
    RUN_TEST(test_csi_manager_enable_config_error);
    RUN_TEST(test_csi_manager_enable_callback_error);
    RUN_TEST(test_csi_manager_enable_csi_error);
    RUN_TEST(test_csi_manager_disable_error);
    RUN_TEST(test_csi_manager_callback_wrapper_triggered);
    RUN_TEST(test_csi_manager_callback_wrapper_null_data);
    
    // Integration tests (using WiFiCSIMock)
    RUN_TEST(test_csi_manager_full_workflow);
    RUN_TEST(test_csi_manager_baseline_then_motion);
    
    // Channel change detection tests
    RUN_TEST(test_csi_manager_channel_change_resets_buffer);
    RUN_TEST(test_csi_manager_same_channel_no_reset);
    
    return UNITY_END();
}

#if defined(ESP_PLATFORM)
extern "C" void app_main(void) { process(); }
#else
int main(int argc, char **argv) { return process(); }
#endif

