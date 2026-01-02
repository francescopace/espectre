/*
 * ESPectre - Serial Streamer Unit Tests
 *
 * Tests the serial streamer command processing and state management.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include <unity.h>
#include <cstdint>
#include <cstring>
#include "serial_streamer.h"
#include "esphome/core/log.h"

using namespace esphome::espectre;

// Test fixture
static SerialStreamer streamer;
static bool threshold_callback_called;
static float threshold_callback_value;
static bool start_callback_called;

void setUp(void) {
    streamer = SerialStreamer();
    streamer.init();
    threshold_callback_called = false;
    threshold_callback_value = 0.0f;
    start_callback_called = false;
}

void tearDown(void) {
    // Nothing to tear down
}

// ============================================================================
// INITIALIZATION TESTS
// ============================================================================

void test_serial_streamer_init(void) {
    SerialStreamer s;
    s.init();
    
    TEST_ASSERT_FALSE(s.is_active());
}

void test_serial_streamer_default_inactive(void) {
    SerialStreamer s;
    
    TEST_ASSERT_FALSE(s.is_active());
}

// ============================================================================
// START/STOP TESTS
// ============================================================================

void test_serial_streamer_start(void) {
    streamer.start();
    
    TEST_ASSERT_TRUE(streamer.is_active());
}

void test_serial_streamer_stop(void) {
    streamer.start();
    TEST_ASSERT_TRUE(streamer.is_active());
    
    streamer.stop();
    TEST_ASSERT_FALSE(streamer.is_active());
}

void test_serial_streamer_start_when_already_active(void) {
    streamer.start();
    TEST_ASSERT_TRUE(streamer.is_active());
    
    // Start again should not change state
    streamer.start();
    TEST_ASSERT_TRUE(streamer.is_active());
}

void test_serial_streamer_stop_when_already_inactive(void) {
    TEST_ASSERT_FALSE(streamer.is_active());
    
    // Stop when already inactive should not crash
    streamer.stop();
    TEST_ASSERT_FALSE(streamer.is_active());
}

// ============================================================================
// CALLBACK TESTS
// ============================================================================

void test_serial_streamer_start_callback(void) {
    streamer.set_start_callback([]() {
        start_callback_called = true;
    });
    
    TEST_ASSERT_FALSE(start_callback_called);
    
    streamer.start();
    
    TEST_ASSERT_TRUE(start_callback_called);
}

void test_serial_streamer_start_callback_not_called_when_already_active(void) {
    streamer.set_start_callback([]() {
        start_callback_called = true;
    });
    
    streamer.start();
    TEST_ASSERT_TRUE(start_callback_called);
    
    // Reset flag
    start_callback_called = false;
    
    // Start again should not call callback
    streamer.start();
    TEST_ASSERT_FALSE(start_callback_called);
}

void test_serial_streamer_threshold_callback_setup(void) {
    bool called = false;
    float received_value = 0.0f;
    
    streamer.set_threshold_callback([&called, &received_value](float value) {
        called = true;
        received_value = value;
    });
    
    // Callback is set but not called yet
    TEST_ASSERT_FALSE(called);
}

void test_serial_streamer_no_callback_no_crash(void) {
    // No callback set, start should not crash
    streamer.start();
    TEST_ASSERT_TRUE(streamer.is_active());
}

// ============================================================================
// SEND DATA TESTS (when USE_USB_SERIAL_JTAG is 0, send_data does nothing)
// ============================================================================

void test_serial_streamer_send_data_when_inactive(void) {
    TEST_ASSERT_FALSE(streamer.is_active());
    
    // Should not crash when inactive
    streamer.send_data(1.5f, 2.0f);
    
    TEST_ASSERT_FALSE(streamer.is_active());
}

void test_serial_streamer_send_data_when_active(void) {
    streamer.start();
    TEST_ASSERT_TRUE(streamer.is_active());
    
    // Should not crash when active (no-op without USB Serial JTAG)
    streamer.send_data(1.5f, 2.0f);
    
    TEST_ASSERT_TRUE(streamer.is_active());
}

// ============================================================================
// CHECK COMMANDS TESTS (when USE_USB_SERIAL_JTAG is 0, check_commands does nothing)
// ============================================================================

void test_serial_streamer_check_commands_no_crash(void) {
    // Should not crash when called
    streamer.check_commands();
    
    TEST_ASSERT_FALSE(streamer.is_active());
}

// ============================================================================
// PROCESS COMMAND TESTS
// ============================================================================

void test_process_command_start(void) {
    TEST_ASSERT_FALSE(streamer.is_active());
    
    streamer.process_command("START");
    
    TEST_ASSERT_TRUE(streamer.is_active());
}

void test_process_command_stop(void) {
    streamer.start();
    TEST_ASSERT_TRUE(streamer.is_active());
    
    streamer.process_command("STOP");
    
    TEST_ASSERT_FALSE(streamer.is_active());
}

void test_process_command_ping(void) {
    streamer.start();
    
    // PING should not change active state
    streamer.process_command("PING");
    
    TEST_ASSERT_TRUE(streamer.is_active());
}

void test_process_command_threshold_valid(void) {
    float received_threshold = 0.0f;
    
    streamer.set_threshold_callback([&received_threshold](float value) {
        received_threshold = value;
    });
    
    streamer.process_command("T:1.50");
    
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 1.50f, received_threshold);
}

void test_process_command_threshold_minimum(void) {
    float received_threshold = 0.0f;
    
    streamer.set_threshold_callback([&received_threshold](float value) {
        received_threshold = value;
    });
    
    streamer.process_command("T:0.1");
    
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 0.1f, received_threshold);
}

void test_process_command_threshold_maximum(void) {
    float received_threshold = 0.0f;
    
    streamer.set_threshold_callback([&received_threshold](float value) {
        received_threshold = value;
    });
    
    streamer.process_command("T:10.0");
    
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 10.0f, received_threshold);
}

void test_process_command_threshold_too_low(void) {
    float received_threshold = -1.0f;  // Sentinel value
    
    streamer.set_threshold_callback([&received_threshold](float value) {
        received_threshold = value;
    });
    
    streamer.process_command("T:0.05");  // Below minimum 0.1
    
    // Callback should NOT be called for invalid values
    TEST_ASSERT_FLOAT_WITHIN(0.01f, -1.0f, received_threshold);
}

void test_process_command_threshold_too_high(void) {
    float received_threshold = -1.0f;  // Sentinel value
    
    streamer.set_threshold_callback([&received_threshold](float value) {
        received_threshold = value;
    });
    
    streamer.process_command("T:15.0");  // Above maximum 10.0
    
    // Callback should NOT be called for invalid values
    TEST_ASSERT_FLOAT_WITHIN(0.01f, -1.0f, received_threshold);
}

void test_process_command_threshold_no_callback(void) {
    // No callback set - should not crash
    streamer.process_command("T:1.50");
    
    // Just verify no crash occurred
    TEST_ASSERT_TRUE(true);
}

void test_process_command_unknown(void) {
    TEST_ASSERT_FALSE(streamer.is_active());
    
    // Unknown command should be ignored
    streamer.process_command("UNKNOWN");
    streamer.process_command("RESTART");
    streamer.process_command("");
    
    TEST_ASSERT_FALSE(streamer.is_active());
}

// ============================================================================
// MAIN
// ============================================================================

int main(int argc, char **argv) {
    UNITY_BEGIN();
    
    // Initialization tests
    RUN_TEST(test_serial_streamer_init);
    RUN_TEST(test_serial_streamer_default_inactive);
    
    // Start/Stop tests
    RUN_TEST(test_serial_streamer_start);
    RUN_TEST(test_serial_streamer_stop);
    RUN_TEST(test_serial_streamer_start_when_already_active);
    RUN_TEST(test_serial_streamer_stop_when_already_inactive);
    
    // Callback tests
    RUN_TEST(test_serial_streamer_start_callback);
    RUN_TEST(test_serial_streamer_start_callback_not_called_when_already_active);
    RUN_TEST(test_serial_streamer_threshold_callback_setup);
    RUN_TEST(test_serial_streamer_no_callback_no_crash);
    
    // Send data tests
    RUN_TEST(test_serial_streamer_send_data_when_inactive);
    RUN_TEST(test_serial_streamer_send_data_when_active);
    
    // Check commands tests
    RUN_TEST(test_serial_streamer_check_commands_no_crash);
    
    // Process command tests
    RUN_TEST(test_process_command_start);
    RUN_TEST(test_process_command_stop);
    RUN_TEST(test_process_command_ping);
    RUN_TEST(test_process_command_threshold_valid);
    RUN_TEST(test_process_command_threshold_minimum);
    RUN_TEST(test_process_command_threshold_maximum);
    RUN_TEST(test_process_command_threshold_too_low);
    RUN_TEST(test_process_command_threshold_too_high);
    RUN_TEST(test_process_command_threshold_no_callback);
    RUN_TEST(test_process_command_unknown);
    
    return UNITY_END();
}

