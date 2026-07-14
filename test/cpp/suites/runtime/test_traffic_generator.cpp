/*
 * ESPectre - Traffic Generator Unit Tests
 *
 * Tests the traffic generator error handling functions.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "test_harness.h"
#include <cstdint>
#include "traffic_generator_manager.h"
#include "traffic_rate_controller.h"
#include "esphome/core/log.h"

using namespace espectre;

static const char *TAG = "test_traffic_generator";

void setUp(void) {
    // Nothing to set up
}

void tearDown(void) {
    // Nothing to tear down
}

// ============================================================================
// SEND ERROR STATE TESTS
// ============================================================================

void test_send_error_state_initialization(void) {
    SendErrorState state;
    
    TEST_ASSERT_EQUAL(0, state.error_count);
    TEST_ASSERT_EQUAL(0, state.last_log_time);
    TEST_ASSERT_EQUAL(1000000, SendErrorState::LOG_INTERVAL_US);
}

// ============================================================================
// HANDLE SEND ERROR TESTS
// ============================================================================

void test_handle_send_error_increments_count(void) {
    SendErrorState state;
    state.last_log_time = 0;  // Will trigger log on first call (time 0 - 0 = 0 which is NOT > 1sec)
    
    // First error at time 0 - condition: 0 - 0 = 0, NOT > 1000000, so NO log
    handle_send_error(state, -1, 11, 0);  // errno 11 = EAGAIN
    
    // Count should be 1 because no logging happened (0 is not > LOG_INTERVAL)
    TEST_ASSERT_EQUAL(1, state.error_count);
}

void test_handle_send_error_rate_limits_logging(void) {
    SendErrorState state;
    state.last_log_time = 0;
    
    // First error at time 0 - condition: 0 - 0 = 0, NOT > 1000000, so NO log
    handle_send_error(state, -1, 11, 0);
    TEST_ASSERT_EQUAL(1, state.error_count);  // Incremented, not reset
    TEST_ASSERT_EQUAL(0, state.last_log_time);  // NOT updated
    
    // Second error at time 500ms - still NOT > 1sec from last_log_time (0)
    handle_send_error(state, -1, 11, 500000);
    TEST_ASSERT_EQUAL(2, state.error_count);  // Incremented
    
    // Third error at time 1.5 seconds - NOW > 1 second since last log (0)
    handle_send_error(state, -1, 11, 1500000);
    TEST_ASSERT_EQUAL(0, state.error_count);  // Reset after log
    TEST_ASSERT_EQUAL(1500000, state.last_log_time);  // Updated
    
    // Fourth error at time 2.0 seconds - NOT > 1 second since last log (1.5s)
    handle_send_error(state, -1, 11, 2000000);
    TEST_ASSERT_EQUAL(1, state.error_count);  // Incremented
    
    // Fifth error at time 2.6 seconds - NOW > 1 second since last log (1.5s)
    handle_send_error(state, -1, 11, 2600000);
    TEST_ASSERT_EQUAL(0, state.error_count);  // Reset after log
    TEST_ASSERT_EQUAL(2600000, state.last_log_time);  // Updated
}

void test_handle_send_error_returns_true_for_enomem(void) {
    SendErrorState state;
    
    // ENOMEM (errno 12) should return true for backoff
    bool needs_backoff = handle_send_error(state, -1, 12, 0);
    TEST_ASSERT_TRUE(needs_backoff);
}

void test_handle_send_error_returns_false_for_other_errors(void) {
    SendErrorState state;
    
    // EAGAIN (errno 11) should return false
    bool needs_backoff = handle_send_error(state, -1, 11, 0);
    TEST_ASSERT_FALSE(needs_backoff);
    
    // Reset state for next test
    state = SendErrorState();
    
    // ECONNREFUSED (errno 111) should return false
    needs_backoff = handle_send_error(state, -1, 111, 2000000);
    TEST_ASSERT_FALSE(needs_backoff);
}

void test_handle_send_error_logs_single_error_message(void) {
    SendErrorState state;
    state.last_log_time = 0;
    
    // Single error at time 1.5s - will trigger log (1.5s - 0 > 1s)
    handle_send_error(state, -1, 11, 1500000);
    
    // State should be reset after logging
    TEST_ASSERT_EQUAL(0, state.error_count);
    TEST_ASSERT_EQUAL(1500000, state.last_log_time);
}

void test_handle_send_error_logs_multiple_errors_summary(void) {
    SendErrorState state;
    state.last_log_time = 0;
    
    // Accumulate errors without logging (all within first second from last_log_time=0)
    handle_send_error(state, -1, 11, 100000);   // 0.1s - no log
    handle_send_error(state, -1, 11, 200000);   // 0.2s - no log
    handle_send_error(state, -1, 11, 300000);   // 0.3s - no log
    handle_send_error(state, -1, 11, 400000);   // 0.4s - no log
    TEST_ASSERT_EQUAL(4, state.error_count);
    
    // Now trigger logging with time > 1 second from last_log_time (0)
    // Should log "Send errors: 5 in last second (errno: 11)"
    handle_send_error(state, -1, 11, 1500000);
    TEST_ASSERT_EQUAL(0, state.error_count);  // Reset after log
    TEST_ASSERT_EQUAL(1500000, state.last_log_time);
}

void test_handle_send_error_handles_negative_sent_value(void) {
    SendErrorState state;
    
    // sendto() returns -1 on error
    bool needs_backoff = handle_send_error(state, -1, 12, 0);
    TEST_ASSERT_TRUE(needs_backoff);
    
    // Reset state
    state = SendErrorState();
    
    // sendto() could also return 0 (no bytes sent)
    needs_backoff = handle_send_error(state, 0, 12, 2000000);
    TEST_ASSERT_TRUE(needs_backoff);
}

void test_adaptive_rate_controller_trims_excess_csi_rate(void) {
    TrafficRateController controller;
    controller.init(100U, true);

    TEST_ASSERT_FALSE(controller.observe(0U, 1));
    TEST_ASSERT_TRUE(controller.observe(400U, 2000001));
    TEST_ASSERT_EQUAL(200U, controller.observed_pps());
    TEST_ASSERT_EQUAL(70U, controller.current_pps());
}

void test_adaptive_rate_controller_holds_inside_tolerance(void) {
    TrafficRateController controller;
    controller.init(100U, true);

    controller.observe(0U, 1);
    TEST_ASSERT_FALSE(controller.observe(202U, 2000001));
    TEST_ASSERT_EQUAL(101U, controller.observed_pps());
    TEST_ASSERT_EQUAL(100U, controller.current_pps());
}

void test_adaptive_rate_controller_recovers_additively(void) {
    TrafficRateController controller;
    controller.init(100U, true);

    controller.observe(0U, 1);
    TEST_ASSERT_TRUE(controller.observe(100U, 2000001));
    TEST_ASSERT_EQUAL(50U, controller.observed_pps());
    TEST_ASSERT_EQUAL(102U, controller.current_pps());
}

void test_fixed_rate_controller_observes_without_adjusting(void) {
    TrafficRateController controller;
    controller.init(100U, false);

    controller.observe(0U, 1);
    TEST_ASSERT_FALSE(controller.observe(400U, 2000001));
    TEST_ASSERT_EQUAL(200U, controller.observed_pps());
    TEST_ASSERT_EQUAL(100U, controller.current_pps());
}

// ============================================================================
// ENTRY POINT
// ============================================================================

int process(void) {
    UNITY_BEGIN();
    
    // SendErrorState tests
    RUN_TEST(test_send_error_state_initialization);
    
    // handle_send_error tests
    RUN_TEST(test_handle_send_error_increments_count);
    RUN_TEST(test_handle_send_error_rate_limits_logging);
    RUN_TEST(test_handle_send_error_returns_true_for_enomem);
    RUN_TEST(test_handle_send_error_returns_false_for_other_errors);
    RUN_TEST(test_handle_send_error_logs_single_error_message);
    RUN_TEST(test_handle_send_error_logs_multiple_errors_summary);
    RUN_TEST(test_handle_send_error_handles_negative_sent_value);
    RUN_TEST(test_adaptive_rate_controller_trims_excess_csi_rate);
    RUN_TEST(test_adaptive_rate_controller_holds_inside_tolerance);
    RUN_TEST(test_adaptive_rate_controller_recovers_additively);
    RUN_TEST(test_fixed_rate_controller_observes_without_adjusting);
    
    return UNITY_END();
}

#if defined(ESP_PLATFORM)
extern "C" void app_main(void) { process(); }
#else
int main(int argc, char **argv) { return process(); }
#endif
