/*
 * ESPectre - Traffic Generator Unit Tests
 *
 * Tests the traffic generator error handling functions.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include <unity.h>
#include <cstdint>
#include "traffic_generator_manager.h"
#include "esphome/core/log.h"

using namespace esphome::espectre;

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

// ============================================================================
// TRAFFIC GENERATOR MODE ENUM TESTS
// ============================================================================

void test_traffic_generator_mode_enum_values(void) {
    // Ensure all three modes can be compared without ambiguity
    TrafficGeneratorMode dns_mode = TrafficGeneratorMode::DNS;
    TrafficGeneratorMode ping_mode = TrafficGeneratorMode::PING;
    TrafficGeneratorMode udp_mode = TrafficGeneratorMode::UDP;

    TEST_ASSERT_TRUE(dns_mode != ping_mode);
    TEST_ASSERT_TRUE(dns_mode != udp_mode);
    TEST_ASSERT_TRUE(ping_mode != udp_mode);
    TEST_ASSERT_TRUE(udp_mode == TrafficGeneratorMode::UDP);
}

void test_traffic_generator_mode_udp_distinct(void) {
    // UDP mode must be a distinct value (not accidentally aliasing DNS or PING)
    TrafficGeneratorMode mode = TrafficGeneratorMode::UDP;
    TEST_ASSERT_FALSE(mode == TrafficGeneratorMode::DNS);
    TEST_ASSERT_FALSE(mode == TrafficGeneratorMode::PING);
}

void test_traffic_generator_udp_host_port_defaults(void) {
    // TrafficGeneratorManager default udp_port_ is 5000
    // We test the manager's setters via the public API
    TrafficGeneratorManager mgr;
    mgr.set_udp_host("192.168.1.1");
    mgr.set_udp_port(4444);
    // No crash — setter accepted valid values
    TEST_ASSERT_TRUE(true);
}

void test_traffic_generator_udp_host_setter_empty(void) {
    // Assigning an empty host is accepted by the setter (validation happens at start())
    TrafficGeneratorManager mgr;
    mgr.set_udp_host("");
    TEST_ASSERT_TRUE(true);  // No crash
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

    // TrafficGeneratorMode UDP enum tests
    RUN_TEST(test_traffic_generator_mode_enum_values);
    RUN_TEST(test_traffic_generator_mode_udp_distinct);
    RUN_TEST(test_traffic_generator_udp_host_port_defaults);
    RUN_TEST(test_traffic_generator_udp_host_setter_empty);

    return UNITY_END();
}

#if defined(ESP_PLATFORM)
extern "C" void app_main(void) { process(); }
#else
int main(int argc, char **argv) { return process(); }
#endif

