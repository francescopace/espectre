/*
 * ESPectre - Traffic Generator Unit Tests
 *
 * Tests the traffic generator error handling functions.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "test_harness.h"
#include <cstdint>
#include <cstring>
#include "esp_wifi.h"
#include "traffic_generator_manager.h"
#include "sdkconfig.h"
#include "esphome/core/log.h"
#include "esp_private/wifi.h"
#include "lwip/sockets.h"
#include "csi_traffic_fakes.h"

using namespace espectre;

namespace {
#ifndef ESPECTRE_TEST_FIXED_STA_RATE
#define ESPECTRE_TEST_FIXED_STA_RATE 1
#endif
#ifndef ESPECTRE_TEST_EXPECTED_TX_RATE
#define ESPECTRE_TEST_EXPECTED_TX_RATE WIFI_PHY_RATE_6M
#endif
constexpr wifi_phy_rate_t EXPECTED_TX_RATE = ESPECTRE_TEST_EXPECTED_TX_RATE;
TrafficGeneratorManager *active_generator = nullptr;
int last_test_socket = -1;
bool fail_socket_creation = false;

int create_test_socket(int domain, int type, int protocol) {
    (void)domain;
    (void)type;
    (void)protocol;
    if (fail_socket_creation) {
        errno = ENOMEM;
        return -1;
    }
    // Startup only needs an owned descriptor; no test sends network traffic.
    last_test_socket = open("/dev/null", O_RDWR);
    return last_test_socket;
}

void finish_stopping_task() {
    const auto function = g_freertos_task_mock.pending_function;
    void *argument = g_freertos_task_mock.pending_argument;
    g_freertos_task_mock.pending_function = nullptr;
    if (function != nullptr) function(argument);
}

void prepare_lifecycle_test() {
    g_lwip_socket_mock_factory = create_test_socket;
    g_freertos_task_mock.defer_execution = true;
    g_freertos_delay_hook = finish_stopping_task;
}

void stop_raw_test_generator() {
    // The FreeRTOS mock runs the task synchronously; stop after bounded sends.
    if (g_esp_wifi_mock.raw_tx_call_count == 3) active_generator->stop();
}
}  // namespace

void setUp(void) {
    esp_wifi_mock_reset();
    g_esp_wifi_fixed_rate_mock = {};
    g_freertos_task_mock = {};
    g_freertos_delay_hook = nullptr;
    g_lwip_socket_mock_factory = nullptr;
    last_test_socket = -1;
    fail_socket_creation = false;
}

void tearDown(void) {
    g_esp_wifi_mock.raw_tx_hook = nullptr;
    active_generator = nullptr;
    g_freertos_delay_hook = nullptr;
    g_lwip_socket_mock_factory = nullptr;
    if (last_test_socket >= 0 && fcntl(last_test_socket, F_GETFD) >= 0) {
        close(last_test_socket);
    }
}

void test_fixed_sta_rate_covers_every_internal_generator_and_running_lifecycle(void) {
    prepare_lifecycle_test();
    g_esp_wifi_mock.current_ap_info.bssid[0] = 0x02U;
    TrafficGeneratorManager manager;
    for (const RuntimeTrafficMode mode : {RuntimeTrafficMode::PING, RuntimeTrafficMode::DNS,
                                         RuntimeTrafficMode::DNS_TCP, RuntimeTrafficMode::WIFI_RAW}) {
        g_esp_wifi_fixed_rate_mock = {};
        g_freertos_task_mock.create_calls = 0U;
        manager.init(100U, mode);
        TEST_ASSERT_TRUE(manager.start(0x0101A8C0U));
        const bool expect_fixed_rate = ESPECTRE_TEST_FIXED_STA_RATE;
        TEST_ASSERT_EQUAL(expect_fixed_rate, g_esp_wifi_fixed_rate_mock.enabled);
        TEST_ASSERT_EQUAL(expect_fixed_rate ? 1U : 0U, g_esp_wifi_fixed_rate_mock.enable_calls);
        TEST_ASSERT_EQUAL(WIFI_IF_STA, g_esp_wifi_fixed_rate_mock.interface);
        if (expect_fixed_rate) {
            TEST_ASSERT_EQUAL(EXPECTED_TX_RATE, g_esp_wifi_fixed_rate_mock.rate);
        }
        if (mode == RuntimeTrafficMode::WIFI_RAW) {
            TEST_ASSERT_EQUAL(EXPECTED_TX_RATE, g_esp_wifi_mock.raw_rate_config.rate);
        }
        TEST_ASSERT_EQUAL(100U, manager.current_rate_pps());
        TEST_ASSERT_TRUE(manager.start(0x0101A8C0U));
        TEST_ASSERT_EQUAL(1U, g_freertos_task_mock.create_calls);
        manager.pause();
        TEST_ASSERT_EQUAL(expect_fixed_rate, g_esp_wifi_fixed_rate_mock.enabled);
        manager.resume();
        manager.stop();
        TEST_ASSERT_FALSE(manager.is_running());
        TEST_ASSERT_FALSE(g_esp_wifi_fixed_rate_mock.enabled);
        TEST_ASSERT_EQUAL(expect_fixed_rate ? 1U : 0U, g_esp_wifi_fixed_rate_mock.disable_calls);
        manager.stop();
        TEST_ASSERT_EQUAL(expect_fixed_rate ? 1U : 0U, g_esp_wifi_fixed_rate_mock.disable_calls);
        if (mode != RuntimeTrafficMode::WIFI_RAW) {
            TEST_ASSERT_EQUAL(-1, fcntl(last_test_socket, F_GETFD));
        }
    }
}

void test_ping_socket_failure_does_not_change_radio_or_create_task(void) {
    prepare_lifecycle_test();
    fail_socket_creation = true;
    TrafficGeneratorManager manager;
    manager.init(100U, RuntimeTrafficMode::PING);
    TEST_ASSERT_FALSE(manager.start(0x0101A8C0U));
    TEST_ASSERT_EQUAL(0U, g_esp_wifi_fixed_rate_mock.enable_calls);
    TEST_ASSERT_EQUAL(0U, g_esp_wifi_fixed_rate_mock.disable_calls);
    TEST_ASSERT_EQUAL(0U, g_freertos_task_mock.create_calls);
    TEST_ASSERT_FALSE(manager.is_running());
}

void test_station_rate_requires_ofdm_capable_access_point(void) {
    prepare_lifecycle_test();
    TrafficGeneratorManager manager;
    for (unsigned phy = 0U; phy < 4U; ++phy) {
        g_esp_wifi_fixed_rate_mock = {};
        g_esp_wifi_mock.current_ap_info.primary = phy == 3U ? 36U : 6U;
        g_esp_wifi_mock.current_ap_info.phy_11g = phy == 1U;
        g_esp_wifi_mock.current_ap_info.phy_11n = phy == 2U;
        manager.init(100U, RuntimeTrafficMode::PING);
        TEST_ASSERT_TRUE(manager.start(0x0101A8C0U));
        const bool expect_fixed_rate = ESPECTRE_TEST_FIXED_STA_RATE && phy != 0U;
        TEST_ASSERT_EQUAL(expect_fixed_rate, g_esp_wifi_fixed_rate_mock.enabled);
        TEST_ASSERT_EQUAL(expect_fixed_rate ? 1U : 0U, g_esp_wifi_fixed_rate_mock.enable_calls);
        manager.stop();
        TEST_ASSERT_FALSE(g_esp_wifi_fixed_rate_mock.enabled);
    }
}

void test_ping_association_lookup_failure_closes_socket_before_task_creation(void) {
    prepare_lifecycle_test();
    g_esp_wifi_mock.get_ap_info_result = ESP_ERR_WIFI_NOT_CONNECT;
    TrafficGeneratorManager manager;
    manager.init(100U, RuntimeTrafficMode::PING);
#if ESPECTRE_TEST_FIXED_STA_RATE
    TEST_ASSERT_FALSE(manager.start(0x0101A8C0U));
    TEST_ASSERT_EQUAL(0U, g_freertos_task_mock.create_calls);
    TEST_ASSERT_EQUAL(-1, fcntl(last_test_socket, F_GETFD));
#else
    TEST_ASSERT_TRUE(manager.start(0x0101A8C0U));
#endif
    TEST_ASSERT_EQUAL(0U, g_esp_wifi_fixed_rate_mock.enable_calls);
    manager.stop();
}

void test_ping_rate_failure_closes_socket_before_task_creation(void) {
    prepare_lifecycle_test();
    g_esp_wifi_fixed_rate_mock.enable_result = ESP_FAIL;
    TrafficGeneratorManager manager;
    manager.init(100U, RuntimeTrafficMode::PING);
#if ESPECTRE_TEST_FIXED_STA_RATE
    TEST_ASSERT_FALSE(manager.start(0x0101A8C0U));
    TEST_ASSERT_EQUAL(1U, g_esp_wifi_fixed_rate_mock.enable_calls);
    TEST_ASSERT_EQUAL(0U, g_freertos_task_mock.create_calls);
    TEST_ASSERT_EQUAL(-1, fcntl(last_test_socket, F_GETFD));
    TEST_ASSERT_FALSE(manager.is_running());
#else
    TEST_ASSERT_TRUE(manager.start(0x0101A8C0U));
    TEST_ASSERT_EQUAL(0U, g_esp_wifi_fixed_rate_mock.enable_calls);
#endif
    manager.stop();
    TEST_ASSERT_EQUAL(0U, g_esp_wifi_fixed_rate_mock.disable_calls);
}

void test_raw_sta_rate_failure_does_not_start_injection(void) {
    prepare_lifecycle_test();
    g_esp_wifi_mock.current_ap_info.bssid[0] = 0x02U;
    g_esp_wifi_fixed_rate_mock.enable_result = ESP_FAIL;
    TrafficGeneratorManager manager;
    manager.init(100U, RuntimeTrafficMode::WIFI_RAW);
#if ESPECTRE_TEST_FIXED_STA_RATE
    TEST_ASSERT_FALSE(manager.start(0U));
    TEST_ASSERT_EQUAL(0U, g_freertos_task_mock.create_calls);
#else
    TEST_ASSERT_TRUE(manager.start(0U));
#endif
    TEST_ASSERT_EQUAL(0U, g_esp_wifi_mock.raw_tx_call_count);
    TEST_ASSERT_FALSE(g_esp_wifi_fixed_rate_mock.enabled);
    manager.stop();
}

void test_ping_task_failure_restores_radio_and_closes_socket(void) {
    prepare_lifecycle_test();
    g_freertos_task_mock.create_result = pdFAIL;
    TrafficGeneratorManager manager;
    manager.init(100U, RuntimeTrafficMode::PING);
    TEST_ASSERT_FALSE(manager.start(0x0101A8C0U));
    TEST_ASSERT_FALSE(manager.is_running());
    TEST_ASSERT_FALSE(g_esp_wifi_fixed_rate_mock.enabled);
    TEST_ASSERT_EQUAL((ESPECTRE_TEST_FIXED_STA_RATE) ? 1U : 0U, g_esp_wifi_fixed_rate_mock.disable_calls);
    TEST_ASSERT_EQUAL(-1, fcntl(last_test_socket, F_GETFD));
}

void test_failed_rate_restore_retains_ownership_until_explicit_cleanup(void) {
    prepare_lifecycle_test();
    TrafficGeneratorManager manager;
    manager.init(100U, RuntimeTrafficMode::PING);
    TEST_ASSERT_TRUE(manager.start(0x0101A8C0U));
    g_esp_wifi_fixed_rate_mock.disable_result = ESP_FAIL;
    manager.stop();
    TEST_ASSERT_FALSE(manager.is_running());
    TEST_ASSERT_EQUAL(ESPECTRE_TEST_FIXED_STA_RATE, g_esp_wifi_fixed_rate_mock.enabled);
    manager.init(100U, RuntimeTrafficMode::DNS);
#if ESPECTRE_TEST_FIXED_STA_RATE
    TEST_ASSERT_FALSE(manager.start(0x0101A8C0U));
    TEST_ASSERT_EQUAL(1U, g_freertos_task_mock.create_calls);
    TEST_ASSERT_TRUE(g_esp_wifi_fixed_rate_mock.enabled);
#endif
    g_esp_wifi_fixed_rate_mock.disable_result = ESP_OK;
    TEST_ASSERT_TRUE(manager.start(0x0101A8C0U));
    TEST_ASSERT_EQUAL(ESPECTRE_TEST_FIXED_STA_RATE,
                      g_esp_wifi_fixed_rate_mock.enabled);
    manager.stop();
    TEST_ASSERT_FALSE(g_esp_wifi_fixed_rate_mock.enabled);
}

void test_failed_startup_cleanup_can_be_retried_by_stop(void) {
    prepare_lifecycle_test();
    g_freertos_task_mock.create_result = pdFAIL;
    g_esp_wifi_fixed_rate_mock.disable_result = ESP_FAIL;
    TrafficGeneratorManager manager;
    manager.init(100U, RuntimeTrafficMode::PING);
    TEST_ASSERT_FALSE(manager.start(0x0101A8C0U));
    TEST_ASSERT_EQUAL(ESPECTRE_TEST_FIXED_STA_RATE, g_esp_wifi_fixed_rate_mock.enabled);
    g_esp_wifi_fixed_rate_mock.disable_result = ESP_OK;
    manager.stop();
    TEST_ASSERT_FALSE(g_esp_wifi_fixed_rate_mock.enabled);
    TEST_ASSERT_EQUAL((ESPECTRE_TEST_FIXED_STA_RATE) ? 2U : 0U, g_esp_wifi_fixed_rate_mock.disable_calls);
}

void test_external_service_start_retries_failed_ping_radio_cleanup(void) {
    prepare_lifecycle_test();
    TrafficGeneratorManager generator;
    espectre::test::FakeCsiTrafficIngress ingress;
    CsiTrafficService service(generator, ingress);
    CsiTrafficServiceConfig config;
    config.mode = CsiTrafficMode::INTERNAL;
    config.traffic_mode = RuntimeTrafficMode::PING;
    config.rate_pps = 100U;
    service.init(config);
    TEST_ASSERT_TRUE(service.start(0x0101A8C0U));
    g_esp_wifi_fixed_rate_mock.disable_result = ESP_FAIL;
    service.stop();
    TEST_ASSERT_FALSE(generator.is_running());
    TEST_ASSERT_EQUAL(ESPECTRE_TEST_FIXED_STA_RATE,
                      g_esp_wifi_fixed_rate_mock.enabled);

    config.mode = CsiTrafficMode::EXTERNAL;
    service.init(config);
    g_esp_wifi_fixed_rate_mock.disable_result = ESP_OK;
    TEST_ASSERT_TRUE(service.start());
    TEST_ASSERT_FALSE(g_esp_wifi_fixed_rate_mock.enabled);
    TEST_ASSERT_FALSE(generator.is_running());
    TEST_ASSERT_TRUE(ingress.is_running());
    service.stop();
}

// ============================================================================
// SEND ERROR STATE TESTS
// ============================================================================

void test_wifi_raw_targets_current_bssid_without_gateway_and_counts_sends(void) {
    TrafficGeneratorManager manager;
    active_generator = &manager;
    g_esp_wifi_mock.raw_tx_hook = stop_raw_test_generator;
    const uint8_t bssid[6] = {0x02, 0x11, 0x22, 0x33, 0x44, 0x55};
    std::memcpy(g_esp_wifi_mock.current_ap_info.bssid, bssid, 6U);
    for (const esp_err_t result : {ESP_OK, ESP_ERR_NO_MEM, ESP_FAIL}) {
        g_esp_wifi_mock.raw_tx_call_count = 0;
        g_esp_wifi_mock.raw_tx_result = result;
        manager.init(100U, RuntimeTrafficMode::WIFI_RAW);
        TEST_ASSERT_TRUE(manager.start(0U));
        TEST_ASSERT_EQUAL(WIFI_IF_STA, g_esp_wifi_mock.raw_tx_interface);
        TEST_ASSERT_TRUE(g_esp_wifi_mock.raw_tx_sys_seq);
        TEST_ASSERT_EQUAL(24, g_esp_wifi_mock.raw_tx_length);
        const uint8_t *frame = g_esp_wifi_mock.raw_tx_frame;
        TEST_ASSERT_EQUAL_UINT8(0x48U, frame[0]);
        TEST_ASSERT_EQUAL_UINT8(0x01U, frame[1]);
        TEST_ASSERT_EQUAL_UINT8_ARRAY(g_esp_wifi_mock.current_ap_info.bssid, frame + 4U, 6U);
        TEST_ASSERT_EQUAL_UINT8_ARRAY(g_esp_wifi_mock.mac, frame + 10U, 6U);
        TEST_ASSERT_EQUAL_UINT8_ARRAY(g_esp_wifi_mock.current_ap_info.bssid, frame + 16U, 6U);
        TEST_ASSERT_EQUAL(0U, frame[22]);
        TEST_ASSERT_EQUAL(0U, frame[23]);
        TEST_ASSERT_EQUAL(result == ESP_OK ? 3U : 0U, manager.send_success_count());
        TEST_ASSERT_EQUAL(result == ESP_OK ? 0U : 3U, manager.send_error_count());
        TEST_ASSERT_EQUAL(100U, manager.current_rate_pps());
        TEST_ASSERT_FALSE(manager.is_running());
        // A later association must rebuild the destination before sending.
        ++g_esp_wifi_mock.current_ap_info.bssid[5];
    }
    TEST_ASSERT_EQUAL(3, g_esp_wifi_mock.get_ap_info_call_count);
    g_esp_wifi_mock.raw_tx_hook = nullptr;
    active_generator = nullptr;
}

void test_wifi_raw_uses_ofdm_for_each_band_and_rejects_rate_configuration_failure(void) {
    TrafficGeneratorManager manager;
    active_generator = &manager;
    g_esp_wifi_mock.raw_tx_hook = stop_raw_test_generator;
    g_esp_wifi_mock.current_ap_info.bssid[0] = 0x02U;
#if !(CONFIG_IDF_TARGET_ESP32C5 || CONFIG_IDF_TARGET_ESP32C6)
    const uint8_t channels[] = {10U};
#else
    const uint8_t channels[] = {10U, 48U};
#endif
    for (uint8_t channel : channels) {
        g_esp_wifi_mock.current_ap_info.primary = channel;
        g_esp_wifi_mock.raw_tx_call_count = 0;
        manager.init(100U, RuntimeTrafficMode::WIFI_RAW);
        TEST_ASSERT_TRUE(manager.start(0U));
        TEST_ASSERT_EQUAL(WIFI_IF_STA, g_esp_wifi_mock.raw_rate_interface);
        TEST_ASSERT_EQUAL(EXPECTED_TX_RATE, g_esp_wifi_mock.raw_rate_config.rate);
#if !(CONFIG_IDF_TARGET_ESP32C5 || CONFIG_IDF_TARGET_ESP32C6)
        TEST_ASSERT_TRUE(g_esp_wifi_mock.raw_rate_legacy);
#else
        TEST_ASSERT_FALSE(g_esp_wifi_mock.raw_rate_legacy);
        TEST_ASSERT_EQUAL(channel > 14U ? WIFI_PHY_MODE_11A : WIFI_PHY_MODE_11G,
                          g_esp_wifi_mock.raw_rate_config.phymode);
        TEST_ASSERT_FALSE(g_esp_wifi_mock.raw_rate_config.ersu);
        TEST_ASSERT_FALSE(g_esp_wifi_mock.raw_rate_config.dcm);
#endif
    }
    g_esp_wifi_mock.raw_tx_call_count = 0;
    g_esp_wifi_mock.raw_rate_result = ESP_FAIL;
    manager.init(100U, RuntimeTrafficMode::WIFI_RAW);
    TEST_ASSERT_FALSE(manager.start(0U));
    TEST_ASSERT_FALSE(manager.is_running());
    TEST_ASSERT_EQUAL(0U, g_esp_wifi_mock.raw_tx_call_count);
    g_esp_wifi_mock.raw_tx_hook = nullptr;
    active_generator = nullptr;
}

void test_wifi_raw_requires_association_and_valid_station_identity(void) {
    TrafficGeneratorManager manager;
    manager.init(100U, RuntimeTrafficMode::WIFI_RAW);
    g_esp_wifi_mock.get_ap_info_result = ESP_ERR_WIFI_NOT_CONNECT;
    TEST_ASSERT_FALSE(manager.start(0U));
    g_esp_wifi_mock.get_ap_info_result = ESP_OK;
    TEST_ASSERT_FALSE(manager.start(0U));
    g_esp_wifi_mock.current_ap_info.bssid[0] = 0x02U;
    g_esp_wifi_mock.get_mac_result = ESP_FAIL;
    TEST_ASSERT_FALSE(manager.start(0U));
    TEST_ASSERT_EQUAL(0, g_esp_wifi_mock.raw_tx_call_count);
    uint8_t frame[24]{};
    TEST_ASSERT_EQUAL(0U, build_null_data_frame(nullptr, g_esp_wifi_mock.mac, frame, sizeof(frame)));
    TEST_ASSERT_EQUAL(0U, build_null_data_frame(g_esp_wifi_mock.current_ap_info.bssid,
                                               g_esp_wifi_mock.mac, frame, sizeof(frame) - 1U));
}

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

void test_handle_send_error_resets_window_after_interval(void) {
    SendErrorState state;
    state.last_log_time = 0;
    
    // Single error at time 1.5s - will trigger log (1.5s - 0 > 1s)
    handle_send_error(state, -1, 11, 1500000);
    
    // State should be reset after logging
    TEST_ASSERT_EQUAL(0, state.error_count);
    TEST_ASSERT_EQUAL(1500000, state.last_log_time);
}

void test_handle_send_error_resets_accumulated_errors_after_interval(void) {
    SendErrorState state;
    state.last_log_time = 0;
    
    // Accumulate errors without logging (all within first second from last_log_time=0)
    handle_send_error(state, -1, 11, 100000);   // 0.1s - no log
    handle_send_error(state, -1, 11, 200000);   // 0.2s - no log
    handle_send_error(state, -1, 11, 300000);   // 0.3s - no log
    handle_send_error(state, -1, 11, 400000);   // 0.4s - no log
    TEST_ASSERT_EQUAL(4, state.error_count);
    
    // The next error after the interval resets the accumulated window.
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

void test_pacing_deadline_starts_from_first_send(void) {
    TEST_ASSERT_TRUE(next_traffic_send_deadline_us(0, 100000, 10000) == 110000);
}

void test_pacing_deadline_preserves_phase_across_small_jitter(void) {
    TEST_ASSERT_TRUE(next_traffic_send_deadline_us(110000, 110750, 10000) == 120000);
    TEST_ASSERT_TRUE(next_traffic_send_deadline_us(120000, 119500, 10000) == 130000);
}

void test_pacing_deadline_resets_instead_of_catching_up(void) {
    TEST_ASSERT_TRUE(next_traffic_send_deadline_us(110000, 119500, 10000) == 129500);
    TEST_ASSERT_TRUE(next_traffic_send_deadline_us(110000, 131000, 10000) == 141000);
}

void test_pacing_deadline_handles_invalid_interval(void) {
    TEST_ASSERT_TRUE(next_traffic_send_deadline_us(120000, 123456, 0) == 123456);
}

void test_dns_tcp_query_frame_adds_length_and_transaction_id(void) {
    uint8_t frame[TRAFFIC_DNS_TCP_FRAME_SIZE] = {};

    TEST_ASSERT_TRUE(
        build_dns_tcp_query_frame(0x1234U, frame, sizeof(frame)) == TRAFFIC_DNS_TCP_FRAME_SIZE);
    TEST_ASSERT_EQUAL_UINT8(0U, frame[0]);
    TEST_ASSERT_EQUAL_UINT8(TRAFFIC_DNS_QUERY_PAYLOAD_SIZE, frame[1]);
    TEST_ASSERT_EQUAL_UINT8(0x12U, frame[2]);
    TEST_ASSERT_EQUAL_UINT8(0x34U, frame[3]);
    TEST_ASSERT_EQUAL_UINT8(0x01U, frame[4]);
    TEST_ASSERT_EQUAL_UINT8(0x00U, frame[5]);
    TEST_ASSERT_EQUAL_UINT8(0x01U, frame[7]);
    TEST_ASSERT_EQUAL_UINT8(0x01U, frame[16]);
    TEST_ASSERT_EQUAL_UINT8(0x01U, frame[18]);
}

void test_dns_udp_query_payload_sets_transaction_id_without_tcp_length(void) {
    uint8_t payload[TRAFFIC_DNS_QUERY_PAYLOAD_SIZE] = {};

    TEST_ASSERT_EQUAL(TRAFFIC_DNS_QUERY_PAYLOAD_SIZE,
                      build_dns_query_payload(0x5678U, payload, sizeof(payload)));
    TEST_ASSERT_EQUAL_UINT8(0x56U, payload[0]);
    TEST_ASSERT_EQUAL_UINT8(0x78U, payload[1]);
    TEST_ASSERT_EQUAL_UINT8(0x01U, payload[2]);
    TEST_ASSERT_EQUAL_UINT8(0x00U, payload[3]);
    TEST_ASSERT_EQUAL_UINT8(0x01U, payload[5]);
    TEST_ASSERT_EQUAL_UINT8(0x01U, payload[14]);
    TEST_ASSERT_EQUAL_UINT8(0x01U, payload[16]);
}

void test_dns_tcp_query_frame_rejects_small_buffer(void) {
    uint8_t frame[TRAFFIC_DNS_TCP_FRAME_SIZE - 1U] = {};

    TEST_ASSERT_EQUAL(0U, build_dns_tcp_query_frame(1U, frame, sizeof(frame)));
    TEST_ASSERT_EQUAL(0U, build_dns_tcp_query_frame(1U, nullptr, 0U));
}

// ============================================================================
// ENTRY POINT
// ============================================================================

void test_traffic_generator_rejects_missing_gateway_or_rate_and_resets_pause(void) {
    TrafficGeneratorManager manager;
    for (RuntimeTrafficMode mode : {RuntimeTrafficMode::PING, RuntimeTrafficMode::DNS, RuntimeTrafficMode::DNS_TCP}) {
        manager.init(0U, mode);
        TEST_ASSERT_FALSE(manager.start(0x0101A8C0U));
        manager.init(100U, mode);
        TEST_ASSERT_FALSE(manager.start(0U));
        TEST_ASSERT_FALSE(manager.is_running());
        TEST_ASSERT_EQUAL(100U, manager.target_rate_pps());
        TEST_ASSERT_EQUAL(manager.target_rate_pps(), manager.current_rate_pps());
        manager.pause();
        TEST_ASSERT_TRUE(manager.is_paused());
        manager.loop();
        manager.resume();
        TEST_ASSERT_FALSE(manager.is_paused());
        manager.stop();
        TEST_ASSERT_EQUAL(0U, manager.send_success_count());
        TEST_ASSERT_EQUAL(0U, manager.send_error_count());
    }
}

int process(void) {
    UNITY_BEGIN();
    RUN_TEST(test_fixed_sta_rate_covers_every_internal_generator_and_running_lifecycle);
    RUN_TEST(test_ping_socket_failure_does_not_change_radio_or_create_task);
    RUN_TEST(test_station_rate_requires_ofdm_capable_access_point);
    RUN_TEST(test_ping_association_lookup_failure_closes_socket_before_task_creation);
    RUN_TEST(test_ping_rate_failure_closes_socket_before_task_creation);
    RUN_TEST(test_raw_sta_rate_failure_does_not_start_injection);
    RUN_TEST(test_ping_task_failure_restores_radio_and_closes_socket);
    RUN_TEST(test_failed_rate_restore_retains_ownership_until_explicit_cleanup);
    RUN_TEST(test_failed_startup_cleanup_can_be_retried_by_stop);
    RUN_TEST(test_external_service_start_retries_failed_ping_radio_cleanup);
    RUN_TEST(test_traffic_generator_rejects_missing_gateway_or_rate_and_resets_pause);
    
    // SendErrorState tests
    RUN_TEST(test_send_error_state_initialization);
    RUN_TEST(test_wifi_raw_targets_current_bssid_without_gateway_and_counts_sends);
    RUN_TEST(test_wifi_raw_uses_ofdm_for_each_band_and_rejects_rate_configuration_failure);
    RUN_TEST(test_wifi_raw_requires_association_and_valid_station_identity);
    
    // handle_send_error tests
    RUN_TEST(test_handle_send_error_increments_count);
    RUN_TEST(test_handle_send_error_rate_limits_logging);
    RUN_TEST(test_handle_send_error_returns_true_for_enomem);
    RUN_TEST(test_handle_send_error_returns_false_for_other_errors);
    RUN_TEST(test_handle_send_error_resets_window_after_interval);
    RUN_TEST(test_handle_send_error_resets_accumulated_errors_after_interval);
    RUN_TEST(test_handle_send_error_handles_negative_sent_value);
    RUN_TEST(test_pacing_deadline_starts_from_first_send);
    RUN_TEST(test_pacing_deadline_preserves_phase_across_small_jitter);
    RUN_TEST(test_pacing_deadline_resets_instead_of_catching_up);
    RUN_TEST(test_pacing_deadline_handles_invalid_interval);
    RUN_TEST(test_dns_tcp_query_frame_adds_length_and_transaction_id);
    RUN_TEST(test_dns_tcp_query_frame_rejects_small_buffer);
    RUN_TEST(test_dns_udp_query_payload_sets_transaction_id_without_tcp_length);
    
    return UNITY_END();
}

#if defined(ESP_PLATFORM)
extern "C" void app_main(void) { process(); }
#else
int main(int argc, char **argv) { return process(); }
#endif
