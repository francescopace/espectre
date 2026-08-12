/*
 * ESPectre - CsiPipeline Unit Tests
 *
 * Tests the CsiPipeline class functionality
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "test_harness.h"
#include <cstdint>
#include <cstring>
#include "lwip/inet.h"
#include "classic_detector.h"
#include "csi_pipeline.h"
#include "utils.h"
#include "wifi_csi_interface.h"
#include "esphome/core/log.h"
#include "esp_wifi.h"

using namespace espectre;


static constexpr uint32_t TEST_PUBLISH_INTERVAL_MS = 1000;
static constexpr uint32_t TEST_MOTION_PHASE_PACKETS = 100;
static constexpr uint32_t TEST_EVALUATION_INTERVAL_MS = 250;
static constexpr uint32_t TEST_PACKETS_PER_EVALUATION_AT_100_PPS = 25;
static constexpr uint32_t TEST_FIRST_EVALUATION_PACKET =
    TEST_PACKETS_PER_EVALUATION_AT_100_PPS + 1U;
static constexpr uint8_t TEST_DEFAULT_MOTION_ON_HITS = 4;
static constexpr uint32_t TEST_MOTION_CALLBACK_TRIGGER_PACKET =
    1U + TEST_PACKETS_PER_EVALUATION_AT_100_PPS * TEST_DEFAULT_MOTION_ON_HITS;

class TransitionDetectorMock : public BaseDetector {
 public:
  TransitionDetectorMock() : BaseDetector(10) {}

  void update_state() override {
    if (total_packets_ >= 2) {
      state_ = MotionState::MOTION;
    }
    current_metric_ = state_ == MotionState::MOTION ? 1.0f : 0.0f;
  }

  bool set_threshold(float threshold) override {
    threshold_ = threshold;
    return true;
  }

  float get_threshold() const override { return threshold_; }
  const char* get_name() const override { return "TransitionMock"; }

 private:
  float threshold_{0.0f};
};

class WindowedTransitionDetectorMock : public BaseDetector {
 public:
  WindowedTransitionDetectorMock() : BaseDetector(10) {}

  void update_state() override {
    if (total_packets_ <= TEST_MOTION_PHASE_PACKETS) {
      state_ = MotionState::MOTION;
    } else {
      state_ = MotionState::IDLE;
    }
    current_metric_ = state_ == MotionState::MOTION ? 1.0f : 0.0f;
  }

  bool set_threshold(float threshold) override {
    threshold_ = threshold;
    return true;
  }

  float get_threshold() const override { return threshold_; }
  const char* get_name() const override { return "WindowedTransitionMock"; }

 private:
  float threshold_{0.0f};
};

static void fill_valid_csi_info_(wifi_csi_info_t* csi_info, int8_t* csi_buf, uint8_t channel = 6) {
  for (int i = 0; i < 128; i++) {
    csi_buf[i] = static_cast<int8_t>(i % 64 - 32);
  }
  std::memset(csi_info, 0, sizeof(*csi_info));
  csi_info->buf = csi_buf;
  csi_info->len = 128;
  csi_info->rx_ctrl.channel = channel;
  // HT20 sensing contract: Classic/ML drop non-HT20 frames in the pipeline.
  csi_info->rx_ctrl.sig_mode = 1;
  csi_info->rx_ctrl.cwb = 0;
}

static void process_timed_packets_(CsiPipeline& manager, wifi_csi_info_t& csi_info,
                                   uint32_t& arrival_us, uint32_t count,
                                   uint32_t interval_us = 10000U) {
  for (uint32_t packet = 0; packet < count; packet++) {
    csi_info.rx_ctrl.timestamp = arrival_us;
    manager.process_packet(&csi_info);
    arrival_us += interval_us;
  }
}

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

void test_csi_pipeline_init(void) {
    ClassicDetector detector(50, 1.0f);
    CsiPipeline manager;
    
    manager.init(&detector, TEST_PUBLISH_INTERVAL_MS, &g_wifi_mock);
    
    TEST_ASSERT_FALSE(manager.is_enabled());
    TEST_ASSERT_NOT_NULL(manager.get_detector());
}

// ============================================================================
// ENABLE/DISABLE TESTS
// ============================================================================

void test_csi_pipeline_enable(void) {
    ClassicDetector detector(50, 1.0f);
    CsiPipeline manager;
    manager.init(&detector, TEST_PUBLISH_INTERVAL_MS, &g_wifi_mock);
    
    esp_err_t err = manager.enable();
    
    TEST_ASSERT_EQUAL(ESP_OK, err);
    TEST_ASSERT_TRUE(manager.is_enabled());
    TEST_ASSERT_TRUE(g_wifi_mock.is_enabled());
}

void test_csi_pipeline_enable_twice_returns_ok(void) {
    ClassicDetector detector(50, 1.0f);
    CsiPipeline manager;
    manager.init(&detector, TEST_PUBLISH_INTERVAL_MS, &g_wifi_mock);
    
    manager.enable();
    esp_err_t err = manager.enable();
    
    TEST_ASSERT_EQUAL(ESP_OK, err);
    TEST_ASSERT_TRUE(manager.is_enabled());
}

void test_csi_pipeline_disable(void) {
    ClassicDetector detector(50, 1.0f);
    CsiPipeline manager;
    manager.init(&detector, TEST_PUBLISH_INTERVAL_MS, &g_wifi_mock);
    
    manager.enable();
    esp_err_t err = manager.disable();
    
    TEST_ASSERT_EQUAL(ESP_OK, err);
    TEST_ASSERT_FALSE(manager.is_enabled());
}

void test_csi_pipeline_disable_preserves_stable_callbacks_for_reenable(void) {
    TransitionDetectorMock detector;
    CsiPipeline manager;
    manager.init(&detector, TEST_PUBLISH_INTERVAL_MS, &g_wifi_mock);
    manager.set_evaluation_interval_ms(10);
    manager.set_motion_on_hits(1);
    manager.set_motion_off_hits(1);

    int motion_callback_count = 0;
    int live_telemetry_callback_count = 0;
    manager.set_motion_state_callback([&](MotionState) {
        motion_callback_count++;
    });
    manager.set_live_telemetry_callback([&](float, float) {
        live_telemetry_callback_count++;
    });

    int8_t csi_buf[128];
    wifi_csi_info_t csi_info = {};
    fill_valid_csi_info_(&csi_info, csi_buf);
    uint32_t arrival_us = 1000000U;

    TEST_ASSERT_EQUAL(ESP_OK, manager.enable());
    process_timed_packets_(manager, csi_info, arrival_us, 2U, 10000U);
    TEST_ASSERT_EQUAL(1, live_telemetry_callback_count);

    TEST_ASSERT_EQUAL(ESP_OK, manager.disable());
    TEST_ASSERT_EQUAL(ESP_OK, manager.enable());
    process_timed_packets_(manager, csi_info, arrival_us, 2U, 10000U);

    TEST_ASSERT_EQUAL(2, live_telemetry_callback_count);
    TEST_ASSERT_TRUE(motion_callback_count >= 1);
}

void test_csi_pipeline_disable_when_not_enabled(void) {
    ClassicDetector detector(50, 1.0f);
    CsiPipeline manager;
    manager.init(&detector, TEST_PUBLISH_INTERVAL_MS, &g_wifi_mock);
    
    esp_err_t err = manager.disable();
    
    TEST_ASSERT_EQUAL(ESP_OK, err);
    TEST_ASSERT_FALSE(manager.is_enabled());
}

// ============================================================================
// THRESHOLD TESTS
// ============================================================================

void test_csi_pipeline_set_threshold(void) {
    ClassicDetector detector(50, 1.0f);
    CsiPipeline manager;
    manager.init(&detector, TEST_PUBLISH_INTERVAL_MS, &g_wifi_mock);
    
    manager.set_threshold(0.75f);
    
    TEST_ASSERT_EQUAL_FLOAT(0.75f, detector.get_threshold());
}

// ============================================================================
// PROCESS PACKET TESTS
// ============================================================================

void test_csi_pipeline_process_packet_null_data(void) {
    ClassicDetector detector(50, 1.0f);
    CsiPipeline manager;
    manager.init(&detector, TEST_PUBLISH_INTERVAL_MS, &g_wifi_mock);
    
    manager.process_packet(nullptr);
    
    TEST_ASSERT_EQUAL(MotionState::IDLE, detector.get_state());
}

void test_csi_pipeline_process_packet_short_data(void) {
    ClassicDetector detector(50, 1.0f);
    CsiPipeline manager;
    manager.init(&detector, TEST_PUBLISH_INTERVAL_MS, &g_wifi_mock);
    
    wifi_csi_info_t csi_info = {};
    int8_t short_buf[5] = {0};
    csi_info.buf = short_buf;
    csi_info.len = 5;
    
    manager.process_packet(&csi_info);
    
    TEST_ASSERT_EQUAL(MotionState::IDLE, detector.get_state());
}

void test_csi_pipeline_counts_valid_local_packets_for_traffic_feedback(void) {
    ClassicDetector detector(50, 1.0f);
    CsiPipeline manager;
    manager.init(&detector, TEST_PUBLISH_INTERVAL_MS, &g_wifi_mock);

    int8_t csi_buf[128];
    wifi_csi_info_t csi_info{};
    fill_valid_csi_info_(&csi_info, csi_buf);
    manager.process_packet(&csi_info);

    TEST_ASSERT_EQUAL(1U, manager.accepted_packets_total());

    csi_info.len = 5U;
    manager.process_packet(&csi_info);
    TEST_ASSERT_EQUAL(1U, manager.accepted_packets_total());
}

void test_csi_pipeline_process_packet_valid_data(void) {
    ClassicDetector detector(50, 1.0f);
    CsiPipeline manager;
    manager.init(&detector, TEST_PUBLISH_INTERVAL_MS, &g_wifi_mock);
    
    // Create valid CSI data (128 bytes for HT20)
    int8_t csi_buf[128];
    wifi_csi_info_t csi_info = {};
    fill_valid_csi_info_(&csi_info, csi_buf);
    
    manager.process_packet(&csi_info);
    
    TEST_ASSERT_EQUAL(1, detector.get_total_packets());
}

void test_csi_pipeline_holds_detector_below_supported_packet_rate(void) {
    ClassicDetector detector(80, 1.0f);
    CsiPipeline manager;
    manager.init(&detector, TEST_PUBLISH_INTERVAL_MS, &g_wifi_mock);

    int8_t csi_buf[128];
    wifi_csi_info_t csi_info = {};
    fill_valid_csi_info_(&csi_info, csi_buf);
    uint32_t timestamp = 1000000U;
    for (int i = 0; i < 20; ++i) {
        csi_info.rx_ctrl.timestamp = timestamp;
        manager.process_packet(&csi_info);
        timestamp += 20000U;
    }
    TEST_ASSERT_EQUAL(0U, detector.get_total_packets());

    for (int i = 0; i < 80; ++i) {
        csi_info.rx_ctrl.timestamp = timestamp;
        manager.process_packet(&csi_info);
        timestamp += 10000U;
    }
    TEST_ASSERT_TRUE(detector.get_total_packets() > 0U);
}

void test_csi_pipeline_filters_duplicate_and_stale_rx_timestamps(void) {
    ClassicDetector detector(10, 1.0f);
    CsiPipeline manager;
    manager.init(&detector, TEST_PUBLISH_INTERVAL_MS, &g_wifi_mock);

    int8_t csi_buf[128];
    wifi_csi_info_t csi_info = {};
    fill_valid_csi_info_(&csi_info, csi_buf);

    csi_info.rx_ctrl.timestamp = 100U;
    manager.process_packet(&csi_info);
    csi_info.rx_ctrl.timestamp = 101U;
    manager.process_packet(&csi_info);
    manager.process_packet(&csi_info);
    csi_info.rx_ctrl.timestamp = 50U;
    manager.process_packet(&csi_info);
    csi_info.rx_ctrl.timestamp = 102U;
    manager.process_packet(&csi_info);

    TEST_ASSERT_EQUAL(3U, detector.get_total_packets());
    TEST_ASSERT_EQUAL(3U, manager.accepted_packets_total());
    TEST_ASSERT_EQUAL(2U, manager.rejected_out_of_order_packets_total());
}

void test_csi_pipeline_accepts_rx_timestamp_wrap(void) {
    ClassicDetector detector(10, 1.0f);
    CsiPipeline manager;
    manager.init(&detector, TEST_PUBLISH_INTERVAL_MS, &g_wifi_mock);

    int8_t csi_buf[128];
    wifi_csi_info_t csi_info = {};
    fill_valid_csi_info_(&csi_info, csi_buf);

    const uint32_t timestamps[] = {UINT32_MAX - 1U, UINT32_MAX, 0U, 1U};
    for (uint32_t timestamp : timestamps) {
        csi_info.rx_ctrl.timestamp = timestamp;
        manager.process_packet(&csi_info);
    }

    TEST_ASSERT_EQUAL(4U, detector.get_total_packets());
    TEST_ASSERT_EQUAL(4U, manager.accepted_packets_total());
    TEST_ASSERT_EQUAL(0U, manager.rejected_out_of_order_packets_total());
}

void test_csi_pipeline_filters_non_ht20_phy(void) {
    ClassicDetector detector(50, 1.0f);
    CsiPipeline manager;
    manager.init(&detector, TEST_PUBLISH_INTERVAL_MS, &g_wifi_mock);

    int8_t csi_buf[128];
    wifi_csi_info_t csi_info = {};
    fill_valid_csi_info_(&csi_info, csi_buf);

    csi_info.rx_ctrl.sig_mode = 0;  // legacy OFDM
    manager.process_packet(&csi_info);
    TEST_ASSERT_EQUAL(0, detector.get_total_packets());
    TEST_ASSERT_EQUAL(0U, manager.accepted_packets_total());

    csi_info.rx_ctrl.sig_mode = 1;
    csi_info.rx_ctrl.cwb = 1;  // HT40
    manager.process_packet(&csi_info);
    TEST_ASSERT_EQUAL(0, detector.get_total_packets());
    TEST_ASSERT_EQUAL(0U, manager.accepted_packets_total());

    csi_info.rx_ctrl.cwb = 0;  // HT20
    manager.process_packet(&csi_info);
    TEST_ASSERT_EQUAL(1, detector.get_total_packets());
    TEST_ASSERT_EQUAL(1U, manager.accepted_packets_total());
}

void test_csi_pipeline_motion_state_callback_fires_before_periodic_publish(void) {
    TransitionDetectorMock detector;
    CsiPipeline manager;
    manager.init(&detector, TEST_PUBLISH_INTERVAL_MS, &g_wifi_mock);
    manager.set_motion_on_hits(1);
    manager.set_motion_off_hits(1);

    int motion_callback_count = 0;
    MotionState last_motion_state = MotionState::IDLE;
    int periodic_callback_count = 0;
    manager.set_live_telemetry_callback([](float, float) {});
    manager.set_motion_state_callback([&](MotionState state) {
        motion_callback_count++;
        last_motion_state = state;
    });

    manager.enable([&](MotionState, uint32_t) {
        periodic_callback_count++;
    });

    int8_t csi_buf[128];
    wifi_csi_info_t csi_info = {};
    fill_valid_csi_info_(&csi_info, csi_buf);
    uint32_t arrival_us = 1000000U;

    process_timed_packets_(manager, csi_info, arrival_us,
                           TEST_FIRST_EVALUATION_PACKET - 1U);

    TEST_ASSERT_EQUAL(0, motion_callback_count);
    TEST_ASSERT_EQUAL(0, periodic_callback_count);

    process_timed_packets_(manager, csi_info, arrival_us, 1U);

    TEST_ASSERT_EQUAL(1, motion_callback_count);
    TEST_ASSERT_EQUAL(MotionState::MOTION, last_motion_state);
    TEST_ASSERT_EQUAL(0, periodic_callback_count);
}

void test_csi_pipeline_motion_state_callback_does_not_repeat_without_new_edge(void) {
    TransitionDetectorMock detector;
    CsiPipeline manager;
    manager.init(&detector, TEST_PUBLISH_INTERVAL_MS, &g_wifi_mock);

    int motion_callback_count = 0;
    manager.set_live_telemetry_callback([](float, float) {});
    manager.set_motion_state_callback([&](MotionState) {
        motion_callback_count++;
    });

    int8_t csi_buf[128];
    wifi_csi_info_t csi_info = {};
    fill_valid_csi_info_(&csi_info, csi_buf);
    uint32_t arrival_us = 1000000U;

    process_timed_packets_(manager, csi_info, arrival_us,
                           TEST_MOTION_CALLBACK_TRIGGER_PACKET);

    TEST_ASSERT_EQUAL(1, motion_callback_count);
}

void test_csi_pipeline_clear_detector_buffer_publishes_idle_edge(void) {
    TransitionDetectorMock detector;
    CsiPipeline manager;
    manager.init(&detector, TEST_PUBLISH_INTERVAL_MS, &g_wifi_mock);
    manager.set_motion_on_hits(1);
    manager.set_motion_off_hits(1);

    int motion_callback_count = 0;
    MotionState last_motion_state = MotionState::IDLE;
    manager.set_live_telemetry_callback([](float, float) {});
    manager.set_motion_state_callback([&](MotionState state) {
        motion_callback_count++;
        last_motion_state = state;
    });

    int8_t csi_buf[128];
    wifi_csi_info_t csi_info = {};
    fill_valid_csi_info_(&csi_info, csi_buf);
    uint32_t arrival_us = 1000000U;

    process_timed_packets_(manager, csi_info, arrival_us,
                           TEST_FIRST_EVALUATION_PACKET);
    manager.clear_detector_buffer();

    TEST_ASSERT_EQUAL(2, motion_callback_count);
    TEST_ASSERT_EQUAL(MotionState::IDLE, last_motion_state);
}

void test_csi_pipeline_motion_state_callback_honors_motion_on_hits(void) {
    TransitionDetectorMock detector;
    CsiPipeline manager;
    manager.init(&detector, TEST_PUBLISH_INTERVAL_MS, &g_wifi_mock);
    manager.set_motion_on_hits(3);

    int motion_callback_count = 0;
    MotionState last_motion_state = MotionState::IDLE;
    manager.set_live_telemetry_callback([](float, float) {});
    manager.set_motion_state_callback([&](MotionState state) {
        motion_callback_count++;
        last_motion_state = state;
    });

    int8_t csi_buf[128];
    wifi_csi_info_t csi_info = {};
    fill_valid_csi_info_(&csi_info, csi_buf);
    uint32_t arrival_us = 1000000U;
    const uint32_t trigger_packet =
        1U + TEST_PACKETS_PER_EVALUATION_AT_100_PPS * 3U;

    process_timed_packets_(manager, csi_info, arrival_us,
                           trigger_packet - 1U);

    TEST_ASSERT_EQUAL(0, motion_callback_count);

    process_timed_packets_(manager, csi_info, arrival_us, 1U);

    TEST_ASSERT_EQUAL(1, motion_callback_count);
    TEST_ASSERT_EQUAL(MotionState::MOTION, last_motion_state);
}

void test_csi_pipeline_motion_state_callback_honors_motion_off_hits(void) {
    WindowedTransitionDetectorMock detector;
    CsiPipeline manager;
    manager.init(&detector, TEST_PUBLISH_INTERVAL_MS, &g_wifi_mock);
    manager.set_motion_on_hits(2);
    manager.set_motion_off_hits(3);
    manager.set_evaluation_interval_ms(TEST_EVALUATION_INTERVAL_MS);

    int motion_callback_count = 0;
    MotionState last_motion_state = MotionState::IDLE;
    manager.set_live_telemetry_callback([](float, float) {});
    manager.set_motion_state_callback([&](MotionState state) {
        motion_callback_count++;
        last_motion_state = state;
    });

    int8_t csi_buf[128];
    wifi_csi_info_t csi_info = {};
    fill_valid_csi_info_(&csi_info, csi_buf);

    // The first tick needs two timestamps to establish elapsed time, so it is
    // packet 26 at 100 pps; subsequent ticks remain 25 packets apart.
    uint32_t arrival_us = 1000000U;
    process_timed_packets_(manager, csi_info, arrival_us, 176U);

    TEST_ASSERT_EQUAL(2, motion_callback_count);
    TEST_ASSERT_EQUAL(MotionState::IDLE, last_motion_state);
}

void test_csi_pipeline_periodic_callback_uses_filtered_motion_state(void) {
    TransitionDetectorMock detector;
    CsiPipeline manager;
    manager.init(&detector, TEST_PUBLISH_INTERVAL_MS, &g_wifi_mock);
    manager.set_motion_on_hits(3);

    int periodic_callback_count = 0;
    MotionState periodic_state = MotionState::MOTION;
    uint32_t periodic_packet_count = 0U;
    manager.set_live_telemetry_callback([](float, float) {});
    manager.enable([&](MotionState state, uint32_t packets_received) {
        periodic_callback_count++;
        periodic_state = state;
        periodic_packet_count = packets_received;
    });

    int8_t csi_buf[128];
    wifi_csi_info_t csi_info = {};
    fill_valid_csi_info_(&csi_info, csi_buf);

    manager.publish_if_due(1000U);
    uint32_t arrival_us = 1000000U;
    process_timed_packets_(manager, csi_info, arrival_us,
                           TEST_FIRST_EVALUATION_PACKET);
    manager.publish_if_due(1999U);

    TEST_ASSERT_EQUAL(0, periodic_callback_count);

    manager.publish_if_due(2000U);

    TEST_ASSERT_EQUAL(1, periodic_callback_count);
    TEST_ASSERT_EQUAL(MotionState::IDLE, periodic_state);
    TEST_ASSERT_EQUAL(TEST_FIRST_EVALUATION_PACKET, periodic_packet_count);
}

void test_csi_pipeline_periodic_callback_reports_zero_packets_when_idle(void) {
    TransitionDetectorMock detector;
    CsiPipeline manager;
    manager.init(&detector, TEST_PUBLISH_INTERVAL_MS, &g_wifi_mock);

    int periodic_callback_count = 0;
    uint32_t periodic_packet_count = UINT32_MAX;
    manager.enable([&](MotionState, uint32_t packets_received) {
        periodic_callback_count++;
        periodic_packet_count = packets_received;
    });

    manager.publish_if_due(1000U);
    manager.publish_if_due(2000U);

    TEST_ASSERT_EQUAL(1, periodic_callback_count);
    TEST_ASSERT_EQUAL(0U, periodic_packet_count);
}

/** Replay one stream at a chosen cadence and count evaluation ticks. */
static int count_evaluations_at_cadence_(uint32_t interval_us, int packets) {
    TransitionDetectorMock detector;
    CsiPipeline manager;
    manager.init(&detector, TEST_PUBLISH_INTERVAL_MS, &g_wifi_mock);
    manager.set_motion_on_hits(1);
    manager.set_motion_off_hits(1);

    int evaluations = 0;
    manager.set_live_telemetry_callback([&](float, float) { evaluations++; });

    int8_t csi_buf[128];
    wifi_csi_info_t csi_info = {};
    fill_valid_csi_info_(&csi_info, csi_buf);

    uint32_t arrival_us = 1000000U;
    for (int i = 0; i < packets; i++) {
        csi_info.rx_ctrl.timestamp = arrival_us;
        manager.process_packet(&csi_info);
        arrival_us += interval_us;
    }
    return evaluations;
}

namespace {

struct InterceptorProbe {
    int calls{0};
    int evaluations_due{0};
    uint32_t last_packets_in_window{0U};
    uint32_t max_packets_in_window{0U};
};

bool interceptor_probe_callback_(void *context, const int8_t *csi_data, size_t csi_len,
                                 int8_t rssi_dbm, bool evaluation_due,
                                 uint32_t packets_in_window) {
    (void) csi_data;
    (void) csi_len;
    (void) rssi_dbm;
    auto *probe = static_cast<InterceptorProbe *>(context);
    probe->calls++;
    if (evaluation_due) {
        probe->evaluations_due++;
        probe->last_packets_in_window = packets_in_window;
        if (packets_in_window > probe->max_packets_in_window) {
            probe->max_packets_in_window = packets_in_window;
        }
    }
    return true;  // consume, exactly like startup calibration does
}

}  // namespace

// Startup calibration consumes every packet through the interceptor. The
// cadence used to be advanced only on the detection path, so the rate estimator
// was starved for the whole ~1000-packet calibration. On an off-nominal stream
// the threshold was then fitted at a resolution the detector never ran at.
void test_csi_pipeline_feeds_cadence_while_interceptor_consumes(void) {
    TransitionDetectorMock detector;
    CsiPipeline manager;
    manager.init(&detector, TEST_PUBLISH_INTERVAL_MS, &g_wifi_mock);

    InterceptorProbe probe;
    manager.set_packet_interceptor(&interceptor_probe_callback_, &probe);

    int8_t csi_buf[128];
    wifi_csi_info_t csi_info = {};
    fill_valid_csi_info_(&csi_info, csi_buf);

    // 500 packets at 500 pps is one second of stream.
    uint32_t arrival_us = 1000000U;
    for (int i = 0; i < 500; i++) {
        csi_info.rx_ctrl.timestamp = arrival_us;
        manager.process_packet(&csi_info);
        arrival_us += 2000U;
    }

    TEST_ASSERT_EQUAL(500, probe.calls);
    // One second at the 250 ms contract is four ticks, not the twenty a packet
    // count of 25 would have produced at this rate.
    TEST_ASSERT_TRUE(probe.evaluations_due >= 3 && probe.evaluations_due <= 6);
    // Each closed window carries its own weight, which is what the calibrator
    // folds in one step.
    TEST_ASSERT_TRUE(probe.max_packets_in_window >= 100U);
}

// The interceptor and the detection path must agree on when a window closes.
void test_csi_pipeline_interceptor_shares_the_detection_cadence(void) {
    const int detection_ticks = count_evaluations_at_cadence_(10000U, 1000);

    TransitionDetectorMock detector;
    CsiPipeline manager;
    manager.init(&detector, TEST_PUBLISH_INTERVAL_MS, &g_wifi_mock);
    InterceptorProbe probe;
    manager.set_packet_interceptor(&interceptor_probe_callback_, &probe);

    int8_t csi_buf[128];
    wifi_csi_info_t csi_info = {};
    fill_valid_csi_info_(&csi_info, csi_buf);
    uint32_t arrival_us = 1000000U;
    for (int i = 0; i < 1000; i++) {
        csi_info.rx_ctrl.timestamp = arrival_us;
        manager.process_packet(&csi_info);
        arrival_us += 10000U;
    }

    TEST_ASSERT_EQUAL(detection_ticks, probe.evaluations_due);
}

void test_csi_pipeline_evaluates_on_elapsed_packet_time(void) {
    // Arrival time is an input, so the cadence is reproducible run to run.
    // The first packet establishes the time origin, leaving 29.99 s of elapsed
    // coverage and therefore 119 complete 250 ms evaluation intervals.
    TEST_ASSERT_EQUAL(119, count_evaluations_at_cadence_(10000U, 3000));

    // The same elapsed span delivered five times faster produces the same
    // number of evaluations because publishing cannot force detector work.
    TEST_ASSERT_EQUAL(119, count_evaluations_at_cadence_(2000U, 15000));
}

void test_csi_pipeline_live_telemetry_callback_does_not_force_every_packet_evaluation(void) {
    TransitionDetectorMock detector;
    CsiPipeline manager;
    manager.init(&detector, TEST_PUBLISH_INTERVAL_MS, &g_wifi_mock);
    manager.set_motion_on_hits(1);
    manager.set_motion_off_hits(1);

    int motion_callback_count = 0;
    int live_telemetry_callback_count = 0;
    manager.set_live_telemetry_callback([&](float, float) {
        live_telemetry_callback_count++;
    });
    manager.set_motion_state_callback([&](MotionState) {
        motion_callback_count++;
    });

    int8_t csi_buf[128];
    wifi_csi_info_t csi_info = {};
    fill_valid_csi_info_(&csi_info, csi_buf);
    uint32_t arrival_us = 1000000U;

    process_timed_packets_(manager, csi_info, arrival_us,
                           TEST_FIRST_EVALUATION_PACKET - 1U);

    TEST_ASSERT_EQUAL(0, motion_callback_count);
    TEST_ASSERT_EQUAL(0, live_telemetry_callback_count);

    process_timed_packets_(manager, csi_info, arrival_us, 1U);

    TEST_ASSERT_EQUAL(1, motion_callback_count);
    TEST_ASSERT_EQUAL(1, live_telemetry_callback_count);
}

// ============================================================================
// STBC PACKET TESTS (GitHub issue #76)
// ============================================================================

void test_csi_pipeline_process_stbc_256_byte_packet(void) {
    ClassicDetector detector(50, 1.0f);
    CsiPipeline manager;
    manager.init(&detector, TEST_PUBLISH_INTERVAL_MS, &g_wifi_mock);
    
    // STBC packet: 256 bytes (2x HT-LTF, 128 SC) — should be truncated to 128
    int8_t csi_buf[256];
    for (int i = 0; i < 256; i++) {
        csi_buf[i] = (int8_t)(i % 64 - 32);
    }
    
    wifi_csi_info_t csi_info = {};
    csi_info.buf = csi_buf;
    csi_info.len = 256;
    csi_info.rx_ctrl.channel = 6;
    csi_info.rx_ctrl.sig_mode = 1;
    csi_info.rx_ctrl.cwb = 0;
    csi_info.rx_ctrl.stbc = 1;
    
    manager.process_packet(&csi_info);
    
    TEST_ASSERT_EQUAL(1, detector.get_total_packets());
}

void test_csi_pipeline_process_short_ht_114_byte_packet(void) {
    ClassicDetector detector(50, 1.0f);
    CsiPipeline manager;
    manager.init(&detector, TEST_PUBLISH_INTERVAL_MS, &g_wifi_mock);

    // Short HT packet: 114 bytes (57 SC) — should be remapped to 128 and processed.
    int8_t csi_buf[114];
    for (int i = 0; i < 114; i++) {
        csi_buf[i] = (int8_t)(i % 64 - 32);
    }

    wifi_csi_info_t csi_info = {};
    csi_info.buf = csi_buf;
    csi_info.len = 114;
    csi_info.rx_ctrl.channel = 6;
    csi_info.rx_ctrl.sig_mode = 1;
    csi_info.rx_ctrl.cwb = 0;

    manager.process_packet(&csi_info);

    TEST_ASSERT_EQUAL(1, detector.get_total_packets());
}

void test_csi_pipeline_process_double_short_ht_228_byte_packet(void) {
    ClassicDetector detector(50, 1.0f);
    CsiPipeline manager;
    manager.init(&detector, TEST_PUBLISH_INTERVAL_MS, &g_wifi_mock);

    // Doubled short HT packet: 228 bytes (2 x 114) — should collapse to 114,
    // then remap to 128 and be processed.
    int8_t csi_buf[228];
    for (int i = 0; i < 228; i++) {
        csi_buf[i] = (int8_t)(i % 64 - 32);
    }

    wifi_csi_info_t csi_info = {};
    csi_info.buf = csi_buf;
    csi_info.len = 228;
    csi_info.rx_ctrl.channel = 6;
    csi_info.rx_ctrl.sig_mode = 1;
    csi_info.rx_ctrl.cwb = 0;

    manager.process_packet(&csi_info);

    TEST_ASSERT_EQUAL(1, detector.get_total_packets());
}

void test_csi_pipeline_process_wrong_length_filtered(void) {
    ClassicDetector detector(50, 1.0f);
    CsiPipeline manager;
    manager.init(&detector, TEST_PUBLISH_INTERVAL_MS, &g_wifi_mock);
    
    // 64 bytes — not HT20 (128) nor STBC (256), must be filtered
    int8_t csi_buf[64];
    memset(csi_buf, 0, sizeof(csi_buf));
    
    wifi_csi_info_t csi_info = {};
    csi_info.buf = csi_buf;
    csi_info.len = 64;
    csi_info.rx_ctrl.channel = 6;
    
    manager.process_packet(&csi_info);
    
    TEST_ASSERT_EQUAL(0, detector.get_total_packets());
}

// ============================================================================
// ERROR PATH TESTS
// ============================================================================

void test_csi_pipeline_enable_config_error(void) {
    ClassicDetector detector(50, 1.0f);
    CsiPipeline manager;
    manager.init(&detector, TEST_PUBLISH_INTERVAL_MS, &g_wifi_mock);
    
    g_wifi_mock.set_config_error(ESP_ERR_INVALID_ARG);
    
    esp_err_t result = manager.enable(nullptr);
    
    TEST_ASSERT_EQUAL(ESP_ERR_INVALID_ARG, result);
    TEST_ASSERT_FALSE(manager.is_enabled());
}

void test_csi_pipeline_enable_callback_error(void) {
    ClassicDetector detector(50, 1.0f);
    CsiPipeline manager;
    manager.init(&detector, TEST_PUBLISH_INTERVAL_MS, &g_wifi_mock);
    
    g_wifi_mock.set_callback_error(ESP_ERR_NO_MEM);
    
    esp_err_t result = manager.enable(nullptr);
    
    TEST_ASSERT_EQUAL(ESP_ERR_NO_MEM, result);
    TEST_ASSERT_FALSE(manager.is_enabled());
}

void test_csi_pipeline_enable_csi_error(void) {
    ClassicDetector detector(50, 1.0f);
    CsiPipeline manager;
    manager.init(&detector, TEST_PUBLISH_INTERVAL_MS, &g_wifi_mock);
    
    g_wifi_mock.set_csi_error(ESP_FAIL);
    
    esp_err_t result = manager.enable(nullptr);
    
    TEST_ASSERT_EQUAL(ESP_FAIL, result);
    TEST_ASSERT_FALSE(manager.is_enabled());
}

void test_csi_pipeline_disable_error(void) {
    ClassicDetector detector(50, 1.0f);
    CsiPipeline manager;
    manager.init(&detector, TEST_PUBLISH_INTERVAL_MS, &g_wifi_mock);
    
    manager.enable(nullptr);
    g_wifi_mock.set_csi_error(ESP_FAIL);
    
    esp_err_t result = manager.disable();
    
    TEST_ASSERT_EQUAL(ESP_FAIL, result);
    TEST_ASSERT_TRUE(manager.is_enabled());
}

// ============================================================================
// CALLBACK WRAPPER TESTS
// ============================================================================

void test_csi_pipeline_callback_wrapper_triggered(void) {
    ClassicDetector detector(50, 1.0f);
    CsiPipeline manager;
    manager.init(&detector, TEST_PUBLISH_INTERVAL_MS, &g_wifi_mock);
    
    manager.enable(nullptr);
    
    int8_t csi_buf[128] = {0};
    wifi_csi_info_t csi_info = {};
    fill_valid_csi_info_(&csi_info, csi_buf);
    
    g_wifi_mock.trigger_callback(&csi_info);
    
    TEST_ASSERT_TRUE(detector.get_total_packets() > 0);
}

void test_csi_pipeline_callback_wrapper_null_data(void) {
    ClassicDetector detector(50, 1.0f);
    CsiPipeline manager;
    manager.init(&detector, TEST_PUBLISH_INTERVAL_MS, &g_wifi_mock);
    
    manager.enable(nullptr);
    
    uint32_t packets_before = detector.get_total_packets();
    
    g_wifi_mock.trigger_callback(nullptr);
    
    TEST_ASSERT_EQUAL(packets_before, detector.get_total_packets());
}

// ============================================================================
// CLEAR DETECTOR BUFFER TEST
// ============================================================================

void test_csi_pipeline_clear_detector_buffer(void) {
    ClassicDetector detector(50, 1.0f);
    CsiPipeline manager;
    manager.init(&detector, TEST_PUBLISH_INTERVAL_MS, &g_wifi_mock);
    
    // Process some packets
    int8_t csi_buf[128] = {0};
    wifi_csi_info_t csi_info = {};
    fill_valid_csi_info_(&csi_info, csi_buf);
    
    for (int i = 0; i < 10; i++) {
        manager.process_packet(&csi_info);
    }
    
    // Clear buffer
    manager.clear_detector_buffer();
    
    // Detector should be reset
    TEST_ASSERT_EQUAL_FLOAT(0.0f, detector.get_motion_metric());
}

void test_csi_pipeline_aggregates_detection_timing_on_evaluation_ticks(void) {
    ClassicDetector detector(10, 1.0f);
    CsiPipeline manager;
    manager.init(&detector, TEST_PUBLISH_INTERVAL_MS, &g_wifi_mock);
    manager.set_evaluation_interval_ms(TEST_EVALUATION_INTERVAL_MS);

    int8_t csi_buf[128];
    wifi_csi_info_t csi_info = {};
    fill_valid_csi_info_(&csi_info, csi_buf);
    uint32_t arrival_us = 1000000U;

    process_timed_packets_(manager, csi_info, arrival_us,
                           TEST_FIRST_EVALUATION_PACKET - 1U);

    DetectionTimingStats timing;
    TEST_ASSERT_FALSE(manager.take_detection_timing(&timing));

    process_timed_packets_(manager, csi_info, arrival_us, 1U);
    process_timed_packets_(manager, csi_info, arrival_us,
                           TEST_PACKETS_PER_EVALUATION_AT_100_PPS);

    TEST_ASSERT_TRUE(manager.take_detection_timing(&timing));
    TEST_ASSERT_EQUAL_INT(2, timing.samples);
    TEST_ASSERT_TRUE(timing.duration_sum_us > 0U);
    TEST_ASSERT_TRUE(timing.minimum_us > 0U);
    TEST_ASSERT_TRUE(timing.maximum_us >= timing.minimum_us);
    TEST_ASSERT_TRUE(timing.duration_sum_us >= timing.minimum_us + timing.maximum_us);
    TEST_ASSERT_FALSE(manager.take_detection_timing(&timing));
    TEST_ASSERT_FALSE(manager.take_detection_timing(nullptr));
}

// ============================================================================
// LEGACY NORMALIZATION TESTS
// ============================================================================

void test_csi_pipeline_filters_unicast_frames_for_other_device(void) {
    ClassicDetector detector(10, 1.0f);
    CsiPipeline manager;
    manager.init(&detector, TEST_PUBLISH_INTERVAL_MS, &g_wifi_mock);

    const uint8_t local_mac[6] = {0x10, 0x20, 0x30, 0x40, 0x50, 0x60};
    const uint8_t other_mac[6] = {0x66, 0x55, 0x44, 0x33, 0x22, 0x11};
    manager.set_local_identity(inet_addr("192.168.1.17"), local_mac);

    int8_t csi_buf[128];
    wifi_csi_info_t csi_info = {};
    fill_valid_csi_info_(&csi_info, csi_buf);
    std::memcpy(csi_info.dmac, other_mac, sizeof(other_mac));

    manager.enable(nullptr);
    manager.process_packet(&csi_info);

    TEST_ASSERT_EQUAL(MotionState::IDLE, detector.get_state());
    TEST_ASSERT_EQUAL_FLOAT(0.0f, detector.get_motion_metric());
    TEST_ASSERT_EQUAL(0U, manager.accepted_packets_total());
}

// ============================================================================
// ENTRY POINT
// ============================================================================

int process(void) {
    UNITY_BEGIN();
    
    // Initialization tests
    RUN_TEST(test_csi_pipeline_init);
    
    // Enable/Disable tests
    RUN_TEST(test_csi_pipeline_enable);
    RUN_TEST(test_csi_pipeline_enable_twice_returns_ok);
    RUN_TEST(test_csi_pipeline_disable);
    RUN_TEST(test_csi_pipeline_disable_preserves_stable_callbacks_for_reenable);
    RUN_TEST(test_csi_pipeline_disable_when_not_enabled);
    
    // Threshold tests
    RUN_TEST(test_csi_pipeline_set_threshold);
    
    // Process packet tests
    RUN_TEST(test_csi_pipeline_process_packet_null_data);
    RUN_TEST(test_csi_pipeline_process_packet_short_data);
    RUN_TEST(test_csi_pipeline_counts_valid_local_packets_for_traffic_feedback);
    RUN_TEST(test_csi_pipeline_process_packet_valid_data);
    RUN_TEST(test_csi_pipeline_holds_detector_below_supported_packet_rate);
    RUN_TEST(test_csi_pipeline_filters_duplicate_and_stale_rx_timestamps);
    RUN_TEST(test_csi_pipeline_accepts_rx_timestamp_wrap);
    RUN_TEST(test_csi_pipeline_filters_non_ht20_phy);
    RUN_TEST(test_csi_pipeline_motion_state_callback_fires_before_periodic_publish);
    RUN_TEST(test_csi_pipeline_motion_state_callback_does_not_repeat_without_new_edge);
    RUN_TEST(test_csi_pipeline_clear_detector_buffer_publishes_idle_edge);
    RUN_TEST(test_csi_pipeline_motion_state_callback_honors_motion_on_hits);
    RUN_TEST(test_csi_pipeline_motion_state_callback_honors_motion_off_hits);
    RUN_TEST(test_csi_pipeline_periodic_callback_uses_filtered_motion_state);
    RUN_TEST(test_csi_pipeline_periodic_callback_reports_zero_packets_when_idle);
    RUN_TEST(test_csi_pipeline_evaluates_on_elapsed_packet_time);
    RUN_TEST(test_csi_pipeline_feeds_cadence_while_interceptor_consumes);
    RUN_TEST(test_csi_pipeline_interceptor_shares_the_detection_cadence);
    RUN_TEST(test_csi_pipeline_live_telemetry_callback_does_not_force_every_packet_evaluation);
    
    // STBC packet tests (issue #76)
    RUN_TEST(test_csi_pipeline_process_stbc_256_byte_packet);
    RUN_TEST(test_csi_pipeline_process_short_ht_114_byte_packet);
    RUN_TEST(test_csi_pipeline_process_double_short_ht_228_byte_packet);
    RUN_TEST(test_csi_pipeline_process_wrong_length_filtered);
    
    // Error path tests
    RUN_TEST(test_csi_pipeline_enable_config_error);
    RUN_TEST(test_csi_pipeline_enable_callback_error);
    RUN_TEST(test_csi_pipeline_enable_csi_error);
    RUN_TEST(test_csi_pipeline_disable_error);
    
    // Callback wrapper tests
    RUN_TEST(test_csi_pipeline_callback_wrapper_triggered);
    RUN_TEST(test_csi_pipeline_callback_wrapper_null_data);
    
    // Clear buffer test
    RUN_TEST(test_csi_pipeline_clear_detector_buffer);
    RUN_TEST(test_csi_pipeline_aggregates_detection_timing_on_evaluation_ticks);
    
    // Legacy normalization tests
    RUN_TEST(test_csi_pipeline_filters_unicast_frames_for_other_device);
    
    return UNITY_END();
}

#if defined(ESP_PLATFORM)
extern "C" void app_main(void) { process(); }
#else
int main(int argc, char **argv) { return process(); }
#endif
