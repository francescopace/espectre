/*
 * ESPectre - Streamer Frontend
 *
 * Streamer frontend adapter over the shared ESP-IDF runtime.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */
#include "streamer_frontend.h"

#include <cstddef>
#include <cstdint>

#include "device_identity.h"
#include "espectre_log.h"
#include "sdkconfig.h"

namespace espectre {

namespace {

static const char *const TAG = "espectre.stream";
constexpr uint8_t kCollectorPacingPayload[] = {'E', 'S', 'P', 'E'};

}  // namespace

RuntimeConfig StreamerFrontend::build_runtime_config_() const {
  RuntimeConfig config;
  config.runtime_profile = RuntimeProfile::STREAM;
  config.csi_traffic_mode = CsiTrafficMode::PACING;
  config.csi_traffic_udp_port = static_cast<uint16_t>(CONFIG_ESPECTRE_TRAFFIC_RX_PORT);
  config.csi_traffic_multicast_group = CONFIG_ESPECTRE_TRAFFIC_RX_MULTICAST_GROUP;
  config.csi_traffic_expected_payload.assign(reinterpret_cast<const char *>(kCollectorPacingPayload),
                                             sizeof(kCollectorPacingPayload));
  config.device_id = derive_runtime_device_id();
  config.collector_port = static_cast<uint16_t>(CONFIG_ESPECTRE_COLLECTOR_PORT);
  config.stream_log_interval_ms = static_cast<uint32_t>(CONFIG_ESPECTRE_STREAM_LOG_INTERVAL_MS);
  config.stream_tx_batch_records = static_cast<uint8_t>(CONFIG_ESPECTRE_STREAM_TX_BATCH_RECORDS);
  return config;
}

bool StreamerFrontend::setup() {
  if (setup_complete_) {
    return true;
  }

  runtime_.set_config(build_runtime_config_());
  if (!runtime_.setup(this)) {
    ESP_LOGE(TAG, "Streamer runtime setup failed");
    return false;
  }

  setup_complete_ = true;
  return true;
}

void StreamerFrontend::loop() {
  if (!setup_complete_) {
    return;
  }
  runtime_.loop();
}

void StreamerFrontend::shutdown() {
  if (!setup_complete_) {
    return;
  }
  runtime_.shutdown();
  setup_complete_ = false;
}

StreamerFrontend::~StreamerFrontend() { shutdown(); }

void StreamerFrontend::on_motion_state_changed(const RuntimeSnapshot &snapshot) { runtime_.record_snapshot(snapshot); }

void StreamerFrontend::on_periodic_update(const RuntimeSnapshot &snapshot, uint32_t packets_received) {
  (void)packets_received;
  runtime_.record_snapshot(snapshot);
}

void StreamerFrontend::on_threshold_changed(const RuntimeSnapshot &snapshot) { runtime_.record_snapshot(snapshot); }

void StreamerFrontend::on_calibration_started(const RuntimeSnapshot &snapshot) { runtime_.record_snapshot(snapshot); }

void StreamerFrontend::on_calibration_finished(const RuntimeSnapshot &snapshot, bool success) {
  (void)success;
  runtime_.record_snapshot(snapshot);
}

void StreamerFrontend::on_live_telemetry(float movement, float threshold) {
  (void)movement;
  (void)threshold;
}

void StreamerFrontend::on_runtime_fault(const char *message) {
  (void)message;
}

}  // namespace espectre
