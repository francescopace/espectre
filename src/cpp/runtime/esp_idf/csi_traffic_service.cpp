#include "csi_traffic_service.h"

#include "espectre_log.h"

namespace esphome {
namespace espectre {

namespace {

static const char *const TAG = "CsiTrafficService";

}  // namespace

void CsiTrafficService::init(const CsiTrafficServiceConfig &config) {
  config_ = config;
  traffic_generator_.init(config_.rate_pps, config_.traffic_mode);
  udp_listener_.init(config_.udp_port);
  if (!config_.multicast_group.empty()) {
    udp_listener_.set_multicast_group(config_.multicast_group.c_str());
  } else {
    udp_listener_.set_multicast_group(nullptr);
  }
  if (!config_.expected_payload.empty()) {
    udp_listener_.set_expected_payload(reinterpret_cast<const uint8_t *>(config_.expected_payload.data()),
                                       config_.expected_payload.size());
  } else {
    udp_listener_.set_expected_payload(nullptr, 0U);
  }
}

bool CsiTrafficService::start() {
  switch (config_.mode) {
    case CsiTrafficMode::INTERNAL:
      return traffic_generator_.is_running() ? true : traffic_generator_.start();
    case CsiTrafficMode::EXTERNAL:
    case CsiTrafficMode::PACING:
      return udp_listener_.is_running() ? true : udp_listener_.start();
    case CsiTrafficMode::DISABLED:
    default:
      ESP_LOGI(TAG, "CSI traffic service disabled");
      return true;
  }
}

void CsiTrafficService::stop() {
  if (traffic_generator_.is_running()) {
    traffic_generator_.stop();
  }
  if (udp_listener_.is_running()) {
    udp_listener_.stop();
  }
}

void CsiTrafficService::loop() {
  if (traffic_generator_.is_running()) {
    traffic_generator_.loop();
  }
  if (udp_listener_.is_running()) {
    udp_listener_.loop();
  }
}

bool CsiTrafficService::is_running() const {
  switch (config_.mode) {
    case CsiTrafficMode::INTERNAL:
      return traffic_generator_.is_running();
    case CsiTrafficMode::EXTERNAL:
    case CsiTrafficMode::PACING:
      return udp_listener_.is_running();
    case CsiTrafficMode::DISABLED:
    default:
      return false;
  }
}

bool CsiTrafficService::get_last_sender(sockaddr_in *out_addr) const { return udp_listener_.get_last_sender(out_addr); }

uint64_t CsiTrafficService::get_packets_received() const { return udp_listener_.get_packets_received(); }

}  // namespace espectre
}  // namespace esphome
