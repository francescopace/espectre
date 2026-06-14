#include "stimulus_service.h"

#include "espectre_log.h"

namespace esphome {
namespace espectre {

namespace {

static const char *const TAG = "StimulusService";

}  // namespace

void StimulusService::init(const StimulusServiceConfig &config) {
  config_ = config;
  traffic_generator_.init(config_.rate_pps, config_.traffic_mode);
  udp_listener_.init(config_.udp_port);
  if (!config_.multicast_group.empty()) {
    udp_listener_.set_multicast_group(config_.multicast_group.c_str());
  } else {
    udp_listener_.set_multicast_group(nullptr);
  }
}

bool StimulusService::start() {
  switch (config_.mode) {
    case StimulusMode::INTERNAL:
      return traffic_generator_.is_running() ? true : traffic_generator_.start();
    case StimulusMode::EXTERNAL:
      return udp_listener_.is_running() ? true : udp_listener_.start();
    case StimulusMode::DISABLED:
    default:
      ESP_LOGI(TAG, "Stimulus service disabled");
      return true;
  }
}

void StimulusService::stop() {
  if (traffic_generator_.is_running()) {
    traffic_generator_.stop();
  }
  if (udp_listener_.is_running()) {
    udp_listener_.stop();
  }
}

void StimulusService::loop() {
  if (traffic_generator_.is_running()) {
    traffic_generator_.loop();
  }
  if (udp_listener_.is_running()) {
    udp_listener_.loop();
  }
}

bool StimulusService::is_running() const {
  switch (config_.mode) {
    case StimulusMode::INTERNAL:
      return traffic_generator_.is_running();
    case StimulusMode::EXTERNAL:
      return udp_listener_.is_running();
    case StimulusMode::DISABLED:
    default:
      return false;
  }
}

bool StimulusService::get_last_sender(sockaddr_in *out_addr) const { return udp_listener_.get_last_sender(out_addr); }

uint64_t StimulusService::get_raw_packets_received() const { return udp_listener_.get_raw_packets_received(); }

uint64_t StimulusService::get_packets_received() const { return udp_listener_.get_packets_received(); }

}  // namespace espectre
}  // namespace esphome
