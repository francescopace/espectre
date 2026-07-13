#pragma once

#include <cstdint>
#include <string>

#include "runtime_capabilities.h"
#include "runtime_events.h"
#include "runtime_snapshot.h"
#include "runtime_sensing_schema.h"
#include "csi_traffic_types.h"
#include "threshold.h"
#include "utils.h"

namespace espectre {

struct RuntimeConfig {
  RuntimeProfile runtime_profile{RuntimeProfile::SENSING};
  DetectionAlgorithm detection_algorithm{DetectionAlgorithm::CLASSIC};
  ThresholdMode threshold_mode{ThresholdMode::AUTO};
  float segmentation_threshold{RUNTIME_SEGMENTATION_THRESHOLD_DEFAULT};
  uint16_t segmentation_window_size{RUNTIME_SEGMENTATION_WINDOW_SIZE_DEFAULT};
  bool classic_recovery_vote_enabled{RUNTIME_CLASSIC_RECOVERY_VOTE_ENABLED_DEFAULT};
  uint32_t traffic_generator_rate{RUNTIME_TRAFFIC_GENERATOR_RATE_DEFAULT};
  RuntimeTrafficMode traffic_generator_mode{RuntimeTrafficMode::PING};
  CsiTrafficMode csi_traffic_mode{CsiTrafficMode::INTERNAL};
  uint16_t csi_traffic_udp_port{RUNTIME_CSI_TRAFFIC_UDP_PORT_DEFAULT};
  std::string csi_traffic_multicast_group;
  std::string csi_traffic_expected_payload;
  uint64_t device_id{0U};
  uint16_t collector_port{RUNTIME_STREAM_COLLECTOR_PORT_DEFAULT};
  uint32_t stream_log_interval_ms{RUNTIME_STREAM_LOG_INTERVAL_MS_DEFAULT};
  uint8_t stream_tx_batch_records{RUNTIME_STREAM_TX_BATCH_RECORDS_DEFAULT};
  uint32_t publish_interval{RUNTIME_PUBLISH_INTERVAL_DEFAULT};
  uint32_t evaluation_interval{RUNTIME_EVALUATION_INTERVAL_DEFAULT};
  uint8_t motion_on_hits{RUNTIME_MOTION_ON_HITS_DEFAULT};
  uint8_t motion_off_hits{RUNTIME_MOTION_OFF_HITS_DEFAULT};
  bool lowpass_enabled{RUNTIME_LOWPASS_ENABLED_DEFAULT};
  float lowpass_cutoff{RUNTIME_LOWPASS_CUTOFF_DEFAULT};
  bool hampel_enabled{RUNTIME_HAMPEL_ENABLED_DEFAULT};
  uint8_t hampel_window{RUNTIME_HAMPEL_WINDOW_DEFAULT};
  float hampel_threshold{RUNTIME_HAMPEL_THRESHOLD_DEFAULT};
};

class IEspectreRuntime {
 public:
  virtual ~IEspectreRuntime() = default;

  virtual bool setup() = 0;
  virtual void shutdown() = 0;
  virtual void loop() = 0;
  virtual void set_services_armed(bool armed) = 0;
  virtual void set_live_telemetry_enabled(bool enabled) = 0;

  virtual bool set_threshold_runtime(float threshold) = 0;
  virtual bool trigger_recalibration() = 0;
  virtual bool is_calibrating() const = 0;

  virtual RuntimeSnapshot get_snapshot() const = 0;
  virtual RuntimeCapabilities get_capabilities() const = 0;

  virtual void set_listener(IRuntimeListener *listener) = 0;
};

}  // namespace espectre
