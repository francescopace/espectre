#pragma once

#include <cstdint>
#include <string>

#include "threshold.h"
#include "utils.h"
#include "runtime_capabilities.h"
#include "runtime_events.h"
#include "runtime_snapshot.h"
#include "csi_traffic_types.h"

namespace espectre {

enum class DetectionAlgorithm {
  CLASSIC,
  ML,
};

enum class RuntimeProfile {
  SENSING,
  STREAM,
};

enum class RuntimeTrafficMode {
  DNS,
  PING,
};

struct RuntimeConfig {
  RuntimeProfile runtime_profile{RuntimeProfile::SENSING};
  DetectionAlgorithm detection_algorithm{DetectionAlgorithm::CLASSIC};
  ThresholdMode threshold_mode{ThresholdMode::AUTO};
  float segmentation_threshold{SEGMENTATION_DEFAULT_THRESHOLD};
  uint16_t segmentation_window_size{DETECTOR_DEFAULT_WINDOW_SIZE};
  uint32_t traffic_generator_rate{100};
  RuntimeTrafficMode traffic_generator_mode{RuntimeTrafficMode::PING};
  CsiTrafficMode csi_traffic_mode{CsiTrafficMode::INTERNAL};
  uint16_t csi_traffic_udp_port{5555};
  std::string csi_traffic_multicast_group;
  std::string csi_traffic_expected_payload;
  uint64_t device_id{0U};
  uint16_t collector_port{5001};
  uint32_t stream_log_interval_ms{1000};
  uint32_t publish_interval{100};
  uint32_t evaluation_interval{25};
  uint8_t motion_on_hits{3};
  uint8_t motion_off_hits{3};
  bool lowpass_enabled{false};
  float lowpass_cutoff{11.0f};
  bool hampel_enabled{true};
  uint8_t hampel_window{7};
  float hampel_threshold{5.0f};
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
