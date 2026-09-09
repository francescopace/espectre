/*
 * ESPectre - Runtime Diagnostics
 *
 * Runtime diagnostics snapshot helpers.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "runtime_diagnostics.h"

#include "counter_helpers.h"
#include "diagnostic_fields.h"
#include "protocol_json.h"
#include <algorithm>
#include <cmath>
#include <cstring>

#include <cstdio>
#include <string>

#include "runtime_config_utils.h"

namespace espectre {

namespace {

float packets_per_second(uint64_t delta, uint32_t elapsed_ms) {
  return elapsed_ms > 0U
             ? static_cast<float>(delta) * 1000.0f / static_cast<float>(elapsed_ms)
             : 0.0f;
}

void append_json_key(std::string *out, const char *key) {
  *out += ",\"";
  *out += key;
  *out += "\":";
}

void append_json_uint(std::string *out, const char *key, uint64_t value) {
  append_json_key(out, key);
  *out += std::to_string(value);
}

void append_json_float(std::string *out, const char *key, float value) {
  char text[32];
  std::snprintf(text, sizeof(text), "%.6g", static_cast<double>(value));
  append_json_key(out, key);
  *out += text;
}

void append_json_bool(std::string *out, const char *key, bool value) {
  append_json_key(out, key);
  *out += value ? "true" : "false";
}

void append_json_null(std::string *out, const char *key) {
  append_json_key(out, key);
  *out += "null";
}

}  // namespace

void append_runtime_csi_quality_diagnostics_json(std::string *out,
                                                const RuntimeDiagnosticsSnapshot &diagnostics) {
  if (out == nullptr) return;
  append_json_uint(out, "csi_rx_error_total", diagnostics.csi_rx_error_total);
  append_json_uint(out, "csi_rx_end_error_total", diagnostics.csi_rx_end_error_total);
  append_json_uint(out, "csi_invalid_estimate_total", diagnostics.csi_invalid_estimate_total);
  append_json_uint(out, "csi_invalid_first_word_total", diagnostics.csi_invalid_first_word_total);
  append_json_uint(out, "csi_sanitized_first_word_total", diagnostics.csi_sanitized_first_word_total);
}

void append_runtime_performance_diagnostics_json(std::string *out,
                                                 const RuntimeDiagnosticsSnapshot &diagnostics,
                                                 bool include_current_memory) {
  if (out == nullptr) {
    return;
  }
  if (include_current_memory) {
    append_json_float(out, "free_memory_kb", static_cast<float>(diagnostics.free_memory_bytes) / 1024.0f);
  }
  append_json_float(out,
                    "minimum_free_memory_kb",
                    static_cast<float>(diagnostics.minimum_free_memory_bytes) / 1024.0f);
  append_json_float(out,
                    "largest_free_memory_kb",
                    static_cast<float>(diagnostics.largest_free_memory_block_bytes) / 1024.0f);
  append_json_uint(out, "cpu_frequency_mhz", diagnostics.cpu_frequency_mhz);
  append_json_bool(out, "performance_window_ready", diagnostics.performance_window_ready);
  if (diagnostics.performance_window_ready) {
    append_json_float(out,
                      "performance_window_ms",
                      static_cast<float>(diagnostics.performance_window_duration_us) / 1000.0f);
    append_json_float(out, "runtime_load_percent", diagnostics.runtime_load_percent);
    append_json_uint(out, "loop_samples", diagnostics.loop_samples);
    append_json_uint(out, "loop_avg_us", diagnostics.loop_average_us);
    append_json_uint(out, "loop_max_us", diagnostics.loop_maximum_us);
  } else {
    append_json_null(out, "performance_window_ms");
    append_json_null(out, "runtime_load_percent");
    append_json_null(out, "loop_samples");
    append_json_null(out, "loop_avg_us");
    append_json_null(out, "loop_max_us");
  }
  append_json_bool(out, "detection_timing_supported", diagnostics.detection_timing_supported);
  if (diagnostics.performance_window_ready && diagnostics.detection_timing_supported) {
    append_json_uint(out, "detection_samples", diagnostics.detection_samples);
    append_json_uint(out, "detection_sum_us", diagnostics.detection_sum_us);
    append_json_uint(out, "detection_avg_us", diagnostics.detection_average_us);
    append_json_uint(out, "detection_min_us", diagnostics.detection_minimum_us);
    append_json_uint(out, "detection_max_us", diagnostics.detection_maximum_us);
  } else {
    append_json_null(out, "detection_samples");
    append_json_null(out, "detection_sum_us");
    append_json_null(out, "detection_avg_us");
    append_json_null(out, "detection_min_us");
    append_json_null(out, "detection_max_us");
  }
}


bool validate_diagnostic_fields(const std::vector<std::string> &fields, unsigned profile) {
  if (fields.size() > ESPECTRE_DIAGNOSTIC_FIELD_COUNT) return false;
  for (const auto &name : fields) {
    if (name == "*") return fields.size() == 1U;
    bool found = false;
    for (const auto &field : espectre_diagnostic_fields) {
      if ((field.profiles & profile) &&
          (name == field.name ||
           (std::strncmp(field.name, name.c_str(), name.size()) == 0 && field.name[name.size()] == '.'))) {
        found = true;
        break;
      }
    }
    if (!found || name.empty()) return false;
  }
  return true;
}

std::string diagnostic_response(const std::vector<std::string> &fields, unsigned profile,
                                const std::function<std::string(const char *)> &value) {
  if (!validate_diagnostic_fields(fields, profile)) return {};
  const bool catalog = fields.empty();
  std::string out = catalog ? "{\"fields\":[" : "{";
  bool first = true;
  std::string group;
  bool group_first = true;
  for (const auto &field : espectre_diagnostic_fields) {
    if (!(field.profiles & profile)) continue;
    const char *dot = std::strchr(field.name, '.');
    const size_t parent_length = dot == nullptr ? 0U : static_cast<size_t>(dot - field.name);
    bool selected = catalog || fields.front() == "*" ||
                    std::strcmp(field.name, "timestamp_ms") == 0 || std::strcmp(field.name, "uptime") == 0;
    for (const auto &requested : fields) {
      selected |= requested == field.name ||
                  (parent_length > 0U && requested.size() == parent_length &&
                   std::strncmp(requested.c_str(), field.name, parent_length) == 0);
    }
    if (!selected) continue;
    if (catalog) {
      if (!first) out += ',';
      first = false;
      out += "{\"name\":";
      append_json_string(&out, field.name);
      out += ",\"type\":";
      append_json_string(&out, field.type);
      out += ",\"unit\":";
      append_json_string(&out, field.unit);
      out += '}';
      continue;
    }
    if (group.size() != parent_length || std::strncmp(group.c_str(), field.name, parent_length) != 0) {
      if (!group.empty()) out += '}';
      group.assign(field.name, parent_length);
      group_first = true;
      if (!group.empty()) {
        if (!first) out += ',';
        first = false;
        append_json_string(&out, group.c_str());
        out += ":{";
      }
    }
    bool &entry_first = group.empty() ? first : group_first;
    if (!entry_first) out += ',';
    entry_first = false;
    append_json_string(&out, dot == nullptr ? field.name : dot + 1);
    out += ':';
    const std::string scalar = value(field.name);
    out += scalar.empty() ? "null" : scalar;
  }
  if (!group.empty()) out += '}';
  out += catalog ? "]}" : "}";
  return out;
}

std::string runtime_diagnostic_value(const char *key, const RuntimeDiagnosticsSample *sample,
                                     const std::function<const RuntimeDiagnosticsSnapshot &()> &snapshot) {
  const auto number = [](double value) {
    if (!std::isfinite(value)) return std::string("null");
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.6g", value);
    return std::string(buffer);
  };
  if (std::strcmp(key, "traffic_tx_pps") == 0) return sample ? number(sample->traffic_tx_pps) : "null";
  if (std::strcmp(key, "csi_callback_pps") == 0) return sample ? number(sample->csi_callback_pps) : "null";
  if (std::strcmp(key, "csi_accepted_pps") == 0) return sample ? number(sample->csi_accepted_pps) : "null";
  if (std::strcmp(key, "csi_admitted_pps") == 0) return sample ? number(sample->csi_admitted_pps) : "null";
  if (std::strcmp(key, "csi_filtered_pps") == 0) return sample ? number(sample->csi_filtered_pps) : "null";
  if (std::strcmp(key, "csi_hw_error_pps") == 0) return sample ? number(sample->csi_hw_error_pps) : "null";
  if (std::strcmp(key, "csi_missing_slots_pps") == 0) return sample ? number(sample->csi_missing_slots_pps) : "null";
  if (std::strcmp(key, "csi_excess_pps") == 0) return sample ? number(sample->csi_excess_pps) : "null";
  if (std::strcmp(key, "csi_stale_pps") == 0) return sample ? number(sample->csi_stale_pps) : "null";
  if (std::strcmp(key, "csi_out_of_order_pps") == 0) return sample ? number(sample->csi_out_of_order_pps) : "null";
  if (std::strcmp(key, "csi_pending_frame_drop_pps") == 0) return sample ? number(sample->csi_pending_frame_drop_pps) : "null";
  if (std::strcmp(key, "csi_occupancy") == 0) return sample ? number(sample->csi_occupancy_ratio) : "null";
  if (std::strcmp(key, "wifi_rssi_dbm") == 0) {
    const int rssi = sample ? sample->wifi_rssi_dbm : snapshot().wifi_rssi_dbm;
    return rssi == INT8_MIN ? "null" : std::to_string(rssi);
  }
  if (std::strcmp(key, "wifi_channel") == 0) return std::to_string(sample ? sample->wifi_channel : snapshot().wifi_channel);
  if (std::strcmp(key, "csi_hw_error_total") == 0) {
    const auto &s = snapshot();
    return std::to_string(s.csi_rx_error_total + s.csi_rx_end_error_total + s.csi_invalid_estimate_total + s.csi_invalid_first_word_total);
  }
  if (std::strcmp(key, "free_memory_kb") == 0) {
    const auto &s = snapshot();
    return number(s.free_memory_bytes / 1024.0);
  }
  if (std::strcmp(key, "minimum_free_memory_kb") == 0) {
    const auto &s = snapshot();
    return number(s.minimum_free_memory_bytes / 1024.0);
  }
  if (std::strcmp(key, "largest_free_memory_kb") == 0) {
    const auto &s = snapshot();
    return number(s.largest_free_memory_block_bytes / 1024.0);
  }
  if (std::strcmp(key, "cpu_frequency_mhz") == 0) {
    const auto &s = snapshot();
    return std::to_string(s.cpu_frequency_mhz);
  }
  if (std::strcmp(key, "performance_window_ready") == 0) {
    const auto &s = snapshot();
    return s.performance_window_ready ? "true" : "false";
  }
  if (std::strcmp(key, "detection_timing_supported") == 0) {
    const auto &s = snapshot();
    return s.detection_timing_supported ? "true" : "false";
  }
  if (std::strcmp(key, "performance_window_ms") == 0) {
    const auto &s = snapshot();
    if (!s.performance_window_ready) return "null";
    return number(s.performance_window_duration_us / 1000.0);
  }
  if (std::strcmp(key, "runtime_load_percent") == 0) {
    const auto &s = snapshot();
    if (!s.performance_window_ready) return "null";
    return number(s.runtime_load_percent);
  }
  if (std::strcmp(key, "loop_samples") == 0) {
    const auto &s = snapshot();
    if (!s.performance_window_ready) return "null";
    return std::to_string(s.loop_samples);
  }
  if (std::strcmp(key, "detection_samples") == 0) {
    const auto &s = snapshot();
    if (!s.performance_window_ready || !s.detection_timing_supported) return "null";
    return std::to_string(s.detection_samples);
  }
  if (std::strcmp(key, "loop_avg_us") == 0) {
    const auto &s = snapshot();
    if (!s.performance_window_ready) return "null";
    return std::to_string(s.loop_average_us);
  }
  if (std::strcmp(key, "loop_max_us") == 0) {
    const auto &s = snapshot();
    if (!s.performance_window_ready) return "null";
    return std::to_string(s.loop_maximum_us);
  }
  if (std::strcmp(key, "detection_sum_us") == 0) {
    const auto &s = snapshot();
    if (!s.performance_window_ready || !s.detection_timing_supported) return "null";
    return std::to_string(s.detection_sum_us);
  }
  if (std::strcmp(key, "detection_avg_us") == 0) {
    const auto &s = snapshot();
    if (!s.performance_window_ready || !s.detection_timing_supported) return "null";
    return std::to_string(s.detection_average_us);
  }
  if (std::strcmp(key, "detection_min_us") == 0) {
    const auto &s = snapshot();
    if (!s.performance_window_ready || !s.detection_timing_supported) return "null";
    return std::to_string(s.detection_minimum_us);
  }
  if (std::strcmp(key, "detection_max_us") == 0) {
    const auto &s = snapshot();
    if (!s.performance_window_ready || !s.detection_timing_supported) return "null";
    return std::to_string(s.detection_maximum_us);
  }
  if (std::strcmp(key, "traffic_packets_total") == 0) {
    const auto &s = snapshot();
    return std::to_string(s.traffic_packets_total);
  }
  if (std::strcmp(key, "csi_callbacks_total") == 0) {
    const auto &s = snapshot();
    return std::to_string(s.csi_callbacks_total);
  }
  if (std::strcmp(key, "csi_accepted_total") == 0) {
    const auto &s = snapshot();
    return std::to_string(s.csi_accepted_total);
  }
  if (std::strcmp(key, "csi_admitted_total") == 0) {
    const auto &s = snapshot();
    return std::to_string(s.csi_admitted_total);
  }
  if (std::strcmp(key, "csi_filtered_total") == 0) {
    const auto &s = snapshot();
    return std::to_string(s.csi_filtered_total);
  }
  if (std::strcmp(key, "csi_missing_slots_total") == 0) {
    const auto &s = snapshot();
    return std::to_string(s.csi_missing_slots_total);
  }
  if (std::strcmp(key, "csi_excess_total") == 0) {
    const auto &s = snapshot();
    return std::to_string(s.csi_excess_total);
  }
  if (std::strcmp(key, "csi_stale_total") == 0) {
    const auto &s = snapshot();
    return std::to_string(s.csi_stale_total);
  }
  if (std::strcmp(key, "csi_out_of_order_total") == 0) {
    const auto &s = snapshot();
    return std::to_string(s.csi_out_of_order_total);
  }
  if (std::strcmp(key, "csi_occupancy_slots") == 0) {
    const auto &s = snapshot();
    return std::to_string(s.csi_occupancy_slots);
  }
  if (std::strcmp(key, "csi_window_slots") == 0) {
    const auto &s = snapshot();
    return std::to_string(s.csi_window_slots);
  }
  if (std::strcmp(key, "csi_provenance_rejected_total") == 0) {
    const auto &s = snapshot();
    return std::to_string(s.csi_provenance_rejected_total);
  }
  if (std::strcmp(key, "csi_pending_frame_drops_total") == 0) {
    const auto &s = snapshot();
    return std::to_string(s.csi_pending_frame_drops_total);
  }
  if (std::strcmp(key, "csi_pending_frames") == 0) {
    const auto &s = snapshot();
    return std::to_string(s.csi_pending_frames);
  }
  if (std::strcmp(key, "csi_pending_frame_capacity") == 0) {
    const auto &s = snapshot();
    return std::to_string(s.csi_pending_frame_capacity);
  }
  if (std::strcmp(key, "csi_rx_error_total") == 0) {
    const auto &s = snapshot();
    return std::to_string(s.csi_rx_error_total);
  }
  if (std::strcmp(key, "csi_rx_end_error_total") == 0) {
    const auto &s = snapshot();
    return std::to_string(s.csi_rx_end_error_total);
  }
  if (std::strcmp(key, "csi_invalid_estimate_total") == 0) {
    const auto &s = snapshot();
    return std::to_string(s.csi_invalid_estimate_total);
  }
  if (std::strcmp(key, "csi_invalid_first_word_total") == 0) {
    const auto &s = snapshot();
    return std::to_string(s.csi_invalid_first_word_total);
  }
  if (std::strcmp(key, "csi_sanitized_first_word_total") == 0) {
    const auto &s = snapshot();
    return std::to_string(s.csi_sanitized_first_word_total);
  }
  return "null";
}

void RuntimeDiagnosticsSampler::reset(const RuntimeDiagnosticsSnapshot &snapshot, uint32_t now_ms) {
  previous_ = snapshot;
  previous_ms_ = now_ms;
  baseline_ready_ = true;
}

RuntimeDiagnosticsSample RuntimeDiagnosticsSampler::sample(const RuntimeDiagnosticsSnapshot &snapshot,
                                                            uint32_t now_ms) {
  RuntimeDiagnosticsSample result;
  result.wifi_rssi_dbm = snapshot.wifi_rssi_dbm;
  result.wifi_channel = snapshot.wifi_channel;
  if (!baseline_ready_) {
    reset(snapshot, now_ms);
    return result;
  }

  const uint32_t elapsed_ms = now_ms - previous_ms_;
  if (elapsed_ms == 0U) {
    return result;
  }
  result.traffic_tx_pps = packets_per_second(
      counter_delta(snapshot.traffic_packets_total, previous_.traffic_packets_total), elapsed_ms);
  result.csi_callback_pps = packets_per_second(
      counter_delta(snapshot.csi_callbacks_total, previous_.csi_callbacks_total), elapsed_ms);
  result.csi_accepted_pps = packets_per_second(
      counter_delta(snapshot.csi_accepted_total, previous_.csi_accepted_total), elapsed_ms);
  result.csi_admitted_pps = packets_per_second(
      counter_delta(snapshot.csi_admitted_total, previous_.csi_admitted_total), elapsed_ms);
  result.csi_filtered_pps = packets_per_second(
      counter_delta(snapshot.csi_filtered_total, previous_.csi_filtered_total), elapsed_ms);
  result.csi_hw_error_pps = packets_per_second(
      counter_delta(snapshot.csi_rx_error_total, previous_.csi_rx_error_total) +
          counter_delta(snapshot.csi_rx_end_error_total, previous_.csi_rx_end_error_total) +
          counter_delta(snapshot.csi_invalid_estimate_total, previous_.csi_invalid_estimate_total) +
          counter_delta(snapshot.csi_invalid_first_word_total, previous_.csi_invalid_first_word_total),
      elapsed_ms);
  result.csi_pending_frame_drop_pps = packets_per_second(
      counter_delta(snapshot.csi_pending_frame_drops_total,
                    previous_.csi_pending_frame_drops_total),
      elapsed_ms);
  result.csi_missing_slots_pps = packets_per_second(
      counter_delta(snapshot.csi_missing_slots_total, previous_.csi_missing_slots_total), elapsed_ms);
  result.csi_excess_pps = packets_per_second(
      counter_delta(snapshot.csi_excess_total, previous_.csi_excess_total), elapsed_ms);
  result.csi_stale_pps = packets_per_second(
      counter_delta(snapshot.csi_stale_total, previous_.csi_stale_total), elapsed_ms);
  result.csi_out_of_order_pps = packets_per_second(
      counter_delta(snapshot.csi_out_of_order_total, previous_.csi_out_of_order_total), elapsed_ms);
  result.csi_occupancy_ratio = snapshot.csi_window_slots > 0U
      ? static_cast<float>(snapshot.csi_occupancy_slots) /
            static_cast<float>(snapshot.csi_window_slots)
      : 0.0f;
  reset(snapshot, now_ms);
  return result;
}

void visit_runtime_diagnostics(const RuntimeConfig &config,
                               const RuntimeSnapshot &snapshot,
                               runtime_diagnostic_visitor_t visitor) {
  if (!visitor) {
    return;
  }

  char value[64];

  std::snprintf(value, sizeof(value), "%.6f", snapshot.threshold);
  visitor("threshold", value);
  std::snprintf(value, sizeof(value), "%u", static_cast<unsigned>(config.segmentation_window_size_ms));
  visitor("window_ms", value);
  visitor("detector", snapshot.detector_name);
  visitor("lowpass", config.lowpass_enabled ? "on" : "off");
  std::snprintf(value, sizeof(value), "%.1f", config.lowpass_cutoff);
  visitor("lowpass_cutoff", value);
  visitor("hampel", config.hampel_enabled ? "on" : "off");
  if (config.hampel_enabled) {
    std::snprintf(value, sizeof(value), "%u", static_cast<unsigned>(config.hampel_window));
    visitor("hampel_window", value);
    std::snprintf(value, sizeof(value), "%.1f", config.hampel_threshold);
    visitor("hampel_threshold", value);
  }
  visitor("traffic_mode", traffic_mode_name(config.traffic_generator_mode));
  visitor("csi_traffic_mode", csi_traffic_mode_name(config.csi_traffic_mode));
  std::snprintf(value, sizeof(value), "%u", static_cast<unsigned>(config.csi_target_pps));
  visitor("csi_target_pps", value);
  std::snprintf(value, sizeof(value), "%u", static_cast<unsigned>(config.evaluation_interval_ms));
  visitor("evaluation_interval_ms", value);
  std::snprintf(value,
                sizeof(value),
                "%u/%u",
                static_cast<unsigned>(config.motion_on_hits),
                static_cast<unsigned>(config.motion_off_hits));
  visitor("motion_hits", value);
}

}  // namespace espectre
