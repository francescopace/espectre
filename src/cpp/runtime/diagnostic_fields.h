// SPDX-License-Identifier: GPL-3.0-only
// Commercial licensing available under separate agreement; see LICENSING.md.
#pragma once

/**
 * @file diagnostic_fields.h
 * @brief C-compatible metadata for the canonical device diagnostic catalog.
 *
 * Profiles are bitmasks of the `ESPECTRE_DIAGNOSTIC_PROFILE_*` constants.
 * Dotted field names select leaves while protocol responses retain their
 * nested objects.
 */

/** Frontend profiles a diagnostic field belongs to. */
enum {
  /** Native firmware. */
  ESPECTRE_DIAGNOSTIC_PROFILE_NATIVE = 1U,
  /** Shared Direct HTTP bridge, used by ESPHome. */
  ESPECTRE_DIAGNOSTIC_PROFILE_BRIDGE = 2U,
  /** Micro-ESPectre. */
  ESPECTRE_DIAGNOSTIC_PROFILE_MICRO = 4U,
  /** Every frontend. */
  ESPECTRE_DIAGNOSTIC_PROFILE_ALL = 7U,
};
typedef struct {
  const char *name;
  const char *type;
  const char *unit;
  unsigned profiles;
} espectre_diagnostic_field_t;

/** Canonical diagnostic fields; filter by profile before exposing a catalog. */
static const espectre_diagnostic_field_t espectre_diagnostic_fields[] = {
  {"timestamp_ms", "integer", "ms", ESPECTRE_DIAGNOSTIC_PROFILE_ALL},
  {"uptime", "integer", "s", ESPECTRE_DIAGNOSTIC_PROFILE_ALL},
  {"free_memory_kb", "number", "KiB", ESPECTRE_DIAGNOSTIC_PROFILE_ALL},
  {"minimum_free_memory_kb", "number", "KiB", ESPECTRE_DIAGNOSTIC_PROFILE_ALL},
  {"largest_free_memory_kb", "number", "KiB", ESPECTRE_DIAGNOSTIC_PROFILE_ALL},
  {"cpu_frequency_mhz", "integer", "MHz", ESPECTRE_DIAGNOSTIC_PROFILE_ALL},
  {"loop_time_ms", "number", "ms", ESPECTRE_DIAGNOSTIC_PROFILE_ALL},
  {"performance_window_ready", "boolean", "", ESPECTRE_DIAGNOSTIC_PROFILE_ALL},
  {"detection_timing_supported", "boolean", "", ESPECTRE_DIAGNOSTIC_PROFILE_ALL},
  {"performance_window_ms", "number", "ms", ESPECTRE_DIAGNOSTIC_PROFILE_ALL},
  {"runtime_load_percent", "number", "%", ESPECTRE_DIAGNOSTIC_PROFILE_ALL},
  {"loop_samples", "integer", "count", ESPECTRE_DIAGNOSTIC_PROFILE_ALL},
  {"detection_samples", "integer", "count", ESPECTRE_DIAGNOSTIC_PROFILE_ALL},
  {"loop_avg_us", "integer", "us", ESPECTRE_DIAGNOSTIC_PROFILE_ALL},
  {"loop_max_us", "integer", "us", ESPECTRE_DIAGNOSTIC_PROFILE_ALL},
  {"detection_sum_us", "integer", "us", ESPECTRE_DIAGNOSTIC_PROFILE_ALL},
  {"detection_avg_us", "integer", "us", ESPECTRE_DIAGNOSTIC_PROFILE_ALL},
  {"detection_min_us", "integer", "us", ESPECTRE_DIAGNOSTIC_PROFILE_ALL},
  {"detection_max_us", "integer", "us", ESPECTRE_DIAGNOSTIC_PROFILE_ALL},
  {"generator_pps", "number", "pps", ESPECTRE_DIAGNOSTIC_PROFILE_ALL},
  {"traffic_tx_pps", "number", "pps", ESPECTRE_DIAGNOSTIC_PROFILE_ALL},
  {"traffic_rx_pps", "number", "pps", ESPECTRE_DIAGNOSTIC_PROFILE_ALL},
  {"csi_callback_pps", "number", "pps", ESPECTRE_DIAGNOSTIC_PROFILE_ALL},
  {"csi_accepted_pps", "number", "pps", ESPECTRE_DIAGNOSTIC_PROFILE_ALL},
  {"csi_admitted_pps", "number", "pps", ESPECTRE_DIAGNOSTIC_PROFILE_ALL},
  {"csi_filtered_pps", "number", "pps", ESPECTRE_DIAGNOSTIC_PROFILE_ALL},
  {"csi_hw_error_pps", "number", "pps", ESPECTRE_DIAGNOSTIC_PROFILE_ALL},
  {"csi_missing_slots_pps", "number", "pps", ESPECTRE_DIAGNOSTIC_PROFILE_ALL},
  {"csi_excess_pps", "number", "pps", ESPECTRE_DIAGNOSTIC_PROFILE_ALL},
  {"csi_stale_pps", "number", "pps", ESPECTRE_DIAGNOSTIC_PROFILE_ALL},
  {"csi_out_of_order_pps", "number", "pps", ESPECTRE_DIAGNOSTIC_PROFILE_ALL},
  {"csi_pending_frame_drop_pps", "number", "pps", ESPECTRE_DIAGNOSTIC_PROFILE_NATIVE | ESPECTRE_DIAGNOSTIC_PROFILE_BRIDGE},
  {"csi_occupancy", "number", "ratio", ESPECTRE_DIAGNOSTIC_PROFILE_ALL},
  {"wifi_channel", "integer", "", ESPECTRE_DIAGNOSTIC_PROFILE_ALL},
  {"wifi_rssi_dbm", "integer", "dBm", ESPECTRE_DIAGNOSTIC_PROFILE_ALL},
  {"csi_hw_error_total", "integer", "count", ESPECTRE_DIAGNOSTIC_PROFILE_ALL},
  {"csi_callbacks_total", "integer", "count", ESPECTRE_DIAGNOSTIC_PROFILE_BRIDGE},
  {"csi_accepted_total", "integer", "count", ESPECTRE_DIAGNOSTIC_PROFILE_BRIDGE},
  {"csi_admitted_total", "integer", "count", ESPECTRE_DIAGNOSTIC_PROFILE_BRIDGE},
  {"csi_filtered_total", "integer", "count", ESPECTRE_DIAGNOSTIC_PROFILE_BRIDGE},
  {"csi_missing_slots_total", "integer", "count", ESPECTRE_DIAGNOSTIC_PROFILE_BRIDGE},
  {"csi_excess_total", "integer", "count", ESPECTRE_DIAGNOSTIC_PROFILE_BRIDGE},
  {"csi_stale_total", "integer", "count", ESPECTRE_DIAGNOSTIC_PROFILE_BRIDGE},
  {"csi_out_of_order_total", "integer", "count", ESPECTRE_DIAGNOSTIC_PROFILE_BRIDGE},
  {"csi_occupancy_slots", "integer", "count", ESPECTRE_DIAGNOSTIC_PROFILE_BRIDGE},
  {"csi_window_slots", "integer", "count", ESPECTRE_DIAGNOSTIC_PROFILE_BRIDGE},
  {"csi_provenance_rejected_total", "integer", "count", ESPECTRE_DIAGNOSTIC_PROFILE_NATIVE | ESPECTRE_DIAGNOSTIC_PROFILE_BRIDGE},
  {"csi_pending_frame_drops_total", "integer", "count", ESPECTRE_DIAGNOSTIC_PROFILE_NATIVE | ESPECTRE_DIAGNOSTIC_PROFILE_BRIDGE},
  {"csi_pending_frames", "integer", "count", ESPECTRE_DIAGNOSTIC_PROFILE_NATIVE | ESPECTRE_DIAGNOSTIC_PROFILE_BRIDGE},
  {"csi_pending_frame_capacity", "integer", "count", ESPECTRE_DIAGNOSTIC_PROFILE_NATIVE | ESPECTRE_DIAGNOSTIC_PROFILE_BRIDGE},
  {"runtime_motion_event_drops_total", "integer", "count", ESPECTRE_DIAGNOSTIC_PROFILE_NATIVE | ESPECTRE_DIAGNOSTIC_PROFILE_BRIDGE},
  {"csi_rx_error_total", "integer", "count", ESPECTRE_DIAGNOSTIC_PROFILE_NATIVE | ESPECTRE_DIAGNOSTIC_PROFILE_BRIDGE},
  {"csi_rx_end_error_total", "integer", "count", ESPECTRE_DIAGNOSTIC_PROFILE_NATIVE | ESPECTRE_DIAGNOSTIC_PROFILE_BRIDGE},
  {"csi_invalid_estimate_total", "integer", "count", ESPECTRE_DIAGNOSTIC_PROFILE_NATIVE | ESPECTRE_DIAGNOSTIC_PROFILE_BRIDGE},
  {"csi_invalid_first_word_total", "integer", "count", ESPECTRE_DIAGNOSTIC_PROFILE_NATIVE | ESPECTRE_DIAGNOSTIC_PROFILE_BRIDGE},
  {"csi_sanitized_first_word_total", "integer", "count", ESPECTRE_DIAGNOSTIC_PROFILE_NATIVE | ESPECTRE_DIAGNOSTIC_PROFILE_BRIDGE},
  {"task_stack_high_water_bytes", "integer", "bytes", ESPECTRE_DIAGNOSTIC_PROFILE_ALL},
  {"direct_http.event_clients", "integer", "count", ESPECTRE_DIAGNOSTIC_PROFILE_ALL},
  {"direct_http.event_client_limit", "integer", "count", ESPECTRE_DIAGNOSTIC_PROFILE_ALL},
  {"direct_http.queue_capacity", "integer", "count", ESPECTRE_DIAGNOSTIC_PROFILE_ALL},
  {"direct_http.queued_messages", "integer", "count", ESPECTRE_DIAGNOSTIC_PROFILE_ALL},
  {"direct_http.accepted_connections", "integer", "count", ESPECTRE_DIAGNOSTIC_PROFILE_ALL},
  {"direct_http.rejected_connections", "integer", "count", ESPECTRE_DIAGNOSTIC_PROFILE_ALL},
  {"direct_http.malformed_requests", "integer", "count", ESPECTRE_DIAGNOSTIC_PROFILE_ALL},
  {"direct_http.oversized_requests", "integer", "count", ESPECTRE_DIAGNOSTIC_PROFILE_ALL},
  {"direct_http.rate_limited_requests", "integer", "count", ESPECTRE_DIAGNOSTIC_PROFILE_ALL},
  {"direct_http.dropped_motion_events", "integer", "count", ESPECTRE_DIAGNOSTIC_PROFILE_ALL},
  {"direct_http.send_failures", "integer", "count", ESPECTRE_DIAGNOSTIC_PROFILE_ALL},
  {"raw_csi.active", "boolean", "", ESPECTRE_DIAGNOSTIC_PROFILE_NATIVE | ESPECTRE_DIAGNOSTIC_PROFILE_BRIDGE},
  {"raw_csi.binary_bound", "boolean", "", ESPECTRE_DIAGNOSTIC_PROFILE_NATIVE | ESPECTRE_DIAGNOSTIC_PROFILE_BRIDGE},
  {"raw_csi.raw_drop_total", "integer", "count", ESPECTRE_DIAGNOSTIC_PROFILE_NATIVE | ESPECTRE_DIAGNOSTIC_PROFILE_BRIDGE},
  {"raw_csi.send_backpressure_total", "integer", "count", ESPECTRE_DIAGNOSTIC_PROFILE_NATIVE | ESPECTRE_DIAGNOSTIC_PROFILE_BRIDGE},
  {"raw_csi.fresh_record_total", "integer", "count", ESPECTRE_DIAGNOSTIC_PROFILE_NATIVE | ESPECTRE_DIAGNOSTIC_PROFILE_BRIDGE},
  {"raw_csi.stream_sequence", "integer", "count", ESPECTRE_DIAGNOSTIC_PROFILE_NATIVE | ESPECTRE_DIAGNOSTIC_PROFILE_BRIDGE},
  {"mqtt.connected", "boolean", "", ESPECTRE_DIAGNOSTIC_PROFILE_NATIVE},
  {"mqtt.queue_capacity", "integer", "count", ESPECTRE_DIAGNOSTIC_PROFILE_NATIVE},
  {"mqtt.outbox_capacity_bytes", "integer", "bytes", ESPECTRE_DIAGNOSTIC_PROFILE_NATIVE},
  {"mqtt.queued_publishes", "integer", "count", ESPECTRE_DIAGNOSTIC_PROFILE_NATIVE},
  {"mqtt.dropped_publishes", "integer", "count", ESPECTRE_DIAGNOSTIC_PROFILE_NATIVE},
  {"mqtt.publish_failures", "integer", "count", ESPECTRE_DIAGNOSTIC_PROFILE_NATIVE},
  {"mqtt.reconnects", "integer", "count", ESPECTRE_DIAGNOSTIC_PROFILE_NATIVE},
};
#define ESPECTRE_DIAGNOSTIC_FIELD_COUNT \
  (sizeof(espectre_diagnostic_fields) / sizeof(espectre_diagnostic_fields[0]))
