/*
 * ESPectre - Traffic Generator Manager
 *
 * Generates paced traffic to the configured IPv4 target or associated AP at the configured
 * CSI target. Scheduling, local send backoff, and stall logging are shared by
 * all backends. Occupancy never changes the send rate.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

/**
 * @file traffic_generator_manager.h
 * @brief ESP-IDF managed traffic for firmware that owns its CSI capture path.
 *
 * Link ESPECTRE_RUNTIME_ESP_IDF_TRAFFIC_SOURCES and its ESP-IDF dependencies.
 * The firmware owns the object and calls stop() before tearing down Wi-Fi.
 * Call lifecycle and control methods from one owner task.
 *
 * stop() only signals the worker and wakes it from a pacing wait, so it never
 * blocks the owner loop. Keep calling loop() after stop(): it reaps the worker
 * once that worker has suspended after leaving its send. Deleting it while it
 * may be inside lwIP, or while vTaskSuspend is still running on another core,
 * can leave the stack locked or corrupt the scheduler. A worker that has not
 * exited within 2 s is logged, and loop() keeps waiting; after 30 s,
 * consume_stop_timeout() is true once. If that worker exits before the flag is
 * read, the timeout is dropped. The radio is free to reconfigure once
 * is_quiescent() is true. Destroying the generator waits until its worker has
 * exited, so keep one generator alive across sensing restarts.
 * A start() made while the previous worker is still exiting returns true and
 * launches the new worker from a later loop(), unless hold_pending_restart()
 * is set. If that launch fails, consume_start_failure() is true and
 * is_running() turns false.
 */

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <sys/types.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "runtime/csi_traffic_service.h"

namespace espectre {

/// @cond INTERNAL
struct SendErrorState {
  uint32_t error_count{0};
  int64_t last_log_time{0};
  static constexpr int64_t LOG_INTERVAL_US = 1000000;
};

inline bool handle_send_error(SendErrorState &state, ssize_t sent, int err_no, int64_t current_time) {
  (void) sent;
  state.error_count++;
  if (current_time - state.last_log_time > SendErrorState::LOG_INTERVAL_US) {
    state.error_count = 0;
    state.last_log_time = current_time;
  }
  return err_no == 12;
}

inline int64_t next_traffic_send_deadline_us(int64_t previous_deadline_us,
                                             int64_t send_started_us,
                                             int64_t interval_us) {
  if (interval_us <= 0) {
    return send_started_us;
  }
  if (previous_deadline_us <= 0) {
    return send_started_us + interval_us;
  }

  const int64_t phase_deadline_us = previous_deadline_us + interval_us;
  const int64_t remaining_us = phase_deadline_us - send_started_us;
  if (remaining_us < interval_us / 2) {
    return send_started_us + interval_us;
  }
  return phase_deadline_us;
}

// Report a blocked or late send at most once per min_interval_us. The log of
// one stall must not become a warning on every following packet.
// `last_report_us` stays negative until the first report.
inline bool should_report_traffic_send_delay(int64_t now_us, int64_t &last_report_us,
                                             int64_t blocked_us, int64_t late_us,
                                             int64_t report_after_us, int64_t min_interval_us) {
  if (blocked_us < report_after_us && late_us < report_after_us) {
    return false;
  }
  // A negative stamp means no report yet, so a report at time zero still counts.
  if (last_report_us >= 0 && now_us - last_report_us < min_interval_us) {
    return false;
  }
  last_report_us = now_us;
  return true;
}

constexpr size_t TRAFFIC_DNS_QUERY_PAYLOAD_SIZE = 17U;
constexpr size_t TRAFFIC_DNS_TCP_FRAME_SIZE = TRAFFIC_DNS_QUERY_PAYLOAD_SIZE + 2U;
constexpr size_t TRAFFIC_NULL_DATA_FRAME_SIZE = 24U;

size_t build_null_data_frame(const uint8_t *bssid, const uint8_t *station_mac,
                             uint8_t *buffer, size_t buffer_len);

size_t build_dns_query_payload(uint16_t transaction_id,
                               uint8_t *buffer,
                               size_t buffer_len);

size_t build_dns_tcp_query_frame(uint16_t transaction_id,
                                 uint8_t *buffer,
                                 size_t buffer_len);
/// @endcond

/** Paced ESP-IDF traffic generator with a firmware-owned lifecycle. */
class TrafficGeneratorManager : public ICsiTrafficGenerator {
 public:
  /**
   * Stop the worker and wait until it exits, however long its socket call takes.
   *
   * Destroy the generator outside a watched loop, or keep it for the device's lifetime.
   */
  ~TrafficGeneratorManager() override;

  /** Configure the send rate and backend while stopped. */
  void init(uint32_t target_pps,
            TrafficGeneratorMode mode = TrafficGeneratorMode::PING) override;

  /** Start sending to an IPv4 address in network byte order; WIFI_RAW ignores the address. */
  bool start(uint32_t target_addr) override;
  /** Finish a pending stop or restart, and report a stalled generator, from the owner task. */
  void loop() override;
  /** Signal the worker to exit and return without waiting for it. */
  void stop() override;
  /** Keep loop() from launching a deferred start while the owner still owns the radio. */
  void hold_pending_restart(bool hold) override { hold_restart_ = hold; }

  /** Whether the worker runs or a deferred start is waiting to launch it. */
  bool is_running() const override {
    return running_.load(std::memory_order_relaxed) || restart_pending_;
  }
  /** True when the worker has left its send, even if loop() has not reaped it yet. */
  bool is_quiescent() const override {
    return !running_.load(std::memory_order_acquire) &&
           task_exited_.load(std::memory_order_acquire);
  }
  /** True when start() has a worker that stop() has not signalled. */
  bool has_live_worker() const override {
    return running_.load(std::memory_order_acquire) &&
           !task_exited_.load(std::memory_order_acquire);
  }
  /** True once, after loop() fails to launch a deferred start. */
  bool consume_start_failure() override {
    const bool failed = start_failed_;
    start_failed_ = false;
    return failed;
  }
  /**
   * True once, after a stopped worker has not exited within 30 s.
   *
   * An unread timeout is dropped when that worker exits.
   */
  bool consume_stop_timeout() override {
    const bool timed_out = stop_timed_out_;
    stop_timed_out_ = false;
    return timed_out;
  }
  /** Suspend sends without destroying the worker. */
  void pause();
  /** Resume sends after pause(). */
  void resume();
  bool is_paused() const { return paused_.load(std::memory_order_relaxed); }

  /** Rate requested by init(), in packets per second. */
  uint32_t target_rate_pps() const { return target_pps_; }
  /** Send rate used by the worker, in packets per second. */
  uint32_t current_rate_pps() const { return current_rate_pps_.load(std::memory_order_relaxed); }
  /** Number of successful sends in the current session. */
  uint32_t send_success_count() const override {
    return send_success_count_.load(std::memory_order_relaxed);
  }
  /** Number of failed sends in the current session. */
  uint32_t send_error_count() const { return send_error_count_.load(std::memory_order_relaxed); }
  /** ICMP identifier used to recognize this generator's ping replies. */
  uint16_t icmp_identifier() const override { return icmp_identifier_; }

 private:
  static void traffic_task_(void *arg);
  bool launch_(uint32_t target_addr);
  bool reap_stopped_task_();
  bool complete_stop_();
  void apply_init_();
  void reset_runtime_state_();
  void wait_for_stop_(TickType_t ticks);

  // Owner-task state: only the owner creates and deletes the worker.
  TaskHandle_t task_handle_{nullptr};
  std::atomic<int> sock_{-1};
  uint32_t target_addr_{0U};
  uint32_t target_pps_{0U};
  uint32_t staged_target_pps_{0U};
  TrafficGeneratorMode mode_{TrafficGeneratorMode::PING};
  TrafficGeneratorMode staged_mode_{TrafficGeneratorMode::PING};
  bool init_pending_{false};
  uint16_t icmp_identifier_{0U};
  uint8_t null_data_frame_[TRAFFIC_NULL_DATA_FRAME_SIZE]{};
  std::atomic<uint32_t> current_rate_pps_{0U};
  std::atomic<bool> running_{false};
  std::atomic<bool> paused_{false};
  std::atomic<bool> task_exited_{true};
  // Owner-task state: a signalled worker not yet reaped, and a start() that
  // waits for it to exit.
  bool stop_pending_{false};
  int64_t stop_started_us_{0};
  bool stop_stall_logged_{false};
  bool stop_fault_reported_{false};
  bool stop_timed_out_{false};
  bool restart_pending_{false};
  bool hold_restart_{false};
  bool start_failed_{false};
  uint32_t restart_target_addr_{0U};
  std::atomic<uint32_t> send_success_count_{0U};
  std::atomic<uint32_t> send_error_count_{0U};
  uint32_t previous_send_success_count_{0U};
  int64_t last_send_progress_us_{0};
  int64_t last_health_check_us_{0};

  static constexpr int64_t HEALTH_CHECK_INTERVAL_US = 1000000;
  static constexpr int64_t SEND_STALL_TIMEOUT_US = 5000000;
  // A stopped worker can only wait inside lwIP: its sockets never block on
  // I/O. Wi-Fi TX stalls seen on hardware last 2-3 s, and a lost AP ends by
  // the station's beacon timeout (6 s by default), which fails pending frames.
  // Past 30 s neither explains the wait, so a stack deadlock is plausible.
  static constexpr int64_t STOP_STALL_LOG_US = 2000000;
  static constexpr int64_t STOP_FAULT_US = 30000000;
  // A send that blocks or starts this late is reported, so a stall can be told
  // apart as a blocked socket or TX path versus a task that did not run.
  // The warning itself is limited to one per interval, so a slow log cannot
  // turn one stall into a warning on every later packet.
  static constexpr int64_t SEND_DELAY_REPORT_US = 100000;
  static constexpr int64_t SEND_DELAY_REPORT_INTERVAL_US = 1000000;
  static constexpr uint32_t CONSECUTIVE_ERROR_REOPEN_THRESHOLD = 32U;
};

}  // namespace espectre
