/*
 * ESPectre - Traffic Generator Manager
 *
 * Generates paced Wi-Fi traffic to the gateway and adapts the send rate from
 * valid local CSI feedback. Scheduling and recovery are shared by the DNS and
 * ICMP protocol backends.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#pragma once

#include <atomic>
#include <cstdint>
#include <sys/types.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "traffic_rate_controller.h"

namespace espectre {

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

enum class TrafficGeneratorMode {
  DNS,
  PING,
};

class TrafficGeneratorManager {
 public:
  void init(uint32_t target_pps,
            TrafficGeneratorMode mode = TrafficGeneratorMode::PING,
            bool adaptive_enabled = true);

  bool start();
  void loop();
  void stop();

  bool is_running() const { return running_.load(std::memory_order_relaxed); }
  void pause();
  void resume();
  bool is_paused() const { return paused_.load(std::memory_order_relaxed); }

  void observe_accepted_csi(uint64_t accepted_csi_total) {
    accepted_csi_total_.store(accepted_csi_total, std::memory_order_relaxed);
  }

  uint32_t target_rate_pps() const { return rate_controller_.target_pps(); }
  uint32_t current_rate_pps() const { return current_rate_pps_.load(std::memory_order_relaxed); }
  uint32_t observed_csi_pps() const { return rate_controller_.observed_pps(); }
  bool adaptive_enabled() const { return rate_controller_.adaptive_enabled(); }
  uint32_t send_success_count() const { return send_success_count_.load(std::memory_order_relaxed); }
  uint32_t send_error_count() const { return send_error_count_.load(std::memory_order_relaxed); }

 private:
  static void traffic_task_(void *arg);
  void reset_runtime_state_();

  TaskHandle_t task_handle_{nullptr};
  int sock_{-1};
  uint32_t gateway_addr_{0U};
  TrafficGeneratorMode mode_{TrafficGeneratorMode::PING};
  TrafficRateController rate_controller_;
  std::atomic<uint32_t> current_rate_pps_{0U};
  std::atomic<uint64_t> accepted_csi_total_{0U};
  std::atomic<bool> running_{false};
  std::atomic<bool> paused_{false};
  std::atomic<bool> task_exited_{true};
  std::atomic<uint32_t> send_success_count_{0U};
  std::atomic<uint32_t> send_error_count_{0U};
  uint32_t previous_send_success_count_{0U};
  int64_t last_send_progress_us_{0};
  int64_t last_health_check_us_{0};

  static constexpr int64_t HEALTH_CHECK_INTERVAL_US = 1000000;
  static constexpr int64_t SEND_STALL_TIMEOUT_US = 5000000;
  static constexpr uint32_t CONSECUTIVE_ERROR_REOPEN_THRESHOLD = 32U;
};

}  // namespace espectre
