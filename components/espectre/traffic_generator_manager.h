/*
 * ESPectre - Traffic Generator Manager
 * 
 * Generates WiFi traffic using UDP/DNS queries or ICMP ping to ensure CSI data availability.
 * Supports two modes:
 *   - dns: UDP queries to gateway:53 (default, lower overhead)
 *   - ping: ICMP echo to gateway (more compatible with all routers)
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#pragma once

#include "esp_err.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "ping/ping_sock.h"
#include <cstdint>
#include <sys/types.h>  // for ssize_t
#include <string>

namespace esphome {
namespace espectre {

/**
 * Send error state for rate-limited logging
 * 
 * Tracks error count and last log time for rate-limited error logging.
 * Used by TrafficGeneratorManager to avoid console spam during memory pressure.
 */
struct SendErrorState {
  uint32_t error_count{0};
  int64_t last_log_time{0};
  static constexpr int64_t LOG_INTERVAL_US = 1000000;  // 1 second
};

/**
 * Handle send error with rate-limited logging and adaptive backoff
 * 
 * @param state Error state (updated in place)
 * @param sent Return value from sendto()
 * @param err_no Current errno value
 * @param current_time Current time in microseconds
 * @return true if backoff delay should be applied (ENOMEM detected)
 */
inline bool handle_send_error(SendErrorState& state, ssize_t sent, int err_no, int64_t current_time) {
  state.error_count++;
  
  // Rate-limit error logging: log at most once per second to avoid console spam
  // during high-load periods (e.g., SPIFFS calibration + UDP can cause ENOMEM)
  if (current_time - state.last_log_time > SendErrorState::LOG_INTERVAL_US) {
    // Logging would happen here on ESP32 (ESP_LOGW)
    // For testing, we just update state
    state.error_count = 0;
    state.last_log_time = current_time;
  }
  
  // Return true if adaptive backoff should be applied (ENOMEM)
  return err_no == 12;  // ENOMEM
}

/**
 * Traffic Generator Mode
 */
enum class TrafficGeneratorMode {
  DNS,   // UDP DNS queries to gateway:53 (default)
  PING   // ICMP echo requests to gateway
};

/**
 * Traffic Generator Manager
 * 
 * Generates continuous WiFi traffic using UDP/DNS queries or ICMP ping
 * to ensure CSI data availability.
 * 
 * DNS mode: fire-and-forget UDP queries, lower overhead
 * Ping mode: ICMP echo requests, more compatible with all routers
 */
class TrafficGeneratorManager {
 public:
  /**
   * Initialize traffic generator with rate and mode
   * 
   * @param rate_pps Packets per second (typically 100)
   * @param mode Traffic generation mode (dns or ping)
   */
  void init(uint32_t rate_pps, TrafficGeneratorMode mode = TrafficGeneratorMode::DNS);
  
  /**
   * Start traffic generator
   * 
   * Uses the rate configured in init().
   * 
   * @return true if started successfully
   */
  bool start();
  
  /**
   * Stop traffic generator
   */
  void stop();
  
  /**
   * Check if traffic generator is running
   * 
   * @return true if running, false otherwise
   */
  bool is_running() const { return running_; }
  
  /**
   * Pause traffic generator
   * 
   * Temporarily stops sending packets without destroying the task.
   * Use resume() to continue. Useful during calibration to avoid
   * wasting CPU cycles on traffic that won't be processed.
   */
  void pause();
  
  /**
   * Resume traffic generator after pause
   */
  void resume();
  
  /**
   * Check if traffic generator is paused
   * 
   * @return true if paused, false otherwise
   */
  bool is_paused() const { return paused_; }
  
 private:
  // FreeRTOS task function (static wrapper) - DNS mode only
  static void dns_traffic_task_(void* arg);
  
  // Ping callback (called by esp_ping for each response)
  static void ping_success_cb_(esp_ping_handle_t hdl, void *args);
  static void ping_timeout_cb_(esp_ping_handle_t hdl, void *args);
  static void ping_end_cb_(esp_ping_handle_t hdl, void *args);
  
  // State
  TaskHandle_t task_handle_{nullptr};
  int sock_{-1};
  esp_ping_handle_t ping_handle_{nullptr};
  uint32_t rate_pps_{0};
  TrafficGeneratorMode mode_{TrafficGeneratorMode::DNS};
  volatile bool running_{false};  // volatile: accessed from main task and FreeRTOS task
  volatile bool paused_{false};   // volatile: accessed from main task and FreeRTOS task
  
  // Mode-specific start/stop
  bool start_dns_();
  bool start_ping_();
  void stop_dns_();
  void stop_ping_();
};

}  // namespace espectre
}  // namespace esphome
