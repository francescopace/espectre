/*
 * ESPectre - Traffic Generator Manager
 * 
 * Generates WiFi traffic using UDP/DNS queries to ensure CSI data availability.
 * Optimized for minimal overhead using raw sockets.
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#pragma once

#include "esp_err.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include <cstdint>

namespace esphome {
namespace espectre {

/**
 * Traffic Generator Manager
 * 
 * Generates continuous WiFi traffic using UDP/DNS queries to ensure CSI data availability.
 * Uses a fire-and-forget approach: sends queries without reading responses.
 * The OS automatically handles socket buffer overflow, making this very efficient.
 */
class TrafficGeneratorManager {
 public:
  /**
   * Initialize traffic generator with rate
   * 
   * @param rate_pps Packets per second (typically 100))
   */
  void init(uint32_t rate_pps);
  
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
   * Get total packet count
   * 
   * @return Number of packets sent
   */
  uint32_t get_packet_count() const { return packet_count_; }
  
  /**
   * Get error count
   * 
   * @return Number of send errors
   */
  uint32_t get_error_count() const { return error_count_; }
  
  /**
   * Set rate while running
   * 
   * Stops and restarts the generator with new rate.
   * 
   * @param rate_pps New packets per second rate
   */
  void set_rate(uint32_t rate_pps);
  
 private:
  // FreeRTOS task function (static wrapper)
  static void traffic_task_(void* arg);
  
  // State
  TaskHandle_t task_handle_{nullptr};
  int sock_{-1};
  uint32_t packet_count_{0};
  uint32_t error_count_{0};
  uint32_t rate_pps_{0};
  bool running_{false};
};

}  // namespace espectre
}  // namespace esphome
