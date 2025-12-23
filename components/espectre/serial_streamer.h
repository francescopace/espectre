/*
 * ESPectre - Serial Streamer
 * 
 * Streams CSI motion data over USB Serial for external clients.
 * Used by the ESPectre browser game and other real-time applications.
 * 
 * Protocol:
 * - Client -> ESP32: "START\n" to enable, "STOP\n" to disable
 *                    "T:X.XX\n" to set threshold (e.g., "T:1.50\n")
 * - ESP32 -> Client: "G<movement>,<threshold>\n" every CSI packet
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <functional>

namespace esphome {
namespace espectre {

/**
 * Serial Streamer
 * 
 * Streams movement and threshold data over USB Serial when activated.
 * No overhead when inactive - only listens for START/STOP commands.
 */
// Forward declaration for callbacks
using ThresholdCallback = std::function<void(float)>;
using StartCallback = std::function<void()>;

class SerialStreamer {
 public:
  /**
   * Initialize serial streamer
   */
  void init();
  
  /**
   * Set callback for threshold changes from serial
   * 
   * @param callback Function to call when threshold is changed via serial
   */
  void set_threshold_callback(ThresholdCallback callback) { threshold_callback_ = callback; }
  
  /**
   * Set callback for start event
   * 
   * @param callback Function to call when streaming starts
   */
  void set_start_callback(StartCallback callback) { start_callback_ = callback; }
  
  /**
   * Check for incoming Serial commands
   * 
   * Should be called periodically (e.g., in loop()).
   * Parses "START\n" and "STOP\n" commands.
   */
  void check_commands();
  
  /**
   * Send data over Serial
   * 
   * Sends "G<movement>,<threshold>\n" format.
   * Only sends if streaming is active.
   * 
   * @param movement Current movement intensity (0.0+)
   * @param threshold Current detection threshold
   */
  void send_data(float movement, float threshold);
  
  /**
   * Check if streaming is currently active
   * 
   * @return true if streaming is active (client requested START)
   */
  bool is_active() const { return active_; }
  
  /**
   * Start streaming
   * 
   * Called when "START\n" command is received.
   */
  void start();
  
  /**
   * Stop streaming
   * 
   * Called when "STOP\n" command is received.
   */
  void stop();
  
 private:
  bool active_{false};  // Streaming currently active
  ThresholdCallback threshold_callback_{nullptr};
  StartCallback start_callback_{nullptr};
  
  // Ping timeout (auto-stop if no PING received)
  static constexpr uint32_t PING_TIMEOUT_MS = 5000;  // 5 seconds
  uint32_t last_ping_time_{0};
  
  // Serial command buffer
  static constexpr size_t CMD_BUFFER_SIZE = 16;
  char cmd_buffer_[CMD_BUFFER_SIZE];
  size_t cmd_buffer_index_{0};
  
  /**
   * Process a complete command line
   * 
   * @param cmd Command string (without newline)
   */
  void process_command_(const char* cmd);
};

}  // namespace espectre
}  // namespace esphome

