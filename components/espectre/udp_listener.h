/*
 * ESPectre - UDP Listener
 * 
 * Listens for UDP packets to trigger CSI generation in external traffic mode.
 * When traffic_generator_rate is 0, this listener allows external sources
 * to generate WiFi traffic that triggers CSI callbacks.
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#pragma once

#include <cstdint>

namespace esphome {
namespace espectre {

/**
 * UDP Listener for External Traffic Mode
 * 
 * Opens a UDP socket to receive packets from external sources.
 * The act of receiving packets triggers CSI callbacks in the WiFi driver.
 * No response is sent (fire-and-forget), minimizing network overhead.
 */
class UDPListener {
 public:
  /**
   * Initialize the UDP listener
   * 
   * @param port UDP port to listen on (default: 5555)
   */
  void init(uint16_t port = 5555);
  
  /**
   * Start listening for UDP packets
   * 
   * Creates a non-blocking UDP socket bound to the configured port.
   * 
   * @return true if started successfully
   */
  bool start();
  
  /**
   * Stop listening and close socket
   */
  void stop();
  
  /**
   * Check if listener is running
   */
  bool is_running() const { return running_; }
  
  /**
   * Get the listening port
   */
  uint16_t get_port() const { return port_; }
  
  /**
   * Process incoming packets (call from loop)
   * 
   * Non-blocking read of any pending UDP packets.
   * Packets are discarded after reading - we only need the WiFi CSI callback.
   */
  void loop();

 private:
  int sock_{-1};
  uint16_t port_{5555};
  bool running_{false};
};

}  // namespace espectre
}  // namespace esphome

