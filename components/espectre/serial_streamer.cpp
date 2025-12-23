/*
 * ESPectre - Serial Streamer Implementation
 * 
 * Streams CSI motion data over USB Serial for external clients.
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "serial_streamer.h"
#include "esphome/core/log.h"
#include "esphome/core/hal.h"
#include <cstring>
#include <cstdio>
#include <cstdlib>

#include "sdkconfig.h"

// USB Serial JTAG is only available on ESP32-C3, C6, S2, S3, H2
#if defined(CONFIG_SOC_USB_SERIAL_JTAG_SUPPORTED) && CONFIG_SOC_USB_SERIAL_JTAG_SUPPORTED
#include "driver/usb_serial_jtag.h"
#define USE_USB_SERIAL_JTAG 1
#else
#define USE_USB_SERIAL_JTAG 0
#endif

namespace esphome {
namespace espectre {

static const char *const TAG = "espectre.serial";
static const char *const STREAM_TAG = "stream";  // Short tag for data stream

void SerialStreamer::init() {
  active_ = false;
  cmd_buffer_index_ = 0;
  memset(cmd_buffer_, 0, CMD_BUFFER_SIZE);
#if !USE_USB_SERIAL_JTAG
  ESP_LOGW(TAG, "USB Serial JTAG not available on this chip - serial streaming disabled");
#endif
}

void SerialStreamer::check_commands() {
#if USE_USB_SERIAL_JTAG
  // Read available bytes from USB Serial JTAG
  uint8_t byte;
  while (usb_serial_jtag_read_bytes(&byte, 1, 0) > 0) {
    if (byte == '\n' || byte == '\r') {
      // End of command
      if (cmd_buffer_index_ > 0) {
        cmd_buffer_[cmd_buffer_index_] = '\0';
        process_command_(cmd_buffer_);
        cmd_buffer_index_ = 0;
      }
    } else if (cmd_buffer_index_ < CMD_BUFFER_SIZE - 1) {
      cmd_buffer_[cmd_buffer_index_++] = (char)byte;
    }
  }
  
  // Check ping timeout (auto-stop if browser disconnected)
  if (active_ && (millis() - last_ping_time_ > PING_TIMEOUT_MS)) {
    ESP_LOGW(TAG, "Ping timeout - stopping stream");
    stop();
  }
#endif
}

void SerialStreamer::process_command_(const char* cmd) {
  if (strcmp(cmd, "START") == 0) {
    start();
  } else if (strcmp(cmd, "STOP") == 0) {
    stop();
  } else if (strcmp(cmd, "PING") == 0) {
    // Reset ping timer (keep-alive from browser)
    last_ping_time_ = millis();
  } else if (strncmp(cmd, "T:", 2) == 0) {
    // Threshold command: T:X.XX
    float threshold = atof(cmd + 2);
    if (threshold >= 0.1f && threshold <= 10.0f) {
      ESP_LOGI(TAG, "Threshold set via serial: %.2f", threshold);
      if (threshold_callback_) {
        threshold_callback_(threshold);
      }
    } else {
      ESP_LOGW(TAG, "Invalid threshold value: %.2f (must be 0.1-10.0)", threshold);
    }
  }
  // Ignore other commands (could be ESPHome debug commands)
}

void SerialStreamer::start() {
  if (!active_) {
    active_ = true;
    last_ping_time_ = millis();  // Initialize ping timer
    ESP_LOGI(TAG, "Serial streaming started");
    
    // Call start callback (used to dump system info)
    if (start_callback_) {
      start_callback_();
    }
  }
}

void SerialStreamer::stop() {
  if (active_) {
    active_ = false;
    ESP_LOGI(TAG, "Serial streaming stopped");
  }
}

void SerialStreamer::send_data(float movement, float threshold) {
#if USE_USB_SERIAL_JTAG
  if (!active_) return;
  
  // Use ESP_LOGI with short tag "G" for clean log format
  // Output: [I][G][]: 0.73,1.50
  // This ensures proper line separation and no mixing with other logs
  ESP_LOGI(STREAM_TAG, "%.2f,%.2f", movement, threshold);
#endif
}

}  // namespace espectre
}  // namespace esphome

