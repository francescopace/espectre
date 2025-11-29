/*
 * ESPectre - Traffic Generator Manager Implementation
 * 
 * Manages UDP/DNS-based traffic generator for CSI packet generation.
 * Optimized for minimal overhead and aligned with micro-espectre.
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "traffic_generator_manager.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_netif.h"
#include "lwip/sockets.h"
#include "lwip/inet.h"
#include <cstring>

namespace esphome {
namespace espectre {

static const char *TAG = "TrafficGen";

// Pre-built DNS query for google.com (type A)
// This is a minimal DNS query that will generate a response
static const uint8_t DNS_QUERY[] = {
    0x00, 0x01,  // Transaction ID
    0x01, 0x00,  // Flags: standard query
    0x00, 0x01,  // Questions: 1
    0x00, 0x00,  // Answer RRs: 0
    0x00, 0x00,  // Authority RRs: 0
    0x00, 0x00,  // Additional RRs: 0
    // Query: google.com
    0x06, 0x67, 0x6f, 0x6f, 0x67, 0x6c, 0x65,  // "google"
    0x03, 0x63, 0x6f, 0x6d,  // "com"
    0x00,  // End of name
    0x00, 0x01,  // Type: A
    0x00, 0x01   // Class: IN
};

// ============================================================================
// PUBLIC API
// ============================================================================

void TrafficGeneratorManager::init(uint32_t rate_pps) {
  task_handle_ = nullptr;
  sock_ = -1;
  rate_pps_ = rate_pps;
  running_ = false;
  
  ESP_LOGD(TAG, "Traffic Generator Manager initialized (rate: %u pps)", rate_pps);
}

bool TrafficGeneratorManager::start() {
  if (running_) {
    ESP_LOGW(TAG, "Traffic generator already running");
    return false;
  }
  
  // Validate rate
  if (rate_pps_ == 0) {
    ESP_LOGE(TAG, "Invalid rate: 0 pps (must be > 0)");
    return false;
  }
  
  // Get gateway IP address
  esp_netif_t *netif = esp_netif_get_handle_from_ifkey("WIFI_STA_DEF");
  if (!netif) {
    ESP_LOGE(TAG, "Failed to get network interface");
    return false;
  }
  
  esp_netif_ip_info_t ip_info;
  if (esp_netif_get_ip_info(netif, &ip_info) != ESP_OK) {
    ESP_LOGE(TAG, "Failed to get IP info");
    return false;
  }
  
  if (ip_info.gw.addr == 0) {
    ESP_LOGE(TAG, "Gateway IP not available");
    return false;
  }
  
  // Log gateway IP
  char gw_str[16];
  snprintf(gw_str, sizeof(gw_str), IPSTR, IP2STR(&ip_info.gw));
  ESP_LOGI(TAG, "Target gateway: %s", gw_str);
  
  // Create UDP socket
  sock_ = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
  if (sock_ < 0) {
    ESP_LOGE(TAG, "Failed to create socket");
    return false;
  }
  
  // Set socket to non-blocking mode for fire-and-forget operation
  int flags = fcntl(sock_, F_GETFL, 0);
  if (fcntl(sock_, F_SETFL, flags | O_NONBLOCK) < 0) {
    ESP_LOGW(TAG, "Failed to set socket non-blocking (continuing anyway)");
  }
  
  // Reset counters
  running_ = true;
  
  // Create FreeRTOS task
  // Stack size: 4096 bytes (increased for safety)
  // Priority: 5 (medium priority, same as other network tasks)
  BaseType_t result = xTaskCreate(
      traffic_task_,
      "traffic_gen",
      4096,
      this,
      5,
      &task_handle_
  );
  
  if (result != pdPASS) {
    ESP_LOGE(TAG, "Failed to create traffic generator task (result: %d)", result);
    close(sock_);
    sock_ = -1;
    running_ = false;
    return false;
  }
  
  // Give task time to start
  vTaskDelay(pdMS_TO_TICKS(100));
  
  uint32_t interval_ms = 1000 / rate_pps_;
  ESP_LOGI(TAG, "📡 Traffic generator started (%u pps, interval: %u ms)", 
           rate_pps_, interval_ms);
  
  return true;
}

void TrafficGeneratorManager::stop() {
  if (!running_) {
    return;
  }
  
  running_ = false;
  
  // Wait for task to finish (max 1 second)
  if (task_handle_) {
    for (int i = 0; i < 10 && eTaskGetState(task_handle_) != eDeleted; i++) {
      vTaskDelay(pdMS_TO_TICKS(100));
    }
    task_handle_ = nullptr;
  }
  
  // Close socket
  if (sock_ >= 0) {
    close(sock_);
    sock_ = -1;
  }
  
  ESP_LOGI(TAG, "📡 Traffic generator stopped");
}

void TrafficGeneratorManager::set_rate(uint32_t rate_pps) {
  if (!running_) {
    ESP_LOGW(TAG, "Cannot set rate: traffic generator not running");
    return;
  }
  
  // Validate rate
  if (rate_pps == 0) {
    ESP_LOGE(TAG, "Invalid rate: 0 pps (must be > 0)");
    return;
  }
  
  if (rate_pps == rate_pps_) {
    return;  // No change needed
  }
  
  // Update rate and restart
  rate_pps_ = rate_pps;
  stop();
  
  // Start new session with new rate
  if (start()) {
    ESP_LOGI(TAG, "📡 Traffic rate changed to %u packets/sec", rate_pps);
  } else {
    ESP_LOGE(TAG, "Failed to restart traffic generator with new rate");
  }
}

// ============================================================================
// PRIVATE TASK
// ============================================================================

void TrafficGeneratorManager::traffic_task_(void* arg) {
  TrafficGeneratorManager* mgr = static_cast<TrafficGeneratorManager*>(arg);
  if (!mgr) {
    ESP_LOGE(TAG, "Invalid manager pointer");
    vTaskDelete(NULL);
    return;
  }
  
  // Get gateway address
  esp_netif_t *netif = esp_netif_get_handle_from_ifkey("WIFI_STA_DEF");
  if (!netif) {
    ESP_LOGE(TAG, "Failed to get netif in task");
    vTaskDelete(NULL);
    return;
  }
  
  esp_netif_ip_info_t ip_info;
  if (esp_netif_get_ip_info(netif, &ip_info) != ESP_OK) {
    ESP_LOGE(TAG, "Failed to get IP info in task");
    vTaskDelete(NULL);
    return;
  }
  
  // Setup destination address (gateway:53 for DNS)
  struct sockaddr_in dest_addr;
  memset(&dest_addr, 0, sizeof(dest_addr));
  dest_addr.sin_family = AF_INET;
  dest_addr.sin_port = htons(53);  // DNS port
  dest_addr.sin_addr.s_addr = ip_info.gw.addr;
  
  // Use microseconds for precise timing with fractional accumulator
  // This compensates for integer division error (e.g., 1000000/400 = 2500µs exact)
  const uint32_t interval_us = 1000000 / mgr->rate_pps_;  // Base interval in microseconds
  const uint32_t remainder_us = 1000000 % mgr->rate_pps_; // Remainder to distribute
  uint32_t accumulator = 0;  // Accumulates fractional microseconds
  
  ESP_LOGI(TAG, "📡 Traffic task started (gateway: " IPSTR ", interval: %u µs, remainder: %u)", 
           IP2STR(&ip_info.gw), interval_us, remainder_us);
  
  int64_t next_send_time = esp_timer_get_time();
  
  while (mgr->running_) {
    // Send DNS query to gateway
    ssize_t sent = sendto(
        mgr->sock_,
        DNS_QUERY,
        sizeof(DNS_QUERY),
        0,
        (struct sockaddr*)&dest_addr,
        sizeof(dest_addr)
    );
    
    if (sent <= 0) {
      // Log occasional errors
      ESP_LOGW(TAG, "Send error: %zd (errno: %d)", sent, errno);
    }
    
    // Calculate next send time with fractional accumulator for precise rate
    accumulator += remainder_us;
    uint32_t extra_us = accumulator / mgr->rate_pps_;
    accumulator %= mgr->rate_pps_;
    
    next_send_time += interval_us + extra_us;
    
    // Sleep until next send time
    int64_t now = esp_timer_get_time();
    int64_t sleep_us = next_send_time - now;
    
    if (sleep_us > 0) {
      // Convert to ticks (round up to avoid drift)
      TickType_t sleep_ticks = pdMS_TO_TICKS((sleep_us + 999) / 1000);
      if (sleep_ticks > 0) {
        vTaskDelay(sleep_ticks);
      }
    } else if (sleep_us < -100000) {
      // We're more than 100ms behind, reset timing
      next_send_time = esp_timer_get_time();
    }
  }
  
  ESP_LOGI(TAG, "📡 Traffic task stopped");
  vTaskDelete(NULL);
}

}  // namespace espectre
}  // namespace esphome
