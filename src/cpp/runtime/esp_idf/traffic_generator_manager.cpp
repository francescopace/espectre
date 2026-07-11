/*
 * ESPectre - Traffic Generator Manager Implementation
 * 
 * Manages traffic generator for CSI packet generation.
 * Supports two modes:
 *   - DNS: UDP queries to gateway:53 (default, lower overhead)
 *   - Ping: ICMP echo to gateway (more compatible with all routers)
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "traffic_generator_manager.h"
#include "espectre_log.h"
#include "esp_timer.h"
#include "esp_netif.h"
#include <cerrno>
#include <fcntl.h>
#include <inttypes.h>
#include <net/if.h>
#include "lwip/sockets.h"
#include "lwip/inet.h"
#include "lwip/ip_addr.h"
#include <cstring>

namespace espectre {

static const char *TRAFFIC_TAG = "TrafficGen";

static esp_netif_t *get_sta_netif() { return esp_netif_get_handle_from_ifkey("WIFI_STA_DEF"); }

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * Get gateway IP address from network interface
 * 
 * @param out_gw Output gateway address
 * @return true if gateway IP was obtained successfully
 */
static bool get_gateway_ip(esp_ip4_addr_t* out_gw) {
    esp_netif_t *netif = get_sta_netif();
    if (!netif) {
        ESP_LOGE(TRAFFIC_TAG, "Failed to get network interface");
        return false;
    }
    
    esp_netif_ip_info_t ip_info;
    if (esp_netif_get_ip_info(netif, &ip_info) != ESP_OK) {
        ESP_LOGE(TRAFFIC_TAG, "Failed to get IP info");
        return false;
    }
    
    if (ip_info.gw.addr == 0) {
        ESP_LOGE(TRAFFIC_TAG, "Gateway IP not available");
        return false;
    }
    
    *out_gw = ip_info.gw;
    return true;
}

static bool get_sta_netif_index(uint32_t *out_index) {
    esp_netif_t *netif = get_sta_netif();
    if (!netif) {
        ESP_LOGE(TRAFFIC_TAG, "Failed to get network interface");
        return false;
    }

    int if_index = esp_netif_get_netif_impl_index(netif);
    if (if_index <= 0) {
        ESP_LOGE(TRAFFIC_TAG, "Invalid STA netif index: %d", if_index);
        return false;
    }

    *out_index = static_cast<uint32_t>(if_index);
    return true;
}

static bool bind_socket_to_sta_interface(int sock) {
    uint32_t if_index = 0;
    if (!get_sta_netif_index(&if_index)) {
        return false;
    }

    struct ifreq iface = {};
    if (if_indextoname(if_index, iface.ifr_name) == nullptr) {
        ESP_LOGW(TRAFFIC_TAG, "Failed to resolve STA interface name for index %" PRIu32, if_index);
        return false;
    }

    if (setsockopt(sock, SOL_SOCKET, SO_BINDTODEVICE, &iface, sizeof(iface)) != 0) {
        ESP_LOGW(TRAFFIC_TAG, "Failed to bind socket to %s (errno=%d)", iface.ifr_name, errno);
        return false;
    }

    ESP_LOGI(TRAFFIC_TAG, "Bound socket to %s (index=%" PRIu32 ")", iface.ifr_name, if_index);
    return true;
}

// IP precedence 6 (TOS 0xC0): the Wi-Fi driver maps the top three TOS bits to
// the 802.11 TID, so generator traffic queues on the WMM/EDCA voice access
// category (AC_VO) for the tightest transmit timing.
static constexpr int TRAFFIC_IP_TOS_AC_VO = 0xC0;

static int create_udp_socket() {
    int sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (sock < 0) {
        ESP_LOGE(TRAFFIC_TAG, "Failed to create UDP socket (errno=%d)", errno);
        return -1;
    }

    if (!bind_socket_to_sta_interface(sock)) {
        ESP_LOGW(TRAFFIC_TAG, "Continuing without explicit UDP socket binding");
    }

    int tos = TRAFFIC_IP_TOS_AC_VO;
    if (setsockopt(sock, IPPROTO_IP, IP_TOS, &tos, sizeof(tos)) != 0) {
        ESP_LOGW(TRAFFIC_TAG, "Failed to set UDP socket TOS (errno=%d)", errno);
    }

    int flags = fcntl(sock, F_GETFL, 0);
    if (flags >= 0) {
        if (fcntl(sock, F_SETFL, flags | O_NONBLOCK) < 0) {
            ESP_LOGW(TRAFFIC_TAG, "Failed to set UDP socket non-blocking (errno=%d)", errno);
        }
    } else {
        ESP_LOGW(TRAFFIC_TAG, "Failed to read UDP socket flags (errno=%d)", errno);
    }

    return sock;
}

// Minimal DNS query for root domain (type A)
// 17 bytes - smallest valid DNS query that generates a response
static const uint8_t DNS_QUERY[] = {
    0x00, 0x01,  // Transaction ID
    0x01, 0x00,  // Flags: standard query
    0x00, 0x01,  // Questions: 1
    0x00, 0x00,  // Answer RRs: 0
    0x00, 0x00,  // Authority RRs: 0
    0x00, 0x00,  // Additional RRs: 0
    0x00,        // Root domain (empty label)
    0x00, 0x01,  // Type: A
    0x00, 0x01   // Class: IN
};

// ============================================================================
// PUBLIC API
// ============================================================================

static const char *traffic_mode_name(TrafficGeneratorMode mode) {
  switch (mode) {
    case TrafficGeneratorMode::PING:
      return "ping";
    case TrafficGeneratorMode::DNS:
      return "dns";
    default:
      return "unknown";
  }
}

void TrafficGeneratorManager::init(uint32_t rate_pps, TrafficGeneratorMode mode) {
  task_handle_ = nullptr;
  sock_ = -1;
  ping_handle_ = nullptr;
  rate_pps_ = rate_pps;
  mode_ = mode;
  running_.store(false);
  paused_.store(false);
  reset_health_state_();

  ESP_LOGD(TRAFFIC_TAG, "Traffic Generator Manager initialized (rate: %u pps, mode: %s)", rate_pps,
           traffic_mode_name(mode));
}

bool TrafficGeneratorManager::start() {
  if (running_.load()) {
    ESP_LOGW(TRAFFIC_TAG, "Traffic generator already running");
    return false;
  }

  // Validate rate
  if (rate_pps_ == 0) {
    ESP_LOGE(TRAFFIC_TAG, "Invalid rate: 0 pps (must be > 0)");
    return false;
  }

  // Start based on mode
  if (mode_ == TrafficGeneratorMode::PING) {
    return start_ping_();
  } else {
    return start_dns_task_();
  }
}

void TrafficGeneratorManager::loop() {
  if (!running_.load() || paused_.load()) {
    return;
  }

  if (mode_ != TrafficGeneratorMode::PING || ping_handle_ == nullptr) {
    return;
  }

  const int64_t now = esp_timer_get_time();
  if (last_health_check_us_ != 0 && (now - last_health_check_us_) < HEALTH_CHECK_INTERVAL_US) {
    return;
  }
  last_health_check_us_ = now;

  uint32_t requests = 0;
  if (esp_ping_get_profile(ping_handle_, ESP_PING_PROF_REQUEST, &requests, sizeof(requests)) != ESP_OK) {
    ESP_LOGW(TRAFFIC_TAG, "Failed to read ping profile");
    return;
  }

  if (requests != last_ping_request_count_) {
    last_ping_request_count_ = requests;
    last_ping_progress_us_ = now;
    return;
  }

  if (last_ping_progress_us_ != 0 && (now - last_ping_progress_us_) >= PING_STALL_TIMEOUT_US) {
    ESP_LOGW(TRAFFIC_TAG, "Ping generator stalled for %.1f s, restarting session",
             static_cast<double>(now - last_ping_progress_us_) / 1000000.0);
    if (!restart_ping_session_()) {
      ESP_LOGE(TRAFFIC_TAG, "Failed to restart stalled ping session");
    }
    last_ping_progress_us_ = now;
  }
}

void TrafficGeneratorManager::pause() {
  if (!paused_.load()) {
    paused_.store(true);
    ESP_LOGD(TRAFFIC_TAG, "Traffic generator paused");
  }
}

void TrafficGeneratorManager::resume() {
  if (paused_.load()) {
    paused_.store(false);
    ESP_LOGD(TRAFFIC_TAG, "Traffic generator resumed");
  }
}

void TrafficGeneratorManager::stop() {
  if (!running_.load()) {
    return;
  }
  
  running_.store(false);
  
  // Stop based on mode
  if (mode_ == TrafficGeneratorMode::PING) {
    stop_ping_();
  } else {
    stop_dns_task_();
  }
  
  ESP_LOGI(TRAFFIC_TAG, "Traffic generator stopped");
}


// ============================================================================
// DNS MODE IMPLEMENTATION
// ============================================================================

bool TrafficGeneratorManager::start_dns_task_() {
  esp_ip4_addr_t gw;
  if (!get_gateway_ip(&gw)) {
    return false;
  }
  
  // Log gateway IP
  char gw_str[16];
  snprintf(gw_str, sizeof(gw_str), IPSTR, IP2STR(&gw));
  ESP_LOGI(TRAFFIC_TAG, "Target gateway: %s", gw_str);
  
  // Create UDP socket
  sock_ = create_udp_socket();
  if (sock_ < 0) {
    return false;
  }

  // Reset counters
  reset_health_state_();
  running_.store(true);

  // Create FreeRTOS task
  // Stack size: 3072 bytes is enough for the fixed DNS payload path.
  // Priority: 5 (medium priority, same as other network tasks)
  BaseType_t result = xTaskCreate(
      dns_traffic_task_,
      "traffic_gen",
      3072,
      this,
      5,
      &task_handle_
  );

  if (result != pdPASS) {
    ESP_LOGE(TRAFFIC_TAG, "Failed to create traffic generator task (result: %d)", result);
    close(sock_);
    sock_ = -1;
    running_.store(false);
    return false;
  }

  // Give task time to start
  vTaskDelay(pdMS_TO_TICKS(100));

  uint32_t interval_ms = 1000 / rate_pps_;
  ESP_LOGI(TRAFFIC_TAG, "Traffic generator started (mode: %s, %u pps, interval: %u ms)",
           traffic_mode_name(mode_), rate_pps_, interval_ms);

  return true;
}

void TrafficGeneratorManager::stop_dns_task_() {
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
}

void TrafficGeneratorManager::dns_traffic_task_(void* arg) {
  TrafficGeneratorManager* mgr = static_cast<TrafficGeneratorManager*>(arg);
  if (!mgr) {
    ESP_LOGE(TRAFFIC_TAG, "Invalid manager pointer");
    vTaskDelete(NULL);
    return;
  }

  // Setup destination address (gateway:53 for DNS).
  struct sockaddr_in dest_addr;
  memset(&dest_addr, 0, sizeof(dest_addr));
  dest_addr.sin_family = AF_INET;
  esp_ip4_addr_t gw;
  if (!get_gateway_ip(&gw)) {
    ESP_LOGE(TRAFFIC_TAG, "Failed to get gateway in task");
    mgr->running_.store(false);
    vTaskDelete(NULL);
    return;
  }
  dest_addr.sin_port = htons(53);  // DNS port
  dest_addr.sin_addr.s_addr = gw.addr;
  ESP_LOGI(TRAFFIC_TAG, "Traffic task target: " IPSTR ":53", IP2STR(&gw));

  // Use microseconds for precise timing with fractional accumulator
  // This compensates for integer division error (e.g., 1000000/400 = 2500µs exact)
  const uint32_t interval_us = 1000000 / mgr->rate_pps_;  // Base interval in microseconds
  const uint32_t remainder_us = 1000000 % mgr->rate_pps_; // Remainder to distribute
  uint32_t accumulator = 0;  // Accumulates fractional microseconds

  ESP_LOGI(TRAFFIC_TAG, "Traffic task started (mode: %s, interval: %u µs, remainder: %u)",
           traffic_mode_name(mgr->mode_), interval_us, remainder_us);

  int64_t next_send_time = esp_timer_get_time();

  // Error state for rate-limited logging
  SendErrorState error_state;
  uint32_t consecutive_send_errors = 0;

  while (mgr->running_.load()) {
    // Check if paused (e.g., during calibration)
    if (mgr->paused_.load()) {
      vTaskDelay(pdMS_TO_TICKS(50));  // Sleep while paused to save CPU
      next_send_time = esp_timer_get_time();  // Reset timing on resume
      continue;
    }

    ssize_t sent = sendto(
        mgr->sock_,
        DNS_QUERY,
        sizeof(DNS_QUERY),
        0,
        (struct sockaddr*)&dest_addr,
        sizeof(dest_addr)
    );

    if (sent <= 0) {
      mgr->send_error_count_.fetch_add(1);
      consecutive_send_errors++;

      // Handle error with rate-limited logging
      const int current_errno = errno;
      const int64_t now_us = esp_timer_get_time();
      bool should_log = (now_us - error_state.last_log_time) > SendErrorState::LOG_INTERVAL_US;
      bool needs_backoff = handle_send_error(error_state, sent, current_errno, now_us);
      if (should_log) {
        ESP_LOGW(TRAFFIC_TAG, "DNS send failed (sent=%d, errno=%d, consecutive=%" PRIu32 ")",
                 static_cast<int>(sent), current_errno, consecutive_send_errors);
      }

      if (consecutive_send_errors >= DNS_CONSECUTIVE_ERROR_RESTART_THRESHOLD) {
        ESP_LOGW(TRAFFIC_TAG, "Recreating DNS socket after %" PRIu32 " consecutive send failures",
                 consecutive_send_errors);
        if (mgr->sock_ >= 0) {
          close(mgr->sock_);
        }
        mgr->sock_ = create_udp_socket();
        consecutive_send_errors = 0;
        next_send_time = esp_timer_get_time();
        if (mgr->sock_ < 0) {
          vTaskDelay(pdMS_TO_TICKS(100));
        }
        continue;
      }

      // Adaptive backoff on ENOMEM: give WiFi stack time to recover
      // during transient memory pressure in the WiFi/LwIP stack.
      if (needs_backoff) {
        vTaskDelay(pdMS_TO_TICKS(5));  // 5ms backoff on memory pressure
      }
    } else {
      mgr->send_success_count_.fetch_add(1);
      consecutive_send_errors = 0;
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

  ESP_LOGI(TRAFFIC_TAG, "DNS traffic task stopped");
  vTaskDelete(NULL);
}

// ============================================================================
// PING MODE IMPLEMENTATION
// ============================================================================

// Ping callbacks (required by esp_ping API but we don't need the data)
void TrafficGeneratorManager::ping_success_cb_(esp_ping_handle_t hdl, void *args) {
  // Ping reply received - CSI was generated, nothing else to do
}

void TrafficGeneratorManager::ping_timeout_cb_(esp_ping_handle_t hdl, void *args) {
  // Ping timeout - still generates CSI on TX, just no reply
  // This is fine for our purposes
}

void TrafficGeneratorManager::ping_end_cb_(esp_ping_handle_t hdl, void *args) {
  // Ping session ended (only called if count is finite)
}

bool TrafficGeneratorManager::start_ping_() {
  // Get gateway IP address
  esp_ip4_addr_t gw;
  if (!get_gateway_ip(&gw)) {
    return false;
  }
  
  // Log gateway IP
  char gw_str[16];
  snprintf(gw_str, sizeof(gw_str), IPSTR, IP2STR(&gw));
  ESP_LOGI(TRAFFIC_TAG, "Target gateway: %s", gw_str);
  
  // Configure ping session
  esp_ping_config_t ping_config = ESP_PING_DEFAULT_CONFIG();
  
  // Set target address
  ip_addr_t target_addr;
  IP_ADDR4(&target_addr, 
           ip4_addr1(&gw), 
           ip4_addr2(&gw), 
           ip4_addr3(&gw), 
           ip4_addr4(&gw));
  ping_config.target_addr = target_addr;

  uint32_t if_index = 0;
  if (get_sta_netif_index(&if_index)) {
    ping_config.interface = if_index;
  } else {
    ESP_LOGW(TRAFFIC_TAG, "Continuing without explicit ping interface binding");
  }
  
  // Configure timing
  ping_config.count = ESP_PING_COUNT_INFINITE;  // Run forever
  ping_config.interval_ms = 1000 / rate_pps_;   // Interval based on rate
  ping_config.timeout_ms = std::min<uint32_t>(1000, std::max<uint32_t>(200, ping_config.interval_ms * 4));
  ping_config.data_size = 0;                    // No payload (header only, smallest possible)
  ping_config.tos = TRAFFIC_IP_TOS_AC_VO;       // WMM/EDCA voice access category
  ping_config.task_stack_size = 2560;           // Stack size for ping task
  ping_config.task_prio = 5;                    // Same priority as DNS mode
  
  // Setup callbacks
  esp_ping_callbacks_t cbs = {
    .cb_args = this,
    .on_ping_success = ping_success_cb_,
    .on_ping_timeout = ping_timeout_cb_,
    .on_ping_end = ping_end_cb_,
  };
  
  // Create ping session
  esp_err_t ret = esp_ping_new_session(&ping_config, &cbs, &ping_handle_);
  if (ret != ESP_OK) {
    ESP_LOGE(TRAFFIC_TAG, "Failed to create ping session: %s", esp_err_to_name(ret));
    return false;
  }
  
  // Start ping session
  ret = esp_ping_start(ping_handle_);
  if (ret != ESP_OK) {
    ESP_LOGE(TRAFFIC_TAG, "Failed to start ping session: %s", esp_err_to_name(ret));
    esp_ping_delete_session(ping_handle_);
    ping_handle_ = nullptr;
    return false;
  }
  
  reset_health_state_();
  running_.store(true);
  
  uint32_t interval_ms = 1000 / rate_pps_;
  ESP_LOGI(TRAFFIC_TAG,
           "Traffic generator started (mode: ping, %u pps, interval: %u ms, timeout: %u ms, if_index: %" PRIu32 ")",
           rate_pps_, interval_ms, ping_config.timeout_ms, ping_config.interface);
  
  return true;
}

void TrafficGeneratorManager::stop_ping_() {
  if (ping_handle_) {
    esp_ping_stop(ping_handle_);
    esp_ping_delete_session(ping_handle_);
    ping_handle_ = nullptr;
  }
}

bool TrafficGeneratorManager::restart_ping_session_() {
  stop_ping_();
  running_.store(false);
  return start_ping_();
}

void TrafficGeneratorManager::reset_health_state_() {
  send_success_count_.store(0);
  send_error_count_.store(0);
  last_ping_request_count_ = 0;
  last_ping_progress_us_ = esp_timer_get_time();
  last_health_check_us_ = 0;
}

}  // namespace espectre
