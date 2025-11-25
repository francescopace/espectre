/*
 * ESPectre - WiFi Traffic Generator Implementation
 * 
 * Generates WiFi traffic by sending ICMP ping requests to the gateway.
 * This ensures bidirectional traffic (request + reply) which reliably
 * triggers CSI packet generation on the ESP32.
 * 
 * Uses ESP-IDF ping component for robust ICMP implementation.
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "traffic_generator.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "ping/ping_sock.h"
#include "lwip/inet.h"
#include <string.h>

static const char *TAG = "TrafficGen";

// Traffic generator state
static esp_ping_handle_t g_ping_handle = NULL;
static uint32_t g_packet_count = 0;
static uint32_t g_timeout_count = 0;
static uint32_t g_rate_pps = 0;
static bool g_running = false;

// Ping success callback
static void on_ping_success(esp_ping_handle_t hdl, void *args) {
    g_packet_count++;
}

// Ping timeout callback
static void on_ping_timeout(esp_ping_handle_t hdl, void *args) {
    g_timeout_count++;
    
    // Log occasional timeouts
    if (g_timeout_count % 10 == 1) {
        ESP_LOGW(TAG, "Ping timeout count: %u", (unsigned int)g_timeout_count);
    }
}

// Ping end callback (called when session ends)
static void on_ping_end(esp_ping_handle_t hdl, void *args) {
    uint32_t transmitted, received;
    
    esp_ping_get_profile(hdl, ESP_PING_PROF_REQUEST, &transmitted, sizeof(transmitted));
    esp_ping_get_profile(hdl, ESP_PING_PROF_REPLY, &received, sizeof(received));
    
    uint32_t loss = 0;
    if (transmitted > 0) {
        loss = (uint32_t)((1 - ((float)received) / transmitted) * 100);
    }
    
    ESP_LOGI(TAG, "Ping session ended: %u transmitted, %u received, %u%% loss",
             (unsigned int)transmitted, (unsigned int)received, (unsigned int)loss);
}

void traffic_generator_init(void) {
    g_ping_handle = NULL;
    g_packet_count = 0;
    g_timeout_count = 0;
    g_rate_pps = 0;
    g_running = false;
    
    ESP_LOGI(TAG, "Traffic generator initialized");
}

bool traffic_generator_start(uint32_t rate_pps) {
    if (g_running) {
        ESP_LOGW(TAG, "Traffic generator already running");
        return false;
    }
    
    if (rate_pps > TRAFFIC_RATE_MAX) {
        ESP_LOGE(TAG, "Invalid rate: %u (must be 0-%u packets/sec)", (unsigned int)rate_pps, TRAFFIC_RATE_MAX);
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
    
    // Configure ping
    esp_ping_config_t config = ESP_PING_DEFAULT_CONFIG();
    
    // Set target address (gateway)
    ip_addr_t target;
    IP_ADDR4(&target, 
             ip4_addr1(&ip_info.gw), 
             ip4_addr2(&ip_info.gw),
             ip4_addr3(&ip_info.gw), 
             ip4_addr4(&ip_info.gw));
    config.target_addr = target;
    
    // Configure ping parameters
    config.count = 0;  // Infinite ping
    config.interval_ms = 1000 / rate_pps;  // Calculate interval from rate
    config.timeout_ms = 1000;
    config.data_size = 32;
    
    // Set callbacks
    esp_ping_callbacks_t cbs = {
        .cb_args = NULL,
        .on_ping_success = on_ping_success,
        .on_ping_timeout = on_ping_timeout,
        .on_ping_end = on_ping_end
    };
    
    // Create ping session
    if (esp_ping_new_session(&config, &cbs, &g_ping_handle) != ESP_OK) {
        ESP_LOGE(TAG, "Failed to create ping session");
        return false;
    }
    
    // Start ping
    if (esp_ping_start(g_ping_handle) != ESP_OK) {
        ESP_LOGE(TAG, "Failed to start ping");
        esp_ping_delete_session(g_ping_handle);
        g_ping_handle = NULL;
        return false;
    }
    
    g_packet_count = 0;
    g_timeout_count = 0;
    g_rate_pps = rate_pps;
    g_running = true;
    
    ESP_LOGI(TAG, "📡 Traffic generator started (%u pps, interval: %u ms)", 
             (unsigned int)rate_pps, (unsigned int)config.interval_ms);
    
    return true;
}

void traffic_generator_stop(void) {
    if (!g_running) {
        return;
    }
    
    if (g_ping_handle) {
        esp_ping_stop(g_ping_handle);
        esp_ping_delete_session(g_ping_handle);
        g_ping_handle = NULL;
    }
    
    ESP_LOGI(TAG, "📡 Traffic generator stopped (%u packets sent, %u timeouts)", 
             (unsigned int)g_packet_count, (unsigned int)g_timeout_count);
    
    g_running = false;
    g_rate_pps = 0;
}

bool traffic_generator_is_running(void) {
    return g_running;
}

uint32_t traffic_generator_get_packet_count(void) {
    return g_packet_count;
}

void traffic_generator_set_rate(uint32_t rate_pps) {
    if (!g_running) {
        ESP_LOGW(TAG, "Cannot set rate: traffic generator not running");
        return;
    }
    
    if (rate_pps > TRAFFIC_RATE_MAX) {
        ESP_LOGE(TAG, "Invalid rate: %u (must be 0-%u packets/sec)", (unsigned int)rate_pps, TRAFFIC_RATE_MAX);
        return;
    }
    
    if (rate_pps == g_rate_pps) {
        return;  // No change needed
    }
    
    // Stop current session
    traffic_generator_stop();
    
    // Start new session with new rate
    if (traffic_generator_start(rate_pps)) {
        ESP_LOGI(TAG, "📡 Traffic rate changed to %u packets/sec", (unsigned int)rate_pps);
    } else {
        ESP_LOGE(TAG, "Failed to restart traffic generator with new rate");
    }
}
