/*
 * ESPectre - WiFi Traffic Generator Implementation
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "traffic_generator.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "lwip/sockets.h"
#include "lwip/netdb.h"
#include <string.h>

static const char *TAG = "TrafficGen";

// Traffic generator state
static esp_timer_handle_t g_timer = NULL;
static int g_socket = -1;
static uint32_t g_packet_count = 0;
static uint32_t g_rate_pps = 0;
static bool g_running = false;

// Timer callback - sends UDP broadcast
static void timer_callback(void *arg) {
    if (g_socket < 0) return;
    
    // Create message with counter
    char message[64];
    snprintf(message, sizeof(message), "tg_%lu", (unsigned long)g_packet_count++);
    
    // Broadcast address
    struct sockaddr_in dest_addr;
    dest_addr.sin_family = AF_INET;
    dest_addr.sin_port = htons(12345);
    dest_addr.sin_addr.s_addr = htonl(INADDR_BROADCAST);
    
    // Send broadcast packet
    sendto(g_socket, message, strlen(message), 0,
           (struct sockaddr *)&dest_addr, sizeof(dest_addr));
}

void traffic_generator_init(void) {
    g_timer = NULL;
    g_socket = -1;
    g_packet_count = 0;
    g_rate_pps = 0;
    g_running = false;
    
    ESP_LOGI(TAG, "Traffic generator initialized");
}

bool traffic_generator_start(uint32_t rate_pps) {
    if (g_running) {
        ESP_LOGW(TAG, "Traffic generator already running");
        return false;
    }
    
    if (rate_pps < 1 || rate_pps > 50) {
        ESP_LOGE(TAG, "Invalid rate: %u (must be 1-50 packets/sec)", (unsigned int)rate_pps);
        return false;
    }
    
    // Create UDP socket
    g_socket = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);
    if (g_socket < 0) {
        ESP_LOGE(TAG, "Failed to create socket");
        return false;
    }
    
    // Enable broadcast
    int broadcast = 1;
    if (setsockopt(g_socket, SOL_SOCKET, SO_BROADCAST, &broadcast, sizeof(broadcast)) < 0) {
        ESP_LOGE(TAG, "Failed to enable broadcast");
        close(g_socket);
        g_socket = -1;
        return false;
    }
    
    // Create periodic timer
    esp_timer_create_args_t timer_args = {
        .callback = timer_callback,
        .name = "traffic_gen"
    };
    
    if (esp_timer_create(&timer_args, &g_timer) != ESP_OK) {
        ESP_LOGE(TAG, "Failed to create timer");
        close(g_socket);
        g_socket = -1;
        return false;
    }
    
    // Calculate period in microseconds
    uint64_t period_us = 1000000 / rate_pps;
    
    // Start timer
    if (esp_timer_start_periodic(g_timer, period_us) != ESP_OK) {
        ESP_LOGE(TAG, "Failed to start timer");
        esp_timer_delete(g_timer);
        g_timer = NULL;
        close(g_socket);
        g_socket = -1;
        return false;
    }
    
    g_packet_count = 0;
    g_rate_pps = rate_pps;
    g_running = true;
    
    ESP_LOGI(TAG, "📡 Traffic generator started (%u packets/sec)", (unsigned int)rate_pps);
    return true;
}

void traffic_generator_stop(void) {
    if (!g_running) {
        return;
    }
    
    if (g_timer) {
        esp_timer_stop(g_timer);
        esp_timer_delete(g_timer);
        g_timer = NULL;
    }
    
    if (g_socket >= 0) {
        close(g_socket);
        g_socket = -1;
    }
    
    ESP_LOGI(TAG, "📡 Traffic generator stopped (%u packets sent)", (unsigned int)g_packet_count);
    
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
    
    if (rate_pps < 1 || rate_pps > 50) {
        ESP_LOGE(TAG, "Invalid rate: %u (must be 1-50 packets/sec)", (unsigned int)rate_pps);
        return;
    }
    
    if (rate_pps == g_rate_pps) {
        return;  // No change needed
    }
    
    // Stop and restart timer with new period
    if (g_timer) {
        esp_timer_stop(g_timer);
        
        uint64_t period_us = 1000000 / rate_pps;
        
        if (esp_timer_start_periodic(g_timer, period_us) == ESP_OK) {
            ESP_LOGI(TAG, "📡 Traffic rate changed: %u -> %u packets/sec", 
                     (unsigned int)g_rate_pps, (unsigned int)rate_pps);
            g_rate_pps = rate_pps;
        } else {
            ESP_LOGE(TAG, "Failed to restart timer with new rate");
        }
    }
}
