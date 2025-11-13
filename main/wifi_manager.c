/*
 * ESPectre - WiFi Manager Module Implementation
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "wifi_manager.h"
#include <string.h>
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"

static const char *TAG = "WiFi_Manager";

static EventGroupHandle_t s_wifi_event_group = NULL;
static wifi_manager_state_t *s_state = NULL;

#define WIFI_CONNECTED_BIT BIT0

static void wifi_event_handler(void* arg, esp_event_base_t event_base,
                               int32_t event_id, void* event_data) {
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        ESP_LOGI(TAG, "WiFi STA started, attempting connection...");
        esp_wifi_connect();
    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        wifi_event_sta_disconnected_t* disconnected = (wifi_event_sta_disconnected_t*) event_data;
        ESP_LOGW(TAG, "WiFi disconnected, reason: %d, reconnecting...", disconnected->reason);
        if (s_state) s_state->connected = false;
        esp_wifi_connect();
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t* event = (ip_event_got_ip_t*) event_data;
        ESP_LOGI(TAG, "WiFi connected, got IP: " IPSTR, IP2STR(&event->ip_info.ip));
        if (s_state) s_state->connected = true;
        if (s_wifi_event_group) {
            xEventGroupSetBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
        }
    }
}

esp_err_t wifi_manager_init(wifi_manager_state_t *state, const wifi_credentials_t *config) {
    if (!state || !config) {
        ESP_LOGE(TAG, "wifi_manager_init: NULL pointer");
        return ESP_ERR_INVALID_ARG;
    }
    
    s_state = state;
    state->connected = false;
    state->initialized = false;
    
    s_wifi_event_group = xEventGroupCreate();
    if (!s_wifi_event_group) {
        ESP_LOGE(TAG, "Failed to create event group");
        return ESP_ERR_NO_MEM;
    }
    
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();
    
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    
    ESP_ERROR_CHECK(esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, NULL));
    ESP_ERROR_CHECK(esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP, &wifi_event_handler, NULL));
    
    wifi_config_t wifi_config = {0};
    strncpy((char *)wifi_config.sta.ssid, config->ssid, sizeof(wifi_config.sta.ssid) - 1);
    wifi_config.sta.ssid[sizeof(wifi_config.sta.ssid) - 1] = '\0';
    strncpy((char *)wifi_config.sta.password, config->password, sizeof(wifi_config.sta.password) - 1);
    wifi_config.sta.password[sizeof(wifi_config.sta.password) - 1] = '\0';
    
    ESP_LOGI(TAG, "WiFi SSID: %s", wifi_config.sta.ssid);
    
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());
    
    state->initialized = true;
    ESP_LOGI(TAG, "WiFi initialization complete");
    
    return ESP_OK;
}

esp_err_t wifi_manager_wait_connected(void) {
    if (!s_wifi_event_group) {
        ESP_LOGE(TAG, "WiFi event group not initialized");
        return ESP_ERR_INVALID_STATE;
    }
    
    ESP_LOGI(TAG, "Waiting for WiFi connection...");
    xEventGroupWaitBits(s_wifi_event_group, WIFI_CONNECTED_BIT, pdFALSE, pdTRUE, portMAX_DELAY);
    
    return ESP_OK;
}

bool wifi_manager_is_connected(const wifi_manager_state_t *state) {
    return state ? state->connected : false;
}

void wifi_manager_cleanup(wifi_manager_state_t *state) {
    if (state) {
        if (state->initialized) {
            // Disable promiscuous mode before stopping WiFi
            esp_wifi_set_promiscuous(false);
            esp_wifi_set_csi(false);
            esp_wifi_stop();
            esp_wifi_deinit();
        }
        if (s_wifi_event_group) {
            vEventGroupDelete(s_wifi_event_group);
            s_wifi_event_group = NULL;
        }
        s_state = NULL;
        state->connected = false;
        state->initialized = false;
    }
}
