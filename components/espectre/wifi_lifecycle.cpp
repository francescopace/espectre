/*
 * ESPectre - WiFi Lifecycle Manager Implementation
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "wifi_lifecycle.h"
#include "esphome/core/log.h"
#include "esp_wifi.h"

namespace esphome {
namespace espectre {

static const char *TAG = "WiFiLifecycle";

  
// Configure WiFi for optimal CSI capture
esp_err_t WiFiLifecycleManager::init() {
  esp_err_t ret;
  
  // Configure WiFi protocol mode (MUST be done before CSI configuration)
  // This initializes internal WiFi structures required for CSI
  // HT20 only: 802.11b/g/n for stable 64 subcarriers
  ret = esp_wifi_set_protocol(WIFI_IF_STA,
      WIFI_PROTOCOL_11B | WIFI_PROTOCOL_11G | WIFI_PROTOCOL_11N);
  if (ret != ESP_OK) {
    ESP_LOGE(TAG, "Failed to set WiFi protocol: 0x%x", ret);
    return ret;
  }
  ESP_LOGI(TAG, "WiFi protocol: 802.11b/g/n (HT20, 64 subcarriers)");

  // HT20 bandwidth for 64 subcarriers
  ret = esp_wifi_set_bandwidth(WIFI_IF_STA, WIFI_BW_HT20);
  if (ret != ESP_OK) {
    ESP_LOGW(TAG, "Failed to set bandwidth: 0x%x", ret);
    // Non-fatal: continue anyway
  }

  // IMPORTANT: Promiscuous mode MUST be called BEFORE configuring CSI
  // This initializes internal WiFi structures required for CSI, even when set to false
  ret = esp_wifi_set_promiscuous(false);
  if (ret != ESP_OK) {
    ESP_LOGE(TAG, "Failed to set promiscuous mode: 0x%x", ret);
    return ret;
  }

  return ESP_OK;
}

esp_err_t WiFiLifecycleManager::register_handlers(wifi_connected_callback_t connected_cb,
                                                  wifi_disconnected_callback_t disconnected_cb) {
  connected_callback_ = connected_cb;
  disconnected_callback_ = disconnected_cb;
  
  // Register WiFi connected event (IP_EVENT_STA_GOT_IP)
  esp_err_t err = esp_event_handler_instance_register(
      IP_EVENT,
      IP_EVENT_STA_GOT_IP,
      &WiFiLifecycleManager::ip_event_handler_,
      this,
      &connected_instance_
  );
  
  if (err != ESP_OK) {
    ESP_LOGE(TAG, "Failed to register connected handler: %s", esp_err_to_name(err));
    return err;
  }
  
  // Register WiFi disconnected event
  err = esp_event_handler_instance_register(
      WIFI_EVENT,
      WIFI_EVENT_STA_DISCONNECTED,
      &WiFiLifecycleManager::wifi_event_handler_,
      this,
      &disconnected_instance_
  );
  
  if (err != ESP_OK) {
    ESP_LOGE(TAG, "Failed to register disconnected handler: %s", esp_err_to_name(err));
    // Cleanup connected handler
    if (connected_instance_) {
      esp_event_handler_instance_unregister(IP_EVENT, IP_EVENT_STA_GOT_IP, connected_instance_);
      connected_instance_ = nullptr;
    }
    return err;
  }
  
  ESP_LOGI(TAG, "WiFi event handlers registered");
  return ESP_OK;
}

void WiFiLifecycleManager::unregister_handlers() {
  if (connected_instance_) {
    esp_event_handler_instance_unregister(IP_EVENT, IP_EVENT_STA_GOT_IP, connected_instance_);
    connected_instance_ = nullptr;
  }
  
  if (disconnected_instance_) {
    esp_event_handler_instance_unregister(WIFI_EVENT, WIFI_EVENT_STA_DISCONNECTED, disconnected_instance_);
    disconnected_instance_ = nullptr;
  }
  
  ESP_LOGI(TAG, "WiFi event handlers unregistered");
}

void WiFiLifecycleManager::ip_event_handler_(void* arg, esp_event_base_t event_base,
                                             int32_t event_id, void* event_data) {
  (void)event_base;
  (void)event_data;
  
  WiFiLifecycleManager* manager = static_cast<WiFiLifecycleManager*>(arg);
  
  // WiFi connected (got IP address)
  if (event_id == IP_EVENT_STA_GOT_IP) {
    ESP_LOGD(TAG, "WiFi connected");
    
    // Log current WiFi parameters for debugging
    bool promiscuous = false;
    esp_wifi_get_promiscuous(&promiscuous);
    ESP_LOGD(TAG, "📡 WiFi Promiscuous mode: %s", promiscuous ? "ENABLED" : "DISABLED");
    
    wifi_ps_type_t ps_type;
    esp_wifi_get_ps(&ps_type);
    const char* ps_str = (ps_type == WIFI_PS_NONE) ? "NONE" : 
                         (ps_type == WIFI_PS_MIN_MODEM) ? "MIN_MODEM" : "MAX_MODEM";
                         ESP_LOGD(TAG, "📡 WiFi Power Save: %s", ps_str);
    
    uint8_t protocol = 0;
    esp_wifi_get_protocol(WIFI_IF_STA, &protocol);
    ESP_LOGD(TAG, "📡 WiFi Protocol: 0x%02X (802.11b=%d, 802.11g=%d, 802.11n=%d)", 
             protocol,
             (protocol & WIFI_PROTOCOL_11B) ? 1 : 0,
             (protocol & WIFI_PROTOCOL_11G) ? 1 : 0,
             (protocol & WIFI_PROTOCOL_11N) ? 1 : 0);
    
    wifi_bandwidth_t bw;
    esp_wifi_get_bandwidth(WIFI_IF_STA, &bw);
    ESP_LOGD(TAG, "📡 WiFi Bandwidth: %s", (bw == WIFI_BW_HT20) ? "HT20" : "HT40");
    
    if (manager->connected_callback_) {
      manager->connected_callback_();
    }
  }
}

void WiFiLifecycleManager::wifi_event_handler_(void* arg, esp_event_base_t event_base,
                                               int32_t event_id, void* event_data) {
  
  WiFiLifecycleManager* manager = static_cast<WiFiLifecycleManager*>(arg);
  
  // WiFi disconnected
  if (event_id == WIFI_EVENT_STA_DISCONNECTED) {
    ESP_LOGW(TAG, "WiFi disconnected");
    if (manager->disconnected_callback_) {
      manager->disconnected_callback_();
    }
  }
}

}  // namespace espectre
}  // namespace esphome
