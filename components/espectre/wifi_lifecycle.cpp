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

namespace {

// HT20-only CSI policy on 2.4 GHz:
// - Prefer 11n-only for deterministic HT20 behavior when supported.
// - Some targets/IDF builds reject 11n-only with ESP_ERR_INVALID_ARG; in that
//   case we fallback to b/g/n to keep the component operational.
constexpr uint16_t WIFI_PROTOCOL_CSI_2G_PREFERRED = WIFI_PROTOCOL_11N;
constexpr uint16_t WIFI_PROTOCOL_CSI_2G_FALLBACK = WIFI_PROTOCOL_11B | WIFI_PROTOCOL_11G | WIFI_PROTOCOL_11N;
constexpr wifi_bandwidth_t WIFI_BANDWIDTH_CSI = WIFI_BW_HT20;

const char *bandwidth_to_str_(wifi_bandwidth_t bw) {
  switch (bw) {
    case WIFI_BW_HT20:
      return "HT20";
    case WIFI_BW_HT40:
      return "HT40";
#ifdef WIFI_BW80
    case WIFI_BW80:
      return "BW80";
#endif
#ifdef WIFI_BW160
    case WIFI_BW160:
      return "BW160";
#endif
#ifdef WIFI_BW80_BW80
    case WIFI_BW80_BW80:
      return "BW80+80";
#endif
    default:
      return "UNKNOWN";
  }
}

#if CONFIG_IDF_TARGET_ESP32C5 || CONFIG_IDF_TARGET_ESP32C6
esp_err_t set_wifi_protocol_for_csi_() {
  esp_err_t ret;

  wifi_protocols_t protocols{};
  protocols.ghz_2g = WIFI_PROTOCOL_CSI_2G_PREFERRED;
#if CONFIG_SOC_WIFI_SUPPORT_5G
  protocols.ghz_5g = WIFI_PROTOCOL_11N;
#ifdef WIFI_PROTOCOL_11A
  protocols.ghz_5g |= WIFI_PROTOCOL_11A;
#endif
#ifdef WIFI_PROTOCOL_11AX
  protocols.ghz_5g |= WIFI_PROTOCOL_11AX;
#endif
#ifdef WIFI_PROTOCOL_11AC
  protocols.ghz_5g |= WIFI_PROTOCOL_11AC;
#endif
#endif
  ret = esp_wifi_set_protocols(WIFI_IF_STA, &protocols);
  if (ret == ESP_OK) {
    return ESP_OK;
  }

  protocols.ghz_2g = WIFI_PROTOCOL_CSI_2G_FALLBACK;
  ret = esp_wifi_set_protocols(WIFI_IF_STA, &protocols);
  if (ret == ESP_OK) {
    ESP_LOGW(TAG, "11n-only protocol not accepted, using 11b/g/n fallback");
  }
  return ret;
}

esp_err_t get_wifi_protocol_for_log_(uint16_t *protocol) {
  if (protocol == nullptr) {
    return ESP_ERR_INVALID_ARG;
  }
  wifi_protocols_t protocols{};
  esp_err_t err = esp_wifi_get_protocols(WIFI_IF_STA, &protocols);
  if (err != ESP_OK) {
    return err;
  }
  uint8_t primary_channel = 0;
  wifi_second_chan_t second_channel = WIFI_SECOND_CHAN_NONE;
  const bool is_5g = (esp_wifi_get_channel(&primary_channel, &second_channel) == ESP_OK)
                         ? (primary_channel > 14)
                         : false;
  *protocol = is_5g ? protocols.ghz_5g : protocols.ghz_2g;
  return ESP_OK;
}

esp_err_t set_wifi_bandwidth_for_csi_() {
  wifi_bandwidths_t bandwidths{};
  bandwidths.ghz_2g = WIFI_BANDWIDTH_CSI;
#if CONFIG_SOC_WIFI_SUPPORT_5G
  bandwidths.ghz_5g = WIFI_BANDWIDTH_CSI;
#endif
  return esp_wifi_set_bandwidths(WIFI_IF_STA, &bandwidths);
}

esp_err_t get_wifi_bandwidth_for_log_(wifi_bandwidth_t *bw) {
  if (bw == nullptr) {
    return ESP_ERR_INVALID_ARG;
  }
  wifi_bandwidths_t bandwidths{};
  esp_err_t err = esp_wifi_get_bandwidths(WIFI_IF_STA, &bandwidths);
  if (err != ESP_OK) {
    return err;
  }
  uint8_t primary_channel = 0;
  wifi_second_chan_t second_channel = WIFI_SECOND_CHAN_NONE;
  const bool is_5g = (esp_wifi_get_channel(&primary_channel, &second_channel) == ESP_OK)
                         ? (primary_channel > 14)
                         : false;
  *bw = is_5g ? bandwidths.ghz_5g : bandwidths.ghz_2g;
  return ESP_OK;
}
#else
esp_err_t set_wifi_protocol_for_csi_() {
  esp_err_t ret = esp_wifi_set_protocol(WIFI_IF_STA, WIFI_PROTOCOL_CSI_2G_PREFERRED);
  if (ret == ESP_OK) {
    return ESP_OK;
  }

  ret = esp_wifi_set_protocol(WIFI_IF_STA, WIFI_PROTOCOL_CSI_2G_FALLBACK);
  if (ret == ESP_OK) {
    ESP_LOGW(TAG, "11n-only protocol not accepted, using 11b/g/n fallback");
  }
  return ret;
}

esp_err_t get_wifi_protocol_for_log_(uint16_t *protocol) {
  if (protocol == nullptr) {
    return ESP_ERR_INVALID_ARG;
  }
  uint8_t protocol_bitmap = 0;
  esp_err_t err = esp_wifi_get_protocol(WIFI_IF_STA, &protocol_bitmap);
  if (err != ESP_OK) {
    return err;
  }
  *protocol = protocol_bitmap;
  return ESP_OK;
}

esp_err_t set_wifi_bandwidth_for_csi_() {
  return esp_wifi_set_bandwidth(WIFI_IF_STA, WIFI_BANDWIDTH_CSI);
}

esp_err_t get_wifi_bandwidth_for_log_(wifi_bandwidth_t *bw) {
  return esp_wifi_get_bandwidth(WIFI_IF_STA, bw);
}
#endif

}  // namespace

  
// Configure WiFi for optimal CSI capture
esp_err_t WiFiLifecycleManager::init() {
  esp_err_t ret;
  
#if CONFIG_IDF_TARGET_ESP32C5
  // ESP32-C5 is dual-band: force 2.4 GHz for stable CSI motion sensing.
  ret = esp_wifi_set_band_mode(WIFI_BAND_MODE_2G_ONLY);
  if (ret != ESP_OK) {
    ESP_LOGW(TAG, "Failed to force 2.4 GHz band mode: 0x%x", ret);
    // Non-fatal: continue, but runtime may still associate on 5 GHz in AUTO mode.
  } else {
    ESP_LOGI(TAG, "WiFi band mode: 2.4 GHz only");
  }
#endif

  // Configure WiFi protocol mode (MUST be done before CSI configuration)
  // This initializes internal WiFi structures required for CSI
  // HT20 only: 802.11b/g/n for stable 64 subcarriers
  ret = set_wifi_protocol_for_csi_();
  if (ret != ESP_OK) {
    ESP_LOGE(TAG, "Failed to set WiFi protocol: 0x%x", ret);
    return ret;
  }
  // HT20 bandwidth for 64 subcarriers
  ret = set_wifi_bandwidth_for_csi_();
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
    ESP_LOGD(TAG, "WiFi Promiscuous mode: %s", promiscuous ? "ENABLED" : "DISABLED");
    
    wifi_ps_type_t ps_type;
    esp_err_t ps_err = esp_wifi_get_ps(&ps_type);
    if (ps_err == ESP_OK) {
      const char* ps_str = (ps_type == WIFI_PS_NONE) ? "NONE" :
                           (ps_type == WIFI_PS_MIN_MODEM) ? "MIN_MODEM" : "MAX_MODEM";
      ESP_LOGD(TAG, "WiFi Power Save: %s", ps_str);
    } else {
      ESP_LOGW(TAG, "WiFi Power Save: unavailable (%s)", esp_err_to_name(ps_err));
    }
    
    uint16_t protocol = 0;
    esp_err_t protocol_err = get_wifi_protocol_for_log_(&protocol);
    if (protocol_err == ESP_OK) {
      const int has_11b = (protocol & WIFI_PROTOCOL_11B) ? 1 : 0;
      const int has_11g = (protocol & WIFI_PROTOCOL_11G) ? 1 : 0;
      const int has_11n = (protocol & WIFI_PROTOCOL_11N) ? 1 : 0;
#ifdef WIFI_PROTOCOL_11AX
      const int has_11ax = (protocol & WIFI_PROTOCOL_11AX) ? 1 : 0;
#else
      const int has_11ax = 0;
#endif
      ESP_LOGD(TAG, "WiFi Protocol: 0x%04X (802.11b=%d, 802.11g=%d, 802.11n=%d, 802.11ax=%d)",
               protocol, has_11b, has_11g, has_11n, has_11ax);
      if ((protocol & WIFI_PROTOCOL_11N) == 0) {
        ESP_LOGW(TAG, "WiFi protocol does not include 11n support: 0x%04X", protocol);
      }
    } else {
      ESP_LOGW(TAG, "WiFi Protocol: unavailable (%s)", esp_err_to_name(protocol_err));
    }
    
    wifi_bandwidth_t bw = WIFI_BW_HT20;
    esp_err_t bw_err = get_wifi_bandwidth_for_log_(&bw);
    if (bw_err == ESP_OK) {
      ESP_LOGD(TAG, "WiFi Bandwidth: %s", bandwidth_to_str_(bw));
    } else {
      ESP_LOGW(TAG, "WiFi Bandwidth: unavailable (%s)", esp_err_to_name(bw_err));
    }
    
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
