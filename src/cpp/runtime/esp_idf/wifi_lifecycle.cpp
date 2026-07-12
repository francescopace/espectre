/*
 * ESPectre - WiFi Lifecycle Manager Implementation
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "wifi_lifecycle.h"
#include "espectre_log.h"
#include "esp_wifi.h"

namespace espectre {

static const char *WIFI_LIFECYCLE_TAG = "WiFiLifecycle";

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

esp_err_t set_wifi_protocol_for_csi_() {
  esp_err_t ret = esp_wifi_set_protocol(WIFI_IF_STA, WIFI_PROTOCOL_CSI_2G_PREFERRED);
  if (ret == ESP_OK) {
    return ESP_OK;
  }

  ret = esp_wifi_set_protocol(WIFI_IF_STA, WIFI_PROTOCOL_CSI_2G_FALLBACK);
  if (ret == ESP_OK) {
    ESP_LOGW(WIFI_LIFECYCLE_TAG, "11n-only protocol not accepted, using 11b/g/n fallback");
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

}  // namespace

esp_err_t WiFiLifecycleManager::apply_csi_wifi_policy() {
  esp_err_t ret;

#if CONFIG_IDF_TARGET_ESP32C5 || CONFIG_IDF_TARGET_ESP32C6
  // Force 2.4 GHz before setting the protocol bitmap. On HE-capable targets
  // this lets esp_wifi_set_protocol() remove 802.11ax cleanly for HT20 CSI.
  ret = esp_wifi_set_band_mode(WIFI_BAND_MODE_2G_ONLY);
  if (ret != ESP_OK) {
    ESP_LOGW(WIFI_LIFECYCLE_TAG, "Failed to force 2.4 GHz band mode: 0x%x", ret);
    // Non-fatal: continue, but runtime may still associate on 5 GHz in AUTO mode.
  } else {
    ESP_LOGI(WIFI_LIFECYCLE_TAG, "WiFi band mode: 2.4 GHz only");
  }
#endif

  // Configure WiFi protocol mode (MUST be done before CSI configuration)
  // This initializes internal WiFi structures required for CSI
  // HT20 only: 802.11b/g/n for stable 64 subcarriers
  ret = set_wifi_protocol_for_csi_();
  if (ret != ESP_OK) {
    ESP_LOGE(WIFI_LIFECYCLE_TAG, "Failed to set WiFi protocol: 0x%x", ret);
    return ret;
  }
  ESP_LOGI(WIFI_LIFECYCLE_TAG, "WiFi protocol: 802.11n HT20 target (802.11ax disabled)");
  // HT20 bandwidth for 64 subcarriers
  ret = set_wifi_bandwidth_for_csi_();
  if (ret != ESP_OK) {
    ESP_LOGW(WIFI_LIFECYCLE_TAG, "Failed to set bandwidth: 0x%x", ret);
    // Non-fatal: continue anyway
  }

  return ESP_OK;
}

// Configure WiFi for optimal CSI capture
esp_err_t WiFiLifecycleManager::init() {
  ESP_LOGI(WIFI_LIFECYCLE_TAG, "Initializing WiFi CSI lifecycle");
  esp_err_t ret = apply_csi_wifi_policy();
  if (ret != ESP_OK) {
    return ret;
  }

  // IMPORTANT: Promiscuous mode MUST be called BEFORE configuring CSI
  // This initializes internal WiFi structures required for CSI, even when set to false
  ret = esp_wifi_set_promiscuous(false);
  if (ret != ESP_OK) {
    ESP_LOGE(WIFI_LIFECYCLE_TAG, "Failed to set promiscuous mode: 0x%x", ret);
    return ret;
  }
  ESP_LOGI(WIFI_LIFECYCLE_TAG, "WiFi CSI lifecycle ready: promiscuous=disabled");

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
    ESP_LOGE(WIFI_LIFECYCLE_TAG, "Failed to register connected handler: %s", esp_err_to_name(err));
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
    ESP_LOGE(WIFI_LIFECYCLE_TAG, "Failed to register disconnected handler: %s", esp_err_to_name(err));
    // Cleanup connected handler
    if (connected_instance_) {
      esp_event_handler_instance_unregister(IP_EVENT, IP_EVENT_STA_GOT_IP, connected_instance_);
      connected_instance_ = nullptr;
    }
    return err;
  }
  
  ESP_LOGI(WIFI_LIFECYCLE_TAG, "WiFi event handlers registered");
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

  connected_event_.clear();
  disconnected_event_.clear();
  ESP_LOGI(WIFI_LIFECYCLE_TAG, "WiFi event handlers unregistered");
}

void WiFiLifecycleManager::process_pending_events() {
  if (disconnected_event_.take()) {
    ESP_LOGW(WIFI_LIFECYCLE_TAG, "WiFi disconnected");
    if (disconnected_callback_) {
      disconnected_callback_();
    }
  }

  if (connected_event_.take()) {
    ESP_LOGD(WIFI_LIFECYCLE_TAG, "WiFi connected");
    log_connect_diagnostics_();
    if (connected_callback_) {
      connected_callback_();
    }
  }
}

void WiFiLifecycleManager::log_connect_diagnostics_() {
  bool promiscuous = false;
  esp_wifi_get_promiscuous(&promiscuous);
  ESP_LOGD(WIFI_LIFECYCLE_TAG, "WiFi Promiscuous mode: %s", promiscuous ? "ENABLED" : "DISABLED");

  wifi_ps_type_t ps_type;
  esp_err_t ps_err = esp_wifi_get_ps(&ps_type);
  if (ps_err == ESP_OK) {
    const char* ps_str = (ps_type == WIFI_PS_NONE) ? "NONE" :
                         (ps_type == WIFI_PS_MIN_MODEM) ? "MIN_MODEM" : "MAX_MODEM";
    ESP_LOGD(WIFI_LIFECYCLE_TAG, "WiFi Power Save: %s", ps_str);
  } else {
    ESP_LOGW(WIFI_LIFECYCLE_TAG, "WiFi Power Save: unavailable (%s)", esp_err_to_name(ps_err));
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
    ESP_LOGD(WIFI_LIFECYCLE_TAG, "WiFi Protocol: 0x%04X (802.11b=%d, 802.11g=%d, 802.11n=%d, 802.11ax=%d)",
             protocol, has_11b, has_11g, has_11n, has_11ax);
    if ((protocol & WIFI_PROTOCOL_11N) == 0) {
      ESP_LOGW(WIFI_LIFECYCLE_TAG, "WiFi protocol does not include 11n support: 0x%04X", protocol);
    }
  } else {
    ESP_LOGW(WIFI_LIFECYCLE_TAG, "WiFi Protocol: unavailable (%s)", esp_err_to_name(protocol_err));
  }

  wifi_bandwidth_t bw = WIFI_BW_HT20;
  esp_err_t bw_err = get_wifi_bandwidth_for_log_(&bw);
  if (bw_err == ESP_OK) {
    ESP_LOGD(WIFI_LIFECYCLE_TAG, "WiFi Bandwidth: %s", bandwidth_to_str_(bw));
  } else {
    ESP_LOGW(WIFI_LIFECYCLE_TAG, "WiFi Bandwidth: unavailable (%s)", esp_err_to_name(bw_err));
  }
}

// The event handlers run on the default event loop task (sys_evt), so they
// must remain short and non-blocking. They only record the event;
// process_pending_events() does the work from the runtime loop.
void WiFiLifecycleManager::ip_event_handler_(void* arg, esp_event_base_t event_base,
                                             int32_t event_id, void* event_data) {
  (void)event_base;
  (void)event_data;

  WiFiLifecycleManager* manager = static_cast<WiFiLifecycleManager*>(arg);

  if (event_id == IP_EVENT_STA_GOT_IP) {
    manager->connected_event_.post();
  }
}

void WiFiLifecycleManager::wifi_event_handler_(void* arg, esp_event_base_t event_base,
                                               int32_t event_id, void* event_data) {

  WiFiLifecycleManager* manager = static_cast<WiFiLifecycleManager*>(arg);

  if (event_id == WIFI_EVENT_STA_DISCONNECTED) {
    manager->disconnected_event_.post();
  }
}

}  // namespace espectre
