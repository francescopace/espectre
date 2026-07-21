/*
 * ESPectre - Wi-Fi Lifecycle Manager
 *
 * Controls STA lifecycle and HT20 CSI compatibility for sensing runtimes.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */
#include "wifi_lifecycle.h"
#include "espectre_log.h"
#include "esp_wifi.h"
#include "sdkconfig.h"

#if defined(CONFIG_ESP_COEX_SW_COEXIST_ENABLE) && __has_include("esp_coexist.h")
#include "esp_coexist.h"
#define ESPECTRE_HAVE_ESP_COEXIST 1
#endif

namespace espectre {

static const char *WIFI_LIFECYCLE_TAG = "WiFiLifecycle";

namespace {

// ESP-IDF exposes 802.11n on 2.4 GHz through the supported b/g/n protocol
// combination. WIFI_PROTOCOL_11N alone is not a valid station mode on the
// published ESPectre targets, so we prefer it first and fall back to b/g/n.
constexpr uint16_t WIFI_PROTOCOL_CSI_2G = WIFI_PROTOCOL_11B | WIFI_PROTOCOL_11G | WIFI_PROTOCOL_11N;
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
  uint8_t current_protocol = 0U;
  if (esp_wifi_get_protocol(WIFI_IF_STA, &current_protocol) == ESP_OK && current_protocol == WIFI_PROTOCOL_CSI_2G) {
    return ESP_OK;
  }
  esp_err_t ret = esp_wifi_set_protocol(WIFI_IF_STA, WIFI_PROTOCOL_11N);
  if (ret == ESP_OK) {
    return ESP_OK;
  }

  ret = esp_wifi_set_protocol(WIFI_IF_STA, WIFI_PROTOCOL_CSI_2G);
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
  wifi_bandwidth_t current_bandwidth = WIFI_BW_HT20;
  if (esp_wifi_get_bandwidth(WIFI_IF_STA, &current_bandwidth) == ESP_OK &&
      current_bandwidth == WIFI_BANDWIDTH_CSI) {
    return ESP_OK;
  }
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
  ESP_LOGI(WIFI_LIFECYCLE_TAG, "WiFi protocol: 802.11b/g/n with HT20 (802.11ax disabled)");
  // HT20 bandwidth for 64 subcarriers
  ret = set_wifi_bandwidth_for_csi_();
  if (ret != ESP_OK) {
    ESP_LOGW(WIFI_LIFECYCLE_TAG, "Failed to set bandwidth: 0x%x", ret);
    // Non-fatal: continue anyway
  }

  return ESP_OK;
}

esp_err_t WiFiLifecycleManager::apply_started_csi_policy() {
#ifdef ESPECTRE_HAVE_ESP_COEXIST
  const esp_err_t coex_err = esp_coex_preference_set(ESP_COEX_PREFER_WIFI);
  if (coex_err != ESP_OK) {
    ESP_LOGW(WIFI_LIFECYCLE_TAG, "Failed to bias Wi-Fi/BT coexistence toward Wi-Fi: %s",
             esp_err_to_name(coex_err));
  }
#endif

  const esp_err_t policy_err = apply_csi_wifi_policy();
  if (policy_err != ESP_OK) {
    ESP_LOGW(WIFI_LIFECYCLE_TAG, "Failed to apply started Wi-Fi CSI policy: %s",
             esp_err_to_name(policy_err));
    return policy_err;
  }
  log_csi_runtime_state(WIFI_LIFECYCLE_TAG);
  ESP_LOGI(WIFI_LIFECYCLE_TAG, "Started Wi-Fi CSI policy applied");
  return ESP_OK;
}

// Configure WiFi for optimal CSI capture
esp_err_t WiFiLifecycleManager::init() {
  if (ready_) {
    return ESP_OK;
  }

  ESP_LOGI(WIFI_LIFECYCLE_TAG, "Initializing WiFi CSI lifecycle");
  const esp_err_t policy_err = started_policy_err_.load(std::memory_order_relaxed);
  if (policy_err != ESP_OK) {
    ESP_LOGE(WIFI_LIFECYCLE_TAG, "Wi-Fi CSI policy was not applied at STA start: %s",
             esp_err_to_name(policy_err));
    return policy_err;
  }

  ESP_LOGI(WIFI_LIFECYCLE_TAG, "WiFi CSI lifecycle ready");
  log_csi_runtime_state(WIFI_LIFECYCLE_TAG);
  ready_ = true;

  return ESP_OK;
}

esp_err_t WiFiLifecycleManager::register_handlers(wifi_connected_callback_t connected_cb,
                                                  wifi_disconnected_callback_t disconnected_cb) {
  connected_callback_ = connected_cb;
  disconnected_callback_ = disconnected_cb;
  
  started_policy_err_.store(ESP_ERR_INVALID_STATE, std::memory_order_relaxed);
  esp_err_t err = esp_event_handler_instance_register(
      WIFI_EVENT,
      WIFI_EVENT_STA_START,
      &WiFiLifecycleManager::wifi_event_handler_,
      this,
      &started_instance_
  );
  if (err != ESP_OK) {
    ESP_LOGE(WIFI_LIFECYCLE_TAG, "Failed to register started handler: %s", esp_err_to_name(err));
    return err;
  }

  // Register WiFi connected event (IP_EVENT_STA_GOT_IP)
  err = esp_event_handler_instance_register(
      IP_EVENT,
      IP_EVENT_STA_GOT_IP,
      &WiFiLifecycleManager::ip_event_handler_,
      this,
      &connected_instance_
  );
  
  if (err != ESP_OK) {
    ESP_LOGE(WIFI_LIFECYCLE_TAG, "Failed to register connected handler: %s", esp_err_to_name(err));
    esp_event_handler_instance_unregister(WIFI_EVENT, WIFI_EVENT_STA_START, started_instance_);
    started_instance_ = nullptr;
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
    if (started_instance_) {
      esp_event_handler_instance_unregister(WIFI_EVENT, WIFI_EVENT_STA_START, started_instance_);
      started_instance_ = nullptr;
    }
    return err;
  }
  
  ESP_LOGI(WIFI_LIFECYCLE_TAG, "WiFi event handlers registered");
  return ESP_OK;
}

void WiFiLifecycleManager::unregister_handlers() {
  if (started_instance_) {
    esp_event_handler_instance_unregister(WIFI_EVENT, WIFI_EVENT_STA_START, started_instance_);
    started_instance_ = nullptr;
  }
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
  started_policy_err_.store(ESP_ERR_INVALID_STATE, std::memory_order_relaxed);
  ready_ = false;
  ESP_LOGI(WIFI_LIFECYCLE_TAG, "WiFi event handlers unregistered");
}

esp_err_t WiFiLifecycleManager::process_pending_events() {
  if (disconnected_event_.take()) {
    ready_ = false;
    ESP_LOGW(WIFI_LIFECYCLE_TAG, "WiFi disconnected");
    if (disconnected_callback_) {
      disconnected_callback_();
    }
  }

  esp_netif_ip_info_t ip_info{};
  if (connected_event_.take(ip_info)) {
    ESP_LOGD(WIFI_LIFECYCLE_TAG, "WiFi connected");
    const esp_err_t err = init();
    if (err != ESP_OK) {
      return err;
    }
    if (connected_callback_) {
      connected_callback_(ip_info);
    }
  }
  return ESP_OK;
}

void WiFiLifecycleManager::log_csi_runtime_state(const char *tag) {
  const char *log_tag = tag != nullptr ? tag : WIFI_LIFECYCLE_TAG;
  bool promiscuous = false;
  esp_wifi_get_promiscuous(&promiscuous);
  ESP_LOGD(log_tag, "WiFi Promiscuous mode: %s", promiscuous ? "ENABLED" : "DISABLED");

  wifi_ps_type_t ps_type;
  esp_err_t ps_err = esp_wifi_get_ps(&ps_type);
  if (ps_err == ESP_OK) {
    const char* ps_str = (ps_type == WIFI_PS_NONE) ? "NONE" :
                         (ps_type == WIFI_PS_MIN_MODEM) ? "MIN_MODEM" : "MAX_MODEM";
    ESP_LOGD(log_tag, "WiFi Power Save: %s", ps_str);
  } else {
    ESP_LOGW(log_tag, "WiFi Power Save: unavailable (%s)", esp_err_to_name(ps_err));
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
    ESP_LOGD(log_tag, "WiFi Protocol: 0x%04X (802.11b=%d, 802.11g=%d, 802.11n=%d, 802.11ax=%d)",
             protocol, has_11b, has_11g, has_11n, has_11ax);
    if ((protocol & WIFI_PROTOCOL_11N) == 0) {
      ESP_LOGW(log_tag, "WiFi protocol does not include 11n support: 0x%04X", protocol);
    }
  } else {
    ESP_LOGW(log_tag, "WiFi Protocol: unavailable (%s)", esp_err_to_name(protocol_err));
  }

  wifi_bandwidth_t bw = WIFI_BW_HT20;
  esp_err_t bw_err = get_wifi_bandwidth_for_log_(&bw);
  if (bw_err == ESP_OK) {
    ESP_LOGD(log_tag, "WiFi Bandwidth: %s", bandwidth_to_str_(bw));
  } else {
    ESP_LOGW(log_tag, "WiFi Bandwidth: unavailable (%s)", esp_err_to_name(bw_err));
  }
}

// The event handlers run on the default event loop task (sys_evt), so they
// must remain short and non-blocking. STA start applies the radio policy before
// association; IP and disconnect events are drained from the runtime loop.
void WiFiLifecycleManager::ip_event_handler_(void* arg, esp_event_base_t event_base,
                                             int32_t event_id, void* event_data) {
  (void)event_base;
  WiFiLifecycleManager* manager = static_cast<WiFiLifecycleManager*>(arg);

  if (manager != nullptr && event_id == IP_EVENT_STA_GOT_IP && event_data != nullptr) {
    const auto *event = static_cast<const ip_event_got_ip_t *>(event_data);
    manager->connected_event_.post(event->ip_info);
  }
}

void WiFiLifecycleManager::wifi_event_handler_(void* arg, esp_event_base_t event_base,
                                               int32_t event_id, void* event_data) {

  (void)event_base;
  (void)event_data;

  WiFiLifecycleManager* manager = static_cast<WiFiLifecycleManager*>(arg);

  if (manager == nullptr) {
    return;
  }
  if (event_id == WIFI_EVENT_STA_START) {
    manager->started_policy_err_.store(apply_started_csi_policy(), std::memory_order_relaxed);
  } else if (event_id == WIFI_EVENT_STA_DISCONNECTED) {
    manager->disconnected_event_.post();
  }
}

}  // namespace espectre
