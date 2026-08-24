/*
 * ESPectre - Shared mDNS Discovery Service
 *
 * Owns the common ESP-IDF mDNS lifecycle used by firmware frontends.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "mdns_discovery_service.h"

#include "espectre_log.h"
#include "esp_err.h"
#include "esp_netif.h"
#include "mdns.h"

namespace espectre {

namespace {

const char *const TAG = "espectre.mdns";
constexpr const char *kStaNetifKey = "WIFI_STA_DEF";

esp_netif_t *get_sta_netif() { return esp_netif_get_handle_from_ifkey(kStaNetifKey); }

bool valid_config(const MdnsDiscoveryServiceConfig &config) {
  const bool identity_valid = config.responder_mode == MdnsResponderMode::USE_EXISTING_RESPONDER ||
                              (!config.hostname.empty() && config.hostname.size() <= 63U);
  return identity_valid && !config.instance_name.empty() &&
         !config.service_type.empty() && !config.service_protocol.empty() && config.port != 0U;
}

}  // namespace

bool MdnsDiscoveryService::setup(const MdnsDiscoveryServiceConfig &config) {
  shutdown();
  if (!valid_config(config)) {
    return false;
  }
  config_ = config;

  if (config_.responder_mode == MdnsResponderMode::OWN_RESPONDER) {
    const esp_err_t init_err = mdns_init();
    if (init_err != ESP_OK && init_err != ESP_ERR_INVALID_STATE) {
      ESP_LOGE(TAG, "mdns_init failed: %s", esp_err_to_name(init_err));
      return false;
    }
    mdns_initialized_ = true;
    owns_mdns_ = init_err == ESP_OK;

    if (mdns_hostname_set(config_.hostname.c_str()) != ESP_OK ||
        mdns_instance_name_set(config_.instance_name.c_str()) != ESP_OK) {
      ESP_LOGE(TAG, "Failed to configure mDNS identity");
      shutdown();
      return false;
    }
  } else {
    mdns_initialized_ = true;
  }
  const esp_err_t service_err = mdns_service_add(config_.instance_name.c_str(),
                                                 config_.service_type.c_str(),
                                                 config_.service_protocol.c_str(),
                                                 config_.port,
                                                 nullptr,
                                                 0U);
  if (service_err != ESP_OK) {
    ESP_LOGE(TAG, "mdns_service_add failed: %s", esp_err_to_name(service_err));
    shutdown();
    return false;
  }
  service_added_ = true;
  service_enabled_ = config_.responder_mode == MdnsResponderMode::USE_EXISTING_RESPONDER;
  if (!set_service_txt_()) {
    shutdown();
    return false;
  }
  return true;
}

bool MdnsDiscoveryService::update_txt(const MdnsTxtRecords &txt_records) {
  config_.txt_records = txt_records;
  return service_added_ && set_service_txt_();
}

void MdnsDiscoveryService::on_wifi_connected() {
  if (!mdns_initialized_ || config_.responder_mode == MdnsResponderMode::USE_EXISTING_RESPONDER) {
    return;
  }
  if (service_enabled_) {
    apply_netif_action_(MDNS_EVENT_ANNOUNCE_IP4);
  } else {
    apply_netif_action_(MDNS_EVENT_ENABLE_IP4);
  }
}

void MdnsDiscoveryService::on_wifi_disconnected() {
  if (mdns_initialized_ && service_enabled_ &&
      config_.responder_mode == MdnsResponderMode::OWN_RESPONDER) {
    apply_netif_action_(MDNS_EVENT_DISABLE_IP4);
  }
}

void MdnsDiscoveryService::shutdown() {
  on_wifi_disconnected();
  if (service_added_) {
    const esp_err_t err = mdns_service_remove(config_.service_type.c_str(), config_.service_protocol.c_str());
    if (err != ESP_OK && err != ESP_ERR_NOT_FOUND) {
      ESP_LOGW(TAG, "mdns_service_remove failed: %s", esp_err_to_name(err));
    }
  }
  if (mdns_initialized_ && owns_mdns_) {
    mdns_free();
  }
  mdns_initialized_ = false;
  owns_mdns_ = false;
  service_added_ = false;
  service_enabled_ = false;
}

bool MdnsDiscoveryService::set_service_txt_() {
  std::vector<mdns_txt_item_t> records;
  records.reserve(config_.txt_records.size());
  for (const auto &record : config_.txt_records) {
    records.push_back(mdns_txt_item_t{record.first.c_str(), record.second.c_str()});
  }
  const esp_err_t err = mdns_service_txt_set(config_.service_type.c_str(),
                                             config_.service_protocol.c_str(),
                                             records.empty() ? nullptr : records.data(),
                                             records.size());
  if (err != ESP_OK) {
    ESP_LOGE(TAG, "mdns_service_txt_set failed: %s", esp_err_to_name(err));
    return false;
  }
  return true;
}

void MdnsDiscoveryService::apply_netif_action_(int action) {
  esp_netif_t *netif = get_sta_netif();
  if (netif == nullptr) {
    ESP_LOGW(TAG, "No STA netif available for mDNS update");
    return;
  }
  const esp_err_t err = mdns_netif_action(netif, static_cast<mdns_event_actions_t>(action));
  if (err != ESP_OK) {
    ESP_LOGW(TAG, "mdns_netif_action(%d) failed: %s", action, esp_err_to_name(err));
    return;
  }
  service_enabled_ = action != MDNS_EVENT_DISABLE_IP4;
}

}  // namespace espectre
