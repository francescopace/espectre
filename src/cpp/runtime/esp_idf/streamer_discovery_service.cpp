/*
 * ESPectre - Streamer Discovery Service
 *
 * Advertises streamer pacing endpoints through mDNS/DNS-SD.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "streamer_discovery_service.h"

#include <array>
#include <cstdio>

#include "espectre_log.h"
#include "espectre_protocol.h"
#include "esp_err.h"
#include "esp_netif.h"
#include "mdns.h"

namespace espectre {

namespace {

static const char *const TAG = "espectre.stream.mdns";
static constexpr const char *kStaNetifKey = "WIFI_STA_DEF";
static constexpr const char *kServiceType = "_espectre-streamer";
static constexpr const char *kServiceProto = "_udp";

std::string build_device_suffix(uint64_t device_id) {
  char buffer[17];
  std::snprintf(buffer, sizeof(buffer), "%016llx", static_cast<unsigned long long>(device_id));
  return std::string(buffer);
}

esp_netif_t *get_sta_netif() { return esp_netif_get_handle_from_ifkey(kStaNetifKey); }

}  // namespace

bool StreamerDiscoveryService::setup(const StreamerDiscoveryServiceConfig &config) {
  config_ = config;
  device_id_text_ = format_espectre_device_id(config_.device_id);
  const std::string suffix = build_device_suffix(config_.device_id);
  hostname_ = "espectre-streamer-" + suffix;
  instance_name_ = "ESPectre Streamer " + device_id_text_;

  if (!initialize_mdns_()) {
    return false;
  }
  if (!configure_identity_()) {
    return false;
  }
  if (!configure_service_()) {
    return false;
  }
  return set_service_txt_();
}

void StreamerDiscoveryService::on_wifi_connected(const StandaloneWifiInfo &wifi_info) {
  (void)wifi_info;
  if (!mdns_initialized_) {
    return;
  }
  if (!set_service_txt_()) {
    return;
  }
  if (service_enabled_) {
    apply_netif_action_(MDNS_EVENT_ANNOUNCE_IP4);
    return;
  }
  apply_netif_action_(MDNS_EVENT_ENABLE_IP4);
}

void StreamerDiscoveryService::on_wifi_disconnected() {
  if (!mdns_initialized_ || !service_enabled_) {
    return;
  }
  apply_netif_action_(MDNS_EVENT_DISABLE_IP4);
}

void StreamerDiscoveryService::shutdown() {
  on_wifi_disconnected();
  if (service_added_) {
    const esp_err_t remove_err = mdns_service_remove(kServiceType, kServiceProto);
    if (remove_err != ESP_OK && remove_err != ESP_ERR_NOT_FOUND) {
      ESP_LOGW(TAG, "mdns_service_remove failed: %s", esp_err_to_name(remove_err));
    }
    service_added_ = false;
  }
  if (mdns_initialized_) {
    mdns_free();
    mdns_initialized_ = false;
  }
  service_enabled_ = false;
}

bool StreamerDiscoveryService::initialize_mdns_() {
  if (mdns_initialized_) {
    return true;
  }
  const esp_err_t err = mdns_init();
  if (err != ESP_OK && err != ESP_ERR_INVALID_STATE) {
    ESP_LOGE(TAG, "mdns_init failed: %s", esp_err_to_name(err));
    return false;
  }
  mdns_initialized_ = true;
  return true;
}

bool StreamerDiscoveryService::configure_identity_() {
  const esp_err_t hostname_err = mdns_hostname_set(hostname_.c_str());
  if (hostname_err != ESP_OK) {
    ESP_LOGE(TAG, "mdns_hostname_set failed: %s", esp_err_to_name(hostname_err));
    return false;
  }
  const esp_err_t instance_err = mdns_instance_name_set(instance_name_.c_str());
  if (instance_err != ESP_OK) {
    ESP_LOGE(TAG, "mdns_instance_name_set failed: %s", esp_err_to_name(instance_err));
    return false;
  }
  return true;
}

bool StreamerDiscoveryService::configure_service_() {
  if (service_added_) {
    return true;
  }
  const esp_err_t err = mdns_service_add(instance_name_.c_str(), kServiceType, kServiceProto, config_.traffic_port, nullptr, 0);
  if (err != ESP_OK) {
    ESP_LOGE(TAG, "mdns_service_add failed: %s", esp_err_to_name(err));
    return false;
  }
  service_added_ = true;
  return true;
}

bool StreamerDiscoveryService::set_service_txt_() {
  std::array<char, 8> traffic_port_text{};
  std::array<char, 8> collector_port_text{};
  std::snprintf(traffic_port_text.data(), traffic_port_text.size(), "%u", static_cast<unsigned>(config_.traffic_port));
  std::snprintf(
      collector_port_text.data(), collector_port_text.size(), "%u", static_cast<unsigned>(config_.collector_port));
  mdns_txt_item_t txt[] = {
      {"device_id", device_id_text_.c_str()},
      {"chip", config_.chip.c_str()},
      {"traffic_port", traffic_port_text.data()},
      {"collector_port", collector_port_text.data()},
  };
  const esp_err_t err = mdns_service_txt_set(kServiceType, kServiceProto, txt, sizeof(txt) / sizeof(txt[0]));
  if (err != ESP_OK) {
    ESP_LOGE(TAG, "mdns_service_txt_set failed: %s", esp_err_to_name(err));
    return false;
  }
  return true;
}

void StreamerDiscoveryService::apply_netif_action_(int action) {
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
