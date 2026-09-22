// SPDX-License-Identifier: GPL-3.0-only
// Commercial licensing available under separate agreement; see LICENSING.md.

#ifndef NO_QSTR

#include "native_wifi.h"
#include "native_log_sink.h"
#include "runtime/esp_idf/network_traffic.h"
#include "esp_event.h"

#include <atomic>

namespace {
esp_event_handler_instance_t rate_handler = nullptr;
std::atomic<esp_err_t> association_rate_result{ESP_OK};

void apply_associated_tx_rate(void *, esp_event_base_t, int32_t, void *) {
  association_rate_result.store(espectre::apply_station_tx_rate());
}
}  // namespace

extern "C" esp_err_t espectre_native_wifi_prepare_tx_rate(void) {
  espectre_native_ensure_log_sink();
  association_rate_result.store(ESP_OK);
  if (rate_handler != nullptr) return ESP_OK;
  // Keep this station policy active across generator stops and Wi-Fi resets.
  // Apply each AP's policy before DHCP, including transparent reassociation.
  return esp_event_handler_instance_register(
      WIFI_EVENT, WIFI_EVENT_STA_CONNECTED, apply_associated_tx_rate, nullptr, &rate_handler);
}

extern "C" esp_err_t espectre_native_wifi_apply_tx_rate(void) {
  espectre_native_ensure_log_sink();
  const esp_err_t result = association_rate_result.load();
  if (result != ESP_OK) return result;
  return espectre::apply_station_tx_rate();
}

extern "C" void espectre_native_wifi_traffic_totals(uint32_t *tx, uint32_t *rx) {
  const espectre::NetworkTrafficSnapshot traffic = espectre::read_network_traffic();
  *tx = traffic.tx_packets;
  *rx = traffic.rx_packets;
}

#endif
