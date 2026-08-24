/*
 * ESPectre - Mock mdns_mock.cpp
 *
 * Host-side mock of mdns API for native C++ tests.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "mdns.h"

#include <string.h>

mdns_mock_state_t g_mdns_mock{};

void mdns_mock_reset(void) {
  g_mdns_mock = {};
  g_mdns_mock.init_result = ESP_OK;
  g_mdns_mock.hostname_set_result = ESP_OK;
  g_mdns_mock.instance_name_set_result = ESP_OK;
  g_mdns_mock.service_add_result = ESP_OK;
  g_mdns_mock.service_txt_set_result = ESP_OK;
  g_mdns_mock.service_remove_result = ESP_OK;
  g_mdns_mock.netif_action_result = ESP_OK;
  g_mdns_mock.last_netif_action = -1;
}

esp_err_t mdns_init(void) {
  g_mdns_mock.init_call_count++;
  return g_mdns_mock.init_result;
}

esp_err_t mdns_hostname_set(const char *hostname) {
  if (hostname != nullptr) {
    strncpy(g_mdns_mock.hostname, hostname, sizeof(g_mdns_mock.hostname) - 1U);
  }
  g_mdns_mock.hostname_set_call_count++;
  return g_mdns_mock.hostname_set_result;
}

esp_err_t mdns_instance_name_set(const char *instance_name) {
  if (instance_name != nullptr) {
    strncpy(g_mdns_mock.instance_name, instance_name, sizeof(g_mdns_mock.instance_name) - 1U);
  }
  g_mdns_mock.instance_name_set_call_count++;
  return g_mdns_mock.instance_name_set_result;
}

esp_err_t mdns_service_add(
    const char *instance_name,
    const char *service_type,
    const char *proto,
    uint16_t port,
    const mdns_txt_item_t *txt,
    size_t num_items) {
  if (instance_name != nullptr) {
    strncpy(g_mdns_mock.instance_name, instance_name, sizeof(g_mdns_mock.instance_name) - 1U);
  }
  if (service_type != nullptr) {
    strncpy(g_mdns_mock.service_type, service_type, sizeof(g_mdns_mock.service_type) - 1U);
  }
  if (proto != nullptr) {
    strncpy(g_mdns_mock.service_proto, proto, sizeof(g_mdns_mock.service_proto) - 1U);
  }
  g_mdns_mock.service_port = port;
  (void)txt;
  (void)num_items;
  g_mdns_mock.service_add_call_count++;
  return g_mdns_mock.service_add_result;
}

esp_err_t mdns_service_txt_set(
    const char *service_type, const char *proto, const mdns_txt_item_t *txt, size_t num_items) {
  if (service_type != nullptr) {
    strncpy(g_mdns_mock.service_type, service_type, sizeof(g_mdns_mock.service_type) - 1U);
  }
  if (proto != nullptr) {
    strncpy(g_mdns_mock.service_proto, proto, sizeof(g_mdns_mock.service_proto) - 1U);
  }
  g_mdns_mock.txt_count = num_items < 12U ? num_items : 12U;
  for (size_t index = 0U; index < g_mdns_mock.txt_count; ++index) {
    strncpy(g_mdns_mock.txt_keys[index], txt[index].key, sizeof(g_mdns_mock.txt_keys[index]) - 1U);
    strncpy(g_mdns_mock.txt_values[index], txt[index].value, sizeof(g_mdns_mock.txt_values[index]) - 1U);
  }
  g_mdns_mock.service_txt_set_call_count++;
  return g_mdns_mock.service_txt_set_result;
}

esp_err_t mdns_service_remove(const char *service_type, const char *proto) {
  (void)service_type;
  (void)proto;
  g_mdns_mock.service_remove_call_count++;
  return g_mdns_mock.service_remove_result;
}

void mdns_free(void) { g_mdns_mock.free_call_count++; }

esp_err_t mdns_netif_action(esp_netif_t *esp_netif, mdns_event_actions_t action) {
  (void)esp_netif;
  g_mdns_mock.netif_action_call_count++;
  g_mdns_mock.last_netif_action = (int)action;
  return g_mdns_mock.netif_action_result;
}

namespace {
struct MdnsMockResetInitializer {
  MdnsMockResetInitializer() { mdns_mock_reset(); }
} g_mdns_mock_reset_initializer;
}  // namespace
