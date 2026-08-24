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
struct mdns_search_once_s {
  int active;
};
static mdns_search_once_t g_mdns_search{1};
static mdns_ip_addr_t g_delegated_address{};
static mdns_host_item_t g_delegated_host{};
static mdns_out_answer_t g_private_answer{};
static mdns_tx_packet_t g_private_packet{};

void mdns_mock_reset(void) {
  g_mdns_mock = {};
  g_mdns_mock.init_result = ESP_OK;
  g_mdns_mock.hostname_set_result = ESP_OK;
  g_mdns_mock.instance_name_set_result = ESP_OK;
  g_mdns_mock.service_add_result = ESP_OK;
  g_mdns_mock.service_txt_set_result = ESP_OK;
  g_mdns_mock.service_remove_result = ESP_OK;
  g_mdns_mock.netif_action_result = ESP_OK;
  g_mdns_mock.delegate_add_result = ESP_OK;
  g_mdns_mock.delegate_set_address_result = ESP_OK;
  g_mdns_mock.delegate_remove_result = ESP_OK;
  g_mdns_mock.async_new_succeeds = true;
  g_mdns_mock.async_delete_result = ESP_OK;
  g_mdns_mock.last_netif_action = -1;
  g_mdns_mock.private_interface_ready[0] = true;
  g_mdns_mock.private_alloc_succeeds = true;
  g_mdns_mock.private_create_answer_succeeds = true;
  g_delegated_address = {};
  g_delegated_host = {};
  g_private_answer = {};
  g_private_packet = {};
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

mdns_search_once_t *mdns_query_async_new(const char *name,
                                         const char *service_type,
                                         const char *proto,
                                         uint16_t type,
                                         uint32_t timeout,
                                         size_t max_results,
                                         mdns_query_notify_t notifier) {
  (void)name;
  (void)notifier;
  if (service_type != nullptr) {
    strncpy(g_mdns_mock.service_type, service_type, sizeof(g_mdns_mock.service_type) - 1U);
  }
  if (proto != nullptr) {
    strncpy(g_mdns_mock.service_proto, proto, sizeof(g_mdns_mock.service_proto) - 1U);
  }
  g_mdns_mock.async_new_call_count++;
  g_mdns_mock.last_query_type = type;
  g_mdns_mock.last_query_timeout_ms = timeout;
  g_mdns_mock.last_query_max_results = max_results;
  return g_mdns_mock.async_new_succeeds ? &g_mdns_search : nullptr;
}

bool mdns_query_async_get_results(mdns_search_once_t *search,
                                  uint32_t timeout,
                                  mdns_result_t **results,
                                  uint8_t *num_results) {
  (void)search;
  g_mdns_mock.async_get_results_call_count++;
  g_mdns_mock.last_get_results_timeout_ms = timeout;
  if (!g_mdns_mock.async_get_results_finished) {
    return false;
  }
  if (results != nullptr) {
    *results = g_mdns_mock.async_results;
  }
  if (num_results != nullptr) {
    *num_results = g_mdns_mock.async_results == nullptr ? 0U : 1U;
  }
  return true;
}

esp_err_t mdns_query_async_delete(mdns_search_once_t *search) {
  (void)search;
  g_mdns_mock.async_delete_call_count++;
  return g_mdns_mock.async_delete_result;
}

void mdns_query_results_free(mdns_result_t *results) {
  g_mdns_mock.query_results_free_call_count++;
  g_mdns_mock.last_freed_results = results;
}

esp_err_t mdns_delegate_hostname_add(const char *hostname, const mdns_ip_addr_t *address_list) {
  g_mdns_mock.delegate_add_call_count++;
  if (hostname != nullptr) {
    strncpy(g_mdns_mock.delegated_hostname,
            hostname,
            sizeof(g_mdns_mock.delegated_hostname) - 1U);
  }
  if (g_mdns_mock.delegate_add_result != ESP_OK) {
    return g_mdns_mock.delegate_add_result;
  }
  if (address_list != nullptr) {
    g_delegated_address = *address_list;
    g_delegated_address.next = nullptr;
    g_mdns_mock.delegated_ipv4_address = address_list->addr.u_addr.ip4.addr;
  }
  g_delegated_host.hostname = g_mdns_mock.delegated_hostname;
  g_delegated_host.address_list = &g_delegated_address;
  g_delegated_host.next = nullptr;
  return ESP_OK;
}

esp_err_t mdns_delegate_hostname_set_address(const char *hostname,
                                             const mdns_ip_addr_t *address_list) {
  g_mdns_mock.delegate_set_address_call_count++;
  if (hostname != nullptr) {
    strncpy(g_mdns_mock.delegated_hostname,
            hostname,
            sizeof(g_mdns_mock.delegated_hostname) - 1U);
  }
  if (g_mdns_mock.delegate_set_address_result != ESP_OK) {
    return g_mdns_mock.delegate_set_address_result;
  }
  if (address_list != nullptr) {
    g_delegated_address = *address_list;
    g_delegated_address.next = nullptr;
    g_mdns_mock.delegated_ipv4_address = address_list->addr.u_addr.ip4.addr;
  }
  return ESP_OK;
}

esp_err_t mdns_delegate_hostname_remove(const char *hostname) {
  g_mdns_mock.delegate_remove_call_count++;
  if (hostname != nullptr) {
    strncpy(g_mdns_mock.delegated_hostname,
            hostname,
            sizeof(g_mdns_mock.delegated_hostname) - 1U);
  }
  if (g_mdns_mock.delegate_remove_result == ESP_OK ||
      g_mdns_mock.delegate_remove_result == ESP_ERR_NOT_FOUND) {
    g_delegated_host = {};
  }
  return g_mdns_mock.delegate_remove_result;
}

mdns_host_item_t *mdns_priv_get_hosts(void) {
  return g_delegated_host.hostname != nullptr ? &g_delegated_host : nullptr;
}

bool mdns_priv_if_ready(mdns_if_t tcpip_if, mdns_ip_protocol_t ip_protocol) {
  return ip_protocol == MDNS_IP_PROTOCOL_V4 && tcpip_if < MDNS_MAX_INTERFACES &&
         g_mdns_mock.private_interface_ready[tcpip_if];
}

mdns_tx_packet_t *mdns_priv_alloc_packet(mdns_if_t tcpip_if,
                                         mdns_ip_protocol_t ip_protocol) {
  g_mdns_mock.private_alloc_call_count++;
  if (!g_mdns_mock.private_alloc_succeeds) {
    return nullptr;
  }
  g_private_packet = {};
  g_private_packet.tcpip_if = tcpip_if;
  g_private_packet.ip_protocol = ip_protocol;
  return &g_private_packet;
}

bool mdns_priv_create_answer(mdns_out_answer_t **destination,
                             uint16_t type,
                             void *service,
                             mdns_host_item_t *host,
                             bool flush,
                             bool bye) {
  g_mdns_mock.private_create_answer_call_count++;
  if (!g_mdns_mock.private_create_answer_succeeds || destination == nullptr) {
    return false;
  }
  g_private_answer = {};
  g_private_answer.type = type;
  g_private_answer.service = service;
  g_private_answer.host = host;
  g_private_answer.flush = flush;
  g_private_answer.bye = bye;
  *destination = &g_private_answer;
  return true;
}

void mdns_priv_dispatch_tx_packet(mdns_tx_packet_t *packet) {
  g_mdns_mock.private_dispatch_call_count++;
  if (packet == nullptr) {
    return;
  }
  g_mdns_mock.last_private_packet_flags = packet->flags;
  if (packet->answers != nullptr) {
    g_mdns_mock.last_private_answer_type = packet->answers->type;
    g_mdns_mock.last_private_answer_flush = packet->answers->flush;
    g_mdns_mock.last_private_answer_goodbye = packet->answers->bye;
    if (packet->answers->bye) {
      g_mdns_mock.private_goodbye_count++;
    } else {
      g_mdns_mock.private_announce_count++;
    }
  }
}

void mdns_priv_free_tx_packet(mdns_tx_packet_t *packet) {
  (void)packet;
  g_mdns_mock.private_free_packet_call_count++;
}

size_t __real_mdns_priv_if_write(mdns_if_t tcpip_if,
                                 mdns_ip_protocol_t ip_protocol,
                                 const esp_ip_addr_t *ip,
                                 uint16_t port,
                                 uint8_t *data,
                                 size_t len) {
  (void)tcpip_if;
  (void)ip_protocol;
  (void)ip;
  (void)port;
  g_mdns_mock.real_write_call_count++;
  g_mdns_mock.last_write_len = len < sizeof(g_mdns_mock.last_write_packet)
                                   ? len
                                   : sizeof(g_mdns_mock.last_write_packet);
  if (data != nullptr && g_mdns_mock.last_write_len > 0U) {
    memcpy(g_mdns_mock.last_write_packet, data, g_mdns_mock.last_write_len);
  }
  return len;
}

namespace {
struct MdnsMockResetInitializer {
  MdnsMockResetInitializer() { mdns_mock_reset(); }
} g_mdns_mock_reset_initializer;
}  // namespace
