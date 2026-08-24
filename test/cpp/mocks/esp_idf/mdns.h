/*
 * ESPectre - Mock mdns.h
 *
 * Host-side mock of mdns.h for native C++ tests.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#ifndef MDNS_H
#define MDNS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "esp_err.h"
#include "esp_netif.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  MDNS_EVENT_ENABLE_IP4 = 0,
  MDNS_EVENT_DISABLE_IP4 = 1,
  MDNS_EVENT_ANNOUNCE_IP4 = 2,
} mdns_event_actions_t;

typedef struct {
  const char *key;
  const char *value;
} mdns_txt_item_t;

#define MDNS_TYPE_PTR 0x000C
#define MDNS_TYPE_A 0x0001
#define MDNS_TYPE_AAAA 0x001C
#define MDNS_NAME_BUF_LEN 65
#define MDNS_MAX_INTERFACES 2
#define MDNS_FLAGS_QR_AUTHORITATIVE 0x8400
#define ESP_IPADDR_TYPE_V4 IPADDR_TYPE_V4

typedef ip_addr_t esp_ip_addr_t;
typedef size_t mdns_if_t;
typedef enum {
  MDNS_IP_PROTOCOL_V4 = 0,
  MDNS_IP_PROTOCOL_V6 = 1,
  MDNS_IP_PROTOCOL_MAX,
} mdns_ip_protocol_t;
typedef struct mdns_search_once_s mdns_search_once_t;
typedef void (*mdns_query_notify_t)(mdns_search_once_t *search);

typedef struct mdns_ip_addr_s {
  esp_ip_addr_t addr;
  struct mdns_ip_addr_s *next;
} mdns_ip_addr_t;

typedef struct mdns_result_s {
  struct mdns_result_s *next;
  esp_netif_t *esp_netif;
  uint32_t ttl;
  int ip_protocol;
  char *instance_name;
  char *service_type;
  char *proto;
  char *hostname;
  uint16_t port;
  mdns_txt_item_t *txt;
  uint8_t *txt_value_len;
  size_t txt_count;
  mdns_ip_addr_t *addr;
} mdns_result_t;

typedef struct mdns_host_item_t {
  const char *hostname;
  mdns_ip_addr_t *address_list;
  struct mdns_host_item_t *next;
} mdns_host_item_t;

typedef struct mdns_out_answer_s {
  struct mdns_out_answer_s *next;
  uint16_t type;
  uint8_t bye;
  uint8_t flush;
  void *service;
  mdns_host_item_t *host;
  const char *custom_instance;
  const char *custom_service;
  const char *custom_proto;
} mdns_out_answer_t;

typedef struct mdns_tx_packet_s {
  struct mdns_tx_packet_s *next;
  uint32_t send_at;
  mdns_if_t tcpip_if;
  mdns_ip_protocol_t ip_protocol;
  esp_ip_addr_t dst;
  uint16_t port;
  uint16_t flags;
  uint8_t distributed;
  void *questions;
  mdns_out_answer_t *answers;
  mdns_out_answer_t *servers;
  mdns_out_answer_t *additional;
  bool queued;
  uint16_t id;
} mdns_tx_packet_t;

typedef struct {
  esp_err_t init_result;
  esp_err_t hostname_set_result;
  esp_err_t instance_name_set_result;
  esp_err_t service_add_result;
  esp_err_t service_txt_set_result;
  esp_err_t service_remove_result;
  esp_err_t netif_action_result;
  esp_err_t delegate_add_result;
  esp_err_t delegate_set_address_result;
  esp_err_t delegate_remove_result;
  int init_call_count;
  int hostname_set_call_count;
  int instance_name_set_call_count;
  int service_add_call_count;
  int service_txt_set_call_count;
  int service_remove_call_count;
  int free_call_count;
  int netif_action_call_count;
  int async_new_call_count;
  int async_get_results_call_count;
  int async_delete_call_count;
  int query_results_free_call_count;
  int delegate_add_call_count;
  int delegate_set_address_call_count;
  int delegate_remove_call_count;
  int private_alloc_call_count;
  int private_create_answer_call_count;
  int private_dispatch_call_count;
  int private_free_packet_call_count;
  int private_announce_count;
  int private_goodbye_count;
  int real_write_call_count;
  int last_netif_action;
  bool async_new_succeeds;
  bool async_get_results_finished;
  bool private_interface_ready[MDNS_MAX_INTERFACES];
  bool private_alloc_succeeds;
  bool private_create_answer_succeeds;
  esp_err_t async_delete_result;
  uint16_t last_query_type;
  uint32_t last_query_timeout_ms;
  uint32_t last_get_results_timeout_ms;
  size_t last_query_max_results;
  mdns_result_t *async_results;
  mdns_result_t *last_freed_results;
  uint32_t delegated_ipv4_address;
  uint16_t last_private_packet_flags;
  uint16_t last_private_answer_type;
  bool last_private_answer_flush;
  bool last_private_answer_goodbye;
  size_t last_write_len;
  uint8_t last_write_packet[512];
  char hostname[64];
  char instance_name[96];
  char service_type[48];
  char service_proto[16];
  char delegated_hostname[MDNS_NAME_BUF_LEN];
  uint16_t service_port;
  size_t txt_count;
  char txt_keys[12][32];
  char txt_values[12][128];
} mdns_mock_state_t;

extern mdns_mock_state_t g_mdns_mock;

void mdns_mock_reset(void);

esp_err_t mdns_init(void);
esp_err_t mdns_hostname_set(const char *hostname);
esp_err_t mdns_instance_name_set(const char *instance_name);
esp_err_t mdns_service_add(
    const char *instance_name,
    const char *service_type,
    const char *proto,
    uint16_t port,
    const mdns_txt_item_t *txt,
    size_t num_items);
esp_err_t mdns_service_txt_set(
    const char *service_type, const char *proto, const mdns_txt_item_t *txt, size_t num_items);
esp_err_t mdns_service_remove(const char *service_type, const char *proto);
void mdns_free(void);
esp_err_t mdns_netif_action(esp_netif_t *esp_netif, mdns_event_actions_t action);
mdns_search_once_t *mdns_query_async_new(const char *name,
                                         const char *service_type,
                                         const char *proto,
                                         uint16_t type,
                                         uint32_t timeout,
                                         size_t max_results,
                                         mdns_query_notify_t notifier);
bool mdns_query_async_get_results(mdns_search_once_t *search,
                                  uint32_t timeout,
                                  mdns_result_t **results,
                                  uint8_t *num_results);
esp_err_t mdns_query_async_delete(mdns_search_once_t *search);
void mdns_query_results_free(mdns_result_t *results);
esp_err_t mdns_delegate_hostname_add(const char *hostname, const mdns_ip_addr_t *address_list);
esp_err_t mdns_delegate_hostname_set_address(const char *hostname,
                                             const mdns_ip_addr_t *address_list);
esp_err_t mdns_delegate_hostname_remove(const char *hostname);

mdns_host_item_t *mdns_priv_get_hosts(void);
bool mdns_priv_if_ready(mdns_if_t tcpip_if, mdns_ip_protocol_t ip_protocol);
mdns_tx_packet_t *mdns_priv_alloc_packet(mdns_if_t tcpip_if,
                                         mdns_ip_protocol_t ip_protocol);
bool mdns_priv_create_answer(mdns_out_answer_t **destination,
                             uint16_t type,
                             void *service,
                             mdns_host_item_t *host,
                             bool flush,
                             bool bye);
void mdns_priv_dispatch_tx_packet(mdns_tx_packet_t *packet);
void mdns_priv_free_tx_packet(mdns_tx_packet_t *packet);
size_t __real_mdns_priv_if_write(mdns_if_t tcpip_if,
                                 mdns_ip_protocol_t ip_protocol,
                                 const esp_ip_addr_t *ip,
                                 uint16_t port,
                                 uint8_t *data,
                                 size_t len);

#ifdef __cplusplus
}
#endif

#endif  // MDNS_H
