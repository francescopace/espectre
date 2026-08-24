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

#include <stddef.h>

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

typedef struct {
  esp_err_t init_result;
  esp_err_t hostname_set_result;
  esp_err_t instance_name_set_result;
  esp_err_t service_add_result;
  esp_err_t service_txt_set_result;
  esp_err_t service_remove_result;
  esp_err_t netif_action_result;
  int init_call_count;
  int hostname_set_call_count;
  int instance_name_set_call_count;
  int service_add_call_count;
  int service_txt_set_call_count;
  int service_remove_call_count;
  int free_call_count;
  int netif_action_call_count;
  int last_netif_action;
  char hostname[64];
  char instance_name[96];
  char service_type[48];
  char service_proto[16];
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

#ifdef __cplusplus
}
#endif

#endif  // MDNS_H
