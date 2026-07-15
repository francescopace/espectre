/*
 * ESPectre - Mock esp_event_mock.cpp
 *
 * Host-side mock of esp_event_mock.cpp for native C++ tests.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */
#include "esp_event.h"

esp_event_mock_state_t g_esp_event_mock{};

void esp_event_mock_reset(void) {
  g_esp_event_mock = {};
}

void esp_event_mock_emit(esp_event_base_t event_base, int32_t event_id,
                         void *event_data) {
  for (int i = 0; i < 8; i++) {
    esp_event_mock_slot_t *slot = &g_esp_event_mock.slots[i];
    if (!slot->active || slot->handler == nullptr) {
      continue;
    }

    const int same_base =
        (slot->event_base == event_base) ||
        (slot->event_base != nullptr && event_base != nullptr &&
         strcmp(slot->event_base, event_base) == 0);
    if (same_base && (slot->event_id == event_id || slot->event_id == ESP_EVENT_ANY_ID)) {
      slot->handler(slot->handler_arg, event_base, event_id, event_data);
    }
  }
}
