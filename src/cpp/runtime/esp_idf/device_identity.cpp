#include "device_identity.h"

#include "espectre_protocol.h"

#include <esp_mac.h>

namespace esphome {
namespace espectre {

uint64_t derive_runtime_device_id() {
  uint8_t mac[6] = {0U, 0U, 0U, 0U, 0U, 0U};
  if (esp_read_mac(mac, ESP_MAC_WIFI_STA) != ESP_OK) {
    return ESPECTRE_DEFAULT_DEVICE_ID;
  }
  return espectre_device_id_from_mac(mac, sizeof(mac));
}

std::string derive_runtime_device_id_string() { return format_espectre_device_id(derive_runtime_device_id()); }

}  // namespace espectre
}  // namespace esphome
