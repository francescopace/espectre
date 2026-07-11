#pragma once

#include <cstddef>
#include <cstdint>

#include "esp_wifi.h"

namespace espectre {

bool csi_frame_matches_local_identity(const wifi_csi_info_t *info,
                                      uint32_t local_ip_addr,
                                      const uint8_t *local_mac_addr);

}  // namespace espectre
