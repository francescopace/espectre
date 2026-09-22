/*
 * ESPectre - Device Identity
 *
 * Derives stable runtime device identifiers from platform identity data.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "device_identity.h"

#include "runtime/espectre_protocol.h"

#include <array>
#include <cstring>

#include <esp_mac.h>
#include <esp_idf_version.h>
#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(6, 0, 0)
#include <psa/crypto.h>
#else
#include <mbedtls/sha256.h>
#endif

namespace espectre {

namespace {

constexpr char kDeviceIdDomain[] = "espectre-device-id-v1";

uint64_t derive_device_id_once() {
  uint8_t mac[6] = {0U, 0U, 0U, 0U, 0U, 0U};
  if (esp_read_mac(mac, ESP_MAC_WIFI_STA) != ESP_OK) {
    return ESPECTRE_DEFAULT_DEVICE_ID;
  }

  std::array<unsigned char, sizeof(kDeviceIdDomain) - 1U + sizeof(mac)> input{};
  std::memcpy(input.data(), kDeviceIdDomain, sizeof(kDeviceIdDomain) - 1U);
  std::memcpy(input.data() + sizeof(kDeviceIdDomain) - 1U, mac, sizeof(mac));

  std::array<unsigned char, 32U> digest{};
#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(6, 0, 0)
  // IDF initializes PSA at startup; repeated initialization is safe and lets
  // this SDK entry point check initialization before using the crypto API.
  if (psa_crypto_init() != PSA_SUCCESS) {
    return ESPECTRE_DEFAULT_DEVICE_ID;
  }
  size_t digest_length = 0U;
  if (psa_hash_compute(PSA_ALG_SHA_256, input.data(), input.size(),
                       digest.data(), digest.size(), &digest_length) != PSA_SUCCESS ||
      digest_length != digest.size()) {
    return ESPECTRE_DEFAULT_DEVICE_ID;
  }
#else
  if (mbedtls_sha256(input.data(), input.size(), digest.data(), 0) != 0) {
    return ESPECTRE_DEFAULT_DEVICE_ID;
  }
#endif

  uint64_t device_id = 0U;
  for (size_t i = 0U; i < sizeof(device_id); ++i) {
    device_id = (device_id << 8U) | static_cast<uint64_t>(digest[i]);
  }
  return device_id;
}

}  // namespace

uint64_t derive_runtime_device_id() {
  static const uint64_t device_id = derive_device_id_once();
  return device_id;
}

std::string derive_runtime_device_id_string() {
  static const std::string device_id = format_espectre_device_id(derive_runtime_device_id());
  return device_id;
}

}  // namespace espectre
