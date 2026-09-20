/*
 * ESPectre - Device Identity Tests
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "test_harness.h"

#include "device_identity.h"
#include "espectre_protocol.h"
#include "esp_idf_version.h"
#include "esp_mac.h"
#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(6, 0, 0)
#include "psa/crypto.h"
#else
#include "mbedtls/sha256.h"
#endif

#include <array>
#include <cstring>

using namespace espectre;

const char* scenario = "success";

void setUp(void) {
  g_esp_mac_mock = {};
#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(6, 0, 0)
  g_psa_crypto_mock = {};
#else
  g_mbedtls_sha256_mock = {};
#endif
}
void tearDown(void) {}

// Host adapters use OpenSSL for real hashes. Firmware builds separately check
// the actual IDF headers and libraries; these tests do not exercise hardware SHA.
void test_sha256_known_vectors(void) {
  struct Vector {
    const char* input;
    const char* digest;
  };
  const Vector vectors[] = {
      {"", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
      {"abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"},
      {"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
       "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"},
  };
  for (const auto& vector : vectors) {
    std::array<unsigned char, 32> digest{};
    const auto* input = reinterpret_cast<const unsigned char*>(vector.input);
#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(6, 0, 0)
    size_t length = 0;
    TEST_ASSERT_EQUAL(PSA_SUCCESS, psa_crypto_init());
    TEST_ASSERT_EQUAL(PSA_SUCCESS, psa_hash_compute(PSA_ALG_SHA_256, input,
                      std::strlen(vector.input), digest.data(), digest.size(), &length));
    TEST_ASSERT_EQUAL(digest.size(), length);
#else
    TEST_ASSERT_EQUAL(0, mbedtls_sha256(input, std::strlen(vector.input), digest.data(), 0));
#endif
    std::string hex;
    for (unsigned char byte : digest) {
      hex += "0123456789abcdef"[byte >> 4];
      hex += "0123456789abcdef"[byte & 15];
    }
    TEST_ASSERT_EQUAL_STRING(vector.digest, hex.c_str());
  }
}

void test_device_identity_uses_station_mac_sha256_pseudonym(void) {
  // SHA-256("espectre-device-id-v1" || 7c:2c:67:42:bb:ac), first 8 bytes,
  // interpreted big-endian. Neither a NUL nor MAC text belongs in the input.
  const uint64_t expected = 0x3cf79180d3a0aca4ULL;
  const unsigned char input[] = {
      'e', 's', 'p', 'e', 'c', 't', 'r', 'e', '-', 'd', 'e', 'v', 'i', 'c', 'e',
      '-', 'i', 'd', '-', 'v', '1', 0x7c, 0x2c, 0x67, 0x42, 0xbb, 0xac,
  };
  TEST_ASSERT_EQUAL(expected, derive_runtime_device_id());
  TEST_ASSERT_EQUAL_STRING("3cf79180d3a0aca4", derive_runtime_device_id_string().c_str());
  g_esp_mac_mock.result = ESP_FAIL;
  TEST_ASSERT_EQUAL(expected, derive_runtime_device_id());
  TEST_ASSERT_EQUAL(1, g_esp_mac_mock.call_count);
#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(6, 0, 0)
  TEST_ASSERT_EQUAL(1, g_psa_crypto_mock.init_count);
  TEST_ASSERT_EQUAL(1, g_psa_crypto_mock.hash_count);
  TEST_ASSERT_EQUAL(PSA_ALG_SHA_256, g_psa_crypto_mock.algorithm);
  TEST_ASSERT_EQUAL(32U, g_psa_crypto_mock.output_size);
  const auto& captured = g_psa_crypto_mock.input;
#else
  TEST_ASSERT_EQUAL(1, g_mbedtls_sha256_mock.call_count);
  TEST_ASSERT_EQUAL(0, g_mbedtls_sha256_mock.is224);
  const auto& captured = g_mbedtls_sha256_mock.input;
#endif
  TEST_ASSERT_EQUAL(sizeof(input), captured.size());
  TEST_ASSERT_EQUAL(0, std::memcmp(input, captured.data(), sizeof(input)));
}

void test_device_identity_failure_returns_cached_default(void) {
  const bool mac_failure = std::strcmp(scenario, "mac_failure") == 0;
  if (mac_failure) {
    g_esp_mac_mock.result = ESP_FAIL;
  }
#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(6, 0, 0)
  const bool init_failure = std::strcmp(scenario, "init_failure") == 0;
  if (init_failure) {
    g_psa_crypto_mock.init_result = PSA_ERROR_GENERIC_ERROR;
  } else if (std::strcmp(scenario, "hash_failure") == 0) {
    g_psa_crypto_mock.hash_result = PSA_ERROR_GENERIC_ERROR;
  } else if (std::strcmp(scenario, "short_digest") == 0) {
    g_psa_crypto_mock.digest_length = 31;
  } else if (std::strcmp(scenario, "long_digest") == 0) {
    g_psa_crypto_mock.digest_length = 33;
  } else if (std::strcmp(scenario, "empty_digest") == 0) {
    g_psa_crypto_mock.digest_length = 0;
  }
#else
  if (std::strcmp(scenario, "hash_failure") == 0) {
    g_mbedtls_sha256_mock.result = -1;
  }
#endif
  TEST_ASSERT_EQUAL(ESPECTRE_DEFAULT_DEVICE_ID, derive_runtime_device_id());
  const auto expected_text = format_espectre_device_id(ESPECTRE_DEFAULT_DEVICE_ID);
  TEST_ASSERT_EQUAL_STRING(expected_text.c_str(), derive_runtime_device_id_string().c_str());
  TEST_ASSERT_EQUAL(ESPECTRE_DEFAULT_DEVICE_ID, derive_runtime_device_id());
  TEST_ASSERT_EQUAL(1, g_esp_mac_mock.call_count);
#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(6, 0, 0)
  TEST_ASSERT_EQUAL(mac_failure ? 0 : 1, g_psa_crypto_mock.init_count);
  TEST_ASSERT_EQUAL(mac_failure || init_failure ? 0 : 1, g_psa_crypto_mock.hash_count);
#else
  TEST_ASSERT_EQUAL(mac_failure ? 0 : 1, g_mbedtls_sha256_mock.call_count);
#endif
}

int main(int argc, char** argv) {
  if (argc > 1) {
    scenario = argv[1];
  }
  UNITY_BEGIN();
  RUN_TEST(test_sha256_known_vectors);
  // Each CTest scenario is a fresh process, including the production ID cache.
  if (std::strcmp(scenario, "success") == 0) {
    RUN_TEST(test_device_identity_uses_station_mac_sha256_pseudonym);
  } else {
    RUN_TEST(test_device_identity_failure_returns_cached_default);
  }
  return UNITY_END();
}
