/*
 * ESPectre - Host PSA Crypto adapter with injectable failures
 *
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <openssl/evp.h>

using psa_status_t = int32_t;
using psa_algorithm_t = uint32_t;
constexpr psa_status_t PSA_SUCCESS = 0;
constexpr psa_status_t PSA_ERROR_GENERIC_ERROR = -132;
constexpr psa_algorithm_t PSA_ALG_SHA_256 = 0x02000009;

struct psa_crypto_mock_state_t {
  psa_status_t init_result{PSA_SUCCESS};
  psa_status_t hash_result{PSA_SUCCESS};
  size_t digest_length{32U};
  int init_count{0};
  int hash_count{0};
  psa_algorithm_t algorithm{0};
  size_t output_size{0};
  std::vector<uint8_t> input;
};

inline psa_crypto_mock_state_t g_psa_crypto_mock{};

inline psa_status_t psa_crypto_init() {
  ++g_psa_crypto_mock.init_count;
  return g_psa_crypto_mock.init_result;
}

inline psa_status_t psa_hash_compute(psa_algorithm_t algorithm, const uint8_t* input,
                                     size_t input_length, uint8_t* hash, size_t hash_size,
                                     size_t* hash_length) {
  auto& state = g_psa_crypto_mock;
  ++state.hash_count;
  state.algorithm = algorithm;
  state.output_size = hash_size;
  state.input.assign(input, input + input_length);
  if (state.hash_result != PSA_SUCCESS) {
    return state.hash_result;
  }
  if (state.init_count == 0 || state.init_result != PSA_SUCCESS ||
      algorithm != PSA_ALG_SHA_256 || hash_size < 32U) {
    return PSA_ERROR_GENERIC_ERROR;
  }
  unsigned int length = 0;
  if (EVP_Digest(input, input_length, hash, &length, EVP_sha256(), nullptr) != 1) {
    return PSA_ERROR_GENERIC_ERROR;
  }
  *hash_length = state.digest_length;
  return PSA_SUCCESS;
}
