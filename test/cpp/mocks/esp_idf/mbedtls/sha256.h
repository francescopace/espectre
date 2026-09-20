/*
 * ESPectre - Mock mbedtls/sha256.h
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <cstddef>
#include <vector>

#include <openssl/evp.h>

struct mbedtls_sha256_mock_state_t {
  int result{0};
  std::vector<unsigned char> input;
  size_t input_len{0U};
  int is224{-1};
  int call_count{0};
};

inline mbedtls_sha256_mock_state_t g_mbedtls_sha256_mock{};

inline int mbedtls_sha256(const unsigned char* input, size_t input_len,
                          unsigned char output[32], int is224) {
  g_mbedtls_sha256_mock.input.assign(input, input + input_len);
  g_mbedtls_sha256_mock.is224 = is224;
  g_mbedtls_sha256_mock.input_len = input_len;
  g_mbedtls_sha256_mock.call_count++;
  if (g_mbedtls_sha256_mock.result == 0 && output != nullptr) {
    unsigned int length = 0;
    return EVP_Digest(input, input_len, output, &length,
                      is224 ? EVP_sha224() : EVP_sha256(), nullptr) == 1 ? 0 : -1;
  }
  return g_mbedtls_sha256_mock.result;
}
