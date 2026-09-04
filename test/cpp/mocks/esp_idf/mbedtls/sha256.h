/*
 * ESPectre - Mock mbedtls/sha256.h
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <cstddef>
#include <cstring>

struct mbedtls_sha256_mock_state_t {
  int result{0};
  unsigned char digest[32]{
      0x3c, 0xf7, 0x91, 0x80, 0xd3, 0xa0, 0xac, 0xa4,
  };
  size_t input_len{0U};
  int call_count{0};
};

inline mbedtls_sha256_mock_state_t g_mbedtls_sha256_mock{};

inline int mbedtls_sha256(const unsigned char* input, size_t input_len,
                          unsigned char output[32], int is224) {
  (void)input;
  (void)is224;
  g_mbedtls_sha256_mock.input_len = input_len;
  g_mbedtls_sha256_mock.call_count++;
  if (g_mbedtls_sha256_mock.result == 0 && output != nullptr) {
    std::memcpy(output, g_mbedtls_sha256_mock.digest,
                sizeof(g_mbedtls_sha256_mock.digest));
  }
  return g_mbedtls_sha256_mock.result;
}
