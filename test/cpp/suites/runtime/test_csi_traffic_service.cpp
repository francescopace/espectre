/*
 * ESPectre - CSI Traffic Service Unit Tests
 *
 * Exercises pacing-mode packet filtering and sender tracking for CSI
 * traffic service.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "test_harness.h"

#include <array>
#include <cstdint>
#include <unistd.h>

#include "csi_traffic_service.h"
#include "lwip/inet.h"
#include "lwip/sockets.h"

using namespace espectre;

namespace {

uint16_t allocate_udp_port() {
  const int sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
  TEST_ASSERT_TRUE(sock >= 0);

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port = 0;
  TEST_ASSERT_TRUE(bind(sock, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0);

  socklen_t addr_len = sizeof(addr);
  TEST_ASSERT_TRUE(getsockname(sock, reinterpret_cast<sockaddr *>(&addr), &addr_len) == 0);
  const uint16_t port = ntohs(addr.sin_port);
  close(sock);
  return port;
}

void send_udp_datagram(uint16_t port, const void *payload, size_t len) {
  const int sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
  TEST_ASSERT_TRUE(sock >= 0);

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port = htons(port);
  TEST_ASSERT_TRUE(sendto(sock, payload, len, 0, reinterpret_cast<const sockaddr *>(&addr), sizeof(addr)) ==
                   static_cast<ssize_t>(len));
  close(sock);
}

void drain_service(CsiTrafficService &service, uint64_t expected_packets) {
  for (int attempt = 0; attempt < 20 && service.get_packets_received() < expected_packets; ++attempt) {
    service.loop();
    usleep(1000);
  }
}

}  // namespace

void setUp(void) {}

void tearDown(void) {}

void test_csi_traffic_service_pacing_mode_filters_payload_and_tracks_sender(void) {
  CsiTrafficService service;
  CsiTrafficServiceConfig config;
  config.mode = CsiTrafficMode::PACING;
  config.udp_port = allocate_udp_port();
  config.expected_payload = "ESPE";
  service.init(config);

  TEST_ASSERT_TRUE(service.start());

  static constexpr std::array<uint8_t, 4> kExpectedPayload{{'E', 'S', 'P', 'E'}};
  static constexpr std::array<uint8_t, 4> kUnexpectedPayload{{'N', 'O', 'P', 'E'}};
  send_udp_datagram(config.udp_port, kUnexpectedPayload.data(), kUnexpectedPayload.size());
  drain_service(service, 1U);
  TEST_ASSERT_EQUAL(0U, service.get_packets_received());

  send_udp_datagram(config.udp_port, kExpectedPayload.data(), kExpectedPayload.size());
  drain_service(service, 1U);
  TEST_ASSERT_EQUAL(1U, service.get_packets_received());

  sockaddr_in sender{};
  TEST_ASSERT_TRUE(service.get_last_sender(&sender));
  TEST_ASSERT_TRUE(sender.sin_port != 0U);

  service.stop();
}

void test_csi_traffic_projection_keeps_mode_separate_from_positive_target(void) {
  RuntimeConfig runtime_config;
  TEST_ASSERT_FALSE(runtime_config.traffic_generator_adaptive);
  runtime_config.csi_target_pps = 94U;
  runtime_config.csi_traffic_mode = CsiTrafficMode::DISABLED;

  CsiTrafficServiceConfig service_config = to_csi_traffic_config(runtime_config);
  TEST_ASSERT_EQUAL(static_cast<int>(CsiTrafficMode::DISABLED),
                    static_cast<int>(service_config.mode));
  TEST_ASSERT_EQUAL(94U, service_config.rate_pps);

  runtime_config.csi_traffic_mode = CsiTrafficMode::INTERNAL;
  service_config = to_csi_traffic_config(runtime_config);
  TEST_ASSERT_EQUAL(static_cast<int>(CsiTrafficMode::INTERNAL),
                    static_cast<int>(service_config.mode));
  TEST_ASSERT_EQUAL(94U, service_config.rate_pps);
}

int process(void) {
  UNITY_BEGIN();
  RUN_TEST(test_csi_traffic_service_pacing_mode_filters_payload_and_tracks_sender);
  RUN_TEST(test_csi_traffic_projection_keeps_mode_separate_from_positive_target);
  return UNITY_END();
}

#if defined(ESP_PLATFORM)
extern "C" void app_main(void) { process(); }
#else
int main(int argc, char **argv) {
  (void) argc;
  (void) argv;
  return process();
}
#endif
