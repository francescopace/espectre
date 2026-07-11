/*
 * ESPectre - UDP Listener Unit Tests
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "test_harness.h"

#include <array>
#include <cstdint>
#include <cstring>

#include "udp_listener.h"
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

void drain_listener(UDPListener &listener, uint64_t expected_packets) {
  for (int attempt = 0; attempt < 20 && listener.get_packets_received() < expected_packets; ++attempt) {
    listener.loop();
    usleep(1000);
  }
}

}  // namespace

void setUp(void) {}

void tearDown(void) {}

void test_udp_listener_accepts_any_payload_by_default(void) {
  UDPListener listener;
  const uint16_t port = allocate_udp_port();
  listener.init(port);
  TEST_ASSERT_TRUE(listener.start());

  static constexpr char kPayload[] = "junk";
  send_udp_datagram(port, kPayload, sizeof(kPayload) - 1U);
  drain_listener(listener, 1U);

  sockaddr_in sender{};
  TEST_ASSERT_EQUAL(1U, listener.get_packets_received());
  TEST_ASSERT_TRUE(listener.get_last_sender(&sender));

  listener.stop();
}

void test_udp_listener_filters_unexpected_payloads_when_configured(void) {
  UDPListener listener;
  const uint16_t port = allocate_udp_port();
  static constexpr std::array<uint8_t, 4> kExpectedPayload{{'E', 'S', 'P', 'E'}};
  static constexpr std::array<uint8_t, 4> kUnexpectedPayload{{'N', 'O', 'P', 'E'}};

  listener.init(port);
  listener.set_expected_payload(kExpectedPayload.data(), kExpectedPayload.size());
  TEST_ASSERT_TRUE(listener.start());

  send_udp_datagram(port, kUnexpectedPayload.data(), kUnexpectedPayload.size());
  drain_listener(listener, 1U);

  sockaddr_in sender{};
  TEST_ASSERT_EQUAL(0U, listener.get_packets_received());
  TEST_ASSERT_FALSE(listener.get_last_sender(&sender));

  send_udp_datagram(port, kExpectedPayload.data(), kExpectedPayload.size());
  drain_listener(listener, 1U);

  TEST_ASSERT_EQUAL(1U, listener.get_packets_received());
  TEST_ASSERT_TRUE(listener.get_last_sender(&sender));

  listener.stop();
}

int process(void) {
  UNITY_BEGIN();
  RUN_TEST(test_udp_listener_accepts_any_payload_by_default);
  RUN_TEST(test_udp_listener_filters_unexpected_payloads_when_configured);
  return UNITY_END();
}

#if defined(ESP_PLATFORM)
extern "C" void app_main(void) { process(); }
#else
int main(int argc, char **argv) { return process(); }
#endif
