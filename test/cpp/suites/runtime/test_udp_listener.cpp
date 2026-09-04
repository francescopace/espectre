/*
 * ESPectre - UDP Listener Unit Tests
 *
 * Exercises UDP ingress behavior through an in-memory socket boundary.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "test_harness.h"

#include <array>
#include <cstdint>

#include "csi_traffic_fakes.h"
#include "runtime_sensing_schema.h"
#include "udp_listener.h"

using namespace espectre;
using namespace espectre::test;

namespace {

constexpr UdpDatagramPeer kPeer{0x0100007FU, 4321U};

}  // namespace

void setUp(void) {}

void tearDown(void) {}

void test_udp_listener_accepts_any_payload_by_default(void) {
  FakeUdpDatagramSocket socket;
  UDPListener listener(socket);
  listener.init(6001U);
  TEST_ASSERT_TRUE(listener.start());
  TEST_ASSERT_EQUAL(6001U, socket.port);

  static constexpr uint8_t kPayload[] = {'j', 'u', 'n', 'k'};
  socket.enqueue(kPayload, sizeof(kPayload), kPeer);
  listener.loop();

  UdpDatagramPeer sender{};
  TEST_ASSERT_EQUAL(1U, listener.get_packets_received());
  TEST_ASSERT_TRUE(listener.get_last_sender(&sender));
  TEST_ASSERT_EQUAL(kPeer.ipv4_addr, sender.ipv4_addr);
  TEST_ASSERT_EQUAL(kPeer.port, sender.port);

  listener.stop();
  TEST_ASSERT_FALSE(socket.opened);
}

void test_udp_listener_filters_unexpected_payloads_when_configured(void) {
  FakeUdpDatagramSocket socket;
  UDPListener listener(socket);
  static constexpr std::array<uint8_t, 1> kPeriodPayload{{'.'}};
  static constexpr std::array<uint8_t, 3> kTruncatedPayload{{0xF0U, 0x9FU, 0x91U}};
  static constexpr std::array<uint8_t, 4> kMalformedPayload{{0xF0U, 0x28U, 0x8CU, 0xBCU}};
  static constexpr std::array<uint8_t, 5> kExtendedPayload{{0xF0U, 0x9FU, 0x91U, 0xBBU, 'x'}};

  listener.init(6002U);
  listener.set_expected_payload(RUNTIME_CSI_TRAFFIC_MARKER_BYTES,
                                RUNTIME_CSI_TRAFFIC_MARKER_LENGTH);
  TEST_ASSERT_TRUE(listener.start());

  socket.enqueue(kPeriodPayload.data(), kPeriodPayload.size(), kPeer);
  socket.enqueue(kTruncatedPayload.data(), kTruncatedPayload.size(), kPeer);
  socket.enqueue(kMalformedPayload.data(), kMalformedPayload.size(), kPeer);
  socket.enqueue(kExtendedPayload.data(), kExtendedPayload.size(), kPeer);
  listener.loop();

  UdpDatagramPeer sender{};
  TEST_ASSERT_EQUAL(0U, listener.get_packets_received());
  TEST_ASSERT_FALSE(listener.get_last_sender(&sender));

  socket.enqueue(RUNTIME_CSI_TRAFFIC_MARKER_BYTES,
                 RUNTIME_CSI_TRAFFIC_MARKER_LENGTH, kPeer);
  listener.loop();

  TEST_ASSERT_EQUAL(1U, listener.get_packets_received());
  TEST_ASSERT_TRUE(listener.get_last_sender(&sender));
}

void test_udp_listener_propagates_socket_open_failure(void) {
  FakeUdpDatagramSocket socket;
  socket.open_result = false;
  UDPListener listener(socket);
  listener.init(6003U);

  TEST_ASSERT_FALSE(listener.start());
  TEST_ASSERT_FALSE(listener.is_running());
  TEST_ASSERT_EQUAL(1U, socket.open_calls);
}

int process(void) {
  UNITY_BEGIN();
  RUN_TEST(test_udp_listener_accepts_any_payload_by_default);
  RUN_TEST(test_udp_listener_filters_unexpected_payloads_when_configured);
  RUN_TEST(test_udp_listener_propagates_socket_open_failure);
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
