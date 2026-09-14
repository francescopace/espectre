/*
 * ESPectre - CSI Traffic Service Unit Tests
 *
 * Exercises shared traffic policy without opening platform sockets.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "test_harness.h"

#include <cstring>

#include "csi_traffic_fakes.h"
#include "csi_traffic_service.h"

using namespace espectre;
using namespace espectre::test;

void setUp(void) {}

void tearDown(void) {}

void test_csi_traffic_service_selects_external_ingress_and_reports_diagnostics(void) {
  FakeCsiTrafficGenerator generator;
  FakeCsiTrafficIngress ingress;
  CsiTrafficService service(generator, ingress);
  CsiTrafficServiceConfig config;
  config.mode = CsiTrafficMode::EXTERNAL;
  config.udp_port = 6001U;
  config.multicast_group = "239.12.12.12";

  service.init(config);
  TEST_ASSERT_EQUAL(6001U, ingress.port);
  TEST_ASSERT_EQUAL_STRING("239.12.12.12", ingress.multicast_group.c_str());
  TEST_ASSERT_EQUAL(RUNTIME_CSI_TRAFFIC_MARKER_LENGTH, ingress.expected_payload.size());
  TEST_ASSERT_TRUE(std::memcmp(RUNTIME_CSI_TRAFFIC_MARKER_BYTES,
                               ingress.expected_payload.data(),
                               RUNTIME_CSI_TRAFFIC_MARKER_LENGTH) == 0);

  TEST_ASSERT_TRUE(service.start());
  TEST_ASSERT_EQUAL(0U, generator.start_calls);
  TEST_ASSERT_EQUAL(1U, ingress.start_calls);

  ingress.packets_received = 3U;
  ingress.last_sender = {0x0100007FU, 1234U};
  service.loop();
  TEST_ASSERT_EQUAL(1U, ingress.loop_calls);
  TEST_ASSERT_EQUAL(3U, service.get_packets_received());
  TEST_ASSERT_EQUAL(3U, service.get_traffic_packets_total());

  UdpDatagramPeer sender{};
  TEST_ASSERT_TRUE(service.get_last_sender(&sender));
  TEST_ASSERT_EQUAL(0x0100007FU, sender.ipv4_addr);
  TEST_ASSERT_EQUAL(1234U, sender.port);

  service.stop();
  TEST_ASSERT_FALSE(service.is_running());
}

void test_csi_traffic_service_selects_internal_generator(void) {
  FakeCsiTrafficGenerator generator;
  FakeCsiTrafficIngress ingress;
  CsiTrafficService service(generator, ingress);
  CsiTrafficServiceConfig config;
  config.mode = CsiTrafficMode::INTERNAL;
  config.rate_pps = 94U;
  config.traffic_mode = RuntimeTrafficMode::DNS_TCP;

  service.init(config);
  TEST_ASSERT_EQUAL(94U, generator.rate_pps);
  TEST_ASSERT_TRUE(generator.mode == RuntimeTrafficMode::DNS_TCP);

  TEST_ASSERT_TRUE(service.start(0x0101A8C0U));
  TEST_ASSERT_EQUAL(1U, generator.start_calls);
  TEST_ASSERT_EQUAL(0x0101A8C0U, generator.gateway_addr);
  TEST_ASSERT_EQUAL(0U, ingress.start_calls);

  generator.send_successes = 7U;
  TEST_ASSERT_EQUAL(7U, service.get_traffic_packets_total());
  TEST_ASSERT_EQUAL(0x1234U, service.internal_icmp_identifier());
}

void test_csi_traffic_projection_keeps_mode_separate_from_positive_target(void) {
  RuntimeConfig runtime_config;
  runtime_config.csi_target_pps = 94U;
  runtime_config.csi_traffic_mode = CsiTrafficMode::INTERNAL;

  CsiTrafficServiceConfig service_config = to_csi_traffic_config(runtime_config);
  TEST_ASSERT_TRUE(service_config.mode == CsiTrafficMode::INTERNAL);
  TEST_ASSERT_EQUAL(94U, service_config.rate_pps);
  TEST_ASSERT_TRUE(service_config.traffic_mode == RuntimeTrafficMode::PING);

  runtime_config.traffic_generator_mode = RuntimeTrafficMode::DNS;
  service_config = to_csi_traffic_config(runtime_config);
  TEST_ASSERT_TRUE(service_config.traffic_mode == RuntimeTrafficMode::DNS);

  runtime_config.traffic_generator_mode = RuntimeTrafficMode::DNS_TCP;
  service_config = to_csi_traffic_config(runtime_config);
  TEST_ASSERT_TRUE(service_config.traffic_mode == RuntimeTrafficMode::DNS_TCP);

  runtime_config.csi_traffic_mode = CsiTrafficMode::EXTERNAL;
  runtime_config.csi_traffic_multicast_group.clear();
  service_config = to_csi_traffic_config(runtime_config);
  TEST_ASSERT_TRUE(service_config.mode == CsiTrafficMode::EXTERNAL);
  TEST_ASSERT_TRUE(service_config.multicast_group.empty());
}

int process(void) {
  UNITY_BEGIN();
  RUN_TEST(test_csi_traffic_service_selects_external_ingress_and_reports_diagnostics);
  RUN_TEST(test_csi_traffic_service_selects_internal_generator);
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
