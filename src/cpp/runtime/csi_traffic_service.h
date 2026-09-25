/*
 * ESPectre - CSI Traffic Service
 *
 * Shared policy for internal CSI traffic generation and external ingress.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "runtime_config.h"
#include "udp_datagram_socket.h"

namespace espectre {

/**
 * Called for each accepted external traffic packet, from the ingress loop().
 *
 * Receives the registered context, the sender, and the cumulative count of
 * accepted packets.
 */
using csi_traffic_packet_callback_t = void (*)(void *, const UdpDatagramPeer &, uint64_t);

/** Traffic policy for CsiTrafficService; build it with to_csi_traffic_config(). */
struct CsiTrafficServiceConfig {
  /** Internal generator mode, or `EXTERNAL` to listen for another host. */
  TrafficGeneratorMode mode{TrafficGeneratorMode::PING};
  /** Internal generator send rate, in packets per second. */
  uint32_t rate_pps{100U};
  /** UDP port the external listener binds. */
  uint16_t udp_port{5555U};
  /** IPv4 multicast group the external listener joins; empty for unicast only. */
  std::string multicast_group;
};

/**
 * Internal traffic source that makes the access point answer the device.
 *
 * TrafficGeneratorManager is the ESP-IDF implementation. Call every method
 * from the owner task.
 */
class ICsiTrafficGenerator {
 public:
  virtual ~ICsiTrafficGenerator() = default;

  /** Configure the send rate and mode while stopped. */
  virtual void init(uint32_t target_pps, TrafficGeneratorMode mode) = 0;
  /** Start sending to an IPv4 address in network byte order; false on failure. */
  virtual bool start(uint32_t target_addr) = 0;
  /** Signal the sender to stop. loop() releases it; this call does not block. */
  virtual void stop() = 0;
  /**
   * Advance periodic work from the owner task.
   *
   * Called while stopped too, so an implementation can finish a stop() that
   * returned before its sender exited.
   */
  virtual void loop() = 0;
  /**
   * Keep a deferred start from launching inside loop().
   *
   * The owner sets this while it still has to disable CSI or touch the radio.
   * start() itself still launches once the previous sender has already exited.
   *
   * @param hold True to park the deferred start.
   */
  virtual void hold_pending_restart(bool hold) { (void)hold; }
  virtual bool is_running() const = 0;
  /** True when no sender is inside a send. A deferred start can still be waiting. */
  virtual bool is_quiescent() const { return !is_running(); }
  /** True after start() has created a worker that stop() has not signalled. */
  virtual bool has_live_worker() const { return is_running(); }
  /** True once, after a deferred launch fails. Synchronous start() does not set this. */
  virtual bool consume_start_failure() { return false; }
  /** True once, after a stopped sender has not exited within its grace period. */
  virtual bool consume_stop_timeout() { return false; }
  /** Successful sends in the current session. */
  virtual uint32_t send_success_count() const = 0;
  /** ICMP identifier used by ping traffic, so its replies can be recognized. */
  virtual uint16_t icmp_identifier() const = 0;
};

/**
 * Listener for traffic sent by another host in `EXTERNAL` mode.
 *
 * Only datagrams whose payload matches the expected marker are counted and
 * reported. Call every method from the owner task.
 */
class ICsiTrafficIngress {
 public:
  virtual ~ICsiTrafficIngress() = default;

  /** Set the UDP port to bind at the next start(). */
  virtual void init(uint16_t port) = 0;
  /** Set the multicast group to join, or `nullptr` for unicast only. */
  virtual void set_multicast_group(const char *group) = 0;
  /** Set the payload a datagram must carry to be accepted. */
  virtual void set_expected_payload(const uint8_t *payload, size_t len) = 0;
  /** Register the per-packet callback; `context` is passed through unchanged. */
  virtual void set_packet_callback(csi_traffic_packet_callback_t callback,
                                   void *context = nullptr) = 0;
  /** Bind the socket and join the group; false on failure. */
  virtual bool start() = 0;
  /** Close the socket. */
  virtual void stop() = 0;
  /** Receive pending datagrams and deliver the callback. */
  virtual void loop() = 0;
  virtual bool is_running() const = 0;
  /** Accepted packets since init(). */
  virtual uint64_t get_packets_received() const = 0;
  /** Sender of the latest accepted packet; false before the first one. */
  virtual bool get_last_sender(UdpDatagramPeer *out_peer) const = 0;
};

/** Project runtime configuration onto transport-independent CSI traffic policy. */
CsiTrafficServiceConfig to_csi_traffic_config(const RuntimeConfig &config);

/**
 * Keeps CSI-bearing traffic flowing in the configured mode.
 *
 * Internal modes drive the generator; `EXTERNAL` drives the listener, which
 * accepts the ESPectre traffic marker. The full runtime owns one of these;
 * firmware that captures CSI itself can use it directly.
 *
 * @par Threading
 * Call every method from one owner task, including loop().
 */
class CsiTrafficService {
 public:
  /** Bind the generator and listener. Neither is owned; both must outlive the service. */
  CsiTrafficService(ICsiTrafficGenerator &traffic_generator,
                    ICsiTrafficIngress &traffic_ingress)
      : traffic_generator_(traffic_generator), traffic_ingress_(traffic_ingress) {}

  /** Configure the mode, rate, and listener while stopped. */
  void init(const CsiTrafficServiceConfig &config);
  /**
   * Start the source for the configured mode; a running source is left as is.
   *
   * `target_addr` is the internal generator's IPv4 destination in network byte
   * order, ignored in `EXTERNAL` mode. Returns false when the source fails to
   * start.
   */
  bool start(uint32_t target_addr = 0U);
  /** Stop the generator and the listener. */
  void stop();
  /** Forward hold_pending_restart() to the generator. */
  void hold_pending_restart(bool hold);
  /** Advance the running source, and let a stopped generator finish its stop. */
  void loop();
  /** Register the callback for accepted external packets. */
  void set_packet_callback(csi_traffic_packet_callback_t callback,
                           void *context = nullptr);

  /** Whether the source for the configured mode is running. */
  bool is_running() const;
  /** True when the generator is not inside a send. */
  bool is_quiescent() const;
  /** True between a generator stop() and the exit of its sender. */
  bool generator_is_stopping() const;
  /** True when the configured source has a live sender, not only a deferred start. */
  bool source_is_active() const;
  /** True once, after a deferred generator launch fails. */
  bool consume_generator_start_failure();
  /** True once, after a stopped generator has not exited within its grace period. */
  bool consume_generator_stop_timeout();
  /** Sender of the latest accepted external packet; false before the first one. */
  bool get_last_sender(UdpDatagramPeer *out_peer) const;
  /** Accepted external packets since init(). */
  uint64_t get_packets_received() const;
  /** Successful internal generator sends; zero in external mode. */
  uint32_t get_generator_packets_total() const;
  /** ICMP identifier of the internal ping traffic. */
  uint16_t internal_icmp_identifier() const;
  /** Mode set by the last init(). */
  TrafficGeneratorMode mode() const { return mode_; }

 private:
  TrafficGeneratorMode mode_{TrafficGeneratorMode::PING};
  ICsiTrafficGenerator &traffic_generator_;
  ICsiTrafficIngress &traffic_ingress_;
};

}  // namespace espectre
