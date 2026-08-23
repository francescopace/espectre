/*
 * ESPectre - Traffic Generator Manager Implementation
 *
 * One task owns pacing, socket draining, and local send-error recovery.
 * Protocol backends only describe the socket and encode one packet.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "traffic_generator_manager.h"

#include <algorithm>
#include <cerrno>
#include <cinttypes>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>

#ifdef ESP_PLATFORM
#include "lwip/tcp.h"
#else
#include <netinet/tcp.h>
#endif

#include "esp_netif.h"
#include "esp_timer.h"
#include "espectre_log.h"
#include "lwip/inet.h"
#include "lwip/sockets.h"
#include "sta_socket_helpers.h"

namespace espectre {

namespace {

static const char *const TAG = "TrafficGen";

constexpr uint8_t DNS_QUERY_TEMPLATE[] = {
    0x00, 0x00, 0x01, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01,
};
// Expedited Forwarding asks Wi-Fi/WMM queues to favor latency over aggregation.
// Gateways may ignore it, so failure to apply the socket option is non-fatal.
constexpr int SENSING_IP_TOS = 46 << 2;

struct __attribute__((packed)) IcmpEchoHeader {
  uint8_t type;
  uint8_t code;
  uint16_t checksum;
  uint16_t identifier;
  uint16_t sequence;
};

uint16_t internet_checksum(const void *data, size_t len) {
  const auto *bytes = static_cast<const uint8_t *>(data);
  uint32_t sum = 0U;
  while (len >= 2U) {
    sum += (static_cast<uint16_t>(bytes[0]) << 8U) | static_cast<uint16_t>(bytes[1]);
    bytes += 2;
    len -= 2U;
  }
  if (len == 1U) {
    sum += static_cast<uint16_t>(bytes[0]) << 8U;
  }
  while ((sum >> 16U) != 0U) {
    sum = (sum & 0xFFFFU) + (sum >> 16U);
  }
  return static_cast<uint16_t>(~sum);
}

class TrafficProtocol {
 public:
  virtual ~TrafficProtocol() = default;
  virtual const char *name() const = 0;
  virtual int socket_type() const = 0;
  virtual int socket_protocol() const = 0;
  virtual uint16_t destination_port() const = 0;
  virtual bool connection_oriented() const { return false; }
  virtual ssize_t send_packet(int sock, const sockaddr_in &destination) = 0;
};

class DnsTrafficProtocol final : public TrafficProtocol {
 public:
  const char *name() const override { return "dns"; }
  int socket_type() const override { return SOCK_STREAM; }
  int socket_protocol() const override { return IPPROTO_TCP; }
  uint16_t destination_port() const override { return 53U; }
  bool connection_oriented() const override { return true; }

  ssize_t send_packet(int sock, const sockaddr_in &destination) override {
    (void)destination;
    uint8_t frame[TRAFFIC_DNS_TCP_FRAME_SIZE];
    const size_t frame_len = build_dns_tcp_query_frame(++transaction_id_, frame, sizeof(frame));
    const ssize_t sent = send(sock, frame, frame_len, MSG_DONTWAIT);
    if (sent >= 0 && static_cast<size_t>(sent) != frame_len) {
      errno = EIO;
      return -1;
    }
    return sent;
  }

 private:
  uint16_t transaction_id_{0U};
};

class IcmpTrafficProtocol final : public TrafficProtocol {
 public:
  explicit IcmpTrafficProtocol(uint16_t identifier) : identifier_(identifier) {}

  const char *name() const override { return "ping"; }
  int socket_type() const override { return SOCK_RAW; }
  int socket_protocol() const override { return IPPROTO_ICMP; }
  uint16_t destination_port() const override { return 0U; }

  ssize_t send_packet(int sock, const sockaddr_in &destination) override {
    IcmpEchoHeader packet{};
    packet.type = 8U;
    packet.identifier = htons(identifier_);
    packet.sequence = htons(++sequence_);
    packet.checksum = htons(internet_checksum(&packet, sizeof(packet)));
    return sendto(sock,
                  &packet,
                  sizeof(packet),
                  0,
                  reinterpret_cast<const sockaddr *>(&destination),
                  sizeof(destination));
  }

 private:
  uint16_t identifier_{0U};
  uint16_t sequence_{0U};
};

const char *traffic_mode_name(TrafficGeneratorMode mode) {
  return mode == TrafficGeneratorMode::PING ? "ping" : "dns";
}

int create_protocol_socket(const TrafficProtocol &protocol) {
  const int sock = socket(AF_INET, protocol.socket_type(), protocol.socket_protocol());
  if (sock < 0) {
    ESP_LOGE(TAG, "Failed to create %s socket (errno=%d)", protocol.name(), errno);
    return -1;
  }

  if (!bind_socket_to_sta_interface(sock, TAG, protocol.name())) {
    ESP_LOGW(TAG, "Continuing without explicit %s socket binding", protocol.name());
  }
  const int sensing_tos = SENSING_IP_TOS;
  if (setsockopt(sock, IPPROTO_IP, IP_TOS, &sensing_tos,
                 sizeof(sensing_tos)) != 0) {
    ESP_LOGW(TAG, "Failed to mark %s traffic as low-latency (errno=%d)",
             protocol.name(), errno);
  }
  if (protocol.connection_oriented()) {
    const int enabled = 1;
    if (setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, &enabled, sizeof(enabled)) != 0) {
      ESP_LOGW(TAG, "Failed to disable Nagle for %s traffic (errno=%d)",
               protocol.name(), errno);
    }
  }
  const int flags = fcntl(sock, F_GETFL, 0);
  if (flags < 0 || fcntl(sock, F_SETFL, flags | O_NONBLOCK) < 0) {
    ESP_LOGW(TAG, "Failed to set %s socket non-blocking (errno=%d)", protocol.name(), errno);
  }
  return sock;
}

enum class SocketDrainResult {
  READY,
  PEER_CLOSED,
  ERROR,
};

SocketDrainResult drain_socket(int sock) {
  uint8_t buffer[128];
  while (true) {
    const ssize_t received = recv(sock, buffer, sizeof(buffer), MSG_DONTWAIT);
    if (received > 0) {
      continue;
    }
    if (received == 0) {
      return SocketDrainResult::PEER_CLOSED;
    }
    if (errno == EAGAIN || errno == EWOULDBLOCK) {
      return SocketDrainResult::READY;
    }
    return SocketDrainResult::ERROR;
  }
}

enum class TcpConnectionState {
  DISCONNECTED,
  CONNECTING,
  CONNECTED,
};

TcpConnectionState start_tcp_connect(int sock, const sockaddr_in &destination) {
  if (connect(sock,
              reinterpret_cast<const sockaddr *>(&destination),
              sizeof(destination)) == 0 || errno == EISCONN) {
    return TcpConnectionState::CONNECTED;
  }
  if (errno == EINPROGRESS || errno == EALREADY) {
    return TcpConnectionState::CONNECTING;
  }
  return TcpConnectionState::DISCONNECTED;
}

TcpConnectionState poll_tcp_connect(int sock) {
  fd_set write_fds;
  fd_set error_fds;
  FD_ZERO(&write_fds);
  FD_ZERO(&error_fds);
  FD_SET(sock, &write_fds);
  FD_SET(sock, &error_fds);
  timeval timeout{};
  const int ready = select(sock + 1, nullptr, &write_fds, &error_fds, &timeout);
  if (ready == 0) {
    return TcpConnectionState::CONNECTING;
  }
  if (ready < 0) {
    return TcpConnectionState::DISCONNECTED;
  }

  int socket_error = 0;
  socklen_t error_len = sizeof(socket_error);
  if (getsockopt(sock, SOL_SOCKET, SO_ERROR, &socket_error, &error_len) != 0 ||
      socket_error != 0) {
    if (socket_error != 0) {
      errno = socket_error;
    }
    return TcpConnectionState::DISCONNECTED;
  }
  return TcpConnectionState::CONNECTED;
}

}  // namespace

size_t build_dns_tcp_query_frame(uint16_t transaction_id,
                                 uint8_t *buffer,
                                 size_t buffer_len) {
  if (buffer == nullptr || buffer_len < TRAFFIC_DNS_TCP_FRAME_SIZE) {
    return 0U;
  }
  buffer[0] = 0U;
  buffer[1] = static_cast<uint8_t>(TRAFFIC_DNS_QUERY_PAYLOAD_SIZE);
  std::memcpy(buffer + 2U, DNS_QUERY_TEMPLATE, sizeof(DNS_QUERY_TEMPLATE));
  buffer[2] = static_cast<uint8_t>(transaction_id >> 8U);
  buffer[3] = static_cast<uint8_t>(transaction_id & 0xFFU);
  return TRAFFIC_DNS_TCP_FRAME_SIZE;
}

void TrafficGeneratorManager::init(uint32_t target_pps, TrafficGeneratorMode mode) {
  task_handle_ = nullptr;
  sock_ = -1;
  gateway_addr_ = 0U;
  mode_ = mode;
  target_pps_ = target_pps;
  current_rate_pps_.store(target_pps, std::memory_order_relaxed);
  running_.store(false, std::memory_order_relaxed);
  paused_.store(false, std::memory_order_relaxed);
  task_exited_.store(true, std::memory_order_relaxed);
  reset_runtime_state_();

  ESP_LOGD(TAG,
           "Traffic generator initialized (target=%" PRIu32 " CSI pps, mode=%s)",
           target_pps,
           traffic_mode_name(mode));
}

bool TrafficGeneratorManager::start(uint32_t gateway_addr) {
  if (running_.load(std::memory_order_relaxed)) {
    return true;
  }
  if (target_pps_ == 0U || gateway_addr == 0U) {
    ESP_LOGE(TAG, "Gateway IP is unavailable in the connection event");
    return false;
  }
  gateway_addr_ = gateway_addr;

  DnsTrafficProtocol dns_protocol;
  IcmpTrafficProtocol icmp_protocol(static_cast<uint16_t>(reinterpret_cast<uintptr_t>(this)));
  const TrafficProtocol &protocol = mode_ == TrafficGeneratorMode::PING
                                        ? static_cast<const TrafficProtocol &>(icmp_protocol)
                                        : static_cast<const TrafficProtocol &>(dns_protocol);
  sock_ = create_protocol_socket(protocol);
  if (sock_ < 0) {
    return false;
  }

  current_rate_pps_.store(target_pps_, std::memory_order_relaxed);
  reset_runtime_state_();
  running_.store(true, std::memory_order_relaxed);
  task_exited_.store(false, std::memory_order_relaxed);
  const BaseType_t result = xTaskCreate(traffic_task_, "traffic_gen", 3072, this, 5, &task_handle_);
  if (result != pdPASS) {
    running_.store(false, std::memory_order_relaxed);
    task_exited_.store(true, std::memory_order_relaxed);
    close(sock_);
    sock_ = -1;
    ESP_LOGE(TAG, "Failed to create traffic generator task (result=%d)", static_cast<int>(result));
    return false;
  }

  char gateway[16];
  const esp_ip4_addr_t gateway_ip{gateway_addr_};
  snprintf(gateway, sizeof(gateway), IPSTR, IP2STR(&gateway_ip));
  ESP_LOGI(TAG,
           "Traffic generator started (mode=%s, target=%" PRIu32 " CSI pps, send=%" PRIu32
           " pps, gateway=%s)",
           traffic_mode_name(mode_),
           target_pps_,
           current_rate_pps(),
           gateway);
  return true;
}

void TrafficGeneratorManager::loop() {
  if (!is_running() || is_paused()) {
    return;
  }

  const int64_t now = esp_timer_get_time();
  if (last_health_check_us_ != 0 && now - last_health_check_us_ < HEALTH_CHECK_INTERVAL_US) {
    return;
  }
  last_health_check_us_ = now;
  const uint32_t successes = send_success_count();
  if (successes != previous_send_success_count_) {
    previous_send_success_count_ = successes;
    last_send_progress_us_ = now;
  } else if (last_send_progress_us_ != 0 && now - last_send_progress_us_ >= SEND_STALL_TIMEOUT_US) {
    ESP_LOGW(TAG, "Traffic generator has not sent a packet for %.1f s",
             static_cast<double>(now - last_send_progress_us_) / 1000000.0);
    last_send_progress_us_ = now;
  }
}

void TrafficGeneratorManager::pause() {
  paused_.store(true, std::memory_order_relaxed);
}

void TrafficGeneratorManager::resume() {
  paused_.store(false, std::memory_order_relaxed);
}

void TrafficGeneratorManager::stop() {
  if (!is_running()) {
    return;
  }
  running_.store(false, std::memory_order_relaxed);
  for (int attempt = 0; attempt < 20 && !task_exited_.load(std::memory_order_relaxed); ++attempt) {
    vTaskDelay(pdMS_TO_TICKS(100));
  }
  if (!task_exited_.load(std::memory_order_relaxed)) {
    ESP_LOGW(TAG, "Traffic generator task did not exit within 2 s");
  }
  ESP_LOGI(TAG, "Traffic generator stopped");
}

void TrafficGeneratorManager::traffic_task_(void *arg) {
  auto *manager = static_cast<TrafficGeneratorManager *>(arg);
  if (manager == nullptr) {
    vTaskDelete(nullptr);
    return;
  }

  DnsTrafficProtocol dns_protocol;
  IcmpTrafficProtocol icmp_protocol(static_cast<uint16_t>(reinterpret_cast<uintptr_t>(manager)));
  TrafficProtocol *protocol = manager->mode_ == TrafficGeneratorMode::PING
                                  ? static_cast<TrafficProtocol *>(&icmp_protocol)
                                  : static_cast<TrafficProtocol *>(&dns_protocol);
  sockaddr_in destination{};
  destination.sin_family = AF_INET;
  destination.sin_port = htons(protocol->destination_port());
  destination.sin_addr.s_addr = manager->gateway_addr_;

  SendErrorState error_state;
  uint32_t consecutive_errors = 0U;
  int64_t next_send_deadline_us = 0;
  TcpConnectionState connection_state = protocol->connection_oriented()
                                            ? TcpConnectionState::DISCONNECTED
                                            : TcpConnectionState::CONNECTED;
  int64_t next_connect_attempt_us = 0;
  constexpr int64_t tcp_reconnect_delay_us = 1000000LL;

  const auto recreate_socket = [&]() {
    if (manager->sock_ >= 0) {
      close(manager->sock_);
    }
    manager->sock_ = create_protocol_socket(*protocol);
    connection_state = protocol->connection_oriented()
                           ? TcpConnectionState::DISCONNECTED
                           : TcpConnectionState::CONNECTED;
    next_connect_attempt_us = esp_timer_get_time() + tcp_reconnect_delay_us;
    next_send_deadline_us = 0;
    return manager->sock_ >= 0;
  };

  while (manager->running_.load(std::memory_order_relaxed)) {
    if (manager->paused_.load(std::memory_order_relaxed)) {
      vTaskDelay(pdMS_TO_TICKS(50));
      continue;
    }

    if (protocol->connection_oriented() && connection_state != TcpConnectionState::CONNECTED) {
      const int64_t now_us = esp_timer_get_time();
      if (manager->sock_ < 0) {
        if (now_us >= next_connect_attempt_us) {
          (void)recreate_socket();
        }
      } else if (connection_state == TcpConnectionState::DISCONNECTED &&
                 now_us >= next_connect_attempt_us) {
        connection_state = start_tcp_connect(manager->sock_, destination);
        if (connection_state == TcpConnectionState::DISCONNECTED) {
          (void)recreate_socket();
        }
      } else if (connection_state == TcpConnectionState::CONNECTING) {
        connection_state = poll_tcp_connect(manager->sock_);
        if (connection_state == TcpConnectionState::DISCONNECTED) {
          (void)recreate_socket();
        } else if (connection_state == TcpConnectionState::CONNECTED) {
          ESP_LOGI(TAG, "%s TCP connection established", protocol->name());
          next_send_deadline_us = 0;
        }
      }
      if (connection_state != TcpConnectionState::CONNECTED) {
        vTaskDelay(pdMS_TO_TICKS(10));
        continue;
      }
    }

    const SocketDrainResult drain_result = drain_socket(manager->sock_);
    if (protocol->connection_oriented() && drain_result != SocketDrainResult::READY) {
      ESP_LOGW(TAG, "%s TCP connection closed while draining responses", protocol->name());
      (void)recreate_socket();
      continue;
    }
    const int64_t send_started_us = esp_timer_get_time();
    const ssize_t sent = protocol->send_packet(manager->sock_, destination);
    if (sent <= 0) {
      manager->send_error_count_.fetch_add(1U, std::memory_order_relaxed);
      consecutive_errors++;
      const int current_errno = errno;
      const int64_t now_us = esp_timer_get_time();
      const bool should_log = now_us - error_state.last_log_time > SendErrorState::LOG_INTERVAL_US;
      const bool needs_backoff = handle_send_error(error_state, sent, current_errno, now_us);
      if (should_log) {
        ESP_LOGW(TAG,
                 "%s send failed (errno=%d, consecutive=%" PRIu32 ")",
                 protocol->name(),
                 current_errno,
                 consecutive_errors);
      }
      const bool transient_error = current_errno == EAGAIN || current_errno == EWOULDBLOCK ||
                                   current_errno == ENOMEM;
      if ((protocol->connection_oriented() && !transient_error) ||
          consecutive_errors >= CONSECUTIVE_ERROR_REOPEN_THRESHOLD) {
        (void)recreate_socket();
        consecutive_errors = 0U;
        if (manager->sock_ < 0) {
          vTaskDelay(pdMS_TO_TICKS(100));
        }
        continue;
      }
      if (needs_backoff) {
        vTaskDelay(pdMS_TO_TICKS(5));
      }
    } else {
      manager->send_success_count_.fetch_add(1U, std::memory_order_relaxed);
      consecutive_errors = 0U;
    }

    const uint32_t rate_pps =
        std::max<uint32_t>(manager->current_rate_pps_.load(std::memory_order_relaxed), 1U);
    const int64_t interval_us = 1000000LL / static_cast<int64_t>(rate_pps);
    // Keep the nominal phase across ordinary scheduler jitter, but reset it
    // whenever recovery would place the next send less than half a period
    // away. This preserves the average cadence without catch-up bursts.
    next_send_deadline_us = next_traffic_send_deadline_us(
        next_send_deadline_us, send_started_us, interval_us);
    const int64_t now_us = esp_timer_get_time();
    const int64_t sleep_us = next_send_deadline_us - now_us;
    if (sleep_us > 0) {
      const TickType_t ticks = pdMS_TO_TICKS((sleep_us + 999LL) / 1000LL);
      if (ticks > 0) {
        vTaskDelay(ticks);
      }
    }
  }

  if (manager->sock_ >= 0) {
    close(manager->sock_);
    manager->sock_ = -1;
  }
  manager->task_handle_ = nullptr;
  manager->task_exited_.store(true, std::memory_order_relaxed);
  vTaskDelete(nullptr);
}

void TrafficGeneratorManager::reset_runtime_state_() {
  send_success_count_.store(0U, std::memory_order_relaxed);
  send_error_count_.store(0U, std::memory_order_relaxed);
  previous_send_success_count_ = 0U;
  last_send_progress_us_ = esp_timer_get_time();
  last_health_check_us_ = 0;
}

}  // namespace espectre
