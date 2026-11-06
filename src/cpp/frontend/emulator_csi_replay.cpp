/*
 * ESPectre - Emulator CSI Replay
 *
 * Delivers recorded CSI from UART1 to the firmware's CSI callback when the
 * firmware runs in esp-emu, which models Wi-Fi but not CSI.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "emulator_csi_replay.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <memory>
#include <mutex>

#include "csi_replay_frame.h"
#include "driver/uart.h"
#include "esp_log.h"
#include "esp_mac.h"
#include "esp_netif.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

namespace espectre {

namespace {

constexpr const char *TAG = "EmulatorCsiReplay";
constexpr uart_port_t kReplayUart = UART_NUM_1;
constexpr int kReplayBaud = 921600;
constexpr int kRxBufferBytes = 4096;
constexpr size_t kReadChunkBytes = 256U;
// Frames held behind the pacer: the jitter buffer plus a stall of a few hundred milliseconds.
constexpr size_t kPendingFrames = 48U;
constexpr int64_t kIdleWaitUs = 20000;

struct ReplayState {
  std::mutex lock;
  wifi_csi_cb_t callback{nullptr};
  void *callback_context{nullptr};
  bool enabled{false};
  TaskHandle_t task{nullptr};
  CsiReplayPacket packet{};
  uint16_t sequence{0U};
};

ReplayState &state() {
  static ReplayState replay_state;
  return replay_state;
}

bool read_identity(CsiReplayIdentity *identity) {
  esp_netif_t *netif = esp_netif_get_handle_from_ifkey("WIFI_STA_DEF");
  esp_netif_ip_info_t ip_info{};
  if (netif == nullptr || esp_netif_get_ip_info(netif, &ip_info) != ESP_OK ||
      ip_info.ip.addr == 0U || esp_wifi_get_mac(WIFI_IF_STA, identity->local_mac_addr) != ESP_OK) {
    return false;
  }
  identity->local_ip_addr = ip_info.ip.addr;
  identity->peer_ip_addr = ip_info.gw.addr;
  return true;
}

void deliver(const CsiReplayFrame &frame) {
  CsiReplayIdentity identity;
  if (!read_identity(&identity)) return;
  uint8_t primary = 0U;
  wifi_second_chan_t second = WIFI_SECOND_CHAN_NONE;
  (void)esp_wifi_get_channel(&primary, &second);

  ReplayState &replay = state();
  std::lock_guard<std::mutex> guard(replay.lock);
  if (!replay.enabled || replay.callback == nullptr ||
      !build_csi_replay_packet(frame, identity, primary, replay.sequence, &replay.packet)) {
    return;
  }
  replay.sequence++;
  replay.callback(replay.callback_context, &replay.packet.info);
}

struct PendingFrame {
  CsiReplayFrame frame;
  int64_t due_us{0};
};

class PendingFrames {
 public:
  bool empty() const { return count_ == 0U; }
  bool full() const { return count_ == kPendingFrames; }
  PendingFrame &front() { return frames_[head_]; }
  void pop() {
    head_ = (head_ + 1U) % kPendingFrames;
    count_--;
  }
  void push(const CsiReplayFrame &frame, int64_t due_us) {
    PendingFrame &slot = frames_[(head_ + count_) % kPendingFrames];
    slot.frame = frame;
    slot.due_us = due_us;
    count_++;
  }

 private:
  std::array<PendingFrame, kPendingFrames> frames_{};
  size_t head_{0U};
  size_t count_{0U};
};

void release_front(PendingFrames &pending, CsiReplayPacer &pacer, int64_t release_us, int64_t now_us) {
  deliver(pending.front().frame);
  pacer.mark_released(release_us, now_us);
  pending.pop();
}

void run(void *) {
  // Heap, not static: only an emulator run allocates the frame buffer.
  std::unique_ptr<PendingFrames> pending(new PendingFrames());
  CsiReplayPacer pacer;
  CsiReplayFrameParser parser;
  CsiReplayFrame frame;
  std::array<uint8_t, kReadChunkBytes> chunk{};
  for (;;) {
    int64_t now_us = esp_timer_get_time();
    while (!pending->empty()) {
      const int64_t release_us = pacer.release_at(pending->front().due_us);
      if (release_us > now_us) break;
      release_front(*pending, pacer, release_us, now_us);
    }
    const int64_t wait_us =
        pending->empty() ? kIdleWaitUs
                         : std::min(kIdleWaitUs, pacer.release_at(pending->front().due_us) - now_us);
    // Wait at least one tick so this task never starves the firmware loop.
    const TickType_t wait_ticks = std::max<TickType_t>(1U, pdMS_TO_TICKS(wait_us / 1000));
    const int read = uart_read_bytes(kReplayUart, chunk.data(), chunk.size(), wait_ticks);
    now_us = esp_timer_get_time();
    for (int i = 0; i < read; ++i) {
      if (!parser.push(chunk[static_cast<size_t>(i)], &frame)) continue;
      if (pending->full()) {
        // A stall longer than the buffer: release the oldest frame now rather than lose it.
        release_front(*pending, pacer, now_us, now_us);
      }
      pending->push(frame, pacer.due(frame.rx_timestamp_us, now_us));
    }
  }
}

esp_err_t start_locked(ReplayState &replay) {
  if (replay.task != nullptr) return ESP_OK;
  const uart_config_t uart_config = {
      .baud_rate = kReplayBaud,
      .data_bits = UART_DATA_8_BITS,
      .parity = UART_PARITY_DISABLE,
      .stop_bits = UART_STOP_BITS_1,
      .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
      .rx_flow_ctrl_thresh = 0,
      .source_clk = UART_SCLK_DEFAULT,
      .flags = {},
  };
  esp_err_t err = uart_driver_install(kReplayUart, kRxBufferBytes, 0, 0, nullptr, 0);
  if (err == ESP_OK) err = uart_param_config(kReplayUart, &uart_config);
  if (err != ESP_OK) {
    ESP_LOGE(TAG, "UART1 setup failed: %s", esp_err_to_name(err));
    return err;
  }
  if (xTaskCreate(&run, "csi_replay", 4096, nullptr, 5, &replay.task) != pdPASS) {
    replay.task = nullptr;
    return ESP_ERR_NO_MEM;
  }
  return ESP_OK;
}

bool detect_emulator() {
  uint8_t factory_mac[6]{};
  if (esp_efuse_mac_get_default(factory_mac) != ESP_OK || !is_emulator_replay_mac(factory_mac)) {
    return false;
  }
  ESP_LOGW(TAG, "esp-emu replay marker in the factory MAC: CSI comes from UART1");
  return true;
}

}  // namespace

bool emulator_csi_replay_active() {
  static const bool active = detect_emulator();
  return active;
}

}  // namespace espectre


extern "C" {

esp_err_t __real_esp_wifi_set_csi_config(const wifi_csi_config_t *config);
esp_err_t __real_esp_wifi_set_csi_rx_cb(wifi_csi_cb_t cb, void *ctx);
esp_err_t __real_esp_wifi_set_csi(bool en);

esp_err_t __wrap_esp_wifi_set_csi_config(const wifi_csi_config_t *config) {
  if (!espectre::emulator_csi_replay_active()) return __real_esp_wifi_set_csi_config(config);
  return config == nullptr ? ESP_ERR_INVALID_ARG : ESP_OK;
}

esp_err_t __wrap_esp_wifi_set_csi_rx_cb(wifi_csi_cb_t cb, void *ctx) {
  if (!espectre::emulator_csi_replay_active()) return __real_esp_wifi_set_csi_rx_cb(cb, ctx);
  auto &replay = espectre::state();
  std::lock_guard<std::mutex> guard(replay.lock);
  replay.callback = cb;
  replay.callback_context = ctx;
  return ESP_OK;
}

esp_err_t __wrap_esp_wifi_set_csi(bool en) {
  if (!espectre::emulator_csi_replay_active()) return __real_esp_wifi_set_csi(en);
  auto &replay = espectre::state();
  std::lock_guard<std::mutex> guard(replay.lock);
  replay.enabled = en;
  return en ? espectre::start_locked(replay) : ESP_OK;
}

}  // extern "C"
