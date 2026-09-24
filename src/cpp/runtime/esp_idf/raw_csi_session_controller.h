/*
 * ESPectre - Raw CSI Session Controller
 *
 * Shared owner-bound raw collection orchestration for Direct frontends.
 *
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <cstdint>
#include <functional>
#include <string>

#include "runtime/direct_http_service.h"
#include "runtime_frontend_controller.h"

namespace espectre {

/**
 * Connects a Direct raw session to the runtime's raw collection.
 *
 * Once configured, a `GET /csi` request on the Direct service starts raw
 * collection through the controller and streams every packet to the client;
 * ending either side stops the other.
 *
 * @par Threading
 * Call every method from the task that owns the runtime and the service.
 */
class RawCsiSessionController {
 public:
  /** Reports why the session ended, after the runtime left raw collection. */
  using StoppedCallback = std::function<void(RawCsiStopReason reason)>;
  /** Reports that a session started. */
  using StartedCallback = std::function<void()>;

  /**
   * Bind the service and runtime, and register the `GET /csi` handler.
   *
   * Neither object is owned; both must outlive the controller. `device_id`
   * must be nonzero and `chip` is the `CONFIG_IDF_TARGET` name; both are
   * stamped on every record.
   */
  void configure(IDirectHttpService *service,
                 RuntimeFrontendController *runtime,
                 uint64_t device_id,
                 std::string chip,
                 StoppedCallback stopped_callback = {},
                 StartedCallback started_callback = {});
  /**
   * Start a session.
   *
   * @return false, with a reason in `message`, when raw CSI is unavailable, a
   *         session is already active, or either side fails to start.
   */
  bool begin(std::string *message = nullptr);
  /** End the session if the runtime left raw collection on its own. Call it from the loop. */
  void ensure_runtime_consistency();
  /** End the active session, if any. */
  void shutdown(RawCsiStopReason reason = RawCsiStopReason::SHUTDOWN);
  /** Whether a session is active. */
  bool active() const { return active_; }

 private:
  static bool offer_packet_(void *context, const RawCsiPacketView &packet);
  void handle_stopped_(RawCsiStopReason reason);

  IDirectHttpService *service_{nullptr};
  RuntimeFrontendController *runtime_{nullptr};
  uint64_t device_id_{0U};
  std::string chip_;
  bool active_{false};
  StoppedCallback stopped_callback_{};
  StartedCallback started_callback_{};
};

}  // namespace espectre
