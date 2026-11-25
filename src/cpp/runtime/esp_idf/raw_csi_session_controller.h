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
#include <memory>
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
 *
 * @par Lifetime
 * The destructor detaches from the service and the runtime, so both must
 * still exist when the controller is destroyed. In an owning class, declare
 * the controller after them.
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
   * Neither object is owned; both must outlive the controller, including its
   * destructor, which detaches from them. `device_id`
   * must be nonzero and `chip` is the `CONFIG_IDF_TARGET` name; both are
   * stamped on every record. Reconfiguring closes the previous binding.
   */
  void configure(IDirectHttpService *service,
                 RuntimeFrontendController *runtime,
                 uint64_t device_id,
                 std::string chip,
                 StoppedCallback stopped_callback = {},
                 StartedCallback started_callback = {});
  /** After configure(), supply a predicate that blocks collection during a Wi-Fi change. */
  void set_wifi_update_pending_callback(std::function<bool()> callback);
  /**
   * Start a session.
   *
   * @return false, with a reason in `message`, when raw CSI is unavailable, a
   *         session is already active, a Wi-Fi change is pending, sensing is
   *         paused, or either side fails to start.
   */
  bool begin(std::string *message = nullptr);
  /** End the session if the runtime left raw collection on its own. Call it from the loop. */
  void ensure_runtime_consistency();
  /** Stop owned capture and keep the service bound for subsequent requests. */
  void shutdown(RawCsiStopReason reason = RawCsiStopReason::SHUTDOWN);
  /**
   * Stop capture and unregister the service. A pending stop is reported with the
   * transport's reason when it dispatches its callback, while the controller
   * still exists. Observer captures must remain valid until that delivery.
   */
  void detach();
  /** Whether a session is active. */
  bool active() const;

 private:
  struct Binding : std::enable_shared_from_this<Binding> {
    IDirectHttpService *service{nullptr};
    RuntimeFrontendController *runtime{nullptr};
    uint64_t device_id{0U};
    std::string chip;
    std::shared_ptr<StoppedCallback> stopped_callback;
    StartedCallback started_callback;
    std::function<bool()> wifi_update_pending;
    struct Session {
      RuntimeFrontendController *runtime;
      std::shared_ptr<StoppedCallback> stopped_callback;
      void stop_capture(RawCsiStopReason reason);
    };
    std::shared_ptr<Session> session;
    std::weak_ptr<void> observer_lifetime;

    ~Binding();
    bool begin(std::string *message);
    void stop(IDirectHttpService &transport, RawCsiStopReason reason);
    void detach();
  };
  std::shared_ptr<Binding> binding_;
  // Destroy first so deferred transport completions cannot call a dead observer.
  std::shared_ptr<void> observer_lifetime_;
};

}  // namespace espectre
