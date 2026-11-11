/*
 * ESPectre - Runtime Guest Access
 *
 * Read-only access to a runtime owned by another part of the firmware.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include "runtime/runtime_events.h"
#include "runtime_frontend_controller.h"

namespace espectre {

/**
 * Lets code that does not own the RuntimeFrontendController follow it.
 *
 * A firmware has one sensing runtime, and one component owns its controller.
 * A second component, the guest, sometimes needs to report that runtime
 * without being handed the controller: an add-on that serves Direct HTTP next
 * to an integration that already owns sensing. The guest gets a read-only
 * view and an event observer, and nothing that changes the runtime. Control
 * changes go back to the owner, for example through the RuntimeControlSink of
 * `RuntimeDirectHttpBridge::setup_guest()`.
 *
 * The first controller whose `setup()` succeeds is the one a guest sees, until
 * that controller is destroyed. It stays visible across `shutdown()` and a
 * later `setup()`; check `is_setup_complete()` on the view before relying on
 * live state.
 *
 * @par Threading
 * Call every method from the task that runs the owner's `loop()`.
 */
class RuntimeGuestAccess {
 public:
  /**
   * Read-only view of the firmware's runtime controller.
   *
   * @return `nullptr` until a controller has completed `setup()`, and again
   *         after that controller is destroyed. Do not keep the pointer
   *         beyond the owner's lifetime.
   */
  static const RuntimeFrontendController *runtime();
  /**
   * Register the guest's event sink; see RuntimeFrontendController::set_observer().
   *
   * @param observer Event sink, or `nullptr` to remove the current one.
   * @return false without a runtime, or when a different observer is already
   *         registered.
   */
  static bool set_observer(IRuntimeListener *observer);

 private:
  friend class RuntimeFrontendController;
  friend class RuntimeDirectHttpBridge;

  static void offer_(RuntimeFrontendController *controller);
  static void release_(const RuntimeFrontendController *controller);
  static RuntimeFrontendController *mutable_runtime_();
};

}  // namespace espectre
