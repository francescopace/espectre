/*
 * ESPectre - Runtime Guest Access
 *
 * Read-only access to a runtime owned by another part of the firmware.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "runtime_guest_access.h"

namespace espectre {

namespace {

// Written and read on the owner task only; see the class threading contract.
RuntimeFrontendController *guest_runtime = nullptr;

}  // namespace

const RuntimeFrontendController *RuntimeGuestAccess::runtime() { return guest_runtime; }

bool RuntimeGuestAccess::set_observer(IRuntimeListener *observer) {
  return guest_runtime != nullptr && guest_runtime->set_observer(observer);
}

void RuntimeGuestAccess::offer_(RuntimeFrontendController *controller) {
  if (guest_runtime == nullptr) {
    guest_runtime = controller;
  }
}

void RuntimeGuestAccess::release_(const RuntimeFrontendController *controller) {
  if (guest_runtime == controller) {
    guest_runtime = nullptr;
  }
}

RuntimeFrontendController *RuntimeGuestAccess::mutable_runtime_() { return guest_runtime; }

}  // namespace espectre
