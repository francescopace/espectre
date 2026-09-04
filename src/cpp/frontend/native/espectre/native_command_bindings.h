/*
 * ESPectre - Native Command Bindings
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <cstdint>
#include <string>

#include "frontend_command_engine.h"

namespace espectre {

class NativeFrontend;

class NativeCommandBindings {
 public:
  explicit NativeCommandBindings(NativeFrontend &owner) : owner_(owner) {}

  FrontendCommandResult execute(const EspectreCommand &command, FrontendCommandOrigin origin, bool allow_local_config,
                                uint64_t connection_token = 0U);
  EspectreCapabilityProfile capability_profile(bool allow_local_config) const;

 private:
  NativeFrontend &owner_;
  FrontendCommandEngine engine_;
};

}  // namespace espectre
