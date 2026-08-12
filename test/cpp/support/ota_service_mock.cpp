/*
 * ESPectre - OTA Service Mock
 *
 * Test double for the OTA service boundary used by native frontend tests.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "ota_service_mock.h"

namespace espectre {
namespace ota_service_mock {

State state{};

void reset() { state = State{}; }

void MockOtaService::loop() { state.loop_calls += 1; }

void MockOtaService::shutdown() { state.shutdown_called = true; }

bool MockOtaService::start_check(const std::string &current_version) {
  state.start_check_calls += 1;
  state.last_current_version = current_version;
  return state.start_check_result;
}

bool MockOtaService::start_update(const std::string &current_version) {
  state.start_update_calls += 1;
  state.last_current_version = current_version;
  return state.start_update_result;
}

EspectreOtaStatus MockOtaService::status() const { return state.status; }

void MockOtaService::set_status_callback(StatusCallback callback) { state.status_callback = std::move(callback); }

void MockOtaService::set_prepare_for_update_callback(PrepareForUpdateCallback callback) {
  state.prepare_callback = std::move(callback);
}

void MockOtaService::emit_status(const EspectreOtaStatus &status) {
  state.status = status;
  if (state.status_callback) {
    state.status_callback(status);
  }
}

void MockOtaService::emit_prepare() {
  if (state.prepare_callback) {
    state.prepare_callback();
  }
}

}  // namespace ota_service_mock
}  // namespace espectre
