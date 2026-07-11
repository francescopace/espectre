#include "ota_service_mock.h"

namespace espectre {
namespace ota_service_mock {

State state{};

void reset() { state = State{}; }

void MockOtaService::loop() { state.loop_calls += 1; }

void MockOtaService::shutdown() { state.shutdown_called = true; }

bool MockOtaService::start_check(const std::string &manifest_url, const std::string &current_version) {
  state.start_check_calls += 1;
  state.last_manifest_url = manifest_url;
  state.last_current_version = current_version;
  return state.start_check_result;
}

bool MockOtaService::start_update(const std::string &manifest_url,
                                  const std::string &image_url,
                                  const std::string &target_version,
                                  const std::string &current_version) {
  state.start_update_calls += 1;
  state.last_manifest_url = manifest_url;
  state.last_image_url = image_url;
  state.last_target_version = target_version;
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
