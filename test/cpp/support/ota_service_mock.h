#pragma once

#include <string>

#include "ota_service.h"

namespace espectre {
namespace ota_service_mock {

struct State {
  bool start_check_result{true};
  bool start_update_result{true};
  bool shutdown_called{false};
  int loop_calls{0};
  int start_check_calls{0};
  int start_update_calls{0};
  std::string last_current_version;
  EspectreOtaStatus status{};
  IOtaService::StatusCallback status_callback;
  IOtaService::PrepareForUpdateCallback prepare_callback;
};

extern State state;

void reset();

class MockOtaService : public IOtaService {
 public:
  void loop() override;
  void shutdown() override;
  bool start_check(const std::string &current_version) override;
  bool start_update(const std::string &current_version) override;
  EspectreOtaStatus status() const override;
  void set_status_callback(StatusCallback callback) override;
  void set_prepare_for_update_callback(PrepareForUpdateCallback callback) override;

  void emit_status(const EspectreOtaStatus &status);
  void emit_prepare();
};

}  // namespace ota_service_mock
}  // namespace espectre
