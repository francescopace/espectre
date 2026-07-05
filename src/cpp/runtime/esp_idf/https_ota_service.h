#pragma once

#include <string>

#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "ota_service.h"

namespace esphome {
namespace espectre {

class HttpsOtaService : public IOtaService {
 public:
  HttpsOtaService() = default;
  ~HttpsOtaService() override;

  void loop() override {}
  void shutdown() override;
  bool start_check(const std::string &manifest_url, const std::string &current_version) override;
  bool start_update(const std::string &manifest_url,
                    const std::string &image_url,
                    const std::string &target_version,
                    const std::string &current_version) override;
  EspectreOtaStatus status() const override;
  void set_status_callback(StatusCallback callback) override;
  void set_prepare_for_update_callback(PrepareForUpdateCallback callback) override;

 private:
  enum class WorkerAction : uint8_t {
    CHECK = 0,
    START_UPDATE,
  };

  struct WorkerRequest {
    WorkerAction action{WorkerAction::CHECK};
    std::string manifest_url;
    std::string image_url;
    std::string target_version;
    std::string current_version;
  };

  struct ManifestInfo {
    std::string version;
    std::string image_url;
  };

  struct WorkerContext {
    HttpsOtaService *service{nullptr};
    WorkerRequest request{};
  };

  static void worker_entry_(void *ctx);
  void run_worker_(const WorkerRequest &request);
  bool begin_request_(const WorkerRequest &request);
  bool ensure_lock_() const;
  void update_status_(const EspectreOtaStatus &status);
  void set_error_status_(const std::string &message,
                         const std::string &current_version,
                         const std::string &target_version,
                         const std::string &manifest_url,
                         const std::string &image_url);
  bool fetch_https_text_(const std::string &url, std::string *body, std::string *error) const;
  bool parse_manifest_(const std::string &body, ManifestInfo *manifest, std::string *error) const;

  mutable SemaphoreHandle_t lock_{nullptr};
  TaskHandle_t worker_task_{nullptr};
  StatusCallback status_callback_{};
  PrepareForUpdateCallback prepare_for_update_callback_{};
  EspectreOtaStatus status_{};
};

}  // namespace espectre
}  // namespace esphome
