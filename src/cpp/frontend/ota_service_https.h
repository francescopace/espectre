/*
 * ESPectre - HTTPS OTA Service
 *
 * Checks OTA manifests and applies HTTPS firmware updates.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <atomic>
#include <array>
#include <cstdint>
#include <memory>
#include <string>

#include "esp_http_client.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "ota_service.h"

namespace espectre {

enum class OtaReleaseChannel : uint8_t {
  RELEASE = 0,
  PREVIEW,
  DEVELOP,
};

class HttpsOtaService : public IOtaService {
 public:
  HttpsOtaService(const char *frontend, const char *chip, OtaReleaseChannel channel);
  ~HttpsOtaService() override;

  void loop() override;
  void shutdown() override;
  bool start_check(const std::string &current_version) override;
  bool start_check(const std::string &current_version, const std::string &channel) override;
  bool start_update(const std::string &current_version) override;
  bool start_update(const std::string &current_version, const std::string &channel) override;
  EspectreOtaStatus status() const override;
  void set_status_callback(StatusCallback callback) override;
  void set_prepare_for_update_callback(PrepareForUpdateCallback callback) override;

 private:
  struct ManifestBuffer : JsonInput {
    static constexpr size_t kChunkBytes = 1024U;
    static constexpr size_t kMaxBytes = 64U * 1024U;
    std::array<std::unique_ptr<char[]>, kMaxBytes / kChunkBytes> chunks{};
    size_t length{0U};
    const char *error{nullptr};
    size_t size() const override { return length; }
    char operator[](size_t offset) const override { return chunks[offset / kChunkBytes][offset % kChunkBytes]; }
  };

  enum class WorkerAction : uint8_t {
    CHECK = 0,
    START_UPDATE,
  };

  struct WorkerRequest {
    WorkerAction action{WorkerAction::CHECK};
    std::string current_version;
    std::string channel;
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
  void finish_worker_();
  bool request_prepare_for_update_();
  bool begin_request_(const WorkerRequest &request);
  bool ensure_lock_() const;
  void update_status_(const EspectreOtaStatus &status);
  void set_error_status_(const std::string &message,
                         const std::string &current_version,
                         const std::string &target_version,
                         const std::string &manifest_url,
                         const std::string &image_url,
                         const std::string &channel);
  static esp_err_t manifest_http_event_(esp_http_client_event_t *event);
  bool fetch_manifest_(const std::string &url, ManifestBuffer *body, std::string *error) const;
  bool parse_manifest_(const ManifestBuffer &body, const std::string &channel,
                       ManifestInfo *manifest, std::string *error) const;

  mutable SemaphoreHandle_t lock_{nullptr};
  SemaphoreHandle_t worker_done_{nullptr};
  SemaphoreHandle_t prepare_done_{nullptr};
  StatusCallback status_callback_{};
  PrepareForUpdateCallback prepare_for_update_callback_{};
  EspectreOtaStatus status_{};
  EspectreOtaStatus pending_status_{};
  bool worker_active_{false};
  bool status_callback_pending_{false};
  bool prepare_callback_pending_{false};
  std::atomic<bool> shutdown_requested_{false};
  std::string frontend_{};
  std::string chip_{};
  std::string default_channel_{};
};

}  // namespace espectre
