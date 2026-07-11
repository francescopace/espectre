#include "ota_service_https.h"

#include <algorithm>
#include <memory>
#include <utility>

#include "esp_crt_bundle.h"
#include "esp_err.h"
#include "esp_http_client.h"
#include "esp_https_ota.h"
#include "esp_log.h"
#include "esp_system.h"
#include "protocol_json.h"

namespace espectre {

namespace {

static const char *const TAG = "espectre.ota";
constexpr size_t kMaxManifestBytes = 4096U;
constexpr uint32_t kHttpTimeoutMs = 30000U;
constexpr uint32_t kPostSuccessDelayMs = 500U;
constexpr uint32_t kWorkerStackSize = 8192U;
constexpr UBaseType_t kWorkerPriority = 5U;

}  // namespace

HttpsOtaService::~HttpsOtaService() { shutdown(); }

void HttpsOtaService::shutdown() {
  if (worker_task_ != nullptr) {
    vTaskDelete(worker_task_);
    worker_task_ = nullptr;
  }
  if (lock_ != nullptr) {
    vSemaphoreDelete(lock_);
    lock_ = nullptr;
  }
}

bool HttpsOtaService::start_check(const std::string &manifest_url, const std::string &current_version) {
  WorkerRequest request;
  request.action = WorkerAction::CHECK;
  request.manifest_url = manifest_url;
  request.current_version = current_version;
  return begin_request_(request);
}

bool HttpsOtaService::start_update(const std::string &manifest_url,
                                   const std::string &image_url,
                                   const std::string &target_version,
                                   const std::string &current_version) {
  WorkerRequest request;
  request.action = WorkerAction::START_UPDATE;
  request.manifest_url = manifest_url;
  request.image_url = image_url;
  request.target_version = target_version;
  request.current_version = current_version;
  return begin_request_(request);
}

EspectreOtaStatus HttpsOtaService::status() const {
  if (!ensure_lock_()) {
    return status_;
  }
  xSemaphoreTake(lock_, portMAX_DELAY);
  const EspectreOtaStatus snapshot = status_;
  xSemaphoreGive(lock_);
  return snapshot;
}

void HttpsOtaService::set_status_callback(StatusCallback callback) { status_callback_ = std::move(callback); }

void HttpsOtaService::set_prepare_for_update_callback(PrepareForUpdateCallback callback) {
  prepare_for_update_callback_ = std::move(callback);
}

void HttpsOtaService::worker_entry_(void *ctx) {
  std::unique_ptr<WorkerContext> context(static_cast<WorkerContext *>(ctx));
  if (context != nullptr && context->service != nullptr) {
    context->service->run_worker_(context->request);
  }
  vTaskDelete(nullptr);
}

void HttpsOtaService::run_worker_(const WorkerRequest &request) {
  const std::string current_version = request.current_version.empty() ? "unknown" : request.current_version;
  ManifestInfo manifest;

  if (request.action == WorkerAction::CHECK ||
      (request.action == WorkerAction::START_UPDATE && !request.manifest_url.empty() && request.image_url.empty())) {
    EspectreOtaStatus checking;
    checking.state = EspectreOtaState::CHECKING;
    checking.busy = true;
    checking.current_version = current_version;
    checking.manifest_url = request.manifest_url;
    checking.target_version = request.target_version;
    update_status_(checking);

    std::string body;
    std::string error;
    if (!fetch_https_text_(request.manifest_url, &body, &error) || !parse_manifest_(body, &manifest, &error)) {
      set_error_status_(error.empty() ? "manifest fetch failed" : error,
                        current_version,
                        request.target_version,
                        request.manifest_url,
                        request.image_url);
      worker_task_ = nullptr;
      return;
    }

    if (request.action == WorkerAction::CHECK) {
      EspectreOtaStatus result;
      result.current_version = current_version;
      result.target_version = manifest.version;
      result.manifest_url = request.manifest_url;
      result.image_url = manifest.image_url;
      result.update_available = manifest.version != current_version && !manifest.version.empty();
      result.busy = false;
      result.state = result.update_available ? EspectreOtaState::UPDATE_AVAILABLE : EspectreOtaState::UP_TO_DATE;
      result.message = result.update_available ? "update available" : "already up to date";
      update_status_(result);
      worker_task_ = nullptr;
      return;
    }
  }

  const std::string image_url = request.image_url.empty() ? manifest.image_url : request.image_url;
  const std::string target_version = request.target_version.empty() ? manifest.version : request.target_version;
  if (image_url.empty()) {
    set_error_status_("missing image_url", current_version, target_version, request.manifest_url, image_url);
    worker_task_ = nullptr;
    return;
  }

  if (prepare_for_update_callback_) {
    prepare_for_update_callback_();
  }

  EspectreOtaStatus downloading;
  downloading.state = EspectreOtaState::DOWNLOADING;
  downloading.busy = true;
  downloading.current_version = current_version;
  downloading.target_version = target_version;
  downloading.manifest_url = request.manifest_url;
  downloading.image_url = image_url;
  downloading.update_available = !target_version.empty() && target_version != current_version;
  downloading.message = "starting https ota";
  update_status_(downloading);

  esp_http_client_config_t http_config{};
  http_config.url = image_url.c_str();
  http_config.timeout_ms = static_cast<int>(kHttpTimeoutMs);
  http_config.crt_bundle_attach = esp_crt_bundle_attach;

  esp_https_ota_config_t ota_config{};
  ota_config.http_config = &http_config;

  const esp_err_t err = esp_https_ota(&ota_config);
  if (err != ESP_OK) {
    set_error_status_(esp_err_to_name(err), current_version, target_version, request.manifest_url, image_url);
    worker_task_ = nullptr;
    return;
  }

  EspectreOtaStatus ready;
  ready.state = EspectreOtaState::REBOOT_SCHEDULED;
  ready.busy = false;
  ready.current_version = current_version;
  ready.target_version = target_version;
  ready.manifest_url = request.manifest_url;
  ready.image_url = image_url;
  ready.update_available = false;
  ready.message = "ota applied, rebooting";
  update_status_(ready);

  vTaskDelay(pdMS_TO_TICKS(kPostSuccessDelayMs));
  esp_restart();
}

bool HttpsOtaService::begin_request_(const WorkerRequest &request) {
  if (!ensure_lock_()) {
    return false;
  }

  xSemaphoreTake(lock_, portMAX_DELAY);
  const bool busy = worker_task_ != nullptr || status_.busy;
  xSemaphoreGive(lock_);
  if (busy) {
    return false;
  }

  auto *context = new WorkerContext{this, request};
  if (context == nullptr) {
    return false;
  }

  if (xTaskCreate(&HttpsOtaService::worker_entry_,
                  "espectre_ota",
                  kWorkerStackSize,
                  context,
                  kWorkerPriority,
                  &worker_task_) != pdPASS) {
    delete context;
    return false;
  }
  return true;
}

bool HttpsOtaService::ensure_lock_() const {
  if (lock_ == nullptr) {
    lock_ = xSemaphoreCreateMutex();
  }
  return lock_ != nullptr;
}

void HttpsOtaService::update_status_(const EspectreOtaStatus &status) {
  if (ensure_lock_()) {
    xSemaphoreTake(lock_, portMAX_DELAY);
    status_ = status;
    xSemaphoreGive(lock_);
  } else {
    status_ = status;
  }
  if (status_callback_) {
    status_callback_(status);
  }
}

void HttpsOtaService::set_error_status_(const std::string &message,
                                        const std::string &current_version,
                                        const std::string &target_version,
                                        const std::string &manifest_url,
                                        const std::string &image_url) {
  EspectreOtaStatus status;
  status.state = EspectreOtaState::ERROR;
  status.busy = false;
  status.current_version = current_version;
  status.target_version = target_version;
  status.manifest_url = manifest_url;
  status.image_url = image_url;
  status.message = message;
  status.update_available = false;
  update_status_(status);
}

bool HttpsOtaService::fetch_https_text_(const std::string &url, std::string *body, std::string *error) const {
  if (body == nullptr) {
    return false;
  }
  body->clear();
  if (url.empty()) {
    if (error != nullptr) {
      *error = "empty url";
    }
    return false;
  }

  esp_http_client_config_t config{};
  config.url = url.c_str();
  config.timeout_ms = static_cast<int>(kHttpTimeoutMs);
  config.crt_bundle_attach = esp_crt_bundle_attach;

  esp_http_client_handle_t client = esp_http_client_init(&config);
  if (client == nullptr) {
    if (error != nullptr) {
      *error = "esp_http_client_init failed";
    }
    return false;
  }

  esp_err_t err = esp_http_client_open(client, 0);
  if (err != ESP_OK) {
    if (error != nullptr) {
      *error = esp_err_to_name(err);
    }
    esp_http_client_cleanup(client);
    return false;
  }

  const int status_code = esp_http_client_fetch_headers(client);
  if (status_code < 0) {
    if (error != nullptr) {
      *error = "failed to fetch headers";
    }
    esp_http_client_close(client);
    esp_http_client_cleanup(client);
    return false;
  }

  char buffer[256];
  while (true) {
    const int read = esp_http_client_read(client, buffer, sizeof(buffer));
    if (read < 0) {
      if (error != nullptr) {
        *error = "manifest read failed";
      }
      esp_http_client_close(client);
      esp_http_client_cleanup(client);
      return false;
    }
    if (read == 0) {
      break;
    }
    if (body->size() + static_cast<size_t>(read) > kMaxManifestBytes) {
      if (error != nullptr) {
        *error = "manifest too large";
      }
      esp_http_client_close(client);
      esp_http_client_cleanup(client);
      return false;
    }
    body->append(buffer, static_cast<size_t>(read));
  }

  esp_http_client_close(client);
  esp_http_client_cleanup(client);
  return true;
}

bool HttpsOtaService::parse_manifest_(const std::string &body, ManifestInfo *manifest, std::string *error) const {
  if (manifest == nullptr) {
    return false;
  }
  manifest->version = extract_json_string(body, "version");
  manifest->image_url = extract_json_string(body, "image_url");
  if (manifest->image_url.empty()) {
    manifest->image_url = extract_json_string(body, "url");
  }
  if (manifest->version.empty() || manifest->image_url.empty()) {
    if (error != nullptr) {
      *error = "invalid manifest";
    }
    return false;
  }
  return true;
}

}  // namespace espectre
