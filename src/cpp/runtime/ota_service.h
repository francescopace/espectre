#pragma once

#include <functional>
#include <string>

#include "espectre_protocol.h"

namespace espectre {

class IOtaService {
 public:
  using StatusCallback = std::function<void(const EspectreOtaStatus &)>;
  using PrepareForUpdateCallback = std::function<void()>;

  virtual ~IOtaService() = default;

  virtual void loop() = 0;
  virtual void shutdown() = 0;
  virtual bool start_check(const std::string &manifest_url, const std::string &current_version) = 0;
  virtual bool start_update(const std::string &manifest_url,
                            const std::string &image_url,
                            const std::string &target_version,
                            const std::string &current_version) = 0;
  virtual EspectreOtaStatus status() const = 0;
  virtual void set_status_callback(StatusCallback callback) = 0;
  virtual void set_prepare_for_update_callback(PrepareForUpdateCallback callback) = 0;
};

}  // namespace espectre
