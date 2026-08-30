// SPDX-License-Identifier: GPL-3.0-only
// Commercial licensing available under separate agreement; see LICENSING.md.

#ifndef NO_QSTR

#include "native_traffic.h"
#include "native_log_sink.h"

#include "runtime/esp_idf/traffic_generator_manager.h"

#include <new>

namespace {

espectre::TrafficGeneratorManager *as_manager(void *handle) {
  return static_cast<espectre::TrafficGeneratorManager *>(handle);
}

espectre::TrafficGeneratorMode resolve_mode(espectre_native_traffic_mode_t mode) {
  switch (mode) {
    case ESPECTRE_NATIVE_TRAFFIC_DNS:
      return espectre::TrafficGeneratorMode::DNS;
    case ESPECTRE_NATIVE_TRAFFIC_DNS_TCP:
      return espectre::TrafficGeneratorMode::DNS_TCP;
    case ESPECTRE_NATIVE_TRAFFIC_PING:
    default:
      return espectre::TrafficGeneratorMode::PING;
  }
}

}  // namespace

extern "C" void *espectre_native_traffic_create(void) {
  espectre_native_ensure_log_sink();
  return new (std::nothrow) espectre::TrafficGeneratorManager();
}

extern "C" void espectre_native_traffic_destroy(void *handle) {
  auto *manager = as_manager(handle);
  if (manager == nullptr) {
    return;
  }
  manager->stop();
  delete manager;
}

extern "C" bool espectre_native_traffic_start(
    void *handle,
    uint32_t gateway_addr,
    uint32_t rate_pps,
    espectre_native_traffic_mode_t mode) {
  auto *manager = as_manager(handle);
  if (manager == nullptr || manager->is_running()) {
    return false;
  }
  manager->init(rate_pps, resolve_mode(mode));
  return manager->start(gateway_addr);
}

extern "C" void espectre_native_traffic_stop(void *handle) {
  auto *manager = as_manager(handle);
  if (manager != nullptr) {
    manager->stop();
  }
}

extern "C" bool espectre_native_traffic_pause(void *handle) {
  auto *manager = as_manager(handle);
  if (manager == nullptr || !manager->is_running()) {
    return false;
  }
  manager->pause();
  return true;
}

extern "C" bool espectre_native_traffic_resume(void *handle) {
  auto *manager = as_manager(handle);
  if (manager == nullptr || !manager->is_running()) {
    return false;
  }
  manager->resume();
  return true;
}

extern "C" bool espectre_native_traffic_is_running(void *handle) {
  auto *manager = as_manager(handle);
  return manager != nullptr && manager->is_running();
}

extern "C" uint32_t espectre_native_traffic_packet_count(void *handle) {
  auto *manager = as_manager(handle);
  return manager == nullptr ? 0U : manager->send_success_count();
}

extern "C" uint32_t espectre_native_traffic_error_count(void *handle) {
  auto *manager = as_manager(handle);
  return manager == nullptr ? 0U : manager->send_error_count();
}

#endif
