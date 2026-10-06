// SPDX-License-Identifier: GPL-3.0-only
// Commercial licensing available under separate agreement; see LICENSING.md.

#ifndef NO_QSTR

#include "native_traffic.h"
#include "native_log_sink.h"

#include "runtime/esp_idf/traffic_generator_service.h"

#include <new>

namespace {

espectre::TrafficGeneratorService *as_generator(void *handle) {
  return static_cast<espectre::TrafficGeneratorService *>(handle);
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
  return new (std::nothrow) espectre::TrafficGeneratorService();
}

extern "C" void espectre_native_traffic_destroy(void *handle) {
  auto *generator = as_generator(handle);
  if (generator == nullptr) {
    return;
  }
  generator->stop();
  delete generator;
}

extern "C" bool espectre_native_traffic_start(
    void *handle,
    uint32_t target_addr,
    uint32_t rate_pps,
    espectre_native_traffic_mode_t mode) {
  auto *generator = as_generator(handle);
  if (generator == nullptr || generator->is_running()) {
    return false;
  }
  generator->init(rate_pps, resolve_mode(mode));
  return generator->start(target_addr);
}

extern "C" void espectre_native_traffic_stop(void *handle) {
  auto *generator = as_generator(handle);
  if (generator != nullptr) {
    generator->stop();
  }
}

extern "C" bool espectre_native_traffic_pause(void *handle) {
  auto *generator = as_generator(handle);
  if (generator == nullptr || !generator->is_running()) {
    return false;
  }
  generator->pause();
  return true;
}

extern "C" bool espectre_native_traffic_resume(void *handle) {
  auto *generator = as_generator(handle);
  if (generator == nullptr || !generator->is_running()) {
    return false;
  }
  generator->resume();
  return true;
}

extern "C" bool espectre_native_traffic_is_running(void *handle) {
  auto *generator = as_generator(handle);
  return generator != nullptr && generator->is_running();
}

extern "C" uint32_t espectre_native_traffic_packet_count(void *handle) {
  auto *generator = as_generator(handle);
  return generator == nullptr ? 0U : generator->send_success_count();
}

extern "C" uint32_t espectre_native_traffic_error_count(void *handle) {
  auto *generator = as_generator(handle);
  return generator == nullptr ? 0U : generator->send_error_count();
}

#endif
