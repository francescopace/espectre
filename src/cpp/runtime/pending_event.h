/*
 * ESPectre - Pending Event Mailbox
 *
 * Single-slot mailbox for deferring work from a callback or event-handler
 * context to the runtime loop task.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */
#pragma once

#include <cstddef>
#include <mutex>
#if defined(ESP_PLATFORM)
#include "freertos/FreeRTOS.h"
#include "freertos/portmacro.h"
#endif
#include <tuple>
#include <utility>

namespace espectre {

namespace detail {

class PendingEventLock {
 public:
  void lock() {
#if defined(ESP_PLATFORM)
    portENTER_CRITICAL_SAFE(&mux_);
#else
    mutex_.lock();
#endif
  }

  void unlock() {
#if defined(ESP_PLATFORM)
    portEXIT_CRITICAL_SAFE(&mux_);
#else
    mutex_.unlock();
#endif
  }

 private:
#if defined(ESP_PLATFORM)
  portMUX_TYPE mux_ = portMUX_INITIALIZER_UNLOCKED;
#else
  std::mutex mutex_;
#endif
};

}  // namespace detail

/**
 * Single-slot mailbox carrying an event with an optional payload.
 *
 * post() records the event and overwrites any unconsumed payload, so events
 * coalesce to the most recent one. take() consumes at most one event per
 * call. Single producer, single consumer.
 *
 * Access is serialized with a lightweight critical section. On ESP-IDF this
 * remains safe from both task and ISR context, including the CSI callback
 * path. On host builds, tests use a regular mutex with the same coalescing
 * semantics.
 */
template <typename... Ts>
class PendingEvent {
 public:
  void post(Ts... values) {
    std::lock_guard<detail::PendingEventLock> lock(lock_);
    store_(std::index_sequence_for<Ts...>{}, values...);
    pending_ = true;
  }

  bool take(Ts &...out) {
    std::lock_guard<detail::PendingEventLock> lock(lock_);
    if (!pending_) {
      return false;
    }
    pending_ = false;
    load_(std::index_sequence_for<Ts...>{}, out...);
    return true;
  }

  void clear() {
    std::lock_guard<detail::PendingEventLock> lock(lock_);
    pending_ = false;
  }

 private:
  template <std::size_t... Is>
  void store_(std::index_sequence<Is...>, Ts... values) {
    ((std::get<Is>(values_) = values), ...);
  }

  template <std::size_t... Is>
  void load_(std::index_sequence<Is...>, Ts &...out) const {
    ((out = std::get<Is>(values_)), ...);
  }

  detail::PendingEventLock lock_{};
  std::tuple<Ts...> values_{};
  bool pending_{false};
};

}  // namespace espectre
