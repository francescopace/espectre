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

#include <atomic>
#include <cstddef>
#include <tuple>
#include <utility>

namespace espectre {

/**
 * Single-slot mailbox carrying an event with an optional payload.
 *
 * post() records the event and overwrites any unconsumed payload, so events
 * coalesce to the most recent one. take() consumes at most one event per
 * call. Single producer, single consumer.
 *
 * Each payload field is an individual lock-free atomic, so fields never
 * tear; a take() racing a post() may pair fields from two consecutive
 * posts, which coalescing channels tolerate by design.
 */
template <typename... Ts>
class PendingEvent {
  static_assert((std::atomic<Ts>::is_always_lock_free && ...),
                "payload fields must be lock-free atomic types");

 public:
  void post(Ts... values) {
    store_(std::index_sequence_for<Ts...>{}, values...);
    pending_.store(true, std::memory_order_release);
  }

  bool take(Ts &...out) {
    if (!pending_.exchange(false, std::memory_order_acquire)) {
      return false;
    }
    load_(std::index_sequence_for<Ts...>{}, out...);
    return true;
  }

  void clear() { pending_.store(false, std::memory_order_relaxed); }

 private:
  template <std::size_t... Is>
  void store_(std::index_sequence<Is...>, Ts... values) {
    (std::get<Is>(values_).store(values, std::memory_order_relaxed), ...);
  }

  template <std::size_t... Is>
  void load_(std::index_sequence<Is...>, Ts &...out) const {
    ((out = std::get<Is>(values_).load(std::memory_order_relaxed)), ...);
  }

  std::tuple<std::atomic<Ts>...> values_{};
  std::atomic<bool> pending_{false};
};

}  // namespace espectre
