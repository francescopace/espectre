/*
 * ESPectre - Deferred Loop Action
 *
 * Defers a one-shot action until the next runtime loop tick.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <atomic>
#include <utility>

namespace espectre {

class DeferredLoopAction {
 public:
  void request() { pending_.store(true, std::memory_order_relaxed); }

  void clear() { pending_.store(false, std::memory_order_relaxed); }

  bool pending() const { return pending_.load(std::memory_order_relaxed); }

  template<typename ReadyFn, typename ActionFn> void flush_if(ReadyFn &&ready, ActionFn &&action) {
    if (!pending()) {
      return;
    }
    if (!std::forward<ReadyFn>(ready)()) {
      return;
    }
    pending_.store(false, std::memory_order_relaxed);
    std::forward<ActionFn>(action)();
  }

 private:
  std::atomic<bool> pending_{false};
};

}  // namespace espectre
