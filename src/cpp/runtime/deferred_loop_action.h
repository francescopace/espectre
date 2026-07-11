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
