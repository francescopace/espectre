/*
 * ESPectre - Fixed Pending Queue
 *
 * Bounded mailbox for transferring trivially copyable records from driver
 * callbacks to an owning runtime loop without allocating.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <array>
#include <cstddef>
#include <mutex>
#include <type_traits>

#include "pending_event.h"

namespace espectre {

/**
 * Fixed-capacity FIFO for callback-to-loop handoff.
 *
 * post() never allocates or blocks: it returns false when the queue is full.
 * The consumer drains records with take() from its owning task. T must remain
 * trivially copyable because ESP-IDF protects the short copy with a critical
 * section that must not run constructors, destructors, or heap allocation.
 */
template <typename T, size_t Capacity>
class PendingQueue {
 public:
  static_assert(Capacity > 0U, "PendingQueue capacity must be positive");
  static_assert(std::is_trivially_copyable<T>::value,
                "PendingQueue records must be trivially copyable");

  bool post(const T &value) {
    std::lock_guard<detail::PendingEventLock> lock(lock_);
    if (count_ == Capacity) {
      return false;
    }
    values_[write_index_] = value;
    write_index_ = next_(write_index_);
    ++count_;
    return true;
  }

  bool take(T &value) {
    std::lock_guard<detail::PendingEventLock> lock(lock_);
    if (count_ == 0U) {
      return false;
    }
    value = values_[read_index_];
    read_index_ = next_(read_index_);
    --count_;
    return true;
  }

  void clear() {
    std::lock_guard<detail::PendingEventLock> lock(lock_);
    read_index_ = 0U;
    write_index_ = 0U;
    count_ = 0U;
  }

  size_t size() const {
    std::lock_guard<detail::PendingEventLock> lock(lock_);
    return count_;
  }

 private:
  static constexpr size_t next_(size_t index) {
    return index + 1U == Capacity ? 0U : index + 1U;
  }

  mutable detail::PendingEventLock lock_{};
  std::array<T, Capacity> values_{};
  size_t read_index_{0U};
  size_t write_index_{0U};
  size_t count_{0U};
};

}  // namespace espectre
