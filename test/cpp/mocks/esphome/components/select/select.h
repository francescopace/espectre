#pragma once

#include <string>

#define LOG_SELECT(tag, name, obj) do {} while (0)

namespace esphome {
namespace select {

class Select {
 public:
  void publish_state(const std::string &state) {
    state_ = state;
    has_state_ = true;
    publish_count_++;
  }
  const std::string &get_state() const { return state_; }
  bool has_state() const { return has_state_; }
  unsigned int get_publish_count() const { return publish_count_; }

 protected:
  virtual void control(const std::string &value) { publish_state(value); }

  std::string state_;
  bool has_state_{false};
  unsigned int publish_count_{0U};
};

}  // namespace select
}  // namespace esphome
