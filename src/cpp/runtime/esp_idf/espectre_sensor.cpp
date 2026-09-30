// SPDX-License-Identifier: GPL-3.0-only
// Commercial licensing available under separate agreement; see LICENSING.md.
#include "espectre_sensor.h"

#include <cstdio>

#include "core/espectre_log.h"
#include "runtime/esp_idf/runtime_sensing_kconfig.h"

#if __has_include(<esp32-hal-log.h>)
#include <esp32-hal-log.h>
#define ESPECTRE_ARDUINO_HAS_LOG 1
#endif

namespace espectre {

namespace {

#ifdef ESPECTRE_ARDUINO_HAS_LOG
// Follow the Core Debug Level, like the core's own log_x() messages.
bool arduino_log_enabled(void *, LogLevel level, const char *) {
  return static_cast<int>(level) <= ARDUHAL_LOG_LEVEL;
}

// The sink can run in Wi-Fi capture context, so it formats on the stack and
// writes through the ROM printf. log_printf() uses a shared static buffer and
// can allocate.
void arduino_log_write(void *, LogLevel level, const char *tag, int, const char *format,
                       va_list args) {
  static constexpr char LEVEL_LETTERS[] = "EWIDV";
  char message[160];
  vsnprintf(message, sizeof(message), format, args);
  ets_printf("[%c][%s] %s\r\n", LEVEL_LETTERS[static_cast<int>(level) - 1], tag, message);
}
#endif

void install_log_sink() {
#ifdef ESPECTRE_ARDUINO_HAS_LOG
  // A sink the sketch registered before begin() takes precedence.
  if (!log_sink_registered()) {
    set_log_sink({nullptr, arduino_log_enabled, arduino_log_write});
  }
#endif
}

}  // namespace

ESPectre::ESPectre() {
  RuntimeConfig config = make_runtime_sensing_config_from_kconfig();
  // A sketch sets its configuration in code, so saved controls must not
  // silently override it after a reboot.
  config.persist_runtime_overrides = false;
  runtime_.set_config(config);
}

bool ESPectre::begin() {
  if (runtime_.is_setup_complete()) {
    return true;
  }
  install_log_sink();
  return runtime_.setup(this);
}

bool ESPectre::motion() const {
  const RuntimeSnapshot &snapshot = runtime_.snapshot();
  return snapshot.ready_to_publish && snapshot.motion_state == MotionState::MOTION;
}

void ESPectre::report_motion_(bool motion) {
  if (motion == reported_motion_) {
    return;
  }
  reported_motion_ = motion;
  if (motion_callback_) {
    motion_callback_(motion);
  }
}

void ESPectre::on_sensing_readiness_changed(const RuntimeSnapshot &snapshot) {
  // The runtime resets motion while it is not ready, and that edge is not
  // published, so close a reported motion before the readiness change.
  if (!snapshot.ready_to_publish) {
    report_motion_(false);
  }
  if (ready_callback_) {
    ready_callback_(snapshot.ready_to_publish);
  }
}

void ESPectre::on_motion_state_changed(const RuntimeSnapshot &snapshot) {
  report_motion_(snapshot.ready_to_publish && snapshot.motion_state == MotionState::MOTION);
}

void ESPectre::on_runtime_fault(const char *message) {
  if (fault_callback_) {
    fault_callback_(message);
  }
}

}  // namespace espectre
