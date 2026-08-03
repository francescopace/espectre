/*
 * ESPectre - OTA Service
 *
 * OTA service boundary used by sensing frontends.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */
#pragma once

#include <functional>
#include <string>

#include "espectre_protocol.h"

namespace espectre {

/**
 * The firmware-update seam.
 *
 * Implement it to drive updates through your own distribution channel, or use
 * the shipped `HttpsOtaService` (`ota_service_https.h`), which resolves a
 * release manifest over HTTPS and applies the image with `esp_https_ota`.
 * Frontends expose the result over MQTT and BLE and never talk to the
 * underlying stack themselves.
 *
 * @par Threading
 * Unlike `IRuntimeListener`, these callbacks are **not** guaranteed to run on
 * your loop task: the shipped implementation runs the check and the download
 * on its own worker task and invokes both callbacks from there. Treat the
 * status and prepare callbacks as cross-task, keep them short, and hand work
 * back to your loop rather than driving the runtime from inside them.
 *
 * @par Operation model
 * `start_check()` and `start_update()` are asynchronous and mutually
 * exclusive: only one operation runs at a time, and the second returns false
 * while the first is in flight. Progress arrives through the status callback,
 * and `status()` is safe to poll at any time.
 */
class IOtaService {
 public:
  /** Update progressed. The argument is the new status, valid for the call. */
  using StatusCallback = std::function<void(const EspectreOtaStatus &)>;
  /** Last chance to quiesce before the download starts. See below. */
  using PrepareForUpdateCallback = std::function<void()>;

  virtual ~IOtaService() = default;

  /**
   * Advance service work from the frontend loop.
   *
   * Implementations that do their work on a private task, including the
   * shipped one, leave this empty. Call it anyway so a synchronous
   * implementation stays a drop-in replacement.
   */
  virtual void loop() = 0;
  /** Abandon any operation in flight and release resources. Safe to repeat. */
  virtual void shutdown() = 0;
  /**
   * Ask whether a newer release exists, without downloading it.
   *
   * @param current_version Version to compare against, normally
   *        `espectre_firmware_version()`. Empty is reported as `"unknown"`.
   * @return false when an operation is already in flight or the worker cannot
   *         start. True only means the check began; the answer arrives as
   *         `UPDATE_AVAILABLE` or `UP_TO_DATE` through the status callback.
   */
  virtual bool start_check(const std::string &current_version) = 0;
  /**
   * Download and apply an update, then schedule the reboot.
   *
   * Performs its own check first, so calling `start_check()` beforehand is
   * optional. The prepare callback fires once the target is resolved and
   * before the download begins: that is where you call
   * `RuntimeFrontendController::quiesce_for_ota()` and stop your own traffic.
   *
   * @return false when an operation is already in flight or the worker cannot
   *         start. A successful update ends in `REBOOT_SCHEDULED`.
   */
  virtual bool start_update(const std::string &current_version) = 0;
  /** Current status. Safe to call from any task, including while an update runs. */
  virtual EspectreOtaStatus status() const = 0;
  /** Install the progress handler. Set it before starting an operation. */
  virtual void set_status_callback(StatusCallback callback) = 0;
  /** Install the pre-download hook. Set it before calling `start_update()`. */
  virtual void set_prepare_for_update_callback(PrepareForUpdateCallback callback) = 0;
};

}  // namespace espectre
