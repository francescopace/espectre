/*
 * ESPectre - SDK Facade
 *
 * Single entry point for firmware integrating the ESPectre sensing engine.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

/**
 * @mainpage ESPectre SDK
 *
 * This reference covers the supported integration surface only. Every
 * declaration included in the reference follows the SDK version contract;
 * implementation dependencies that merely ship in the bundle are internal and
 * may change in any release.
 *
 * Start with the
 * [SDK README](https://github.com/francescopace/espectre/blob/main/docs/SDK.md)
 * for installation and a minimal application. Use espectre_sdk.h for the
 * sensing runtime, espectre_core_sdk.h for custom capture pipelines,
 * espectre_protocol_sdk.h for the ESPectre Protocol and its transports,
 * espectre_services_sdk.h for optional services, and espectre_mqtt_sdk.h
 * for the ESP-IDF MQTT implementation.
 *
 * @ref sdk_integration documents lifecycle, threading, source compatibility,
 * and advanced integration contracts for this version of the SDK.
 */

/**
 * @file espectre_sdk.h
 * @brief The public ESPectre integration surface, in one include.
 *
 * ESPectre turns ordinary Wi-Fi traffic into a motion signal: it captures
 * Channel State Information from the radio, extracts features, and reports a
 * debounced motion state. This header is the supported entry point for
 * firmware that embeds the sensing engine in its own application.
 *
 * @code
 * #include "espectre_sdk.h"
 *
 * class ProductFrontend : public espectre::IRuntimeListener {
 *  public:
 *   bool setup() {
 *     runtime_.set_config(espectre::make_runtime_sensing_config_from_kconfig());
 *     return runtime_.setup(this);
 *   }
 *
 *   void loop() { runtime_.loop(); }
 *   void shutdown() { runtime_.shutdown(); }
 *
 *   void on_motion_state_changed(const espectre::RuntimeSnapshot &snapshot) override {
 *     if (!snapshot.ready_to_publish) return;
 *     publish_motion(snapshot.motion_state == espectre::MotionState::MOTION);
 *   }
 *
 *  private:
 *   void publish_motion(bool motion);  // Queue work for the application.
 *   espectre::RuntimeFrontendController runtime_;
 * };
 * @endcode
 *
 * @section sdk_paths Two integration paths
 *
 * - **Full runtime (recommended).** Your firmware owns boot, provisioning,
 *   networking, OTA, and the product surface. ESPectre owns Wi-Fi CSI capture,
 *   calibration, detection, and eventing behind
 *   `espectre::RuntimeFrontendController` and `espectre::IRuntimeListener`.
 *   Requires ESP-IDF >= 5.5.3.
 * - **Core-only.** Your firmware already captures CSI. Include
 *   `espectre_core_sdk.h` and drive `espectre::LightweightDetector` or
 *   `espectre::HighAccuracyDetector` directly; see @ref integration_core_only.
 *
 * @section sdk_threading Threading
 *
 * Run `setup()`, `loop()`, `shutdown()`, and every control call on one owner
 * task. Listener callbacks run on that task and must stay bounded and
 * non-blocking. Raw CSI packet callbacks are the exception: they run in Wi-Fi
 * capture context. @ref integration_threading has the full contract.
 *
 * @section sdk_versioning Versioning
 *
 * `ESPECTRE_SDK_VERSION_STRING` and `ESPECTRE_SDK_VERSION_AT_LEAST()` identify
 * the SDK sources you compiled against. See `runtime/espectre_sdk_version.h`
 * for how that differs from your firmware version.
 *
 * @section sdk_stability Stability tiers
 *
 * Everything reachable from this header is the stable runtime surface and
 * follows the SDK version contract. The opt-in `espectre_core_sdk.h` facade is
 * the lower-level detector extension. The opt-in `espectre_protocol_sdk.h`
 * facade adds the protocol and transport contracts. The optional services and
 * MQTT facades expose supported ESP-IDF integration contracts. Headers included only as
 * implementation dependencies can change in any release. See
 * @ref integration_versioning for the exact guarantees.
 *
 * @section sdk_licensing Licensing
 *
 * ESPectre is dual-licensed: GPLv3, or a separately offered commercial license
 * for proprietary firmware. See `LICENSING.md`.
 */

// SDK identity.
#include "runtime/espectre_sdk_version.h"

// Optional frontend-owned logging sink. No sink is installed by default.
#include "core/espectre_log.h"

// Runtime contracts. Platform-agnostic and host-testable.
#include "runtime/csi_capture_profile.h"
#include "runtime/runtime_capabilities.h"
#include "runtime/runtime_config_utils.h"
#include "runtime/runtime_diagnostics.h"
#include "runtime/runtime_events.h"
#include "runtime/runtime_config.h"
#include "runtime/csi_raw_record.h"
#include "runtime/raw_csi.h"
#include "runtime/runtime_sensing_schema.h"
#include "runtime/runtime_snapshot.h"

// Recommended entry point. The declaration is portable; linking it requires
// the ESP-IDF runtime sources.
#include "runtime/esp_idf/device_identity.h"
#include "runtime/esp_idf/runtime_frontend_controller.h"
#include "runtime/esp_idf/runtime_sensing_kconfig.h"
