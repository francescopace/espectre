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
 * This reference covers the supported integration surface only. Every header
 * listed here follows the SDK version contract; anything else in the bundle is
 * internal and may change in any release.
 *
 * Start at espectre_sdk.h for the integration model, the threading contract,
 * and a working example. The repository guide `docs/EMBEDDING.md` covers build
 * integration, install surfaces, and release channels.
 */

/**
 * @file espectre_sdk.h
 * @brief The public ESPectre integration surface, in one include.
 *
 * ESPectre turns ordinary Wi-Fi traffic into a motion signal: it captures
 * Channel State Information from the radio, extracts features, and reports a
 * debounced motion state. This header is the supported entry point for
 * firmware that embeds that engine instead of flashing one of the published
 * frontends.
 *
 * @code
 * #include "espectre_sdk.h"
 *
 * class ProductFrontend : public espectre::IRuntimeListener {
 *  public:
 *   bool setup() {
 *     espectre::RuntimeConfig config;  // documented defaults, ready to use
 *     runtime_.set_config(config);
 *     return runtime_.setup(this);
 *   }
 *
 *   void loop() { runtime_.loop(); }
 *
 *   void on_motion_state_changed(const espectre::RuntimeSnapshot &snapshot) override {
 *     if (!snapshot.ready_to_publish) return;
 *     publish(snapshot.motion_state == espectre::MotionState::MOTION);
 *   }
 *
 *  private:
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
 *   Requires ESP-IDF >= 5.5.
 * - **Core-only.** Your firmware already captures CSI. Drive
 *   `espectre::LightweightDetector` or `espectre::HighAccuracyDetector` directly; they need
 *   nothing but the C++17 standard library. `runtime/esp_idf/csi_pipeline.cpp`
 *   is the reference for normalization, evaluation cadence, and hit filtering.
 *
 * @section sdk_threading Threading contract
 *
 * The control surface is single-owner. Internal bounded mailboxes protect
 * callback-to-loop handoff, but they do not make control calls thread-safe.
 *
 * - Run `setup()`, `loop()`, and `shutdown()` on one task. These are the calls
 *   that build and tear down runtime state, and they are not safe to race.
 * - Every `IRuntimeListener` callback is delivered on the caller's task: from
 *   `loop()` for sensing events, or inline on the task that invoked a control
 *   method. Work raised in the Wi-Fi CSI callback is deferred through an
 *   internal mailbox first, so no listener callback runs in interrupt or Wi-Fi
 *   driver context.
 * - Because callbacks run on your own task, you may block in them (publish
 *   over MQTT, write NVS). The cost is loop latency, not a dropped CSI frame.
 * - Call `set_*_runtime()` only from the owner task. The shipped MQTT, BLE, and
 *   OTA adapters queue stack events and deliver their application callbacks
 *   from the frontend loop, so Native follows this rule without external locks.
 * - Do not drive the controller from inside `on_runtime_fault()` beyond
 *   `shutdown()`.
 *
 * @section sdk_versioning Versioning
 *
 * `ESPECTRE_SDK_VERSION_STRING` and `ESPECTRE_SDK_VERSION_AT_LEAST()` identify
 * the SDK sources you compiled against. See `runtime/espectre_sdk_version.h`
 * for how that differs from your firmware version.
 *
 * @section sdk_stability Stability tiers
 *
 * Everything reachable from this header is the supported surface and follows
 * the SDK version contract. Headers that this facade does not pull in are
 * internal: they ship in the bundle because the runtime needs them to compile,
 * and they can change in any release. `docs/EMBEDDING.md` lists the tiers.
 *
 * @section sdk_licensing Licensing
 *
 * ESPectre is dual-licensed: GPLv3, or a separately offered commercial license
 * for proprietary firmware. See `LICENSING.md`.
 */

// SDK identity.
#include "runtime/espectre_sdk_version.h"

// Detectors and CSI format. Portable, C++17 standard library only.
#include "core/base_detector.h"
#include "core/filtered_turbulence_ring.h"
#include "core/lightweight_detector.h"
#include "core/csi_format.h"
#include "core/high_accuracy_detector.h"

// Runtime contracts. Platform-agnostic and host-testable.
#include "runtime/firmware_version.h"
#include "runtime/runtime_capabilities.h"
#include "runtime/runtime_config_utils.h"
#include "runtime/runtime_diagnostics.h"
#include "runtime/runtime_events.h"
#include "runtime/runtime_interface.h"
#include "runtime/runtime_sensing_schema.h"
#include "runtime/runtime_snapshot.h"

// Boundary interfaces you implement to reach your own transports.
#include "runtime/ble_bindings.h"
#include "runtime/espectre_protocol.h"
#include "runtime/mqtt_transport.h"
#include "runtime/ota_service.h"

// Recommended entry point. The declaration is portable; linking it requires
// the ESP-IDF runtime sources.
#include "runtime/esp_idf/runtime_frontend_controller.h"
#include "runtime/esp_idf/runtime_sensing_kconfig.h"
