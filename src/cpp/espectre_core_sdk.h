/*
 * ESPectre - Core-only SDK Facade
 *
 * Explicit opt-in surface for integrations that already capture normalized
 * CSI and drive the detectors directly.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

/**
 * @file espectre_core_sdk.h
 * @brief Core-only detector API for custom CSI capture integrations.
 *
 * Prefer `espectre_sdk.h` and `RuntimeFrontendController` when ESPectre should
 * own CSI capture, temporal admission, calibration, and event delivery. Include
 * this facade only when the embedding firmware already implements those parts.
 */

#include "runtime/espectre_sdk_version.h"
#include "core/base_detector.h"
#include "core/csi_format.h"
#include "core/high_accuracy_detector.h"
#include "core/lightweight_detector.h"
#include "core/temporal_csi_sampler.h"
