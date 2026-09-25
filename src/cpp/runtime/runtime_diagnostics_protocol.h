/*
 * ESPectre - Runtime Diagnostics Protocol
 *
 * Diagnostics serialization for the ESPectre Protocol.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <functional>
#include <string>
#include <vector>

#include "diagnostic_fields.h"
#include "runtime_diagnostics.h"
#include "runtime_snapshot.h"

/**
 * @file runtime_diagnostics_protocol.h
 * @brief Diagnostic field selection and JSON serialization for protocol responses.
 */

namespace espectre {

/**
 * Append shared platform and performance fields to an existing JSON object.
 *
 * The object must already contain at least one field. Metrics from an
 * incomplete aggregation window are emitted as `null`; unsupported detector
 * timing is identified separately by `detection_timing_supported`.
 */
void append_runtime_performance_diagnostics_json(std::string *out,
                                                 const RuntimeDiagnosticsSnapshot &diagnostics,
                                                 bool include_current_memory = true);

/** Validate diagnostic paths, groups, or an exclusive wildcard against the canonical registry. */
bool validate_diagnostic_fields(const std::vector<std::string> &fields, unsigned profile = ESPECTRE_DIAGNOSTIC_PROFILE_ALL);
/** Serialize the catalog without reading values, or only selected values from the supplied provider.
 * `profile` is a mask of `ESPECTRE_DIAGNOSTIC_PROFILE_*` values. Empty selections return the catalog.
 * The provider returns one JSON scalar. Unknown profile fields produce an empty response.
 */
std::string diagnostic_response(const std::vector<std::string> &fields, unsigned profile,
                                const std::function<std::string(const char *)> &value);
/** Read a shared scalar from a cached rate sample or a lazily acquired runtime snapshot. */
std::string runtime_diagnostic_value(const char *key, const RuntimeDiagnosticsSample *sample,
                                     const std::function<const RuntimeDiagnosticsSnapshot &()> &snapshot);

}  // namespace espectre
