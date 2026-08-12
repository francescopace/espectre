/*
 * ESPectre - SDK Version
 *
 * Compile-time version of the embedded ESPectre SDK.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

/**
 * @file espectre_sdk_version.h
 * @brief Compile-time identity of the ESPectre SDK sources you compiled against.
 *
 * This is the version of the *SDK*, not of your firmware. The two are
 * deliberately separate:
 *
 * - `ESPECTRE_SDK_VERSION_STRING` is baked in at compile time and identifies
 *   the ESPectre sources in your build. Use it in diagnostics, bug reports,
 *   and to guard code against SDK releases.
 * - `espectre::espectre_firmware_version()` (`firmware_version.h`) reports the
 *   *application* version supplied by your build system. In an integration it
 *   is your product's version, not ESPectre's.
 *
 * The value here is the single source of truth inside the C++ tree, and the
 * release tooling keeps it in step with `src/cpp/idf_component.yml`. Snapshot
 * bundles stamp a prerelease suffix onto the component manifest only; the
 * macros below always carry the numeric release they were branched from.
 */

/** Major version. Changes when the published integration surface breaks. */
#define ESPECTRE_SDK_VERSION_MAJOR 3
/** Minor version. Changes when the surface grows in a backward-compatible way. */
#define ESPECTRE_SDK_VERSION_MINOR 0
/** Patch version. Changes for fixes that keep the surface identical. */
#define ESPECTRE_SDK_VERSION_PATCH 0

/** Dotted version string, for example `"3.0.0"`. */
#define ESPECTRE_SDK_VERSION_STRING "3.0.0"

/**
 * Single comparable integer for the SDK version, as `MMmmpp`.
 *
 * Example: `3.0.0` becomes `30000`.
 */
#define ESPECTRE_SDK_VERSION_NUMBER \
  ((ESPECTRE_SDK_VERSION_MAJOR * 10000) + (ESPECTRE_SDK_VERSION_MINOR * 100) + ESPECTRE_SDK_VERSION_PATCH)

/**
 * Compile-time feature guard.
 *
 * Use it to keep one integration compiling against several SDK releases:
 * @code
 * #if ESPECTRE_SDK_VERSION_AT_LEAST(3, 1, 0)
 *   controller.set_motion_hits_runtime(3, 5);
 * #endif
 * @endcode
 */
#define ESPECTRE_SDK_VERSION_AT_LEAST(major, minor, patch) \
  (ESPECTRE_SDK_VERSION_NUMBER >= ((major) * 10000 + (minor) * 100 + (patch)))

namespace espectre {

/**
 * The SDK version as a string, usable where a macro is not.
 *
 * @return `ESPECTRE_SDK_VERSION_STRING`, valid for the process lifetime.
 */
constexpr const char *espectre_sdk_version() { return ESPECTRE_SDK_VERSION_STRING; }

}  // namespace espectre
