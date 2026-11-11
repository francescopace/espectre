/*
 * ESPectre - Mock ESPHome Version
 *
 * Minimal mock for esphome/core/version.h, so that host checks can evaluate
 * the version guards in ESPHome frontend code.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#define VERSION_CODE(major, minor, patch) ((major) << 16 | (minor) << 8 | (patch))
#ifndef ESPHOME_VERSION_CODE
#define ESPHOME_VERSION "2026.11.0"
#define ESPHOME_VERSION_CODE VERSION_CODE(2026, 11, 0)
#endif
