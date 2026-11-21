// SPDX-License-Identifier: GPL-3.0-only
// Commercial licensing available under separate agreement; see LICENSING.md.
#pragma once

#include "esphome/core/defines.h"

// The IDF console component supplies this header only for the USB CDC logger.
#ifdef USE_ESPECTRE_USB_CONSOLE
#include "primary_console.h"
#endif
