// SPDX-License-Identifier: GPL-3.0-only
// Commercial licensing available under separate agreement; see LICENSING.md.
#pragma once

#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

esp_err_t espectre_native_wifi_prepare_tx_rate(void);
esp_err_t espectre_native_wifi_apply_tx_rate(void);

#ifdef __cplusplus
}
#endif
