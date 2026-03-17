#ifndef CSI_MANAGER_H
#define CSI_MANAGER_H

#include <Arduino.h>
#include <WiFi.h>
#include "esp_wifi.h"
#include "esp_wifi_types.h"
#include <functional>

#define NUM_SUBCARRIERS 64

/**
 * CSI Manager - Handles CSI hardware configuration and callbacks
 * Adapted from ESPectre components/espectre/csi_manager.h
 */
class CSIManager {
public:
    using CSICallback = std::function<void(const wifi_csi_info_t*)>;

    CSIManager();

    /**
     * Initialize CSI hardware and register callback
     */
    bool begin();

    /**
     * Set user callback for CSI packets
     */
    void setCallback(CSICallback callback);

    /**
     * Get number of dropped CSI packets
     */
    uint32_t getDroppedCount() const { return dropped_count_; }

    /**
     * Get total CSI packets received
     */
    uint32_t getTotalCount() const { return total_count_; }

private:
    static void IRAM_ATTR csiRecvCallback(void* ctx, wifi_csi_info_t* data);

    CSICallback user_callback_;
    static uint32_t dropped_count_;
    static uint32_t total_count_;
    static CSIManager* instance_;
};

#endif
