#include "csi_manager.h"

uint32_t CSIManager::dropped_count_ = 0;
uint32_t CSIManager::total_count_ = 0;
CSIManager* CSIManager::instance_ = nullptr;

CSIManager::CSIManager() : user_callback_(nullptr) {
    instance_ = this;
}

bool CSIManager::begin() {
    Serial.println("\n--- Initializing CSI ---");

    // Configure CSI
    wifi_csi_config_t csi_config = {
        .lltf_en = true,
        .htltf_en = true,
        .stbc_htltf2_en = true,
        .ltf_merge_en = true,
        .channel_filter_en = false,
        .manu_scale = false,
    };

    Serial.println("Setting CSI configuration...");
    esp_err_t err = esp_wifi_set_csi_config(&csi_config);
    if (err != ESP_OK) {
        Serial.printf("❌ Failed to set CSI config: 0x%x (%s)\n", err, esp_err_to_name(err));
        return false;
    }
    Serial.println("✓ CSI config set");

    Serial.println("Registering CSI callback...");
    err = esp_wifi_set_csi_rx_cb(&CSIManager::csiRecvCallback, this);
    if (err != ESP_OK) {
        Serial.printf("❌ Failed to set CSI callback: 0x%x (%s)\n", err, esp_err_to_name(err));
        return false;
    }
    Serial.println("✓ CSI callback registered");

    Serial.println("Enabling CSI...");
    err = esp_wifi_set_csi(true);
    if (err != ESP_OK) {
        Serial.printf("❌ Failed to enable CSI: 0x%x (%s)\n", err, esp_err_to_name(err));
        return false;
    }
    Serial.println("✓ CSI enabled");

    Serial.println("✓ CSI initialization complete\n");
    return true;
}

void CSIManager::setCallback(CSICallback callback) {
    user_callback_ = callback;
}

void IRAM_ATTR CSIManager::csiRecvCallback(void* ctx, wifi_csi_info_t* data) {
    if (!data) {
        dropped_count_++;
        return;
    }

    if (!instance_) {
        dropped_count_++;
        return;
    }

    total_count_++;

    // Debug: Log first few packets
    static uint32_t debug_count = 0;
    if (debug_count < 5) {
        debug_count++;
        // Note: Can't use Serial.print in ISR, just increment counter
    }

    // Call user callback if registered
    if (instance_->user_callback_) {
        instance_->user_callback_(data);
    }
}
