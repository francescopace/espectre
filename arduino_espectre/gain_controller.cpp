#include "gain_controller.h"
#include "esp_system.h"

// PHY register access functions (ESP-IDF)
// Note: These may not be available in all Arduino-ESP32 versions
// We'll use weak symbols to allow compilation without them
extern "C" {
    void phy_force_rx_gain(bool enable, uint8_t agc_gain, uint8_t fft_gain) __attribute__((weak));
    uint8_t phy_get_rx_gain_agc() __attribute__((weak));
    uint8_t phy_get_rx_gain_fft() __attribute__((weak));
}

GainController::GainController()
    : agc_gain_(0), fft_gain_(0), locked_(false), supported_(true) {

#if CONFIG_IDF_TARGET_ESP32
    // ESP32 original doesn't support gain lock
    supported_ = false;
#endif

    // Check if PHY functions are available at runtime
    if (phy_force_rx_gain == nullptr ||
        phy_get_rx_gain_agc == nullptr ||
        phy_get_rx_gain_fft == nullptr) {
        Serial.println("Warning: PHY gain functions not available in this Arduino-ESP32 version");
        Serial.println("Gain lock will be disabled. CSI will still work but may be less stable.");
        supported_ = false;
    }
}

bool GainController::isSupported() const {
    return supported_;
}

bool GainController::lockGain() {
    if (!supported_) {
        Serial.println("Gain lock not available - continuing without it");
        Serial.println("(CSI will still work, may have slightly reduced stability)");
        return true;  // Return true to allow continuing
    }

    // Read current gain values
    if (!readCurrentGains()) {
        Serial.println("Failed to read current gains - continuing without lock");
        return true;  // Return true to allow continuing
    }

    // Lock gains at current values
    if (phy_force_rx_gain != nullptr) {
        phy_force_rx_gain(true, agc_gain_, fft_gain_);
        locked_ = true;
        Serial.printf("Gain lock enabled: AGC=%d, FFT=%d\n", agc_gain_, fft_gain_);
    } else {
        Serial.println("phy_force_rx_gain not available - continuing without lock");
    }

    return true;
}

bool GainController::unlockGain() {
    if (!supported_ || !locked_) {
        return false;
    }

    // Restore auto-gain control
    if (phy_force_rx_gain != nullptr) {
        phy_force_rx_gain(false, 0, 0);
        locked_ = false;
        Serial.println("Gain lock disabled");
        return true;
    }

    return false;
}

bool GainController::readCurrentGains() {
    if (!supported_) {
        return false;
    }

    if (phy_get_rx_gain_agc != nullptr && phy_get_rx_gain_fft != nullptr) {
        agc_gain_ = phy_get_rx_gain_agc();
        fft_gain_ = phy_get_rx_gain_fft();
        return true;
    }

    return false;
}
