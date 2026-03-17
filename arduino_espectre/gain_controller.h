#ifndef GAIN_CONTROLLER_H
#define GAIN_CONTROLLER_H

#include <Arduino.h>

/**
 * Gain Controller - Handles AGC/FFT gain lock for stable CSI measurements
 * Adapted from ESPectre components/espectre/gain_controller.h
 *
 * Supported on: ESP32-S3, ESP32-C3, ESP32-C5, ESP32-C6
 * Not supported on: ESP32 (original)
 */
class GainController {
public:
    GainController();

    /**
     * Lock AGC and FFT gains at current values
     * Call after 3 seconds of WiFi activity
     */
    bool lockGain();

    /**
     * Unlock gains (restore auto-gain control)
     */
    bool unlockGain();

    /**
     * Check if gain lock is supported on this chip
     */
    bool isSupported() const;

    /**
     * Get current AGC gain value
     */
    uint8_t getAgcGain() const { return agc_gain_; }

    /**
     * Get current FFT gain value
     */
    uint8_t getFftGain() const { return fft_gain_; }

    /**
     * Check if gains are locked
     */
    bool isLocked() const { return locked_; }

private:
    bool readCurrentGains();

    uint8_t agc_gain_;
    uint8_t fft_gain_;
    bool locked_;
    bool supported_;
};

#endif
