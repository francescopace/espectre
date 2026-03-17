#ifndef NBVI_CALIBRATOR_H
#define NBVI_CALIBRATOR_H

#include <Arduino.h>
#include <vector>

/**
 * NBVI Calibrator - Non-consecutive Band Variance Index
 * Adapted from ESPectre components/espectre/nbvi_calibrator.h
 *
 * Selects optimal 12 non-consecutive subcarriers for motion detection
 * Accuracy: F1-score ~97% with zero manual configuration
 */
class NBVICalibrator {
public:
    NBVICalibrator();

    /**
     * Collect a single CSI sample during calibration phase
     * @param csi_data Raw CSI data (128 bytes: 64 subcarriers × 2 bytes I/Q)
     */
    void collectSample(const int8_t* csi_data);

    /**
     * Check if calibration has collected enough samples
     */
    bool isComplete() const { return sample_count_ >= CALIBRATION_SAMPLES; }

    /**
     * Get current sample count
     */
    size_t getSampleCount() const { return sample_count_; }

    /**
     * Select optimal 12-subcarrier band using NBVI algorithm
     * Must be called after isComplete() returns true
     * @return Vector of 12 subcarrier indices (sorted)
     */
    std::vector<uint8_t> selectBand();

    /**
     * Calculate adaptive threshold using P95 method
     * threshold = P95(moving_variance) × 1.4
     * @param band Selected subcarrier indices
     * @return Adaptive threshold value
     */
    float calculateAdaptiveThreshold(const std::vector<uint8_t>& band);

    /**
     * Reset calibrator state
     */
    void reset();

private:
    /**
     * Calculate moving variance for a single subcarrier
     */
    float calculateSubcarrierVariance(uint8_t sc_idx);

    /**
     * Calculate NBVI score for a band
     */
    float calculateNBVIScore(const std::vector<uint8_t>& band);

    /**
     * Check if a band has proper non-consecutive spacing
     */
    bool isNonConsecutive(const std::vector<uint8_t>& band);

    static constexpr size_t CALIBRATION_SAMPLES = 700;  // ~7 seconds at 100 pps
    static constexpr size_t BAND_SIZE = 12;
    static constexpr uint8_t GUARD_BAND_LOW = 11;   // Skip first 11 subcarriers
    static constexpr uint8_t GUARD_BAND_HIGH = 52;  // Skip last 11 subcarriers
    static constexpr uint8_t MIN_SPACING = 2;       // Minimum gap between subcarriers

    std::vector<std::vector<float>> magnitude_buffer_;  // [64][samples]
    size_t sample_count_;
};

#endif
