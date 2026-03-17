#include "nbvi_calibrator.h"
#include "utils.h"
#include <cmath>
#include <algorithm>

NBVICalibrator::NBVICalibrator() : sample_count_(0) {
    // Initialize magnitude buffer: 64 subcarriers × CALIBRATION_SAMPLES
    magnitude_buffer_.resize(64);
    for (auto& vec : magnitude_buffer_) {
        vec.reserve(CALIBRATION_SAMPLES);
    }
}

void NBVICalibrator::collectSample(const int8_t* csi_data) {
    if (!csi_data || sample_count_ >= CALIBRATION_SAMPLES) {
        return;
    }

    // Extract magnitude for each subcarrier
    for (size_t sc = 0; sc < 64; sc++) {
        int8_t I = csi_data[sc * 2];
        int8_t Q = csi_data[sc * 2 + 1];
        float magnitude = sqrt(static_cast<float>(I * I + Q * Q));
        magnitude_buffer_[sc].push_back(magnitude);
    }

    sample_count_++;
}

std::vector<uint8_t> NBVICalibrator::selectBand() {
    if (!isComplete()) {
        Serial.println("Warning: Calibration not complete!");
        return {};
    }

    Serial.println("Selecting optimal band using NBVI...");

    // Calculate variance for each subcarrier
    std::vector<std::pair<uint8_t, float>> sc_variances;
    for (uint8_t sc = GUARD_BAND_LOW; sc <= GUARD_BAND_HIGH; sc++) {
        float variance = calculateSubcarrierVariance(sc);
        sc_variances.push_back({sc, variance});
    }

    // Sort by variance (ascending - we want stable subcarriers)
    std::sort(sc_variances.begin(), sc_variances.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });

    // Select top 12 stable subcarriers with non-consecutive constraint
    std::vector<uint8_t> selected_band;
    selected_band.reserve(BAND_SIZE);

    for (const auto& [sc, variance] : sc_variances) {
        if (selected_band.size() >= BAND_SIZE) {
            break;
        }

        // Check if this subcarrier is non-consecutive with already selected ones
        bool is_valid = true;
        for (uint8_t existing_sc : selected_band) {
            if (abs(static_cast<int>(sc) - static_cast<int>(existing_sc)) < MIN_SPACING) {
                is_valid = false;
                break;
            }
        }

        if (is_valid) {
            selected_band.push_back(sc);
        }
    }

    // Sort selected band
    std::sort(selected_band.begin(), selected_band.end());

    Serial.print("Selected band: ");
    for (uint8_t sc : selected_band) {
        Serial.print(sc);
        Serial.print(" ");
    }
    Serial.println();

    return selected_band;
}

float NBVICalibrator::calculateAdaptiveThreshold(const std::vector<uint8_t>& band) {
    if (band.empty() || !isComplete()) {
        Serial.println("Warning: Cannot calculate threshold - invalid band or incomplete calibration");
        return 1.0f;
    }

    Serial.println("Calculating adaptive threshold...");

    // Calculate turbulence for each calibration sample
    std::vector<float> turbulence_samples;
    turbulence_samples.reserve(sample_count_);

    for (size_t i = 0; i < sample_count_; i++) {
        // Get amplitudes for this sample across selected band
        std::vector<float> amplitudes;
        amplitudes.reserve(band.size());

        for (uint8_t sc : band) {
            amplitudes.push_back(magnitude_buffer_[sc][i]);
        }

        // Calculate spatial std dev (turbulence)
        float turbulence = calculateStdDev(amplitudes);
        turbulence_samples.push_back(turbulence);
    }

    // Calculate moving variance with window size 50
    const size_t window_size = 50;
    std::vector<float> moving_variances;

    for (size_t i = window_size; i < turbulence_samples.size(); i++) {
        std::vector<float> window(
            turbulence_samples.begin() + i - window_size,
            turbulence_samples.begin() + i
        );
        float variance = calculateVariance(window);
        moving_variances.push_back(variance);
    }

    // Calculate P95 (95th percentile)
    float p95 = calculatePercentile(moving_variances, 0.95);

    // Adaptive threshold = P95 × 2.5 (increased from 1.4 for better stability)
    // This reduces false positives from WiFi interference
    float threshold = p95 * 2.5f;

    Serial.print("P95 moving variance: ");
    Serial.println(p95, 3);
    Serial.print("Adaptive threshold: ");
    Serial.println(threshold, 3);

    return threshold;
}

float NBVICalibrator::calculateSubcarrierVariance(uint8_t sc_idx) {
    if (sc_idx >= 64 || magnitude_buffer_[sc_idx].empty()) {
        return 999999.0f;  // Invalid subcarrier
    }

    return calculateVariance(magnitude_buffer_[sc_idx]);
}

float NBVICalibrator::calculateNBVIScore(const std::vector<uint8_t>& band) {
    // Calculate combined variance of selected band
    float total_variance = 0.0f;

    for (uint8_t sc : band) {
        total_variance += calculateSubcarrierVariance(sc);
    }

    return total_variance / band.size();
}

bool NBVICalibrator::isNonConsecutive(const std::vector<uint8_t>& band) {
    for (size_t i = 1; i < band.size(); i++) {
        if (band[i] - band[i-1] < MIN_SPACING) {
            return false;
        }
    }
    return true;
}

void NBVICalibrator::reset() {
    sample_count_ = 0;
    for (auto& vec : magnitude_buffer_) {
        vec.clear();
    }
}
