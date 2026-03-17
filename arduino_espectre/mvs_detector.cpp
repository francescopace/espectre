#include "mvs_detector.h"
#include "utils.h"
#include <cmath>

MVSDetector::MVSDetector(size_t window_size)
    : window_size_(window_size),
      threshold_(1.0f),
      state_(IDLE),
      motion_metric_(0.0f),
      total_packets_(0) {
    turbulence_buffer_.reserve(window_size);
}

void MVSDetector::processPacket(const int8_t* csi_data, const std::vector<uint8_t>& selected_scs) {
    if (!csi_data || selected_scs.empty()) {
        return;
    }

    // Calculate spatial turbulence for this packet
    float turbulence = calculateTurbulence(csi_data, selected_scs);

    // Add to circular buffer
    turbulence_buffer_.push_back(turbulence);
    if (turbulence_buffer_.size() > window_size_) {
        turbulence_buffer_.erase(turbulence_buffer_.begin());
    }

    total_packets_++;
}

void MVSDetector::updateState() {
    if (!isReady()) {
        return;
    }

    // Calculate moving variance (motion metric)
    motion_metric_ = calculateMovingVariance();

    // State machine: simple threshold comparison
    if (motion_metric_ > threshold_) {
        state_ = MOTION;
    } else {
        state_ = IDLE;
    }
}

float MVSDetector::calculateTurbulence(const int8_t* csi_data, const std::vector<uint8_t>& selected_scs) {
    // Extract amplitudes from selected subcarriers
    std::vector<float> amplitudes;
    amplitudes.reserve(selected_scs.size());

    for (uint8_t sc : selected_scs) {
        // Each subcarrier has I and Q components (2 bytes)
        int8_t I = csi_data[sc * 2];
        int8_t Q = csi_data[sc * 2 + 1];

        // Calculate amplitude = sqrt(I² + Q²)
        float amplitude = sqrt(static_cast<float>(I * I + Q * Q));
        amplitudes.push_back(amplitude);
    }

    // Turbulence = spatial standard deviation
    return calculateStdDev(amplitudes);
}

float MVSDetector::calculateMovingVariance() {
    // Moving variance = temporal variance of turbulence
    return calculateVariance(turbulence_buffer_);
}

void MVSDetector::reset() {
    turbulence_buffer_.clear();
    state_ = IDLE;
    motion_metric_ = 0.0f;
    total_packets_ = 0;
}
