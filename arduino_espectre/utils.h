#ifndef UTILS_H
#define UTILS_H

#include <Arduino.h>
#include <vector>
#include <cmath>
#include <algorithm>

// Calculate mean of a vector
inline float calculateMean(const std::vector<float>& data) {
    if (data.empty()) return 0.0f;

    float sum = 0.0f;
    for (float val : data) {
        sum += val;
    }
    return sum / data.size();
}

// Calculate variance of a vector
inline float calculateVariance(const std::vector<float>& data) {
    if (data.size() < 2) return 0.0f;

    float mean = calculateMean(data);
    float sum_sq_diff = 0.0f;

    for (float val : data) {
        float diff = val - mean;
        sum_sq_diff += diff * diff;
    }

    return sum_sq_diff / data.size();
}

// Calculate standard deviation of a vector
inline float calculateStdDev(const std::vector<float>& data) {
    return sqrt(calculateVariance(data));
}

// Calculate median of a vector (modifies copy)
inline float calculateMedian(std::vector<float> data) {
    if (data.empty()) return 0.0f;

    size_t n = data.size();
    std::sort(data.begin(), data.end());

    if (n % 2 == 0) {
        return (data[n/2 - 1] + data[n/2]) / 2.0f;
    } else {
        return data[n/2];
    }
}

// Calculate percentile (p in range 0-1)
inline float calculatePercentile(std::vector<float> data, float p) {
    if (data.empty()) return 0.0f;

    std::sort(data.begin(), data.end());
    size_t idx = static_cast<size_t>(p * (data.size() - 1));
    return data[idx];
}

// Calculate Euclidean distance between two points
inline float euclideanDistance(float x1, float y1, float x2, float y2) {
    float dx = x2 - x1;
    float dy = y2 - y1;
    return sqrt(dx * dx + dy * dy);
}

#endif
