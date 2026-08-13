// SPDX-License-Identifier: GPL-3.0-only
// Commercial licensing available under separate agreement; see LICENSING.md.
/*
 * ESPectre - Lightweight IQR Resource Benchmark
 *
 * Compare the C++ hot paths of the normal- and aggregated-turbulence IQR
 * Lightweight candidates. This is a host microbenchmark: it uses the production
 * feature primitives and models the optimized aggregated path that builds one
 * packet-wide magnitude frame, then derives both turbulence streams from it.
 *
 * Build and run:
 *   c++ -O3 -DNDEBUG -std=c++17 -Isrc/cpp/core \
 *       test/cpp/support/benchmark_lightweight_iqr_resources.cpp \
 *       src/cpp/core/filters.cpp \
 *       -o /tmp/benchmark_lightweight_iqr_resources
 *   /tmp/benchmark_lightweight_iqr_resources
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "csi_features.h"
#include "csi_format.h"
#include "filters.h"
#include "utils.h"

namespace {

using Clock = std::chrono::steady_clock;

constexpr uint16_t kWindow = 100U;
constexpr uint32_t kPacketRate = 100U;
constexpr uint32_t kEvaluationRate = 4U;
constexpr size_t kCorpusPackets = 4096U;
constexpr size_t kSamples = 15U;
constexpr size_t kPacketIterations = 200000U;
constexpr size_t kEvaluationIterations = 200000U;

volatile double benchmark_sink = 0.0;

struct Summary {
  double p10_ns;
  double median_ns;
  double p90_ns;
};

Summary summarize(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const auto percentile = [&values](double quantile) {
    const double position = quantile * static_cast<double>(values.size() - 1U);
    const size_t lower = static_cast<size_t>(position);
    const size_t upper = std::min(lower + 1U, values.size() - 1U);
    const double fraction = position - static_cast<double>(lower);
    return values[lower] * (1.0 - fraction) + values[upper] * fraction;
  };
  return Summary{percentile(0.10), percentile(0.50), percentile(0.90)};
}

template <typename Function>
Summary measure(size_t iterations, Function&& function) {
  for (size_t i = 0U; i < iterations / 10U; ++i) {
    benchmark_sink += function(i);
  }
  std::vector<double> samples;
  samples.reserve(kSamples);
  for (size_t sample = 0U; sample < kSamples; ++sample) {
    const auto started = Clock::now();
    double local_sink = 0.0;
    for (size_t i = 0U; i < iterations; ++i) {
      local_sink += function(i + sample);
    }
    const auto elapsed = std::chrono::duration<double, std::nano>(
        Clock::now() - started).count();
    benchmark_sink += local_sink;
    samples.push_back(elapsed / static_cast<double>(iterations));
  }
  return summarize(std::move(samples));
}

using Packet = std::array<int8_t, espectre::HT20_CSI_LEN>;

std::vector<Packet> make_packets() {
  std::mt19937 generator(20260813U);
  std::uniform_int_distribution<int> sample(-96, 96);
  std::vector<Packet> packets(kCorpusPackets);
  for (Packet& packet : packets) {
    for (int8_t& value : packet) {
      value = static_cast<int8_t>(sample(generator));
    }
  }
  return packets;
}

uint8_t fill_packet_amplitudes(const Packet& packet, float* out) {
  for (uint8_t bin = 0U; bin < espectre::HT20_NUM_SUBCARRIERS; ++bin) {
    const float imag = static_cast<float>(packet[bin * 2U]);
    const float real = static_cast<float>(packet[bin * 2U + 1U]);
    out[bin] = std::sqrt(real * real + imag * imag);
  }
  return espectre::HT20_NUM_SUBCARRIERS;
}

uint8_t select_normal_amplitudes(const float* packet_amplitudes,
                                 uint8_t packet_count, float* out) {
  uint8_t written = 0U;
  for (uint8_t i = 0U; i < espectre::HT20_SELECTED_BAND_SIZE; ++i) {
    const uint8_t bin = espectre::DEFAULT_SUBCARRIERS[i];
    if (bin < packet_count) {
      out[written++] = packet_amplitudes[bin];
    }
  }
  return written;
}

uint8_t select_aggregated_amplitudes(const float* packet_amplitudes,
                                     uint8_t packet_count, float* out) {
  constexpr uint8_t width = espectre::TURB_IQR_AGGREGATION_WIDTH;
  constexpr int half = static_cast<int>((width - 1U) / 2U);
  uint8_t written = 0U;
  for (uint8_t i = 0U; i < espectre::HT20_SELECTED_BAND_SIZE; ++i) {
    int low = static_cast<int>(espectre::DEFAULT_SUBCARRIERS[i]) - half;
    int high = low + static_cast<int>(width) - 1;
    if (low < espectre::HT20_GUARD_BAND_LOW) {
      low = espectre::HT20_GUARD_BAND_LOW;
      high = espectre::HT20_GUARD_BAND_LOW + width - 1;
    }
    if (high > espectre::HT20_GUARD_BAND_HIGH) {
      low = espectre::HT20_GUARD_BAND_HIGH - width + 1;
      high = espectre::HT20_GUARD_BAND_HIGH;
    }
    float total = 0.0f;
    uint8_t count = 0U;
    for (int bin = low; bin <= high; ++bin) {
      if (bin == espectre::HT20_DC_SUBCARRIER || bin < 0 ||
          bin >= packet_count) {
        continue;
      }
      total += packet_amplitudes[bin];
      ++count;
    }
    if (count > 0U) {
      out[written++] = total / static_cast<float>(count);
    }
  }
  return written;
}

struct TurbulencePacketState {
  std::array<float, kWindow> ring{};
  uint16_t slot{0U};
  espectre::hampel_filter_state_t hampel{};
  espectre::lowpass_filter_state_t lowpass{};

  TurbulencePacketState() {
    espectre::hampel_turbulence_init(
        &hampel, espectre::HAMPEL_TURBULENCE_WINDOW_DEFAULT,
        espectre::HAMPEL_TURBULENCE_THRESHOLD_DEFAULT, true);
    espectre::lowpass_filter_init(
        &lowpass, espectre::LOWPASS_CUTOFF_DEFAULT,
        espectre::LOWPASS_SAMPLE_RATE, false);
  }

  float push(float value) {
    const float hampel_value =
        espectre::hampel_filter_turbulence(&hampel, value);
    const float filtered =
        espectre::lowpass_filter_apply(&lowpass, hampel_value);
    ring[slot] = filtered;
    slot = static_cast<uint16_t>((slot + 1U) % kWindow);
    return filtered;
  }
};

float normal_packet_turbulence(const Packet& packet,
                               TurbulencePacketState& normal_state) {
  float amplitudes[espectre::HT20_SELECTED_BAND_SIZE]{};
  const uint8_t count = espectre::extract_subcarrier_amplitudes(
      packet.data(), packet.size(), espectre::DEFAULT_SUBCARRIERS,
      espectre::HT20_SELECTED_BAND_SIZE, amplitudes,
      espectre::HT20_SELECTED_BAND_SIZE);
  const float turbulence =
      espectre::calculate_spatial_turbulence_from_amplitudes(
          amplitudes, count);
  return normal_state.push(turbulence);
}

float aggregated_packet_turbulence(const Packet& packet, float* packet_frame,
                                   TurbulencePacketState& normal_state,
                                   TurbulencePacketState& aggregated_state) {
  const uint8_t packet_count = fill_packet_amplitudes(packet, packet_frame);
  float normal[espectre::HT20_SELECTED_BAND_SIZE]{};
  float aggregated[espectre::HT20_SELECTED_BAND_SIZE]{};
  const uint8_t normal_count =
      select_normal_amplitudes(packet_frame, packet_count, normal);
  const uint8_t aggregated_count =
      select_aggregated_amplitudes(packet_frame, packet_count, aggregated);
  const float normal_turbulence =
      espectre::calculate_spatial_turbulence_from_amplitudes(
          normal, normal_count);
  const float aggregated_turbulence =
      espectre::calculate_spatial_turbulence_from_amplitudes(
          aggregated, aggregated_count);
  return normal_state.push(normal_turbulence) +
         aggregated_state.push(aggregated_turbulence);
}

float iqr_over_mean(float* sorted, uint16_t count, float mean) {
  std::sort(sorted, sorted + count);
  const float q25 = espectre::percentile_from_sorted(sorted, count, 0.25f);
  const float q75 = espectre::percentile_from_sorted(sorted, count, 0.75f);
  return (q75 - q25) / std::max(std::fabs(mean), 1e-6f);
}

float fuse(float autocorr, float iqr) {
  constexpr float center_autocorr = 0.39f;
  constexpr float scale_autocorr = 0.38f;
  constexpr float center_iqr = 0.21f;
  constexpr float scale_iqr = 0.17f;
  constexpr float weight_autocorr = 5.8f;
  constexpr float weight_iqr = 4.1f;
  constexpr float intercept = 0.8f;
  const float logit = intercept +
      weight_autocorr * ((autocorr - center_autocorr) / scale_autocorr) +
      weight_iqr * ((iqr - center_iqr) / scale_iqr);
  return 1.0f / (1.0f + std::exp(-logit));
}

float evaluate_normal_iqr(const float* normal, float* scratch) {
  std::memcpy(scratch, normal, kWindow * sizeof(float));
  const espectre::MeanVariance moments =
      espectre::calculate_mean_variance_two_pass(scratch, kWindow);
  const float autocorr = espectre::calc_autocorrelation(
      scratch, kWindow, moments.mean, moments.variance, 1U);
  return fuse(autocorr, iqr_over_mean(scratch, kWindow, moments.mean));
}

float evaluate_aggregated_iqr(const float* normal, const float* aggregated,
                              float* scratch) {
  std::memcpy(scratch, normal, kWindow * sizeof(float));
  const espectre::MeanVariance moments =
      espectre::calculate_mean_variance_two_pass(scratch, kWindow);
  const float autocorr = espectre::calc_autocorrelation(
      scratch, kWindow, moments.mean, moments.variance, 1U);
  std::memcpy(scratch, aggregated, kWindow * sizeof(float));
  const float aggregated_mean = espectre::calculate_mean(scratch, kWindow);
  return fuse(
      autocorr, iqr_over_mean(scratch, kWindow, aggregated_mean));
}

std::array<float, kWindow> make_series(uint32_t seed) {
  std::mt19937 generator(seed);
  std::uniform_real_distribution<float> noise(-0.07f, 0.07f);
  std::array<float, kWindow> values{};
  for (uint16_t i = 0U; i < kWindow; ++i) {
    const float x = static_cast<float>(i);
    values[i] = 0.20f + 0.04f * std::sin(0.17f * x) + noise(generator);
  }
  return values;
}

void print_summary(const char* label, const Summary& summary) {
  std::cout << std::left << std::setw(30) << label << std::right
            << std::fixed << std::setprecision(1)
            << " p10=" << std::setw(8) << summary.p10_ns
            << " ns  median=" << std::setw(8) << summary.median_ns
            << " ns  p90=" << std::setw(8) << summary.p90_ns << " ns\n";
}

}  // namespace

int main() {
  const std::vector<Packet> packets = make_packets();
  std::array<float, espectre::HT20_NUM_SUBCARRIERS> packet_frame{};
  std::array<float, kWindow> scratch{};
  const std::array<float, kWindow> normal = make_series(20260814U);
  const std::array<float, kWindow> aggregated = make_series(20260815U);
  TurbulencePacketState normal_packet_state;
  TurbulencePacketState aggregated_normal_packet_state;
  TurbulencePacketState aggregated_packet_state;

  const Summary normal_packet = measure(kPacketIterations, [&](size_t i) {
    return normal_packet_turbulence(
        packets[i % packets.size()], normal_packet_state);
  });
  const Summary aggregated_packet = measure(kPacketIterations, [&](size_t i) {
    return aggregated_packet_turbulence(
        packets[i % packets.size()], packet_frame.data(),
        aggregated_normal_packet_state,
        aggregated_packet_state);
  });
  const Summary normal_evaluation = measure(kEvaluationIterations, [&](size_t) {
    return evaluate_normal_iqr(normal.data(), scratch.data());
  });
  const Summary aggregated_evaluation = measure(
      kEvaluationIterations, [&](size_t) {
        return evaluate_aggregated_iqr(
            normal.data(), aggregated.data(), scratch.data());
      });
  const Summary pure_fusion = measure(kEvaluationIterations * 5U, [&](size_t i) {
    return fuse(0.35f + static_cast<float>(i % 17U) * 0.001f,
                0.20f + static_cast<float>(i % 13U) * 0.001f);
  });

  const size_t common_dynamic_bytes =
      2U * static_cast<size_t>(kWindow) * sizeof(float);
  const size_t aggregated_ring_bytes =
      static_cast<size_t>(kWindow) * sizeof(float);
  const size_t aggregated_filter_bytes =
      sizeof(espectre::hampel_filter_state_t) +
      sizeof(espectre::lowpass_filter_state_t);
  const double normal_cpu_us_per_second =
      (normal_packet.median_ns * kPacketRate +
       normal_evaluation.median_ns * kEvaluationRate) / 1000.0;
  const double aggregated_cpu_us_per_second =
      (aggregated_packet.median_ns * kPacketRate +
       aggregated_evaluation.median_ns * kEvaluationRate) / 1000.0;

  std::cout << "Lightweight IQR C++ host microbenchmark\n"
            << "window=" << kWindow << ", packet_rate=" << kPacketRate
            << " pps, evaluation_rate=" << kEvaluationRate << " Hz\n\n";
  print_summary("normal IQR packet path", normal_packet);
  print_summary("aggregated IQR packet path", aggregated_packet);
  print_summary("normal IQR evaluation", normal_evaluation);
  print_summary("aggregated IQR evaluation", aggregated_evaluation);
  print_summary("pure two-feature fusion", pure_fusion);

  std::cout << "\nModeled requested persistent feature state\n"
            << "  normal IQR:     " << common_dynamic_bytes << " B dynamic\n"
            << "  aggregated IQR: "
            << common_dynamic_bytes + aggregated_ring_bytes
            << " B dynamic + " << aggregated_filter_bytes
            << " B fixed filter state\n"
            << "  delta:          "
            << aggregated_ring_bytes + aggregated_filter_bytes << " B\n";
  std::cout << "\nModeled steady-state host CPU at nominal cadence\n"
            << std::fixed << std::setprecision(2)
            << "  normal IQR:     " << normal_cpu_us_per_second << " us/s\n"
            << "  aggregated IQR: " << aggregated_cpu_us_per_second << " us/s\n"
            << "  ratio:          "
            << aggregated_cpu_us_per_second / normal_cpu_us_per_second
            << "x\n"
            << "\nbenchmark_sink=" << benchmark_sink << "\n";
  return 0;
}
