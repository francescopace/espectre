// SPDX-License-Identifier: GPL-3.0-only
// Commercial licensing available under separate agreement; see LICENSING.md.
/*
 * ESPectre - Production Detector Resource Benchmark
 *
 * Measures the current C++ Classic and ML detector implementations on the
 * host. The report generator compiles this support source once per revision,
 * then executes the cached binary for every report so timing is always fresh.
 */

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <new>
#include <random>
#include <string>
#include <vector>

#include "classic_detector.h"
#include "csi_format.h"
#include "detector_limits.h"
#include "ml_detector.h"

namespace allocation_tracker {

struct alignas(std::max_align_t) Header {
  std::size_t size;
};

std::atomic<std::size_t> live_bytes{0U};
std::atomic<std::size_t> peak_bytes{0U};

void observe_peak(std::size_t value) {
  std::size_t peak = peak_bytes.load(std::memory_order_relaxed);
  while (value > peak &&
         !peak_bytes.compare_exchange_weak(peak, value,
                                           std::memory_order_relaxed)) {
  }
}

void* allocate(std::size_t size) {
  if (size == 0U) size = 1U;
  void* raw = std::malloc(sizeof(Header) + size);
  if (raw == nullptr) throw std::bad_alloc();
  auto* header = static_cast<Header*>(raw);
  header->size = size;
  const std::size_t live = live_bytes.fetch_add(
                               size, std::memory_order_relaxed) +
                           size;
  observe_peak(live);
  return header + 1;
}

void release(void* pointer) noexcept {
  if (pointer == nullptr) return;
  auto* header = static_cast<Header*>(pointer) - 1;
  live_bytes.fetch_sub(header->size, std::memory_order_relaxed);
  std::free(header);
}

}  // namespace allocation_tracker

void* operator new(std::size_t size) {
  return allocation_tracker::allocate(size);
}

void* operator new[](std::size_t size) {
  return allocation_tracker::allocate(size);
}

void operator delete(void* pointer) noexcept {
  allocation_tracker::release(pointer);
}

void operator delete[](void* pointer) noexcept {
  allocation_tracker::release(pointer);
}

void operator delete(void* pointer, std::size_t) noexcept {
  allocation_tracker::release(pointer);
}

void operator delete[](void* pointer, std::size_t) noexcept {
  allocation_tracker::release(pointer);
}

namespace {

using Clock = std::chrono::steady_clock;
using Packet = std::array<int8_t, espectre::HT20_CSI_LEN>;

constexpr std::size_t kCorpusPackets = 2048U;
constexpr std::size_t kSamples = 11U;
constexpr std::size_t kPacketIterations = 80000U;
constexpr std::size_t kInferenceIterations = 80000U;
constexpr double kPacketRate = 100.0;
constexpr double kInferenceRate = 4.0;

volatile double benchmark_sink = 0.0;

struct TimingSummary {
  double median_ns;
  double p90_ns;
};

struct DetectorSummary {
  const char* name;
  std::size_t object_bytes;
  std::size_t heap_bytes;
  std::size_t persistent_bytes;
  std::size_t transient_heap_bytes;
  TimingSummary packet;
  TimingSummary inference;
  double cpu_us_per_second;
};

TimingSummary summarize(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const auto percentile = [&values](double quantile) {
    const double position = quantile * static_cast<double>(values.size() - 1U);
    const std::size_t lower = static_cast<std::size_t>(position);
    const std::size_t upper = std::min(lower + 1U, values.size() - 1U);
    const double fraction = position - static_cast<double>(lower);
    return values[lower] * (1.0 - fraction) + values[upper] * fraction;
  };
  return TimingSummary{percentile(0.50), percentile(0.90)};
}

template <typename Function>
TimingSummary measure(std::size_t iterations, Function&& function) {
  for (std::size_t i = 0U; i < iterations / 10U; ++i) {
    benchmark_sink += function(i);
  }
  std::vector<double> samples;
  samples.reserve(kSamples);
  for (std::size_t sample = 0U; sample < kSamples; ++sample) {
    const auto started = Clock::now();
    double local_sink = 0.0;
    for (std::size_t i = 0U; i < iterations; ++i) {
      local_sink += function(i + sample);
    }
    const double elapsed_ns =
        std::chrono::duration<double, std::nano>(Clock::now() - started)
            .count();
    benchmark_sink += local_sink;
    samples.push_back(elapsed_ns / static_cast<double>(iterations));
  }
  return summarize(std::move(samples));
}

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

template <typename Detector>
DetectorSummary benchmark_detector(const char* name,
                                   const std::vector<Packet>& packets) {
  const std::size_t live_before =
      allocation_tracker::live_bytes.load(std::memory_order_relaxed);
  Detector detector(espectre::DETECTOR_DEFAULT_WINDOW_SIZE);
  const std::size_t live_after_construction =
      allocation_tracker::live_bytes.load(std::memory_order_relaxed);
  const std::size_t heap_bytes = live_after_construction - live_before;

  uint64_t timestamp_us = 0U;
  for (std::size_t i = 0U; i < espectre::DETECTOR_DEFAULT_WINDOW_SIZE * 2U;
       ++i) {
    detector.set_packet_timestamp_us(timestamp_us);
    detector.process_packet(
        packets[i % packets.size()].data(), packets[i % packets.size()].size(),
        espectre::DEFAULT_SUBCARRIERS, espectre::HT20_SELECTED_BAND_SIZE, -55);
    timestamp_us += 10000U;
  }
  detector.update_state();

  allocation_tracker::peak_bytes.store(
      allocation_tracker::live_bytes.load(std::memory_order_relaxed),
      std::memory_order_relaxed);
  detector.set_packet_timestamp_us(timestamp_us);
  detector.process_packet(packets.front().data(), packets.front().size(),
                          espectre::DEFAULT_SUBCARRIERS,
                          espectre::HT20_SELECTED_BAND_SIZE, -55);
  detector.update_state();
  const std::size_t live_now =
      allocation_tracker::live_bytes.load(std::memory_order_relaxed);
  const std::size_t transient_heap_bytes =
      allocation_tracker::peak_bytes.load(std::memory_order_relaxed) - live_now;

  const TimingSummary packet = measure(kPacketIterations, [&](std::size_t i) {
    detector.set_packet_timestamp_us(timestamp_us + i * 10000U);
    detector.process_packet(
        packets[i % packets.size()].data(), packets[i % packets.size()].size(),
        espectre::DEFAULT_SUBCARRIERS, espectre::HT20_SELECTED_BAND_SIZE, -55);
    return static_cast<double>(detector.get_last_turbulence());
  });
  const TimingSummary inference =
      measure(kInferenceIterations, [&](std::size_t) {
        detector.update_state();
        return static_cast<double>(detector.get_motion_metric());
      });
  const double cpu_us_per_second =
      (packet.median_ns * kPacketRate + inference.median_ns * kInferenceRate) /
      1000.0;
  return DetectorSummary{name,
                         sizeof(Detector),
                         heap_bytes,
                         sizeof(Detector) + heap_bytes,
                         transient_heap_bytes,
                         packet,
                         inference,
                         cpu_us_per_second};
}

void print_detector(const DetectorSummary& summary) {
  std::cout << "    \"" << summary.name << "\": {\n"
            << "      \"object_bytes\": " << summary.object_bytes << ",\n"
            << "      \"heap_bytes\": " << summary.heap_bytes << ",\n"
            << "      \"persistent_bytes\": " << summary.persistent_bytes
            << ",\n"
            << "      \"transient_heap_bytes\": "
            << summary.transient_heap_bytes << ",\n"
            << "      \"packet_median_ns\": " << summary.packet.median_ns
            << ",\n"
            << "      \"packet_p90_ns\": " << summary.packet.p90_ns << ",\n"
            << "      \"inference_median_ns\": "
            << summary.inference.median_ns << ",\n"
            << "      \"inference_p90_ns\": " << summary.inference.p90_ns
            << ",\n"
            << "      \"cpu_us_per_second\": "
            << summary.cpu_us_per_second << "\n"
            << "    }";
}

}  // namespace

int main() {
  const std::vector<Packet> packets = make_packets();
  const DetectorSummary classic =
      benchmark_detector<espectre::ClassicDetector>("classic", packets);
  const DetectorSummary ml =
      benchmark_detector<espectre::MLDetector>("ml", packets);

  std::cout << std::fixed << std::setprecision(3)
            << "{\n"
            << "  \"benchmark\": \"production_cpp_host\",\n"
            << "  \"window_packets\": "
            << espectre::DETECTOR_DEFAULT_WINDOW_SIZE << ",\n"
            << "  \"packet_rate_hz\": " << kPacketRate << ",\n"
            << "  \"inference_rate_hz\": " << kInferenceRate << ",\n"
            << "  \"detectors\": {\n";
  print_detector(classic);
  std::cout << ",\n";
  print_detector(ml);
  std::cout << "\n  },\n"
            << "  \"benchmark_sink\": " << benchmark_sink << "\n"
            << "}\n";
  return std::isfinite(benchmark_sink) ? 0 : 1;
}
