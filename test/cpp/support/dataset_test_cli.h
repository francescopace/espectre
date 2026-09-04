/*
 * ESPectre - Dataset Test CLI
 *
 * Test-only argument parsing shared by data-backed C++ integration suites.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <cstdio>
#include <cstring>
#include <string>

#include "csi_test_data.h"

namespace espectre::test::dataset_cli {

constexpr int kSkippedExitCode = 77;

struct Options {
  bool aggregate{true};
  bool has_chip{false};
  csi_test_data::ChipType chip{csi_test_data::ChipType::ESP32};
  std::string gate;
};

inline bool parse(int argc, char** argv, const char* expected_gate, Options& options) {
  options = Options{};
  if (argc <= 1) {
    return true;
  }
  options.aggregate = false;
  for (int index = 1; index < argc; ++index) {
    if (std::strcmp(argv[index], "--chip") == 0 && index + 1 < argc) {
      if (!csi_test_data::chip_from_string(argv[++index], options.chip)) {
        std::fprintf(stderr, "Unsupported dataset test chip: %s\n", argv[index]);
        return false;
      }
      options.has_chip = true;
      continue;
    }
    if (std::strcmp(argv[index], "--gate") == 0 && index + 1 < argc) {
      options.gate = argv[++index];
      continue;
    }
    std::fprintf(stderr, "Unknown dataset test argument: %s\n", argv[index]);
    return false;
  }
  if (!options.has_chip || options.gate.empty() ||
      (expected_gate != nullptr && options.gate != expected_gate)) {
    std::fprintf(stderr, "Expected --chip <chip> --gate %s\n",
                 expected_gate == nullptr ? "<gate>" : expected_gate);
    return false;
  }
  return true;
}

inline bool matches(const Options& options, csi_test_data::ChipType chip) {
  return options.aggregate || (options.has_chip && options.chip == chip);
}

inline int no_eligible_dataset(const Options& options) {
  std::printf("No eligible %s dataset for chip %s\n", options.gate.c_str(),
              csi_test_data::chip_name(options.chip));
  return kSkippedExitCode;
}

}  // namespace espectre::test::dataset_cli
