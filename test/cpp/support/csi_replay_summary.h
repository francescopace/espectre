/*
 * ESPectre - Summary table helpers for replay-based C++ tests
 *
 * Shared formatting for chip-level Classic/ML summary tables emitted by native
 * integration suites.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */
#pragma once

#include <cstdio>
#include <vector>

namespace espectre::test::summary {

struct DetectorSummaryCell {
  bool valid{false};
  float recall{0.0f};
  float fp_rate{0.0f};
};

struct DualDetectorSummaryRow {
  const char* chip_name{nullptr};
  int dataset_count{0};
  DetectorSummaryCell classic{};
  DetectorSummaryCell ml{};
};

inline void format_recall_fp(char* buffer,
                             size_t buffer_size,
                             const DetectorSummaryCell& cell) {
  if (!cell.valid) {
    std::snprintf(buffer, buffer_size, "N/A");
    return;
  }
  std::snprintf(buffer, buffer_size, "%.1f%% R, %.1f%% FP", cell.recall, cell.fp_rate);
}

inline void print_dual_detector_summary_table(
    const char* title,
    const std::vector<DualDetectorSummaryRow>& rows,
    const char* footer_line = "Legend: R = Recall, FP = False Positive Rate",
    const char* extra_footer_line = nullptr) {
  std::printf("\n");
  std::printf("================================================================================\n");
  std::printf("%s\n", title);
  std::printf("================================================================================\n");
  std::printf("| Chip   | Datasets | Classic                 | ML                      |\n");
  std::printf("|--------|----------|-------------------------|-------------------------|\n");

  for (const DualDetectorSummaryRow& row : rows) {
    char classic_str[32] = "N/A";
    char ml_str[32] = "N/A";
    format_recall_fp(classic_str, sizeof(classic_str), row.classic);
    format_recall_fp(ml_str, sizeof(ml_str), row.ml);
    std::printf("| %-6s | %8d | %-23s | %-23s |\n",
                row.chip_name != nullptr ? row.chip_name : "Unknown",
                row.dataset_count,
                classic_str,
                ml_str);
  }

  std::printf("--------------------------------------------------------------------------------\n");
  if (footer_line != nullptr) {
    std::printf("%s\n", footer_line);
  }
  if (extra_footer_line != nullptr) {
    std::printf("%s\n", extra_footer_line);
  }
}

}  // namespace espectre::test::summary
