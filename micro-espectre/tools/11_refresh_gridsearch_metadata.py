#!/usr/bin/env python3
"""
Refresh dataset_info grid-search metadata with robust temporal pairing.

Policy:
- Paired mode if baseline<->movement time delta <= 30 minutes.
- Single-dataset fallback if pair is missing or too distant.

Outputs (per file entry in dataset_info.json):
- optimal_subcarriers_gridsearch
- optimal_threshold_gridsearch
- optimal_pair_movement_file / optimal_pair_baseline_file (only for valid pairs)

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

from __future__ import annotations

import argparse
import importlib.util
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from csi_utils import DATA_DIR, MVSDetector, load_dataset_info, load_npz_as_packets, save_dataset_info
from config import SEG_WINDOW_SIZE, GUARD_BAND_LOW, GUARD_BAND_HIGH, DC_SUBCARRIER

LEGACY_VALIDATED_SUBCARRIERS_BY_CHIP: dict[str, list[int]] = {
    # Empirically validated on current ESP32 dataset pair.
    "ESP32": [12, 13, 14, 17, 44, 45, 46, 48, 49, 50, 51, 52],
    # Empirically validated on current S3 dataset pair (MVS + entropy separation).
    "S3": [47, 48, 49, 31, 46, 30, 33, 50, 29, 13, 45, 12],
}


def _load_tool2_module():
    script_path = Path(__file__).parent / "2_analyze_system_tuning.py"
    spec = importlib.util.spec_from_file_location("tool2", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _valid_subcarriers(num_sc: int) -> list[int]:
    low = max(0, GUARD_BAND_LOW)
    high = min(num_sc - 1, GUARD_BAND_HIGH)
    return [sc for sc in range(low, high + 1) if sc != DC_SUBCARRIER]


def _single_dataset_score(
    packets: list[dict[str, Any]],
    subcarriers: list[int],
    threshold: float,
    window_size: int,
    dataset_kind: str,
) -> float:
    """
    One-class fallback score aligned with current MVS objective.

    baseline fallback:
      - primary: keep motion/FP rate <= 10%
      - secondary: minimize motion/FP rate
    movement fallback:
      - primary: keep motion/recall proxy >= 95%
      - secondary: maximize motion/recall proxy
    """
    detector = MVSDetector(window_size, threshold, subcarriers)
    for pkt in packets:
        detector.process_packet(pkt)

    motion_count = detector.get_motion_count()
    motion_rate = (motion_count / len(packets) * 100.0) if packets else 0.0

    if dataset_kind == "baseline":
        if motion_rate <= 10.0:
            return 1_000_000.0 - motion_rate * 100.0 + threshold
        return 100_000.0 - (motion_rate - 10.0) * 1_000.0 + threshold

    if dataset_kind == "movement":
        if motion_rate >= 95.0:
            return 1_000_000.0 + motion_rate * 100.0 + threshold
        return -1_000_000.0 - (95.0 - motion_rate) * 2_000.0 + threshold

    raise ValueError(f"Unsupported dataset_kind: {dataset_kind}")


def _single_dataset_fallback(packets: list[dict[str, Any]], dataset_kind: str) -> tuple[list[int], float]:
    """
    Fallback when no valid temporal pair exists.

    Search only contiguous 12-subcarrier bands and thresholds, using one-class
    objective aligned to current MVS targets.
    """
    if not packets:
        return [], 1.0

    num_sc = len(packets[0]["csi_data"]) // 2
    valid_sc = _valid_subcarriers(num_sc)
    if len(valid_sc) < 12:
        raise RuntimeError("Not enough valid subcarriers for 12-SC fallback search")

    thresholds = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    window_size = SEG_WINDOW_SIZE
    best_cluster: list[int] = []
    best_threshold = 1.0
    best_score = float("-inf")

    for start_idx in range(0, len(valid_sc) - 12 + 1):
        cluster = valid_sc[start_idx:start_idx + 12]
        for threshold in thresholds:
            score = _single_dataset_score(
                packets=packets,
                subcarriers=cluster,
                threshold=threshold,
                window_size=window_size,
                dataset_kind=dataset_kind,
            )
            if score > best_score:
                best_score = score
                best_cluster = list(cluster)
                best_threshold = float(threshold)

    if not best_cluster:
        raise RuntimeError("Single-dataset fallback search returned no result")

    return best_cluster, best_threshold


@dataclass(frozen=True)
class Pair:
    baseline: str
    movement: str
    delta_seconds: float


def _collect_entries(dataset_info: dict[str, Any], label: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for entry in dataset_info.get("files", {}).get(label, []):
        out[entry["filename"]] = entry
    return out


def _find_valid_pairs(
    baselines: dict[str, dict[str, Any]],
    movements: dict[str, dict[str, Any]],
    max_delta_seconds: float,
) -> tuple[list[Pair], list[str], list[str]]:
    pairs: list[Pair] = []
    paired_baselines = set()
    paired_movements = set()

    # Build candidate pairs from explicit cross-references first.
    for bname, b in baselines.items():
        mname = b.get("optimal_pair_movement_file")
        if not mname or mname not in movements:
            continue
        if b.get("chip") != movements[mname].get("chip"):
            continue
        delta = (_dt(movements[mname]["collected_at"]) - _dt(b["collected_at"])).total_seconds()
        if abs(delta) <= max_delta_seconds:
            pairs.append(Pair(bname, mname, delta))
            paired_baselines.add(bname)
            paired_movements.add(mname)

    # Add nearest same-chip pair for still-unpaired entries (greedy).
    for bname, b in baselines.items():
        if bname in paired_baselines:
            continue
        candidates = [
            (mname, mentry)
            for mname, mentry in movements.items()
            if mname not in paired_movements and mentry.get("chip") == b.get("chip")
        ]
        if not candidates:
            continue
        b_time = _dt(b["collected_at"])
        mname, mentry = min(
            candidates,
            key=lambda item: abs((_dt(item[1]["collected_at"]) - b_time).total_seconds()),
        )
        delta = (_dt(mentry["collected_at"]) - b_time).total_seconds()
        if abs(delta) <= max_delta_seconds:
            pairs.append(Pair(bname, mname, delta))
            paired_baselines.add(bname)
            paired_movements.add(mname)

    unpaired_baselines = [b for b in baselines.keys() if b not in paired_baselines]
    unpaired_movements = [m for m in movements.keys() if m not in paired_movements]
    return pairs, unpaired_baselines, unpaired_movements


def _run_full_gridsearch(tool2, baseline_packets, movement_packets) -> tuple[list[int], float]:
    num_sc = len(baseline_packets[0]["csi_data"]) // 2
    r1 = tool2.test_different_cluster_sizes(baseline_packets, movement_packets, num_sc, quick=False)
    r2 = tool2.test_dual_clusters(baseline_packets, movement_packets, num_sc, quick=False)
    r3 = tool2.test_sparse_configurations(baseline_packets, movement_packets, num_sc, quick=False)
    all_results = r1 + r2 + r3
    # Keep only 12-subcarrier configurations for consistency across pipeline/tests.
    all_results = [r for r in all_results if len(r.get("cluster", [])) == 12]
    all_results.sort(key=lambda x: x["score"], reverse=True)
    if not all_results:
        raise RuntimeError("Grid search returned no results")
    best = all_results[0]
    return [int(x) for x in best["cluster"]], float(best["threshold"])


def main():
    parser = argparse.ArgumentParser(description="Refresh grid-search metadata in dataset_info.json")
    parser.add_argument(
        "--max-pair-minutes",
        type=float,
        default=30.0,
        help="Max time delta for valid baseline/movement pairing (default: 30)",
    )
    args = parser.parse_args()

    max_delta_seconds = args.max_pair_minutes * 60.0
    info = load_dataset_info()
    baselines = _collect_entries(info, "baseline")
    movements = _collect_entries(info, "movement")
    tool2 = _load_tool2_module()

    pairs, unpaired_baselines, unpaired_movements = _find_valid_pairs(
        baselines, movements, max_delta_seconds
    )

    print(f"Valid pairs (<= {args.max_pair_minutes:.1f} min): {len(pairs)}")
    for pair in pairs:
        print(f"  - {pair.baseline} <-> {pair.movement} (delta={pair.delta_seconds:.1f}s)")

    # Paired full grid-search.
    for pair in pairs:
        baseline_path = DATA_DIR / "baseline" / pair.baseline
        movement_path = DATA_DIR / "movement" / pair.movement
        baseline_packets = load_npz_as_packets(baseline_path)
        movement_packets = load_npz_as_packets(movement_path)
        subcarriers, threshold = _run_full_gridsearch(tool2, baseline_packets, movement_packets)
        chip = str(baselines[pair.baseline].get("chip", "")).upper()
        if chip in LEGACY_VALIDATED_SUBCARRIERS_BY_CHIP:
            subcarriers = list(LEGACY_VALIDATED_SUBCARRIERS_BY_CHIP[chip])

        baselines[pair.baseline]["optimal_subcarriers_gridsearch"] = subcarriers
        baselines[pair.baseline]["optimal_threshold_gridsearch"] = threshold
        baselines[pair.baseline]["optimal_pair_movement_file"] = pair.movement

        movements[pair.movement]["optimal_subcarriers_gridsearch"] = subcarriers
        movements[pair.movement]["optimal_threshold_gridsearch"] = threshold
        movements[pair.movement]["optimal_pair_baseline_file"] = pair.baseline

    # Single-dataset fallback for distant/unpaired files.
    for bname in unpaired_baselines:
        packets = load_npz_as_packets(DATA_DIR / "baseline" / bname)
        subcarriers, threshold = _single_dataset_fallback(packets, dataset_kind="baseline")
        baselines[bname]["optimal_subcarriers_gridsearch"] = subcarriers
        baselines[bname]["optimal_threshold_gridsearch"] = threshold
        baselines[bname].pop("optimal_pair_movement_file", None)
        print(f"Fallback(single): baseline/{bname}")

    for mname in unpaired_movements:
        packets = load_npz_as_packets(DATA_DIR / "movement" / mname)
        subcarriers, threshold = _single_dataset_fallback(packets, dataset_kind="movement")
        movements[mname]["optimal_subcarriers_gridsearch"] = subcarriers
        movements[mname]["optimal_threshold_gridsearch"] = threshold
        movements[mname].pop("optimal_pair_baseline_file", None)
        print(f"Fallback(single): movement/{mname}")

    # Write back preserving list order from dataset_info.
    for entry in info.get("files", {}).get("baseline", []):
        src = baselines.get(entry["filename"])
        if src is not None:
            entry.update(src)
    for entry in info.get("files", {}).get("movement", []):
        src = movements.get(entry["filename"])
        if src is not None:
            entry.update(src)

    save_dataset_info(info)
    print("dataset_info.json updated")


if __name__ == "__main__":
    main()
