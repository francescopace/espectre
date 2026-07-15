"""
ESPectre - Dataset Metadata

Dataset metadata helpers for tool-side workflows.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .bootstrap import setup_paths
from .repo_paths import data_dir

setup_paths()

try:
    import config
except ImportError:
    import src.config as config

try:
    from classic_detector import ClassicDetector
    from threshold import (
        StartupThresholdCalibrator,
        get_detector_auto_factor,
        get_detector_startup_gate,
    )
except ImportError:  # pragma: no cover
    from src.classic_detector import ClassicDetector
    from src.threshold import (
        StartupThresholdCalibrator,
        get_detector_auto_factor,
        get_detector_startup_gate,
    )


DATASET_FORMAT_VERSION = "1.2"
DATA_DIR = data_dir()
DATASET_INFO_FILE = DATA_DIR / "dataset_info.json"


@dataclass(frozen=True)
class ResolvedDataset:
    """One dataset entry resolved from dataset_info plus optional pair metadata."""

    label: str
    entry: Dict[str, Any]
    path: Path
    counterpart_label: Optional[str] = None
    counterpart_entry: Optional[Dict[str, Any]] = None
    counterpart_path: Optional[Path] = None


@dataclass(frozen=True)
class ResolvedPair:
    """Resolved static_presence/motion pair.

    Detection thresholds are intentionally not resolved here: they are
    detector-specific, so consumers replay the startup calibration of the
    detector they evaluate (`estimate_runtime_threshold` for classic, the
    variance sweep calibration for the moving-variance baseline) on the static
    capture of the pair.
    """

    static_presence: ResolvedDataset
    motion: ResolvedDataset
    chip: str
    num_subcarriers: int


def load_dataset_info() -> Dict[str, Any]:
    """Load or create dataset info."""
    if DATASET_INFO_FILE.exists():
        with open(DATASET_INFO_FILE, "r", encoding="utf-8") as handle:
            return json.load(handle)

    now = datetime.now().isoformat()
    return {
        "format_version": DATASET_FORMAT_VERSION,
        "created_at": now,
        "updated_at": now,
        "labels": {},
        "files": {},
        "contributors": [],
        "environments": [],
    }


def save_dataset_info(info: Dict[str, Any]) -> None:
    """Persist dataset info to disk."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATASET_INFO_FILE, "w", encoding="utf-8") as handle:
        json.dump(info, handle, indent=2)


def get_dataset_stats() -> Dict[str, Any]:
    """Get dataset statistics by scanning dataset directories."""
    stats = {
        "labels": {},
        "total_samples": 0,
        "total_packets": 0,
        "labels_count": 0,
    }
    if not DATA_DIR.exists():
        return stats

    for label_dir in DATA_DIR.iterdir():
        if label_dir.is_dir() and not label_dir.name.startswith("."):
            samples = list(label_dir.glob("*.npz"))
            if not samples:
                continue
            stats["labels"][label_dir.name] = {"samples": len(samples)}
            stats["total_samples"] += len(samples)
            stats["labels_count"] += 1

    return stats


def _dataset_sort_key(entry: Dict[str, Any]) -> Tuple[str, str]:
    """Sort entries newest-first by collected_at, then by filename."""
    return (str(entry.get("collected_at") or ""), str(entry.get("filename") or ""))


def _entry_matches_filters(
    entry: Dict[str, Any],
    *,
    label: Optional[str] = None,
    chip: Optional[str] = None,
    environment: Optional[str] = None,
    num_sc: Optional[int] = None,
) -> bool:
    """Return True when one dataset_info entry matches the requested filters."""
    if label is not None and str(label) != str(entry.get("_label", label)):
        return False
    if chip is not None and str(entry.get("chip", "")).upper() != str(chip).upper():
        return False
    if environment is not None and str(entry.get("environment", "")) != str(environment):
        return False
    if num_sc is not None and int(entry.get("subcarriers", 0) or 0) != int(num_sc):
        return False
    return True


def _resolve_entry_path(label: str, entry: Dict[str, Any]) -> Path:
    """Resolve one dataset_info entry to its NPZ path."""
    relative_path = entry.get("relative_path")
    if relative_path:
        return DATA_DIR / str(relative_path)
    filename = entry.get("filename")
    if not filename:
        raise KeyError("filename")
    return DATA_DIR / str(label) / str(filename)


def _threshold_mode_from_config() -> str:
    """Return the shared startup-threshold mode used by the repo config."""
    return str(config.SEG_THRESHOLD) if isinstance(config.SEG_THRESHOLD, str) else "auto"


def estimate_runtime_threshold(
    packets: Iterable[Dict[str, Any]],
    *,
    threshold_mode: Optional[str] = None,
    selected_subcarriers: Optional[Iterable[int]] = None,
) -> Optional[float]:
    """Replay the classic startup calibration and return a production-aligned threshold."""
    selected_mode = _threshold_mode_from_config() if threshold_mode is None else str(threshold_mode)
    detector = ClassicDetector(
        window_size=config.SEG_WINDOW_SIZE,
        threshold=1.0,
        enable_lowpass=config.ENABLE_LOWPASS_FILTER,
        lowpass_cutoff=config.LOWPASS_CUTOFF,
        enable_hampel=config.ENABLE_HAMPEL_FILTER,
        hampel_window=config.HAMPEL_WINDOW,
        hampel_threshold=config.HAMPEL_THRESHOLD,
    )
    calibrator = StartupThresholdCalibrator(
        config.CALIBRATION_BUFFER_SIZE,
        auto_factor=get_detector_auto_factor(detector),
        gate_enabled=get_detector_startup_gate(detector),
    )
    band = config.DEFAULT_SUBCARRIERS if selected_subcarriers is None else tuple(selected_subcarriers)
    packets_since_evaluation = 0
    for pkt in packets:
        csi_data = pkt["csi_data"] if isinstance(pkt, dict) else pkt
        detector.process_packet(csi_data, band)
        packets_since_evaluation += 1
        if packets_since_evaluation < config.EVALUATION_INTERVAL:
            continue
        detector.update_state()
        calibrator.observe_detector(
            detector,
            packet_weight=packets_since_evaluation,
        )
        packets_since_evaluation = 0
        if calibrator.is_complete():
            break
    if not calibrator.is_successful():
        return None
    threshold, _ = calibrator.calculate_threshold(selected_mode)
    return float(threshold)


def build_calibrated_classic_detector(
    packets: Iterable[Dict[str, Any]],
    *,
    threshold_mode: Optional[str] = None,
    selected_subcarriers: Optional[Iterable[int]] = None,
    threshold: float = 1.0,
) -> Optional[Tuple[ClassicDetector, float]]:
    """
    Return a ClassicDetector calibrated exactly like the production startup flow.

    The returned detector has its startup-calibrated threshold applied and its
    frozen variance floor preserved across the warm reset that follows calibration.
    """
    selected_mode = _threshold_mode_from_config() if threshold_mode is None else str(threshold_mode)
    detector = ClassicDetector(
        window_size=config.SEG_WINDOW_SIZE,
        threshold=threshold,
        enable_lowpass=config.ENABLE_LOWPASS_FILTER,
        lowpass_cutoff=config.LOWPASS_CUTOFF,
        enable_hampel=config.ENABLE_HAMPEL_FILTER,
        hampel_window=config.HAMPEL_WINDOW,
        hampel_threshold=config.HAMPEL_THRESHOLD,
    )
    calibrator = StartupThresholdCalibrator(
        config.CALIBRATION_BUFFER_SIZE,
        auto_factor=get_detector_auto_factor(detector),
        gate_enabled=get_detector_startup_gate(detector),
    )
    band = config.DEFAULT_SUBCARRIERS if selected_subcarriers is None else tuple(selected_subcarriers)
    packets_since_evaluation = 0
    for pkt in packets:
        csi_data = pkt["csi_data"] if isinstance(pkt, dict) else pkt
        detector.process_packet(csi_data, band)
        packets_since_evaluation += 1
        if packets_since_evaluation < config.EVALUATION_INTERVAL:
            continue
        detector.update_state()
        calibrator.observe_detector(
            detector,
            packet_weight=packets_since_evaluation,
        )
        packets_since_evaluation = 0
        if calibrator.is_complete():
            break
    if not calibrator.is_successful():
        return None
    startup_threshold, _ = calibrator.calculate_threshold(selected_mode)
    if hasattr(calibrator, "get_floor_snapshot") and hasattr(detector, "apply_startup_floor"):
        floor_value, vote_enabled, sample_count = calibrator.get_floor_snapshot()
        detector.apply_startup_floor(floor_value, vote_enabled, sample_count)
    detector.set_adaptive_threshold(float(startup_threshold))
    detector.reset()
    return detector, float(startup_threshold)


def resolve_dataset_selection(
    dataset: Optional[str] = None,
    *,
    label: Optional[str] = None,
    chip: Optional[str] = None,
    environment: Optional[str] = None,
    num_sc: Optional[int] = 64,
    require_pair: bool = False,
    prefer_latest: bool = True,
    dataset_info: Optional[Dict[str, Any]] = None,
) -> ResolvedDataset:
    """Resolve one dataset entry from dataset_info, optionally by filename or stem."""
    info = load_dataset_info() if dataset_info is None else dataset_info
    files_section = info.get("files", {})

    dataset_key = str(dataset).strip() if dataset is not None else None
    dataset_key_lower = dataset_key.lower() if dataset_key else None
    candidates: List[Tuple[str, Dict[str, Any]]] = []
    for current_label, entries in files_section.items():
        for raw_entry in entries:
            entry = dict(raw_entry)
            entry["_label"] = current_label
            if not _entry_matches_filters(
                entry,
                label=label,
                chip=chip,
                environment=environment,
                num_sc=num_sc,
            ):
                continue
            if dataset_key_lower:
                filename = str(entry.get("filename") or "")
                stem = Path(filename).stem
                dataset_id = (
                    f"{str(entry.get('chip', '')).lower()}_"
                    f"{str(entry.get('environment', 'unknown'))}_{stem}"
                )
                if dataset_key_lower not in {
                    filename.lower(),
                    stem.lower(),
                    dataset_id.lower(),
                }:
                    continue
            if require_pair:
                if current_label not in ("static_presence", "motion"):
                    continue
                pair_field = (
                    "optimal_pair_motion_file"
                    if current_label == "static_presence"
                    else "optimal_pair_static_presence_file"
                )
                if not entry.get(pair_field):
                    continue
            candidates.append((current_label, entry))

    if not candidates:
        detail = dataset_key or chip or label or "requested filters"
        raise FileNotFoundError(f"No dataset found in dataset_info.json for {detail}")

    if len(candidates) > 1:
        candidates.sort(key=lambda item: _dataset_sort_key(item[1]), reverse=prefer_latest)

    resolved_label, entry = candidates[0]
    resolved_path = _resolve_entry_path(resolved_label, entry)
    counterpart_label = None
    counterpart_entry = None
    counterpart_path = None
    if resolved_label == "static_presence":
        counterpart_label = "motion"
        counterpart_name = entry.get("optimal_pair_motion_file")
    elif resolved_label == "motion":
        counterpart_label = "static_presence"
        counterpart_name = entry.get("optimal_pair_static_presence_file")
    else:
        counterpart_name = None

    if counterpart_label and counterpart_name:
        for raw_entry in files_section.get(counterpart_label, []):
            if raw_entry.get("filename") == counterpart_name:
                counterpart_entry = dict(raw_entry)
                counterpart_path = _resolve_entry_path(counterpart_label, counterpart_entry)
                break

    return ResolvedDataset(
        label=resolved_label,
        entry=entry,
        path=resolved_path,
        counterpart_label=counterpart_label,
        counterpart_entry=counterpart_entry,
        counterpart_path=counterpart_path,
    )


def resolve_explicit_pair(
    dataset: Optional[str] = None,
    *,
    chip: Optional[str] = None,
    environment: Optional[str] = None,
    num_sc: int = 64,
    prefer_latest: bool = True,
    dataset_info: Optional[Dict[str, Any]] = None,
) -> ResolvedPair:
    """Resolve one explicit static_presence/motion pair from dataset metadata."""
    resolved = resolve_dataset_selection(
        dataset,
        chip=chip,
        environment=environment,
        num_sc=num_sc,
        require_pair=True,
        prefer_latest=prefer_latest,
        dataset_info=dataset_info,
    )
    if resolved.label not in ("static_presence", "motion"):
        raise ValueError(
            f"Dataset '{resolved.path.name}' has label '{resolved.label}', "
            "expected static_presence or motion"
        )
    if resolved.counterpart_entry is None or resolved.counterpart_path is None:
        raise FileNotFoundError(
            f"Dataset '{resolved.path.name}' is missing explicit pair metadata in dataset_info.json"
        )

    if resolved.label == "static_presence":
        static_dataset = resolved
        motion_dataset = ResolvedDataset(
            label="motion",
            entry=resolved.counterpart_entry,
            path=resolved.counterpart_path,
            counterpart_label="static_presence",
            counterpart_entry=resolved.entry,
            counterpart_path=resolved.path,
        )
    else:
        static_dataset = ResolvedDataset(
            label="static_presence",
            entry=resolved.counterpart_entry,
            path=resolved.counterpart_path,
            counterpart_label="motion",
            counterpart_entry=resolved.entry,
            counterpart_path=resolved.path,
        )
        motion_dataset = resolved

    return ResolvedPair(
        static_presence=static_dataset,
        motion=motion_dataset,
        chip=str(static_dataset.entry.get("chip", motion_dataset.entry.get("chip", "UNKNOWN"))).upper(),
        num_subcarriers=int(static_dataset.entry.get("subcarriers", num_sc) or num_sc),
    )


def select_dataset_interactively(
    *,
    label: Optional[str] = None,
    chip: Optional[str] = None,
    environment: Optional[str] = None,
    num_sc: Optional[int] = 64,
    require_pair: bool = False,
    prompt: str = "Select dataset",
    dataset_info: Optional[Dict[str, Any]] = None,
) -> ResolvedDataset:
    """Interactive chooser backed by dataset_info metadata."""
    info = load_dataset_info() if dataset_info is None else dataset_info
    files_section = info.get("files", {})
    options: List[Tuple[str, Dict[str, Any]]] = []
    for current_label, entries in files_section.items():
        for raw_entry in entries:
            entry = dict(raw_entry)
            entry["_label"] = current_label
            if not _entry_matches_filters(
                entry,
                label=label,
                chip=chip,
                environment=environment,
                num_sc=num_sc,
            ):
                continue
            if require_pair:
                if current_label not in ("static_presence", "motion"):
                    continue
                pair_field = (
                    "optimal_pair_motion_file"
                    if current_label == "static_presence"
                    else "optimal_pair_static_presence_file"
                )
                if not entry.get(pair_field):
                    continue
            options.append((current_label, entry))

    if not options:
        raise FileNotFoundError("No datasets available for the requested interactive selection")

    options.sort(
        key=lambda item: (
            str(item[1].get("environment", "unknown")),
            str(item[1].get("chip", "unknown")).upper(),
            str(item[1].get("collected_at", "")),
            str(item[1].get("filename", "")),
        ),
        reverse=True,
    )
    print()
    print(prompt)
    last_environment = None
    last_chip = None
    for idx, (current_label, entry) in enumerate(options, start=1):
        filename = str(entry.get("filename", "<missing>"))
        chip_label = str(entry.get("chip", "unknown")).upper()
        env_label = str(entry.get("environment", "unknown"))
        collected_at = str(entry.get("collected_at", "unknown"))
        if env_label != last_environment:
            print(f"\nEnvironment: {env_label}")
            last_environment = env_label
            last_chip = None
        if chip_label != last_chip:
            print(f"  Chip: {chip_label}")
            last_chip = chip_label
        print(f"    {idx:>2}. [{current_label}] {filename} | at={collected_at}")

    while True:
        try:
            selection = input("Enter dataset number: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nSelection cancelled.")
            raise SystemExit(130)
        try:
            selected_index = int(selection)
        except ValueError:
            print("Invalid selection. Enter a number from the list.")
            continue
        if 1 <= selected_index <= len(options):
            _, selected_entry = options[selected_index - 1]
            return resolve_dataset_selection(
                str(selected_entry.get("filename")),
                dataset_info=info,
                prefer_latest=True,
            )
        print("Selection out of range.")
