# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Dataset-gate discovery with strict broken-catalog handling."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pytest

from tools.lib.dataset_metadata import ADMITTED_DATASET_ROLES
from tools.lib.repo_paths import data_dir

from support.chip_matrix import DETECTION_CHIPS, chip_label


DATA_DIR = data_dir()
DATASET_INFO_PATH = DATA_DIR / "dataset_info.json"
RESERVED_ROLES = frozenset({"selection", "holdout"})


@dataclass(frozen=True)
class DatasetGateCase:
    """One eligible dataset entry or pair for a chip-level replay gate."""

    chip: str
    gate: str
    label: str
    path: Path
    counterpart_path: Path | None = None
    entry: dict[str, Any] | None = None
    counterpart_entry: dict[str, Any] | None = None


def load_test_dataset_info(path: Path = DATASET_INFO_PATH) -> dict[str, Any]:
    """Load the test catalog, treating absence and malformed JSON as failures."""
    if not path.is_file():
        raise AssertionError(f"Dataset catalog is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AssertionError(f"Dataset catalog is not readable JSON: {path}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("files"), dict):
        raise AssertionError("Dataset catalog must contain a files object")
    return payload


def _entries(info: dict[str, Any], label: str) -> list[dict[str, Any]]:
    entries = info["files"].get(label, [])
    if not isinstance(entries, list) or any(not isinstance(entry, dict) for entry in entries):
        raise AssertionError(f"Dataset catalog group {label!r} must be a list of objects")
    return entries


def _role(entry: dict[str, Any]) -> str:
    return str(entry.get("dataset_role", "exclude")).strip().lower()


def _path(label: str, entry: dict[str, Any], root: Path = DATA_DIR) -> Path:
    filename = entry.get("filename")
    if not isinstance(filename, str) or not filename.strip():
        raise AssertionError(f"Admitted {label} entry has no filename")
    return root / label / filename


def _rate(entry: dict[str, Any]) -> float:
    value = entry.get("average_packet_rate")
    try:
        rate = float(value)
    except (TypeError, ValueError):
        rate = 0.0
    if rate > 0.0:
        return rate
    try:
        duration_ms = float(entry.get("duration_ms", 0.0) or 0.0)
        packets = int(entry.get("num_packets", 0) or 0)
    except (TypeError, ValueError):
        return 0.0
    return packets * 1000.0 / duration_ms if duration_ms > 0.0 and packets > 0 else 0.0


def _validate_npz(path: Path) -> None:
    """Validate the archive directory and required CSI member without materializing it."""
    try:
        with np.load(path, allow_pickle=False) as archive:
            if "csi_data" not in archive.files:
                raise AssertionError(f"Dataset NPZ has no csi_data member: {path}")
    except AssertionError:
        raise
    except (OSError, ValueError, EOFError) as exc:
        raise AssertionError(f"Dataset NPZ is corrupt or unreadable: {path}") from exc


def validate_testable_catalog(path: Path = DATASET_INFO_PATH) -> None:
    """Fail when admitted test data is missing or a declared pair is broken."""
    info = load_test_dataset_info(path)
    root = path.parent
    motion_entries = {
        entry.get("filename"): entry
        for entry in _entries(info, "motion")
        if isinstance(entry.get("filename"), str)
    }
    for label in ("static_presence", "motion", "empty"):
        for entry in _entries(info, label):
            if _role(entry) not in ADMITTED_DATASET_ROLES:
                continue
            entry_path = _path(label, entry, root)
            if not entry_path.is_file():
                raise AssertionError(f"Catalogued dataset file is missing: {entry_path}")
            _validate_npz(entry_path)
            if not entry.get("chip"):
                raise AssertionError(f"Admitted dataset has no chip: {entry_path}")
            if int(entry.get("subcarriers", 0) or 0) <= 0:
                raise AssertionError(f"Admitted dataset has invalid subcarriers: {entry_path}")

            if label != "static_presence" or not entry.get("optimal_pair_motion_file"):
                continue
            counterpart = motion_entries.get(entry["optimal_pair_motion_file"])
            if counterpart is None:
                raise AssertionError(
                    f"Catalogued pair is missing motion metadata: {entry_path}"
                )
            counterpart_path = _path("motion", counterpart, root)
            if not counterpart_path.is_file():
                raise AssertionError(f"Catalogued pair file is missing: {counterpart_path}")
            if _role(counterpart) != _role(entry):
                raise AssertionError(f"Catalogued pair roles differ: {entry_path}")
            if chip_label(counterpart.get("chip", "")) != chip_label(entry["chip"]):
                raise AssertionError(f"Catalogued pair chips differ: {entry_path}")
            if counterpart.get("subcarriers") != entry.get("subcarriers"):
                raise AssertionError(f"Catalogued pair subcarrier counts differ: {entry_path}")


def dataset_gate_cases(gate: str) -> tuple[DatasetGateCase, ...]:
    """Return all eligible cases for a named performance gate."""
    info = load_test_dataset_info()
    motion_entries = {
        str(entry.get("filename")): entry
        for entry in _entries(info, "motion")
        if entry.get("filename")
    }
    cases: list[DatasetGateCase] = []

    if gate in {"normal", "reserved", "weak", "packet_rate"}:
        for entry in _entries(info, "static_presence"):
            role = _role(entry)
            motion = motion_entries.get(str(entry.get("optimal_pair_motion_file", "")))
            if role not in ADMITTED_DATASET_ROLES or motion is None or _role(motion) != role:
                continue
            if bool(entry.get("synthetic")) or bool(motion.get("synthetic")):
                continue
            if entry.get("subcarriers") != 64 or motion.get("subcarriers") != 64:
                continue
            weak = bool(entry.get("low_rssi")) or bool(motion.get("low_rssi"))
            if gate == "normal" and weak:
                continue
            if gate == "reserved" and (weak or role not in RESERVED_ROLES):
                continue
            if gate == "weak" and not weak:
                continue
            if gate == "packet_rate" and (
                _rate(entry) < 500.0 or _rate(motion) < 500.0
            ):
                continue
            chip = chip_label(str(entry.get("chip", "")))
            cases.append(
                DatasetGateCase(
                    chip=chip,
                    gate=gate,
                    label=Path(str(entry["filename"])).stem,
                    path=_path("static_presence", entry),
                    counterpart_path=_path("motion", motion),
                    entry=entry,
                    counterpart_entry=motion,
                )
            )
    elif gate in {"empty", "long"}:
        explicit_long = any(
            bool(entry.get("long_recording"))
            and _role(entry) in ADMITTED_DATASET_ROLES
            for entry in _entries(info, "empty")
        )
        groups: Iterable[tuple[str, dict[str, Any]]]
        if gate == "long" and not explicit_long:
            groups = (("test", entry) for entry in _entries(info, "test"))
        else:
            groups = (("empty", entry) for entry in _entries(info, "empty"))
        for label, entry in groups:
            if _role(entry) not in ADMITTED_DATASET_ROLES:
                continue
            is_long = bool(entry.get("long_recording")) or label == "test"
            if is_long != (gate == "long") or entry.get("subcarriers") != 64:
                continue
            chip = chip_label(str(entry.get("chip", "")))
            cases.append(
                DatasetGateCase(
                    chip=chip,
                    gate=gate,
                    label=Path(str(entry["filename"])).stem,
                    path=_path(label, entry),
                    entry=entry,
                )
            )
    else:
        raise ValueError(f"Unknown dataset gate: {gate}")

    return tuple(sorted(cases, key=lambda case: (case.chip, case.label)))


def per_chip_params(gate: str, *, all_matches: bool = True) -> list[object]:
    """Create one visible pytest parameter per chip, skipping only no-data cases."""
    cases = dataset_gate_cases(gate)
    params: list[object] = []
    for chip in DETECTION_CHIPS:
        matches = [case for case in cases if case.chip == chip]
        if not matches:
            params.append(
                pytest.param(
                    None,
                    marks=pytest.mark.skip(
                        reason=f"No eligible {gate} dataset for chip {chip}"
                    ),
                    id=f"{chip.lower()}_no_eligible_{gate}_dataset",
                )
            )
            continue
        selected = matches if all_matches else matches[:1]
        params.extend(
            pytest.param(case, id=f"{chip.lower()}_{case.label}") for case in selected
        )
    return params
