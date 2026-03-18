#!/usr/bin/env python3
"""
Micro-ESPectre - Wifi-Sensing-HAR CSV to NPZ Converter

Convert approved wifi-sensing-har CSV files into ESPectre NPZ sample format.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.request import urlopen, Request

import numpy as np


MICRO_DIR = Path(__file__).resolve().parent.parent
REGISTRY_PATH = MICRO_DIR / "data" / "external_dataset_registry.json"
DEFAULT_INPUT_DIR = MICRO_DIR / ".tmp" / "wifi_sensing_har_csv"
DEFAULT_OUTPUT_DIR = MICRO_DIR / "data"
GITHUB_CSI_DATA_API = "https://api.github.com/repos/jasminkarki/wifi-sensing-har/contents/csi_data?ref=main"
DATASET_INFO_PATH = MICRO_DIR / "data" / "dataset_info.json"

CSI_LINE_RE = re.compile(r",(?P<len>\d+),\[(?P<body>[^\]]+)\]\s*$")


def _load_registry() -> dict[str, Any]:
    with REGISTRY_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _require_dataset_approved(registry: dict[str, Any], dataset_id: str = "wifi_sensing_har") -> None:
    item = next((d for d in registry.get("datasets", []) if d.get("id") == dataset_id), None)
    if item is None:
        raise ValueError(f"Dataset '{dataset_id}' not found in registry.")
    if item.get("status") != "approved":
        raise ValueError(f"Dataset '{dataset_id}' is not approved (status={item.get('status')}).")
    if item.get("license", {}).get("license_conflict_flag", False):
        raise ValueError(f"Dataset '{dataset_id}' has legal conflict flag enabled.")


def _download_all_csv_from_github(target_dir: Path) -> int:
    target_dir.mkdir(parents=True, exist_ok=True)
    req = Request(GITHUB_CSI_DATA_API, headers={"User-Agent": "espectre-dataset-converter"})
    with urlopen(req, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))

    csv_entries = [item for item in payload if item.get("name", "").endswith(".csv") and item.get("download_url")]
    downloaded = 0
    for entry in csv_entries:
        out_path = target_dir / entry["name"]
        if out_path.exists() and out_path.stat().st_size > 0:
            continue
        data_req = Request(entry["download_url"], headers={"User-Agent": "espectre-dataset-converter"})
        with urlopen(data_req, timeout=120) as data_response:
            content = data_response.read()
        out_path.write_bytes(content)
        downloaded += 1
        print(f"Downloaded {entry['name']} ({len(content)} bytes)")
    return len(csv_entries)


def _label_description(label: str) -> str:
    if label == "walking":
        return "Walking activity from wifi-sensing-har public dataset"
    if label == "run":
        return "Running/jogging activity from wifi-sensing-har public dataset"
    if label == "no_presence":
        return "No-activity baseline from wifi-sensing-har public dataset"
    return "Imported from wifi-sensing-har public dataset"


def _update_dataset_info(records: list[dict[str, Any]]) -> None:
    if not DATASET_INFO_PATH.exists():
        return
    with DATASET_INFO_PATH.open("r", encoding="utf-8") as handle:
        info = json.load(handle)

    info.setdefault("labels", {})
    info.setdefault("files", {})

    for rec in records:
        label = rec["label"]
        info["labels"].setdefault(label, {"description": _label_description(label)})
        info["files"].setdefault(label, [])
        info["files"][label].append(
            {
                "filename": rec["filename"],
                "chip": "ESP32",
                "subcarriers": rec["num_subcarriers"],
                "contributor": "external:wifi_sensing_har",
                "collected_at": rec["collected_at"],
                "duration_ms": rec["duration_ms"],
                "num_packets": rec["packets"],
                "gain_locked": False,
                "description": f"Imported from wifi-sensing-har ({rec['source_csv']})",
            }
        )

    info["updated_at"] = datetime.now().isoformat()
    with DATASET_INFO_PATH.open("w", encoding="utf-8") as handle:
        json.dump(info, handle, indent=2)
        handle.write("\n")


def _infer_label_from_filename(csv_path: Path) -> str:
    stem = csv_path.stem.lower()
    if "noactivity" in stem:
        return "no_presence"
    if "walk" in stem:
        return "walking"
    if "jog" in stem or "run" in stem:
        return "run"
    return "unknown"


def _extract_iq_vector(line: str) -> np.ndarray | None:
    if not line.startswith("CSI_DATA"):
        return None
    match = CSI_LINE_RE.search(line.strip())
    if not match:
        return None
    declared_len = int(match.group("len"))
    body = match.group("body")
    values = np.fromstring(body, sep=" ", dtype=np.int16)
    if values.size == 0:
        return None
    if values.size != declared_len:
        # Keep behavior robust: truncate/pad to declared length.
        if values.size > declared_len:
            values = values[:declared_len]
        else:
            values = np.pad(values, (0, declared_len - values.size), mode="constant")
    values = np.clip(values, -128, 127).astype(np.int8, copy=False)
    return values


def _convert_csv(
    csv_path: Path,
    output_dir: Path,
    sampling_hz: int,
    keep_all_lengths: bool = False,
    max_lines_per_file: int = 0,
) -> list[tuple[Path, int, int, int]]:
    label = _infer_label_from_filename(csv_path)
    label_dir = output_dir / label
    label_dir.mkdir(parents=True, exist_ok=True)

    vectors: list[np.ndarray] = []
    lengths: list[int] = []
    with csv_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for index, line in enumerate(handle):
            if max_lines_per_file > 0 and index >= max_lines_per_file:
                break
            iq = _extract_iq_vector(line)
            if iq is not None:
                vectors.append(iq)
                lengths.append(int(iq.shape[0]))

    if not vectors:
        return []

    # Some files may contain mixed CSI lengths (e.g., LLTF-only and full CSI).
    # Default mode keeps one preferred length; keep_all_lengths emits one NPZ per length.
    uniq_lengths, uniq_counts = np.unique(np.array(lengths, dtype=np.int32), return_counts=True)
    preferred_lengths = [128, 104]  # 64 and 52 subcarriers in I/Q interleaved format
    if keep_all_lengths:
        selected_lengths = [int(v) for v in sorted(uniq_lengths.tolist())]
    else:
        selected_len: int | None = None
        for candidate in preferred_lengths:
            if candidate in uniq_lengths:
                selected_len = candidate
                break
        if selected_len is None:
            selected_len = int(uniq_lengths[int(np.argmax(uniq_counts))])
        selected_lengths = [selected_len]

    outputs: list[tuple[Path, int, int, int]] = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for selected_len in selected_lengths:
        filtered = [v for v in vectors if int(v.shape[0]) == int(selected_len)]
        if not filtered:
            continue
        csi_data = np.stack(filtered)
        num_subcarriers = int(csi_data.shape[1] // 2)
        duration_ms = int((len(filtered) / float(sampling_hz)) * 1000.0)

        out_name = f"{label}_esp32_{num_subcarriers}sc_{timestamp}_{csv_path.stem}.npz"
        out_path = label_dir / out_name
        np.savez_compressed(
            out_path,
            csi_data=csi_data,
            num_subcarriers=num_subcarriers,
            label=label,
            chip="esp32",
            gain_locked=False,
            collected_at=datetime.now().isoformat(),
            duration_ms=duration_ms,
            format_version="1.0",
            source_dataset="wifi_sensing_har",
            source_file=csv_path.name,
            source_iq_len=int(selected_len),
        )
        outputs.append((out_path, len(filtered), num_subcarriers, int(selected_len)))

    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert wifi-sensing-har CSV dataset to ESPectre NPZ format.")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR), help="Input directory containing CSV files.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory for generated NPZ.")
    parser.add_argument("--sampling-hz", type=int, default=100, help="Nominal sampling rate used for duration estimation.")
    parser.add_argument(
        "--keep-all-lengths",
        action="store_true",
        help="Keep all detected CSI vector lengths and emit one NPZ per length per CSV (lossless by length).",
    )
    parser.add_argument("--max-files", type=int, default=0, help="Limit number of CSV files to convert (0 = all).")
    parser.add_argument("--max-lines-per-file", type=int, default=0, help="Limit lines per CSV file (0 = all lines).")
    parser.add_argument(
        "--download-all",
        action="store_true",
        help="Download all available CSV files from jasminkarki/wifi-sensing-har before conversion.",
    )
    parser.add_argument(
        "--skip-registry-check",
        action="store_true",
        help="Skip legal gate validation from external_dataset_registry.json.",
    )
    parser.add_argument("--strict", action="store_true", help="Fail if no files were converted.")
    args = parser.parse_args()

    if args.skip_registry_check:
        print("WARNING: skipping registry legal gate validation (--skip-registry-check).")
    else:
        if not REGISTRY_PATH.exists():
            print(
                "WARNING: external dataset registry not found; "
                "continuing without registry legal gate validation."
            )
        else:
            registry = _load_registry()
            _require_dataset_approved(registry, "wifi_sensing_har")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.download_all:
        total_remote = _download_all_csv_from_github(input_dir)
        print(f"Remote CSV files available: {total_remote}")

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    csv_files = sorted(input_dir.glob("*.csv"))
    if args.max_files > 0:
        csv_files = csv_files[: args.max_files]

    converted = 0
    packets_total = 0
    manifest: dict[str, Any] = {
        "dataset_id": "wifi_sensing_har",
        "created_at": datetime.now().isoformat(),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "files": [],
    }

    converted_records: list[dict[str, Any]] = []
    for csv_path in csv_files:
        outputs = _convert_csv(
            csv_path=csv_path,
            output_dir=output_dir,
            sampling_hz=args.sampling_hz,
            keep_all_lengths=args.keep_all_lengths,
            max_lines_per_file=args.max_lines_per_file,
        )
        if not outputs:
            manifest["files"].append(
                {
                    "source_csv": csv_path.name,
                    "status": "skipped",
                    "reason": "no CSI_DATA rows parsed",
                }
            )
            continue

        for out_path, packets, num_subcarriers, source_iq_len in outputs:
            converted += 1
            packets_total += packets
            out_rel = str(out_path.relative_to(output_dir))
            out_filename = out_path.name
            collected_at = datetime.now().isoformat()
            duration_ms = int((packets / float(args.sampling_hz)) * 1000.0)
            manifest["files"].append(
                {
                    "source_csv": csv_path.name,
                    "output_npz": out_rel,
                    "packets": packets,
                    "label": _infer_label_from_filename(csv_path),
                    "num_subcarriers": num_subcarriers,
                    "source_iq_len": source_iq_len,
                    "status": "converted",
                }
            )
            converted_records.append(
                {
                    "filename": out_filename,
                    "source_csv": csv_path.name,
                    "label": _infer_label_from_filename(csv_path),
                    "packets": packets,
                    "num_subcarriers": num_subcarriers,
                    "duration_ms": duration_ms,
                    "collected_at": collected_at,
                }
            )
            print(
                f"Converted {csv_path.name} -> {out_path.name} "
                f"({packets} packets, iq_len={source_iq_len}, sc={num_subcarriers})"
            )

    manifest["summary"] = {
        "csv_files_seen": len(csv_files),
        "npz_files_created": converted,
        "packets_total": packets_total,
    }
    manifest_path = output_dir / "wifi_sensing_har_conversion_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    _update_dataset_info(converted_records)

    print(f"Wrote conversion manifest: {manifest_path}")
    print(f"Created NPZ files: {converted}")

    if args.strict and converted == 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
