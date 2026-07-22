#!/usr/bin/env python3
"""
ESPectre - Synthetic low-RSSI dataset generator

Derive a reproducible weak-link CSI capture from one registered real dataset.
Reference matching jointly reproduces the observed phase-specific Core-6
behavior; shared-session generation fits the quiet phase and reuses the same
impairment parameters for the paired motion derivative.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import datetime
import hashlib
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, Optional

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.lib.bootstrap import setup_paths  # noqa: E402, F401
from tools.lib import dataset_metadata  # noqa: E402
from tools.lib.performance_report import (  # noqa: E402
    compute_classic_packet_result,
)
from tools.lib.csi_io import load_npz_as_packets  # noqa: E402

import config  # noqa: E402
from ml_detector import FEATURE_NAMES, MLDetector  # noqa: E402
from runtime_policy import make_evaluation_cadence  # noqa: E402


GENERATOR_VERSION = 4
SUPPORTED_SOURCE_LABELS = ("empty", "static_presence", "motion")
QUIET_LABELS = ("empty", "static_presence")
CALIBRATION_MAX_PACKETS = 6000
CALIBRATION_ROUNDS = 6
MAX_JITTER_SIGMA = 3.0
SHARED_SESSION_MOTION_SPATIAL_BOOST = 1.1


@dataclass(frozen=True)
class ImpairmentParameters:
    """Fitted weak-link parameters applied consistently to one capture."""

    source_retention: float
    jitter_sigma: float
    temporal_rho: float
    noise_sigma_abs: float
    spatial_spread: float
    turbulence_retention: float
    turbulence_noise: float
    turbulence_rho: float
    turbulence_skew: float


IMPAIRMENT_PARAMETER_NAMES = (
    "source_retention",
    "jitter_sigma",
    "temporal_rho",
    "noise_sigma_abs",
    "spatial_spread",
    "turbulence_retention",
    "turbulence_noise",
    "turbulence_rho",
    "turbulence_skew",
)


@dataclass(frozen=True)
class LowRssiProfile:
    """One weak-link behavior profile derived from real captures."""

    name: str
    reference_chip: str
    target_quiet_rssi_dbm: float
    target_motion_rssi_dbm: float
    target_quiet_l1_delta: float
    target_motion_l1_delta: float
    attenuation_db: float
    noise_sigma_abs: float
    temporal_rho: float
    packet_loss: float
    burst_loss_probability: float
    burst_loss_length: int
    deformation_mode: str
    reference_feature_medians: Dict[str, Dict[str, float]]
    reference_datasets: tuple[str, str]


LOW_RSSI_PROFILES = {
    "c3_weak_link": LowRssiProfile(
        name="c3_weak_link",
        reference_chip="C3",
        target_quiet_rssi_dbm=-77.0,
        target_motion_rssi_dbm=-77.0,
        target_quiet_l1_delta=0.1068241023,
        target_motion_l1_delta=0.1162957716,
        attenuation_db=1.3,
        noise_sigma_abs=0.55,
        temporal_rho=0.0,
        packet_loss=0.0010,
        burst_loss_probability=0.0,
        burst_loss_length=1,
        deformation_mode="gain_jitter",
        reference_feature_medians={
            "static_presence": {
                "turb_mad_over_mean": 0.0141460914,
                "turb_skewness": 0.6029982003,
                "turb_autocorr": 0.0038923421,
                "l1_delta": 0.1068241023,
                "l1_delta_std": 0.0263707299,
                "l1_delta_waveform_length": 2.6515621811,
            },
            "motion": {
                "turb_mad_over_mean": 0.0265381753,
                "turb_skewness": 0.9141968319,
                "turb_autocorr": 0.5726245162,
                "l1_delta": 0.1162957716,
                "l1_delta_std": 0.0320751730,
                "l1_delta_waveform_length": 2.9129450872,
            },
        },
        reference_datasets=(
            "static_presence_c3_64sc_dev0000acebe64ae708_20260722_210321_712831_0001.npz",
            "motion_c3_64sc_dev0000acebe64ae708_20260722_210523_413343_0001.npz",
        ),
    ),
    "c5_moderate_link": LowRssiProfile(
        name="c5_moderate_link",
        reference_chip="C5",
        target_quiet_rssi_dbm=-75.0,
        target_motion_rssi_dbm=-71.0,
        target_quiet_l1_delta=0.0324460606,
        target_motion_l1_delta=0.0359959860,
        attenuation_db=0.0,
        noise_sigma_abs=0.0,
        temporal_rho=0.2,
        packet_loss=0.0,
        burst_loss_probability=0.0,
        burst_loss_length=1,
        deformation_mode="rank_preserving",
        reference_feature_medians={
            "static_presence": {
                "turb_mad_over_mean": 0.0317348919,
                "turb_skewness": 0.0108864711,
                "turb_autocorr": 0.0146959884,
                "l1_delta": 0.0324460606,
                "l1_delta_std": 0.0072907729,
                "l1_delta_waveform_length": 0.7344873603,
            },
            "motion": {
                "turb_mad_over_mean": 0.1035929297,
                "turb_skewness": 0.2776024626,
                "turb_autocorr": 0.6678587802,
                "l1_delta": 0.0359959860,
                "l1_delta_std": 0.0128352709,
                "l1_delta_waveform_length": 0.8823076765,
            },
        },
        reference_datasets=(
            "static_presence_c5_64sc_dev000030eda0e46278_20260722_205156_405317_0001.npz",
            "motion_c5_64sc_dev000030eda0e46278_20260722_205350_355335_0001.npz",
        ),
    ),
    "s3_weak_link": LowRssiProfile(
        name="s3_weak_link",
        reference_chip="S3",
        target_quiet_rssi_dbm=-77.0,
        target_motion_rssi_dbm=-75.0,
        target_quiet_l1_delta=0.0829442076,
        target_motion_l1_delta=0.0797444073,
        attenuation_db=0.0,
        noise_sigma_abs=0.0,
        temporal_rho=0.1,
        packet_loss=0.0,
        burst_loss_probability=0.0,
        burst_loss_length=1,
        deformation_mode="gain_jitter",
        reference_feature_medians={
            "static_presence": {
                "turb_mad_over_mean": 0.0079209775,
                "turb_skewness": 0.7076436922,
                "turb_autocorr": 0.0704195681,
                "l1_delta": 0.0829442076,
                "l1_delta_std": 0.0207895529,
                "l1_delta_waveform_length": 2.0497956170,
            },
            "motion": {
                "turb_mad_over_mean": 0.0136436264,
                "turb_skewness": 1.1579034260,
                "turb_autocorr": 0.6778138926,
                "l1_delta": 0.0797444073,
                "l1_delta_std": 0.0257041642,
                "l1_delta_waveform_length": 2.0729831844,
            },
        },
        reference_datasets=(
            "static_presence_s3_64sc_dev000010b41de8ec00_20260722_172043_630431_0001.npz",
            "motion_s3_64sc_dev000010b41de8ec00_20260722_172305_879358_0001.npz",
        ),
    ),
    "c6_moderate_link": LowRssiProfile(
        name="c6_moderate_link",
        reference_chip="C6",
        target_quiet_rssi_dbm=-69.0,
        target_motion_rssi_dbm=-66.0,
        target_quiet_l1_delta=0.0336792292,
        target_motion_l1_delta=0.0320762532,
        attenuation_db=0.0,
        noise_sigma_abs=0.0,
        temporal_rho=0.1,
        packet_loss=0.0,
        burst_loss_probability=0.0,
        burst_loss_length=1,
        deformation_mode="gain_jitter",
        reference_feature_medians={
            "static_presence": {
                "turb_mad_over_mean": 0.0872638584,
                "turb_skewness": 0.0774848815,
                "turb_autocorr": 0.0249507884,
                "l1_delta": 0.0336792292,
                "l1_delta_std": 0.0075417523,
                "l1_delta_waveform_length": 0.7617675375,
            },
            "motion": {
                "turb_mad_over_mean": 0.0883550303,
                "turb_skewness": 0.2033364546,
                "turb_autocorr": 0.6303078168,
                "l1_delta": 0.0320762532,
                "l1_delta_std": 0.0100869223,
                "l1_delta_waveform_length": 0.7641816740,
            },
        },
        reference_datasets=(
            "static_presence_c6_64sc_dev00007c2c6742bbac_20260722_191653_148862_0001.npz",
            "motion_c6_64sc_dev00007c2c6742bbac_20260722_191914_560463_0001.npz",
        ),
    ),
}


def _scalar(value: np.ndarray | Any) -> Any:
    """Return a Python scalar from one NPZ scalar value."""
    array = np.asarray(value)
    return array.item() if array.ndim == 0 else value


def _stable_seed(seed: int, value: str) -> int:
    digest = hashlib.sha256(str(value).encode("utf-8")).digest()
    fragment = int.from_bytes(digest[:4], byteorder="little")
    return (int(seed) ^ fragment) & 0x7FFFFFFF


def _find_catalog_entry(
    info: Dict[str, Any], source_path: Path
) -> tuple[str, Dict[str, Any]]:
    source_resolved = source_path.resolve()
    for label, entries in info.get("files", {}).items():
        for entry in entries:
            if (
                dataset_metadata.resolve_entry_path(label, entry).resolve()
                == source_resolved
            ):
                return str(label), entry
    raise FileNotFoundError(
        f"Source dataset is not registered in data/dataset_info.json: {source_path}"
    )


def resolve_source_dataset(
    value: str | Path,
    *,
    info: Optional[Dict[str, Any]] = None,
) -> tuple[Path, str, Dict[str, Any], Dict[str, Any]]:
    """Resolve one real source path and its catalog entry."""
    dataset_info = dataset_metadata.load_dataset_info() if info is None else info
    candidate = Path(value)
    if candidate.exists():
        source_path = candidate.resolve()
        label, entry = _find_catalog_entry(dataset_info, source_path)
    else:
        resolved = dataset_metadata.resolve_dataset_selection(
            str(value), num_sc=None, dataset_info=dataset_info
        )
        source_path = resolved.path.resolve()
        label = resolved.label
        entry = resolved.entry

    if label not in SUPPORTED_SOURCE_LABELS:
        supported = ", ".join(SUPPORTED_SOURCE_LABELS)
        raise ValueError(f"Source label must be one of {supported}; got {label!r}")
    if bool(entry.get("synthetic")):
        raise ValueError("Synthetic datasets cannot be used as generator sources")
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    return source_path, label, entry, dataset_info


def _source_pair_names(label: str, entry: Dict[str, Any]) -> tuple[str, ...]:
    filename = str(entry.get("filename", ""))
    counterpart_field = {
        "static_presence": "optimal_pair_motion_file",
        "motion": "optimal_pair_static_presence_file",
    }.get(label)
    counterpart = str(entry.get(counterpart_field, "")) if counterpart_field else ""
    return tuple(sorted(name for name in (filename, counterpart) if name))


def build_generation_group(
    label: str,
    entry: Dict[str, Any],
    profile_name: str,
    seed: int,
    generation_mode: str = "reference_match",
) -> str:
    """Return a pair-stable identifier for one synthetic weak-link session."""
    source_names = _source_pair_names(label, entry)
    payload = "\0".join(source_names + (profile_name, generation_mode, str(int(seed))))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"low-rssi-{profile_name}-{generation_mode}-s{int(seed)}-{digest}"


def build_output_path(
    source_path: Path,
    label: str,
    profile_name: str,
    generation_mode: str,
    seed: int,
) -> Path:
    """Return the deterministic default path for one generated dataset."""
    output_name = (
        f"{source_path.stem}__synthetic_low_rssi_"
        f"{profile_name}_{generation_mode}_seed{int(seed)}.npz"
    )
    return dataset_metadata.DATA_DIR / label / output_name


def _drop_mask(
    packet_count: int, profile: LowRssiProfile, rng: np.random.Generator
) -> np.ndarray:
    drop = np.zeros(packet_count, dtype=bool)
    if profile.packet_loss > 0.0:
        drop |= rng.random(packet_count) < profile.packet_loss
    if profile.burst_loss_probability <= 0.0 or profile.burst_loss_length <= 0:
        return drop
    index = 0
    while index < packet_count:
        if rng.random() < profile.burst_loss_probability:
            length = max(1, int(profile.burst_loss_length))
            drop[index : index + length] = True
            index += length
        index += 1
    return drop


def _apply_packet_mask(
    payload: Dict[str, np.ndarray], keep: np.ndarray
) -> Dict[str, np.ndarray]:
    """Filter every packet-aligned NPZ array with the same mask."""
    packet_count = len(keep)
    filtered: Dict[str, np.ndarray] = {}
    for key, raw_value in payload.items():
        value = np.asarray(raw_value)
        if value.ndim > 0 and value.shape[0] == packet_count:
            filtered[key] = value[keep]
        else:
            filtered[key] = value
    return filtered


def _temporal_profile_field(
    innovations: np.ndarray,
    rho: float,
) -> np.ndarray:
    """Build a stationary AR(1) per-subcarrier profile perturbation."""
    if rho <= 0.0:
        return innovations.astype(np.float32, copy=False)
    field = np.empty_like(innovations, dtype=np.float32)
    field[0] = innovations[0]
    innovation_gain = float(np.sqrt(max(0.0, 1.0 - rho * rho)))
    for index in range(1, len(field)):
        field[index] = rho * field[index - 1] + innovation_gain * innovations[index]
    return field


def _temporal_turbulence_field(
    innovations: np.ndarray, rho: float, skew: float
) -> np.ndarray:
    """Build a scalar AR field with controllable positive-tail asymmetry."""
    values = np.asarray(innovations, dtype=np.float32)
    if skew > 0.0:
        values = values + np.float32(skew) * (values * values - 1.0)
        values -= np.mean(values)
        std = float(np.std(values))
        if std > 1e-6:
            values /= np.float32(std)
    return _temporal_profile_field(values[:, None], rho)[:, 0]


def _normalized_baseline_profile(csi_data: np.ndarray) -> np.ndarray:
    """Return the robust per-subcarrier amplitude profile for one capture."""
    csi = np.asarray(csi_data)
    pairs = csi.astype(np.float32, copy=False).reshape(len(csi), -1, 2)
    amplitudes = np.sqrt(np.sum(pairs * pairs, axis=2))
    packet_means = np.mean(amplitudes, axis=1, keepdims=True)
    normalized = np.divide(
        amplitudes,
        packet_means,
        out=np.ones_like(amplitudes),
        where=packet_means > 1e-6,
    )
    baseline = np.median(normalized, axis=0).astype(np.float32)
    baseline_mean = float(np.mean(baseline))
    if baseline_mean > 1e-6:
        baseline /= np.float32(baseline_mean)
    return baseline


def degrade_csi(
    csi_data: np.ndarray,
    *,
    profile: LowRssiProfile,
    parameters: ImpairmentParameters,
    baseline_profile: np.ndarray,
    profile_field: np.ndarray,
    turbulence_field: np.ndarray,
    noise_field: np.ndarray,
) -> np.ndarray:
    """Apply one deterministic weak-link impairment to signed I/Q payloads."""
    csi = np.asarray(csi_data)
    if csi.ndim != 2 or csi.shape[1] % 2 != 0:
        raise ValueError("csi_data must have shape [packets, subcarriers * 2]")
    subcarriers = csi.shape[1] // 2
    if profile_field.shape != (len(csi), subcarriers):
        raise ValueError("profile_field shape does not match csi_data")
    if turbulence_field.shape != (len(csi),):
        raise ValueError("turbulence_field shape does not match csi_data")
    if noise_field.shape != (len(csi), subcarriers, 2):
        raise ValueError("noise_field shape does not match csi_data")
    if baseline_profile.shape != (subcarriers,):
        raise ValueError("baseline_profile shape does not match csi_data")
    sensing_indices = np.asarray(
        [index for index in config.DEFAULT_SUBCARRIERS if 0 <= index < subcarriers],
        dtype=np.intp,
    )
    if len(sensing_indices) < 2:
        raise ValueError("csi_data does not contain the configured sensing band")

    pairs = csi.astype(np.float32, copy=True).reshape(len(csi), subcarriers, 2)
    if parameters.source_retention < 1.0:
        amplitudes = np.sqrt(np.sum(pairs * pairs, axis=2))
        packet_means = np.mean(amplitudes, axis=1, keepdims=True)
        normalized = np.divide(
            amplitudes,
            packet_means,
            out=np.ones_like(amplitudes),
            where=packet_means > 1e-6,
        )
        mixed = baseline_profile[None, :] + np.float32(
            parameters.source_retention
        ) * (normalized - baseline_profile[None, :])
        target_amplitudes = np.maximum(mixed, 0.0) * packet_means
        amplitude_scale = np.divide(
            target_amplitudes,
            amplitudes,
            out=np.ones_like(amplitudes),
            where=amplitudes > 1e-6,
        )
        pairs *= amplitude_scale[:, :, None]
    pairs *= np.float32(10.0 ** (-profile.attenuation_db / 20.0))
    if parameters.jitter_sigma > 0.0:
        if profile.deformation_mode == "gain_jitter":
            gains = np.exp(np.float32(parameters.jitter_sigma) * profile_field)
            gains /= np.mean(gains, axis=1, keepdims=True)
            pairs *= gains[:, :, None]
        elif profile.deformation_mode == "rank_preserving":
            sensing_pairs = pairs[:, sensing_indices]
            amplitudes = np.sqrt(np.sum(sensing_pairs * sensing_pairs, axis=2))
            jittered_scores = amplitudes * np.exp(
                np.float32(parameters.jitter_sigma)
                * profile_field[:, sensing_indices]
            )
            source_order = np.argsort(amplitudes, axis=1)
            target_order = np.argsort(jittered_scores, axis=1)
            rows = np.arange(len(pairs))[:, None]
            sorted_pairs = sensing_pairs[rows, source_order].copy()
            sensing_pairs[rows, target_order] = sorted_pairs
            pairs[:, sensing_indices] = sensing_pairs
        else:
            raise ValueError(
                f"Unsupported deformation mode: {profile.deformation_mode}"
            )
    pre_control_amplitudes = np.sqrt(
        np.sum(pairs[:, sensing_indices] * pairs[:, sensing_indices], axis=2)
    )
    pre_control_means = np.mean(pre_control_amplitudes, axis=1, keepdims=True)
    pre_control_normalized = np.divide(
        pre_control_amplitudes,
        pre_control_means,
        out=np.ones_like(pre_control_amplitudes),
        where=pre_control_means > 1e-6,
    )
    spatial_turbulence = np.std(pre_control_normalized, axis=1)
    median_turbulence = float(np.median(spatial_turbulence))
    target_turbulence = median_turbulence + parameters.turbulence_retention * (
        spatial_turbulence - median_turbulence
    )
    target_turbulence += (
        median_turbulence * parameters.turbulence_noise * turbulence_field
    )
    target_turbulence *= parameters.spatial_spread
    target_turbulence = np.maximum(target_turbulence, 1e-4)
    if parameters.noise_sigma_abs > 0.0:
        pairs += np.float32(parameters.noise_sigma_abs) * noise_field
    for _ in range(3):
        sensing_pairs = pairs[:, sensing_indices]
        amplitudes = np.sqrt(np.sum(sensing_pairs * sensing_pairs, axis=2))
        packet_means = np.mean(amplitudes, axis=1, keepdims=True)
        normalized = np.divide(
            amplitudes,
            packet_means,
            out=np.ones_like(amplitudes),
            where=packet_means > 1e-6,
        )
        current_turbulence = np.std(normalized, axis=1)
        scale = np.divide(
            target_turbulence,
            current_turbulence,
            out=np.ones_like(current_turbulence),
            where=current_turbulence > 1e-6,
        )
        controlled = 1.0 + (normalized - 1.0) * scale[:, None]
        controlled = np.maximum(controlled, 0.0)
        target_amplitudes = controlled * packet_means
        amplitude_scale = np.divide(
            target_amplitudes,
            amplitudes,
            out=np.ones_like(amplitudes),
            where=amplitudes > 1e-6,
        )
        sensing_pairs *= amplitude_scale[:, :, None]
        pairs[:, sensing_indices] = sensing_pairs
        pairs = np.clip(np.rint(pairs), -128, 127)
    return pairs.reshape(csi.shape).astype(np.int8)


def extract_feature_medians(csi_data: np.ndarray) -> Dict[str, float]:
    """Measure production Core-6 medians at the runtime evaluation cadence."""
    detector = MLDetector(window_size=config.SEG_WINDOW_SIZE)
    cadence = make_evaluation_cadence(config.EVALUATION_INTERVAL)
    rows = []
    for packet_index, packet in enumerate(csi_data):
        detector.process_packet(packet, config.DEFAULT_SUBCARRIERS)
        if not cadence.note_evaluation_tick():
            continue
        if packet_index < config.SEG_WINDOW_SIZE or not detector.is_ready():
            continue
        rows.append(tuple(float(value) for value in detector._extract_features()))
    if not rows:
        raise ValueError(
            "Dataset needs more than "
            f"{config.SEG_WINDOW_SIZE} packets for feature extraction"
        )
    matrix = np.asarray(rows, dtype=np.float64)
    return {
        name: float(np.median(matrix[:, index]))
        for index, name in enumerate(FEATURE_NAMES)
    }


FEATURE_ERROR_FLOORS = {
    "turb_mad_over_mean": 0.02,
    "turb_skewness": 0.25,
    "turb_autocorr": 0.10,
    "l1_delta": 0.02,
    "l1_delta_std": 0.01,
    "l1_delta_waveform_length": 0.50,
}


def feature_fit_errors(
    achieved: Dict[str, float], target: Dict[str, float]
) -> Dict[str, float]:
    """Return scale-aware absolute errors for the production feature set."""
    return {
        name: abs(achieved[name] - target[name])
        / max(abs(target[name]), FEATURE_ERROR_FLOORS[name])
        for name in FEATURE_NAMES
    }


def _feature_fit_score(
    achieved: Dict[str, float], target: Dict[str, float]
) -> float:
    errors = feature_fit_errors(achieved, target)
    return float(np.mean(tuple(errors.values())))


def calibrate_impairment_parameters(
    csi_data: np.ndarray,
    *,
    profile: LowRssiProfile,
    target_metrics: Dict[str, float],
    innovations: np.ndarray,
    turbulence_innovations: np.ndarray,
    noise_field: np.ndarray,
    baseline_profile: np.ndarray,
    preserve_motion: bool = False,
) -> tuple[ImpairmentParameters, Dict[str, float], Dict[str, float]]:
    """Fit the weak-link transform jointly against all production ML features."""
    sample_count = min(len(csi_data), CALIBRATION_MAX_PACKETS)
    sample_csi = csi_data[:sample_count]
    sample_innovations = innovations[:sample_count]
    sample_turbulence_innovations = turbulence_innovations[:sample_count]
    sample_noise = noise_field[:sample_count]
    field_cache: Dict[float, np.ndarray] = {}
    turbulence_field_cache: Dict[tuple[float, float], np.ndarray] = {}

    def measure(parameters: ImpairmentParameters) -> Dict[str, float]:
        rho_key = round(parameters.temporal_rho, 6)
        profile_field = field_cache.get(rho_key)
        if profile_field is None:
            profile_field = _temporal_profile_field(sample_innovations, rho_key)
            field_cache[rho_key] = profile_field
        turbulence_key = (
            round(parameters.turbulence_rho, 6),
            round(parameters.turbulence_skew, 6),
        )
        turbulence_field = turbulence_field_cache.get(turbulence_key)
        if turbulence_field is None:
            turbulence_field = _temporal_turbulence_field(
                sample_turbulence_innovations, *turbulence_key
            )
            turbulence_field_cache[turbulence_key] = turbulence_field
        degraded = degrade_csi(
            sample_csi,
            profile=profile,
            parameters=parameters,
            baseline_profile=baseline_profile,
            profile_field=profile_field,
            turbulence_field=turbulence_field,
            noise_field=sample_noise,
        )
        return extract_feature_medians(degraded)

    current = ImpairmentParameters(
        source_retention=1.0,
        jitter_sigma=0.0,
        temporal_rho=profile.temporal_rho,
        noise_sigma_abs=profile.noise_sigma_abs,
        spatial_spread=1.0,
        turbulence_retention=0.75 if preserve_motion else 0.5,
        turbulence_noise=min(
            0.5, target_metrics["turb_mad_over_mean"] / 0.6745
        ),
        turbulence_rho=min(
            0.95, max(0.0, target_metrics["turb_autocorr"])
        ),
        turbulence_skew=min(2.0, max(0.0, target_metrics["turb_skewness"] / 2.0)),
    )
    current_metrics = measure(current)
    target_l1 = target_metrics["l1_delta"]
    lower_sigma = 0.0
    upper_sigma = MAX_JITTER_SIGMA
    best_l1_parameters = current
    best_l1_metrics = current_metrics
    best_l1_error = abs(current_metrics["l1_delta"] - target_l1)
    for _ in range(8):
        candidate_sigma = (lower_sigma + upper_sigma) / 2.0
        candidate = replace(current, jitter_sigma=candidate_sigma)
        candidate_metrics = measure(candidate)
        candidate_error = abs(candidate_metrics["l1_delta"] - target_l1)
        if candidate_error < best_l1_error:
            best_l1_parameters = candidate
            best_l1_metrics = candidate_metrics
            best_l1_error = candidate_error
        if candidate_metrics["l1_delta"] < target_l1:
            lower_sigma = candidate_sigma
        else:
            upper_sigma = candidate_sigma
    current = best_l1_parameters
    current_metrics = best_l1_metrics
    current_score = _feature_fit_score(current_metrics, target_metrics)
    steps = {
        "source_retention": 0.25,
        "jitter_sigma": 0.04,
        "temporal_rho": 0.25,
        "spatial_spread": 0.5,
        "turbulence_retention": 0.4,
        "turbulence_noise": 0.04,
        "turbulence_rho": 0.25,
        "turbulence_skew": 0.5,
        "noise_sigma_abs": max(0.1, profile.noise_sigma_abs / 2.0),
    }
    bounds = {
        "source_retention": (0.75 if preserve_motion else 0.0, 1.0),
        "jitter_sigma": (0.0, MAX_JITTER_SIGMA),
        "temporal_rho": (0.0, 0.95),
        "spatial_spread": (0.5, 3.0),
        "noise_sigma_abs": (0.0, max(1.0, profile.noise_sigma_abs * 2.0)),
        "turbulence_retention": (0.65 if preserve_motion else 0.0, 1.0),
        "turbulence_noise": (0.0, 0.5),
        "turbulence_rho": (0.0, 0.95),
        "turbulence_skew": (0.0, 3.0),
    }
    parameter_names = tuple(steps)
    for _ in range(CALIBRATION_ROUNDS):
        for name in parameter_names:
            for direction in (-1.0, 1.0):
                candidate_values = {
                    field: getattr(current, field) for field in parameter_names
                }
                lower, upper = bounds[name]
                candidate_values[name] = min(
                    upper,
                    max(lower, candidate_values[name] + direction * steps[name]),
                )
                candidate = ImpairmentParameters(**candidate_values)
                if candidate == current:
                    continue
                candidate_metrics = measure(candidate)
                candidate_score = _feature_fit_score(candidate_metrics, target_metrics)
                if candidate_score < current_score:
                    current = candidate
                    current_metrics = candidate_metrics
                    current_score = candidate_score
        steps = {name: value / 2.0 for name, value in steps.items()}

    return current, current_metrics, feature_fit_errors(current_metrics, target_metrics)


def _find_group_parameters(
    info: Dict[str, Any], group_id: str
) -> Optional[ImpairmentParameters]:
    for label, entries in info.get("files", {}).items():
        for entry in entries:
            if not entry.get("synthetic"):
                continue
            path = dataset_metadata.resolve_entry_path(label, entry)
            with np.load(path, allow_pickle=False) as generated:
                if "generation_group" not in generated:
                    continue
                stored_group = str(_scalar(generated["generation_group"]))
                if stored_group != group_id:
                    continue
                missing = [
                    name for name in IMPAIRMENT_PARAMETER_NAMES if name not in generated
                ]
                if missing:
                    missing_fields = ", ".join(missing)
                    raise ValueError(
                        f"Synthetic quiet dataset lacks parameters: {missing_fields}"
                    )
                return ImpairmentParameters(
                    **{
                        name: float(generated[name])
                        for name in IMPAIRMENT_PARAMETER_NAMES
                    }
                )
    return None


def _effective_impairment_parameters(
    parameters: ImpairmentParameters,
    *,
    label: str,
    generation_mode: str,
) -> ImpairmentParameters:
    """Preserve source motion dynamics while reusing the quiet link model."""
    if generation_mode != "shared_session" or label != "motion":
        return parameters
    return replace(
        parameters,
        source_retention=1.0,
        spatial_spread=(
            parameters.spatial_spread * SHARED_SESSION_MOTION_SPATIAL_BOOST
        ),
        turbulence_retention=1.0,
    )


def _validate_registered_output_path(path: Path, label: str) -> None:
    expected_parent = (dataset_metadata.DATA_DIR / label).resolve()
    if path.parent.resolve() != expected_parent:
        raise ValueError(
            f"Registered output must be stored directly under data/{label}/"
        )


def _format_device_id(entry: Dict[str, Any]) -> str:
    return str(entry.get("device_id", ""))


def _build_output_entry(
    *,
    source_entry: Dict[str, Any],
    source_path: Path,
    output_path: Path,
    profile: LowRssiProfile,
    packet_count: int,
    generated_at: str,
) -> Dict[str, Any]:
    description = (
        f"Synthetic low-RSSI derivative of {source_path.name} using the "
        f"{profile.name} profile. Not a real capture."
    )
    entry = {
        "filename": output_path.name,
        "chip": str(source_entry.get("chip", "unknown")).upper(),
        "subcarriers": int(source_entry.get("subcarriers", 0) or 0),
        "contributor": str(source_entry.get("contributor", "")),
        "collected_at": str(source_entry.get("collected_at", "")),
        "generated_at": generated_at,
        "duration_ms": int(source_entry.get("duration_ms", 0) or 0),
        "num_packets": int(packet_count),
        "description": description,
        "environment": str(source_entry.get("environment", "")),
        "device_id": _format_device_id(source_entry),
        "low_rssi": True,
        "synthetic": True,
    }
    return entry


def _upsert_entry(info: Dict[str, Any], label: str, entry: Dict[str, Any]) -> None:
    entries = info.setdefault("files", {}).setdefault(label, [])
    for index, current in enumerate(entries):
        if current.get("filename") == entry["filename"]:
            entries[index] = entry
            return
    entries.append(entry)


def _link_synthetic_pair(
    info: Dict[str, Any], group_id: str
) -> Optional[tuple[Dict[str, Any], Dict[str, Any]]]:
    matches: Dict[str, Dict[str, Any]] = {}
    for label in ("static_presence", "motion"):
        for entry in info.get("files", {}).get(label, []):
            if not entry.get("synthetic"):
                continue
            path = dataset_metadata.resolve_entry_path(label, entry)
            if not path.exists():
                continue
            with np.load(path, allow_pickle=False) as generated:
                if "generation_group" not in generated:
                    continue
                stored_group = str(_scalar(generated["generation_group"]))
            if stored_group == group_id:
                matches[label] = entry
    if set(matches) != {"static_presence", "motion"}:
        return None
    static_entry = matches["static_presence"]
    motion_entry = matches["motion"]
    static_entry["optimal_pair_motion_file"] = motion_entry["filename"]
    motion_entry["optimal_pair_static_presence_file"] = static_entry["filename"]
    return static_entry, motion_entry


def _print_feature_report(
    source_metrics: Dict[str, float],
    achieved_metrics: Dict[str, float],
    profile: LowRssiProfile,
    label: str,
) -> None:
    reference_label = "static_presence" if label == "empty" else label
    reference = profile.reference_feature_medians.get(reference_label, {})
    print("\nFeature medians:")
    print("  feature                         source   synthetic   reference")
    for name in FEATURE_NAMES:
        print(
            f"  {name:<30} {source_metrics[name]:>8.4f} "
            f"{achieved_metrics[name]:>11.4f} "
            f"{reference.get(name, float('nan')):>11.4f}"
        )


def _print_classic_pair_result(
    pair: tuple[Dict[str, Any], Dict[str, Any]],
) -> None:
    static_entry, motion_entry = pair
    static_path = dataset_metadata.resolve_entry_path("static_presence", static_entry)
    motion_path = dataset_metadata.resolve_entry_path("motion", motion_entry)
    static_packets = load_npz_as_packets(static_path)
    motion_packets = load_npz_as_packets(motion_path)
    result = compute_classic_packet_result(
        static_packets,
        motion_packets,
        tuple(config.DEFAULT_SUBCARRIERS),
        config.SEG_WINDOW_SIZE,
    )
    if result is None:
        print("\nClassic synthetic-pair replay: startup calibration failed")
        return
    threshold, metrics = result
    print("\nClassic synthetic-pair replay:")
    print(f"  threshold: {threshold:.6f}")
    print(f"  recall:    {metrics['recall']:.2f}%")
    print(f"  FP rate:   {metrics['fp_rate']:.2f}%")
    print(f"  precision: {metrics['precision']:.2f}%")
    print(f"  alarms:    {metrics['effective_alarms']}")


def generate_dataset(
    source: str | Path,
    *,
    profile_name: str,
    seed: int,
    generation_mode: str = "reference_match",
    output_path: Optional[Path] = None,
    jitter_sigma: Optional[float] = None,
    register: bool = True,
    force: bool = False,
    dataset_info_path: Optional[Path] = None,
) -> tuple[Path, Optional[Dict[str, Any]]]:
    """Generate and optionally register one synthetic low-RSSI dataset."""
    info = dataset_metadata.load_dataset_info(dataset_info_path)
    source_path, label, source_entry, info = resolve_source_dataset(source, info=info)
    profile = LOW_RSSI_PROFILES[profile_name]
    if generation_mode not in ("reference_match", "shared_session"):
        raise ValueError(f"Unsupported generation mode: {generation_mode}")
    group_id = build_generation_group(
        label, source_entry, profile.name, seed, generation_mode
    )

    if output_path is None:
        output_path = build_output_path(
            source_path,
            label,
            profile.name,
            generation_mode,
            seed,
        )
    output_path = Path(output_path).resolve()
    if output_path.suffix.lower() != ".npz":
        raise ValueError("Output path must use the .npz extension")
    if output_path == source_path:
        raise ValueError("Output path must not overwrite the real source dataset")
    if output_path.exists() and not force:
        raise FileExistsError(f"Output already exists: {output_path}")
    if register:
        _validate_registered_output_path(output_path, label)

    with np.load(source_path, allow_pickle=False) as source_npz:
        payload = {key: np.asarray(source_npz[key]) for key in source_npz.files}
    if "csi_data" not in payload:
        raise ValueError(f"No csi_data array in {source_path}")
    embedded_label = str(_scalar(payload.get("label", label)))
    if embedded_label != label:
        raise ValueError(
            f"Source label mismatch: catalog={label!r}, NPZ={embedded_label!r}"
        )

    packet_count = len(payload["csi_data"])
    rng = np.random.default_rng(_stable_seed(seed, source_path.name))
    drop = _drop_mask(packet_count, profile, rng)
    payload = _apply_packet_mask(payload, ~drop)
    csi = np.asarray(payload["csi_data"], dtype=np.int8)
    subcarriers = csi.shape[1] // 2
    innovations = rng.normal(0.0, 1.0, size=(len(csi), subcarriers)).astype(np.float32)
    turbulence_innovations = rng.normal(0.0, 1.0, size=len(csi)).astype(np.float32)
    noise_field = rng.normal(0.0, 1.0, size=(len(csi), subcarriers, 2)).astype(
        np.float32
    )

    source_metrics = extract_feature_medians(csi)
    reference_label = "static_presence" if label == "empty" else label
    target_metrics = profile.reference_feature_medians[reference_label]
    baseline_profile = _normalized_baseline_profile(csi)
    if jitter_sigma is not None:
        parameters = ImpairmentParameters(
            source_retention=1.0,
            jitter_sigma=float(jitter_sigma),
            temporal_rho=profile.temporal_rho,
            noise_sigma_abs=profile.noise_sigma_abs,
            spatial_spread=1.0,
            turbulence_retention=1.0,
            turbulence_noise=0.0,
            turbulence_rho=profile.temporal_rho,
            turbulence_skew=0.0,
        )
    elif generation_mode == "reference_match" or label in QUIET_LABELS:
        parameters, _, _ = calibrate_impairment_parameters(
            csi,
            profile=profile,
            target_metrics=target_metrics,
            innovations=innovations,
            turbulence_innovations=turbulence_innovations,
            noise_field=noise_field,
            baseline_profile=baseline_profile,
            preserve_motion=generation_mode == "shared_session",
        )
    else:
        parameters = _find_group_parameters(info, group_id)
        if parameters is None:
            raise ValueError(
                "Generate the paired static_presence dataset first, or pass "
                "--jitter-sigma explicitly. Motion must reuse quiet-session "
                "calibration."
            )
    if parameters.jitter_sigma < 0.0 or parameters.jitter_sigma > MAX_JITTER_SIGMA:
        raise ValueError(f"jitter_sigma must be between 0 and {MAX_JITTER_SIGMA}")

    profile_field = _temporal_profile_field(innovations, parameters.temporal_rho)
    turbulence_field = _temporal_turbulence_field(
        turbulence_innovations,
        parameters.turbulence_rho,
        parameters.turbulence_skew,
    )
    effective_parameters = _effective_impairment_parameters(
        parameters,
        label=label,
        generation_mode=generation_mode,
    )
    degraded = degrade_csi(
        csi,
        profile=profile,
        parameters=effective_parameters,
        baseline_profile=baseline_profile,
        profile_field=profile_field,
        turbulence_field=turbulence_field,
        noise_field=noise_field,
    )
    payload["csi_data"] = degraded

    source_rssi = payload.get("rssi_dbm")
    target_rssi_dbm = (
        profile.target_motion_rssi_dbm
        if generation_mode == "reference_match" and label == "motion"
        else profile.target_quiet_rssi_dbm
    )
    if source_rssi is None or np.asarray(source_rssi).ndim == 0:
        payload["rssi_dbm"] = np.full(
            len(degraded), int(round(target_rssi_dbm)), dtype=np.int16
        )
    else:
        rssi = np.asarray(source_rssi, dtype=np.float64)
        shift = target_rssi_dbm - float(np.median(rssi))
        payload["rssi_dbm"] = np.rint(rssi + shift).astype(np.int16)

    achieved_metrics = extract_feature_medians(degraded)
    fit_errors = feature_fit_errors(achieved_metrics, target_metrics)
    generated_at = datetime.now().isoformat()
    payload["low_rssi"] = np.asarray(True)
    payload["synthetic"] = np.asarray(True)
    payload["source_dataset"] = np.asarray(source_path.name)
    payload["low_rssi_profile"] = np.asarray(profile.name)
    payload["deformation_mode"] = np.asarray(profile.deformation_mode)
    payload["reference_datasets"] = np.asarray(profile.reference_datasets)
    payload["generation_seed"] = np.asarray(int(seed), dtype=np.int64)
    payload["generation_group"] = np.asarray(group_id)
    payload["generation_mode"] = np.asarray(generation_mode)
    payload["generated_at"] = np.asarray(generated_at)
    payload["generator_version"] = np.asarray(GENERATOR_VERSION, dtype=np.int16)
    payload["feature_names"] = np.asarray(FEATURE_NAMES)
    payload["source_feature_medians"] = np.asarray(
        [source_metrics[name] for name in FEATURE_NAMES], dtype=np.float32
    )
    payload["target_feature_medians"] = np.asarray(
        [target_metrics[name] for name in FEATURE_NAMES], dtype=np.float32
    )
    payload["synthetic_feature_medians"] = np.asarray(
        [achieved_metrics[name] for name in FEATURE_NAMES], dtype=np.float32
    )
    payload["feature_relative_errors"] = np.asarray(
        [fit_errors[name] for name in FEATURE_NAMES], dtype=np.float32
    )
    payload["mean_feature_relative_error"] = np.asarray(
        np.mean(tuple(fit_errors.values())), dtype=np.float32
    )
    payload["source_retention"] = np.asarray(
        effective_parameters.source_retention, dtype=np.float32
    )
    payload["jitter_sigma"] = np.asarray(
        effective_parameters.jitter_sigma, dtype=np.float32
    )
    payload["temporal_rho"] = np.asarray(
        effective_parameters.temporal_rho, dtype=np.float32
    )
    payload["noise_sigma_abs"] = np.asarray(
        effective_parameters.noise_sigma_abs, dtype=np.float32
    )
    payload["spatial_spread"] = np.asarray(
        effective_parameters.spatial_spread, dtype=np.float32
    )
    payload["turbulence_retention"] = np.asarray(
        effective_parameters.turbulence_retention, dtype=np.float32
    )
    payload["turbulence_noise"] = np.asarray(
        effective_parameters.turbulence_noise, dtype=np.float32
    )
    payload["turbulence_rho"] = np.asarray(
        effective_parameters.turbulence_rho, dtype=np.float32
    )
    payload["turbulence_skew"] = np.asarray(
        effective_parameters.turbulence_skew, dtype=np.float32
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)

    output_entry = None
    pair = None
    if register:
        output_entry = _build_output_entry(
            source_entry=source_entry,
            source_path=source_path,
            output_path=output_path,
            profile=profile,
            packet_count=len(degraded),
            generated_at=generated_at,
        )
        _upsert_entry(info, label, output_entry)
        pair = _link_synthetic_pair(info, group_id)
        info["updated_at"] = generated_at
        dataset_metadata.save_dataset_info(info, dataset_info_path)

    print(f"Generated: {output_path}")
    print(f"Profile:   {profile.name} (reference {profile.reference_chip})")
    print(f"Mode:      {generation_mode}")
    print(f"Packets:   {packet_count} -> {len(degraded)}")
    print(f"Retention: {effective_parameters.source_retention:.6f}")
    print(f"Jitter:    {effective_parameters.jitter_sigma:.6f}")
    print(f"Temp rho:  {effective_parameters.temporal_rho:.6f}")
    print(f"I/Q noise: {effective_parameters.noise_sigma_abs:.6f}")
    print(f"Spread:    {effective_parameters.spatial_spread:.6f}")
    print(f"Turb keep: {effective_parameters.turbulence_retention:.6f}")
    print(f"Turb noise:{effective_parameters.turbulence_noise:>10.6f}")
    print(f"Turb rho:  {effective_parameters.turbulence_rho:.6f}")
    print(f"Turb skew: {effective_parameters.turbulence_skew:.6f}")
    print(f"Fit error: {float(np.mean(tuple(fit_errors.values()))):.4f}")
    print(f"RSSI:      median {float(np.median(payload['rssi_dbm'])):.1f} dBm")
    print("Synthetic: true")
    print(f"Group:     {group_id}")
    _print_feature_report(source_metrics, achieved_metrics, profile, label)
    if pair is not None:
        _print_classic_pair_result(pair)
    return output_path, output_entry


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a registered synthetic low-RSSI derivative from one real "
            "CSI dataset. "
            "In shared_session mode, generate static_presence first and motion second "
            "with the same profile and seed."
        )
    )
    parser.add_argument(
        "dataset", help="Registered source filename, stem, dataset id, or path"
    )
    parser.add_argument(
        "--profile",
        choices=tuple(LOW_RSSI_PROFILES),
        default="c3_weak_link",
        help="Reference weak-link behavior profile (default: c3_weak_link)",
    )
    parser.add_argument(
        "--seed", type=int, default=20260722, help="Reproducible generation seed"
    )
    parser.add_argument(
        "--mode",
        choices=("reference_match", "shared_session"),
        default="reference_match",
        help=(
            "reference_match reproduces each real phase for Classic testing; "
            "shared_session reuses quiet impairment for training candidates"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output NPZ path (default: data/<label>/...)",
    )
    parser.add_argument(
        "--jitter-sigma",
        type=float,
        default=None,
        help=(
            "Override quiet-session jitter calibration; normally reused "
            "automatically for motion"
        ),
    )
    parser.add_argument(
        "--no-register",
        action="store_true",
        help="Do not update data/dataset_info.json",
    )
    parser.add_argument(
        "--force", action="store_true", help="Replace an existing deterministic output"
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        generate_dataset(
            args.dataset,
            profile_name=args.profile,
            seed=args.seed,
            generation_mode=args.mode,
            output_path=args.output,
            jitter_sigma=args.jitter_sigma,
            register=not args.no_register,
            force=args.force,
        )
    except (FileNotFoundError, FileExistsError, KeyError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
