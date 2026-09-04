# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""Deterministic packet augmentation and stream provenance."""

from __future__ import annotations

from tools.lib.bootstrap import setup_paths

setup_paths()

import argparse
import copy
import hashlib
import inspect
import json
import numpy as np
from functools import lru_cache
from pathlib import Path
from tools.lib import npz_cache
from tools.lib.timing_quality import summarize_capture_timing
from tools.lib.csi_io import load_npz_packet_view
from tools.lib.repo_paths import tools_lib_dir

def derive_seed(base_seed, *offsets):
    """Derive a stable int32-compatible seed from a base seed."""
    if base_seed is None:
        return None
    seed = int(base_seed) & 0x7FFFFFFF
    for offset in offsets:
        seed = (seed * 1103515245 + 12345 + int(offset) * 1009) & 0x7FFFFFFF
    return seed or 1


FIXED_PACKET_AUGMENTATION_SEEDS = (20260807, 20260808)


FIXED_PACKET_AUGMENTATION_SEED = FIXED_PACKET_AUGMENTATION_SEEDS[0]


TRAINING_AUGMENT_COMPONENT_ORDER = (
    "base",
    "drift",
    "burst-loss",
)


DEFAULT_TRAINING_AUGMENT_COMPONENTS = (
    "base",
    "drift",
    "burst-loss",
)


def parse_augmentation_components(value):
    """Normalize one augmentation component set, keeping CLI compatibility."""
    if value in (None, False):
        return tuple()
    if value is True:
        return DEFAULT_TRAINING_AUGMENT_COMPONENTS
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return tuple()
        parts = text.split(",")
    elif isinstance(value, (list, tuple, set)):
        parts = value
    else:
        parts = str(value).split(",")

    normalized = []
    seen = set()
    for part in parts:
        name = str(part).strip().lower()
        if not name:
            continue
        if name not in TRAINING_AUGMENT_COMPONENT_ORDER:
            raise argparse.ArgumentTypeError(
                "augment components must be chosen from: "
                + ", ".join(TRAINING_AUGMENT_COMPONENT_ORDER)
            )
        if name in seen:
            continue
        seen.add(name)
        normalized.append(name)
    if not normalized:
        return tuple()
    ordered = [
        name for name in TRAINING_AUGMENT_COMPONENT_ORDER
        if name in seen
    ]
    return tuple(ordered)


def format_augmentation_components(components):
    """Return a stable, user-facing component list."""
    normalized = parse_augmentation_components(components)
    return "none" if not normalized else ",".join(normalized)


def _first_non_empty(mapping, keys):
    """Return the first non-empty string-like value for a list of keys."""
    for key in keys:
        value = mapping.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _prepare_feature_packets_for_record(record, packet_augmentation=None, augmentation_seed=None):
    """Return one per-file packet list ready for feature extraction."""
    packets = []
    for idx, packet in enumerate(_ensure_record_packets(record)):
        copied = dict(packet)
        copied['source_file'] = record['path'].name
        copied['packet_index'] = idx
        packets.append(copied)
    if packet_augmentation:
        return augment_csi_packets(packets, packet_augmentation, augmentation_seed)
    return packets


@lru_cache(maxsize=None)
def _implementation_source_digest(*objects):
    """Hash only the implementation objects that affect one cached stream."""
    digest = hashlib.sha256()
    for implementation in objects:
        source = inspect.getsource(implementation)
        digest.update(implementation.__qualname__.encode('utf-8'))
        digest.update(b'\0')
        digest.update(source.encode('utf-8'))
        digest.update(b'\0')
    return digest.hexdigest()


@lru_cache(maxsize=64)
def _packet_augmentation_stream_provenance_cached(config_json, augmentation_seed):
    """Build one immutable-by-convention packet-transform provenance value."""
    return {
        'transform': 'training_packet_augmentation_v2',
        'config': json.loads(config_json),
        'seed': int(augmentation_seed),
        'implementation_sha256': _implementation_source_digest(
            _prepare_feature_packets_for_record,
            _ensure_record_packets,
            _normalize_augmentation_range,
            _estimate_packet_rate_pps,
            _resample_stable_packet_rate,
            _smoothed_iq_profile,
            _add_scaled_iq_noise,
            _stable_text_seed,
            derive_seed,
            augment_csi_packets,
        ),
        'timing_quality': npz_cache.source_manifest(
            tools_lib_dir() / 'timing_quality.py'
        ),
    }


def _packet_augmentation_stream_provenance(packet_augmentation, augmentation_seed):
    """Return a stable cache identity for one deterministic packet transform."""
    if not packet_augmentation or augmentation_seed is None:
        return None
    config_json = json.dumps(
        dict(packet_augmentation),
        sort_keys=True,
        separators=(',', ':'),
    )
    return copy.deepcopy(
        _packet_augmentation_stream_provenance_cached(
            config_json,
            int(augmentation_seed),
        )
    )


def training_packet_augmentation_seed(packet_augmentation):
    """Return the primary fixed seed used by single-view stress diagnostics."""
    if not packet_augmentation:
        return None
    return FIXED_PACKET_AUGMENTATION_SEED


def training_packet_augmentation_seeds(packet_augmentation):
    """Return the promoted fixed augmentation views used for model training."""
    if not packet_augmentation:
        return tuple()
    return FIXED_PACKET_AUGMENTATION_SEEDS


def _normalize_augmentation_range(value, *, name, integer=False, minimum=0.0):
    """Return one validated inclusive numeric (min, max) range."""
    if isinstance(value, np.ndarray):
        parts = value.tolist()
    elif isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        parts = [item.strip() for item in str(value).split(',') if str(item).strip()]
    if len(parts) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    try:
        first = int(parts[0]) if integer else float(parts[0])
        second = int(parts[1]) if integer else float(parts[1])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain valid numeric values") from exc
    if first < minimum or second < minimum or second < first:
        raise ValueError(f"{name} must satisfy {minimum} <= min <= max")
    return first, second


def _estimate_packet_rate_pps(packets):
    """Estimate effective source throughput through the shared timing policy."""
    summary = summarize_capture_timing(packets)
    return max(1e-9, float(summary["packet_rate_pps"]))


DEFAULT_MIN_TARGET_RATE_PPS = 70.0


def _resample_stable_packet_rate(packets, rate_scale, min_target_rate_pps=DEFAULT_MIN_TARGET_RATE_PPS):
    """Return a lower-rate stable stream without modelling the drop as loss."""
    scale = float(rate_scale)
    min_target_rate_pps = float(min_target_rate_pps)
    if scale >= 1.0 or len(packets) < 2:
        return [dict(packet) for packet in packets]

    source_rate_pps = _estimate_packet_rate_pps(packets)
    target_rate_pps = max(min_target_rate_pps, source_rate_pps * scale)
    if target_rate_pps >= source_rate_pps:
        return [dict(packet) for packet in packets]

    stride = source_rate_pps / target_rate_pps
    interval_us = max(1, int(round(1_000_000.0 / target_rate_pps)))
    selected = []
    cursor = 0.0
    next_seq_num = None
    next_device_ticks_us = None
    next_wifi_rx_ts_us = None
    next_wifi_rx_start_ts_ns = None
    while True:
        source_index = int(round(cursor))
        if source_index >= len(packets):
            break
        packet = dict(packets[source_index])
        if next_seq_num is None:
            next_seq_num = int(packet.get('seq_num', packet.get('stream_seq_num', 0)) or 0)
        else:
            next_seq_num += 1
        packet['seq_num'] = next_seq_num
        packet['stream_seq_num'] = next_seq_num
        if packet.get('device_ticks_us') is not None:
            if next_device_ticks_us is None:
                next_device_ticks_us = int(packet['device_ticks_us'])
            else:
                next_device_ticks_us += interval_us
            packet['device_ticks_us'] = next_device_ticks_us
        if packet.get('wifi_rx_ts_us') is not None:
            if next_wifi_rx_ts_us is None:
                next_wifi_rx_ts_us = int(packet['wifi_rx_ts_us'])
            else:
                next_wifi_rx_ts_us = (next_wifi_rx_ts_us + interval_us) % (1 << 32)
            packet['wifi_rx_ts_us'] = next_wifi_rx_ts_us
        if packet.get('wifi_rx_start_ts_ns') is not None:
            if next_wifi_rx_start_ts_ns is None:
                next_wifi_rx_start_ts_ns = int(packet['wifi_rx_start_ts_ns'])
            else:
                next_wifi_rx_start_ts_ns += interval_us * 1000
            packet['wifi_rx_start_ts_ns'] = next_wifi_rx_start_ts_ns
        selected.append(packet)
        cursor += stride
    return selected


def _smoothed_iq_profile(rng, usable_subcarriers):
    """Return one smooth, unit-RMS per-tone I/Q perturbation template."""
    profile = rng.normal(0.0, 1.0, size=(int(usable_subcarriers), 2))
    if usable_subcarriers > 1:
        kernel = np.asarray([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float64)
        kernel /= np.sum(kernel)
        for axis in range(profile.shape[1]):
            profile[:, axis] = np.convolve(profile[:, axis], kernel, mode='same')
    rms = float(np.sqrt(np.mean(np.square(profile))))
    if rms <= 0.0:
        return profile
    return profile / rms


def _constant_object_array(value, length):
    """Return one object array filled with one repeated value."""
    array = np.empty(int(length), dtype=object)
    array[:] = value
    return array


def _build_sample_context_for_replay_rows(record, rows):
    """Return per-row sample context for one canonical replay-row payload."""
    count = int(len(rows['X']))
    return {
        'chip': _constant_object_array(str(record['chip']).upper(), count),
        'source_file': _constant_object_array(record['path'].name, count),
        'lineage_group': _constant_object_array(record['lineage_group'], count),
        'session_group': _constant_object_array(record['session_group'], count),
        'environment_group': _constant_object_array(record['environment_group'], count),
        'pair_id': _constant_object_array(record['pair_id'], count),
        'day_group': _constant_object_array(record['day_group'], count),
        'dataset_role': _constant_object_array(record['dataset_role'], count),
        'timing_quality_status': _constant_object_array(record['timing_quality_status'], count),
        'timing_quality_bucket': _constant_object_array(record['timing_quality_bucket'], count),
        'synthetic': np.full(count, bool(record['synthetic']), dtype=bool),
        'label_name': _constant_object_array(record['label_name'], count),
        'packet_index': np.asarray(rows['packet_index'], dtype=np.int32),
        'window_index': np.asarray(rows['evaluation_index'], dtype=np.int32),
        'reset_index': np.asarray(rows['reset_index'], dtype=np.int32),
    }


def _ensure_record_packets(record):
    """Materialize record packets only when a cache miss needs them."""
    packets = record.get('packets')
    if packets is not None:
        return packets
    loader = record.get('packets_loader')
    if callable(loader):
        packets = loader()
    else:
        packets = _load_npz_packets_cached(record['path'])
    record['packets'] = packets
    return packets


def _stable_text_seed(value):
    digest = hashlib.sha256(str(value).encode('utf-8')).digest()
    return int.from_bytes(digest[:4], byteorder='little') & 0x7FFFFFFF


def _add_scaled_iq_noise(raw, usable, noise_sigma, rng):
    """Add per-tone relative Gaussian noise with one vectorized RNG draw."""
    if usable <= 0 or noise_sigma <= 0.0:
        return
    tone_view = raw[:2 * usable].reshape(usable, 2)
    magnitudes = np.maximum(1.0, np.linalg.norm(tone_view, axis=1))
    noise_scale = noise_sigma * magnitudes[:, None] / np.sqrt(2.0)
    tone_view += rng.normal(0.0, noise_scale, size=(usable, 2))


def augment_csi_packets(packets, config, seed):
    """Return a deterministic packet-level augmented copy for training only."""
    if not config:
        return list(packets)
    noise_sigma = float(config.get('noise_sigma', 0.0))
    packet_loss = float(config.get('packet_loss', 0.0))
    stutter_probability = float(config.get('stutter_probability', 0.0))
    drift_sigma = float(config.get('drift_sigma', 0.0))
    drift_episode_count = int(config.get('drift_episode_count', 0))
    drift_duration_seconds = _normalize_augmentation_range(
        config.get('drift_duration_seconds', (20.0, 60.0)),
        name='drift_duration_seconds',
    )
    burst_loss_starts_per_minute = float(config.get('burst_loss_starts_per_minute', 0.0))
    burst_length_packets = _normalize_augmentation_range(
        config.get('burst_length_packets', (2, 6)),
        name='burst_length_packets',
        integer=True,
        minimum=1,
    )
    packet_rate_scale = _normalize_augmentation_range(
        config.get('packet_rate_scale', (1.0, 1.0)),
        name='packet_rate_scale',
        minimum=0.0,
    )
    min_target_rate_pps = float(config.get('min_target_rate_pps', DEFAULT_MIN_TARGET_RATE_PPS))
    if (
        min(
            noise_sigma,
            packet_loss,
            stutter_probability,
            drift_sigma,
            burst_loss_starts_per_minute,
            min_target_rate_pps,
        ) < 0.0
        or packet_loss >= 1.0
        or stutter_probability > 1.0
        or drift_episode_count < 0
        or packet_rate_scale[0] <= 0.0
        or packet_rate_scale[1] > 1.0
        or min_target_rate_pps <= 0.0
    ):
        raise ValueError("invalid packet augmentation parameters")

    grouped = {}
    for packet in packets:
        grouped.setdefault(str(packet.get('source_file', '__single_stream__')), []).append(packet)
    augmented = []
    for source in sorted(grouped):
        rng = np.random.default_rng(derive_seed(seed, _stable_text_seed(source)))
        source_packets = grouped[source]
        rate_scale = float(rng.uniform(*packet_rate_scale))
        source_packets = _resample_stable_packet_rate(
            source_packets,
            rate_scale,
            min_target_rate_pps=min_target_rate_pps,
        )
        packet_rate_pps = _estimate_packet_rate_pps(source_packets)
        burst_start_probability = min(
            1.0,
            burst_loss_starts_per_minute / 60.0 / max(packet_rate_pps, 1e-9),
        )
        min_burst_packets, max_burst_packets = burst_length_packets
        drift_episodes = []
        if drift_sigma > 0.0 and drift_episode_count > 0 and source_packets:
            example_raw = np.asarray(source_packets[0]['csi_data'], dtype=np.float64)
            usable = len(example_raw) // 2
            if usable > 0:
                source_matrix = np.asarray(
                    [np.asarray(packet['csi_data'], dtype=np.float64)[:2 * usable] for packet in source_packets],
                    dtype=np.float64,
                ).reshape(len(source_packets), usable, 2)
                tone_scale = np.median(np.linalg.norm(source_matrix, axis=2), axis=0)
                tone_scale = np.maximum(tone_scale, 1.0)[:, None] / np.sqrt(2.0)
                min_seconds, max_seconds = drift_duration_seconds
                for _ in range(drift_episode_count):
                    duration_seconds = rng.uniform(min_seconds, max_seconds)
                    duration_packets = int(round(duration_seconds * packet_rate_pps))
                    duration_packets = max(2, min(duration_packets, len(source_packets)))
                    max_start = max(0, len(source_packets) - duration_packets)
                    start_index = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
                    template = _smoothed_iq_profile(rng, usable) * tone_scale
                    amplitude = float(rng.normal(0.0, drift_sigma))
                    drift_episodes.append((start_index, start_index + duration_packets, amplitude, template))
        previous_emitted = None
        burst_skip_remaining = 0
        for packet_index, packet in enumerate(source_packets):
            if burst_skip_remaining > 0:
                burst_skip_remaining -= 1
                continue
            if burst_start_probability > 0.0 and rng.random() < burst_start_probability:
                burst_skip_remaining = int(rng.integers(min_burst_packets, max_burst_packets + 1))
                if burst_skip_remaining > 0:
                    burst_skip_remaining -= 1
                    continue
            if packet_loss > 0.0 and rng.random() < packet_loss:
                continue
            raw = np.asarray(packet['csi_data'], dtype=np.float64)
            stuttered = False
            if (
                stutter_probability > 0.0
                and previous_emitted is not None
                and rng.random() < stutter_probability
            ):
                raw = previous_emitted.copy()
                stuttered = True
            else:
                raw = raw.copy()
            usable = len(raw) // 2
            if usable > 0 and drift_episodes and not stuttered:
                tone_view = raw[:2 * usable].reshape(usable, 2)
                for start_index, stop_index, amplitude, template in drift_episodes:
                    if start_index <= packet_index < stop_index:
                        phase = np.pi * float((packet_index - start_index) + 1) / float(
                            (stop_index - start_index) + 1
                        )
                        tone_view += np.sin(phase) * amplitude * template[:usable]
            _add_scaled_iq_noise(raw, usable, noise_sigma, rng)
            copied = dict(packet)
            emitted = np.clip(np.rint(raw), -128, 127).astype(np.int8)
            copied['csi_data'] = emitted
            augmented.append(copied)
            previous_emitted = emitted.astype(np.float64)
    return augmented


ROBUSTNESS_WINNER_NAME = (
    'baseline_standard__feature_jitter_010__packet_rate_noise_loss_stutter_moderate'
)


ROBUSTNESS_WINNER_FEATURE_AUGMENTATION = {'jitter_sigma': 0.10}


ROBUSTNESS_WINNER_PACKET_AUGMENTATION = {
    'noise_sigma': 0.01,
    'packet_loss': 0.05,
    'stutter_probability': 0.08,
    'packet_rate_scale': (0.7, 1.0),
    'min_target_rate_pps': 70.0,
}


CORRELATED_DRIFT_PACKET_AUGMENTATION = {
    'drift_sigma': 0.035,
    'drift_episode_count': 1,
    'drift_duration_seconds': (20.0, 60.0),
}


BURST_LOSS_PACKET_AUGMENTATION = {
    'burst_loss_starts_per_minute': 0.6,
    'burst_length_packets': (2, 6),
}


TRAINING_AUGMENT_COMPONENTS = {
    'base': {
        'feature': dict(ROBUSTNESS_WINNER_FEATURE_AUGMENTATION),
        'packet': dict(ROBUSTNESS_WINNER_PACKET_AUGMENTATION),
    },
    'drift': {
        'feature': {},
        'packet': dict(CORRELATED_DRIFT_PACKET_AUGMENTATION),
    },
    'burst-loss': {
        'feature': {},
        'packet': dict(BURST_LOSS_PACKET_AUGMENTATION),
    },
}


def _merge_augmentation_dicts(*configs):
    """Merge augmentation config fragments into one plain dict."""
    merged = {}
    for config in configs:
        for key, value in dict(config or {}).items():
            merged[key] = tuple(value) if isinstance(value, (list, tuple)) else value
    return merged


def resolve_training_augmentation(augment):
    """Return active components plus merged augmentation configs."""
    components = parse_augmentation_components(augment)
    if not components:
        return tuple(), {}, {}
    feature_parts = [TRAINING_AUGMENT_COMPONENTS[name]['feature'] for name in components]
    packet_parts = [TRAINING_AUGMENT_COMPONENTS[name]['packet'] for name in components]
    return (
        components,
        _merge_augmentation_dicts(*feature_parts),
        _merge_augmentation_dicts(*packet_parts),
    )


def format_augmentation_config(feature_augmentation=None, packet_augmentation=None, *, components=None):
    """Compact one-line description of an active training augmentation recipe."""
    feature_augmentation = dict(feature_augmentation or {})
    packet_augmentation = dict(packet_augmentation or {})
    if not feature_augmentation and not packet_augmentation:
        return 'none'
    normalized_components = parse_augmentation_components(components)
    parts = []
    if normalized_components:
        parts.append(f"components={'+'.join(normalized_components)}")
    elif feature_augmentation == ROBUSTNESS_WINNER_FEATURE_AUGMENTATION and packet_augmentation == ROBUSTNESS_WINNER_PACKET_AUGMENTATION:
        parts.append(f"recipe={ROBUSTNESS_WINNER_NAME}")
    if feature_augmentation:
        parts.append(
            'feature={'
            + ', '.join(f'{key}={value}' for key, value in sorted(feature_augmentation.items()))
            + '}'
        )
    if packet_augmentation:
        parts.append(
            'packet={'
            + ', '.join(f'{key}={value}' for key, value in sorted(packet_augmentation.items()))
            + '}'
        )
    return '; '.join(parts)


def _append_augmented_training_rows(X_train_scaled, y_train, scaler, X_aug, y_aug,
                                    groups_aug, train_groups, sample_weight=None):
    """Append packet-augmented rows whose groups belong to the train split."""
    if X_aug is None or y_aug is None or groups_aug is None:
        return X_train_scaled, y_train, sample_weight
    train_group_set = {str(group) for group in train_groups}
    if not train_group_set:
        return X_train_scaled, y_train, sample_weight
    aug_mask = np.isin(np.asarray(groups_aug).astype(str), list(train_group_set))
    if not np.any(aug_mask):
        return X_train_scaled, y_train, sample_weight
    X_extra = scaler.transform(np.asarray(X_aug)[aug_mask])
    y_extra = np.asarray(y_aug)[aug_mask]
    X_out = np.concatenate((np.asarray(X_train_scaled), X_extra), axis=0)
    y_out = np.concatenate((np.asarray(y_train), y_extra), axis=0)
    if sample_weight is None:
        return X_out, y_out, None
    sw_out = np.concatenate((
        np.asarray(sample_weight, dtype=np.float32),
        np.ones(int(np.sum(aug_mask)), dtype=np.float32),
    ))
    return X_out, y_out, sw_out


def _load_npz_packets_cached(path):
    """Load NPZ packets through the shared packet-view cache."""
    packets = load_npz_packet_view(Path(path))
    if not packets:
        raise RuntimeError(
            f"{Path(path).name} has no HT20/HT-LTF/64-SC sensing packets after format filtering"
        )
    return packets
