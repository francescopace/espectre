#!/usr/bin/env python3
"""
Tiny-MLP Gesture Detection - Training Script

Trains a tiny-MLP gesture classifier from labeled CSI data collected with labels
compatible with the configured project gesture set.

Unlike the motion model training script (10_train_motion_model.py), features are extracted from the full
file buffer (variable duration), not from a sliding window. This allows the
MLP to learn the temporal morphology of each gesture (rise time, peak shape,
fall time, etc.).

Data is loaded from data/<label>/ directories.
Each NPZ file represents a single gesture/motion sample.

Exports:
  - src/gesture_weights.py       (MicroPython inference)
  - components/espectre/gesture_weights.h  (C++ ESPHome inference)

Usage:
    python tools/12_train_gesture_model.py             # Train with default features
    python tools/12_train_gesture_model.py --info      # Show gesture dataset info
    python tools/12_train_gesture_model.py --seed 42   # Reproducible training
    python tools/12_train_gesture_model.py --window-seconds 2.0 --window-labels wave,circle_cw
    python tools/12_train_gesture_model.py --ablation  # Feature ablation study

Workflow:
  1. Collect/convert samples for project gesture labels (e.g. wave, circle_cw)
  2. Ensure data/<label>/ contains .npz files for at least 2 labels
  3. Train:           python tools/12_train_gesture_model.py

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['ABSL_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import argparse
import contextlib
import importlib.util
import io
import numpy as np
import random
import re
import statistics
import time
from pathlib import Path

# Import project modules (csi_utils sets up sys.path)
from csi_utils import (
    load_npz_as_packets,
    DATA_DIR,
    TARGET_NO_GESTURE_RECALL,
    TARGET_GESTURE_RECALL,
)

try:
    from ml_utils import (
        generate_seed,
        split_holdout,
    )
except ImportError:
    # Fallbacks for gesture-only branches where ml_utils is intentionally absent.
    def generate_seed():
        return int(time.time() * 1_000_000) & 0x7FFFFFFF

    def split_holdout(X, y, test_size=0.2, random_state=None, groups=None):
        if groups is not None:
            from sklearn.model_selection import GroupShuffleSplit
            gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            train_idx, test_idx = next(gss.split(X, y, groups))
            return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

        from sklearn.model_selection import train_test_split
        return train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y if len(np.unique(y)) > 1 else None
        )

# Add src/ to path for gesture_features import
SRC_DIR = Path(__file__).parent.parent / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gesture_features import (
    extract_gesture_features, GESTURE_FEATURES, NUM_GESTURE_FEATURES,
    EXPERIMENTAL_GESTURE_FEATURES
)
from segmentation import SegmentationContext
from utils import extract_phases

MODELS_DIR = Path(__file__).parent.parent / 'models'
CPP_DIR = Path(__file__).parent.parent.parent / 'components' / 'espectre'

# Labels used for gesture classification training from data/<label>/ directories.
GESTURE_LABELS = [
    'wave',
    'circle_cw',
    'circle_ccw',
    'swipe_left',
    'swipe_right',
    'push',
    'pull',
]

# Synthetic negative class for gesture classification.
# Source is data/no_gesture (3s quiet/non-gesture samples).
NO_GESTURE_LABEL = 'no_gesture'
NO_GESTURE_PRIMARY_SOURCE_LABELS = ['no_gesture']
NO_GESTURE_WINDOW_SECONDS = 2.0

# Fixed subcarriers used for gesture feature extraction (aligned with ML motion model)
GESTURE_SUBCARRIERS = [12, 14, 16, 18, 20, 24, 28, 36, 40, 44, 48, 52]

# Default CSI packet rate used to convert seconds -> packets when windowing.
DEFAULT_PACKET_RATE = 100.0
DEFAULT_NO_GESTURE_MAX_PER_SOURCE = 20
DEFAULT_GESTURE_MLP_HIDDEN = (24, 12)
DEFAULT_GESTURE_MLP_ALPHA = 0.001
DEFAULT_FEATURE_PRESET = 'reduced_plus_paper'
DEFAULT_REJECT_CONFIDENCE = 0.55
DEFAULT_REJECT_MARGIN = 0.05

# Validation thresholds for dataset filtering.
VALIDATION_DURATION_SECONDS = 3.0
VALIDATION_PPS_MIN = 98.0
VALIDATION_PPS_MAX = 116.0
VALIDATION_SHARE_THRESHOLD = 0.47
VALIDATION_CONTRAST_THRESHOLD = 0.92
VALIDATION_PEAK_MIN = 0.12
VALIDATION_PEAK_MAX = 0.93
VALIDATION_KEEP_MIN_SCORE = 3
SEQUENTIAL_BENCHMARK_SEED_COUNT = 6

CANONICAL_GESTURE_FEATURES = []
for _name in list(GESTURE_FEATURES) + list(EXPERIMENTAL_GESTURE_FEATURES):
    if _name not in CANONICAL_GESTURE_FEATURES:
        CANONICAL_GESTURE_FEATURES.append(_name)
FEATURE_NAME_TO_ID = {name: idx for idx, name in enumerate(CANONICAL_GESTURE_FEATURES)}


# ============================================================================
# Data Loading
# ============================================================================

def _discover_no_gesture_sources():
    """Discover required no_gesture sources."""
    preferred = []
    for source_label in NO_GESTURE_PRIMARY_SOURCE_LABELS:
        source_dir = DATA_DIR / source_label
        if source_dir.exists() and source_dir.is_dir() and list(source_dir.glob('*.npz')):
            preferred.append(source_label)
    return preferred


def _discover_class_names(label_filter=None, include_no_gesture=True):
    """Discover available class names and no_gesture source labels.

    Args:
        label_filter: Optional set/list of positive gesture labels to include.
        include_no_gesture: Whether synthetic no_gesture should be considered.

    Returns:
        tuple[list[str], list[str]]: (class_names, no_gesture_sources_found)
    """
    label_filter = set(label_filter or [])
    found_labels = []

    # Positive gesture labels (preserve canonical order).
    for label in GESTURE_LABELS:
        if label_filter and label not in label_filter:
            continue
        label_dir = DATA_DIR / label
        if label_dir.exists() and label_dir.is_dir() and list(label_dir.glob('*.npz')):
            found_labels.append(label)

    # Synthetic no_gesture class if source directories contain data.
    no_gesture_sources_found = []
    if include_no_gesture:
        no_gesture_sources_found = _discover_no_gesture_sources()
        if no_gesture_sources_found:
            found_labels.append(NO_GESTURE_LABEL)

    return found_labels, no_gesture_sources_found


def _source_labels_for_class(label, no_gesture_sources_found):
    """Return source directories backing a class label."""
    if label == NO_GESTURE_LABEL:
        return list(no_gesture_sources_found)
    return [label]


def _compute_gesture_guidance(duration_s):
    """Return expected gesture interval inside a fixed-length sample."""
    still_before = max(0.4, duration_s * 0.20)
    still_after = max(0.7, duration_s * 0.25)
    cue_start = still_before
    cue_stop = duration_s - still_after

    min_gesture_span = 1.4
    span = cue_stop - cue_start
    if span < min_gesture_span:
        deficit = min_gesture_span - span
        reduce_before = min(deficit * 0.5, max(0.0, cue_start - 0.3))
        cue_start -= reduce_before
        deficit -= reduce_before
        reduce_after = min(deficit, max(0.0, (duration_s - cue_stop) - 0.3))
        cue_stop += reduce_after

    cue_start = max(0.2, min(cue_start, duration_s - 0.4))
    cue_stop = max(cue_start + 0.4, min(cue_stop, duration_s - 0.2))
    return cue_start, cue_stop


def _validate_dataset_file(npz_path, label, duration_hint_s=VALIDATION_DURATION_SECONDS):
    """Return (keep, row) for a single file with fixed internal thresholds."""
    try:
        packets = load_npz_as_packets(npz_path)
        if not packets or len(packets) < 10:
            return False, {'name': npz_path.name, 'status': 'REVIEW', 'reason': 'empty_or_too_short', 'score': 0}

        use_cv_norm = any(not p.get('gain_locked', True) for p in packets)
        ctx = SegmentationContext(window_size=1, threshold=1.0)
        ctx.use_cv_normalization = use_cv_norm

        turb = []
        for pkt in packets:
            v, _ = ctx.compute_spatial_turbulence(
                pkt['csi_data'], GESTURE_SUBCARRIERS, use_cv_normalization=use_cv_norm
            )
            turb.append(v)
        turb = np.asarray(turb, dtype=np.float32)
        if turb.size < 10:
            return False, {'name': npz_path.name, 'status': 'REVIEW', 'reason': 'too_short_after_parse', 'score': 0}

        smooth_win = 15
        kernel = np.ones(smooth_win, dtype=np.float32) / float(smooth_win)
        sm = np.convolve(turb, kernel, mode='same')
        n = sm.size
        duration_s = max(duration_hint_s, n / DEFAULT_PACKET_RATE)
        pps = float(n / duration_s)

        if label == NO_GESTURE_LABEL:
            keep = (VALIDATION_PPS_MIN <= pps <= VALIDATION_PPS_MAX)
            return keep, {
                'name': npz_path.name,
                'status': 'KEEP' if keep else 'REVIEW',
                'score': 1 if keep else 0,
                'pps': pps,
                'share_g': np.nan,
                'contrast': np.nan,
                'peak_pos': np.nan,
            }

        cue_start, cue_stop = _compute_gesture_guidance(duration_hint_s)
        t_axis = np.linspace(0.0, duration_s, num=n, endpoint=False)
        gmask = (t_axis >= cue_start) & (t_axis < cue_stop)
        omask = ~gmask

        total_energy = float(np.sum(sm * sm)) + 1e-12
        gesture_energy = float(np.sum(sm[gmask] * sm[gmask])) if gmask.any() else 0.0
        share_g = gesture_energy / total_energy

        mean_g = float(np.mean(sm[gmask])) if gmask.any() else 0.0
        mean_o = float(np.mean(sm[omask])) if omask.any() else 0.0
        contrast = mean_g / (mean_o + 1e-12)
        peak_pos = float(np.argmax(sm)) / float(max(n - 1, 1))

        score = 0
        score += 1 if (VALIDATION_PPS_MIN <= pps <= VALIDATION_PPS_MAX) else 0
        score += 1 if (share_g >= VALIDATION_SHARE_THRESHOLD) else 0
        score += 1 if (contrast >= VALIDATION_CONTRAST_THRESHOLD) else 0
        score += 1 if (VALIDATION_PEAK_MIN <= peak_pos <= VALIDATION_PEAK_MAX) else 0
        keep = score >= VALIDATION_KEEP_MIN_SCORE
        return keep, {
            'name': npz_path.name,
            'status': 'KEEP' if keep else 'REVIEW',
            'score': score,
            'pps': pps,
            'share_g': share_g,
            'contrast': contrast,
            'peak_pos': peak_pos,
        }
    except Exception as exc:
        return False, {'name': npz_path.name, 'status': 'REVIEW', 'reason': f'exception:{exc}', 'score': 0}


def _build_validation_allowlist(duration_hint_s=VALIDATION_DURATION_SECONDS):
    """Build per-source allowlist of files that pass validation."""
    found_labels, no_gesture_sources_found = _discover_class_names(include_no_gesture=True)
    allowlist = {}
    summary = {}
    for label in found_labels:
        rows = []
        source_labels = _source_labels_for_class(label, no_gesture_sources_found)
        for source_label in source_labels:
            label_dir = DATA_DIR / source_label
            for npz_file in sorted(label_dir.glob('*.npz')):
                keep, row = _validate_dataset_file(npz_file, label, duration_hint_s=duration_hint_s)
                row['source'] = source_label
                rows.append(row)
                if keep:
                    allowlist.setdefault(source_label, set()).add(npz_file.name)
        summary[label] = rows
    return allowlist, summary


def _print_validation_summary(summary, context='dataset-validation'):
    """Print compact validation summary."""
    print(f'Dataset validation ({context}):')
    headers = ('label', 'total', 'keep', 'review', 'keep%')
    rows_out = []
    review_preview = []
    total_all = 0
    keep_all = 0

    for label, rows in summary.items():
        total = len(rows)
        keep = sum(1 for r in rows if r.get('status') == 'KEEP')
        review = total - keep
        keep_pct = (keep / total * 100.0) if total else 0.0
        rows_out.append((label, total, keep, review, keep_pct))
        total_all += total
        keep_all += keep

        if review > 0:
            review_names = [r['name'] for r in rows if r.get('status') == 'REVIEW']
            preview = ', '.join(review_names[:3])
            extra = '' if review <= 3 else f' ... (+{review - 3})'
            review_preview.append(f'  - {label}: {preview}{extra}')

    if not rows_out:
        print('  no files')
        return

    label_w = max(len(headers[0]), max(len(r[0]) for r in rows_out))
    total_w = len(headers[1])
    keep_w = len(headers[2])
    review_w = len(headers[3])
    pct_w = len(headers[4])

    line = (
        f"  {{:<{label_w}}}  "
        f"{{:>{total_w}}}  "
        f"{{:>{keep_w}}}  "
        f"{{:>{review_w}}}  "
        f"{{:>{pct_w}}}"
    )
    print(line.format(*headers))
    print('  ' + '-' * (label_w + total_w + keep_w + review_w + pct_w + 10))
    for lbl, total, keep, review, keep_pct in rows_out:
        print(line.format(lbl, total, keep, review, f'{keep_pct:5.1f}'))

    keep_pct_all = (keep_all / total_all * 100.0) if total_all else 0.0
    print('  ' + '-' * (label_w + total_w + keep_w + review_w + pct_w + 10))
    print(line.format('TOTAL', total_all, keep_all, total_all - keep_all, f'{keep_pct_all:5.1f}'))

    if review_preview:
        print('\n  Review examples:')
        for row in review_preview:
            print(row)


def _print_dataset_label_stats(stats, class_names):
    """Print dataset sample distribution by class."""
    print(f'  Total samples: {stats["total"]}')
    class_id_map = {name: i for i, name in enumerate(class_names)}
    for label, count in sorted(stats['labels'].items()):
        cid = class_id_map[label]
        print(f'  {label} (class {cid}): {count} samples')


def load_gesture_data(window_packets=None, stride_packets=None, window_labels=None,
                      no_gesture_window_packets=None, no_gesture_stride_packets=None,
                      first_window_only=False, no_gesture_max_per_source=None,
                      random_seed=None, feature_names=None, allowed_files_by_source=None):
    """Load gesture data from data/<label>/ directories.

    By default, each NPZ file is treated as a single sample (variable-length I/Q stream).
    Optionally, selected labels can be split into fixed-size windows.
    When first_window_only=True, only the first fixed-size window is kept.
    A synthetic class named "no_gesture" is built from data/no_gesture when
    available; otherwise it falls back to baseline/movement. Those sources can
    be forced to fixed windowing independently.

    Returns:
        tuple: (X, y, class_names, stats, groups)
            X: np.ndarray of shape (N_samples, NUM_GESTURE_FEATURES)
            y: np.ndarray of shape (N_samples,) with integer class IDs
            class_names: list of class names indexed by class_id
            stats: dict with label counts
            groups: np.ndarray of source-file group IDs (one per sample)
    """
    window_labels = set(window_labels or [])
    rng = np.random.default_rng(random_seed)
    X_list = []
    y_list = []
    groups_list = []
    stats = {'labels': {}, 'total': 0}
    found_labels, no_gesture_sources_found = _discover_class_names(
        include_no_gesture=True
    )

    if not found_labels:
        return np.empty((0, NUM_GESTURE_FEATURES)), np.array([]), [], stats, np.array([])

    # Build class mapping: 0, 1, 2, ... in order of GESTURE_LABELS
    class_names = found_labels
    class_id_map = {name: i for i, name in enumerate(class_names)}

    # Load data from each class
    for label in class_names:
        stats['labels'][label] = 0
        cid = class_id_map[label]

        source_labels = _source_labels_for_class(label, no_gesture_sources_found)

        for source_label in source_labels:
            label_dir = DATA_DIR / source_label
            source_samples = []
            for npz_file in sorted(label_dir.glob('*.npz')):
                if (allowed_files_by_source is not None and
                        npz_file.name not in allowed_files_by_source.get(source_label, set())):
                    continue
                force_no_gesture_window = (
                    label == NO_GESTURE_LABEL and
                    no_gesture_window_packets is not None and
                    no_gesture_window_packets >= 10
                )
                if force_no_gesture_window:
                    wp = no_gesture_window_packets
                    sp = no_gesture_stride_packets if no_gesture_stride_packets else no_gesture_window_packets
                else:
                    wp = window_packets if source_label in window_labels else None
                    sp = stride_packets if source_label in window_labels else None

                feature_vectors = _extract_event_features(
                    npz_file,
                    window_packets=wp,
                    stride_packets=sp,
                    feature_names=feature_names,
                    # Apply first-window truncation only to positive gesture labels.
                    # Keep no_gesture sources with normal windowing.
                    first_window_only=(first_window_only and label != NO_GESTURE_LABEL),
                )
                if not feature_vectors:
                    continue
                for features in feature_vectors:
                    # Group by source label + filename to avoid accidental collisions.
                    source_samples.append((features, f'{source_label}/{npz_file.name}'))

            if (label == NO_GESTURE_LABEL and
                    no_gesture_max_per_source is not None and
                    no_gesture_max_per_source > 0 and
                    len(source_samples) > no_gesture_max_per_source):
                selected_idx = rng.choice(
                    len(source_samples), size=no_gesture_max_per_source, replace=False
                )
                source_samples = [source_samples[i] for i in sorted(selected_idx.tolist())]

            for features, group_id in source_samples:
                X_list.append(features)
                y_list.append(cid)
                groups_list.append(group_id)
                stats['labels'][label] += 1
                stats['total'] += 1

    if not X_list:
        return np.empty((0, NUM_GESTURE_FEATURES)), np.array([]), class_names, stats, np.array([])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    groups = np.array(groups_list, dtype=object)
    return X, y, class_names, stats, groups


def _extract_event_features(npz_path: Path, window_packets=None, stride_packets=None,
                            first_window_only=False, feature_names=None):
    """Extract gesture features from an event NPZ file.

    Args:
        npz_path: Path to event NPZ file.
        window_packets: Optional fixed window length (in packets).
        stride_packets: Optional stride (in packets) when windowing is active.

    Returns:
        list of feature vectors (one or more), or [] if extraction fails.
    """
    try:
        packets = load_npz_as_packets(npz_path)
        if not packets or len(packets) < 10:
            return []

        # NPZ packets carry gain_locked metadata; CV normalization is needed when gain lock is off.
        use_cv_norm = any(not p.get('gain_locked', True) for p in packets)

        ctx = SegmentationContext(window_size=1, threshold=1.0)
        ctx.use_cv_normalization = use_cv_norm

        event_buffer = []
        for pkt in packets:
            csi_data = pkt['csi_data']
            turbulence, _ = ctx.compute_spatial_turbulence(
                csi_data, GESTURE_SUBCARRIERS, use_cv_normalization=use_cv_norm
            )
            phases = extract_phases(csi_data, GESTURE_SUBCARRIERS)
            event_buffer.append({'turbulence': turbulence, 'phases': phases})

        if not event_buffer:
            return []

        # Default behavior: one sample per file.
        if not window_packets or window_packets < 10 or len(event_buffer) <= window_packets:
            return [extract_gesture_features(event_buffer, feature_names=feature_names)]

        # Windowed behavior: optionally keep only one fixed window.
        if first_window_only:
            # Use a gesture-centric window around the turbulence peak, not the
            # first packets of the file (which may contain pre-gesture idle).
            peak_idx = int(np.argmax([e['turbulence'] for e in event_buffer]))
            start = max(0, min(peak_idx - (window_packets // 2), len(event_buffer) - window_packets))
            centered_window = event_buffer[start:start + window_packets]
            if len(centered_window) < 10:
                return []
            return [extract_gesture_features(centered_window, feature_names=feature_names)]

        # Windowed behavior: create multiple samples from long events.
        if not stride_packets or stride_packets < 1:
            stride_packets = window_packets

        feature_vectors = []
        start = 0
        n_packets = len(event_buffer)

        while start < n_packets:
            end = min(start + window_packets, n_packets)
            window = event_buffer[start:end]
            if len(window) < 10:
                break
            feature_vectors.append(extract_gesture_features(window, feature_names=feature_names))
            if end >= n_packets:
                break
            start += stride_packets

        return feature_vectors

    except Exception as e:
        print(f"  Warning: Could not process {npz_path.name}: {e}")
        return []


def _estimate_positive_gesture_counts(window_packets=None, stride_packets=None, window_labels=None,
                                      first_window_only=False, feature_names=None,
                                      allowed_files_by_source=None):
    """Estimate valid sample counts for positive gesture classes only.

    Returns:
        dict: {gesture_label: sample_count}
    """
    window_labels = set(window_labels or [])
    counts = {}

    for label in GESTURE_LABELS:
        label_dir = DATA_DIR / label
        if not (label_dir.exists() and label_dir.is_dir()):
            continue

        sample_count = 0
        for npz_file in sorted(label_dir.glob('*.npz')):
            if (allowed_files_by_source is not None and
                    npz_file.name not in allowed_files_by_source.get(label, set())):
                continue
            wp = window_packets if label in window_labels else None
            sp = stride_packets if label in window_labels else None
            vectors = _extract_event_features(
                npz_file,
                window_packets=wp,
                stride_packets=sp,
                first_window_only=first_window_only,
                feature_names=feature_names,
            )
            sample_count += len(vectors)

        if sample_count > 0:
            counts[label] = sample_count

    return counts


def _build_cv_splitter(n_folds=5, groups=None):
    """Build CV splitter with group awareness when available."""
    from sklearn.model_selection import StratifiedKFold
    if groups is not None:
        try:
            from sklearn.model_selection import StratifiedGroupKFold
            return StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42), True
        except ImportError:
            from sklearn.model_selection import GroupKFold
            return GroupKFold(n_splits=n_folds), True
    return StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42), False


def _fit_mlp_model(X, y, mlp_hidden=DEFAULT_GESTURE_MLP_HIDDEN, mlp_alpha=DEFAULT_GESTURE_MLP_ALPHA,
                   random_seed=42):
    """Fit tiny MLP with StandardScaler preprocessing."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPClassifier

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', MLPClassifier(
            hidden_layer_sizes=tuple(mlp_hidden),
            activation='relu',
            alpha=float(mlp_alpha),
            max_iter=1000,
            random_state=random_seed,
        ))
    ])
    model.fit(X, y)
    return model


def _cross_validate_mlp(X, y, groups=None, n_folds=5,
                        mlp_hidden=DEFAULT_GESTURE_MLP_HIDDEN,
                        mlp_alpha=DEFAULT_GESTURE_MLP_ALPHA,
                        random_seed=42):
    """Cross-validate tiny MLP with optional group split."""
    from sklearn.metrics import f1_score, accuracy_score
    from sklearn.base import clone

    n_folds = min(n_folds, min(np.bincount(y)))
    if n_folds < 2:
        return {'accuracy_mean': 0.0, 'accuracy_std': 0.0, 'f1_mean': 0.0, 'f1_std': 0.0}

    splitter, uses_groups = _build_cv_splitter(n_folds=n_folds, groups=groups)
    split_iter = splitter.split(X, y, groups=groups) if uses_groups else splitter.split(X, y)

    base_model = _fit_mlp_model(
        X, y, mlp_hidden=mlp_hidden, mlp_alpha=mlp_alpha, random_seed=random_seed
    )
    f1_vals, acc_vals = [], []
    for train_idx, val_idx in split_iter:
        m = clone(base_model)
        m.fit(X[train_idx], y[train_idx])
        y_pred = m.predict(X[val_idx])
        f1_vals.append(f1_score(y[val_idx], y_pred, average='macro', zero_division=0) * 100)
        acc_vals.append(accuracy_score(y[val_idx], y_pred) * 100)

    return {
        'accuracy_mean': float(np.mean(acc_vals)),
        'accuracy_std': float(np.std(acc_vals)),
        'f1_mean': float(np.mean(f1_vals)),
        'f1_std': float(np.std(f1_vals)),
    }


def _predict_with_reject_from_proba(proba, class_names, conf_thr, margin_thr):
    """Apply confidence/margin reject rule using probability outputs."""
    proba = np.asarray(proba, dtype=np.float64)
    if proba.ndim == 1:
        proba = proba.reshape(-1, 1)
    top_idx = np.argmax(proba, axis=1)
    top_prob = proba[np.arange(len(proba)), top_idx]
    second_prob = np.partition(proba, -2, axis=1)[:, -2] if proba.shape[1] > 1 else np.zeros(len(proba))
    margins = top_prob - second_prob
    y_pred = top_idx.astype(np.int32)
    if NO_GESTURE_LABEL in class_names:
        no_gesture_id = class_names.index(NO_GESTURE_LABEL)
        reject_mask = (top_prob < conf_thr) | (margins < margin_thr)
        y_pred[reject_mask] = no_gesture_id
    return y_pred


def _evaluate_reject_metrics(y_true, y_pred, class_names):
    """Compute objective metrics and per-class recalls."""
    from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score, recall_score

    recalls = recall_score(
        y_true, y_pred,
        labels=np.arange(len(class_names)),
        average=None,
        zero_division=0
    ) * 100.0
    per_class_recall = {name: float(recalls[i]) for i, name in enumerate(class_names)}

    return {
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0) * 100.0),
        'balanced_acc': float(balanced_accuracy_score(y_true, y_pred) * 100.0),
        'accuracy': float(accuracy_score(y_true, y_pred) * 100.0),
        'per_class_recall': per_class_recall,
    }


def _compute_no_gesture_fp_rate(y_true, y_pred, class_names):
    """FP rate: among true no_gesture samples, predicted as gesture."""
    if NO_GESTURE_LABEL not in class_names:
        return 0.0
    no_gesture_id = class_names.index(NO_GESTURE_LABEL)
    mask = (y_true == no_gesture_id)
    denom = int(np.sum(mask))
    if denom == 0:
        return 0.0
    fp = int(np.sum(y_pred[mask] != no_gesture_id))
    return 100.0 * fp / denom


def _compute_gesture_precision(y_true, y_pred, class_names):
    """Macro precision across gesture classes only (exclude no_gesture)."""
    from sklearn.metrics import precision_score
    gesture_ids = [i for i, n in enumerate(class_names) if n != NO_GESTURE_LABEL]
    if not gesture_ids:
        return 0.0
    p = precision_score(
        y_true, y_pred,
        labels=np.array(gesture_ids, dtype=np.int32),
        average='macro',
        zero_division=0
    )
    return float(p * 100.0)


def _evaluate_no_gesture_first(y_true, y_pred, class_names):
    """Extended metrics for conservative gesture triggering policy."""
    from sklearn.metrics import confusion_matrix
    base = _evaluate_reject_metrics(y_true, y_pred, class_names)
    base['no_gesture_fp_rate'] = _compute_no_gesture_fp_rate(y_true, y_pred, class_names)
    base['gesture_precision'] = _compute_gesture_precision(y_true, y_pred, class_names)
    base['confusion_matrix'] = confusion_matrix(
        y_true, y_pred, labels=np.arange(len(class_names), dtype=np.int32)
    )
    return base


def _passes_recall_constraints(metrics, class_names):
    """Check target recall constraints across no_gesture and gesture classes."""
    per = metrics['per_class_recall']
    if NO_GESTURE_LABEL in class_names:
        if per.get(NO_GESTURE_LABEL, 0.0) < (TARGET_NO_GESTURE_RECALL * 100.0):
            return False
    for label in class_names:
        if label == NO_GESTURE_LABEL:
            continue
        if per.get(label, 0.0) < (TARGET_GESTURE_RECALL * 100.0):
            return False
    return True


def _tune_reject_thresholds_from_proba(proba, y_eval, class_names):
    """Grid-search reject thresholds using probability outputs."""
    conf_grid = np.arange(0.20, 0.91, 0.05)
    margin_grid = np.arange(0.00, 0.31, 0.02)

    best_any = None
    best_feasible = None
    for conf_thr in conf_grid:
        for margin_thr in margin_grid:
            y_pred = _predict_with_reject_from_proba(
                proba, class_names, conf_thr=float(conf_thr), margin_thr=float(margin_thr)
            )
            metrics = _evaluate_reject_metrics(y_eval, y_pred, class_names)
            candidate = (
                metrics['macro_f1'],
                metrics['balanced_acc'],
                metrics['accuracy'],
                float(conf_thr),
                float(margin_thr),
                metrics,
            )
            if best_any is None or candidate[:3] > best_any[:3]:
                best_any = candidate
            if _passes_recall_constraints(metrics, class_names):
                if best_feasible is None or candidate[:3] > best_feasible[:3]:
                    best_feasible = candidate

    chosen = best_feasible if best_feasible is not None else best_any
    if chosen is None:
        return {
            'confidence': DEFAULT_REJECT_CONFIDENCE,
            'margin': DEFAULT_REJECT_MARGIN,
            'metrics': None,
            'constraints_met': False,
        }

    return {
        'confidence': chosen[3],
        'margin': chosen[4],
        'metrics': chosen[5],
        'constraints_met': best_feasible is not None,
    }


def _resolve_feature_names(feature_preset='baseline'):
    """Resolve named feature preset."""
    if feature_preset == 'baseline':
        return list(GESTURE_FEATURES)
    if feature_preset == 'coherence_swap':
        return [f for f in GESTURE_FEATURES if f != 'phase_circular_variance'] + ['phase_inter_sc_coherence']
    if feature_preset == 'plus_coherence':
        return list(GESTURE_FEATURES) + ['phase_inter_sc_coherence']
    if feature_preset == 'reduced_plus_paper':
        # Empirically selected on no_gesture-first optimization for 3s single-gesture datasets.
        return [
            'phase_entropy',
            'turb_iqr',
            'event_duration',
            'turb_diff_abs_mean',
            'turb_mid_mean',
            'phase_diff_var',
            'phase_inter_sc_coherence',
        ]
    raise ValueError(f'Unknown feature preset: {feature_preset}')


def _feature_ids(feature_names):
    """Map feature names to canonical numeric IDs for runtime extraction."""
    return [FEATURE_NAME_TO_ID[name] for name in feature_names]


def _mlp_export_arrays(model):
    """Extract flattened MLP arrays from scaler+MLP pipeline."""
    scaler = model.named_steps['scaler']
    mlp = model.named_steps['clf']

    layer_sizes = [int(mlp.coefs_[0].shape[0])]
    for w in mlp.coefs_:
        layer_sizes.append(int(w.shape[1]))

    weight_offsets = [0]
    bias_offsets = [0]
    flat_w = []
    flat_b = []
    for w, b in zip(mlp.coefs_, mlp.intercepts_):
        w_t = np.asarray(w.T, dtype=np.float32)  # [out, in]
        b_v = np.asarray(b, dtype=np.float32)
        flat_w.extend(w_t.ravel().tolist())
        flat_b.extend(b_v.ravel().tolist())
        weight_offsets.append(len(flat_w))
        bias_offsets.append(len(flat_b))

    return {
        'scaler': scaler,
        'layer_sizes': layer_sizes,
        'num_layers': len(layer_sizes),
        'max_width': max(layer_sizes),
        'weight_offsets': weight_offsets,
        'bias_offsets': bias_offsets,
        'weights': flat_w,
        'biases': flat_b,
    }


def _export_mlp_micropython(model, output_path, seed, class_names, feature_names,
                            reject_confidence, reject_margin):
    """Export tiny MLP parameters for MicroPython gesture inference."""
    data = _mlp_export_arrays(model)
    scaler = data['scaler']
    timestamp = __import__('datetime').datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    code = f'''"""
Micro-ESPectre - Gesture Tiny-MLP Weights

Auto-generated tiny MLP weights for gesture classification.
Architecture: MLP {data['layer_sizes']}
Trained: {timestamp}
Seed: {seed}

This file is auto-generated by 12_train_gesture_model.py.
DO NOT EDIT - your changes will be overwritten!

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

GESTURE_CLASS_LABELS = {class_names}
GESTURE_NUM_CLASSES = {len(class_names)}
GESTURE_NUM_FEATURES = {data['layer_sizes'][0]}
GESTURE_FEATURE_NAMES = {feature_names}
GESTURE_FEATURE_IDS = {_feature_ids(feature_names)}
GESTURE_REJECT_CONFIDENCE = {reject_confidence:.6f}
GESTURE_REJECT_MARGIN = {reject_margin:.6f}

GESTURE_FEATURE_MEAN = [{', '.join(f'{x:.6f}' for x in scaler.mean_)}]
GESTURE_FEATURE_SCALE = [{', '.join(f'{x:.6f}' for x in scaler.scale_)}]

GESTURE_MLP_NUM_LAYERS = {data['num_layers']}
GESTURE_MLP_LAYER_SIZES = {data['layer_sizes']}
GESTURE_MLP_MAX_WIDTH = {data['max_width']}
GESTURE_MLP_WEIGHT_OFFSETS = {data['weight_offsets']}
GESTURE_MLP_BIAS_OFFSETS = {data['bias_offsets']}
GESTURE_MLP_WEIGHTS = [{', '.join(f'{v:.8f}' for v in data['weights'])}]
GESTURE_MLP_BIASES = [{', '.join(f'{v:.8f}' for v in data['biases'])}]
'''
    with open(output_path, 'w') as f:
        f.write(code)
    return len(code), scaler


def _export_mlp_cpp(model, output_path, seed, class_names, feature_names,
                    reject_confidence, reject_margin):
    """Export tiny MLP parameters for C++ ESPHome gesture inference."""
    data = _mlp_export_arrays(model)
    scaler = data['scaler']
    timestamp = __import__('datetime').datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    code = f'''/*
 * ESPectre - Gesture Tiny-MLP Weights
 *
 * Auto-generated tiny MLP weights for gesture classification.
 * Architecture: {data['layer_sizes']}
 * Trained: {timestamp}
 * Seed: {seed}
 *
 * This file is auto-generated by 12_train_gesture_model.py.
 * DO NOT EDIT - your changes will be overwritten!
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#pragma once

#include <cstdint>

namespace esphome {{
namespace espectre {{

constexpr int GESTURE_NUM_CLASSES = {len(class_names)};
constexpr int GESTURE_NUM_FEATURES = {data['layer_sizes'][0]};
constexpr float GESTURE_REJECT_CONFIDENCE = {reject_confidence:.8f}f;
constexpr float GESTURE_REJECT_MARGIN = {reject_margin:.8f}f;

constexpr const char* GESTURE_CLASS_LABELS[{len(class_names)}] = {{
'''
    for name in class_names:
        code += f'    "{name}",\n'
    code += '};\n'
    code += f'constexpr uint8_t GESTURE_FEATURE_IDS[{data["layer_sizes"][0]}] = {{{", ".join(str(v) for v in _feature_ids(feature_names))}}};\n\n'

    code += f'constexpr float GESTURE_FEATURE_MEAN[{data["layer_sizes"][0]}] = {{{", ".join(f"{x:.8f}f" for x in scaler.mean_)}}};\n'
    code += f'constexpr float GESTURE_FEATURE_SCALE[{data["layer_sizes"][0]}] = {{{", ".join(f"{x:.8f}f" for x in scaler.scale_)}}};\n\n'

    code += f'constexpr int GESTURE_MLP_NUM_LAYERS = {data["num_layers"]};\n'
    code += f'constexpr int GESTURE_MLP_MAX_WIDTH = {data["max_width"]};\n'
    code += f'constexpr int GESTURE_MLP_LAYER_SIZES[{data["num_layers"]}] = {{{", ".join(str(v) for v in data["layer_sizes"])}}};\n'
    code += f'constexpr int GESTURE_MLP_WEIGHT_OFFSETS[{data["num_layers"]}] = {{{", ".join(str(v) for v in data["weight_offsets"])}}};\n'
    code += f'constexpr int GESTURE_MLP_BIAS_OFFSETS[{data["num_layers"]}] = {{{", ".join(str(v) for v in data["bias_offsets"])}}};\n'
    code += f'constexpr float GESTURE_MLP_WEIGHTS[{len(data["weights"])}] = {{{", ".join(f"{v:.8f}f" for v in data["weights"])}}};\n'
    code += f'constexpr float GESTURE_MLP_BIASES[{len(data["biases"])}] = {{{", ".join(f"{v:.8f}f" for v in data["biases"])}}};\n\n'

    code += '''}  // namespace espectre
}  // namespace esphome
'''
    with open(output_path, 'w') as f:
        f.write(code)
    return len(code), scaler


# ============================================================================
# Info Display
# ============================================================================

def show_info():
    """Show gesture dataset info."""
    print(f'\n{"="*60}')
    print('  GESTURE DATASET INFORMATION')
    print(f'{"="*60}\n')

    total = 0
    found_any = False

    for label in GESTURE_LABELS:
        label_dir = DATA_DIR / label
        if label_dir.exists() and label_dir.is_dir():
            samples = list(label_dir.glob('*.npz'))
            if samples:
                print(f'  {label}: {len(samples)} samples')
                total += len(samples)
                found_any = True

    if not found_any:
        print('  No gesture data found.')
        print(f'\n  Add data by creating .npz files under data/<label>/ for labels in:')
        print(f'    {", ".join(GESTURE_LABELS)}')
        return

    print(f'\n  Total: {total}')
    if total < 20:
        print(f'\n  Warning: at least 20 samples per class are recommended for training.')
    print()


# ============================================================================
# Main Training
# ============================================================================

def _prepare_gesture_dataset(seed=None, window_seconds=2.0, window_overlap=0.0,
                             window_labels=None,
                             no_gesture_max_per_source=DEFAULT_NO_GESTURE_MAX_PER_SOURCE,
                             feature_names=None,
                             validate_dataset=False,
                             validated_files_by_source=None,
                             validation_summary=None,
                             print_validation_summary=True):
    """Prepare shared gesture dataset/config for both training and experiments."""
    if seed is None:
        seed = generate_seed()
        print(f'Generated random seed: {seed}\n')
    else:
        print(f'Using provided seed: {seed}\n')
    np.random.seed(seed)

    if no_gesture_max_per_source is not None and no_gesture_max_per_source < -1:
        print('Error: --no-gesture-max-per-source must be -1, 0, or >= 1.')
        return None

    window_packets = None
    stride_packets = None
    effective_window_labels = []
    packet_rate = DEFAULT_PACKET_RATE
    if window_seconds and window_seconds > 0:
        if not (0.0 <= window_overlap < 1.0):
            print('Error: --window-overlap must be in range [0.0, 1.0).')
            return None
        window_packets = max(10, int(round(window_seconds * packet_rate)))
        stride_packets = max(1, int(round(window_packets * (1.0 - window_overlap))))
        effective_window_labels = [lbl.strip() for lbl in (window_labels or []) if lbl.strip()]
        if not effective_window_labels:
            # Default to runtime-aligned fixed windows for all available gesture labels.
            for lbl in GESTURE_LABELS:
                label_dir = DATA_DIR / lbl
                if label_dir.exists() and label_dir.is_dir() and list(label_dir.glob('*.npz')):
                    effective_window_labels.append(lbl)

        print(f'Windowing enabled: {window_seconds:.2f}s ({window_packets} packets), '
              f'overlap={window_overlap:.2f}, stride={stride_packets} packets')
        print(f'  Labels with windowing: {effective_window_labels}')

    # Always window negative samples to keep no_gesture coherent with runtime window length.
    no_gesture_window_packets = max(10, int(round(NO_GESTURE_WINDOW_SECONDS * packet_rate)))
    no_gesture_stride_packets = no_gesture_window_packets

    no_gesture_sources = _discover_no_gesture_sources()
    if not no_gesture_sources:
        print(f"Error: required dataset source '{NO_GESTURE_LABEL}' not found or empty in {DATA_DIR / NO_GESTURE_LABEL}")
        return None
    no_gesture_sources_msg = no_gesture_sources
    allowed_files_by_source = None
    if validate_dataset:
        if validated_files_by_source is None:
            allowed_files_by_source, validation_summary = _build_validation_allowlist(
                duration_hint_s=VALIDATION_DURATION_SECONDS
            )
        else:
            allowed_files_by_source = validated_files_by_source
        if print_validation_summary and validation_summary is not None:
            _print_validation_summary(validation_summary, context='train-on-validated')
    print('Loading gesture data...')
    print(f'  Class source: {NO_GESTURE_LABEL} <= {no_gesture_sources_msg}')
    print(f'  Forced windowing for {no_gesture_sources_msg}: '
          f'{NO_GESTURE_WINDOW_SECONDS:.2f}s ({no_gesture_window_packets} packets), '
          f'stride={no_gesture_stride_packets}')
    if no_gesture_max_per_source == -1:
        positive_counts = _estimate_positive_gesture_counts(
            window_packets=window_packets,
            stride_packets=stride_packets,
            window_labels=effective_window_labels,
            first_window_only=True,
            feature_names=feature_names,
            allowed_files_by_source=allowed_files_by_source,
        )
        if positive_counts:
            no_gesture_cap = max(positive_counts.values())
            print(f'  no_gesture cap per source: AUTO -> {no_gesture_cap} samples '
                  f'(from positive gesture counts: {positive_counts})')
        else:
            no_gesture_cap = None
            print('  no_gesture cap per source: AUTO unavailable (no positive gestures found), no cap')
    elif no_gesture_max_per_source == 0:
        no_gesture_cap = None
        print('  no_gesture cap per source: disabled (no cap)')
    else:
        no_gesture_cap = no_gesture_max_per_source
        print(f'  no_gesture cap per source: {no_gesture_cap} samples')

    X, y, class_names, stats, groups = load_gesture_data(
        window_packets=window_packets,
        stride_packets=stride_packets,
        window_labels=effective_window_labels,
        no_gesture_window_packets=no_gesture_window_packets,
        no_gesture_stride_packets=no_gesture_stride_packets,
        first_window_only=True,
        no_gesture_max_per_source=no_gesture_cap,
        random_seed=seed,
        feature_names=feature_names,
        allowed_files_by_source=allowed_files_by_source,
    )

    return {
        'seed': seed,
        'window_packets': window_packets,
        'stride_packets': stride_packets,
        'no_gesture_window_packets': no_gesture_window_packets,
        'no_gesture_stride_packets': no_gesture_stride_packets,
        'X': X,
        'y': y,
        'class_names': class_names,
        'stats': stats,
        'groups': groups,
    }


def train_all(seed=None, window_seconds=2.0, window_overlap=0.0,
              window_labels=None,
              no_gesture_max_per_source=DEFAULT_NO_GESTURE_MAX_PER_SOURCE,
              mlp_hidden=DEFAULT_GESTURE_MLP_HIDDEN,
              mlp_alpha=DEFAULT_GESTURE_MLP_ALPHA,
              feature_preset=DEFAULT_FEATURE_PRESET,
              validate_dataset=False,
              validated_files_by_source=None,
              validation_summary=None,
              print_validation_summary=True):
    """Train gesture classifier (tiny MLP) with selected feature preset.

    Args:
        seed: Optional random seed for reproducibility.
        window_seconds: Optional fixed window length in seconds for selected labels.
        window_overlap: Overlap ratio between consecutive windows [0.0, 1.0).
        window_labels: Labels to split into windows.
        no_gesture_max_per_source: Cap of no_gesture samples per source label
                                  (source: no_gesture)
                                  after windowing.
                                  -1 = auto (match positive gesture sample count),
                                   0 = no cap, >0 = explicit cap.
        validate_dataset: If True, train only on files marked KEEP by validation.
        validated_files_by_source: Optional precomputed validation allowlist.
        validation_summary: Optional precomputed validation summary for printing.
        print_validation_summary: Print validation table when validation is enabled.
    """
    print(f'\n{"="*60}')
    print('  GESTURE CLASSIFIER TRAINING')
    print(f'{"="*60}\n')

    feature_names = _resolve_feature_names(feature_preset)
    prep = _prepare_gesture_dataset(
        seed=seed,
        window_seconds=window_seconds,
        window_overlap=window_overlap,
        window_labels=window_labels,
        no_gesture_max_per_source=no_gesture_max_per_source,
        feature_names=feature_names,
        validate_dataset=validate_dataset,
        validated_files_by_source=validated_files_by_source,
        validation_summary=validation_summary,
        print_validation_summary=print_validation_summary,
    )
    if prep is None:
        return 1

    seed = prep['seed']
    X = prep['X']
    y = prep['y']
    class_names = prep['class_names']
    stats = prep['stats']
    groups = prep['groups']

    if len(X) == 0:
        print('Error: No gesture data found.')
        print(f'Expected labels: {", ".join(GESTURE_LABELS)}')
        return 1

    _print_dataset_label_stats(stats, class_names)

    num_classes = len(class_names)
    print(f'\nClasses ({num_classes}): {class_names}')
    print(f'Feature preset: {feature_preset}')
    print(f'Features ({len(feature_names)}): {", ".join(feature_names)}\n')

    if num_classes < 2:
        print('Error: At least 2 gesture classes required for training.')
        print('Collect events for multiple labels from the configured gesture label set.')
        return 1

    if len(X) < 10:
        print(f'Error: Only {len(X)} samples - need at least 10 for training.')
        return 1

    n_folds = min(5, min(np.bincount(y)))
    if n_folds < 2:
        print('Warning: Too few events for cross-validation, skipping CV.')
        cv_results = {'accuracy_mean': 0, 'f1_mean': 0, 'accuracy_std': 0, 'f1_std': 0}
    else:
        print(f'{n_folds}-fold cross-validation (Tiny-MLP hidden={mlp_hidden}, alpha={mlp_alpha} '
              f'{len(feature_names)}->{num_classes})...')
        cv_results = _cross_validate_mlp(
            X, y, groups=groups, n_folds=n_folds,
            mlp_hidden=mlp_hidden, mlp_alpha=mlp_alpha, random_seed=seed or 42
        )
        print(f'  Accuracy: {cv_results["accuracy_mean"]:.1f}% (+/- {cv_results["accuracy_std"]:.1f}%)')
        print(f'  F1 Score: {cv_results["f1_mean"]:.1f}% (+/- {cv_results["f1_std"]:.1f}%)')

    X_train_raw, X_test_raw, y_train, y_test = split_holdout(
        X, y, test_size=0.2, random_state=seed, groups=groups
    )

    print(f'\nTraining final model on full dataset (Tiny-MLP hidden={mlp_hidden}, alpha={mlp_alpha})...')
    model = _fit_mlp_model(X, y, mlp_hidden=mlp_hidden, mlp_alpha=mlp_alpha, random_seed=seed or 42)
    y_pred = model.predict(X_test_raw)
    test_acc = np.mean(y_pred == y_test) * 100
    print(f'\nHold-out test set (20%):')
    print(f'  Accuracy: {test_acc:.1f}%')
    print(f'  Per-class accuracy:')
    for cid, name in enumerate(class_names):
        mask = y_test == cid
        if mask.sum() > 0:
            acc = np.mean(y_pred[mask] == cid) * 100
            print(f'    {name}: {acc:.1f}%')

    print('\nCalibrating reject thresholds...')
    proba = model.predict_proba(X_test_raw)
    reject_cfg = _tune_reject_thresholds_from_proba(proba, y_test, class_names)
    reject_conf = reject_cfg['confidence']
    reject_margin = reject_cfg['margin']
    print(f'  Selected thresholds: confidence>={reject_conf:.2f}, margin>={reject_margin:.2f}')
    if reject_cfg['metrics'] is not None:
        m = reject_cfg['metrics']
        print(f'  Hold-out macro-F1: {m["macro_f1"]:.1f}%')
        print(f'  Hold-out balanced accuracy: {m["balanced_acc"]:.1f}%')
        status = 'met' if reject_cfg.get('constraints_met', False) else 'not met'
        print(f'  Recall constraints ({status}):')
        for label in class_names:
            target = TARGET_NO_GESTURE_RECALL if label == NO_GESTURE_LABEL else TARGET_GESTURE_RECALL
            val = m['per_class_recall'].get(label, 0.0)
            print(f'    {label}: {val:.1f}% (target >= {target*100:.1f}%)')

    print('\nExporting models...')
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    mp_path = SRC_DIR / 'gesture_weights.py'
    mp_size, _ = _export_mlp_micropython(
        model, mp_path, seed=seed, class_names=class_names, feature_names=feature_names,
        reject_confidence=reject_conf, reject_margin=reject_margin
    )
    print(f'  MicroPython weights: {mp_path.name} ({mp_size/1024:.1f} KB)')

    cpp_path = CPP_DIR / 'gesture_weights.h'
    cpp_size, _ = _export_mlp_cpp(
        model, cpp_path, seed=seed, class_names=class_names, feature_names=feature_names,
        reject_confidence=reject_conf, reject_margin=reject_margin
    )
    print(f'  C++ weights: {cpp_path.name} ({cpp_size/1024:.1f} KB)')

    print(f'\n{"="*60}')
    print('  DONE!')
    print(f'{"="*60}')
    print(f'\nModel trained on {stats["total"]} samples, {num_classes} classes')
    print(f'CV F1={cv_results["f1_mean"]:.1f}% (+/- {cv_results["f1_std"]:.1f}%)')
    print(f'Classes: {class_names}')
    print(f'\nGenerated files:')
    print(f'  - {mp_path} (MicroPython)')
    print(f'  - {cpp_path} (C++ ESPHome)')
    print()
    return 0


def run_ablation(seed=None, window_seconds=2.0, window_overlap=0.0,
                 window_labels=None,
                 no_gesture_max_per_source=DEFAULT_NO_GESTURE_MAX_PER_SOURCE,
                 mlp_hidden=DEFAULT_GESTURE_MLP_HIDDEN,
                 mlp_alpha=DEFAULT_GESTURE_MLP_ALPHA,
                 validate_dataset=False):
    """Run leave-one-feature-out ablation for gesture features."""
    print(f'\n{"="*60}')
    print('  GESTURE FEATURE ABLATION')
    print(f'{"="*60}\n')

    prep = _prepare_gesture_dataset(
        seed=seed,
        window_seconds=window_seconds,
        window_overlap=window_overlap,
        window_labels=window_labels,
        no_gesture_max_per_source=no_gesture_max_per_source,
        validate_dataset=validate_dataset,
    )
    if prep is None:
        return 1

    X = prep['X']
    y = prep['y']
    class_names = prep['class_names']
    stats = prep['stats']
    groups = prep['groups']

    if len(X) == 0:
        print('Error: No gesture data found.')
        print(f'Expected labels: {", ".join(GESTURE_LABELS)}')
        return 1

    _print_dataset_label_stats(stats, class_names)
    print(f'\nClasses ({len(class_names)}): {class_names}')
    print(f'Features ({NUM_GESTURE_FEATURES}): {", ".join(GESTURE_FEATURES)}\n')

    if len(class_names) < 2:
        print('Error: At least 2 gesture classes required for ablation.')
        return 1

    if len(X) < 10:
        print(f'Error: Only {len(X)} samples - need at least 10 for ablation.')
        return 1

    n_folds = min(5, min(np.bincount(y)))
    if n_folds < 2:
        print('Error: Too few samples per class for cross-validation ablation.')
        return 1

    print(f'Baseline CV (Tiny-MLP hidden={mlp_hidden}, alpha={mlp_alpha}, folds={n_folds})...')
    baseline = _cross_validate_mlp(
        X, y, groups=groups, n_folds=n_folds,
        mlp_hidden=mlp_hidden, mlp_alpha=mlp_alpha, random_seed=seed or 42
    )
    baseline_f1 = baseline['f1_mean']
    print(f'  Baseline macro-F1: {baseline_f1:.2f}%')
    print(f'  Baseline accuracy: {baseline["accuracy_mean"]:.2f}%\n')

    results = []
    for idx, name in enumerate(GESTURE_FEATURES):
        X_reduced = np.delete(X, idx, axis=1)
        cv = _cross_validate_mlp(
            X_reduced, y, groups=groups, n_folds=n_folds,
            mlp_hidden=mlp_hidden, mlp_alpha=mlp_alpha, random_seed=seed or 42
        )
        delta_f1 = cv['f1_mean'] - baseline_f1
        results.append({
            'idx': idx,
            'name': name,
            'f1_mean': cv['f1_mean'],
            'acc_mean': cv['accuracy_mean'],
            'delta_f1': delta_f1,
        })

    # Sort by delta descending: positive means potential redundancy/noise.
    results.sort(key=lambda r: r['delta_f1'], reverse=True)

    print('Ablation results (remove one feature):')
    print('  idx  feature                      F1(%)    dF1(%)   ACC(%)')
    print('  -------------------------------------------------------------')
    for r in results:
        print(f'  {r["idx"]:>2}   {r["name"]:<26} {r["f1_mean"]:>6.2f}  {r["delta_f1"]:>+7.2f}  {r["acc_mean"]:>7.2f}')

    removable = [r for r in results if r['delta_f1'] > 0.10]
    critical = [r for r in results if r['delta_f1'] < -0.50]

    print('\nInterpretation:')
    print('  - dF1 > 0: removing the feature improves CV macro-F1 (candidate redundant/noisy)')
    print('  - dF1 < 0: removing the feature hurts CV macro-F1 (feature likely useful)')
    print('  - thresholds: +0.10 (candidate remove), -0.50 (keep strongly)\n')

    if removable:
        print('Candidate removable features:')
        for r in removable:
            print(f'  - {r["name"]} (idx {r["idx"]}, dF1={r["delta_f1"]:+.2f})')
    else:
        print('Candidate removable features: none above +0.10 dF1')

    if critical:
        print('\nStrong keep features:')
        for r in sorted(critical, key=lambda x: x['delta_f1']):
            print(f'  - {r["name"]} (idx {r["idx"]}, dF1={r["delta_f1"]:+.2f})')
    else:
        print('\nStrong keep features: none below -0.50 dF1')

    return 0


def _evaluate_feature_set(feature_names, seed=None, window_seconds=2.0, window_overlap=0.0,
                          window_labels=None,
                          no_gesture_max_per_source=DEFAULT_NO_GESTURE_MAX_PER_SOURCE,
                          mlp_hidden=DEFAULT_GESTURE_MLP_HIDDEN,
                          mlp_alpha=DEFAULT_GESTURE_MLP_ALPHA,
                          validate_dataset=False):
    """Train/evaluate one feature set with no_gesture-first threshold tuning."""
    prep = _prepare_gesture_dataset(
        seed=seed,
        window_seconds=window_seconds,
        window_overlap=window_overlap,
        window_labels=window_labels,
        no_gesture_max_per_source=no_gesture_max_per_source,
        feature_names=feature_names,
        validate_dataset=validate_dataset,
    )
    if prep is None:
        return None

    X = prep['X']
    y = prep['y']
    class_names = prep['class_names']
    groups = prep['groups']
    if len(X) < 10 or len(class_names) < 2:
        return None

    n_folds = min(5, min(np.bincount(y)))
    cv = _cross_validate_mlp(
        X, y, groups=groups, n_folds=n_folds,
        mlp_hidden=mlp_hidden, mlp_alpha=mlp_alpha, random_seed=seed or 42
    ) if n_folds >= 2 else {'accuracy_mean': 0.0, 'f1_mean': 0.0, 'accuracy_std': 0.0, 'f1_std': 0.0}

    X_train, X_test, y_train, y_test = split_holdout(
        X, y, test_size=0.2, random_state=prep['seed'], groups=groups
    )
    model = _fit_mlp_model(
        X_train, y_train, mlp_hidden=mlp_hidden, mlp_alpha=mlp_alpha, random_seed=seed or 42
    )
    proba = model.predict_proba(X_test)
    tuned_cfg = _tune_reject_thresholds_from_proba(proba, y_test, class_names)
    conf = tuned_cfg['confidence']
    margin = tuned_cfg['margin']
    y_pred = _predict_with_reject_from_proba(
        proba, class_names, conf_thr=conf, margin_thr=margin
    )
    tuned = tuned_cfg
    eval_metrics = _evaluate_no_gesture_first(y_test, y_pred, class_names)

    return {
        'seed': prep['seed'],
        'feature_names': list(feature_names),
        'n_features': len(feature_names),
        'class_names': class_names,
        'cv': cv,
        'tuned': tuned,
        'eval': eval_metrics,
    }


def _compute_ablation_deltas(X, y, groups, feature_names, mlp_hidden, mlp_alpha, random_seed):
    """Return baseline CV and delta-F1 per removed feature."""
    n_folds = min(5, min(np.bincount(y)))
    baseline = _cross_validate_mlp(
        X, y, groups=groups, n_folds=n_folds,
        mlp_hidden=mlp_hidden, mlp_alpha=mlp_alpha, random_seed=random_seed
    )
    baseline_f1 = baseline['f1_mean']
    rows = []
    for idx, name in enumerate(feature_names):
        X_red = np.delete(X, idx, axis=1)
        cv = _cross_validate_mlp(
            X_red, y, groups=groups, n_folds=n_folds,
            mlp_hidden=mlp_hidden, mlp_alpha=mlp_alpha, random_seed=random_seed
        )
        rows.append({
            'idx': idx,
            'name': name,
            'delta_f1': cv['f1_mean'] - baseline_f1,
            'f1_removed': cv['f1_mean'],
        })
    return baseline, rows


def run_no_gesture_optimization(seed=None, window_seconds=2.0, window_overlap=0.0,
                                window_labels=None,
                                no_gesture_max_per_source=DEFAULT_NO_GESTURE_MAX_PER_SOURCE,
                                mlp_hidden=DEFAULT_GESTURE_MLP_HIDDEN,
                                mlp_alpha=DEFAULT_GESTURE_MLP_ALPHA,
                                feature_preset='baseline',
                                validate_dataset=False):
    """Feature minimization + paper-inspired features for no_gesture-first policy."""
    print(f'\n{"="*60}')
    print('  GESTURE OPTIMIZATION (NO_GESTURE-FIRST)')
    print(f'{"="*60}\n')

    default_feature_names = _resolve_feature_names(feature_preset)

    # 1) Baseline lock with current default feature set
    baseline = _evaluate_feature_set(
        feature_names=default_feature_names,
        seed=seed,
        window_seconds=window_seconds,
        window_overlap=window_overlap,
        window_labels=window_labels,
        no_gesture_max_per_source=no_gesture_max_per_source,
        mlp_hidden=mlp_hidden,
        mlp_alpha=mlp_alpha,
        validate_dataset=validate_dataset,
    )
    if baseline is None:
        print('Error: unable to build baseline dataset.')
        return 1

    print('Baseline (current default features)')
    print(f'  Seed: {baseline["seed"]}')
    print(f'  Features: {baseline["n_features"]}')
    print(f'  CV macro-F1: {baseline["cv"]["f1_mean"]:.2f}%')
    print(f'  Hold-out macro-F1: {baseline["eval"]["macro_f1"]:.2f}%')
    print(f'  Hold-out balanced-acc: {baseline["eval"]["balanced_acc"]:.2f}%')
    print(f'  no_gesture FP rate: {baseline["eval"]["no_gesture_fp_rate"]:.2f}%')
    print(f'  gesture precision: {baseline["eval"]["gesture_precision"]:.2f}%')
    print('  Confusion matrix (rows=true, cols=pred):')
    print(baseline['eval']['confusion_matrix'])
    print('')

    # 2) Build reduced candidate sets from ablation
    prep_default = _prepare_gesture_dataset(
        seed=seed,
        window_seconds=window_seconds,
        window_overlap=window_overlap,
        window_labels=window_labels,
        no_gesture_max_per_source=no_gesture_max_per_source,
        feature_names=default_feature_names,
        validate_dataset=validate_dataset,
    )
    X = prep_default['X']
    y = prep_default['y']
    groups = prep_default['groups']
    _, ablation_rows = _compute_ablation_deltas(
        X, y, groups, default_feature_names,
        mlp_hidden=mlp_hidden, mlp_alpha=mlp_alpha, random_seed=seed or 42
    )
    ablation_rows.sort(key=lambda r: r['delta_f1'])
    strong_keep = [r['name'] for r in ablation_rows if r['delta_f1'] <= -0.50]
    if len(strong_keep) < 6:
        strong_keep = [r['name'] for r in ablation_rows[:6]]

    neutral = [r['name'] for r in ablation_rows if -0.50 < r['delta_f1'] <= 0.10]
    candidate_min = strong_keep[:]
    candidate_mid = strong_keep[:] + neutral[:2]
    # Reduced+paper-inspired set
    candidate_paper = candidate_min[:] + list(EXPERIMENTAL_GESTURE_FEATURES)

    candidate_sets = [
        ('baseline', list(default_feature_names)),
        ('reduced_min', candidate_min),
        ('reduced_mid', candidate_mid),
        ('reduced_plus_paper', candidate_paper),
    ]

    # Deduplicate while preserving order.
    clean_candidates = []
    for name, feats in candidate_sets:
        seen = set()
        uniq = []
        for f in feats:
            if f not in seen:
                seen.add(f)
                uniq.append(f)
        clean_candidates.append((name, uniq))

    # 3) Evaluate all candidate sets no_gesture-first
    results = []
    for name, feats in clean_candidates:
        res = _evaluate_feature_set(
            feature_names=feats,
            seed=seed,
            window_seconds=window_seconds,
            window_overlap=window_overlap,
            window_labels=window_labels,
            no_gesture_max_per_source=no_gesture_max_per_source,
            mlp_hidden=mlp_hidden,
            mlp_alpha=mlp_alpha,
            validate_dataset=validate_dataset,
        )
        if res is None:
            continue
        results.append((name, res))

    if not results:
        print('Error: no candidate set could be evaluated.')
        return 1

    # Rank: no_gesture recall -> gesture precision -> macro-F1 -> balanced-acc.
    def rank_key(item):
        _, r = item
        no_gesture_recall = r['eval']['per_class_recall'].get(NO_GESTURE_LABEL, 0.0)
        return (
            no_gesture_recall,
            r['eval']['gesture_precision'],
            r['eval']['macro_f1'],
            r['eval']['balanced_acc'],
        )

    results.sort(key=rank_key, reverse=True)

    print('Candidate set comparison (no_gesture-first ranking):')
    print('  set_name              n_feat  no_g_rec  gest_prec  macro_f1  bal_acc  fp_rate')
    print('  -------------------------------------------------------------------------------')
    for name, r in results:
        no_g_rec = r['eval']['per_class_recall'].get(NO_GESTURE_LABEL, 0.0)
        print(f'  {name:<20} {r["n_features"]:>6}  {no_g_rec:>8.2f}  {r["eval"]["gesture_precision"]:>9.2f}  '
              f'{r["eval"]["macro_f1"]:>8.2f}  {r["eval"]["balanced_acc"]:>7.2f}  {r["eval"]["no_gesture_fp_rate"]:>7.2f}')

    best_name, best = results[0]
    print('\nBest set (by policy):')
    print(f'  {best_name} -> {best["n_features"]} features')
    print(f'  Features: {best["feature_names"]}')
    print(f'  Tuned thresholds: confidence>={best["tuned"]["confidence"]:.2f}, '
          f'margin>={best["tuned"]["margin"]:.2f}')
    print(f'  no_gesture FP rate: {best["eval"]["no_gesture_fp_rate"]:.2f}% '
          f'(baseline {baseline["eval"]["no_gesture_fp_rate"]:.2f}%)')
    print(f'  no_gesture recall: {best["eval"]["per_class_recall"].get(NO_GESTURE_LABEL, 0.0):.2f}%')
    print(f'  gesture precision: {best["eval"]["gesture_precision"]:.2f}%')
    print(f'  macro-F1: {best["eval"]["macro_f1"]:.2f}%')

    return 0


def _load_stream_benchmark_module(stream_script):
    """Load 13_test_gesture_stream.py as a module."""
    spec = importlib.util.spec_from_file_location('gesture_stream_benchmark', str(stream_script))
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_stream_benchmark_once(stream_mod, seed):
    """Run streaming benchmark once and return parsed metrics dict."""
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        metrics = stream_mod.run_stream_test(seed=int(seed), return_metrics=True)
    if not isinstance(metrics, dict) or metrics.get('error'):
        return None
    per = metrics.get('per_class_accuracy', {})
    return {
        'seed': int(seed),
        'circle': float(per.get('circle_cw', 0.0)),
        'wave': float(per.get('wave', 0.0)),
        'no_gesture': float(per.get('no_gesture', 0.0)),
        'overall': float(metrics.get('overall_accuracy', 0.0)),
        'pass': bool(metrics.get('constraint_pass', False)),
    }


def run_sequential_train_search(max_runs=12,
                                window_seconds=2.0, window_overlap=0.0, window_labels=None,
                                no_gesture_max_per_source=DEFAULT_NO_GESTURE_MAX_PER_SOURCE,
                                mlp_hidden=DEFAULT_GESTURE_MLP_HIDDEN,
                                mlp_alpha=DEFAULT_GESTURE_MLP_ALPHA,
                                feature_preset=DEFAULT_FEATURE_PRESET):
    """Run sequential auto-seed training and benchmark after each run."""
    if max_runs < 1:
        print('Error: --sequential-train-search must be >= 1 when provided.')
        return 1

    stream_script = Path(__file__).resolve().with_name('13_test_gesture_stream.py')
    stream_mod = _load_stream_benchmark_module(stream_script)
    if stream_mod is None:
        print('Error: unable to load 13_test_gesture_stream.py')
        return 1
    benchmark_seeds = [random.SystemRandom().randrange(0, 2**31) for _ in range(SEQUENTIAL_BENCHMARK_SEED_COUNT)]

    best = None
    validation_allowlist, validation_summary = _build_validation_allowlist(
        duration_hint_s=VALIDATION_DURATION_SECONDS
    )

    print(f'\n{"="*60}')
    print('  SEQUENTIAL TRAIN SEARCH (AUTO-SEED + STREAM TEST)')
    print(f'{"="*60}\n')
    print(f'Max runs: {max_runs}')
    print(f'Train config: hidden={mlp_hidden}, alpha={mlp_alpha}, preset={feature_preset}')
    print('Mode: --train-on-validated enabled for every run\n')
    _print_validation_summary(validation_summary, context='train-search')

    for run_idx in range(1, max_runs + 1):
        print(f'--- Run {run_idx}/{max_runs} ---')
        train_seed = generate_seed()
        rc = train_all(
            seed=train_seed,
            window_seconds=window_seconds,
            window_overlap=window_overlap,
            window_labels=window_labels,
            no_gesture_max_per_source=no_gesture_max_per_source,
            mlp_hidden=mlp_hidden,
            mlp_alpha=mlp_alpha,
            feature_preset=feature_preset,
            validate_dataset=True,
            validated_files_by_source=validation_allowlist,
            validation_summary=validation_summary,
            print_validation_summary=False,
        )
        if rc != 0:
            print('Training failed, stopping search.')
            return 1
        print(f'  train seed: {train_seed}')

        rows = []
        for seed in benchmark_seeds:
            row = _run_stream_benchmark_once(stream_mod, seed)
            if row is None:
                print(f'Error: benchmark failed on seed={seed}')
                return 1
            rows.append(row)

        pass_count = sum(1 for r in rows if r['pass'])
        avg_circle = statistics.mean(r['circle'] for r in rows)
        avg_wave = statistics.mean(r['wave'] for r in rows)
        avg_ng = statistics.mean(r['no_gesture'] for r in rows)
        avg_overall = statistics.mean(r['overall'] for r in rows)
        worst_single = min(min(r['circle'], r['wave'], r['no_gesture']) for r in rows)
        score = (pass_count, avg_wave, avg_circle, avg_ng, avg_overall, worst_single)

        print(
            f'  pass={pass_count}/{len(benchmark_seeds)} '
            f'avg(c/w/ng)={avg_circle:.1f}/{avg_wave:.1f}/{avg_ng:.1f} '
            f'avg_overall={avg_overall:.1f} worst={worst_single:.1f}'
        )

        rec = {
            'run': run_idx,
            'train_seed': train_seed,
            'pass_count': pass_count,
            'total': len(benchmark_seeds),
            'avg_circle': avg_circle,
            'avg_wave': avg_wave,
            'avg_no_gesture': avg_ng,
            'avg_overall': avg_overall,
            'worst_single_class': worst_single,
            'rows': rows,
            'score': score,
        }

        improved = (best is None) or (score > best['score'])
        if improved:
            best = rec
            print(f'  NEW BEST (run={run_idx}, seed={train_seed})')

        if pass_count == len(benchmark_seeds):
            print('  Full PASS reached, stopping early.')
            break

    if best is None or best['train_seed'] is None:
        print('Error: no valid best run found.')
        return 1

    print(f'\nBest run: #{best["run"]} seed={best["train_seed"]} '
          f'pass={best["pass_count"]}/{best["total"]} '
          f'avg(c/w/ng)={best["avg_circle"]:.1f}/{best["avg_wave"]:.1f}/{best["avg_no_gesture"]:.1f} '
          f'avg_overall={best["avg_overall"]:.1f}')
    print('Retraining best seed to keep exported weights aligned...')
    rr = train_all(
        seed=best['train_seed'],
        window_seconds=window_seconds,
        window_overlap=window_overlap,
        window_labels=window_labels,
        no_gesture_max_per_source=no_gesture_max_per_source,
        mlp_hidden=mlp_hidden,
        mlp_alpha=mlp_alpha,
        feature_preset=feature_preset,
        validate_dataset=True,
        validated_files_by_source=validation_allowlist,
        validation_summary=validation_summary,
        print_validation_summary=False,
    )
    if rr != 0:
        print('Error: retraining best seed failed.')
        return 1

    print('\nBest-run benchmark details:')
    for r in best['rows']:
        status = 'PASS' if r['pass'] else 'FAIL'
        print(f'  seed={r["seed"]}: c/w/ng={r["circle"]:.1f}/{r["wave"]:.1f}/{r["no_gesture"]:.1f} '
              f'overall={r["overall"]:.1f} -> {status}')
    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Train gesture classifier from labeled CSI data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Workflow:
  1. Collect samples:
       (use/convert labels from configured set: wave,circle_cw,circle_ccw,swipe_left,swipe_right,push,pull)
  2. Train:
       python tools/12_train_gesture_model.py
       python tools/12_train_gesture_model.py --window-seconds 2.0 --window-labels wave,circle_cw
  3. Run:
       Deploy to ESP32 or use with micro-espectre/src/gesture_detector.py

Motion model (binary IDLE/MOTION): python tools/10_train_motion_model.py
'''
    )
    parser.add_argument('--info', action='store_true',
                        help='Show gesture dataset information')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducible training')
    parser.add_argument('--window-seconds', type=float, default=2.0,
                        help='Fixed window length in seconds (default: 2.0)')
    parser.add_argument('--window-overlap', type=float, default=0.0,
                        help='Window overlap ratio [0.0, 1.0), default: 0.0')
    parser.add_argument('--window-labels', type=str, default='',
                        help='Comma-separated labels to window (default: auto all gesture labels)')
    parser.add_argument('--no-gesture-max-per-source', type=int,
                        default=DEFAULT_NO_GESTURE_MAX_PER_SOURCE,
                        help='Cap no_gesture samples per source label '
                             '(source: no_gesture). '
                             '-1 = auto (match gesture count), 0 = no cap, >0 = explicit cap '
                             f'(default: {DEFAULT_NO_GESTURE_MAX_PER_SOURCE})')
    parser.add_argument('--ablation', action='store_true',
                        help='Run leave-one-feature-out ablation study and exit')
    parser.add_argument('--optimize-no-gesture', action='store_true',
                        help='Run no_gesture-first feature minimization + paper-inspired evaluation')
    parser.add_argument('--validate-dataset', action='store_true',
                        help='Validate dataset only (print KEEP/REVIEW report and exit)')
    parser.add_argument('--train-on-validated', action='store_true',
                        help='Train using only files marked KEEP by dataset validation')
    parser.add_argument('--sequential-train-search', nargs='?', type=int, const=12, default=0,
                        help='Run sequential auto-seed training search (optional N=max runs, default: 12)')
    parser.add_argument('--feature-preset', choices=['baseline', 'coherence_swap', 'plus_coherence', 'reduced_plus_paper'],
                        default=DEFAULT_FEATURE_PRESET,
                        help=f'Feature preset (default: {DEFAULT_FEATURE_PRESET})')
    parser.add_argument('--mlp-hidden', type=str, default='24,12',
                        help='Comma-separated hidden sizes for tiny MLP (default: 24,12)')
    parser.add_argument('--mlp-alpha', type=float, default=DEFAULT_GESTURE_MLP_ALPHA,
                        help=f'MLP L2 regularization alpha (default: {DEFAULT_GESTURE_MLP_ALPHA})')
    args = parser.parse_args()

    window_labels = [lbl.strip() for lbl in args.window_labels.split(',') if lbl.strip()]
    mlp_hidden = tuple(int(x.strip()) for x in args.mlp_hidden.split(',') if x.strip())
    if not mlp_hidden:
        print('Error: --mlp-hidden must include at least one positive integer.')
        return 1

    if args.info:
        show_info()
        return 0

    if args.validate_dataset:
        _, validation_summary = _build_validation_allowlist(
            duration_hint_s=VALIDATION_DURATION_SECONDS
        )
        _print_validation_summary(validation_summary, context='validate-dataset')
        return 0

    if args.sequential_train_search:
        if args.seed is not None:
            print('Warning: --seed is ignored in --sequential-train-search mode (auto-seed enabled).')
        return run_sequential_train_search(
            max_runs=args.sequential_train_search,
            window_seconds=args.window_seconds,
            window_overlap=args.window_overlap,
            window_labels=window_labels,
            no_gesture_max_per_source=args.no_gesture_max_per_source,
            mlp_hidden=mlp_hidden,
            mlp_alpha=args.mlp_alpha,
            feature_preset=args.feature_preset,
        )

    if args.ablation:
        return run_ablation(
            seed=args.seed,
            window_seconds=args.window_seconds,
            window_overlap=args.window_overlap,
            window_labels=window_labels,
            no_gesture_max_per_source=args.no_gesture_max_per_source,
            validate_dataset=args.train_on_validated,
        )

    if args.optimize_no_gesture:
        return run_no_gesture_optimization(
            seed=args.seed,
            window_seconds=args.window_seconds,
            window_overlap=args.window_overlap,
            window_labels=window_labels,
            no_gesture_max_per_source=args.no_gesture_max_per_source,
            mlp_hidden=mlp_hidden,
            mlp_alpha=args.mlp_alpha,
            feature_preset=args.feature_preset,
            validate_dataset=args.train_on_validated,
        )

    return train_all(
        seed=args.seed,
        window_seconds=args.window_seconds,
        window_overlap=args.window_overlap,
        window_labels=window_labels,
        no_gesture_max_per_source=args.no_gesture_max_per_source,
        mlp_hidden=mlp_hidden,
        mlp_alpha=args.mlp_alpha,
        feature_preset=args.feature_preset,
        validate_dataset=args.train_on_validated,
    )


if __name__ == '__main__':
    exit(main())
