#!/usr/bin/env python3
"""
ML Gesture Detection - Training Script

Trains a gesture classifier from labeled CSI data collected with labels
compatible with the configured project gesture set.

Unlike the motion model training script (10_train_motion_model.py), features are extracted from the full
file buffer (variable duration), not from a sliding window. This allows the
model to learn the temporal morphology of each gesture (rise time, peak shape,
fall time, etc.).

Data is loaded from data/<label>/ directories.
Each NPZ file represents a single gesture/motion sample.

Exports:
  - src/gesture_weights.py       (MicroPython inference)
  - components/espectre/gesture_weights.h  (C++ ESPHome inference)
  - models/gesture_scaler.npz    (normalization parameters)
  - models/gesture_test_data.npz (test data for cross-validation)

Usage:
    python tools/11_train_gesture_model.py             # Train with default features
    python tools/11_train_gesture_model.py --info      # Show gesture dataset info
    python tools/11_train_gesture_model.py --experiment
    python tools/11_train_gesture_model.py --feature-ablation
    python tools/11_train_gesture_model.py --feature-ablation --ablation-variants all,no_phase,phase_only
    python tools/11_train_gesture_model.py --seed 42   # Reproducible training
    python tools/11_train_gesture_model.py --window-seconds 2.0 --window-labels wave,circle_cw

Workflow:
  1. Collect/convert samples for project gesture labels (e.g. wave, circle_cw)
  2. Ensure data/<label>/ contains .npz files for at least 2 labels
  3. Train:           python tools/11_train_gesture_model.py

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['ABSL_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import argparse
import numpy as np
import time
from pathlib import Path

# Import project modules (csi_utils sets up sys.path)
from csi_utils import load_npz_as_packets, DATA_DIR

from ml_utils import (
    suppress_stderr,
    generate_seed,
    setup_tf_logging,
    experiment_architectures,
    split_holdout,
)

# Add src/ to path for gesture_features import
SRC_DIR = Path(__file__).parent.parent / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gesture_features import (
    extract_gesture_features, GESTURE_FEATURES, NUM_GESTURE_FEATURES
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
# Data is sourced from baseline/ and movement/ directories.
NO_GESTURE_LABEL = 'no_gesture'
NO_GESTURE_SOURCE_LABELS = ['baseline', 'movement']
NO_GESTURE_WINDOW_SECONDS = 2.0

# Fixed subcarriers used for gesture feature extraction
GESTURE_SUBCARRIERS = [11, 14, 17, 21, 24, 28, 31, 35, 39, 42, 46, 49]

# Default CSI packet rate used to convert seconds -> packets when windowing.
DEFAULT_PACKET_RATE = 100.0

# Feature groups used for ablation analyses.
MORPH_FEATURES = [f for f in GESTURE_FEATURES if not f.startswith('phase_')]
PHASE_FEATURES = [f for f in GESTURE_FEATURES if f.startswith('phase_')]


# ============================================================================
# Data Loading
# ============================================================================

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
        for source_label in NO_GESTURE_SOURCE_LABELS:
            source_dir = DATA_DIR / source_label
            if source_dir.exists() and source_dir.is_dir() and list(source_dir.glob('*.npz')):
                no_gesture_sources_found.append(source_label)
        if no_gesture_sources_found:
            found_labels.append(NO_GESTURE_LABEL)

    return found_labels, no_gesture_sources_found


def _source_labels_for_class(label, no_gesture_sources_found):
    """Return source directories backing a class label."""
    if label == NO_GESTURE_LABEL:
        return list(no_gesture_sources_found)
    return [label]


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
                      random_seed=None):
    """Load gesture data from data/<label>/ directories.

    By default, each NPZ file is treated as a single sample (variable-length I/Q stream).
    Optionally, selected labels can be split into fixed-size windows.
    When first_window_only=True, only the first fixed-size window is kept.
    A synthetic class named "no_gesture" is built by aggregating baseline/movement
    directories. Those sources can be forced to fixed windowing independently.

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
                    # Apply first-window truncation only to positive gesture labels.
                    # Keep no_gesture (baseline/movement) with normal windowing.
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
                            first_window_only=False):
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

        use_cv_norm = any(p.get('use_cv_normalization', False) for p in packets)

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
            return [extract_gesture_features(event_buffer)]

        # Windowed behavior: optionally keep only the first fixed window.
        if first_window_only:
            first_window = event_buffer[:window_packets]
            if len(first_window) < 10:
                return []
            return [extract_gesture_features(first_window)]

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
            feature_vectors.append(extract_gesture_features(window))
            if end >= n_packets:
                break
            start += stride_packets

        return feature_vectors

    except Exception as e:
        print(f"  Warning: Could not process {npz_path.name}: {e}")
        return []


def _estimate_positive_gesture_counts(window_packets=None, stride_packets=None, window_labels=None,
                                      first_window_only=False):
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
            wp = window_packets if label in window_labels else None
            sp = stride_packets if label in window_labels else None
            vectors = _extract_event_features(
                npz_file,
                window_packets=wp,
                stride_packets=sp,
                first_window_only=first_window_only,
            )
            sample_count += len(vectors)

        if sample_count > 0:
            counts[label] = sample_count

    return counts


def _extract_turbulence_windows(npz_path: Path, window_packets, stride_packets):
    """Extract fixed-length turbulence windows from an event NPZ file.

    Args:
        npz_path: Path to event NPZ file.
        window_packets: Fixed window length in packets (>0).
        stride_packets: Sliding stride in packets (>0).

    Returns:
        list[np.ndarray]: Windows shaped (window_packets, 1), dtype float32.
    """
    try:
        packets = load_npz_as_packets(npz_path)
        if not packets or len(packets) < max(10, window_packets):
            return []

        use_cv_norm = any(p.get('use_cv_normalization', False) for p in packets)
        ctx = SegmentationContext(window_size=1, threshold=1.0)
        ctx.use_cv_normalization = use_cv_norm

        turbulence = []
        for pkt in packets:
            csi_data = pkt['csi_data']
            turb, _ = ctx.compute_spatial_turbulence(
                csi_data, GESTURE_SUBCARRIERS, use_cv_normalization=use_cv_norm
            )
            turbulence.append(float(turb))

        if len(turbulence) < window_packets:
            return []

        if not stride_packets or stride_packets < 1:
            stride_packets = window_packets

        seq = np.array(turbulence, dtype=np.float32)
        windows = []
        start = 0
        while start + window_packets <= len(seq):
            w = seq[start:start + window_packets].reshape(window_packets, 1)
            windows.append(w)
            start += stride_packets

        return windows
    except Exception as e:
        print(f"  Warning: Could not process {npz_path.name}: {e}")
        return []


def load_gesture_temporal_data(window_packets, stride_packets, window_labels=None,
                               no_gesture_window_packets=None, no_gesture_stride_packets=None):
    """Load fixed-length temporal windows for sequence models.

    Args:
        window_packets: Fixed window length in packets.
        stride_packets: Sliding stride in packets.
        window_labels: Labels to include.
        no_gesture_window_packets: Optional window size for synthetic no_gesture class.
        no_gesture_stride_packets: Optional stride for synthetic no_gesture class.

    Returns:
        tuple: (X_seq, y, class_names, stats, groups)
            X_seq: np.ndarray (N, window_packets, 1)
            y: np.ndarray (N,)
            class_names: list[str]
            stats: dict
            groups: np.ndarray group IDs (source file names)
    """
    if not window_packets or window_packets < 10:
        return np.empty((0, 0, 1), dtype=np.float32), np.array([]), [], {'labels': {}, 'total': 0}, np.array([])

    label_filter = set(window_labels or [])
    X_list, y_list, groups_list = [], [], []
    stats = {'labels': {}, 'total': 0}
    no_gesture_requested = (not label_filter) or (NO_GESTURE_LABEL in label_filter)
    found_labels, no_gesture_sources_found = _discover_class_names(
        label_filter=label_filter,
        include_no_gesture=no_gesture_requested,
    )

    if not found_labels:
        return np.empty((0, window_packets, 1), dtype=np.float32), np.array([]), [], stats, np.array([])

    class_names = found_labels
    class_id_map = {name: i for i, name in enumerate(class_names)}

    for label in class_names:
        stats['labels'][label] = 0
        cid = class_id_map[label]

        source_labels = _source_labels_for_class(label, no_gesture_sources_found)

        for source_label in source_labels:
            label_dir = DATA_DIR / source_label
            for npz_file in sorted(label_dir.glob('*.npz')):
                if label == NO_GESTURE_LABEL and no_gesture_window_packets and no_gesture_window_packets >= 10:
                    wp = no_gesture_window_packets
                    sp = no_gesture_stride_packets if no_gesture_stride_packets else no_gesture_window_packets
                else:
                    wp = window_packets
                    sp = stride_packets

                windows = _extract_turbulence_windows(npz_file, wp, sp)
                if not windows:
                    continue
                for w in windows:
                    X_list.append(w)
                    y_list.append(cid)
                    groups_list.append(f'{source_label}/{npz_file.name}')
                    stats['labels'][label] += 1
                    stats['total'] += 1

    if not X_list:
        return np.empty((0, window_packets, 1), dtype=np.float32), np.array([]), class_names, stats, np.array([])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    groups = np.array(groups_list, dtype=object)
    return X, y, class_names, stats, groups


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


def _run_classical_experiments(X, y, class_names, groups=None):
    """Evaluate classical ML models on handcrafted features."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score, accuracy_score
    from sklearn.base import clone

    print("\n" + "=" * 60)
    print("       GESTURE CLASSICAL EXPERIMENT")
    print("=" * 60 + "\n")

    models = {
        'LogReg': Pipeline([
            ('scaler', StandardScaler()),
            # Keep constructor compatible across older/newer sklearn versions.
            ('clf', LogisticRegression(max_iter=2000))
        ]),
        'SVM-RBF': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(kernel='rbf', C=2.0, gamma='scale'))
        ]),
        'RandomForest': RandomForestClassifier(
            n_estimators=300, max_depth=None, random_state=42, n_jobs=-1
        ),
    }

    n_folds = min(5, min(np.bincount(y)))
    if n_folds < 2:
        print("Not enough samples per class for classical CV.")
        return []

    splitter, uses_groups = _build_cv_splitter(n_folds=n_folds, groups=groups)
    split_iter = splitter.split(X, y, groups=groups) if uses_groups else splitter.split(X, y)

    results = []
    for name, model in models.items():
        f1_vals, acc_vals = [], []
        for train_idx, val_idx in split_iter:
            m = clone(model)
            m.fit(X[train_idx], y[train_idx])
            y_pred = m.predict(X[val_idx])
            f1_vals.append(f1_score(y[val_idx], y_pred, average='macro', zero_division=0) * 100)
            acc_vals.append(accuracy_score(y[val_idx], y_pred) * 100)

        # Rebuild iterator for next model
        split_iter = splitter.split(X, y, groups=groups) if uses_groups else splitter.split(X, y)

        # Micro-benchmark inference
        m = clone(model)
        m.fit(X, y)
        sample = X[:1]
        for _ in range(20):
            m.predict(sample)
        n_bench = 1000
        t0 = time.perf_counter()
        for _ in range(n_bench):
            m.predict(sample)
        infer_us = (time.perf_counter() - t0) / n_bench * 1e6

        result = {
            'name': name,
            'f1_mean': float(np.mean(f1_vals)),
            'f1_std': float(np.std(f1_vals)),
            'accuracy_mean': float(np.mean(acc_vals)),
            'inference_us': float(infer_us),
        }
        results.append(result)
        print(f"{name:<14} F1={result['f1_mean']:.1f}% (+/- {result['f1_std']:.1f}%) "
              f"Acc={result['accuracy_mean']:.1f}% Infer={result['inference_us']:.1f}us")

    return results


def _build_tiny_temporal_cnn(seq_len, num_classes):
    """Build a tiny 1D CNN for temporal turbulence windows."""
    import tensorflow as tf
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(seq_len, 1)),
        tf.keras.layers.Conv1D(16, 5, padding='same', activation='relu'),
        tf.keras.layers.SeparableConv1D(24, 5, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.SeparableConv1D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax'),
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def _run_temporal_experiment(X_seq, y, class_names, groups=None):
    """Evaluate tiny temporal CNN on sequence windows."""
    if len(X_seq) == 0:
        print("Temporal experiment skipped: no temporal windows available.")
        return None

    try:
        with suppress_stderr():
            import tensorflow as tf
            setup_tf_logging()
    except ImportError as e:
        print(f"Temporal experiment skipped (missing dependency): {e}")
        return None

    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score, accuracy_score

    print("\n" + "=" * 60)
    print("       GESTURE TEMPORAL EXPERIMENT")
    print("=" * 60 + "\n")

    n_folds = min(5, min(np.bincount(y)))
    if n_folds < 2:
        print("Not enough samples per class for temporal CV.")
        return None

    splitter, uses_groups = _build_cv_splitter(n_folds=n_folds, groups=groups)
    split_iter = splitter.split(X_seq, y, groups=groups) if uses_groups else splitter.split(X_seq, y)

    f1_vals, acc_vals = [], []
    seq_len = X_seq.shape[1]

    for train_idx, val_idx in split_iter:
        X_train = X_seq[train_idx]
        X_val = X_seq[val_idx]

        # Fold-wise normalization over flattened temporal input.
        scaler = StandardScaler()
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_val_flat = X_val.reshape(X_val.shape[0], -1)
        X_train_scaled = scaler.fit_transform(X_train_flat).reshape(X_train.shape[0], seq_len, 1)
        X_val_scaled = scaler.transform(X_val_flat).reshape(X_val.shape[0], seq_len, 1)

        with suppress_stderr():
            model = _build_tiny_temporal_cnn(seq_len=seq_len, num_classes=len(class_names))
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=6, restore_best_weights=True, min_delta=1e-4
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6
                ),
            ]
            model.fit(
                X_train_scaled, y[train_idx],
                epochs=30,
                batch_size=32,
                validation_split=0.15,
                callbacks=callbacks,
                verbose=0,
            )
            probs = model.predict(X_val_scaled, verbose=0)

        y_pred = np.argmax(probs, axis=1)
        f1_vals.append(f1_score(y[val_idx], y_pred, average='macro', zero_division=0) * 100)
        acc_vals.append(accuracy_score(y[val_idx], y_pred) * 100)

    # Benchmark on full dataset fit.
    scaler = StandardScaler()
    X_flat = X_seq.reshape(X_seq.shape[0], -1)
    X_scaled = scaler.fit_transform(X_flat).reshape(X_seq.shape[0], seq_len, 1)
    with suppress_stderr():
        model = _build_tiny_temporal_cnn(seq_len=seq_len, num_classes=len(class_names))
        model.fit(X_scaled, y, epochs=8, batch_size=32, verbose=0)
        sample = X_scaled[:1]
        for _ in range(20):
            model.predict(sample, verbose=0)
        n_bench = 500
        t0 = time.perf_counter()
        for _ in range(n_bench):
            model.predict(sample, verbose=0)
        infer_us = (time.perf_counter() - t0) / n_bench * 1e6

    result = {
        'name': f'TinyCNN-1D ({seq_len}x1)',
        'f1_mean': float(np.mean(f1_vals)),
        'f1_std': float(np.std(f1_vals)),
        'accuracy_mean': float(np.mean(acc_vals)),
        'inference_us': float(infer_us),
    }
    print(f"{result['name']:<24} F1={result['f1_mean']:.1f}% (+/- {result['f1_std']:.1f}%) "
          f"Acc={result['accuracy_mean']:.1f}% Infer={result['inference_us']:.1f}us")
    return result


def _print_experiment_summary(mlp_best, classical_results, temporal_result):
    """Print unified summary across all experiment families."""
    print("\n" + "=" * 78)
    print("                    ALL EXPERIMENTS SUMMARY")
    print("=" * 78 + "\n")
    print(f"{'Family':<12} {'Model':<24} {'F1 (CV)':>14} {'Accuracy':>10} {'Infer (us)':>12}")
    print("-" * 78)

    rows = []
    if mlp_best is not None:
        rows.append({
            'family': 'MLP',
            'name': mlp_best['name'],
            'f1_mean': mlp_best['f1_mean'],
            'f1_std': mlp_best['f1_std'],
            'acc': mlp_best['accuracy_mean'],
            'infer': mlp_best['inference_us'],
        })
    for r in classical_results or []:
        rows.append({
            'family': 'Classical',
            'name': r['name'],
            'f1_mean': r['f1_mean'],
            'f1_std': r['f1_std'],
            'acc': r['accuracy_mean'],
            'infer': r['inference_us'],
        })
    if temporal_result is not None:
        rows.append({
            'family': 'Temporal',
            'name': temporal_result['name'],
            'f1_mean': temporal_result['f1_mean'],
            'f1_std': temporal_result['f1_std'],
            'acc': temporal_result['accuracy_mean'],
            'infer': temporal_result['inference_us'],
        })

    if not rows:
        print("No experiment results.")
        return

    best = max(rows, key=lambda r: (r['f1_mean'], -r['infer']))
    for r in rows:
        marker = " **" if r is best else "   "
        print(f"{marker}{r['family']:<9} {r['name']:<24} "
              f"{r['f1_mean']:>6.1f}+/-{r['f1_std']:<5.1f} "
              f"{r['acc']:>9.1f}% {r['infer']:>11.1f}")

    print("-" * 78)
    print(f"\nBest overall: {best['family']} / {best['name']} (F1={best['f1_mean']:.1f}%)\n")


def _fit_logreg_model(X, y):
    """Fit standardized multinomial Logistic Regression for gestures."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=3000)),
    ])
    model.fit(X, y)
    return model


def _logreg_to_multiclass_logits(model, num_features, num_classes):
    """Return (W, b) logits matrix compatible with softmax inference.

    W shape: (num_features, num_classes)
    b shape: (num_classes,)
    """
    clf = model.named_steps['clf']
    coef = np.asarray(clf.coef_, dtype=np.float32)
    intercept = np.asarray(clf.intercept_, dtype=np.float32)

    # sklearn binary logistic stores a single hyperplane (class 1).
    # Convert to two-logit representation equivalent to softmax:
    #   logit_0 = 0, logit_1 = z
    if coef.shape[0] == 1 and num_classes == 2:
        W = np.zeros((num_features, 2), dtype=np.float32)
        b = np.zeros((2,), dtype=np.float32)
        W[:, 1] = coef[0]
        b[1] = intercept[0]
        return W, b

    # Multiclass case (coef rows = classes)
    if coef.shape[0] != num_classes:
        raise ValueError(f"Unexpected coef shape {coef.shape} for {num_classes} classes")
    W = coef.T.copy()  # features x classes
    b = intercept.copy()
    return W, b


def _cross_validate_logreg(X, y, groups=None, n_folds=5):
    """Cross-validate logistic regression with optional group split."""
    from sklearn.metrics import f1_score, accuracy_score
    from sklearn.base import clone

    n_folds = min(n_folds, min(np.bincount(y)))
    if n_folds < 2:
        return {'accuracy_mean': 0.0, 'accuracy_std': 0.0, 'f1_mean': 0.0, 'f1_std': 0.0}

    splitter, uses_groups = _build_cv_splitter(n_folds=n_folds, groups=groups)
    split_iter = splitter.split(X, y, groups=groups) if uses_groups else splitter.split(X, y)

    base_model = _fit_logreg_model(X, y)
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


def _slice_features(X, selected_features):
    """Select feature columns by name from the full gesture feature matrix."""
    index_map = {name: i for i, name in enumerate(GESTURE_FEATURES)}
    selected_idx = [index_map[name] for name in selected_features]
    return X[:, selected_idx], selected_idx


def _feature_importance_from_logreg(model):
    """Return standardized feature importance from trained logistic regression."""
    clf = model.named_steps['clf']
    coef = np.asarray(clf.coef_, dtype=np.float32)
    if coef.ndim == 1:
        coef = coef.reshape(1, -1)
    return np.mean(np.abs(coef), axis=0)


def _print_feature_importance(selected_features, model, top_n=12):
    """Print sorted feature importance based on mean |coef| across classes."""
    importances = _feature_importance_from_logreg(model)
    pairs = list(zip(selected_features, importances.tolist()))
    pairs.sort(key=lambda x: x[1], reverse=True)

    print("\nFeature importance (LogReg mean |coef|):")
    for rank, (name, score) in enumerate(pairs[:max(1, top_n)], start=1):
        print(f"  {rank:>2}. {name:<24} {score:.4f}")


def _build_feature_ablation_variants(variant_names):
    """Resolve ablation variant names to concrete feature lists."""
    presets = {
        'all': list(GESTURE_FEATURES),
        'no_phase': [f for f in GESTURE_FEATURES if f not in PHASE_FEATURES],
        'phase_only': list(PHASE_FEATURES),
        'no_duration': [f for f in GESTURE_FEATURES if f != 'event_duration'],
        'no_peak_position': [f for f in GESTURE_FEATURES if f != 'peak_position'],
        'no_asymmetry': [f for f in GESTURE_FEATURES if f != 'rise_fall_asymmetry'],
        'compact_shape': [
            f for f in GESTURE_FEATURES
            if f not in {'peak_position', 'rise_fall_asymmetry'}
        ],
    }

    variants = []
    for name in variant_names:
        key = name.strip()
        if not key:
            continue
        if key not in presets:
            print(f"Warning: Unknown ablation variant '{key}', skipping.")
            continue
        features = presets[key]
        if not features:
            print(f"Warning: Variant '{key}' has no features, skipping.")
            continue
        variants.append((key, features))

    if not variants:
        variants = [('all', list(GESTURE_FEATURES))]
    return variants


def run_feature_ablation(seed=None, window_seconds=2.0, window_overlap=0.0,
                         window_labels=None, packet_rate=DEFAULT_PACKET_RATE,
                         no_gesture_max_per_source=-1,
                         variants_csv='all,no_phase,no_duration,compact_shape,phase_only'):
    """Run feature ablation study for gesture LogReg without exporting artifacts."""
    print(f'\n{"="*60}')
    print('  GESTURE FEATURE ABLATION (LogReg)')
    print(f'{"="*60}\n')

    prep = _prepare_gesture_dataset(
        seed=seed,
        window_seconds=window_seconds,
        window_overlap=window_overlap,
        window_labels=window_labels,
        packet_rate=packet_rate,
        no_gesture_max_per_source=no_gesture_max_per_source,
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
        return 1
    if len(class_names) < 2:
        print('Error: At least 2 gesture classes required for ablation.')
        return 1

    _print_dataset_label_stats(stats, class_names)
    print(f'\nClasses ({len(class_names)}): {class_names}')

    variant_names = [v.strip() for v in variants_csv.split(',') if v.strip()]
    variants = _build_feature_ablation_variants(variant_names)
    print(f"\nVariants: {[name for name, _ in variants]}")

    n_folds = min(5, min(np.bincount(y)))
    if n_folds < 2:
        print('Error: Too few samples per class for CV ablation.')
        return 1

    rows = []
    for name, selected_features in variants:
        X_sel, _ = _slice_features(X, selected_features)
        cv = _cross_validate_logreg(X_sel, y, groups=groups, n_folds=n_folds)
        model = _fit_logreg_model(X_sel, y)
        holdout_seed = prep['seed']
        _, X_test_raw, _, y_test = split_holdout(
            X_sel, y, test_size=0.2, random_state=holdout_seed, groups=groups
        )
        y_pred = model.predict(X_test_raw)
        holdout_acc = float(np.mean(y_pred == y_test) * 100.0)
        rows.append({
            'variant': name,
            'n_features': len(selected_features),
            'f1_mean': cv['f1_mean'],
            'f1_std': cv['f1_std'],
            'acc_mean': cv['accuracy_mean'],
            'holdout_acc': holdout_acc,
            'features': selected_features,
            'model': model,
        })

    rows.sort(key=lambda r: (r['f1_mean'], r['acc_mean'], r['holdout_acc']), reverse=True)

    print("\n" + "-" * 86)
    print(f"{'Variant':<18} {'#Feat':>5} {'CV F1':>14} {'CV Acc':>12} {'Holdout Acc':>14}")
    print("-" * 86)
    for r in rows:
        print(f"{r['variant']:<18} {r['n_features']:>5d} "
              f"{r['f1_mean']:>6.1f}+/-{r['f1_std']:<5.1f} "
              f"{r['acc_mean']:>10.1f}% {r['holdout_acc']:>13.1f}%")
    print("-" * 86)

    best = rows[0]
    print(f"\nBest variant: {best['variant']} (CV F1={best['f1_mean']:.1f}%)")
    print(f"Features ({best['n_features']}): {best['features']}")
    _print_feature_importance(best['features'], best['model'], top_n=best['n_features'])
    print("\nAblation completed. No model export performed.\n")
    return 0


def _export_logreg_micropython(model, output_path, seed, class_names):
    """Export logistic regression as single affine layer (W1/B1) for MicroPython."""
    scaler = model.named_steps['scaler']
    num_features = len(scaler.mean_)
    num_classes = len(class_names)
    W, b = _logreg_to_multiclass_logits(model, num_features, num_classes)

    timestamp = __import__('datetime').datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    code = f'''"""
Micro-ESPectre - Gesture LogReg Weights

Auto-generated logistic regression weights for gesture classification.
Architecture: {num_features} -> {num_classes} (softmax)
Classes: {num_classes}
Trained: {timestamp}
Seed: {seed}

This file is auto-generated by 11_train_gesture_model.py.
DO NOT EDIT - your changes will be overwritten!

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

GESTURE_FEATURE_MEAN = [{', '.join(f'{x:.6f}' for x in scaler.mean_)}]
GESTURE_FEATURE_SCALE = [{', '.join(f'{x:.6f}' for x in scaler.scale_)}]
GESTURE_CLASS_LABELS = {class_names}
GESTURE_NUM_CLASSES = {num_classes}

# Single affine layer logits = x_norm @ W1 + B1
W1 = [
'''
    for i in range(W.shape[0]):
        code += '    [' + ', '.join(f'{v:.6f}' for v in W[i]) + '],\n'
    code += ']\n'
    code += 'B1 = [' + ', '.join(f'{v:.6f}' for v in b) + ']\n'

    with open(output_path, 'w') as f:
        f.write(code)
    return len(code), scaler, W, b


def _export_logreg_cpp(model, output_path, seed, class_names):
    """Export logistic regression weights for C++ ESPHome gesture runtime."""
    scaler = model.named_steps['scaler']
    num_features = len(scaler.mean_)
    num_classes = len(class_names)
    W, b = _logreg_to_multiclass_logits(model, num_features, num_classes)

    timestamp = __import__('datetime').datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    code = f'''/*
 * ESPectre - Gesture LogReg Weights
 *
 * Auto-generated logistic regression weights for gesture classification.
 * Architecture: {num_features} -> {num_classes} (softmax)
 * Classes: {num_classes}
 * Trained: {timestamp}
 * Seed: {seed}
 *
 * This file is auto-generated by 11_train_gesture_model.py.
 * DO NOT EDIT - your changes will be overwritten!
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#pragma once

namespace esphome {{
namespace espectre {{

constexpr int GESTURE_NUM_CLASSES = {num_classes};

constexpr float GESTURE_FEATURE_MEAN[{num_features}] = {{{', '.join(f'{x:.6f}f' for x in scaler.mean_)}}};
constexpr float GESTURE_FEATURE_SCALE[{num_features}] = {{{', '.join(f'{x:.6f}f' for x in scaler.scale_)}}};

constexpr const char* GESTURE_CLASS_LABELS[{num_classes}] = {{
'''
    for name in class_names:
        code += f'    "{name}",\n'
    code += '};\n\n'
    code += f'constexpr float GESTURE_W1[{num_features}][{num_classes}] = {{\n'
    for i in range(W.shape[0]):
        code += '    {' + ', '.join(f'{v:.6f}f' for v in W[i]) + '},\n'
    code += '};\n'
    code += f'constexpr float GESTURE_B1[{num_classes}] = {{{", ".join(f"{v:.6f}f" for v in b)}}};\n\n'
    code += '''}  // namespace espectre
}  // namespace esphome
'''
    with open(output_path, 'w') as f:
        f.write(code)
    return len(code), scaler


def _export_logreg_test_data(model, X_test_raw, y_test, output_path):
    """Export raw test features and expected probabilities for C++ parity tests."""
    probs = model.predict_proba(X_test_raw).astype(np.float32)
    y_pred = np.argmax(probs, axis=1).astype(np.int32)
    np.savez(
        output_path,
        features=X_test_raw.astype(np.float32),
        labels=y_test.astype(np.int32),
        expected_probs=probs,
        expected_pred=y_pred,
    )
    return len(X_test_raw)


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
                             window_labels=None, packet_rate=DEFAULT_PACKET_RATE,
                             no_gesture_max_per_source=-1):
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
    if window_seconds and window_seconds > 0:
        if not (0.0 <= window_overlap < 1.0):
            print('Error: --window-overlap must be in range [0.0, 1.0).')
            return None
        if packet_rate <= 0:
            print('Error: --packet-rate must be > 0.')
            return None
        window_packets = max(10, int(round(window_seconds * packet_rate)))
        stride_packets = max(1, int(round(window_packets * (1.0 - window_overlap))))
        effective_window_labels = [lbl.strip() for lbl in (window_labels or []) if lbl.strip()]

        print(f'Windowing enabled: {window_seconds:.2f}s ({window_packets} packets), '
              f'overlap={window_overlap:.2f}, stride={stride_packets} packets')
        print(f'  Labels with windowing: {effective_window_labels}')

    # Always window negative samples to keep no_gesture coherent with runtime window length.
    no_gesture_window_packets = max(10, int(round(NO_GESTURE_WINDOW_SECONDS * packet_rate)))
    no_gesture_stride_packets = no_gesture_window_packets

    print('Loading gesture data...')
    print(f'  Synthetic class: {NO_GESTURE_LABEL} <= {NO_GESTURE_SOURCE_LABELS}')
    print(f'  Forced windowing for {NO_GESTURE_SOURCE_LABELS}: '
          f'{NO_GESTURE_WINDOW_SECONDS:.2f}s ({no_gesture_window_packets} packets), '
          f'stride={no_gesture_stride_packets}')
    if no_gesture_max_per_source == -1:
        positive_counts = _estimate_positive_gesture_counts(
            window_packets=window_packets,
            stride_packets=stride_packets,
            window_labels=effective_window_labels,
            first_window_only=True,
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


def train_all(seed=None, hidden_layers=None, window_seconds=2.0, window_overlap=0.0,
              window_labels=None, packet_rate=DEFAULT_PACKET_RATE,
              no_gesture_max_per_source=-1):
    """Train gesture classifier using logistic regression.

    Args:
        seed: Optional random seed for reproducibility.
        hidden_layers: MLP hidden layer sizes (default: [24]).
        window_seconds: Optional fixed window length in seconds for selected labels.
        window_overlap: Overlap ratio between consecutive windows [0.0, 1.0).
        window_labels: Labels to split into windows.
        packet_rate: Packets per second used for seconds->packets conversion.
        no_gesture_max_per_source: Cap of no_gesture samples per source label
                                  (baseline/movement) after windowing.
                                  -1 = auto (match positive gesture sample count),
                                   0 = no cap, >0 = explicit cap.
    """
    if hidden_layers is not None:
        print("Note: --hidden-layers is ignored for gesture training (LogReg mode).")

    print(f'\n{"="*60}')
    print('  GESTURE CLASSIFIER TRAINING')
    print(f'{"="*60}\n')

    prep = _prepare_gesture_dataset(
        seed=seed,
        window_seconds=window_seconds,
        window_overlap=window_overlap,
        window_labels=window_labels,
        packet_rate=packet_rate,
        no_gesture_max_per_source=no_gesture_max_per_source,
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
    print(f'Features ({NUM_GESTURE_FEATURES}): {", ".join(GESTURE_FEATURES)}\n')

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
        print(f'{n_folds}-fold cross-validation (LogReg {NUM_GESTURE_FEATURES}->{num_classes})...')
        cv_results = _cross_validate_logreg(X, y, groups=groups, n_folds=n_folds)
        print(f'  Accuracy: {cv_results["accuracy_mean"]:.1f}% (+/- {cv_results["accuracy_std"]:.1f}%)')
        print(f'  F1 Score: {cv_results["f1_mean"]:.1f}% (+/- {cv_results["f1_std"]:.1f}%)')

    X_train_raw, X_test_raw, y_train, y_test = split_holdout(
        X, y, test_size=0.2, random_state=seed, groups=groups
    )

    print('\nTraining final model on full dataset (LogReg)...')
    model = _fit_logreg_model(X, y)
    probs = model.predict_proba(X_test_raw)

    y_pred = np.argmax(probs, axis=1)
    test_acc = np.mean(y_pred == y_test) * 100
    print(f'\nHold-out test set (20%):')
    print(f'  Accuracy: {test_acc:.1f}%')
    print(f'  Per-class accuracy:')
    for cid, name in enumerate(class_names):
        mask = y_test == cid
        if mask.sum() > 0:
            acc = np.mean(y_pred[mask] == cid) * 100
            print(f'    {name}: {acc:.1f}%')

    print('\nExporting models...')
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    mp_path = SRC_DIR / 'gesture_weights.py'
    mp_size, mp_scaler, _, _ = _export_logreg_micropython(
        model, mp_path, seed=seed, class_names=class_names
    )
    print(f'  MicroPython weights: {mp_path.name} ({mp_size/1024:.1f} KB)')

    cpp_path = CPP_DIR / 'gesture_weights.h'
    cpp_size, cpp_scaler = _export_logreg_cpp(
        model, cpp_path, seed=seed, class_names=class_names
    )
    print(f'  C++ weights: {cpp_path.name} ({cpp_size/1024:.1f} KB)')

    scaler_path = MODELS_DIR / 'gesture_scaler.npz'
    np.savez(scaler_path, mean=mp_scaler.mean_, scale=mp_scaler.scale_)
    print(f'  Scaler: {scaler_path.name}')

    test_data_path = MODELS_DIR / 'gesture_test_data.npz'
    n_test = _export_logreg_test_data(model, X_test_raw, y_test, test_data_path)
    print(f'  Test data: {test_data_path.name} ({n_test} samples)')

    print(f'\n{"="*60}')
    print('  DONE!')
    print(f'{"="*60}')
    print(f'\nModel trained on {stats["total"]} samples, {num_classes} classes')
    print(f'CV F1={cv_results["f1_mean"]:.1f}% (+/- {cv_results["f1_std"]:.1f}%)')
    print(f'Classes: {class_names}')
    print(f'\nGenerated files:')
    print(f'  - {mp_path} (MicroPython)')
    print(f'  - {cpp_path} (C++ ESPHome)')
    print(f'  - {scaler_path}')
    print(f'  - {test_data_path}')
    print()
    return 0


def run_experiment(seed=None, window_seconds=2.0, window_overlap=0.0, window_labels=None,
                   packet_rate=DEFAULT_PACKET_RATE, no_gesture_max_per_source=-1):
    """Compare MLP, classical, and temporal models."""
    try:
        with suppress_stderr():
            import tensorflow as tf
            setup_tf_logging()
    except ImportError as e:
        print(f"Error: Missing dependency - {e}")
        return 1

    prep = _prepare_gesture_dataset(
        seed=seed,
        window_seconds=window_seconds,
        window_overlap=window_overlap,
        window_labels=window_labels,
        packet_rate=packet_rate,
        no_gesture_max_per_source=no_gesture_max_per_source,
    )
    if prep is None:
        return 1

    window_packets = prep['window_packets']
    stride_packets = prep['stride_packets']
    no_gesture_window_packets = prep['no_gesture_window_packets']
    no_gesture_stride_packets = prep['no_gesture_stride_packets']
    X = prep['X']
    y = prep['y']
    class_names = prep['class_names']
    stats = prep['stats']
    groups = prep['groups']

    if len(X) == 0:
        print("Error: No gesture data found.")
        return 1

    if len(class_names) < 2:
        print("Error: At least 2 gesture classes required for experiment.")
        return 1

    _print_dataset_label_stats(stats, class_names)

    num_classes = len(class_names)
    mlp_best = experiment_architectures(
        X, y, num_classes, class_names,
        num_features=NUM_GESTURE_FEATURES,
        title="GESTURE ARCHITECTURE EXPERIMENT",
        groups=groups,
    )

    classical_results = _run_classical_experiments(X, y, class_names, groups=groups)

    # Temporal experiments require fixed windows; include all available gesture labels.
    temporal_labels = class_names
    X_seq, y_seq, seq_class_names, _, seq_groups = load_gesture_temporal_data(
        window_packets=window_packets,
        stride_packets=stride_packets,
        window_labels=temporal_labels,
        no_gesture_window_packets=no_gesture_window_packets,
        no_gesture_stride_packets=no_gesture_stride_packets,
    )
    temporal_result = None
    if len(X_seq) > 0 and len(seq_class_names) >= 2:
        print(f"\nLoading temporal data... {len(X_seq)} windows "
              f"({X_seq.shape[1]} packets each), classes={seq_class_names}")
        temporal_result = _run_temporal_experiment(
            X_seq, y_seq, seq_class_names, groups=seq_groups
        )
    else:
        print("\nTemporal experiment skipped: insufficient fixed-window temporal data.")

    _print_experiment_summary(mlp_best, classical_results, temporal_result)
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
       python tools/11_train_gesture_model.py
       python tools/11_train_gesture_model.py --window-seconds 2.0 --window-labels wave,circle_cw
  3. Run:
       Deploy to ESP32 or use with micro-espectre/src/gesture_detector.py

Motion model (binary IDLE/MOTION): python tools/10_train_motion_model.py
'''
    )
    parser.add_argument('--info', action='store_true',
                        help='Show gesture dataset information')
    parser.add_argument('--experiment', action='store_true',
                        help='Compare MLP, classical, and temporal models')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducible training')
    parser.add_argument('--hidden-layers', type=int, nargs='+', default=[24],
                        help='Hidden layer sizes (default: 24)')
    parser.add_argument('--window-seconds', type=float, default=2.0,
                        help='Fixed window length in seconds (default: 2.0)')
    parser.add_argument('--window-overlap', type=float, default=0.0,
                        help='Window overlap ratio [0.0, 1.0), default: 0.0')
    parser.add_argument('--window-labels', type=str, default='',
                        help='Comma-separated labels to window (default: none)')
    parser.add_argument('--packet-rate', type=float, default=DEFAULT_PACKET_RATE,
                        help=f'Packets/sec for seconds->packets conversion (default: {DEFAULT_PACKET_RATE})')
    parser.add_argument('--no-gesture-max-per-source', type=int, default=-1,
                        help='Cap no_gesture samples per source label (baseline/movement). '
                             '-1 = auto (match gesture count), 0 = no cap, >0 = explicit cap '
                             '(default: -1)')
    parser.add_argument('--feature-ablation', action='store_true',
                        help='Run feature ablation study with LogReg (no export)')
    parser.add_argument('--ablation-variants', type=str,
                        default='all,no_phase,no_duration,compact_shape,phase_only',
                        help='Comma-separated ablation variants. '
                             'Supported: all,no_phase,phase_only,no_duration,'
                             'no_peak_position,no_asymmetry,compact_shape')
    args = parser.parse_args()

    window_labels = [lbl.strip() for lbl in args.window_labels.split(',') if lbl.strip()]

    if args.info:
        show_info()
        return 0

    if args.experiment:
        return run_experiment(
            seed=args.seed,
            window_seconds=args.window_seconds,
            window_overlap=args.window_overlap,
            window_labels=window_labels,
            packet_rate=args.packet_rate,
            no_gesture_max_per_source=args.no_gesture_max_per_source,
        )

    if args.feature_ablation:
        return run_feature_ablation(
            seed=args.seed,
            window_seconds=args.window_seconds,
            window_overlap=args.window_overlap,
            window_labels=window_labels,
            packet_rate=args.packet_rate,
            no_gesture_max_per_source=args.no_gesture_max_per_source,
            variants_csv=args.ablation_variants,
        )

    return train_all(
        seed=args.seed,
        hidden_layers=args.hidden_layers,
        window_seconds=args.window_seconds,
        window_overlap=args.window_overlap,
        window_labels=window_labels,
        packet_rate=args.packet_rate,
        no_gesture_max_per_source=args.no_gesture_max_per_source,
    )


if __name__ == '__main__':
    exit(main())
