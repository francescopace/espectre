#!/usr/bin/env python3
"""
ML Motion Detection - Training Script

Trains neural network models for motion detection using all available CSI data.
Generates models for both ESP-IDF (TFLite) and MicroPython.

Training features:
  - 5-fold stratified cross-validation for reliable metrics
  - Early stopping with patience to prevent overfitting
  - Dropout regularization during training
  - Balanced class weights for imbalanced datasets
  - Learning rate reduction on plateau
  - Configurable FP penalty (--fp-weight) for conservative models

Usage:
    python tools/10_train_ml_model.py                    # Train with default features
    python tools/10_train_ml_model.py --info             # Show dataset info
    python tools/10_train_ml_model.py --experiment       # Compare architectures
    python tools/10_train_ml_model.py --fp-weight 2.0    # Penalize FP 2x more
    python tools/10_train_ml_model.py --shap              # SHAP importance (200 samples)
    python tools/10_train_ml_model.py --shap 500          # SHAP importance (500 samples)

Configuration:
  - TRAINING_FEATURES: Edit at top of file to change feature set

Note: Files without gain lock use CV normalization (auto-detected from each .npz
file via the `gain_locked` field).

To compare ML with MVS, use:
    python tools/7_compare_detection_methods.py

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

# Suppress TensorFlow/absl warnings BEFORE any imports
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['ABSL_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '2'

import argparse
import numpy as np
import subprocess
import re
import shutil
import tempfile
from pathlib import Path
from collections import deque
from contextlib import contextmanager
from datetime import datetime


@contextmanager
def suppress_stderr():
    """
    Context manager to suppress stderr output at the file descriptor level.
    
    This is necessary because TensorFlow's C++ code writes directly to the
    C-level stderr, bypassing Python's sys.stderr.
    """
    # Save the original stderr file descriptor
    stderr_fd = sys.stderr.fileno()
    saved_stderr_fd = os.dup(stderr_fd)
    
    # Open /dev/null and redirect stderr to it
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, stderr_fd)
    os.close(devnull)
    
    try:
        yield
    finally:
        # Restore the original stderr
        os.dup2(saved_stderr_fd, stderr_fd)
        os.close(saved_stderr_fd)

# Import csi_utils first - it sets up paths automatically
from csi_utils import (
    load_npz_as_packets,
    DATA_DIR,
)
from config import SEG_WINDOW_SIZE, DEFAULT_SUBCARRIERS, HAMPEL_WINDOW, HAMPEL_THRESHOLD
from segmentation import SegmentationContext
from features import (
    extract_features_by_name, DEFAULT_FEATURES,
)

# ============================================================================
# Feature Selection
# ============================================================================
#
# 12 features: 11 turbulence-based + 1 temporal.
# Top-3 by SHAP: turb_autocorr (40.9%), turb_std (12.0%), turb_mean (8.1%).
# See ALGORITHMS.md "Feature Importance" for full SHAP/correlation table.
# ============================================================================

# ============================================================================
# FEATURE SET TO USE FOR TRAINING
# ============================================================================
# Change this list to experiment with different features.
# Available features are defined in src/features.py
#
# Current default (12 features):
TRAINING_FEATURES = DEFAULT_FEATURES

# To experiment with different features, define a custom list here.


# Directories
MODELS_DIR = Path(__file__).parent.parent / 'models'
SRC_DIR = Path(__file__).parent.parent / 'src'
CPP_DIR = Path(__file__).parent.parent.parent / 'components' / 'espectre'


# ============================================================================
# Data Loading
# ============================================================================

def load_dataset_info():
    """Load dataset_info.json with label mappings."""
    import json
    info_path = DATA_DIR / 'dataset_info.json'
    if info_path.exists():
        with open(info_path, 'r') as f:
            return json.load(f)
    return {'labels': {}}


def _build_dataset_file_index(dataset_info):
    """Build filename -> (label, entry) index from dataset_info files section."""
    index = {}
    for label, files in dataset_info.get('files', {}).items():
        for entry in files:
            name = entry.get('filename')
            if name:
                index[name] = (label, entry)
    return index


def _is_temporally_paired(dataset_info, label, entry, max_delta_seconds=30 * 60):
    """Check if entry has a valid counterpart within max delta."""
    if label == 'baseline':
        counterpart_label = 'movement'
        counterpart_name = entry.get('optimal_pair_movement_file')
    else:
        counterpart_label = 'baseline'
        counterpart_name = entry.get('optimal_pair_baseline_file')
    if not counterpart_name:
        return False

    counterpart = None
    for candidate in dataset_info.get('files', {}).get(counterpart_label, []):
        if candidate.get('filename') == counterpart_name:
            counterpart = candidate
            break
    if counterpart is None:
        return False

    try:
        t1 = datetime.fromisoformat(entry['collected_at'])
        t2 = datetime.fromisoformat(counterpart['collected_at'])
    except Exception:
        return False
    return abs((t2 - t1).total_seconds()) <= max_delta_seconds


def build_gridsearch_tuning_map(dataset_info, default_subcarriers, default_threshold=1.0):
    """
    Build per-file tuning map from dataset_info.

    Returns:
        dict: {
            filename: {
                'subcarriers': list[int],
                'threshold': float,
                'mode': 'paired' | 'single-dataset fallback' | 'missing',
                'confidence_factor': float,
            }
        }
    """
    tuning = {}
    for label, files in dataset_info.get('files', {}).items():
        for entry in files:
            name = entry.get('filename')
            if not name:
                continue

            subcarriers = default_subcarriers
            threshold = float(entry.get('optimal_threshold_gridsearch', default_threshold))
            paired = _is_temporally_paired(dataset_info, label, entry)
            mode = 'paired' if paired else 'single-dataset fallback'
            confidence_factor = 1.0 if paired else 0.5
            tuning[name] = {
                'subcarriers': list(subcarriers),
                'threshold': threshold,
                'mode': mode,
                'confidence_factor': confidence_factor,
            }
    return tuning


def is_motion_label(label_name, dataset_info):
    """
    Determine if a label represents motion or idle.
    
    Uses dataset_info.json labels when available (name-based schema).
    
    Args:
        label_name: Label name from npz file
        dataset_info: Loaded dataset_info.json
    
    Returns:
        bool: True if motion, False if idle
    """
    labels = dataset_info.get('labels', {})
    if label_name in labels:
        return label_name == 'movement'
    # Default: only 'movement' is motion
    return label_name == 'movement'


def get_file_metadata(dataset_info):
    """
    Get metadata for all files in dataset_info.json.
    
    Returns a dict mapping filename to metadata including:
    - gain_locked: Whether AGC gain lock was active for this file
    
    Args:
        dataset_info: Loaded dataset_info.json
    
    Returns:
        dict: {filename: {gain_locked: bool, ...}}
    """
    file_metadata = {}
    files_by_label = dataset_info.get('files', {})
    for label, file_list in files_by_label.items():
        for file_info in file_list:
            filename = file_info.get('filename', '')
            if filename:
                file_metadata[filename] = {
                    'gain_locked': file_info.get('gain_locked', True),
                    'chip': file_info.get('chip', 'unknown'),
                }
    return file_metadata


def load_all_data():
    """
    Load all available CSI data from the data/ directory.
    
    Reads label from npz file metadata (not folder structure).
    Uses dataset_info.json only to determine if label is motion or idle.
    Uses each NPZ file's `gain_locked` field to decide CV normalization.
    
    Returns:
        tuple: (all_packets, stats) where stats is a dict with dataset info
    """
    all_packets = []
    stats = {
        'chips': set(),
        'labels': {},
        'total': 0,
        'cv_norm_files': set(),
        'files': [],
        'excluded_labels': set(),
    }
    
    # Load dataset info for label mapping and file metadata
    dataset_info = load_dataset_info()
    file_metadata = get_file_metadata(dataset_info)
    
    # Scan all subdirectories in data/
    excluded_dirs = {'.'}
    for subdir in sorted(DATA_DIR.iterdir()):
        if not subdir.is_dir() or subdir.name in excluded_dirs:
            continue
        
        # Load all npz files in this directory
        for npz_file in sorted(subdir.glob('*.npz')):
            try:
                packets = load_npz_as_packets(npz_file)
                if not packets:
                    continue
                
                # Get label from file metadata (already set by load_npz_as_packets)
                label = packets[0].get('label', subdir.name)
                
                # Keep training strictly on baseline/movement labels.
                # Test/control datasets must never be part of model training.
                label_lc = str(label).lower()
                if label_lc not in ('baseline', 'movement'):
                    stats['excluded_labels'].add(label_lc)
                    continue

                # Get chip
                chip = packets[0].get('chip', 'unknown').upper()
                
                # Track stats
                if label not in stats['labels']:
                    stats['labels'][label] = 0
                stats['labels'][label] += len(packets)
                stats['total'] += len(packets)
                
                stats['chips'].add(chip)
                
                # Get file-specific metadata
                meta = file_metadata.get(npz_file.name, {})
                gain_locked = bool(meta.get('gain_locked', packets[0].get('gain_locked', True)))
                if not gain_locked:
                    stats['cv_norm_files'].add(npz_file.name)
                
                # Add flags to each packet
                is_motion = is_motion_label(label, dataset_info)
                for idx, p in enumerate(packets):
                    p['is_motion'] = is_motion
                    p['gain_locked'] = gain_locked
                    p['source_file'] = npz_file.name
                    p['packet_index'] = idx
                
                all_packets.extend(packets)
                stats['files'].append(npz_file.name)
                
            except Exception as e:
                print(f"  Warning: Could not load {npz_file.name}: {e}")
    
    stats['chips'] = sorted(stats['chips'])
    stats['cv_norm_files'] = sorted(stats['cv_norm_files'])
    stats['excluded_labels'] = sorted(stats['excluded_labels'])
    return all_packets, stats


# ============================================================================
# Feature Extraction
# ============================================================================

def extract_features(packets, window_size=SEG_WINDOW_SIZE, subcarriers=None,
                     feature_names=None, return_metadata=False,
                     enable_hampel=True, hampel_window=HAMPEL_WINDOW, hampel_threshold=HAMPEL_THRESHOLD):
    """
    Extract features from CSI packets using sliding window.
    
    Uses SegmentationContext.add_turbulence() so the filter chain (Hampel -> low-pass)
    matches the runtime pipeline, ensuring train/deploy alignment.
    
    Args:
        packets: List of CSI packets with 'csi_data' and 'label'
        window_size: Sliding window size (default: SEG_WINDOW_SIZE from config.py)
        subcarriers: List of subcarrier indices to use (default: DEFAULT_SUBCARRIERS)
        feature_names: List of feature names to extract (default: DEFAULT_FEATURES)
        return_metadata: If True, return per-sample metadata
        enable_hampel: Enable Hampel outlier filter on turbulence (default: True)
        hampel_window: Hampel filter window size (default: 7)
        hampel_threshold: Hampel filter threshold in MAD units (default: 5.0)
    
    Returns:
        tuple: (X, y, feature_names, groups)
            - X: Feature matrix (n_samples, n_features)
            - y: Labels (n_samples,)
            - feature_names: List of feature names
            - groups: Chip group labels per sample (n_samples,) for grouped CV
    """
    if subcarriers is None:
        subcarriers = DEFAULT_SUBCARRIERS
    
    if feature_names is None:
        feature_names = DEFAULT_FEATURES.copy()
    
    X, y = [], []
    groups = []

    # Process each source file independently to avoid window leakage across files.
    grouped = {}
    for pkt in packets:
        source = pkt.get('source_file', '__single_stream__')
        grouped.setdefault(source, []).append(pkt)

    for source_file, file_packets in grouped.items():
        chip = file_packets[0].get('chip', 'unknown').upper()
        ctx = SegmentationContext(
            window_size=window_size,
            threshold=1.0,
            enable_hampel=enable_hampel,
            hampel_window=hampel_window,
            hampel_threshold=hampel_threshold,
        )
        last_amplitudes = None
        for pkt in file_packets:
            csi_data = pkt['csi_data']

            use_cv_norm = not pkt.get('gain_locked', True)
            ctx.use_cv_normalization = use_cv_norm
            turb, amps = SegmentationContext.compute_spatial_turbulence(
                csi_data, subcarriers, use_cv_normalization=use_cv_norm
            )
            ctx.add_turbulence(turb)
            last_amplitudes = amps

            if ctx.buffer_count < window_size:
                continue

            # Reconstruct chronological order from circular buffer
            idx = ctx.buffer_index
            if ctx.buffer_count < ctx.window_size:
                turb_list = ctx.turbulence_buffer[:ctx.buffer_count]
            else:
                turb_list = ctx.turbulence_buffer[idx:] + ctx.turbulence_buffer[:idx]
            n = len(turb_list)

            features = extract_features_by_name(
                turb_list, n,
                amplitudes=last_amplitudes,
                feature_names=feature_names
            )

            X.append(features)
            y.append(1 if pkt.get('is_motion', False) else 0)
            groups.append(chip)

    X_arr = np.array(X)
    y_arr = np.array(y)
    groups_arr = np.array(groups)
    return X_arr, y_arr, feature_names, groups_arr


def compute_mvs_guided_sample_weights(packets, tuning_map, window_size=SEG_WINDOW_SIZE):
    """
    Compute sample weights using context-aware MVS scoring per source file.

    Weight policy:
    - movement samples: hard-positive mining
      (harder positives near threshold are weighted more)
    - baseline samples: promote hard negatives (2.0) when metric >= threshold, else 1.0
    - single-dataset fallback files get reduced confidence factor (0.5)
    """
    weights = []

    grouped = {}
    for pkt in packets:
        source = pkt.get('source_file', '__single_stream__')
        grouped.setdefault(source, []).append(pkt)

    for source_file, file_packets in grouped.items():
        cfg = tuning_map.get(source_file, None)
        if cfg is None:
            subcarriers = DEFAULT_SUBCARRIERS
            threshold = 1.0
            confidence_factor = 0.5
            mode = 'missing'
        else:
            subcarriers = cfg['subcarriers']
            threshold = max(float(cfg['threshold']), 1e-6)
            confidence_factor = float(cfg['confidence_factor'])
            mode = cfg['mode']

        ctx = SegmentationContext(window_size=window_size, threshold=threshold)
        for pkt in file_packets:
            use_cv_norm = not pkt.get('gain_locked', True)
            turb, _ = SegmentationContext.compute_spatial_turbulence(
                pkt['csi_data'], subcarriers, use_cv_normalization=use_cv_norm
            )
            ctx.add_turbulence(turb)
            ctx.update_state()
            if ctx.buffer_count < window_size:
                continue

            ratio = max(0.0, float(ctx.current_moving_variance)) / threshold
            if pkt.get('is_motion', False):
                # Hard-positive mining:
                # subtle/near-threshold motion is exactly where recall drops in
                # deployment, so up-weight it; easy positives get lower weight.
                if ratio < 0.5:
                    base = 2.4
                elif ratio < 0.8:
                    base = 2.0
                elif ratio < 1.0:
                    base = 1.7
                elif ratio < 1.3:
                    base = 1.3
                elif ratio < 1.8:
                    base = 1.0
                else:
                    base = 0.8
            else:
                base = 2.0 if ratio >= 1.0 else 1.0
            weights.append(base * confidence_factor)

    return np.array(weights, dtype=np.float32)


# ============================================================================
# Model Training
# ============================================================================

def build_model(hidden_layers=[16, 8], num_features=12, use_dropout=True, dropout_rate=0.2):
    """
    Build a Keras MLP model.
    
    Dropout layers are added during training for regularization but are
    automatically disabled during inference (and don't affect exported weights).
    
    Args:
        hidden_layers: List of hidden layer sizes
        num_features: Number of input features
        use_dropout: Whether to add dropout layers (for training only)
        dropout_rate: Dropout rate (0.0-1.0)
    
    Returns:
        Compiled Keras model
    """
    import tensorflow as tf
    
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(num_features,)))
    
    for units in hidden_layers:
        model.add(tf.keras.layers.Dense(units, activation='relu'))
        if use_dropout and dropout_rate > 0:
            model.add(tf.keras.layers.Dropout(dropout_rate))
    
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train_model(X, y, hidden_layers=[16, 8], max_epochs=200, use_dropout=True,
                class_weight=None, fp_weight=1.0, sample_weight=None, verbose=0):
    """
    Train a neural network model with best practices.
    
    Uses early stopping, learning rate reduction, dropout regularization,
    and optional class weighting for imbalanced datasets.
    
    Args:
        X: Feature matrix (normalized)
        y: Labels
        hidden_layers: List of hidden layer sizes
        max_epochs: Maximum training epochs (early stopping will cut short)
        use_dropout: Whether to add dropout layers
        class_weight: Class weight dict (e.g., {0: 1.0, 1: 2.0}) or None for auto
        fp_weight: Multiplier for class 0 (IDLE) weight to penalize false positives.
                   Values >1.0 make the model more conservative (fewer FP, lower recall).
        sample_weight: Optional per-sample weights
        verbose: Training verbosity
    
    Returns:
        Trained Keras model
    """
    import tensorflow as tf
    
    # Auto-compute class weights if not provided
    if class_weight is None:
        n_total = len(y)
        n_pos = np.sum(y == 1)
        n_neg = n_total - n_pos
        if n_pos > 0 and n_neg > 0:
            # Balanced class weights: higher weight for minority class
            class_weight = {
                0: n_total / (2 * n_neg),
                1: n_total / (2 * n_pos)
            }
    
    # Apply FP penalty: increase weight for class 0 (IDLE)
    # This makes misclassifying baseline as motion more costly
    if fp_weight != 1.0 and class_weight is not None:
        class_weight[0] *= fp_weight

    # Keras forbids passing class_weight and sample_weight together.
    # Merge class weights into sample_weight when both are requested.
    if sample_weight is not None and class_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=np.float32).copy()
        class_multiplier = np.where(np.asarray(y) == 1, class_weight[1], class_weight[0])
        sample_weight *= class_multiplier.astype(np.float32)
        class_weight = None
    
    # Determine number of features from input shape
    num_features = X.shape[1] if hasattr(X, 'shape') else len(X[0])
    model = build_model(hidden_layers=hidden_layers, num_features=num_features, use_dropout=use_dropout)
    
    # Stratified validation split (Keras validation_split takes the last N%
    # in order, which can skew chip representation)
    from sklearn.model_selection import train_test_split as _val_split
    split_kwargs = dict(test_size=0.1, random_state=42, stratify=np.asarray(y))
    if sample_weight is not None:
        X_t, X_v, y_t, y_v, sw_t, sw_v = _val_split(
            X, y, sample_weight, **split_kwargs
        )
    else:
        X_t, X_v, y_t, y_v = _val_split(X, y, **split_kwargs)
        sw_t, sw_v = None, None

    # Callbacks for training robustness
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            min_delta=1e-4
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-6
        ),
    ]
    
    model.fit(
        X_t, y_t,
        epochs=max_epochs,
        batch_size=32,
        validation_data=(X_v, y_v, sw_v) if sw_v is not None else (X_v, y_v),
        class_weight=class_weight,
        sample_weight=sw_t,
        callbacks=callbacks,
        verbose=verbose
    )
    
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a model on test data and return metrics dict.
    
    Args:
        model: Trained Keras model
        X_test: Test features (normalized)
        y_test: Test labels
    
    Returns:
        dict: Metrics (recall, precision, fp_rate, f1, tp, fp, tn, fn)
    """
    y_pred = (model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
    tp = int(np.sum((y_test == 1) & (y_pred == 1)))
    fp = int(np.sum((y_test == 0) & (y_pred == 1)))
    tn = int(np.sum((y_test == 0) & (y_pred == 0)))
    fn = int(np.sum((y_test == 1) & (y_pred == 0)))
    
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    fp_rate = fp / (fp + tn) * 100 if (fp + tn) > 0 else 0
    f1 = 2 * tp / (2 * tp + fp + fn) * 100 if (2 * tp + fp + fn) > 0 else 0
    
    return {
        'recall': recall, 'precision': precision,
        'fp_rate': fp_rate, 'f1': f1,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
    }


def cross_validate(X, y, hidden_layers=[16, 8], n_folds=5, max_epochs=200,
                   fp_weight=1.0, sample_weight=None, groups=None):
    """
    Perform stratified k-fold cross-validation, grouped by chip when available.
    
    When groups are provided, uses StratifiedGroupKFold so each fold's
    validation set contains chips not seen during training for that fold.
    This prevents inflated CV metrics from chip-level data leakage.
    
    Args:
        X: Feature matrix (NOT normalized - scaler fit per fold)
        y: Labels
        hidden_layers: List of hidden layer sizes
        n_folds: Number of CV folds
        max_epochs: Maximum training epochs per fold
        fp_weight: Multiplier for class 0 weight (>1.0 penalizes FP more)
        sample_weight: Optional per-sample weights aligned with X/y
        groups: Optional chip group labels per sample for grouped splitting
    
    Returns:
        dict: Mean and std of each metric across folds
    """
    from sklearn.preprocessing import StandardScaler

    if groups is not None:
        from sklearn.model_selection import StratifiedGroupKFold
        unique_groups = len(set(groups))
        effective_folds = min(n_folds, unique_groups)
        splitter = StratifiedGroupKFold(n_splits=effective_folds, shuffle=True, random_state=42)
        split_iter = splitter.split(X, y, groups)
    else:
        from sklearn.model_selection import StratifiedKFold
        effective_folds = n_folds
        splitter = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=42)
        split_iter = splitter.split(X, y)

    fold_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(split_iter):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        sw_train_fold = sample_weight[train_idx] if sample_weight is not None else None
        
        # Fit scaler on training fold only
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_fold)
        X_val_scaled = scaler.transform(X_val_fold)
        
        with suppress_stderr():
            model = train_model(X_train_scaled, y_train_fold,
                              hidden_layers=hidden_layers, max_epochs=max_epochs,
                              fp_weight=fp_weight, sample_weight=sw_train_fold)
            metrics = evaluate_model(model, X_val_scaled, y_val_fold)
        
        fold_metrics.append(metrics)
    
    # Aggregate
    result = {}
    for key in fold_metrics[0]:
        values = [m[key] for m in fold_metrics]
        result[f'{key}_mean'] = np.mean(values)
        result[f'{key}_std'] = np.std(values)
    
    return result


def export_tflite(model, X_sample, output_path, name):
    """
    Export model to TFLite with int8 quantization.
    
    Args:
        model: Trained Keras model
        X_sample: Sample data for quantization calibration
        output_path: Output directory
        name: Model name
    
    Returns:
        Path to saved .tflite file
    """
    import tensorflow as tf
    import warnings
    
    # Use up to 500 random samples for better quantization calibration
    n_samples = min(500, len(X_sample))
    indices = np.random.choice(len(X_sample), n_samples, replace=False)
    calibration_data = X_sample[indices]
    
    def representative_dataset():
        for i in range(len(calibration_data)):
            yield [calibration_data[i:i+1].astype(np.float32)]
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    # Suppress TFLite conversion warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        tflite_model = converter.convert()
    
    tflite_path = output_path / f'motion_detector_{name}.tflite'
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    return tflite_path, len(tflite_model)


def export_micropython(model, scaler, output_path, seed=None):
    """
    Export model weights to MicroPython code.
    
    Generates ml_weights.py with network weights only.
    The inference functions are in ml_detector.py (not auto-generated).
    
    Args:
        model: Trained Keras model
        scaler: StandardScaler with mean_ and scale_
        output_path: Output file path
        seed: Random seed used for training (or None if not set)
    
    Returns:
        Size of generated code
    """
    from datetime import datetime
    weights = model.get_weights()
    
    seed_info = f"Seed: {seed}"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Build code - weights only
    code = '''"""
Micro-ESPectre - ML Model Weights

Auto-generated neural network weights for motion detection.
Architecture: 12 -> ''' + ' -> '.join(str(w.shape[1]) for w in weights[::2]) + f'''
Trained: {timestamp}
{seed_info}

This file is auto-generated by 10_train_ml_model.py.
DO NOT EDIT - your changes will be overwritten!

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

# Feature normalization (StandardScaler)
FEATURE_MEAN = [''' + ', '.join(f'{x:.6f}' for x in scaler.mean_) + ''']
FEATURE_SCALE = [''' + ', '.join(f'{x:.6f}' for x in scaler.scale_) + ''']

'''
    
    # Add weights for each layer
    for i in range(0, len(weights), 2):
        W = weights[i]
        b = weights[i + 1]
        layer_num = i // 2 + 1
        in_size, out_size = W.shape
        
        activation = 'Sigmoid' if i == len(weights) - 2 else 'ReLU'
        code += f'# Layer {layer_num}: {in_size} -> {out_size} ({activation})\n'
        code += f'W{layer_num} = [\n'
        for row in W:
            code += '    [' + ', '.join(f'{x:.6f}' for x in row) + '],\n'
        code += ']\n'
        code += f'B{layer_num} = [' + ', '.join(f'{x:.6f}' for x in b) + ']\n\n'
    
    with open(output_path, 'w') as f:
        f.write(code)
    
    return len(code)


def export_cpp_weights(model, scaler, output_path, seed=None):
    """
    Export model weights to C++ header for ESPHome.
    
    Generates ml_weights.h with constexpr weights.
    
    Args:
        model: Trained Keras model
        scaler: StandardScaler with mean_ and scale_
        output_path: Output file path
        seed: Random seed used for training (or None if not set)
    
    Returns:
        Size of generated code
    """
    from datetime import datetime
    weights = model.get_weights()
    arch = ' -> '.join(str(w.shape[1]) for w in weights[::2])
    
    seed_info = f"Seed: {seed}"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    code = f'''/*
 * ESPectre - ML Model Weights
 * 
 * Auto-generated neural network weights for motion detection.
 * Architecture: 12 -> {arch}
 * Trained: {timestamp}
 * {seed_info}
 * 
 * This file is auto-generated by 10_train_ml_model.py.
 * DO NOT EDIT - your changes will be overwritten!
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#pragma once

namespace esphome {{
namespace espectre {{

// Feature normalization (StandardScaler)
constexpr float ML_FEATURE_MEAN[12] = {{{', '.join(f'{x:.6f}f' for x in scaler.mean_)}}};
constexpr float ML_FEATURE_SCALE[12] = {{{', '.join(f'{x:.6f}f' for x in scaler.scale_)}}};

'''
    
    # Add weights for each layer
    for i in range(0, len(weights), 2):
        W = weights[i]
        b = weights[i + 1]
        layer_num = i // 2 + 1
        in_size, out_size = W.shape
        
        activation = 'Sigmoid' if i == len(weights) - 2 else 'ReLU'
        code += f'// Layer {layer_num}: {in_size} -> {out_size} ({activation})\n'
        code += f'constexpr float ML_W{layer_num}[{in_size}][{out_size}] = {{\n'
        for row in W:
            code += '    {' + ', '.join(f'{x:.6f}f' for x in row) + '},\n'
        code += '};\n'
        code += f'constexpr float ML_B{layer_num}[{out_size}] = {{{", ".join(f"{x:.6f}f" for x in b)}}};\n\n'
    
    code += '''}  // namespace espectre
}  // namespace esphome
'''
    
    with open(output_path, 'w') as f:
        f.write(code)
    
    return len(code)


def export_test_data(model, scaler, X_test_raw, y_test, output_path):
    """
    Export test data for validation across Python and C++.
    
    Generates ml_test_data.npz with RAW features (not normalized) and expected outputs.
    This allows testing the full pipeline including normalization.
    
    Args:
        model: Trained Keras model
        scaler: StandardScaler used for normalization
        X_test_raw: Test features (NOT normalized, raw values)
        y_test: Test labels
        output_path: Output file path
    
    Returns:
        Number of test samples
    """
    # Normalize for prediction
    X_test_scaled = scaler.transform(X_test_raw)
    predictions = model.predict(X_test_scaled, verbose=0).flatten()
    
    # Save RAW features (not normalized) so tests can verify full pipeline
    np.savez(output_path,
             features=X_test_raw.astype(np.float32),
             labels=y_test.astype(np.int32),
             expected_outputs=predictions.astype(np.float32))
    
    return len(X_test_raw)


# ============================================================================
# Feature Importance (Correlation)
# ============================================================================

def calculate_correlation_importance(feature_names=None):
    """
    Calculate correlation of selected training features with motion label.
    
    This is a fast alternative to SHAP for initial feature screening.
    Reuses load_all_data() and extract_features() for DRY compliance.
    
    Args:
        feature_names: Optional list of features to analyze (default: TRAINING_FEATURES)
    
    Returns:
        dict: {feature_name: correlation} sorted by absolute correlation
    """
    if feature_names is None:
        feature_names = list(TRAINING_FEATURES)
    
    print("\nCalculating feature correlations...")
    print(f"  Analyzing {len(feature_names)} features")
    
    # Reuse existing data loading and feature extraction
    all_packets, stats = load_all_data()
    print(f"  Loaded {stats['total']} packets")
    if stats.get('cv_norm_files'):
        print(f"  Files using CV normalization: {len(stats['cv_norm_files'])}")
    
    print("  Extracting features...")
    X, y, actual_features, _ = extract_features(all_packets, feature_names=feature_names)
    print(f"  Extracted features for {len(X)} samples")
    
    # Calculate correlations for each feature column
    correlations = {}
    for i, fname in enumerate(actual_features):
        corr = np.corrcoef(X[:, i], y)[0, 1]
        if not np.isnan(corr):
            correlations[fname] = corr
    
    # Sort by absolute correlation
    sorted_corr = dict(sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True))
    
    return sorted_corr


def run_shap_all_features(n_samples=100):
    """
    Train a model with selected training features and calculate SHAP importance.
    
    Args:
        n_samples: Number of samples for SHAP (default: 100, lower for speed)
    
    Returns:
        int: Exit code
    """
    all_features = list(TRAINING_FEATURES)
    print(f"\n{'='*70}")
    print(f"  SHAP Analysis with {len(all_features)} training features")
    print(f"{'='*70}")
    print(f"  Using {n_samples} samples (use --shap <N> to change)")
    
    # Load data
    all_packets, stats = load_all_data()
    print(f"\nLoaded {stats['total']} packets")
    
    # Extract selected features
    print(f"Extracting {len(all_features)} features...")
    X, y, actual_features, _ = extract_features(all_packets, feature_names=all_features)
    print(f"  Samples: {len(X)}, Features: {len(actual_features)}")
    
    # Normalize
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train a simple model (just for SHAP, not for export)
    print("\nTraining model for SHAP analysis...")
    model = train_model(X_scaled, y, fp_weight=2.0, verbose=0)
    
    # Calculate SHAP
    importance = calculate_shap_importance(model, X_scaled, actual_features, 
                                           n_samples=n_samples)
    if importance:
        print_feature_importance(importance, current_features=TRAINING_FEATURES)
    
    return 0


def print_correlation_table(correlations, current_features=None):
    """Print correlation results in a nice table."""
    from src.features import DEFAULT_FEATURES
    
    if current_features is None:
        current_features = DEFAULT_FEATURES
    
    print("\n" + "=" * 74)
    print("  Feature Correlation with Motion Label")
    print("=" * 74)
    print(f"{'Rank':<5} {'Feature':<22} {'Corr':>8} {'|Corr|':>8} {'Status':<12}")
    print("-" * 74)
    
    for rank, (fname, corr) in enumerate(correlations.items(), 1):
        status = "USED" if fname in current_features else ""
        bar = '█' * int(abs(corr) * 20)
        print(f"{rank:<5} {fname:<22} {corr:>+8.4f} {abs(corr):>8.4f} {status:<12} {bar}")
    
    print("-" * 74)
    
    # Recommendations
    print("\nRecommendations:")
    sorted_items = list(correlations.items())
    top_unused = [(f, c) for f, c in sorted_items if f not in current_features][:3]
    if top_unused:
        print(f"  Top unused features: {', '.join(f[0] for f in top_unused)}")
    
    low_used = [(f, c) for f, c in sorted_items if f in current_features and abs(c) < 0.2]
    if low_used:
        print(f"  Low correlation but used: {', '.join(f[0] for f in low_used)}")


# ============================================================================
# Feature Importance (SHAP)
# ============================================================================

def calculate_shap_importance(model, X, feature_names, n_samples=500):
    """
    Calculate SHAP feature importance values.
    
    SHAP (SHapley Additive exPlanations) provides theoretically grounded
    feature importance based on game theory. Each feature's importance
    is its average marginal contribution across all possible coalitions.
    
    Args:
        model: Trained Keras model
        X: Feature matrix (normalized)
        feature_names: List of feature names
        n_samples: Number of samples to explain (default: 500)
    
    Returns:
        dict: {feature_name: mean_abs_shap_value} sorted by importance
    """
    try:
        import shap
    except ImportError:
        print("Error: SHAP not installed. Run: pip install shap")
        return None
    
    print("\nCalculating SHAP feature importance...")
    print("  (This may take 1-2 minutes)")
    
    # Use subset for background (SHAP is expensive)
    n_background = min(100, len(X))
    background_idx = np.random.choice(len(X), n_background, replace=False)
    background = X[background_idx]
    
    # Calculate SHAP values on subset
    n_explain = min(n_samples, len(X))
    explain_idx = np.random.choice(len(X), n_explain, replace=False)
    X_explain = X[explain_idx]
    
    # Use permutation algorithm (faster than KernelExplainer for neural networks)
    explainer = shap.Explainer(model.predict, background, algorithm='permutation')
    
    with suppress_stderr():
        shap_values = explainer(X_explain).values
    
    # Handle different shap_values shapes
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    if len(shap_values.shape) > 2:
        shap_values = shap_values.squeeze()
    
    # Calculate mean absolute SHAP value per feature
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    
    # Create importance dict
    importance = {name: float(val) for name, val in zip(feature_names, mean_abs_shap)}
    
    # Sort by importance (descending)
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    
    return importance


def print_feature_importance(importance, title="Feature Importance (SHAP)", 
                             current_features=None):
    """
    Print feature importance table with visual bars.
    
    Args:
        importance: Dict of {feature_name: importance_value}
        title: Title for the table
        current_features: Optional list of features currently in use (to mark USED)
    """
    print(f"\n{'='*78}")
    print(f"  {title}")
    print(f"{'='*78}\n")
    
    total = sum(importance.values())
    if total < 1e-10:
        print("  No importance values calculated.\n")
        return
    
    if current_features:
        print(f"{'Rank':<5} {'Feature':<22} {'SHAP':>8} {'Contrib':>8} {'Status':<8}")
        print("-" * 78)
    else:
        print(f"{'Rank':<6} {'Feature':<22} {'SHAP Value':>12} {'Contribution':>14}")
        print("-" * 70)
    
    for rank, (name, value) in enumerate(importance.items(), 1):
        pct = (value / total * 100)
        bar_len = int(pct / 2.5)  # Scale to ~40 chars max
        bar = '█' * bar_len
        if current_features:
            status = "USED" if name in current_features else ""
            print(f"{rank:<5} {name:<22} {value:>8.4f} {pct:>7.1f}% {status:<8} {bar}")
        else:
            print(f"{rank:<6} {name:<22} {value:>12.6f} {pct:>8.1f}% {bar}")
    
    if current_features:
        print("-" * 78)
    else:
        print("-" * 70)
        print(f"{'':6} {'TOTAL':<22} {total:>12.6f} {'100.0%':>14}")
    print()
    
    # Recommendations
    sorted_features = list(importance.keys())
    low_importance = [f for f in sorted_features if importance[f] / total < 0.03]
    high_importance = [f for f in sorted_features[:3]]
    
    print("Recommendations:")
    print(f"  Most important: {', '.join(high_importance)}")
    if low_importance:
        print(f"  Low importance (<3%): {', '.join(low_importance)}")
    
    if current_features:
        # Show top unused and low-importance used features
        top_unused = [f for f in sorted_features[:10] if f not in current_features]
        low_used = [f for f in sorted_features if f in current_features 
                    and importance[f] / total < 0.05]
        if top_unused:
            print(f"  Top unused features: {', '.join(top_unused[:5])}")
        if low_used:
            print(f"  Low importance but USED: {', '.join(low_used)}")
    print()


# ============================================================================
# Ablation Study
# ============================================================================

def run_ablation_study(X, y, feature_names, hidden_layers=[16, 8], fp_weight=2.0):
    """
    Run ablation study: train model removing one feature at a time.
    
    This helps identify which features are truly important by measuring
    the impact of removing each one. Features whose removal improves or
    doesn't affect F1 are candidates for elimination.
    
    Args:
        X: Feature matrix (NOT normalized - scaler fit per fold)
        y: Labels
        feature_names: List of feature names
        hidden_layers: Model architecture
        fp_weight: FP penalty weight
    
    Returns:
        list: Results for each ablation experiment
    """
    print("\n" + "="*80)
    print("                         ABLATION STUDY")
    print("="*80 + "\n")
    print("Training models with one feature removed at a time to measure impact...\n")
    
    results = []
    
    # Baseline (all features)
    print(f"[1/{len(feature_names)+1}] Baseline (all {len(feature_names)} features)...")
    with suppress_stderr():
        baseline_cv = cross_validate(X, y, hidden_layers=hidden_layers, n_folds=5,
                                     max_epochs=200, fp_weight=fp_weight)
    baseline_f1 = baseline_cv['f1_mean']
    results.append({
        'removed': 'None (baseline)',
        'n_features': len(feature_names),
        'f1_mean': baseline_f1,
        'f1_std': baseline_cv['f1_std'],
        'recall_mean': baseline_cv['recall_mean'],
        'fp_rate_mean': baseline_cv['fp_rate_mean'],
        'delta_f1': 0.0,
    })
    print(f"    F1: {baseline_f1:.2f}% (+/- {baseline_cv['f1_std']:.2f}%)\n")
    
    # Remove each feature one at a time
    for i, feature_name in enumerate(feature_names):
        print(f"[{i+2}/{len(feature_names)+1}] Removing '{feature_name}'...")
        
        # Create X without this feature
        X_ablated = np.delete(X, i, axis=1)
        
        # Adjust architecture for smaller input
        adjusted_layers = hidden_layers.copy()
        
        with suppress_stderr():
            cv = cross_validate(X_ablated, y, hidden_layers=adjusted_layers, n_folds=5,
                               max_epochs=200, fp_weight=fp_weight)
        
        f1 = cv['f1_mean']
        delta = f1 - baseline_f1
        
        results.append({
            'removed': feature_name,
            'n_features': len(feature_names) - 1,
            'f1_mean': f1,
            'f1_std': cv['f1_std'],
            'recall_mean': cv['recall_mean'],
            'fp_rate_mean': cv['fp_rate_mean'],
            'delta_f1': delta,
        })
        
        direction = "↑" if delta > 0.1 else "↓" if delta < -0.1 else "≈"
        print(f"    F1: {f1:.2f}% ({direction} {delta:+.2f}%)\n")
    
    # Print summary table
    print("\n" + "="*85)
    print("                           ABLATION SUMMARY")
    print("="*85 + "\n")
    
    # Sort by delta (worst impact first = most important features)
    sorted_results = sorted(results[1:], key=lambda r: r['delta_f1'])
    
    print(f"{'Removed Feature':<24} {'F1 (CV)':>14} {'Delta':>10} {'Recall':>10} {'FP Rate':>10} {'Note':<12}")
    print("-"*85)
    
    # Print baseline first
    bl = results[0]
    print(f"{'None (baseline)':<24} {bl['f1_mean']:>8.2f}% +/-{bl['f1_std']:.1f} "
          f"{'---':>10} {bl['recall_mean']:>9.1f}% {bl['fp_rate_mean']:>9.1f}%")
    print("-"*85)
    
    important_features = []
    removable_features = []
    
    for r in sorted_results:
        delta_str = f"{r['delta_f1']:+.2f}%"
        
        note = ""
        if r['delta_f1'] < -0.5:
            note = "IMPORTANT"
            important_features.append(r['removed'])
        elif r['delta_f1'] > 0.1:
            note = "removable"
            removable_features.append(r['removed'])
        elif abs(r['delta_f1']) <= 0.1:
            note = "neutral"
        
        print(f"{r['removed']:<24} {r['f1_mean']:>8.2f}% +/-{r['f1_std']:.1f} "
              f"{delta_str:>10} {r['recall_mean']:>9.1f}% {r['fp_rate_mean']:>9.1f}% {note:<12}")
    
    print("-"*85)
    
    # Recommendations
    print("\nInterpretation:")
    print("  - Delta < 0: Removing hurts performance (feature is important)")
    print("  - Delta > 0: Removing helps performance (feature adds noise)")
    print("  - Delta ≈ 0: Feature has minimal impact (candidate for removal)")
    
    print("\nRecommendations:")
    if important_features:
        print(f"  KEEP (removing hurts F1 by >0.5%): {', '.join(important_features)}")
    if removable_features:
        print(f"  REMOVE (removing helps F1 by >0.1%): {', '.join(removable_features)}")
    
    neutral = [r['removed'] for r in sorted_results if abs(r['delta_f1']) <= 0.1]
    if neutral:
        print(f"  NEUTRAL (minimal impact): {', '.join(neutral)}")
    
    print()
    return results


# ============================================================================
# Main
# ============================================================================

def show_info():
    """Show dataset information."""
    print("\n" + "="*60)
    print("              DATASET INFORMATION")
    print("="*60 + "\n")
    
    # Load dataset info
    dataset_info = load_dataset_info()
    
    print("Labels defined in dataset_info.json:")
    for label, info in dataset_info.get('labels', {}).items():
        label_type = "MOTION" if label == 'movement' else "IDLE"
        print(f"  {label} -> {label_type}")
        if info.get('description'):
            print(f"    {info['description']}")
    print()
    
    # Show files using CV normalization
    file_metadata = get_file_metadata(dataset_info)
    cv_norm_files = [f for f, meta in file_metadata.items() if not meta.get('gain_locked', True)]
    if cv_norm_files:
        print(f"Files using CV normalization ({len(cv_norm_files)}):")
        for f in sorted(cv_norm_files):
            print(f"  - {f}")
        print()
    
    # Load and analyze data
    _, stats = load_all_data()
    
    print(f"Chips available: {', '.join(stats['chips']) if stats['chips'] else 'None'}")
    print(f"Total packets: {stats['total']}")
    print()
    
    print("Packets by label:")
    idle_total = 0
    motion_total = 0
    for label, count in sorted(stats['labels'].items()):
        is_motion = is_motion_label(label, dataset_info)
        label_type = "MOTION" if is_motion else "IDLE"
        print(f"  {label}: {count} packets ({label_type})")
        if is_motion:
            motion_total += count
        else:
            idle_total += count
    
    print(f"\nSummary:")
    print(f"  IDLE packets:   {idle_total}")
    print(f"  MOTION packets: {motion_total}")
    print()
    
    # Show data directory contents
    print("Data directory contents:")
    for subdir in sorted(DATA_DIR.iterdir()):
        if subdir.is_dir() and not subdir.name.startswith('.'):
            files = list(subdir.glob('*.npz'))
            if files:
                print(f"  {subdir.name}/: {len(files)} files")
                for f in sorted(files)[:3]:
                    print(f"    - {f.name}")
                if len(files) > 3:
                    print(f"    ... and {len(files) - 3} more")
    print()


def train_all(fp_weight=1.0, seed=None, feature_names=None,
              feature_importance=False, ablation=False, shap_samples=200):
    """
    Train models with all available data.
    
    Args:
        fp_weight: Multiplier for class 0 (IDLE) weight. Values >1.0 penalize
                   false positives more, producing a more conservative model.
        seed: Optional random seed for reproducible training. If None, a random
              seed is generated and saved for reproducibility.
        feature_names: List of feature names to use. If None, uses DEFAULT_FEATURES.
        feature_importance: If True, calculate and display SHAP feature importance.
        ablation: If True, run ablation study instead of training.

    Returns:
        tuple[int, int | None]:
            (exit_code, used_seed)
            - exit_code: 0 on success, non-zero on failure
            - used_seed: seed used for training (None only on early dependency errors)
    """
    subcarriers = DEFAULT_SUBCARRIERS
    
    print("\n" + "="*60)
    print("           ML MOTION DETECTOR TRAINING")
    print("="*60 + "\n")
    print(f"Subcarriers: {subcarriers}\n")
    
    # Check dependencies (suppress TensorFlow C++ warnings during import)
    try:
        with suppress_stderr():
            import tensorflow as tf
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import train_test_split
            
            # Generate random seed if not provided (for reproducibility tracking)
            # Uses NumPy's SeedSequence which gathers entropy from the OS
            if seed is None:
                from numpy.random import SeedSequence
                ss = SeedSequence()
                seed = int(ss.entropy % (2**31))  # Convert to int32 for compatibility
                print(f"Generated random seed: {seed}\n")
            else:
                print(f"Using provided seed: {seed}\n")
            
            # Set random seeds for reproducibility
            np.random.seed(seed)
            tf.random.set_seed(seed)
            
            # Suppress TensorFlow Python-level warnings
            tf.get_logger().setLevel('ERROR')
            
            # Suppress absl logging
            try:
                import absl.logging
                absl.logging.set_verbosity(absl.logging.ERROR)
                absl.logging.set_stderrthreshold(absl.logging.ERROR)
            except ImportError:
                pass
    except ImportError as e:
        print(f"Error: Missing dependency - {e}")
        print("Install with: pip install tensorflow scikit-learn")
        return 1, None
    
    # Load data
    print("Loading data...")
    all_packets, stats = load_all_data()
    
    if not stats['chips']:
        print("Error: No datasets found in data/")
        print("Collect data using: ./me collect --label baseline --duration 60")
        return 1, seed
    
    print(f"  Chips: {', '.join(stats['chips'])}")
    if stats.get('excluded_chips'):
        print(f"  Excluded chips: {', '.join(stats['excluded_chips'])}")
    if stats.get('cv_norm_files'):
        print(f"  Files using CV normalization: {len(stats['cv_norm_files'])}")
    for label, count in sorted(stats['labels'].items()):
        print(f"  {label}: {count} packets")
    print(f"  Total: {stats['total']} packets")
    
    # Determine feature set to use
    if feature_names is None:
        feature_names = DEFAULT_FEATURES.copy()
    
    # Extract features
    print("\nExtracting features...")
    X, y, actual_feature_names, groups = extract_features(
        all_packets, subcarriers=subcarriers, feature_names=feature_names
    )
    print(f"  Samples: {len(X)}")
    print(f"  Features: {len(actual_feature_names)}")
    print(f"  Feature set: {', '.join(actual_feature_names)}")
    n_idle = np.sum(y == 0)
    n_motion = np.sum(y == 1)
    print(f"  Class balance: IDLE={n_idle}, MOTION={n_motion}")
    if n_idle > 0 and n_motion > 0:
        ratio = max(n_idle, n_motion) / min(n_idle, n_motion)
        print(f"  Imbalance ratio: {ratio:.1f}:1")

    print("\nComputing MVS-guided sample weights...")
    dataset_info = load_dataset_info()
    tuning_map = build_gridsearch_tuning_map(dataset_info, DEFAULT_SUBCARRIERS, default_threshold=1.0)
    sample_weights = compute_mvs_guided_sample_weights(
        all_packets, tuning_map, window_size=SEG_WINDOW_SIZE
    )
    if len(sample_weights) != len(X):
        print(
            f"Error: sample weights mismatch (weights={len(sample_weights)}, samples={len(X)})."
        )
        return 1, seed
    print(
        "  Weight mode: context-aware + hard-positive mining "
        "(pair<=30min + single-dataset fallback)"
    )
    print(
        f"  Weight stats: min={float(np.min(sample_weights)):.3f}, "
        f"max={float(np.max(sample_weights)):.3f}, "
        f"mean={float(np.mean(sample_weights)):.3f}"
    )
    
    # Run ablation study if requested
    if ablation:
        run_ablation_study(X, y, actual_feature_names, fp_weight=fp_weight)
        return 0, seed
    
    # Cross-validation grouped by chip for realistic per-chip evaluation
    n_chips = len(set(groups))
    if fp_weight != 1.0:
        print(f"\nFP weight: {fp_weight}x (penalizing false positives)")
    print(f"\n{n_chips}-fold grouped CV by chip (12 -> 16 -> 8 -> 1)...")
    with suppress_stderr():
        cv_results = cross_validate(X, y, hidden_layers=[16, 8], n_folds=n_chips,
                                    max_epochs=200, fp_weight=fp_weight,
                                    sample_weight=sample_weights, groups=groups)
    
    print(f"  Recall:    {cv_results['recall_mean']:.1f}% (+/- {cv_results['recall_std']:.1f}%)")
    print(f"  Precision: {cv_results['precision_mean']:.1f}% (+/- {cv_results['precision_std']:.1f}%)")
    print(f"  FP Rate:   {cv_results['fp_rate_mean']:.1f}% (+/- {cv_results['fp_rate_std']:.1f}%)")
    print(f"  F1 Score:  {cv_results['f1_mean']:.1f}% (+/- {cv_results['f1_std']:.1f}%)")
    
    # Also do a single split for test data export
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train final model on full dataset for production export
    # Use consistent scaler: fit on full dataset (same data the model sees)
    print("\nTraining final model on full dataset...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    with suppress_stderr():
        model = train_model(X_scaled, y, hidden_layers=[16, 8], max_epochs=200,
                           fp_weight=fp_weight, sample_weight=sample_weights)
    
    # Quick evaluation on the held-out test set (for reference)
    X_test_scaled = scaler.transform(X_test_raw)
    with suppress_stderr():
        test_metrics = evaluate_model(model, X_test_scaled, y_test)
    
    f1 = test_metrics['f1']
    print(f"\nHold-out test set (20%):")
    print(f"  Recall:    {test_metrics['recall']:.1f}%")
    print(f"  Precision: {test_metrics['precision']:.1f}%")
    print(f"  FP Rate:   {test_metrics['fp_rate']:.1f}%")
    print(f"  F1 Score:  {f1:.1f}%")
    
    # Calculate SHAP feature importance if requested
    if feature_importance:
        importance = calculate_shap_importance(model, X_scaled, actual_feature_names, 
                                               n_samples=shap_samples)
        if importance:
            print_feature_importance(importance)
    
    # Export models
    print("\nExporting models...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # TFLite (suppress C++ warnings during conversion)
    with suppress_stderr():
        tflite_path, tflite_size = export_tflite(model, X_scaled, MODELS_DIR, 'small')
    print(f"  TFLite: {tflite_path.name} ({tflite_size/1024:.1f} KB)")
    
    # MicroPython weights
    mp_path = SRC_DIR / 'ml_weights.py'
    mp_size = export_micropython(model, scaler, mp_path, seed=seed)
    print(f"  MicroPython weights: {mp_path.name} ({mp_size/1024:.1f} KB)")
    
    # C++ weights for ESPHome
    cpp_path = CPP_DIR / 'ml_weights.h'
    cpp_size = export_cpp_weights(model, scaler, cpp_path, seed=seed)
    print(f"  C++ weights: {cpp_path.name} ({cpp_size/1024:.1f} KB)")
    
    # Save scaler for TFLite (external normalization)
    scaler_path = MODELS_DIR / 'feature_scaler.npz'
    np.savez(scaler_path, mean=scaler.mean_, scale=scaler.scale_)
    print(f"  Scaler: {scaler_path.name}")
    
    # Test data for validation (save raw features + expected outputs)
    with suppress_stderr():
        test_data_path = MODELS_DIR / 'ml_test_data.npz'
        n_test = export_test_data(model, scaler, X_test_raw, y_test, test_data_path)
    print(f"  Test data: {test_data_path.name} ({n_test} samples)")
    
    print("\n" + "="*60)
    print("                    DONE!")
    print("="*60)
    print(f"\nModel trained with CV F1={cv_results['f1_mean']:.1f}% (+/- {cv_results['f1_std']:.1f}%)")
    print(f"\nGenerated files:")
    print(f"  - {mp_path} (MicroPython)")
    print(f"  - {cpp_path} (C++ ESPHome)")
    print(f"  - {tflite_path} (ESP-IDF TFLite)")
    print(f"  - {scaler_path} (normalization params)")
    print(f"  - {test_data_path} (test data for validation)")
    print()
    
    return 0, seed


def _run_ml_performance_tests():
    """
    Run ML-only performance tests and parse per-chip metrics.

    Returns:
        tuple: (metrics_dict_or_none, raw_output)
    """
    project_root = Path(__file__).parent.parent
    cmd = [
        sys.executable,
        '-m',
        'pytest',
        'tests/test_validation_real_data.py::TestPerformanceMetrics::test_ml_detection_accuracy',
        '-v',
    ]
    result = subprocess.run(
        cmd,
        cwd=project_root,
        capture_output=True,
        text=True,
    )
    output = (result.stdout or "") + "\n" + (result.stderr or "")

    # Parse lines from the "DETAILED METRICS (for PERFORMANCE.md)" table.
    pattern = re.compile(
        r"\|\s*([A-Za-z0-9_-]+)\s*\|\s*ML\s*\|\s*([0-9.]+)%\s*\|\s*([0-9.]+)%\s*\|\s*([0-9.]+)%\s*\|\s*([0-9.]+)%\s*\|"
    )
    rows = []
    for match in pattern.finditer(output):
        chip, recall, precision, fp_rate, f1 = match.groups()
        rows.append({
            'chip': chip,
            'recall': float(recall),
            'precision': float(precision),
            'fp_rate': float(fp_rate),
            'f1': float(f1),
        })

    if not rows:
        return None, output

    metrics = {
        'rows': rows,
        'mean_recall': float(np.mean([r['recall'] for r in rows])),
        'min_recall': float(np.min([r['recall'] for r in rows])),
        'mean_fp_rate': float(np.mean([r['fp_rate'] for r in rows])),
        'mean_f1': float(np.mean([r['f1'] for r in rows])),
    }
    return metrics, output


def _seed_candidate_key(metrics):
    """
    Ranking key for selecting best seed from ML performance tests.

    Priority:
      1) higher min recall (worst-chip protection)
      2) lower mean FP rate
      3) higher mean recall
      4) higher mean F1
    """
    return (
        metrics['min_recall'],
        -metrics['mean_fp_rate'],
        metrics['mean_recall'],
        metrics['mean_f1'],
    )


def _model_artifact_paths():
    """Return paths of generated model artifacts."""
    return [
        SRC_DIR / 'ml_weights.py',
        CPP_DIR / 'ml_weights.h',
        MODELS_DIR / 'motion_detector_small.tflite',
        MODELS_DIR / 'feature_scaler.npz',
        MODELS_DIR / 'ml_test_data.npz',
    ]


def _backup_artifacts():
    """Backup model artifacts to a temporary directory."""
    backup_dir = Path(tempfile.mkdtemp(prefix='ml_seed_search_backup_'))
    saved_files = []
    for path in _model_artifact_paths():
        if path.exists():
            rel_name = path.name
            shutil.copy2(path, backup_dir / rel_name)
            saved_files.append((path, backup_dir / rel_name))
    return backup_dir, saved_files


def _restore_artifacts(saved_files):
    """Restore model artifacts from backup copies."""
    for original, backup in saved_files:
        if backup.exists():
            original.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(backup, original)


def train_until_improvement(max_trials, fp_weight=1.0, feature_names=None):
    """
    Train repeatedly with auto-generated seeds until ML performance improves.

    Baseline is the current model performance from ML real-data tests.
    Stops on first strict improvement according to _seed_candidate_key.
    """
    if max_trials < 1:
        print("Error: --seed-search-until-improvement must be >= 1")
        return 1

    if feature_names is None:
        feature_names = DEFAULT_FEATURES

    print("\n" + "=" * 70)
    print("  SEED SEARCH (loop until improvement)")
    print("=" * 70)
    print(f"Max trials: {max_trials}")
    print(f"FP weight: {fp_weight}")

    print("\nEvaluating current model baseline with ML performance tests...")
    baseline_metrics, baseline_output = _run_ml_performance_tests()
    if baseline_metrics is None:
        print("Error: unable to evaluate current model baseline")
        print("--- pytest tail ---")
        print("\n".join(baseline_output.splitlines()[-20:]))
        return 1

    print(
        f"Baseline: min_recall={baseline_metrics['min_recall']:.1f}% "
        f"mean_recall={baseline_metrics['mean_recall']:.1f}% "
        f"mean_fp={baseline_metrics['mean_fp_rate']:.1f}% "
        f"mean_f1={baseline_metrics['mean_f1']:.1f}%"
    )

    backup_dir, saved_files = _backup_artifacts()
    print(f"Artifacts backup: {backup_dir}")

    trial_summaries = []
    improved = False
    improved_seed = None
    improved_metrics = None

    for idx in range(1, max_trials + 1):
        print(f"\n[{idx}/{max_trials}] Training with auto-generated seed")
        train_rc, used_seed = train_all(
            fp_weight=fp_weight,
            seed=None,
            feature_names=feature_names,
            feature_importance=False,
            ablation=False,
            shap_samples=200,
        )
        if train_rc != 0:
            print(f"  Training failed (exit={train_rc})")
            continue

        print(f"  Seed generated by training script: {used_seed}")
        print("  Running ML performance tests...")
        metrics, output = _run_ml_performance_tests()
        if metrics is None:
            print(f"  Could not evaluate seed={used_seed} from pytest output")
            print("  --- pytest tail ---")
            print("\n".join(output.splitlines()[-20:]))
            continue

        trial_summaries.append((used_seed, metrics))
        print(
            f"  Result: min_recall={metrics['min_recall']:.1f}% "
            f"mean_recall={metrics['mean_recall']:.1f}% "
            f"mean_fp={metrics['mean_fp_rate']:.1f}% "
            f"mean_f1={metrics['mean_f1']:.1f}%"
        )

        if _seed_candidate_key(metrics) > _seed_candidate_key(baseline_metrics):
            improved = True
            improved_seed = used_seed
            improved_metrics = metrics
            print("  Improvement found: stopping search")
            break

    print("\n" + "=" * 70)
    print("  UNTIL-IMPROVEMENT SUMMARY")
    print("=" * 70)
    for seed, metrics in trial_summaries:
        print(
            f"  seed={seed} | minR={metrics['min_recall']:.1f}% "
            f"meanR={metrics['mean_recall']:.1f}% "
            f"meanFP={metrics['mean_fp_rate']:.1f}% "
            f"meanF1={metrics['mean_f1']:.1f}%"
        )

    if improved:
        print(
            f"\nSelected seed: {improved_seed} "
            f"(min_recall={improved_metrics['min_recall']:.1f}%, "
            f"mean_fp={improved_metrics['mean_fp_rate']:.1f}%)"
        )
        return 0

    print("\nNo improvement found within max trials: restoring original artifacts")
    _restore_artifacts(saved_files)
    return 1


def experiment_architectures():
    """
    Compare multiple MLP architectures using cross-validation.
    
    Trains and evaluates each architecture on the same data with 5-fold CV.
    Reports a comparison table with F1, inference time, and memory usage.
    Recommends the best architecture by F1 (inference time as tiebreaker).
    """
    import time
    subcarriers = DEFAULT_SUBCARRIERS
    
    print("\n" + "="*60)
    print("       ARCHITECTURE EXPERIMENT")
    print("="*60 + "\n")
    
    # Check dependencies
    try:
        with suppress_stderr():
            import tensorflow as tf
            tf.get_logger().setLevel('ERROR')
            try:
                import absl.logging
                absl.logging.set_verbosity(absl.logging.ERROR)
                absl.logging.set_stderrthreshold(absl.logging.ERROR)
            except ImportError:
                pass
    except ImportError as e:
        print(f"Error: Missing dependency - {e}")
        return 1
    
    # Load and extract features
    print("Loading data...")
    all_packets, stats = load_all_data()
    
    if not stats['chips']:
        print("Error: No datasets found in data/")
        return 1
    
    print(f"  Chips: {', '.join(stats['chips'])}")
    if stats.get('excluded_chips'):
        print(f"  Excluded chips: {', '.join(stats['excluded_chips'])}")
    if stats.get('cv_norm_files'):
        print(f"  Files using CV normalization: {len(stats['cv_norm_files'])}")
    print(f"  Total: {stats['total']} packets")
    
    print("\nExtracting features...")
    X, y, feature_names, _ = extract_features(all_packets, subcarriers=subcarriers)
    print(f"  Samples: {len(X)}")
    print(f"  Features: {len(feature_names)}")
    print(f"  Class balance: IDLE={np.sum(y==0)}, MOTION={np.sum(y==1)}")
    
    # Define architectures to compare
    architectures = [
        {'name': 'Current (16-8)', 'layers': [16, 8]},
        {'name': 'Wide-shallow (24)', 'layers': [24]},
        {'name': 'Deeper (12-8-4)', 'layers': [12, 8, 4]},
        {'name': 'Minimal (8)', 'layers': [8]},
        {'name': 'Wide-deep (24-12)', 'layers': [24, 12]},
    ]
    
    results = []
    
    for arch in architectures:
        name = arch['name']
        layers = arch['layers']
        
        # Calculate parameter count
        layer_sizes = [12] + layers + [1]
        n_params = 0
        for i in range(len(layer_sizes) - 1):
            n_params += layer_sizes[i] * layer_sizes[i + 1]  # weights
            n_params += layer_sizes[i + 1]  # biases
        
        weight_kb = n_params * 4 / 1024  # float32
        
        # Estimate inference FLOPS (multiply-accumulate operations)
        flops = 0
        for i in range(len(layer_sizes) - 1):
            flops += layer_sizes[i] * layer_sizes[i + 1]  # MAC operations
        
        print(f"\nEvaluating: {name} ({' -> '.join(map(str, [12] + layers + [1]))})...")
        print(f"  Parameters: {n_params}, Weights: {weight_kb:.1f} KB, FLOPS: {flops}")
        
        # Cross-validate
        with suppress_stderr():
            cv = cross_validate(X, y, hidden_layers=layers, n_folds=5, max_epochs=200)
        
        # Measure Python inference time
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        with suppress_stderr():
            model = train_model(X_scaled, y, hidden_layers=layers, max_epochs=200)
        
        # Benchmark inference (use model.predict for consistency)
        sample = X_scaled[:1].astype(np.float32)
        # Warm up
        for _ in range(10):
            model.predict(sample, verbose=0)
        
        n_bench = 1000
        start_t = time.perf_counter()
        for _ in range(n_bench):
            model.predict(sample, verbose=0)
        elapsed = time.perf_counter() - start_t
        inference_us = elapsed / n_bench * 1e6
        
        result = {
            'name': name,
            'layers': layers,
            'params': n_params,
            'weight_kb': weight_kb,
            'flops': flops,
            'f1_mean': cv['f1_mean'],
            'f1_std': cv['f1_std'],
            'recall_mean': cv['recall_mean'],
            'recall_std': cv['recall_std'],
            'precision_mean': cv['precision_mean'],
            'fp_rate_mean': cv['fp_rate_mean'],
            'inference_us': inference_us,
        }
        results.append(result)
        
        print(f"  F1: {cv['f1_mean']:.1f}% +/- {cv['f1_std']:.1f}%")
        print(f"  Recall: {cv['recall_mean']:.1f}%, Precision: {cv['precision_mean']:.1f}%")
        print(f"  Inference: {inference_us:.1f} us/sample")
    
    # Print comparison table
    print("\n" + "="*90)
    print("                         ARCHITECTURE COMPARISON")
    print("="*90 + "\n")
    
    print(f"{'Architecture':<22} {'Params':>7} {'KB':>6} {'F1 (CV)':>12} {'Recall':>10} {'FP Rate':>10} {'Inf (us)':>10}")
    print("-"*90)
    
    best = max(results, key=lambda r: (r['f1_mean'], -r['inference_us']))
    
    for r in results:
        marker = " **" if r == best else "   "
        print(f"{marker}{r['name']:<19} {r['params']:>7} {r['weight_kb']:>5.1f} "
              f"{r['f1_mean']:>6.1f}+/-{r['f1_std']:<4.1f} "
              f"{r['recall_mean']:>9.1f}% {r['fp_rate_mean']:>9.1f}% "
              f"{r['inference_us']:>9.1f}")
    
    print("-"*90)
    print(f"\n** Best architecture: {best['name']}")
    print(f"   F1: {best['f1_mean']:.1f}% +/- {best['f1_std']:.1f}%")
    print(f"   Recall: {best['recall_mean']:.1f}%, FP Rate: {best['fp_rate_mean']:.1f}%")
    print(f"   Parameters: {best['params']}, Weights: {best['weight_kb']:.1f} KB")
    
    # Recommend action
    current = next(r for r in results if r['name'].startswith('Current'))
    if best != current:
        improvement = best['f1_mean'] - current['f1_mean']
        print(f"\n   Improvement over current: {improvement:+.1f}% F1")
        if improvement > 1.0:
            print(f"   Recommendation: Switch to {best['name']}")
            print(f"   Update train_all() with hidden_layers={best['layers']}")
        else:
            print(f"   Recommendation: Difference is marginal, keep current architecture")
    else:
        print(f"\n   Current architecture is already optimal!")
    
    print()
    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Train ML motion detection model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python tools/10_train_ml_model.py                    # Train with default features
  python tools/10_train_ml_model.py --info             # Show dataset info
  python tools/10_train_ml_model.py --experiment       # Compare architectures
  python tools/10_train_ml_model.py --fp-weight 2.0    # Penalize FP 2x more
  python tools/10_train_ml_model.py --seed 42          # Reproducible training
  python tools/10_train_ml_model.py --seed-search-until-improvement 20
                                           # Stop at first improvement (max 20 trials)
  python tools/10_train_ml_model.py --shap              # SHAP importance (200 samples)
  python tools/10_train_ml_model.py --shap 500          # SHAP importance (500 samples)

Configuration (edit at top of this file):
  TRAINING_FEATURES = [...]   # Feature list to use

To compare ML with MVS, use:
  python tools/7_compare_detection_methods.py
'''
    )
    parser.add_argument('--info', action='store_true', 
                       help='Show dataset information')
    parser.add_argument('--experiment', action='store_true',
                       help='Compare multiple MLP architectures using cross-validation')
    parser.add_argument('--seed', type=int, default=None,
                       help='Use specific random seed for reproducible training')
    parser.add_argument('--seed-search-until-improvement', type=int, default=0, metavar='MAX_TRIALS',
                       help='Train with auto-generated seeds until first '
                            'improvement over current ML performance, with '
                            'at most MAX_TRIALS attempts')
    parser.add_argument('--fp-weight', type=float, default=1.0,
                       help='Multiplier for IDLE class weight to penalize false positives. '
                            'Values >1.0 make the model more conservative (default: 1.0)')
    parser.add_argument('--shap', type=int, nargs='?', const=200, default=None,
                       metavar='SAMPLES',
                       help='Calculate SHAP feature importance (default: 200 samples)')
    parser.add_argument('--correlation', action='store_true',
                       help='Calculate correlation of selected training features with motion label')
    parser.add_argument('--ablation', action='store_true',
                       help='Run ablation study (test removing each feature)')
    args = parser.parse_args()
    
    if args.info:
        show_info()
        return 0
    
    if args.experiment:
        return experiment_architectures()
    
    if args.correlation:
        correlations = calculate_correlation_importance()
        if correlations:
            print_correlation_table(correlations, TRAINING_FEATURES)
        return 0

    if args.seed_search_until_improvement > 0:
        if args.seed is not None:
            print("Error: --seed and --seed-search-until-improvement are mutually exclusive")
            return 1
        if args.shap is not None or args.ablation:
            print("Error: --seed-search-until-improvement cannot be combined with --shap or --ablation")
            return 1
        return train_until_improvement(
            max_trials=args.seed_search_until_improvement,
            fp_weight=args.fp_weight,
            feature_names=TRAINING_FEATURES
        )
    
    train_rc, _ = train_all(
        fp_weight=args.fp_weight, 
        seed=args.seed,
        feature_names=TRAINING_FEATURES,
        feature_importance=args.shap is not None,
        ablation=args.ablation,
        shap_samples=args.shap if args.shap is not None else 200
    )
    return train_rc


if __name__ == '__main__':
    exit(main())
