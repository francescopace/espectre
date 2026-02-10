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
    python tools/10_train_ml_model.py                    # Train (excludes control sets automatically)
    python tools/10_train_ml_model.py --info              # Show dataset info
    python tools/10_train_ml_model.py --experiment        # Compare architectures
    python tools/10_train_ml_model.py --fp-weight 2.0    # Penalize FP 2x more
    python tools/10_train_ml_model.py --include-all-chips # Include all chips (even ESP32)

Note: Files marked as "excluded from ML training" in dataset_info.json are
automatically skipped. This includes control sets and datasets collected
without gain lock.

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
from pathlib import Path
from collections import deque
from contextlib import contextmanager


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
    load_baseline_and_movement,
    load_npz_as_packets,
    DATA_DIR,
    DEFAULT_SUBCARRIERS,
)
from segmentation import SegmentationContext
from features import (
    calc_skewness, calc_kurtosis, calc_entropy_turb,
    calc_zero_crossing_rate, calc_autocorrelation, calc_mad,
)

# Directories
MODELS_DIR = Path(__file__).parent.parent / 'models'
SRC_DIR = Path(__file__).parent.parent / 'src'
CPP_DIR = Path(__file__).parent.parent.parent / 'components' / 'espectre'

# Feature names (12 features - all non-redundant, all turbulence-based)
FEATURE_NAMES = [
    'turb_mean', 'turb_std', 'turb_max', 'turb_min', 'turb_zcr', 'turb_skewness',
    'turb_kurtosis', 'turb_entropy', 'turb_autocorr', 'turb_mad', 'turb_slope', 'turb_delta'
]


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


def is_motion_label(label_name, dataset_info):
    """
    Determine if a label represents motion or idle.
    
    Uses dataset_info.json to map labels:
    - label_id 1 = MOTION
    - label_id 0, 2, ... = IDLE (baseline, baseline_noisy, etc.)
    
    Args:
        label_name: Label name from npz file
        dataset_info: Loaded dataset_info.json
    
    Returns:
        bool: True if motion, False if idle
    """
    labels = dataset_info.get('labels', {})
    if label_name in labels:
        return labels[label_name].get('id') == 1
    # Default: only 'movement' is motion
    return label_name == 'movement'


def get_file_metadata(dataset_info):
    """
    Get metadata for all files in dataset_info.json.
    
    Returns a dict mapping filename to metadata including:
    - use_cv_normalization: Whether to use CV normalization for this file
    
    Args:
        dataset_info: Loaded dataset_info.json
    
    Returns:
        dict: {filename: {use_cv_normalization: bool, ...}}
    """
    file_metadata = {}
    files_by_label = dataset_info.get('files', {})
    for label, file_list in files_by_label.items():
        for file_info in file_list:
            filename = file_info.get('filename', '')
            if filename:
                file_metadata[filename] = {
                    'use_cv_normalization': file_info.get('use_cv_normalization', False),
                    'chip': file_info.get('chip', 'unknown'),
                }
    return file_metadata


def load_all_data(exclude_chips=None):
    """
    Load all available CSI data from the data/ directory.
    
    Reads label from npz file metadata (not folder structure).
    Uses dataset_info.json to determine if label is motion or idle.
    Sets use_cv_normalization flag on each packet based on dataset_info.json.
    
    Args:
        exclude_chips: Optional list of chip names to exclude (e.g. ['ESP32'])
    
    Returns:
        tuple: (all_packets, stats) where stats is a dict with dataset info
    """
    all_packets = []
    stats = {'chips': set(), 'labels': {}, 'total': 0, 'excluded_chips': set(), 'cv_norm_files': set()}
    
    # Normalize excluded chips to uppercase for comparison
    excluded = set(c.upper() for c in (exclude_chips or []))
    
    # Load dataset info for label mapping and file metadata
    dataset_info = load_dataset_info()
    file_metadata = get_file_metadata(dataset_info)
    
    # Scan all subdirectories in data/
    # Exclude baseline_noisy - has very different characteristics that distort the scaler
    excluded_dirs = {'.'}
    for subdir in DATA_DIR.iterdir():
        if not subdir.is_dir() or subdir.name in excluded_dirs:
            continue
        
        # Load all npz files in this directory
        for npz_file in subdir.glob('*.npz'):
            try:
                packets = load_npz_as_packets(npz_file)
                if not packets:
                    continue
                
                # Get label from file metadata (already set by load_npz_as_packets)
                label = packets[0].get('label', subdir.name)
                
                # Get chip
                chip = packets[0].get('chip', 'unknown').upper()
                
                # Skip excluded chips
                if chip in excluded:
                    stats['excluded_chips'].add(chip)
                    continue
                
                # Track stats
                if label not in stats['labels']:
                    stats['labels'][label] = 0
                stats['labels'][label] += len(packets)
                stats['total'] += len(packets)
                
                stats['chips'].add(chip)
                
                # Get file-specific metadata
                meta = file_metadata.get(npz_file.name, {})
                use_cv_norm = meta.get('use_cv_normalization', False)
                if use_cv_norm:
                    stats['cv_norm_files'].add(npz_file.name)
                
                # Add flags to each packet
                is_motion = is_motion_label(label, dataset_info)
                for p in packets:
                    p['is_motion'] = is_motion
                    p['use_cv_normalization'] = use_cv_norm
                
                all_packets.extend(packets)
                
            except Exception as e:
                print(f"  Warning: Could not load {npz_file.name}: {e}")
    
    stats['chips'] = sorted(stats['chips'])
    stats['excluded_chips'] = sorted(stats['excluded_chips'])
    stats['cv_norm_files'] = sorted(stats['cv_norm_files'])
    return all_packets, stats


# ============================================================================
# Feature Extraction
# ============================================================================

def extract_features(packets, window_size=50, subcarriers=None):
    """
    Extract features from CSI packets using sliding window.
    
    Args:
        packets: List of CSI packets with 'csi_data' and 'label'
        window_size: Sliding window size (default: 50)
        subcarriers: List of subcarrier indices to use (default: DEFAULT_SUBCARRIERS)
    
    Returns:
        tuple: (X, y) feature matrix and labels
    """
    if subcarriers is None:
        subcarriers = DEFAULT_SUBCARRIERS
    
    X, y = [], []
    turb_buffer = deque(maxlen=window_size)
    
    for pkt in packets:
        # Calculate turbulence, using CV normalization for files that need it
        # (collected without gain lock, e.g. ESP32 or C3 without gain lock)
        use_cv_norm = pkt.get('use_cv_normalization', False)
        turb, amps = SegmentationContext.compute_spatial_turbulence(
            pkt['csi_data'], subcarriers, use_cv_normalization=use_cv_norm
        )
        turb_buffer.append(turb)
        
        # Wait for buffer to fill
        if len(turb_buffer) < window_size:
            continue
        
        turb_list = list(turb_buffer)
        n = len(turb_list)
        turb_mean = np.mean(turb_list)
        turb_std = np.std(turb_list)
        turb_var = turb_std * turb_std
        
        # Extract 12 features (all non-redundant, all turbulence-based)
        features = [
            turb_mean,                                              # turb_mean
            turb_std,                                               # turb_std
            np.max(turb_list),                                      # turb_max
            np.min(turb_list),                                      # turb_min
            calc_zero_crossing_rate(turb_list, n, mean=turb_mean),  # turb_zcr
            calc_skewness(turb_list, n, turb_mean, turb_std),       # turb_skewness
            calc_kurtosis(turb_list, n, turb_mean, turb_std),       # turb_kurtosis
            calc_entropy_turb(turb_list, n),                        # turb_entropy
            calc_autocorrelation(turb_list, n, mean=turb_mean, variance=turb_var),  # turb_autocorr
            calc_mad(turb_list, n),                                 # turb_mad
            np.polyfit(range(n), turb_list, 1)[0],                  # turb_slope
            turb_list[-1] - turb_list[0],                           # turb_delta
        ]
        
        X.append(features)
        # Label: 0 = IDLE, 1 = MOTION (from metadata)
        y.append(1 if pkt.get('is_motion', False) else 0)
    
    return np.array(X), np.array(y)


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
                class_weight=None, fp_weight=1.0, verbose=0):
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
    
    model = build_model(hidden_layers=hidden_layers, use_dropout=use_dropout)
    
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
        X, y,
        epochs=max_epochs,
        batch_size=32,
        validation_split=0.1,
        class_weight=class_weight,
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
                   fp_weight=1.0):
    """
    Perform stratified k-fold cross-validation.
    
    Args:
        X: Feature matrix (NOT normalized - scaler fit per fold)
        y: Labels
        hidden_layers: List of hidden layer sizes
        n_folds: Number of CV folds
        max_epochs: Maximum training epochs per fold
        fp_weight: Multiplier for class 0 weight (>1.0 penalizes FP more)
    
    Returns:
        dict: Mean and std of each metric across folds
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Fit scaler on training fold only
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_fold)
        X_val_scaled = scaler.transform(X_val_fold)
        
        with suppress_stderr():
            model = train_model(X_train_scaled, y_train_fold,
                              hidden_layers=hidden_layers, max_epochs=max_epochs,
                              fp_weight=fp_weight)
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
    
    seed_info = f"Seed: {seed}" if seed is not None else "Seed: not set (non-deterministic)"
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
    
    seed_info = f"Seed: {seed}" if seed is not None else "Seed: not set (non-deterministic)"
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
        label_type = "MOTION" if info.get('id') == 1 else "IDLE"
        print(f"  {label} (id={info.get('id')}) -> {label_type}")
        if info.get('description'):
            print(f"    {info['description']}")
    print()
    
    # Show files using CV normalization
    file_metadata = get_file_metadata(dataset_info)
    cv_norm_files = [f for f, meta in file_metadata.items() if meta.get('use_cv_normalization')]
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


def train_all(fp_weight=2.0, exclude_chips=None, seed=None):
    """
    Train models with all available data.
    
    Args:
        fp_weight: Multiplier for class 0 (IDLE) weight. Values >1.0 penalize
                   false positives more, producing a more conservative model.
        exclude_chips: Optional list of chip names to exclude from training
        seed: Optional random seed for reproducible training
    """
    from ml_detector import ML_SUBCARRIERS
    subcarriers = ML_SUBCARRIERS
    
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
            
            # Set random seeds for reproducibility if specified
            if seed is not None:
                print(f"Using random seed: {seed}\n")
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
        return 1
    
    # Load data
    if exclude_chips:
        print(f"Loading data (excluding chips: {', '.join(exclude_chips)})...")
    else:
        print("Loading data from npz metadata...")
    all_packets, stats = load_all_data(exclude_chips=exclude_chips)
    
    if not stats['chips']:
        print("Error: No datasets found in data/")
        print("Collect data using: ./me collect --label baseline --duration 60")
        return 1
    
    print(f"  Chips: {', '.join(stats['chips'])}")
    if stats.get('excluded_chips'):
        print(f"  Excluded chips: {', '.join(stats['excluded_chips'])}")
    if stats.get('cv_norm_files'):
        print(f"  Files using CV normalization: {len(stats['cv_norm_files'])}")
    for label, count in sorted(stats['labels'].items()):
        print(f"  {label}: {count} packets")
    print(f"  Total: {stats['total']} packets")
    
    # Extract features
    print("\nExtracting features...")
    X, y = extract_features(all_packets, subcarriers=subcarriers)
    print(f"  Samples: {len(X)}")
    print(f"  Features: {len(FEATURE_NAMES)}")
    n_idle = np.sum(y == 0)
    n_motion = np.sum(y == 1)
    print(f"  Class balance: IDLE={n_idle}, MOTION={n_motion}")
    if n_idle > 0 and n_motion > 0:
        ratio = max(n_idle, n_motion) / min(n_idle, n_motion)
        print(f"  Imbalance ratio: {ratio:.1f}:1")
    
    # 5-fold cross-validation for reliable evaluation
    if fp_weight != 1.0:
        print(f"\nFP weight: {fp_weight}x (penalizing false positives)")
    print("\n5-fold cross-validation (12 -> 16 -> 8 -> 1)...")
    with suppress_stderr():
        cv_results = cross_validate(X, y, hidden_layers=[16, 8], n_folds=5,
                                    max_epochs=200, fp_weight=fp_weight)
    
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
                           fp_weight=fp_weight)
    
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
    
    return 0


def experiment_architectures(exclude_chips=None):
    """
    Compare multiple MLP architectures using cross-validation.
    
    Trains and evaluates each architecture on the same data with 5-fold CV.
    Reports a comparison table with F1, inference time, and memory usage.
    Recommends the best architecture by F1 (inference time as tiebreaker).
    
    Args:
        exclude_chips: Optional list of chip names to exclude from training
    """
    import time
    from ml_detector import ML_SUBCARRIERS
    subcarriers = ML_SUBCARRIERS
    
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
    if exclude_chips:
        print(f"Loading data (excluding chips: {', '.join(exclude_chips)})...")
    else:
        print("Loading data...")
    all_packets, stats = load_all_data(exclude_chips=exclude_chips)
    
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
    X, y = extract_features(all_packets, subcarriers=subcarriers)
    print(f"  Samples: {len(X)}")
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
  python tools/10_train_ml_model.py                    # Train with all chips
  python tools/10_train_ml_model.py --info              # Show dataset info
  python tools/10_train_ml_model.py --experiment        # Compare architectures
  python tools/10_train_ml_model.py --fp-weight 2.0    # Penalize FP 2x more
  python tools/10_train_ml_model.py --seed 42          # Train with specific seed

Files with use_cv_normalization=true in dataset_info.json use CV normalization
during feature extraction (for data collected without gain lock).
  
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
    parser.add_argument('--fp-weight', type=float, default=2.0,
                       help='Multiplier for IDLE class weight to penalize false positives. '
                            'Values >1.0 make the model more conservative (default: 2.0)')
    parser.add_argument('--exclude-chip', type=str, action='append', default=None,
                       help='Exclude a chip from training data (can be repeated). '
                            'Default: ESP32 (no gain lock, inconsistent amplitudes). '
                            'Use --include-all-chips to override.')
    parser.add_argument('--include-all-chips', action='store_true',
                       help='Include all chips in training (overrides default ESP32 exclusion)')
    
    args = parser.parse_args()
    
    if args.info:
        show_info()
        return 0
    
    # Default: exclude ESP32 (no gain lock → inconsistent amplitudes)
    exclude_chips = args.exclude_chip
    if exclude_chips is None and not args.include_all_chips:
        exclude_chips = ['ESP32']
    
    if args.experiment:
        return experiment_architectures(exclude_chips=exclude_chips)
    
    return train_all(fp_weight=args.fp_weight, exclude_chips=exclude_chips, seed=args.seed)


if __name__ == '__main__':
    exit(main())
