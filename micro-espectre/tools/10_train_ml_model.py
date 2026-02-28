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
    python tools/10_train_ml_model.py --shap             # Show SHAP feature importance

Configuration:
  - TRAINING_FEATURES: Edit at top of file to change feature set

Note: Files marked as "excluded from ML training" in dataset_info.json are
automatically skipped. Files without gain lock use CV normalization.

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
from config import SEG_WINDOW_SIZE
from segmentation import SegmentationContext
from features import (
    calc_skewness, calc_kurtosis, calc_entropy_turb,
    calc_zero_crossing_rate, calc_autocorrelation, calc_mad,
    extract_features_by_name, DEFAULT_FEATURES,
)

# ============================================================================
# Feature Selection (v2.5.1)
# ============================================================================
#
# Features ordered by category: Statistical (8), Temporal (3), Amplitude (1).
# Multi-lag autocorr captures temporal patterns at 10ms, 20ms, 50ms scales.
#
# USED FEATURES (12):
# | Idx | Feature            | SHAP   | Corr   | Type       | Description                  |
# |-----|--------------------|--------|--------|------------|------------------------------|
# |  1  | turb_mean          | 0.030  | -0.486 | Statistical| Mean turbulence              |
# |  2  | turb_std           | 0.020  | -0.116 | Statistical| Standard deviation           |
# |  3  | turb_max           | 0.014  | -0.388 | Statistical| Maximum value                |
# |  4  | turb_min           | 0.017  | -0.276 | Statistical| Minimum value                |
# |  5  | turb_zcr           | 0.036  | -0.539 | Statistical| Zero-crossing rate           |
# |  6  | turb_skewness      | 0.064  | +0.333 | Statistical| Asymmetry (3rd moment)       |
# |  7  | turb_entropy       | 0.108  | +0.447 | Statistical| Shannon entropy (highest)    |
# |  8  | turb_mad           | 0.080  | +0.287 | Statistical| Median absolute deviation    |
# |  9  | turb_autocorr      | 0.033  | +0.737 | Temporal   | Lag-1 autocorr (10ms)        |
# | 10  | turb_autocorr_lag2 | 0.066  | +0.701 | Temporal   | Lag-2 autocorr (20ms)        |
# | 11  | turb_autocorr_lag5 | 0.007  | +0.488 | Temporal   | Lag-5 autocorr (50ms)        |
# | 12  | amp_entropy        | 0.037  | -0.124 | Amplitude  | Amplitude distribution       |
#
# EXCLUDED FEATURES:
# | Feature            | SHAP   | Corr   | Type       | Reason                        |
# |--------------------|--------|--------|------------|-------------------------------|
# | turb_kurtosis      | 0.016  | -0.409 | Statistical| Low importance in 12-set      |
# | turb_slope         | 0.004  | -0.020 | Statistical| Low importance                |
# | turb_delta         | 0.002  | -0.015 | Statistical| Lowest importance             |
# | turb_periodicity   | 0.005  | +0.063 | Temporal   | High overhead (FFT)           |
# | amp_range          | 0.015  | -0.451 | Amplitude  | Low importance in 12-set      |
# | amp_skewness       | 0.003  | -0.058 | Amplitude  | Low importance                |
# | amp_kurtosis       | 0.004  | +0.030 | Amplitude  | Low importance                |
# | phase_diff_var     | 0.004  | -0.296 | Phase      | Low importance                |
# | phase_std          | 0.014  | -0.369 | Phase      | Low importance                |
# | phase_entropy      | 0.005  | +0.213 | Phase      | Low importance                |
# | phase_range        | 0.010  | -0.392 | Phase      | Low importance                |
# | fft_total_energy   | 0.004  | -0.494 | Energy     | High overhead (FFT)           |
# | fft_low_energy     | 0.004  | -0.496 | Energy     | High overhead (FFT)           |
# | fft_mid_energy     | 0.012  | -0.453 | Energy     | High overhead (FFT)           |
# | fft_high_energy    | 0.016  | -0.328 | Energy     | High overhead (FFT)           |
# | fft_energy_ratio   | 0.012  | -0.454 | Energy     | High overhead (FFT)           |
# | fft_dominant_freq  | 0.003  | +0.062 | Energy     | High overhead (FFT)           |
# | spectral_centroid  | 0.018  | +0.336 | Spectral   | High overhead (FFT)           |
# | spectral_flatness  | 0.024  | +0.364 | Spectral   | High overhead (FFT)           |
# | spectral_rolloff   | 0.007  | +0.331 | Spectral   | High overhead (FFT)           |
# ============================================================================

# ============================================================================
# FEATURE SET TO USE FOR TRAINING
# ============================================================================
# Change this list to experiment with different features.
# Available features are defined in src/features.py
#
# Current default (12 features, optimized via SHAP analysis):
TRAINING_FEATURES = DEFAULT_FEATURES

# To experiment with different features, define a custom list here:
# See ALL_AVAILABLE_FEATURES in features.py for the full list


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


def get_class_id(label_name, dataset_info):
    """
    Get multi-class ID for a label.
    
    Uses class_id from dataset_info.json:
    - class_id 0 = IDLE (baseline, baseline_noisy)
    - class_id 1 = MOTION (movement)
    - class_id 2+ = specific gestures (wave, swipe, etc.)
    
    Falls back to binary mapping if class_id not defined.
    
    Args:
        label_name: Label name from npz file
        dataset_info: Loaded dataset_info.json
    
    Returns:
        int: Class ID
    """
    labels = dataset_info.get('labels', {})
    if label_name in labels:
        return labels[label_name].get('class_id', 0)
    return 1 if label_name == 'movement' else 0


def get_class_names(dataset_info):
    """
    Get ordered list of class names from dataset_info.json.
    
    Returns list indexed by class_id: ['idle', 'motion', 'wave', ...]
    
    Args:
        dataset_info: Loaded dataset_info.json
    
    Returns:
        list: Class names ordered by class_id
    """
    labels = dataset_info.get('labels', {})
    class_map = {}
    for label_name, info in labels.items():
        if 'class_id' in info:
            cid = info['class_id']
            if cid not in class_map:
                class_map[cid] = label_name
    
    if not class_map:
        return ['idle', 'motion']
    
    # Build ordered list - use first label found for each class_id
    # For class_id=0 prefer canonical name 'idle'/'baseline'
    max_id = max(class_map.keys())
    result = []
    for i in range(max_id + 1):
        name = class_map.get(i, f'class_{i}')
        # Use canonical names for 0 and 1
        if i == 0:
            name = 'idle'
        elif i == 1:
            name = 'motion'
        result.append(name)
    return result


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


def load_all_data():
    """
    Load all available CSI data from the data/ directory.
    
    Reads label from npz file metadata (not folder structure).
    Uses dataset_info.json to determine if label is motion or idle.
    Sets use_cv_normalization flag on each packet based on dataset_info.json.
    
    Returns:
        tuple: (all_packets, stats) where stats is a dict with dataset info
    """
    all_packets = []
    stats = {'chips': set(), 'labels': {}, 'total': 0, 'cv_norm_files': set()}
    
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
                class_id = get_class_id(label, dataset_info)
                for p in packets:
                    p['class_id'] = class_id
                    p['use_cv_normalization'] = use_cv_norm
                
                all_packets.extend(packets)
                
            except Exception as e:
                print(f"  Warning: Could not load {npz_file.name}: {e}")
    
    stats['chips'] = sorted(stats['chips'])
    stats['cv_norm_files'] = sorted(stats['cv_norm_files'])
    return all_packets, stats


# ============================================================================
# Feature Extraction
# ============================================================================

def extract_features(packets, window_size=SEG_WINDOW_SIZE, subcarriers=None,
                     feature_names=None):
    """
    Extract features from CSI packets using sliding window.
    
    Args:
        packets: List of CSI packets with 'csi_data' and 'label'
        window_size: Sliding window size (default: SEG_WINDOW_SIZE from config.py)
        subcarriers: List of subcarrier indices to use (default: DEFAULT_SUBCARRIERS)
        feature_names: List of feature names to extract (default: DEFAULT_FEATURES)
    
    Returns:
        tuple: (X, y, feature_names) feature matrix, labels, and actual feature names
    """
    from utils import extract_phases
    
    if subcarriers is None:
        subcarriers = DEFAULT_SUBCARRIERS
    
    if feature_names is None:
        feature_names = DEFAULT_FEATURES.copy()
    
    X, y = [], []
    turb_buffer = deque(maxlen=window_size)
    last_amplitudes = None
    last_phases = None
    
    for pkt in packets:
        csi_data = pkt['csi_data']
        
        # Calculate turbulence, using CV normalization for files that need it
        # (collected without gain lock, e.g. ESP32 or C3 without gain lock)
        use_cv_norm = pkt.get('use_cv_normalization', False)
        turb, amps = SegmentationContext.compute_spatial_turbulence(
            csi_data, subcarriers, use_cv_normalization=use_cv_norm
        )
        turb_buffer.append(turb)
        last_amplitudes = amps
        
        # Extract phases for phase-based features
        last_phases = extract_phases(csi_data, subcarriers)
        
        # Wait for buffer to fill
        if len(turb_buffer) < window_size:
            continue
        
        turb_list = list(turb_buffer)
        n = len(turb_list)
        
        # Extract features using configurable feature set
        features = extract_features_by_name(
            turb_list, n, 
            amplitudes=last_amplitudes,
            feature_names=feature_names,
            phases=last_phases
        )
        
        X.append(features)
        y.append(pkt.get('class_id', 0))
    
    return np.array(X), np.array(y), feature_names


# ============================================================================
# Model Training
# ============================================================================

def build_model(num_classes, hidden_layers=[16, 8], num_features=12, use_dropout=True, dropout_rate=0.2):
    """
    Build a Keras MLP model (softmax multiclass output).

    Dropout layers are added during training for regularization but are
    automatically disabled during inference (and don't affect exported weights).

    Args:
        num_classes: Number of output classes
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

    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def train_model(X, y, num_classes, hidden_layers=[16, 8], max_epochs=200, use_dropout=True,
                class_weight=None, fp_weight=1.0, verbose=0):
    """
    Train a neural network model with best practices.

    Uses early stopping, learning rate reduction, dropout regularization,
    and balanced class weighting for imbalanced datasets.

    Args:
        X: Feature matrix (normalized)
        y: Labels (class IDs)
        num_classes: Number of output classes
        hidden_layers: List of hidden layer sizes
        max_epochs: Maximum training epochs (early stopping will cut short)
        use_dropout: Whether to add dropout layers
        class_weight: Class weight dict or None for auto-balanced
        fp_weight: Multiplier for class 0 (IDLE) weight to penalize false positives
        verbose: Training verbosity

    Returns:
        Trained Keras model
    """
    import tensorflow as tf
    from sklearn.utils.class_weight import compute_class_weight

    if class_weight is None:
        unique_classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=unique_classes, y=y)
        class_weight = dict(zip(unique_classes.tolist(), weights.tolist()))

    # Apply FP penalty to class 0 (IDLE) to reduce false positives
    if fp_weight != 1.0 and 0 in class_weight:
        class_weight[0] *= fp_weight

    num_features = X.shape[1] if hasattr(X, 'shape') else len(X[0])
    model = build_model(num_classes, hidden_layers=hidden_layers, num_features=num_features,
                        use_dropout=use_dropout)
    
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


def evaluate_model_multiclass(model, X_test, y_test, class_names):
    """
    Evaluate a multiclass model on test data.
    
    Args:
        model: Trained Keras model (softmax output)
        X_test: Test features (normalized)
        y_test: Test labels (class IDs)
        class_names: List of class names indexed by class_id
    
    Returns:
        dict: Metrics (accuracy, per_class_accuracy, confusion)
    """
    from sklearn.metrics import confusion_matrix, classification_report
    
    probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    
    accuracy = np.mean(y_pred == y_test) * 100
    
    per_class = {}
    for cid, name in enumerate(class_names):
        mask = y_test == cid
        if mask.sum() > 0:
            per_class[name] = np.mean(y_pred[mask] == cid) * 100
    
    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(class_names))))
    report = classification_report(y_test, y_pred, target_names=class_names,
                                   zero_division=0, output_dict=True)
    
    return {
        'accuracy': accuracy,
        'per_class': per_class,
        'confusion': cm,
        'report': report,
        'f1': report.get('macro avg', {}).get('f1-score', 0) * 100,
    }


def cross_validate(X, y, num_classes, class_names, hidden_layers=[16, 8], n_folds=5,
                   max_epochs=200, fp_weight=1.0):
    """
    Perform stratified k-fold cross-validation.

    Args:
        X: Feature matrix (NOT normalized - scaler fit per fold)
        y: Labels (class IDs)
        num_classes: Number of output classes
        class_names: Class names for evaluation report
        hidden_layers: List of hidden layer sizes
        n_folds: Number of CV folds
        max_epochs: Maximum training epochs per fold
        fp_weight: Multiplier for class 0 weight (>1.0 penalizes FP more)

    Returns:
        dict: Mean and std of each scalar metric across folds
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_metrics = []

    for train_idx, val_idx in skf.split(X, y):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_val = scaler.transform(X[val_idx])

        with suppress_stderr():
            model = train_model(X_train, y[train_idx], num_classes,
                                hidden_layers=hidden_layers, max_epochs=max_epochs,
                                fp_weight=fp_weight)
            metrics = evaluate_model_multiclass(model, X_val, y[val_idx], class_names)

        fold_metrics.append(metrics)

    result = {}
    scalar_keys = [k for k, v in fold_metrics[0].items() if isinstance(v, (int, float))]
    for key in scalar_keys:
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


def export_micropython(model, scaler, output_path, seed=None, class_names=None):
    """
    Export model weights to MicroPython code (ml_weights.py).

    Args:
        model: Trained Keras model
        scaler: StandardScaler with mean_ and scale_
        output_path: Output file path
        seed: Random seed used for training
        class_names: List of class names (required for multiclass)

    Returns:
        Size of generated code
    """
    from datetime import datetime
    weights = model.get_weights()

    num_classes = len(class_names) if class_names else 2
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    code = '''"""
Micro-ESPectre - ML Model Weights

Auto-generated neural network weights for motion/gesture detection.
Architecture: 12 -> ''' + ' -> '.join(str(w.shape[1]) for w in weights[::2]) + f'''
Classes: {num_classes}
Trained: {timestamp}
Seed: {seed}

This file is auto-generated by 10_train_ml_model.py.
DO NOT EDIT - your changes will be overwritten!

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

# Feature normalization (StandardScaler)
FEATURE_MEAN = [''' + ', '.join(f'{x:.6f}' for x in scaler.mean_) + ''']
FEATURE_SCALE = [''' + ', '.join(f'{x:.6f}' for x in scaler.scale_) + ''']

'''

    code += f'# Class labels (indexed by class_id)\n'
    code += f'CLASS_LABELS = {class_names}\n'
    code += f'NUM_CLASSES = {num_classes}\n\n'

    for i in range(0, len(weights), 2):
        W = weights[i]
        b = weights[i + 1]
        layer_num = i // 2 + 1
        in_size, out_size = W.shape

        activation = 'Softmax' if i == len(weights) - 2 else 'ReLU'
        code += f'# Layer {layer_num}: {in_size} -> {out_size} ({activation})\n'
        code += f'W{layer_num} = [\n'
        for row in W:
            code += '    [' + ', '.join(f'{x:.6f}' for x in row) + '],\n'
        code += ']\n'
        code += f'B{layer_num} = [' + ', '.join(f'{x:.6f}' for x in b) + ']\n\n'
    
    with open(output_path, 'w') as f:
        f.write(code)
    
    return len(code)


def export_cpp_weights(model, scaler, output_path, seed=None, class_names=None):
    """
    Export model weights to C++ header for ESPHome.
    
    Generates ml_weights.h with constexpr weights.
    
    Args:
        model: Trained Keras model
        scaler: StandardScaler with mean_ and scale_
        output_path: Output file path
        seed: Random seed used for training (or None if not set)
        class_names: List of class names for multiclass model (None = binary)
    
    Returns:
        Size of generated code
    """
    from datetime import datetime
    weights = model.get_weights()
    arch = ' -> '.join(str(w.shape[1]) for w in weights[::2])
    num_classes = len(class_names) if class_names else 2
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    code = f'''/*
 * ESPectre - ML Model Weights
 *
 * Auto-generated neural network weights for motion/gesture detection.
 * Architecture: 12 -> {arch}
 * Classes: {num_classes}
 * Trained: {timestamp}
 * Seed: {seed}
 *
 * This file is auto-generated by 10_train_ml_model.py.
 * DO NOT EDIT - your changes will be overwritten!
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#pragma once
#include <array>

namespace esphome {{
namespace espectre {{

// Feature normalization (StandardScaler)
constexpr float ML_FEATURE_MEAN[12] = {{{', '.join(f'{x:.6f}f' for x in scaler.mean_)}}};
constexpr float ML_FEATURE_SCALE[12] = {{{', '.join(f'{x:.6f}f' for x in scaler.scale_)}}};

constexpr int ML_NUM_CLASSES = {num_classes};

// Class labels (indexed by class_id)
constexpr const char* ML_CLASS_LABELS[{num_classes}] = {{
'''
    for name in (class_names or [f'class_{i}' for i in range(num_classes)]):
        code += f'    "{name}",\n'
    code += '};\n\n'

    for i in range(0, len(weights), 2):
        W = weights[i]
        b = weights[i + 1]
        layer_num = i // 2 + 1
        in_size, out_size = W.shape

        activation = 'Softmax' if i == len(weights) - 2 else 'ReLU'
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
    X_test_scaled = scaler.transform(X_test_raw)
    probs = model.predict(X_test_scaled, verbose=0)

    # expected_outputs = 1 - prob[idle], matching MLDetector::predict() in both C++ and Python.
    expected_outputs = (1.0 - probs[:, 0]).astype(np.float32)

    # Save RAW features (not normalized) so tests can verify the full pipeline
    np.savez(output_path,
             features=X_test_raw.astype(np.float32),
             labels=y_test.astype(np.int32),
             expected_outputs=expected_outputs)
    
    return len(X_test_raw)


# ============================================================================
# Feature Importance (Correlation)
# ============================================================================

def calculate_correlation_importance(feature_names=None):
    """
    Calculate correlation of ALL available features with motion label.
    
    This is a fast alternative to SHAP for initial feature screening.
    Reuses load_all_data() and extract_features() for DRY compliance.
    
    Args:
        feature_names: Optional list of features to analyze (default: ALL_AVAILABLE_FEATURES)
    
    Returns:
        dict: {feature_name: correlation} sorted by absolute correlation
    """
    from src.features import ALL_AVAILABLE_FEATURES
    
    if feature_names is None:
        feature_names = list(ALL_AVAILABLE_FEATURES)
    
    print("\nCalculating feature correlations...")
    print(f"  Analyzing {len(feature_names)} features")
    
    # Reuse existing data loading and feature extraction
    all_packets, stats = load_all_data()
    print(f"  Loaded {stats['total']} packets")
    if stats.get('cv_norm_files'):
        print(f"  Files using CV normalization: {len(stats['cv_norm_files'])}")
    
    print("  Extracting features...")
    X, y, actual_features = extract_features(all_packets, feature_names=feature_names)
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
    Train a model with ALL available features and calculate SHAP importance.
    
    This is useful for comparing importance of all features, not just the
    default 12. Uses fewer samples for speed since we have more features.
    
    Args:
        n_samples: Number of samples for SHAP (default: 100, lower for speed)
    
    Returns:
        int: Exit code
    """
    from src.features import ALL_AVAILABLE_FEATURES
    
    all_features = list(ALL_AVAILABLE_FEATURES)
    print(f"\n{'='*70}")
    print(f"  SHAP Analysis with ALL {len(all_features)} features")
    print(f"{'='*70}")
    print(f"  Using {n_samples} samples (use --shap-samples to change)")
    
    # Load data
    all_packets, stats = load_all_data()
    print(f"\nLoaded {stats['total']} packets")
    
    # Extract ALL features
    print(f"Extracting {len(all_features)} features...")
    X, y, actual_features = extract_features(all_packets, feature_names=all_features)
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
        class_id = info.get('class_id', '?')
        print(f"  {label} (class_id={class_id})")
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
    for label, count in sorted(stats['labels'].items()):
        class_id = get_class_id(label, dataset_info)
        print(f"  {label}: {count} packets (class_id={class_id})")
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


def train_all(fp_weight=2.0, seed=None, feature_names=None,
              feature_importance=False, ablation=False, shap_samples=200):
    """
    Train models with all available data.

    Args:
        fp_weight: Multiplier for class 0 (IDLE) weight. Values >1.0 penalize
                   false positives more, producing a more conservative model.
        seed: Optional random seed for reproducible training.
        feature_names: List of feature names to use. If None, uses DEFAULT_FEATURES.
        feature_importance: If True, calculate and display SHAP feature importance.
        ablation: If True, run ablation study instead of training.
    """
    from ml_detector import ML_SUBCARRIERS
    subcarriers = ML_SUBCARRIERS
    hidden_layers = [24]

    print("\n" + "="*60)
    print("        ML MOTION/GESTURE DETECTOR TRAINING")
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
        return 1
    
    # Load data
    print("Loading data...")
    all_packets, stats = load_all_data()
    
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
    
    # Determine feature set to use
    if feature_names is None:
        feature_names = DEFAULT_FEATURES.copy()
    
    dataset_info = load_dataset_info()
    class_names = get_class_names(dataset_info)
    num_classes = len(class_names)

    print(f"Classes ({num_classes}): {class_names}\n")

    # Extract features
    print("\nExtracting features...")
    X, y, actual_feature_names = extract_features(all_packets, subcarriers=subcarriers,
                                                   feature_names=feature_names)
    print(f"  Samples: {len(X)}")
    print(f"  Features: {len(actual_feature_names)}")
    print(f"  Feature set: {', '.join(actual_feature_names)}")
    for cid, name in enumerate(class_names):
        print(f"  {name} (class {cid}): {np.sum(y == cid)} samples")
    
    # Run ablation study if requested
    if ablation:
        run_ablation_study(X, y, actual_feature_names, fp_weight=fp_weight)
        return 0
    
    # 5-fold cross-validation for reliable evaluation
    if fp_weight != 1.0:
        print(f"\nFP weight: {fp_weight}x (penalizing false positives)")
    arch_str = ' -> '.join(map(str, [12] + hidden_layers + [num_classes]))
    print(f"\n5-fold cross-validation ({arch_str})...")
    with suppress_stderr():
        cv_results = cross_validate(X, y, num_classes, class_names, hidden_layers=hidden_layers,
                                    n_folds=5, max_epochs=200, fp_weight=fp_weight)

    print(f"  Accuracy:  {cv_results['accuracy_mean']:.1f}% (+/- {cv_results['accuracy_std']:.1f}%)")
    print(f"  F1 Score:  {cv_results['f1_mean']:.1f}% (+/- {cv_results['f1_std']:.1f}%)")
    
    # Also do a single split for test data export
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("\nTraining final model on full dataset...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    with suppress_stderr():
        model = train_model(X_scaled, y, num_classes, hidden_layers=hidden_layers, max_epochs=200,
                            fp_weight=fp_weight)

    X_test_scaled = scaler.transform(X_test_raw)
    with suppress_stderr():
        test_metrics = evaluate_model_multiclass(model, X_test_scaled, y_test, class_names)

    print(f"\nHold-out test set (20%):")
    print(f"  Accuracy:  {test_metrics['accuracy']:.1f}%")
    print(f"  F1 Score:  {test_metrics['f1']:.1f}%")
    print(f"  Per-class accuracy:")
    for name, acc in test_metrics['per_class'].items():
        print(f"    {name}: {acc:.1f}%")
    
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
    mp_size = export_micropython(model, scaler, mp_path, seed=seed, class_names=class_names)
    print(f"  MicroPython weights: {mp_path.name} ({mp_size/1024:.1f} KB)")
    
    # C++ weights for ESPHome
    cpp_path = CPP_DIR / 'ml_weights.h'
    cpp_size = export_cpp_weights(model, scaler, cpp_path, seed=seed, class_names=class_names)
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
    print(f"Classes: {class_names}")
    print(f"\nGenerated files:")
    print(f"  - {mp_path} (MicroPython)")
    print(f"  - {cpp_path} (C++ ESPHome)")
    print(f"  - {tflite_path} (ESP-IDF TFLite)")
    print(f"  - {scaler_path} (normalization params)")
    print(f"  - {test_data_path} (test data for validation)")
    print()
    
    return 0


def experiment_architectures():
    """
    Compare multiple MLP architectures using cross-validation.
    
    Trains and evaluates each architecture on the same data with 5-fold CV.
    Reports a comparison table with F1, inference time, and memory usage.
    Recommends the best architecture by F1 (inference time as tiebreaker).
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
    
    dataset_info = load_dataset_info()
    class_names = get_class_names(dataset_info)
    num_classes = len(class_names)

    print("\nExtracting features...")
    X, y, feature_names = extract_features(all_packets, subcarriers=subcarriers)
    print(f"  Samples: {len(X)}")
    print(f"  Features: {len(feature_names)}")
    for cid, name in enumerate(class_names):
        print(f"  {name} (class {cid}): {np.sum(y == cid)} samples")
    
    # Define architectures to compare
    architectures = [
        {'name': 'Current (24)', 'layers': [24]},
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
        layer_sizes = [12] + layers + [num_classes]
        n_params = sum(
            layer_sizes[i] * layer_sizes[i + 1] + layer_sizes[i + 1]
            for i in range(len(layer_sizes) - 1)
        )
        weight_kb = n_params * 4 / 1024
        flops = sum(layer_sizes[i] * layer_sizes[i + 1] for i in range(len(layer_sizes) - 1))

        arch_str = ' -> '.join(map(str, layer_sizes))
        print(f"\nEvaluating: {name} ({arch_str})...")
        print(f"  Parameters: {n_params}, Weights: {weight_kb:.1f} KB, FLOPS: {flops}")

        # Cross-validate
        with suppress_stderr():
            cv = cross_validate(X, y, num_classes, class_names, hidden_layers=layers,
                                n_folds=5, max_epochs=200)

        # Measure Python inference time
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        with suppress_stderr():
            model = train_model(X_scaled, y, num_classes, hidden_layers=layers, max_epochs=200)
        
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
            'accuracy_mean': cv.get('accuracy_mean', 0.0),
            'inference_us': inference_us,
        }
        results.append(result)

        print(f"  F1: {cv['f1_mean']:.1f}% +/- {cv['f1_std']:.1f}%")
        print(f"  Accuracy: {cv.get('accuracy_mean', 0.0):.1f}%")
        print(f"  Inference: {inference_us:.1f} us/sample")

    # Print comparison table
    print("\n" + "="*75)
    print("                    ARCHITECTURE COMPARISON")
    print("="*75 + "\n")

    print(f"{'Architecture':<22} {'Params':>7} {'KB':>6} {'F1 (CV)':>12} {'Accuracy':>10} {'Inf (us)':>10}")
    print("-"*75)

    best = max(results, key=lambda r: (r['f1_mean'], -r['inference_us']))

    for r in results:
        marker = " **" if r == best else "   "
        print(f"{marker}{r['name']:<19} {r['params']:>7} {r['weight_kb']:>5.1f} "
              f"{r['f1_mean']:>6.1f}+/-{r['f1_std']:<4.1f} "
              f"{r['accuracy_mean']:>9.1f}% "
              f"{r['inference_us']:>9.1f}")

    print("-"*75)
    print(f"\n** Best architecture: {best['name']}")
    print(f"   F1: {best['f1_mean']:.1f}% +/- {best['f1_std']:.1f}%")
    print(f"   Accuracy: {best['accuracy_mean']:.1f}%")
    print(f"   Parameters: {best['params']}, Weights: {best['weight_kb']:.1f} KB")
    
    # Recommend action
    current = next((r for r in results if r['name'].startswith('Current')), None)
    if current and best != current:
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
  python tools/10_train_ml_model.py --shap             # Show SHAP feature importance

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
    parser.add_argument('--fp-weight', type=float, default=2.0,
                       help='Multiplier for IDLE class weight to penalize false positives. '
                            'Values >1.0 make the model more conservative (default: 2.0)')
    parser.add_argument('--shap', action='store_true',
                       help='Calculate and display SHAP feature importance (12 default features)')
    parser.add_argument('--shap-all', action='store_true',
                       help='Calculate SHAP for ALL available features (slower, uses fewer samples)')
    parser.add_argument('--shap-samples', type=int, default=200,
                       help='Number of samples for SHAP analysis (default: 200)')
    parser.add_argument('--correlation', action='store_true',
                       help='Calculate correlation of ALL available features with motion label')
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
    
    if args.shap_all:
        return run_shap_all_features(n_samples=args.shap_samples)
    
    return train_all(
        fp_weight=args.fp_weight,
        seed=args.seed,
        feature_names=TRAINING_FEATURES,
        feature_importance=args.shap,
        ablation=args.ablation,
        shap_samples=args.shap_samples,
    )


if __name__ == '__main__':
    exit(main())
