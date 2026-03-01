#!/usr/bin/env python3
"""
ML Motion Detection - Training Script (binary IDLE/MOTION detector)

Trains neural network models for motion detection using all available CSI data.
Generates models for both ESP-IDF (TFLite) and MicroPython.

This script trains a binary IDLE/MOTION detector.

Only motion-detection labels are used for training:
  - baseline -> IDLE
  - movement -> MOTION

Gesture-specific labels (wave, swipe_left, etc.) are excluded.

Training features:
  - 5-fold stratified cross-validation for reliable metrics
  - Early stopping with patience to prevent overfitting
  - Dropout regularization during training
  - Balanced class weights for imbalanced datasets
  - Learning rate reduction on plateau
  - Configurable FP penalty (--fp-weight) for conservative models

Usage:
    python tools/10_train_motion_model.py                    # Train with default features
    python tools/10_train_motion_model.py --info             # Show dataset info
    python tools/10_train_motion_model.py --experiment       # Compare architectures
    python tools/10_train_motion_model.py --fp-weight 2.0    # Penalize FP 2x more
    python tools/10_train_motion_model.py --shap             # Show SHAP feature importance

Configuration:
  - TRAINING_FEATURES: Edit at top of file to change feature set
  - TRAINING_LABELS: Labels included in motion training

Files without gain lock use CV normalization.

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

# Import csi_utils first - it sets up paths automatically
from csi_utils import (
    load_npz_as_packets,
    DATA_DIR,
    DEFAULT_SUBCARRIERS,
)
from config import SEG_WINDOW_SIZE
from segmentation import SegmentationContext
from features import (
    extract_features_by_name, DEFAULT_FEATURES,
)
from ml_utils import (
    suppress_stderr,
    generate_seed,
    setup_tf_logging,
    train_model,
    evaluate_model_multiclass,
    cross_validate,
    export_tflite,
    export_micropython,
    export_cpp_weights,
    export_test_data,
    calculate_shap_importance,
    print_feature_importance,
    print_correlation_table,
    experiment_architectures,
    split_holdout,
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
# | 10  | turb_mad           | 0.080  | +0.287 | Statistical| Median absolute deviation    |
# | 11  | turb_slope         | 0.004  | -0.020 | Statistical| Linear trend slope           |
# | 12  | amp_entropy        | 0.037  | -0.124 | Amplitude  | Amplitude distribution       |
#
# EXCLUDED FEATURES:
# | Feature            | SHAP   | Corr   | Type       | Reason                        |
# |--------------------|--------|--------|------------|-------------------------------|
# | turb_autocorr_lag2 | 0.066  | +0.701 | Temporal   | Not in current DEFAULT_FEATURES |
# | turb_autocorr_lag5 | 0.007  | +0.488 | Temporal   | Not in current DEFAULT_FEATURES |
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
# See DEFAULT_FEATURES in features.py for the current feature set


# Directories
MODELS_DIR = Path(__file__).parent.parent / 'models'
SRC_DIR = Path(__file__).parent.parent / 'src'
CPP_DIR = Path(__file__).parent.parent.parent / 'components' / 'espectre'

# Motion-training classes: IDLE (0) vs MOTION (1)
CLASS_ID_BY_LABEL = {'baseline': 0, 'movement': 1}
CLASS_NAMES = ['idle', 'motion']
TRAINING_LABELS = list(CLASS_ID_BY_LABEL.keys())


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
    return {'labels': {}, 'files': {}}


def get_file_metadata(dataset_info):
    """Get metadata for all files in dataset_info.json.

    Returns:
        dict: {filename: {use_cv_normalization: bool, label: str, chip: str}}
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
                    'label': label,
                }
    return file_metadata


def load_all_data():
    """Load motion-training CSI data from the data/ directory.

    Only loads TRAINING_LABELS (baseline, movement).
    Gesture-specific labels are excluded.

    Returns:
        tuple: (all_packets, stats) where stats is a dict with dataset info.
    """
    all_packets = []
    stats = {'chips': set(), 'labels': {}, 'total': 0, 'cv_norm_files': set(),
             'excluded_labels': set()}

    dataset_info = load_dataset_info()
    file_metadata = get_file_metadata(dataset_info)

    for subdir in DATA_DIR.iterdir():
        if not subdir.is_dir():
            continue

        label = subdir.name

        # Only load motion-training labels
        class_id = CLASS_ID_BY_LABEL.get(label)
        if class_id is None:
            stats['excluded_labels'].add(label)
            continue

        for npz_file in subdir.glob('*.npz'):
            try:
                packets = load_npz_as_packets(npz_file)
                if not packets:
                    continue

                chip = packets[0].get('chip', 'unknown').upper()

                if label not in stats['labels']:
                    stats['labels'][label] = 0
                stats['labels'][label] += len(packets)
                stats['total'] += len(packets)
                stats['chips'].add(chip)

                meta = file_metadata.get(npz_file.name, {})
                use_cv_norm = meta.get('use_cv_normalization', False)
                if use_cv_norm:
                    stats['cv_norm_files'].add(npz_file.name)

                for p in packets:
                    p['class_id'] = class_id
                    p['use_cv_normalization'] = use_cv_norm
                    p['source_file'] = npz_file.name

                all_packets.extend(packets)

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
                     feature_names=None, return_groups=False):
    """Extract features from CSI packets using sliding window.

    Args:
        packets: List of CSI packets with 'csi_data' and 'label'.
        window_size: Sliding window size (default: SEG_WINDOW_SIZE from config.py).
        subcarriers: List of subcarrier indices to use (default: DEFAULT_SUBCARRIERS).
        feature_names: List of feature names to extract (default: DEFAULT_FEATURES).
        return_groups: If True, also return source group per extracted window.

    Returns:
        tuple:
            - (X, y, feature_names) when return_groups=False
            - (X, y, feature_names, groups) when return_groups=True
    """
    if subcarriers is None:
        subcarriers = DEFAULT_SUBCARRIERS

    if feature_names is None:
        feature_names = DEFAULT_FEATURES.copy()

    X, y, groups = [], [], []
    turb_buffer = deque(maxlen=window_size)
    last_amplitudes = None
    current_source = None

    for pkt in packets:
        source_file = pkt.get('source_file', 'unknown')
        if source_file != current_source:
            # Prevent windows from crossing file/session boundaries.
            turb_buffer.clear()
            last_amplitudes = None
            current_source = source_file

        csi_data = pkt['csi_data']

        use_cv_norm = pkt.get('use_cv_normalization', False)
        turb, amps = SegmentationContext.compute_spatial_turbulence(
            csi_data, subcarriers, use_cv_normalization=use_cv_norm
        )
        turb_buffer.append(turb)
        last_amplitudes = amps

        if len(turb_buffer) < window_size:
            continue

        turb_list = list(turb_buffer)
        n = len(turb_list)

        features = extract_features_by_name(
            turb_list, n,
            amplitudes=last_amplitudes,
            feature_names=feature_names
        )

        X.append(features)
        y.append(pkt.get('class_id', 0))
        if return_groups:
            groups.append(source_file)

    if return_groups:
        return np.array(X), np.array(y), feature_names, np.array(groups)
    return np.array(X), np.array(y), feature_names


# ============================================================================
# Feature Importance (Correlation)
# ============================================================================

def calculate_correlation_importance(feature_names=None):
    """Calculate correlation of available features with motion label.

    Fast alternative to SHAP for initial feature screening.

    Args:
        feature_names: Optional list of features to analyze (default: DEFAULT_FEATURES).

    Returns:
        dict: {feature_name: correlation} sorted by absolute correlation.
    """
    if feature_names is None:
        feature_names = list(DEFAULT_FEATURES)

    print("\nCalculating feature correlations...")
    print(f"  Analyzing {len(feature_names)} features")

    all_packets, stats = load_all_data()
    print(f"  Loaded {stats['total']} packets")
    if stats.get('cv_norm_files'):
        print(f"  Files using CV normalization: {len(stats['cv_norm_files'])}")

    print("  Extracting features...")
    X, y, actual_features = extract_features(all_packets, feature_names=feature_names)
    print(f"  Extracted features for {len(X)} samples")

    correlations = {}
    for i, fname in enumerate(actual_features):
        corr = np.corrcoef(X[:, i], y)[0, 1]
        if not np.isnan(corr):
            correlations[fname] = corr

    return dict(sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True))


# ============================================================================
# Ablation Study
# ============================================================================

def run_ablation_study(X, y, num_classes, class_names, feature_names,
                       hidden_layers=None, fp_weight=2.0, groups=None):
    """Run ablation study: train model removing one feature at a time.

    Args:
        X: Feature matrix (NOT normalized).
        y: Labels.
        num_classes: Number of output classes.
        class_names: Class names list.
        feature_names: List of feature names.
        hidden_layers: Model architecture.
        fp_weight: FP penalty weight.
        groups: Optional group IDs (source files) for group-aware CV.

    Returns:
        list: Results for each ablation experiment.
    """
    if hidden_layers is None:
        hidden_layers = [16, 8]

    print("\n" + "="*80)
    print("                         ABLATION STUDY")
    print("="*80 + "\n")
    print("Training models with one feature removed at a time to measure impact...\n")

    results = []

    print(f"[1/{len(feature_names)+1}] Baseline (all {len(feature_names)} features)...")
    with suppress_stderr():
        baseline_cv = cross_validate(X, y, num_classes, class_names,
                                     hidden_layers=hidden_layers, n_folds=5,
                                     max_epochs=200, fp_weight=fp_weight,
                                     groups=groups)
    baseline_f1 = baseline_cv['f1_mean']
    results.append({
        'removed': 'None (baseline)',
        'n_features': len(feature_names),
        'f1_mean': baseline_f1,
        'f1_std': baseline_cv['f1_std'],
        'accuracy_mean': baseline_cv['accuracy_mean'],
        'delta_f1': 0.0,
    })
    print(f"    F1: {baseline_f1:.2f}% (+/- {baseline_cv['f1_std']:.2f}%)\n")

    for i, feature_name in enumerate(feature_names):
        print(f"[{i+2}/{len(feature_names)+1}] Removing '{feature_name}'...")

        X_ablated = np.delete(X, i, axis=1)

        with suppress_stderr():
            cv = cross_validate(X_ablated, y, num_classes, class_names,
                                hidden_layers=hidden_layers, n_folds=5,
                                max_epochs=200, fp_weight=fp_weight,
                                groups=groups)

        f1 = cv['f1_mean']
        delta = f1 - baseline_f1

        results.append({
            'removed': feature_name,
            'n_features': len(feature_names) - 1,
            'f1_mean': f1,
            'f1_std': cv['f1_std'],
            'accuracy_mean': cv['accuracy_mean'],
            'delta_f1': delta,
        })

        direction = "↑" if delta > 0.1 else "↓" if delta < -0.1 else "≈"
        print(f"    F1: {f1:.2f}% ({direction} {delta:+.2f}%)\n")

    print("\n" + "="*85)
    print("                           ABLATION SUMMARY")
    print("="*85 + "\n")

    sorted_results = sorted(results[1:], key=lambda r: r['delta_f1'])

    print(f"{'Removed Feature':<24} {'F1 (CV)':>14} {'Delta':>10} {'Accuracy':>10} {'Note':<12}")
    print("-"*85)

    bl = results[0]
    print(f"{'None (baseline)':<24} {bl['f1_mean']:>8.2f}% +/-{bl['f1_std']:.1f} "
          f"{'---':>10} {bl['accuracy_mean']:>9.1f}%")
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
              f"{delta_str:>10} {r['accuracy_mean']:>9.1f}% {note:<12}")

    print("-"*85)
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

    dataset_info = load_dataset_info()

    print("Labels defined in dataset_info.json:")
    for label, info in dataset_info.get('labels', {}).items():
        status = "included" if label in TRAINING_LABELS else "excluded"
        print(f"  {label} ({status})")
        if info.get('description'):
            print(f"    {info['description']}")
    print()

    file_metadata = get_file_metadata(dataset_info)
    cv_norm_files = [f for f, meta in file_metadata.items() if meta.get('use_cv_normalization')]
    if cv_norm_files:
        print(f"Files using CV normalization ({len(cv_norm_files)}):")
        for f in sorted(cv_norm_files):
            print(f"  - {f}")
        print()

    _, stats = load_all_data()

    if stats['excluded_labels']:
        print(f"Excluded labels (gesture labels): {', '.join(stats['excluded_labels'])}")
        print()

    idle_labels = [lbl for lbl, cid in CLASS_ID_BY_LABEL.items() if cid == 0]
    motion_labels = [lbl for lbl, cid in CLASS_ID_BY_LABEL.items() if cid == 1]

    print("Motion training data:")
    print(f"  IDLE labels: {', '.join(idle_labels)}")
    print(f"  MOTION labels: {', '.join(motion_labels)}")
    print(f"  Chips: {', '.join(stats['chips']) if stats['chips'] else 'None'}")
    print(f"  Total packets: {stats['total']}")
    for label, count in sorted(stats['labels'].items()):
        class_name = "IDLE" if CLASS_ID_BY_LABEL.get(label) == 0 else "MOTION"
        print(f"  {label}: {count} packets ({class_name})")
    print()

    print("All labels in data/ directory:")
    for subdir in sorted(DATA_DIR.iterdir()):
        if subdir.is_dir() and not subdir.name.startswith('.'):
            files = list(subdir.glob('*.npz'))
            if files:
                status = "included" if subdir.name in TRAINING_LABELS else "excluded"
                print(f"  {subdir.name}: {len(files)} files ({status})")
    print()


def _setup_training_runtime(seed=None):
    """Import training dependencies, initialize logging, and resolve seed."""
    try:
        with suppress_stderr():
            import tensorflow as tf  # noqa: F401
            from sklearn.preprocessing import StandardScaler  # noqa: F401
    except ImportError as e:
        print(f"Error: Missing dependency - {e}")
        print("Install with: pip install tensorflow scikit-learn")
        return None

    if seed is None:
        seed = generate_seed()
        print(f"Generated random seed: {seed}\n")
    else:
        print(f"Using provided seed: {seed}\n")

    setup_tf_logging(seed=seed)
    return seed


def _print_motion_data_summary(stats):
    """Print packet-level summary for motion training labels."""
    print(f"  Chips: {', '.join(stats['chips'])}")
    if stats.get('excluded_files'):
        print(f"  Files excluded: {len(stats['excluded_files'])}")
    if stats.get('cv_norm_files'):
        print(f"  Files using CV normalization: {len(stats['cv_norm_files'])}")
    for label, count in sorted(stats['labels'].items()):
        print(f"  {label}: {count} packets")
    print(f"  Total: {stats['total']} packets")


def _prepare_motion_training_data(subcarriers, feature_names=None):
    """Load motion packets, extract features, and normalize class IDs."""
    print("Loading motion training data (gesture recordings excluded)...")
    all_packets, stats = load_all_data()

    if not stats['chips']:
        print("Error: No datasets found in data/")
        print("Collect data using: ./me collect --label baseline --duration 60")
        return None

    _print_motion_data_summary(stats)

    if feature_names is None:
        feature_names = DEFAULT_FEATURES.copy()

    print("\nExtracting features...")
    X, y, actual_feature_names, groups = extract_features(
        all_packets,
        subcarriers=subcarriers,
        feature_names=feature_names,
        return_groups=True
    )

    observed_ids = sorted(np.unique(y).tolist())
    class_names = [CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else f'class_{cid}'
                   for cid in observed_ids]
    id_remap = {old: new for new, old in enumerate(observed_ids)}
    y_remapped = np.array([id_remap[v] for v in y])

    return {
        'X': X,
        'y': y_remapped,
        'actual_feature_names': actual_feature_names,
        'groups': groups,
        'class_names': class_names,
        'num_classes': len(class_names),
    }



def train_all(fp_weight=2.0, seed=None, feature_names=None,
              feature_importance=False, ablation=False, shap_samples=200):
    """Train models with all available data.

    Args:
        fp_weight: Multiplier for class 0 (IDLE) weight. Values >1.0 penalize
                   false positives more, producing a more conservative model.
        seed: Optional random seed for reproducible training.
        feature_names: List of feature names to use. If None, uses DEFAULT_FEATURES.
        feature_importance: If True, calculate and display SHAP feature importance.
        ablation: If True, run ablation study instead of training.
        shap_samples: Number of samples for SHAP analysis.
    """
    from ml_detector import ML_SUBCARRIERS
    subcarriers = ML_SUBCARRIERS
    hidden_layers = [24]

    print("\n" + "="*60)
    print("            ML MOTION DETECTOR TRAINING")
    print("="*60 + "\n")
    print(f"Subcarriers: {subcarriers}\n")

    seed = _setup_training_runtime(seed=seed)
    if seed is None:
        return 1

    prep = _prepare_motion_training_data(subcarriers=subcarriers, feature_names=feature_names)
    if prep is None:
        return 1

    from sklearn.preprocessing import StandardScaler
    X = prep['X']
    y = prep['y']
    actual_feature_names = prep['actual_feature_names']
    groups = prep['groups']
    class_names = prep['class_names']
    num_classes = prep['num_classes']

    print(f"  Samples: {len(X)}")
    print(f"  Features: {len(actual_feature_names)}")
    print(f"  Feature set: {', '.join(actual_feature_names)}")
    print(f"\nClasses ({num_classes}): {class_names}")
    for new_id, name in enumerate(class_names):
        print(f"  {name} (class {new_id}): {np.sum(y == new_id)} samples")

    if ablation:
        run_ablation_study(X, y, num_classes, class_names, actual_feature_names,
                           hidden_layers=hidden_layers, fp_weight=fp_weight,
                           groups=groups)
        return 0

    if fp_weight != 1.0:
        print(f"\nFP weight: {fp_weight}x (penalizing false positives)")
    arch_str = ' -> '.join(map(str, [12] + hidden_layers + [num_classes]))
    print(f"\n5-fold cross-validation ({arch_str})...")
    with suppress_stderr():
        cv_results = cross_validate(X, y, num_classes, class_names, hidden_layers=hidden_layers,
                                    n_folds=5, max_epochs=200, fp_weight=fp_weight,
                                    groups=groups)

    print(f"  Accuracy:  {cv_results['accuracy_mean']:.1f}% (+/- {cv_results['accuracy_std']:.1f}%)")
    print(f"  F1 Score:  {cv_results['f1_mean']:.1f}% (+/- {cv_results['f1_std']:.1f}%)")
    if 'motion_recall_mean' in cv_results and 'fp_rate_mean' in cv_results:
        print(f"  Motion Recall: {cv_results['motion_recall_mean']:.1f}% "
              f"(+/- {cv_results.get('motion_recall_std', 0.0):.1f}%)")
        print(f"  FP Rate:       {cv_results['fp_rate_mean']:.1f}% "
              f"(+/- {cv_results.get('fp_rate_std', 0.0):.1f}%)")

    # Fixed selection metric aligned with performance tests.
    selection_score = cv_results.get('motion_recall_mean', 0.0) - cv_results.get('fp_rate_mean', 0.0)
    selection_desc = "motion_recall - fp_rate"
    print(f"  Selection score ({selection_desc}): {selection_score:.2f}")

    unique_groups = np.unique(groups)
    if len(unique_groups) >= 2:
        X_train_raw, X_test_raw, y_train, y_test = split_holdout(
            X, y, test_size=0.2, random_state=seed, groups=groups
        )
    else:
        # Fallback for edge case with a single source group.
        print("Warning: only one source group available; using non-group split.")
        X_train_raw, X_test_raw, y_train, y_test = split_holdout(
            X, y, test_size=0.2, random_state=seed
        )

    print("\nTraining final model on full dataset...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    with suppress_stderr():
        model = train_model(X_scaled, y, num_classes, hidden_layers=hidden_layers,
                            max_epochs=200, fp_weight=fp_weight)

    X_test_scaled = scaler.transform(X_test_raw)
    with suppress_stderr():
        test_metrics = evaluate_model_multiclass(model, X_test_scaled, y_test, class_names)

    print(f"\nHold-out test set (20%):")
    print(f"  Accuracy:  {test_metrics['accuracy']:.1f}%")
    print(f"  F1 Score:  {test_metrics['f1']:.1f}%")
    print(f"  Per-class accuracy:")
    for name, acc in test_metrics['per_class'].items():
        print(f"    {name}: {acc:.1f}%")

    if feature_importance:
        importance = calculate_shap_importance(model, X_scaled, actual_feature_names,
                                               n_samples=shap_samples)
        if importance:
            print_feature_importance(importance)

    print("\nExporting models...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    with suppress_stderr():
        tflite_path, tflite_size = export_tflite(model, X_scaled, MODELS_DIR, 'small')
    print(f"  TFLite: {tflite_path.name} ({tflite_size/1024:.1f} KB)")

    mp_path = SRC_DIR / 'ml_weights.py'
    mp_size = export_micropython(model, scaler, mp_path, seed=seed, class_names=class_names,
                                 py_prefix='', generator_script='10_train_motion_model.py')
    print(f"  MicroPython weights: {mp_path.name} ({mp_size/1024:.1f} KB)")

    cpp_path = CPP_DIR / 'ml_weights.h'
    cpp_size = export_cpp_weights(model, scaler, cpp_path, seed=seed, class_names=class_names,
                                  cpp_prefix='ML_', generator_script='10_train_motion_model.py')
    print(f"  C++ weights: {cpp_path.name} ({cpp_size/1024:.1f} KB)")

    scaler_path = MODELS_DIR / 'feature_scaler.npz'
    np.savez(scaler_path, mean=scaler.mean_, scale=scaler.scale_)
    print(f"  Scaler: {scaler_path.name}")

    with suppress_stderr():
        test_data_path = MODELS_DIR / 'ml_test_data.npz'
        n_test = export_test_data(model, scaler, X_test_raw, y_test, test_data_path)
    print(f"  Test data: {test_data_path.name} ({n_test} samples)")

    print("\n" + "="*60)
    print("                    DONE!")
    print("="*60)
    print(f"\nModel trained with CV F1={cv_results['f1_mean']:.1f}% (+/- {cv_results['f1_std']:.1f}%)")
    print(f"Selection metric ({selection_desc}): {selection_score:.2f}")
    print(f"Classes: {class_names}")
    print(f"\nGenerated files:")
    print(f"  - {mp_path} (MicroPython)")
    print(f"  - {cpp_path} (C++ ESPHome)")
    print(f"  - {tflite_path} (ESP-IDF TFLite)")
    print(f"  - {scaler_path} (normalization params)")
    print(f"  - {test_data_path} (test data for validation)")
    print()

    return 0


def run_experiment(seed=None, feature_names=None):
    """Compare multiple MLP architectures using cross-validation."""
    from ml_detector import ML_SUBCARRIERS
    subcarriers = ML_SUBCARRIERS

    seed = _setup_training_runtime(seed=seed)
    if seed is None:
        return 1

    prep = _prepare_motion_training_data(subcarriers=subcarriers, feature_names=feature_names)
    if prep is None:
        return 1

    X = prep['X']
    y = prep['y']
    actual_feature_names = prep['actual_feature_names']
    groups = prep['groups']
    class_names = prep['class_names']
    num_classes = prep['num_classes']

    experiment_architectures(
        X, y, num_classes, class_names,
        num_features=len(actual_feature_names),
        title="STAGE 1 ARCHITECTURE EXPERIMENT",
        groups=groups
    )
    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Train ML motion model (binary IDLE/MOTION detector)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
This script trains a binary IDLE/MOTION detector.
Only baseline and movement labels are used (gesture labels are excluded).

Examples:
  python tools/10_train_motion_model.py                    # Train with default features
  python tools/10_train_motion_model.py --info             # Show dataset info
  python tools/10_train_motion_model.py --experiment       # Compare architectures
  python tools/10_train_motion_model.py --fp-weight 2.0    # Penalize FP 2x more
  python tools/10_train_motion_model.py --seed 42          # Reproducible training
  python tools/10_train_motion_model.py --shap             # Show SHAP feature importance

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
                       help='Calculate and display SHAP feature importance')
    parser.add_argument('--shap-samples', type=int, default=200,
                       help='Number of samples for SHAP analysis (default: 200)')
    parser.add_argument('--correlation', action='store_true',
                       help='Calculate correlation of features with motion label')
    parser.add_argument('--ablation', action='store_true',
                       help='Run ablation study (test removing each feature)')
    args = parser.parse_args()

    if args.info:
        show_info()
        return 0

    if args.experiment:
        return run_experiment(seed=args.seed, feature_names=TRAINING_FEATURES)

    if args.correlation:
        correlations = calculate_correlation_importance()
        if correlations:
            print_correlation_table(correlations, TRAINING_FEATURES)
        return 0

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
