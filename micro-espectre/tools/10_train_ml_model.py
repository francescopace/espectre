#!/usr/bin/env python3
"""
ML Motion Detection - Training Script

Trains neural network models for motion detection using all available CSI data.
Generates models for both ESP-IDF (TFLite) and MicroPython.

Usage:
    python tools/10_train_ml_model.py              # Train with all data
    python tools/10_train_ml_model.py --info       # Show dataset info

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

# Add paths for imports
_src_path = str(Path(__file__).parent.parent / 'src')
_micro_espectre_path = str(Path(__file__).parent.parent)
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)
if _micro_espectre_path not in sys.path:
    sys.path.insert(0, _micro_espectre_path)

from csi_utils import (
    load_baseline_and_movement,
    load_npz_as_packets,
    DATA_DIR,
)
from segmentation import SegmentationContext
from features import calc_skewness, calc_kurtosis, calc_iqr_turb, calc_entropy_turb

# Directories
MODELS_DIR = Path(__file__).parent.parent / 'models'
SRC_DIR = Path(__file__).parent.parent / 'src'
CPP_DIR = Path(__file__).parent.parent.parent / 'components' / 'espectre'

# Feature names (12 features)
FEATURE_NAMES = [
    'turb_mean', 'turb_std', 'turb_max', 'turb_min', 'turb_range', 'turb_variance',
    'turb_iqr', 'turb_entropy', 'amp_skewness', 'amp_kurtosis', 'turb_slope', 'turb_delta'
]

# Default subcarriers (consecutive band, as selected by NBVI/P95)
DEFAULT_SUBCARRIERS = list(range(11, 23))


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


def load_all_data():
    """
    Load all available CSI data from the data/ directory.
    
    Reads label from npz file metadata (not folder structure).
    Uses dataset_info.json to determine if label is motion or idle.
    
    Returns:
        tuple: (all_packets, stats) where stats is a dict with dataset info
    """
    all_packets = []
    stats = {'chips': set(), 'labels': {}, 'total': 0}
    
    # Load dataset info for label mapping
    dataset_info = load_dataset_info()
    
    # Scan all subdirectories in data/
    for subdir in DATA_DIR.iterdir():
        if not subdir.is_dir() or subdir.name.startswith('.'):
            continue
        
        # Load all npz files in this directory
        for npz_file in subdir.glob('*.npz'):
            try:
                packets = load_npz_as_packets(npz_file)
                if not packets:
                    continue
                
                # Get label from file metadata (already set by load_npz_as_packets)
                label = packets[0].get('label', subdir.name)
                
                # Track stats
                if label not in stats['labels']:
                    stats['labels'][label] = 0
                stats['labels'][label] += len(packets)
                stats['total'] += len(packets)
                
                # Get chip
                chip = packets[0].get('chip', 'unknown').upper()
                stats['chips'].add(chip)
                
                # Add motion flag based on dataset_info
                is_motion = is_motion_label(label, dataset_info)
                for p in packets:
                    p['is_motion'] = is_motion
                
                all_packets.extend(packets)
                
            except Exception as e:
                print(f"  Warning: Could not load {npz_file.name}: {e}")
    
    stats['chips'] = sorted(stats['chips'])
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
        # Calculate turbulence
        turb, amps = SegmentationContext.compute_spatial_turbulence(
            pkt['csi_data'], subcarriers
        )
        turb_buffer.append(turb)
        
        # Wait for buffer to fill
        if len(turb_buffer) < window_size:
            continue
        
        turb_list = list(turb_buffer)
        
        # Extract 12 features
        features = [
            np.mean(turb_list),                              # turb_mean
            np.std(turb_list),                               # turb_std
            np.max(turb_list),                               # turb_max
            np.min(turb_list),                               # turb_min
            np.max(turb_list) - np.min(turb_list),          # turb_range
            np.var(turb_list),                               # turb_variance
            calc_iqr_turb(turb_list, len(turb_list)),       # turb_iqr
            calc_entropy_turb(turb_list, len(turb_list)),   # turb_entropy
            calc_skewness(list(amps)) if amps is not None else 0,  # amp_skewness
            calc_kurtosis(list(amps)) if amps is not None else 0,  # amp_kurtosis
            np.polyfit(range(len(turb_list)), turb_list, 1)[0],    # turb_slope
            turb_list[-1] - turb_list[0],                   # turb_delta
        ]
        
        X.append(features)
        # Label: 0 = IDLE, 1 = MOTION (from metadata)
        y.append(1 if pkt.get('is_motion', False) else 0)
    
    return np.array(X), np.array(y)


# ============================================================================
# Model Training
# ============================================================================

def train_model(X, y, hidden_layers=[16, 8], epochs=50):
    """
    Train a neural network model.
    
    Args:
        X: Feature matrix (normalized)
        y: Labels
        hidden_layers: List of hidden layer sizes
        epochs: Training epochs
    
    Returns:
        Trained Keras model
    """
    import tensorflow as tf
    
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(12,)))
    
    for units in hidden_layers:
        model.add(tf.keras.layers.Dense(units, activation='relu'))
    
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=epochs, batch_size=32, validation_split=0.1, verbose=0)
    
    return model


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


def export_micropython(model, scaler, output_path):
    """
    Export model weights to MicroPython code.
    
    Generates ml_weights.py with network weights only.
    The inference functions are in ml_detector.py (not auto-generated).
    
    Args:
        model: Trained Keras model
        scaler: StandardScaler with mean_ and scale_
        output_path: Output file path
    
    Returns:
        Size of generated code
    """
    weights = model.get_weights()
    
    # Build code - weights only
    code = '''"""
Micro-ESPectre - ML Model Weights

Auto-generated neural network weights for motion detection.
Architecture: 12 -> ''' + ' -> '.join(str(w.shape[1]) for w in weights[::2]) + '''

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


def export_cpp_weights(model, scaler, output_path):
    """
    Export model weights to C++ header for ESPHome.
    
    Generates ml_weights.h with constexpr weights.
    
    Args:
        model: Trained Keras model
        scaler: StandardScaler with mean_ and scale_
        output_path: Output file path
    
    Returns:
        Size of generated code
    """
    weights = model.get_weights()
    arch = ' -> '.join(str(w.shape[1]) for w in weights[::2])
    
    code = f'''/*
 * ESPectre - ML Model Weights
 * 
 * Auto-generated neural network weights for motion detection.
 * Architecture: 12 -> {arch}
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


def train_all():
    """
    Train models with all available data.
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
    print("Loading data from npz metadata...")
    all_packets, stats = load_all_data()
    
    if not stats['chips']:
        print("Error: No datasets found in data/")
        print("Collect data using: ./me collect --label baseline --duration 60")
        return 1
    
    print(f"  Chips: {', '.join(stats['chips'])}")
    for label, count in sorted(stats['labels'].items()):
        print(f"  {label}: {count} packets")
    print(f"  Total: {stats['total']} packets")
    
    # Extract features
    print("\nExtracting features...")
    X, y = extract_features(all_packets, subcarriers=subcarriers)
    print(f"  Samples: {len(X)}")
    print(f"  Features: {len(FEATURE_NAMES)}")
    print(f"  Class balance: IDLE={np.sum(y==0)}, MOTION={np.sum(y==1)}")
    
    # Split for evaluation (BEFORE normalization to preserve raw test features)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Normalize (fit on training data only, apply to test)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    
    # Also scale full dataset for final training
    X_scaled = scaler.fit_transform(X)
    
    # Train model (best architecture: 16 -> 8 -> 1)
    print("\nTraining neural network (12 -> 16 -> 8 -> 1)...")
    with suppress_stderr():
        model = train_model(X_train, y_train, hidden_layers=[16, 8], epochs=50)
    
    # Evaluate
    with suppress_stderr():
        y_pred = (model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
    tp = np.sum((y_test == 1) & (y_pred == 1))
    fp = np.sum((y_test == 0) & (y_pred == 1))
    tn = np.sum((y_test == 0) & (y_pred == 0))
    fn = np.sum((y_test == 1) & (y_pred == 0))
    
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    fp_rate = fp / (fp + tn) * 100 if (fp + tn) > 0 else 0
    f1 = 2 * tp / (2 * tp + fp + fn) * 100 if (2 * tp + fp + fn) > 0 else 0
    
    print(f"\nEvaluation (20% test set):")
    print(f"  Recall:    {recall:.1f}%")
    print(f"  Precision: {precision:.1f}%")
    print(f"  FP Rate:   {fp_rate:.1f}%")
    print(f"  F1 Score:  {f1:.1f}%")
    
    # Retrain on full dataset for production
    print("\nRetraining on full dataset...")
    with suppress_stderr():
        model = train_model(X_scaled, y, hidden_layers=[16, 8], epochs=50)
    
    # Export models
    print("\nExporting models...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # TFLite (suppress C++ warnings during conversion)
    with suppress_stderr():
        tflite_path, tflite_size = export_tflite(model, X_scaled, MODELS_DIR, 'small')
    print(f"  TFLite: {tflite_path.name} ({tflite_size/1024:.1f} KB)")
    
    # MicroPython weights
    mp_path = SRC_DIR / 'ml_weights.py'
    mp_size = export_micropython(model, scaler, mp_path)
    print(f"  MicroPython weights: {mp_path.name} ({mp_size/1024:.1f} KB)")
    
    # C++ weights for ESPHome
    cpp_path = CPP_DIR / 'ml_weights.h'
    cpp_size = export_cpp_weights(model, scaler, cpp_path)
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
    print(f"\nModel trained with F1={f1:.1f}%")
    print(f"\nGenerated files:")
    print(f"  - {mp_path} (MicroPython)")
    print(f"  - {cpp_path} (C++ ESPHome)")
    print(f"  - {tflite_path} (ESP-IDF TFLite)")
    print(f"  - {scaler_path} (normalization params)")
    print(f"  - {test_data_path} (test data for validation)")
    print()
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Train ML motion detection model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python tools/10_train_ml_model.py           # Train with fixed subcarriers
  python tools/10_train_ml_model.py --info    # Show dataset info
  
To compare ML with MVS, use:
  python tools/7_compare_detection_methods.py
'''
    )
    parser.add_argument('--info', action='store_true', 
                       help='Show dataset information')
    
    args = parser.parse_args()
    
    if args.info:
        show_info()
        return 0
    
    return train_all()


if __name__ == '__main__':
    exit(main())
