#!/usr/bin/env python3
"""
ML training infrastructure shared between motion and gesture training scripts.

Provides model building, training, evaluation, export, and analysis utilities.
All TF/sklearn imports are lazy (inside functions) to keep import time fast.

Used by:
  - tools/10_train_motion_model.py   (binary IDLE/MOTION detector)
  - tools/11_train_gesture_model.py  (gesture classifier)

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import os
import sys
from contextlib import contextmanager

import numpy as np


# ============================================================================
# Utilities
# ============================================================================

@contextmanager
def suppress_stderr():
    """Context manager to suppress stderr at the file-descriptor level.

    Necessary because TensorFlow's C++ code writes directly to the C-level
    stderr, bypassing Python's sys.stderr.
    """
    stderr_fd = sys.stderr.fileno()
    saved = os.dup(stderr_fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, stderr_fd)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(saved, stderr_fd)
        os.close(saved)


def generate_seed():
    """Generate a non-deterministic random seed from OS entropy."""
    from numpy.random import SeedSequence
    ss = SeedSequence()
    return int(ss.entropy % (2**31))


def setup_tf_logging(seed=None):
    """Suppress TensorFlow Python-level and absl logging, and set random seeds.

    Call after importing tensorflow. TF C++ warnings must be suppressed via
    env vars (TF_CPP_MIN_LOG_LEVEL etc.) before any import.

    Args:
        seed: Optional random seed. If provided, sets both numpy and TF seeds.
    """
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    try:
        import absl.logging
        absl.logging.set_verbosity(absl.logging.ERROR)
        absl.logging.set_stderrthreshold(absl.logging.ERROR)
    except ImportError:
        pass
    if seed is not None:
        np.random.seed(seed)
        tf.random.set_seed(seed)


# ============================================================================
# Model Building and Training
# ============================================================================

def build_model(num_classes, hidden_layers, num_features=12,
                use_dropout=True, dropout_rate=0.2):
    """Build a Keras MLP model.

    For binary classification (num_classes=2), uses a single sigmoid output
    which is more efficient than softmax. For multiclass, uses softmax.

    Args:
        num_classes: Number of output classes.
        hidden_layers: List of hidden layer sizes.
        num_features: Number of input features.
        use_dropout: Whether to add dropout layers (training only).
        dropout_rate: Dropout rate (0.0-1.0).

    Returns:
        Compiled Keras model.
    """
    import tensorflow as tf

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(num_features,)))
    for units in hidden_layers:
        model.add(tf.keras.layers.Dense(units, activation='relu'))
        if use_dropout and dropout_rate > 0:
            model.add(tf.keras.layers.Dropout(dropout_rate))

    if num_classes == 2:
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy'],
        )
    else:
        model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
        )
    return model


def train_model(X, y, num_classes=None, hidden_layers=None, max_epochs=200,
                use_dropout=True, class_weight=None, fp_weight=1.0, verbose=0,
                batch_size=32, validation_split=0.1):
    """Train a neural network with early stopping and LR reduction.

    Args:
        X: Feature matrix (normalized).
        y: Labels (class IDs).
        num_classes: Number of output classes (derived from y if None).
        hidden_layers: List of hidden layer sizes (default: [16, 8]).
        max_epochs: Maximum training epochs.
        use_dropout: Whether to add dropout layers.
        class_weight: Class weight dict or None for auto-balanced.
        fp_weight: Multiplier for class 0 (IDLE) weight to penalize FP.
        verbose: Training verbosity.
        batch_size: Mini-batch size.
        validation_split: Fraction of data held out for validation.

    Returns:
        Trained Keras model.
    """
    import tensorflow as tf
    from sklearn.utils.class_weight import compute_class_weight

    if num_classes is None:
        num_classes = len(np.unique(y))
    if hidden_layers is None:
        hidden_layers = [16, 8]

    if class_weight is None:
        unique_classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=unique_classes, y=y)
        class_weight = dict(zip(unique_classes.tolist(), weights.tolist()))

    if fp_weight != 1.0 and 0 in class_weight:
        class_weight[0] *= fp_weight

    num_features = X.shape[1] if hasattr(X, 'shape') else len(X[0])
    model = build_model(num_classes, hidden_layers, num_features=num_features,
                        use_dropout=use_dropout)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=15, restore_best_weights=True, min_delta=1e-4,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6,
        ),
    ]
    model.fit(
        X, y,
        epochs=max_epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=verbose,
    )
    return model


def evaluate_model_multiclass(model, X_test, y_test, class_names):
    """Evaluate a model on test data.

    Handles both binary (sigmoid) and multiclass (softmax) models.

    Args:
        model: Trained Keras model (sigmoid or softmax output).
        X_test: Test features (normalized).
        y_test: Test labels (class IDs).
        class_names: List of class names indexed by class_id.

    Returns:
        dict with keys:
          - accuracy, per_class, confusion, report, f1 (macro)
          - motion_recall, motion_precision, fp_rate (binary only, else 0.0)
    """
    from sklearn.metrics import confusion_matrix, classification_report

    probs = model.predict(X_test, verbose=0)

    if len(class_names) == 2 and probs.shape[1] == 1:
        y_pred = (probs[:, 0] > 0.5).astype(int)
    else:
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
    # Binary-specific metrics aligned with runtime performance tests:
    # class 0 = idle, class 1 = motion
    motion_recall = 0.0
    motion_precision = 0.0
    fp_rate = 0.0
    if len(class_names) == 2 and cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        motion_recall = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
        motion_precision = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
        fp_rate = (fp / (tn + fp) * 100.0) if (tn + fp) > 0 else 0.0

    return {
        'accuracy': accuracy,
        'per_class': per_class,
        'confusion': cm,
        'report': report,
        'f1': report.get('macro avg', {}).get('f1-score', 0) * 100,
        'motion_recall': motion_recall,
        'motion_precision': motion_precision,
        'fp_rate': fp_rate,
    }


def cross_validate(X, y, num_classes, class_names, hidden_layers=None, n_folds=5,
                   max_epochs=200, fp_weight=1.0, batch_size=32, validation_split=0.1,
                   groups=None):
    """Perform stratified k-fold cross-validation.

    Args:
        X: Feature matrix (NOT normalized — scaler fit per fold).
        y: Labels (class IDs).
        num_classes: Number of output classes.
        class_names: Class names for evaluation report.
        hidden_layers: List of hidden layer sizes (default: [16, 8]).
        n_folds: Number of CV folds.
        max_epochs: Maximum training epochs per fold.
        fp_weight: Multiplier for class 0 weight (>1.0 penalizes FP more).
        batch_size: Mini-batch size.
        validation_split: Fraction held out per fold for early stopping.
        groups: Optional group IDs (e.g., source file/session) for group-aware CV.

    Returns:
        dict: Mean and std of each scalar metric across folds
              (accuracy_mean, accuracy_std, f1_mean, f1_std).
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    if hidden_layers is None:
        hidden_layers = [16, 8]

    if groups is not None:
        groups = np.asarray(groups)
        try:
            from sklearn.model_selection import StratifiedGroupKFold
            splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
        except ImportError:
            # Backward compatibility with older scikit-learn versions.
            from sklearn.model_selection import GroupKFold
            splitter = GroupKFold(n_splits=n_folds)
        split_iter = splitter.split(X, y, groups=groups)
    else:
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        split_iter = splitter.split(X, y)
    fold_metrics = []

    for train_idx, val_idx in split_iter:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_val = scaler.transform(X[val_idx])
        with suppress_stderr():
            model = train_model(
                X_train, y[train_idx], num_classes,
                hidden_layers=hidden_layers, max_epochs=max_epochs,
                fp_weight=fp_weight, batch_size=batch_size,
                validation_split=validation_split,
            )
            metrics = evaluate_model_multiclass(model, X_val, y[val_idx], class_names)
        fold_metrics.append(metrics)

    result = {}
    scalar_keys = [k for k, v in fold_metrics[0].items() if isinstance(v, (int, float))]
    for key in scalar_keys:
        values = [m[key] for m in fold_metrics]
        result[f'{key}_mean'] = np.mean(values)
        result[f'{key}_std'] = np.std(values)
    return result


def split_holdout(X, y, test_size=0.2, random_state=42, groups=None):
    """Split data into train/test with optional group awareness.

    Args:
        X: Feature matrix.
        y: Labels array.
        test_size: Fraction for test split.
        random_state: RNG seed.
        groups: Optional group IDs to prevent leakage across related samples.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    if groups is not None:
        groups = np.asarray(groups)
        unique_groups = np.unique(groups)
        if len(unique_groups) >= 2:
            from sklearn.model_selection import GroupShuffleSplit
            gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            train_idx, test_idx = next(gss.split(X, y, groups=groups))
            return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

    from sklearn.model_selection import train_test_split
    stratify = y if len(np.unique(y)) > 1 else None
    try:
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)
    except ValueError:
        # Fallback when stratified split is not feasible (e.g., too few samples per class)
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=None)


# ============================================================================
# Model Export
# ============================================================================

def export_tflite(model, X_sample, output_path, name):
    """Export model to TFLite with int8 quantization.

    Args:
        model: Trained Keras model.
        X_sample: Sample data for quantization calibration.
        output_path: Output directory (Path).
        name: Model name suffix for filename.

    Returns:
        tuple: (Path to .tflite file, file size in bytes).
    """
    import tensorflow as tf
    import warnings

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

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        tflite_model = converter.convert()

    tflite_path = output_path / f'motion_detector_{name}.tflite'
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    return tflite_path, len(tflite_model)


def export_micropython(model, scaler, output_path, seed, class_names,
                       py_prefix='', generator_script=''):
    """Export model weights to MicroPython code.

    Args:
        model: Trained Keras model.
        scaler: StandardScaler with mean_ and scale_.
        output_path: Output file path (Path).
        seed: Random seed used for training.
        class_names: List of class names.
        py_prefix: Variable name prefix.
                   '' for motion-model exports (FEATURE_MEAN, CLASS_LABELS, ...),
                   'GESTURE_' for gesture-model exports (GESTURE_FEATURE_MEAN, ...).
        generator_script: Name of the generating script for the file header.

    Returns:
        Size of generated code in bytes.
    """
    from datetime import datetime

    weights = model.get_weights()
    num_classes = len(class_names) if class_names else 2
    num_inputs = weights[0].shape[0]
    arch = ' -> '.join(str(w.shape[1]) for w in weights[::2])
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    p = py_prefix

    code = f'''"""
Micro-ESPectre - ML Model Weights

Auto-generated neural network weights for motion/gesture detection.
Architecture: {num_inputs} -> {arch}
Classes: {num_classes}
Trained: {timestamp}
Seed: {seed}

This file is auto-generated by {generator_script}.
DO NOT EDIT - your changes will be overwritten!

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

# Feature normalization (StandardScaler)
{p}FEATURE_MEAN = [{', '.join(f'{x:.6f}' for x in scaler.mean_)}]
{p}FEATURE_SCALE = [{', '.join(f'{x:.6f}' for x in scaler.scale_)}]

# Class labels (indexed by class_id)
{p}CLASS_LABELS = {class_names}
{p}NUM_CLASSES = {num_classes}

'''
    for i in range(0, len(weights), 2):
        W = weights[i]
        b = weights[i + 1]
        layer_num = i // 2 + 1
        in_size, out_size = W.shape
        if i == len(weights) - 2:
            activation = 'Sigmoid' if (num_classes == 2 and out_size == 1) else 'Softmax'
        else:
            activation = 'ReLU'
        code += f'# Layer {layer_num}: {in_size} -> {out_size} ({activation})\n'
        code += f'W{layer_num} = [\n'
        for row in W:
            code += '    [' + ', '.join(f'{x:.6f}' for x in row) + '],\n'
        code += ']\n'
        code += f'B{layer_num} = [' + ', '.join(f'{x:.6f}' for x in b) + ']\n\n'

    with open(output_path, 'w') as f:
        f.write(code)
    return len(code)


def export_cpp_weights(model, scaler, output_path, seed, class_names,
                       cpp_prefix='ML_', generator_script=''):
    """Export model weights to C++ header for ESPHome.

    Args:
        model: Trained Keras model.
        scaler: StandardScaler with mean_ and scale_.
        output_path: Output file path (Path).
        seed: Random seed used for training.
        class_names: List of class names.
        cpp_prefix: C++ variable name prefix.
                    'ML_' for motion-model exports (ML_FEATURE_MEAN, ML_W1, ...),
                    'GESTURE_' for gesture-model exports (GESTURE_FEATURE_MEAN, GESTURE_W1, ...).
        generator_script: Name of the generating script for the file header.

    Returns:
        Size of generated code in bytes.
    """
    from datetime import datetime

    weights = model.get_weights()
    arch = ' -> '.join(str(w.shape[1]) for w in weights[::2])
    num_inputs = weights[0].shape[0]
    num_classes = len(class_names) if class_names else 2
    n_features = len(scaler.mean_)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    p = cpp_prefix

    code = f'''/*
 * ESPectre - ML Model Weights
 *
 * Auto-generated neural network weights for motion/gesture detection.
 * Architecture: {num_inputs} -> {arch}
 * Classes: {num_classes}
 * Trained: {timestamp}
 * Seed: {seed}
 *
 * This file is auto-generated by {generator_script}.
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
constexpr float {p}FEATURE_MEAN[{n_features}] = {{{', '.join(f'{x:.6f}f' for x in scaler.mean_)}}};
constexpr float {p}FEATURE_SCALE[{n_features}] = {{{', '.join(f'{x:.6f}f' for x in scaler.scale_)}}};

constexpr int {p}NUM_CLASSES = {num_classes};

// Class labels (indexed by class_id)
constexpr const char* {p}CLASS_LABELS[{num_classes}] = {{
'''
    for name in (class_names or [f'class_{i}' for i in range(num_classes)]):
        code += f'    "{name}",\n'
    code += '};\n\n'

    for i in range(0, len(weights), 2):
        W = weights[i]
        b = weights[i + 1]
        layer_num = i // 2 + 1
        in_size, out_size = W.shape
        if i == len(weights) - 2:
            activation = 'Sigmoid' if (num_classes == 2 and out_size == 1) else 'Softmax'
        else:
            activation = 'ReLU'
        code += f'// Layer {layer_num}: {in_size} -> {out_size} ({activation})\n'
        code += f'constexpr float {p}W{layer_num}[{in_size}][{out_size}] = {{\n'
        for row in W:
            code += '    {' + ', '.join(f'{x:.6f}f' for x in row) + '},\n'
        code += '};\n'
        code += (f'constexpr float {p}B{layer_num}[{out_size}] = '
                 f'{{{", ".join(f"{x:.6f}f" for x in b)}}};\n\n')

    code += '''}  // namespace espectre
}  // namespace esphome
'''
    with open(output_path, 'w') as f:
        f.write(code)
    return len(code)


def export_test_data(model, scaler, X_test_raw, y_test, output_path):
    """Export test data for cross-platform (Python/C++) validation.

    Saves RAW features (not normalized) and expected model outputs so tests
    can verify the full pipeline including normalization.

    Args:
        model: Trained Keras model.
        scaler: StandardScaler used for normalization.
        X_test_raw: Test features (NOT normalized).
        y_test: Test labels.
        output_path: Output file path (Path).

    Returns:
        Number of test samples.
    """
    X_test_scaled = scaler.transform(X_test_raw)
    probs = model.predict(X_test_scaled, verbose=0)
    # For binary sigmoid: output is already prob[motion]
    # For softmax: use 1 - prob[idle]
    if probs.shape[1] == 1:
        expected_outputs = probs[:, 0].astype(np.float32)
    else:
        expected_outputs = (1.0 - probs[:, 0]).astype(np.float32)
    np.savez(output_path,
             features=X_test_raw.astype(np.float32),
             labels=y_test.astype(np.int32),
             expected_outputs=expected_outputs)
    return len(X_test_raw)


# ============================================================================
# Feature Importance (SHAP and Correlation)
# ============================================================================

def calculate_shap_importance(model, X, feature_names, n_samples=500):
    """Calculate SHAP feature importance using the permutation algorithm.

    Args:
        model: Trained Keras model.
        X: Feature matrix (normalized).
        feature_names: List of feature names.
        n_samples: Number of samples to explain (lower = faster).

    Returns:
        dict: {feature_name: mean_abs_shap_value} sorted by importance,
              or None if shap is not installed.
    """
    try:
        import shap
    except ImportError:
        print("Error: SHAP not installed. Run: pip install shap")
        return None

    print("\nCalculating SHAP feature importance...")
    print("  (This may take 1-2 minutes)")

    n_background = min(100, len(X))
    background = X[np.random.choice(len(X), n_background, replace=False)]
    n_explain = min(n_samples, len(X))
    X_explain = X[np.random.choice(len(X), n_explain, replace=False)]

    explainer = shap.Explainer(model.predict, background, algorithm='permutation')
    with suppress_stderr():
        shap_values = explainer(X_explain).values

    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    if len(shap_values.shape) > 2:
        shap_values = shap_values.squeeze()

    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    importance = {name: float(val) for name, val in zip(feature_names, mean_abs_shap)}
    return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))


def print_feature_importance(importance, title="Feature Importance (SHAP)",
                             current_features=None):
    """Print feature importance table with visual bars.

    Args:
        importance: Dict of {feature_name: importance_value}.
        title: Title for the table.
        current_features: Optional list of features currently in use (marks USED).
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
        pct = value / total * 100
        bar = '█' * int(pct / 2.5)
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

    sorted_features = list(importance.keys())
    low_importance = [f for f in sorted_features if importance[f] / total < 0.03]
    high_importance = sorted_features[:3]

    print("Recommendations:")
    print(f"  Most important: {', '.join(high_importance)}")
    if low_importance:
        print(f"  Low importance (<3%): {', '.join(low_importance)}")

    if current_features:
        top_unused = [f for f in sorted_features[:10] if f not in current_features]
        low_used = [f for f in sorted_features
                    if f in current_features and importance[f] / total < 0.05]
        if top_unused:
            print(f"  Top unused features: {', '.join(top_unused[:5])}")
        if low_used:
            print(f"  Low importance but USED: {', '.join(low_used)}")
    print()


def print_correlation_table(correlations, current_features=None):
    """Print correlation results in a nice table.

    Args:
        correlations: Dict of {feature_name: correlation_with_label}.
        current_features: Optional list of features currently in use (marks USED).
    """
    print("\n" + "=" * 74)
    print("  Feature Correlation with Motion Label")
    print("=" * 74)
    print(f"{'Rank':<5} {'Feature':<22} {'Corr':>8} {'|Corr|':>8} {'Status':<12}")
    print("-" * 74)

    for rank, (fname, corr) in enumerate(correlations.items(), 1):
        status = "USED" if (current_features and fname in current_features) else ""
        bar = '█' * int(abs(corr) * 20)
        print(f"{rank:<5} {fname:<22} {corr:>+8.4f} {abs(corr):>8.4f} {status:<12} {bar}")

    print("-" * 74)
    print("\nRecommendations:")
    sorted_items = list(correlations.items())
    top_unused = [(f, c) for f, c in sorted_items
                  if not current_features or f not in current_features][:3]
    if top_unused:
        print(f"  Top unused features: {', '.join(f[0] for f in top_unused)}")
    if current_features:
        low_used = [(f, c) for f, c in sorted_items
                    if f in current_features and abs(c) < 0.2]
        if low_used:
            print(f"  Low correlation but used: {', '.join(f[0] for f in low_used)}")


# ============================================================================
# Architecture Experiment
# ============================================================================

def experiment_architectures(X, y, num_classes, class_names, num_features=12,
                             title="ARCHITECTURE EXPERIMENT", groups=None):
    """Compare multiple MLP architectures using cross-validation.

    Args:
        X: Feature matrix (N_samples, num_features).
        y: Labels array (N_samples,).
        num_classes: Number of output classes.
        class_names: List of class names.
        num_features: Number of input features (default: 12).
        title: Title for the experiment output.
        groups: Optional group IDs for group-aware cross-validation.

    Returns:
        dict: Best architecture result with keys: name, layers, f1_mean, etc.
    """
    import time
    from sklearn.preprocessing import StandardScaler

    print("\n" + "=" * 60)
    print(f"       {title}")
    print("=" * 60 + "\n")

    print(f"  Samples: {len(X)}")
    print(f"  Features: {num_features}")
    print(f"  Classes: {num_classes} ({', '.join(class_names)})")
    for cid, name in enumerate(class_names):
        print(f"    {name}: {np.sum(y == cid)} samples")

    architectures = [
        {'name': 'Current (24)', 'layers': [24]},
        {'name': 'Deep (16-8)', 'layers': [16, 8]},
        {'name': 'Deeper (12-8-4)', 'layers': [12, 8, 4]},
        {'name': 'Minimal (8)', 'layers': [8]},
        {'name': 'Wide-deep (24-12)', 'layers': [24, 12]},
    ]

    results = []

    for arch in architectures:
        name = arch['name']
        layers = arch['layers']

        layer_sizes = [num_features] + layers + [num_classes]
        n_params = sum(
            layer_sizes[i] * layer_sizes[i + 1] + layer_sizes[i + 1]
            for i in range(len(layer_sizes) - 1)
        )
        weight_kb = n_params * 4 / 1024
        flops = sum(layer_sizes[i] * layer_sizes[i + 1] for i in range(len(layer_sizes) - 1))

        arch_str = ' -> '.join(map(str, layer_sizes))
        print(f"\nEvaluating: {name} ({arch_str})...")
        print(f"  Parameters: {n_params}, Weights: {weight_kb:.1f} KB, FLOPS: {flops}")

        with suppress_stderr():
            cv = cross_validate(X, y, num_classes, class_names, hidden_layers=layers,
                                n_folds=5, max_epochs=200, groups=groups)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        with suppress_stderr():
            model = train_model(X_scaled, y, num_classes, hidden_layers=layers, max_epochs=200)

        sample = X_scaled[:1].astype(np.float32)
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

    print("\n" + "=" * 75)
    print("                    ARCHITECTURE COMPARISON")
    print("=" * 75 + "\n")

    print(f"{'Architecture':<22} {'Params':>7} {'KB':>6} {'F1 (CV)':>12} {'Accuracy':>10} {'Inf (us)':>10}")
    print("-" * 75)

    best = max(results, key=lambda r: (r['f1_mean'], -r['inference_us']))

    for r in results:
        marker = " **" if r == best else "   "
        print(f"{marker}{r['name']:<19} {r['params']:>7} {r['weight_kb']:>5.1f} "
              f"{r['f1_mean']:>6.1f}+/-{r['f1_std']:<4.1f} "
              f"{r['accuracy_mean']:>9.1f}% "
              f"{r['inference_us']:>9.1f}")

    print("-" * 75)
    print(f"\n** Best architecture: {best['name']}")
    print(f"   F1: {best['f1_mean']:.1f}% +/- {best['f1_std']:.1f}%")
    print(f"   Accuracy: {best['accuracy_mean']:.1f}%")
    print(f"   Parameters: {best['params']}, Weights: {best['weight_kb']:.1f} KB")

    current = next((r for r in results if r['name'].startswith('Current')), None)
    if current and best != current:
        improvement = best['f1_mean'] - current['f1_mean']
        print(f"\n   Improvement over current: {improvement:+.1f}% F1")
        if improvement > 1.0:
            print(f"   Recommendation: Switch to {best['name']}")
            print(f"   Update hidden_layers={best['layers']}")
        else:
            print(f"   Recommendation: Difference is marginal, keep current architecture")
    else:
        print(f"\n   Current architecture is already optimal!")

    print()
    return best
