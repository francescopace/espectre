#!/usr/bin/env python3
"""
ML Motion Detection - Training Script

Trains neural network models for motion detection using all available CSI data.
Generates exported weights for both C++ and MicroPython runtimes.

Training features:
  - Grouped cross-validation with blocked out-of-fold scoring
  - Early stopping with patience to prevent overfitting
  - Dropout regularization during training
  - Balanced class weights for imbalanced datasets
  - Learning rate reduction on plateau
  - Configurable FP penalty (--fp-weight) and feature normalization (--scaler)

Usage:
    python tools/10_train_ml_model.py                    # Train with current production defaults
    python tools/10_train_ml_model.py --info             # Show dataset info
    python tools/10_train_ml_model.py --experiment       # Run the FP-first MLP topology campaign
    python tools/10_train_ml_model.py --experiment --experiment-promote
                                                    # Promote the winner if it beats the baseline
    python tools/10_train_ml_model.py --fp-weight 2.0    # Penalize FP 2x more
    python tools/10_train_ml_model.py --scaler clipped_standard
                                                    # Robust clipping + z-score
    python tools/10_train_ml_model.py --batch-size 32
                                                    # Smaller batch size experiment
    python tools/10_train_ml_model.py --device cuda # Force CUDA when available
    python tools/10_train_ml_model.py --device mps  # Force Apple GPU when available
    python tools/10_train_ml_model.py --shap              # SHAP importance (200 samples)
    python tools/10_train_ml_model.py --shap 500          # SHAP importance (500 samples)

Configuration:
  - TRAINING_FEATURES: Edit at top of file to change feature set

Note: turbulence normalization follows the gain mode: gain-locked streams use
raw std, streams without gain lock use CV normalization (std/mean).

To compare ML with MVS, use:
    python tools/7_compare_detection_methods.py

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import os
import sys

import argparse
import copy
import hashlib
import json
import numpy as np
import random
import subprocess
import re
import shutil
import tempfile
from pathlib import Path
from collections import deque
from dataclasses import dataclass

from repo_paths import (
    cpp_core_dir,
    generated_data_dir,
    models_dir,
    python_src_dir,
    python_tests_dir,
    repo_root,
)
from contextlib import contextmanager
from datetime import datetime
from time import perf_counter

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

TorchModuleBase = nn.Module if nn is not None else object


@contextmanager
def suppress_stderr():
    """
    Context manager to suppress stderr output at the file descriptor level.
    
    Some native libraries write directly to the C-level stderr, bypassing
    Python's sys.stderr.
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


def format_duration(seconds):
    """Render elapsed time in a compact human-readable form."""
    seconds = float(seconds)
    if seconds < 1.0:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 60.0:
        return f"{seconds:.2f} s"
    minutes, rem = divmod(seconds, 60.0)
    return f"{int(minutes)}m {rem:.1f}s"


def derive_seed(base_seed, *offsets):
    """Derive a stable int32-compatible seed from a base seed."""
    if base_seed is None:
        return None
    seed = int(base_seed) & 0x7FFFFFFF
    for offset in offsets:
        seed = (seed * 1103515245 + 12345 + int(offset) * 1009) & 0x7FFFFFFF
    return seed or 1


def set_global_determinism(seed, torch_module=None):
    """
    Best-effort deterministic runtime configuration for a fixed seed.

    This resets Python, NumPy, and PyTorch RNG state immediately before
    stochastic training steps. `PYTHONHASHSEED` only affects new processes,
    but setting it here still documents the intended seed in subprocesses.
    """
    if seed is None:
        return

    seed = int(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch_mod = torch_module if torch_module is not None else torch
    if torch_mod is None:
        return

    torch_mod.manual_seed(seed)
    if torch_mod.cuda.is_available():
        torch_mod.cuda.manual_seed_all(seed)
    try:
        torch_mod.use_deterministic_algorithms(True)
    except (AttributeError, RuntimeError, ValueError):
        pass
    try:
        torch_mod.backends.cudnn.deterministic = True
        torch_mod.backends.cudnn.benchmark = False
    except AttributeError:
        pass


def ensure_torch_available():
    """Return the torch module or raise ImportError with a stable message."""
    if torch is None or nn is None:
        raise ImportError("No module named 'torch'")
    return torch


def set_active_torch_device(device):
    """Set the process-wide PyTorch training device preference."""
    global ACTIVE_TORCH_DEVICE
    ACTIVE_TORCH_DEVICE = str(device or DEFAULT_TORCH_DEVICE).strip().lower()


def resolve_torch_device(device=None, torch_module=None):
    """Resolve a PyTorch device name from cpu/cuda/mps."""
    torch_mod = torch_module if torch_module is not None else ensure_torch_available()
    requested = str(device or ACTIVE_TORCH_DEVICE or DEFAULT_TORCH_DEVICE).strip().lower()
    if requested == 'cuda':
        if not torch_mod.cuda.is_available():
            raise RuntimeError("CUDA device requested, but torch.cuda.is_available() is false")
        return torch_mod.device('cuda')
    if requested == 'mps':
        if not (hasattr(torch_mod.backends, 'mps') and torch_mod.backends.mps.is_available()):
            raise RuntimeError("MPS device requested, but torch.backends.mps.is_available() is false")
        return torch_mod.device('mps')
    if requested == 'cpu':
        return torch_mod.device('cpu')
    raise ValueError(f"Unsupported torch device: {device!r}")


def describe_torch_device(device=None):
    """Return a compact human-readable training device description."""
    torch_mod = ensure_torch_available()
    resolved = resolve_torch_device(device, torch_module=torch_mod)
    if resolved.type == 'cuda':
        name = torch_mod.cuda.get_device_name(resolved)
        return f"cuda ({name})"
    if resolved.type == 'mps':
        return "mps (Apple Metal)"
    return "cpu"


def model_torch_device(model):
    """Return the device where a TorchMLP stores its parameters."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return resolve_torch_device('cpu')


def resolve_training_seed(seed, trailing_newline=False):
    """Resolve and print the seed used for a training/evaluation run."""
    suffix = "\n" if trailing_newline else ""
    if seed is None:
        from numpy.random import SeedSequence
        ss = SeedSequence()
        seed = int(ss.entropy % (2**31))
        print(f"Generated random seed: {seed}{suffix}")
    else:
        print(f"Using provided seed: {seed}{suffix}")
    return seed


def _init_linear(layer, seed=None):
    """Initialize a Linear layer with Glorot uniform weights and zero bias."""
    if torch is None:
        raise ImportError("No module named 'torch'")
    if seed is None:
        nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(layer.bias)
        return

    rng_state = torch.get_rng_state()
    try:
        torch.manual_seed(int(seed))
        nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(layer.bias)
    finally:
        torch.set_rng_state(rng_state)


class TorchMLP(TorchModuleBase):
    """Dense binary classifier with export helpers for runtime artifacts."""

    def __init__(self, num_features, hidden_layers=None, use_dropout=True,
                 dropout_rate=0.2, seed=None):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = list(DEFAULT_HIDDEN_LAYERS)
        self.num_features = int(num_features)
        self.hidden_layers = [int(units) for units in hidden_layers]
        self.dropout_rate = float(dropout_rate)
        self.use_dropout = bool(use_dropout and dropout_rate > 0.0)

        self.linears = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        in_features = self.num_features
        for layer_idx, units in enumerate(self.hidden_layers):
            linear = nn.Linear(in_features, units)
            _init_linear(linear, derive_seed(seed, layer_idx, 0))
            self.linears.append(linear)
            if self.use_dropout:
                self.dropouts.append(nn.Dropout(self.dropout_rate))
            in_features = units

        self.output = nn.Linear(in_features, 1)
        _init_linear(self.output, derive_seed(seed, len(self.hidden_layers), 0))

    def forward_logits(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32)
        activations = x
        for layer_idx, linear in enumerate(self.linears):
            activations = torch.relu(linear(activations))
            if self.use_dropout:
                activations = self.dropouts[layer_idx](activations)
        return self.output(activations)

    def forward(self, x):
        return torch.sigmoid(self.forward_logits(x))

    def predict(self, X, verbose=0):
        probs = predict_probabilities(self, X)
        return probs.reshape(-1, 1)

    def get_weights(self):
        return extract_model_weights(self)


def extract_model_weights(model):
    """Return dense-layer weights in the export layout expected by the runtimes."""
    if isinstance(model, TorchMLP):
        weights = []
        for linear in list(model.linears) + [model.output]:
            kernel = linear.weight.detach().cpu().numpy().T.copy()
            bias = linear.bias.detach().cpu().numpy().copy()
            weights.extend((kernel, bias))
        return weights
    if hasattr(model, 'get_weights'):
        return model.get_weights()
    raise TypeError(f"Unsupported model type for weight export: {type(model)!r}")


def predict_logits(model, X):
    """Return flat logits for a dense binary classifier."""
    ensure_torch_available()
    X = np.asarray(X, dtype=np.float32)
    if X.size == 0:
        return np.asarray([], dtype=np.float32)
    if not isinstance(model, TorchMLP):
        raise TypeError(f"Unsupported model type for logits: {type(model)!r}")
    model.eval()
    device = model_torch_device(model)
    with torch.no_grad():
        logits = model.forward_logits(torch.from_numpy(X).to(device))
    return logits.detach().cpu().numpy().reshape(-1)

# Import csi_utils first - it sets up paths automatically
TESTS_DIR = python_tests_dir()
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from csi_utils import (
    find_static_presence_motion_dataset,
    load_npz_as_packets,
    DATA_DIR,
)
from config import SEG_WINDOW_SIZE, DEFAULT_SUBCARRIERS, HAMPEL_WINDOW, HAMPEL_THRESHOLD
from segmentation import SegmentationContext
from features import (
    extract_features_by_name, DEFAULT_FEATURES, RAW_FEATURES, RELATIVE_FEATURES,
    ROBUST_RELATIVE_FEATURES,
)

# ============================================================================
# Feature Selection
# ============================================================================
#
# Production MLP uses the feature set in src/features.DEFAULT_FEATURES.
# See ALGORITHMS.md "Feature Importance" for SHAP/correlation rankings.
# ============================================================================

TRAINING_FEATURES = DEFAULT_FEATURES
BINARY_TRAINING_LABELS = ('empty', 'static_presence', 'motion')
FEATURE_SET_CHOICES = {
    'production': DEFAULT_FEATURES,
    'relative': RELATIVE_FEATURES,
    'robust_relative': ROBUST_RELATIVE_FEATURES,
}
# Directories
MODELS_DIR = models_dir()
GENERATED_DATA_DIR = generated_data_dir()
SRC_DIR = python_src_dir()
CPP_DIR = cpp_core_dir()

# Default training/evaluation configuration
DEFAULT_HIDDEN_LAYERS = [32, 16]
DEFAULT_FP_WEIGHT = 2.0
DEFAULT_SCALER_MODE = 'standard'
DEFAULT_BATCH_SIZE = 1024
DEFAULT_TORCH_DEVICE = 'cpu'
SAMPLE_WEIGHT_MODES = ('none', 'mvs_global', 'mvs_gridsearch', 'mvs_hard_negative')
DEFAULT_SAMPLE_WEIGHT_MODE = 'mvs_hard_negative'
TRAINING_CACHE_VERSION = 1
DEFAULT_ML_TEMPERATURE = 5.0
# All chips included: MLDetector keeps MVS CV normalization disabled, then
# extracts the exported raw/relative feature set from the same turbulence base.
DEFAULT_EXCLUDED_CHIPS = ()
DEFAULT_ARCHITECTURE_SWEEP = (
    {'name': 'Legacy (16-8)', 'layers': [16, 8]},
    {'name': 'Previous default (24-12)', 'layers': [24, 12]},
    {'name': 'Shallow (24)', 'layers': [24]},
    {'name': 'Current default (32-16)', 'layers': [32, 16]},
    {'name': 'Deep (24-12-6)', 'layers': [24, 12, 6]},
)
DEFAULT_EXPERIMENT_OUTPUT = MODELS_DIR / 'mlp_architecture_experiment.json'
DEFAULT_EXPERIMENT_SCREENING_SEED = 20260519
DEFAULT_EXPERIMENT_INITIAL_SEEDS = (20260518, 20260519, 20260520)
DEFAULT_EXPERIMENT_FINAL_SEEDS = (20260518, 20260519, 20260520, 20260521, 20260522)
DEFAULT_PAIRED_GATE_CHIPS = ('C3', 'C5', 'C6', 'ESP32', 'S3')
DEFAULT_LONG_GATE_CHIPS = ('C3', 'C5', 'C6', 'S3')
DEFAULT_GAIN_STRESS_SCALES = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
GAIN_INVARIANT_FEATURES = ('turb_skewness', 'turb_autocorr')
GAIN_SENSITIVE_FEATURES = tuple(
    name for name in RAW_FEATURES if name not in GAIN_INVARIANT_FEATURES
)
GAIN_RELATIVE_FEATURES = tuple(
    name for name in RELATIVE_FEATURES if name not in GAIN_INVARIANT_FEATURES
)
GAIN_EXTRA_RELATIVE_FEATURES = (
    'turb_range_over_mean',
    'turb_peak_over_mad',
)
GAIN_FEATURE_EXPERIMENT_SETS = (
    'raw',
    'relative',
    'relative_plus_range',
    'relative_plus_peak',
    'hybrid',
)
PYTEST_PAIRED_ML_GATE = (
    'test/python/test_validation_real_data.py::'
    'TestPerformanceMetrics::test_ml_detection_accuracy'
)
PYTEST_LONG_ML_GATE = (
    'test/python/test_validation_long_recordings.py::'
    'TestLongRecordings::test_ml_vs_test_recordings'
)
DEFAULT_MAX_EPOCHS = 100
DEFAULT_EARLY_STOP_PATIENCE = 8
DEFAULT_LR_PATIENCE = 4
DEFAULT_CLIP_PERCENTILES = (1.0, 99.0)
DEFAULT_PRIMARY_GROUP_KEY = 'session_group'
DEFAULT_BLOCK_GROUP_KEY = 'source_file'
DEFAULT_CV_FOLDS = 3
DEFAULT_REPORT_GROUP_KEYS = (
    'chip',
    'environment_group',
    'session_group',
    'source_file',
)
MAX_PROMOTION_TOTAL_FP_INCREASE = 0
MAX_PROMOTION_FP_RATE_INCREASE = 0.0
ACTIVE_TORCH_DEVICE = DEFAULT_TORCH_DEVICE
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


def parse_environment_filter(value):
    """Normalize a comma-separated environment filter into a set."""
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        items = value
    else:
        items = str(value).split(',')
    normalized = {str(item).strip() for item in items if str(item).strip()}
    return normalized or None


def parse_chip_filter(value):
    """Normalize a comma-separated chip filter into an uppercase set."""
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        items = value
    else:
        items = str(value).split(',')
    normalized = {str(item).strip().upper() for item in items if str(item).strip()}
    return normalized or None


def normalize_allowed_labels(labels):
    """Normalize an iterable of labels to a lowercase set."""
    if labels is None:
        return None
    return {str(label).strip().lower() for label in labels if str(label).strip()} or None


def normalize_sample_weight_mode(value):
    """Return a supported sample-weight mode."""
    mode = str(value or DEFAULT_SAMPLE_WEIGHT_MODE).strip().lower()
    if mode not in SAMPLE_WEIGHT_MODES:
        raise argparse.ArgumentTypeError(
            f"unsupported sample weight mode {mode!r}; expected one of {SAMPLE_WEIGHT_MODES}"
        )
    return mode


def parse_hidden_layers(value):
    """Parse comma-separated hidden layer widths into a positive integer list."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        layers = [int(v) for v in value]
    else:
        text = str(value).strip()
        if not text:
            return None
        try:
            layers = [int(part.strip()) for part in text.split(',') if part.strip()]
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "hidden layers must be a comma-separated list of integers, e.g. 24,12"
            ) from exc
    if not layers or any(layer <= 0 for layer in layers):
        raise argparse.ArgumentTypeError(
            "hidden layers must contain one or more positive integers"
        )
    return layers


def format_hidden_layers(layers):
    """Return hidden layers as a stable dash-separated string."""
    return '-'.join(str(int(layer)) for layer in layers)


def resolve_training_feature_set(name):
    """Resolve a named binary-training feature set."""
    key = str(name or 'production').strip().lower()
    if key not in FEATURE_SET_CHOICES:
        raise argparse.ArgumentTypeError(
            f"unsupported feature set {name!r}; expected one of {sorted(FEATURE_SET_CHOICES)}"
        )
    return list(FEATURE_SET_CHOICES[key])


def normalize_architecture_specs(architectures):
    """Normalize architecture definitions into {name, layers} dicts."""
    specs = []
    seen = set()
    for idx, arch in enumerate(architectures):
        if isinstance(arch, dict):
            layers = parse_hidden_layers(arch.get('layers'))
            name = str(arch.get('name') or f"MLP ({format_hidden_layers(layers)})")
        else:
            layers = parse_hidden_layers(arch)
            name = f"MLP ({format_hidden_layers(layers)})"
        key = tuple(layers)
        if key in seen:
            continue
        seen.add(key)
        specs.append({
            'name': name,
            'layers': list(layers),
        })
    return specs


def parse_architecture_sweep(value):
    """Parse semicolon-separated hidden-layer specs for --experiment-architectures."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return normalize_architecture_specs(value)

    text = str(value).strip()
    if not text:
        return None

    specs = []
    for idx, chunk in enumerate(text.split(';'), start=1):
        item = chunk.strip()
        if not item:
            continue
        if '=' in item:
            name, layer_text = item.split('=', 1)
            layers = parse_hidden_layers(layer_text)
            specs.append({'name': name.strip() or f"MLP #{idx}", 'layers': layers})
        else:
            layers = parse_hidden_layers(item)
            specs.append({'name': f"MLP ({format_hidden_layers(layers)})", 'layers': layers})
    if not specs:
        raise argparse.ArgumentTypeError(
            "experiment architectures must contain one or more layer specs, e.g. 16,8;24,12;32,16"
        )
    return normalize_architecture_specs(specs)


def parse_positive_chip_boost(value):
    """
    Parse chip=multiplier pairs for motion-sample boosting.

    Example:
        ESP32=1.2,S3=1.1
    """
    if value is None:
        return None
    if isinstance(value, dict):
        boosts = {}
        for chip, factor in value.items():
            chip_name = str(chip).strip().upper()
            factor_value = float(factor)
            if not chip_name:
                raise argparse.ArgumentTypeError("chip name cannot be empty in positive chip boost")
            if factor_value <= 0.0:
                raise argparse.ArgumentTypeError("positive chip boost factors must be > 0")
            boosts[chip_name] = factor_value
        return boosts or None
    text = str(value).strip()
    if not text:
        return None

    boosts = {}
    for part in text.split(','):
        item = part.strip()
        if not item:
            continue
        if '=' not in item:
            raise argparse.ArgumentTypeError(
                "positive chip boost must use CHIP=FACTOR entries, e.g. ESP32=1.2"
            )
        chip, factor = item.split('=', 1)
        chip = chip.strip().upper()
        if not chip:
            raise argparse.ArgumentTypeError("chip name cannot be empty in positive chip boost")
        try:
            factor_value = float(factor.strip())
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"invalid boost factor for {chip!r}: {factor!r}"
            ) from exc
        if factor_value <= 0.0:
            raise argparse.ArgumentTypeError(
                "positive chip boost factors must be > 0"
            )
        boosts[chip] = factor_value
    return boosts or None


def parse_gain_stress_scales(value):
    """Parse comma-separated positive gain stress multipliers."""
    if value is None:
        return tuple(DEFAULT_GAIN_STRESS_SCALES)
    if isinstance(value, (list, tuple)):
        parts = value
    else:
        parts = str(value).split(',')

    scales = []
    for item in parts:
        text = str(item).strip()
        if not text:
            continue
        try:
            scale = float(text)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid gain stress scale '{text}'"
            ) from exc
        if scale <= 0.0:
            raise argparse.ArgumentTypeError("gain stress scales must be > 0")
        scales.append(scale)
    if not scales:
        raise argparse.ArgumentTypeError("at least one gain stress scale is required")
    return tuple(scales)


def apply_positive_chip_boost(sample_weights, sample_context, y, chip_boosts):
    """
    Boost motion samples for specific chips, then renormalize overall mean to 1.0.
    """
    if chip_boosts is None:
        return sample_weights, {}
    if sample_context is None or 'chip' not in sample_context:
        return sample_weights, {}

    weights = np.asarray(sample_weights, dtype=np.float32).copy()
    chips = np.asarray(sample_context['chip']).astype(str)
    labels = np.asarray(y)
    summary = {}

    for chip, factor in sorted(chip_boosts.items()):
        mask = (chips == chip) & (labels == 1)
        affected = int(np.sum(mask))
        if affected == 0:
            summary[chip] = {'factor': factor, 'affected': 0}
            continue
        weights[mask] *= np.float32(factor)
        summary[chip] = {'factor': factor, 'affected': affected}

    mean_weight = float(np.mean(weights))
    if mean_weight > 1e-6:
        weights /= np.float32(mean_weight)
    return weights, summary


def _build_dataset_file_index(dataset_info):
    """Build filename -> (label, entry) index from dataset_info files section."""
    index = {}
    for label, files in dataset_info.get('files', {}).items():
        for entry in files:
            name = entry.get('filename')
            if name:
                index[name] = (label, entry)
    return index


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


def _parse_iso_timestamp(value):
    """Parse ISO timestamps from dataset metadata."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def _resolve_counterpart_name(label, entry, dataset_info, max_delta_seconds=30 * 60):
    """Resolve the paired baseline/movement file from metadata or nearest timestamp."""
    if label not in ('static_presence', 'motion'):
        return None

    counterpart_field = (
        'optimal_pair_motion_file'
        if label == 'static_presence'
        else 'optimal_pair_static_presence_file'
    )
    explicit = entry.get(counterpart_field)
    if explicit:
        return str(explicit)

    target_label = 'motion' if label == 'static_presence' else 'static_presence'
    timestamp = _parse_iso_timestamp(entry.get('collected_at'))
    if timestamp is None:
        return None

    chip = str(entry.get('chip', '')).upper()
    best_name = None
    best_delta = None
    for candidate in dataset_info.get('files', {}).get(target_label, []):
        if chip and str(candidate.get('chip', '')).upper() != chip:
            continue
        candidate_ts = _parse_iso_timestamp(candidate.get('collected_at'))
        candidate_name = candidate.get('filename')
        if candidate_ts is None or not candidate_name:
            continue

        delta = abs((candidate_ts - timestamp).total_seconds())
        if delta > max_delta_seconds:
            continue
        if best_delta is None or delta < best_delta:
            best_delta = delta
            best_name = str(candidate_name)
    return best_name


def _build_pair_id(label, entry, dataset_info=None):
    """Build a stable pair/session id shared by baseline and movement files."""
    if label not in ('static_presence', 'motion'):
        return None

    filename = entry.get('filename')
    if not filename:
        return None

    counterpart = None
    if dataset_info is not None:
        counterpart = _resolve_counterpart_name(label, entry, dataset_info)
    if counterpart is None:
        counterpart_field = (
            'optimal_pair_motion_file'
            if label == 'static_presence'
            else 'optimal_pair_static_presence_file'
        )
        counterpart = entry.get(counterpart_field)
    if not counterpart:
        return None

    names = sorted([str(filename), str(counterpart)])
    return f"pair:{names[0]}::{names[1]}"


def _build_file_context(label, file_info, dataset_info=None):
    """Derive grouping metadata used for honest evaluation and reporting."""
    filename = str(file_info.get('filename', ''))
    chip = str(file_info.get('chip', 'unknown')).upper()
    gain_locked = bool(file_info.get('gain_locked', True))
    collected_at = _parse_iso_timestamp(file_info.get('collected_at'))
    explicit_environment = _first_non_empty(
        file_info,
        (
            'environment',
            'environment_id',
            'environment_name',
        ),
    )
    explicit_session = _first_non_empty(
        file_info,
        (
            'session',
            'session_id',
            'session_name',
        ),
    )
    pair_id = _build_pair_id(label, file_info, dataset_info=dataset_info)
    day_group = collected_at.date().isoformat() if collected_at else 'unknown-day'

    return {
        'gain_locked': gain_locked,
        'chip': chip,
        'collected_at': collected_at.isoformat() if collected_at else '',
        'day_group': day_group,
        'pair_id': pair_id or '',
        # Session grouping is the primary evaluation key. Use explicit session
        # metadata when available, otherwise fall back to the paired capture or file.
        'session_group': explicit_session or pair_id or f"file:{filename or 'unknown'}",
        # Keep a dedicated environment field so future datasets can report
        # room/environment worst-groups without changing the training code again.
        'environment_group': explicit_environment or 'unknown-environment',
    }


def _fallback_file_context(filename, label, packet):
    """Create grouping metadata for files missing from dataset_info.json."""
    fallback = {
        'filename': filename,
        'chip': packet.get('chip', 'unknown'),
        'gain_locked': packet.get('gain_locked', True),
        'collected_at': packet.get('collected_at', ''),
    }
    return _build_file_context(label, fallback)


def _is_temporally_paired(dataset_info, label, entry, max_delta_seconds=30 * 60):
    """Check if entry has a valid counterpart within max delta."""
    if label == 'static_presence':
        counterpart_label = 'motion'
        counterpart_name = entry.get('optimal_pair_motion_file')
    else:
        counterpart_label = 'static_presence'
        counterpart_name = entry.get('optimal_pair_static_presence_file')
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


def build_gridsearch_tuning_map(dataset_info, default_threshold=1.0):
    """
    Build per-file tuning map from dataset_info.

    Returns:
        dict: {
            filename: {
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

            threshold = float(entry.get('optimal_threshold_gridsearch', default_threshold))
            paired = _is_temporally_paired(dataset_info, label, entry)
            mode = 'paired' if paired else 'single-dataset fallback'
            confidence_factor = 1.0 if paired else 0.5
            tuning[name] = {
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
        return label_name == 'motion'
    # Default: only 'motion' is motion
    return label_name == 'motion'


def get_file_metadata(dataset_info):
    """
    Get metadata for all files in dataset_info.json.

    Returns a dict mapping filename to metadata including normalization flags and
    grouping context used by training/evaluation.

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
                file_metadata[filename] = _build_file_context(label, file_info, dataset_info=dataset_info)
    return file_metadata


def load_all_data(environment_filter=None, excluded_chips=None,
                  allowed_labels=BINARY_TRAINING_LABELS,
                  require_sync_metadata=False):
    """
    Load all available CSI data from the data/ directory.

    Reads label from npz file metadata (not folder structure).
    Uses dataset_info.json only to determine if label is motion or idle.
    Uses each NPZ file's `gain_locked` field to decide CV normalization and
    attaches grouping metadata (file, session/pair, environment when available).

    Args:
        environment_filter: Optional set/string of environment names to keep.
        excluded_chips: Optional set/string of chip names to exclude.
        allowed_labels: Iterable of lowercase labels to include.

    Returns:
        tuple: (all_packets, stats) where stats is a dict with dataset info
    """
    environment_filter = parse_environment_filter(environment_filter)
    excluded_chips = parse_chip_filter(excluded_chips)
    allowed_labels = normalize_allowed_labels(allowed_labels)
    all_packets = []
    stats = {
        'chips': set(),
        'labels': {},
        'total': 0,
        'cv_norm_files': set(),
        'files': [],
        'excluded_labels': set(),
        'excluded_chips': set(),
        'excluded_environments': set(),
        'excluded_missing_sync_metadata': set(),
        'session_groups': set(),
        'environment_groups': set(),
        'sync_metadata_files': set(),
    }
    
    # Load dataset info for label mapping and file metadata
    dataset_info = load_dataset_info()
    file_metadata = get_file_metadata(dataset_info)
    
    # Scan raw dataset subdirectories only. Generated artifacts such as
    # data/auto_generated/*.npz are not packet captures and should not be
    # parsed as CSI inputs.
    excluded_dirs = {'.', GENERATED_DATA_DIR.name}
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
                
                label_lc = str(label).lower()
                if allowed_labels is not None and label_lc not in allowed_labels:
                    stats['excluded_labels'].add(label_lc)
                    continue

                # Get chip
                chip = packets[0].get('chip', 'unknown').upper()
                if excluded_chips is not None and chip in excluded_chips:
                    stats['excluded_chips'].add(chip)
                    continue
                
                # Get file-specific metadata
                meta = file_metadata.get(npz_file.name)
                if meta is None:
                    meta = _fallback_file_context(npz_file.name, label_lc, packets[0])

                environment_group = meta.get('environment_group', 'unknown-environment')
                if environment_filter is not None and environment_group not in environment_filter:
                    stats['excluded_environments'].add(environment_group)
                    continue

                gain_locked = bool(meta.get('gain_locked', packets[0].get('gain_locked', True)))
                has_sync_metadata = any(
                    p.get('stimulus_id') is not None and (
                        p.get('wifi_rx_start_ts_ns') is not None or p.get('device_ticks_us') is not None
                    )
                    for p in packets
                )
                if require_sync_metadata and not has_sync_metadata:
                    stats['excluded_missing_sync_metadata'].add(npz_file.name)
                    continue
                if has_sync_metadata:
                    stats['sync_metadata_files'].add(npz_file.name)

                # Track stats after all active filters
                if label not in stats['labels']:
                    stats['labels'][label] = 0
                stats['labels'][label] += len(packets)
                stats['total'] += len(packets)
                stats['chips'].add(chip)
                if not gain_locked:
                    stats['cv_norm_files'].add(npz_file.name)

                stats['session_groups'].add(meta.get('session_group', f"file:{npz_file.name}"))
                if environment_group != 'unknown-environment':
                    stats['environment_groups'].add(environment_group)

                # Add flags to each packet
                is_motion = is_motion_label(label, dataset_info)
                for idx, p in enumerate(packets):
                    p['is_motion'] = is_motion
                    p['label_name'] = label_lc
                    p['gain_locked'] = gain_locked
                    p['source_file'] = npz_file.name
                    p['packet_index'] = idx
                    p['chip'] = meta.get('chip', chip)
                    p['collected_at'] = meta.get('collected_at', '')
                    p['day_group'] = meta.get('day_group', 'unknown-day')
                    p['pair_id'] = meta.get('pair_id', '')
                    p['session_group'] = meta.get('session_group', f"file:{npz_file.name}")
                    p['environment_group'] = environment_group
                
                all_packets.extend(packets)
                stats['files'].append(npz_file.name)
                
            except Exception as e:
                print(f"  Warning: Could not load {npz_file.name}: {e}")
    
    stats['chips'] = sorted(stats['chips'])
    stats['cv_norm_files'] = sorted(stats['cv_norm_files'])
    stats['excluded_labels'] = sorted(stats['excluded_labels'])
    stats['excluded_chips'] = sorted(stats['excluded_chips'])
    stats['excluded_environments'] = sorted(stats['excluded_environments'])
    stats['excluded_missing_sync_metadata'] = sorted(stats['excluded_missing_sync_metadata'])
    stats['session_groups'] = sorted(stats['session_groups'])
    stats['environment_groups'] = sorted(stats['environment_groups'])
    stats['sync_metadata_files'] = sorted(stats['sync_metadata_files'])
    return all_packets, stats


def _training_cache_manifest(feature_names, environment_filter=None,
                             excluded_chips=None,
                             allowed_labels=BINARY_TRAINING_LABELS,
                             sample_weight_mode=DEFAULT_SAMPLE_WEIGHT_MODE,
                             window_size=SEG_WINDOW_SIZE,
                             enable_hampel=True,
                             hampel_window=HAMPEL_WINDOW,
                             hampel_threshold=HAMPEL_THRESHOLD):
    """Build a stable manifest for cached training matrices."""
    allowed_labels = sorted(normalize_allowed_labels(allowed_labels) or ())
    environment_filter = sorted(parse_environment_filter(environment_filter) or ())
    excluded_chips = sorted(parse_chip_filter(excluded_chips) or ())
    sample_weight_mode = normalize_sample_weight_mode(sample_weight_mode)
    files = []
    for npz_file in sorted(DATA_DIR.glob('*/*.npz')):
        if allowed_labels and npz_file.parent.name.lower() not in allowed_labels:
            continue
        try:
            stat = npz_file.stat()
        except OSError:
            continue
        files.append({
            'path': str(npz_file.relative_to(DATA_DIR)),
            'size': int(stat.st_size),
            'mtime_ns': int(stat.st_mtime_ns),
        })

    info_path = DATA_DIR / 'dataset_info.json'
    dataset_info = None
    if info_path.exists():
        stat = info_path.stat()
        dataset_info = {
            'path': str(info_path.relative_to(DATA_DIR)),
            'size': int(stat.st_size),
            'mtime_ns': int(stat.st_mtime_ns),
        }

    return {
        'version': TRAINING_CACHE_VERSION,
        'feature_names': list(feature_names),
        'allowed_labels': allowed_labels,
        'environment_filter': environment_filter,
        'excluded_chips': excluded_chips,
        'sample_weight_mode': sample_weight_mode,
        'default_subcarriers': [int(sc) for sc in DEFAULT_SUBCARRIERS],
        'window_size': int(window_size),
        'enable_hampel': bool(enable_hampel),
        'hampel_window': int(hampel_window),
        'hampel_threshold': float(hampel_threshold),
        'dataset_info': dataset_info,
        'files': files,
    }


def _training_cache_path(manifest):
    """Return the on-disk cache path for a training manifest."""
    payload = json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode('utf-8')
    digest = hashlib.sha256(payload).hexdigest()[:16]
    return GENERATED_DATA_DIR / f'training_matrix_{digest}.npz'


def _load_training_matrix_cache(cache_path, manifest):
    """Load cached feature matrix, labels, context, stats, and base weights."""
    if not cache_path.exists():
        return None
    try:
        with np.load(cache_path, allow_pickle=True) as data:
            cached_manifest = json.loads(str(data['manifest_json'].item()))
            if cached_manifest != manifest:
                return None
            sample_context = {
                key: data[f'ctx_{key}']
                for key in data['context_keys'].tolist()
            }
            stats = json.loads(str(data['stats_json'].item()))
            return {
                'X': data['X'].astype(np.float32, copy=False),
                'y': data['y'].astype(np.int8, copy=False),
                'feature_names': data['feature_names'].astype(str).tolist(),
                'sample_context': sample_context,
                'sample_weights': data['sample_weights'].astype(np.float32, copy=False),
                'stats': stats,
            }
    except Exception as exc:
        print(f"  Warning: ignoring invalid training cache {cache_path.name}: {exc}")
        return None


def _save_training_matrix_cache(cache_path, manifest, X, y, feature_names,
                                sample_context, sample_weights, stats):
    """Persist a training matrix cache for repeated local runs."""
    try:
        GENERATED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            'manifest_json': np.asarray(json.dumps(manifest, sort_keys=True)),
            'stats_json': np.asarray(json.dumps(stats, sort_keys=True)),
            'X': np.asarray(X, dtype=np.float32),
            'y': np.asarray(y, dtype=np.int8),
            'feature_names': np.asarray(feature_names, dtype=object),
            'sample_weights': np.asarray(sample_weights, dtype=np.float32),
            'context_keys': np.asarray(list(sample_context.keys()), dtype=object),
        }
        for key, values in sample_context.items():
            payload[f'ctx_{key}'] = np.asarray(values)
        np.savez(cache_path, **payload)
    except Exception as exc:
        print(f"  Warning: could not write training cache {cache_path.name}: {exc}")


def load_training_matrix(environment_filter=None, excluded_chips=None,
                         feature_names=None, sample_weight_mode=DEFAULT_SAMPLE_WEIGHT_MODE,
                         use_cache=True):
    """Load or build the cached training matrix used by binary ML training."""
    if feature_names is None:
        feature_names = DEFAULT_FEATURES.copy()
    feature_names = list(feature_names)
    sample_weight_mode = normalize_sample_weight_mode(sample_weight_mode)

    manifest = _training_cache_manifest(
        feature_names,
        environment_filter=environment_filter,
        excluded_chips=excluded_chips,
        allowed_labels=BINARY_TRAINING_LABELS,
        sample_weight_mode=sample_weight_mode,
    )
    cache_path = _training_cache_path(manifest)
    if use_cache:
        cached = _load_training_matrix_cache(cache_path, manifest)
        if cached is not None:
            print(f"  Training matrix cache: hit ({cache_path.name})")
            return cached, None
        print(f"  Training matrix cache: miss ({cache_path.name})")
    else:
        print("  Training matrix cache: disabled")

    load_start = perf_counter()
    all_packets, stats = load_all_data(
        environment_filter=environment_filter,
        excluded_chips=excluded_chips,
    )
    print(f"  Load time: {format_duration(perf_counter() - load_start)}")

    if not stats['chips']:
        return {
            'X': np.empty((0, len(feature_names)), dtype=np.float32),
            'y': np.asarray([], dtype=np.int8),
            'feature_names': feature_names,
            'sample_context': {},
            'sample_weights': np.asarray([], dtype=np.float32),
            'stats': stats,
        }, all_packets

    print("\nExtracting features...")
    features_start = perf_counter()
    X, y, actual_feature_names, sample_context = extract_features(
        all_packets, feature_names=feature_names
    )
    print(f"  Feature extraction time: {format_duration(perf_counter() - features_start)}")

    print(f"\nComputing sample weights ({sample_weight_mode})...")
    weights_start = perf_counter()
    sample_weights = compute_sample_weights(
        all_packets,
        y,
        sample_weight_mode=sample_weight_mode,
        window_size=SEG_WINDOW_SIZE,
    )
    print(f"  Sample weights time: {format_duration(perf_counter() - weights_start)}")

    matrix = {
        'X': np.asarray(X, dtype=np.float32),
        'y': np.asarray(y, dtype=np.int8),
        'feature_names': list(actual_feature_names),
        'sample_context': sample_context,
        'sample_weights': np.asarray(sample_weights, dtype=np.float32),
        'stats': stats,
    }
    if use_cache:
        _save_training_matrix_cache(
            cache_path,
            manifest,
            matrix['X'],
            matrix['y'],
            matrix['feature_names'],
            matrix['sample_context'],
            matrix['sample_weights'],
            matrix['stats'],
        )
        print(f"  Training matrix cache: wrote {cache_path.name}")
    return matrix, all_packets


def extract_features(packets, window_size=SEG_WINDOW_SIZE,
                     feature_names=None, return_metadata=False,
                     enable_hampel=True, hampel_window=HAMPEL_WINDOW, hampel_threshold=HAMPEL_THRESHOLD,
                     ):
    """
    Extract features from CSI packets using sliding window.
    
    Uses SegmentationContext.add_turbulence() so the filter chain (Hampel -> low-pass)
    matches the runtime pipeline, ensuring train/deploy alignment.
    
    Args:
        packets: List of CSI packets with 'csi_data' and 'label'
        window_size: Sliding window size (default: SEG_WINDOW_SIZE from config.py)
        feature_names: List of feature names to extract (default: DEFAULT_FEATURES)
        return_metadata: If True, return per-sample metadata
        enable_hampel: Enable Hampel outlier filter on turbulence (default: True)
        hampel_window: Hampel filter window size (default: 7)
        hampel_threshold: Hampel filter threshold in MAD units (default: 5.0)
    
    Returns:
        tuple: (X, y, feature_names, sample_context)
            - X: Feature matrix (n_samples, n_features)
            - y: Labels (n_samples,)
            - feature_names: List of feature names
            - sample_context: Dict of aligned per-sample grouping metadata
    """
    if feature_names is None:
        feature_names = DEFAULT_FEATURES.copy()
    feature_names = list(feature_names)
    
    X, y = [], []
    sample_context = {
        'chip': [],
        'source_file': [],
        'session_group': [],
        'environment_group': [],
        'pair_id': [],
        'day_group': [],
        'label_name': [],
        'packet_index': [],
        'window_index': [],
    }

    # Process each source file independently to avoid window leakage across files.
    grouped = {}
    for pkt in packets:
        source = pkt.get('source_file', '__single_stream__')
        grouped.setdefault(source, []).append(pkt)

    for source_file, file_packets in grouped.items():
        chip = file_packets[0].get('chip', 'unknown').upper()
        file_context = {
            'chip': chip,
            'source_file': source_file,
            'session_group': file_packets[0].get('session_group', f"file:{source_file}"),
            'environment_group': file_packets[0].get('environment_group', 'unknown-environment'),
            'pair_id': file_packets[0].get('pair_id', ''),
            'day_group': file_packets[0].get('day_group', 'unknown-day'),
        }
        ctx = SegmentationContext(
            window_size=window_size,
            threshold=1.0,
            enable_hampel=enable_hampel,
            hampel_window=hampel_window,
            hampel_threshold=hampel_threshold,
        )
        last_amplitudes = None
        window_index = 0
        for pkt in file_packets:
            csi_data = pkt['csi_data']
            ctx.use_cv_normalization = not bool(pkt.get('gain_locked', True))
            turb, amps = ctx.calculate_spatial_turbulence(
                csi_data,
                DEFAULT_SUBCARRIERS,
                return_amplitudes=True,
            )
            ctx.add_turbulence(turb)
            last_amplitudes = amps

            if ctx.buffer_count < window_size:
                continue

            # Reconstruct chronological order from circular buffer
            idx = ctx.buffer_index
            turb_list = ctx.turbulence_buffer[idx:] + ctx.turbulence_buffer[:idx]
            n = len(turb_list)

            features = extract_features_by_name(
                turb_list, n,
                amplitudes=last_amplitudes,
                feature_names=feature_names
            )

            X.append(features)
            y.append(1 if pkt.get('is_motion', False) else 0)
            for key, value in file_context.items():
                sample_context[key].append(value)
            sample_context['label_name'].append(str(pkt.get('label_name', 'unknown')))
            sample_context['packet_index'].append(int(pkt.get('packet_index', window_index)))
            sample_context['window_index'].append(window_index)
            window_index += 1

    X_arr = np.asarray(X, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.int8)
    context_arrays = {
        key: np.asarray(values, dtype=np.int32 if key in ('packet_index', 'window_index') else object)
        for key, values in sample_context.items()
    }
    return X_arr, y_arr, feature_names, context_arrays


def compute_sample_weights(packets, labels, sample_weight_mode=DEFAULT_SAMPLE_WEIGHT_MODE,
                           window_size=SEG_WINDOW_SIZE):
    """Compute base sample weights for the selected training policy."""
    mode = normalize_sample_weight_mode(sample_weight_mode)
    if mode == 'none':
        return np.ones(len(labels), dtype=np.float32)

    dataset_info = load_dataset_info()
    if mode == 'mvs_global':
        tuning_map = {}
        mvs_policy = 'full'
    elif mode == 'mvs_gridsearch':
        tuning_map = build_gridsearch_tuning_map(dataset_info, default_threshold=1.0)
        mvs_policy = 'full'
    elif mode == 'mvs_hard_negative':
        tuning_map = build_gridsearch_tuning_map(dataset_info, default_threshold=1.0)
        mvs_policy = 'hard_negative'
    else:
        raise AssertionError(f"Unhandled sample weight mode: {mode}")

    return compute_mvs_guided_sample_weights(
        packets,
        tuning_map,
        window_size=window_size,
        policy=mvs_policy,
    )


def compute_mvs_guided_sample_weights(packets, tuning_map, window_size=SEG_WINDOW_SIZE,
                                      policy='full'):
    """
    Compute sample weights using context-aware MVS scoring per source file.

    Weight policy:
    - full policy: hard-positive mining for movement samples
    - hard_negative policy: leave movement samples neutral
    - baseline samples: promote hard negatives (2.0) when metric >= threshold, else 1.0
    - per-file thresholds fall back toward the default threshold when pairing
      confidence is low, reducing overfitting to file-specific grid-search tuning
    - weights are normalized per file so no single recording dominates training
    """
    weights = []

    grouped = {}
    for pkt in packets:
        source = pkt.get('source_file', '__single_stream__')
        grouped.setdefault(source, []).append(pkt)

    for source_file, file_packets in grouped.items():
        cfg = tuning_map.get(source_file, None)
        if cfg is None:
            threshold = 1.0
            confidence_factor = 0.5
        else:
            threshold = max(float(cfg['threshold']), 1e-6)
            confidence_factor = float(cfg['confidence_factor'])
        effective_threshold = (
            confidence_factor * threshold
            + (1.0 - confidence_factor) * 1.0
        )

        ctx = SegmentationContext(window_size=window_size, threshold=effective_threshold)
        file_weights = []
        for pkt in file_packets:
            ctx.use_cv_normalization = not bool(pkt.get('gain_locked', True))
            turb = ctx.calculate_spatial_turbulence(
                pkt['csi_data'],
                DEFAULT_SUBCARRIERS,
            )
            ctx.add_turbulence(turb)
            ctx.update_state()
            if ctx.buffer_count < window_size:
                continue

            ratio = max(0.0, float(ctx.current_moving_variance)) / max(effective_threshold, 1e-6)
            if pkt.get('is_motion', False):
                # Hard-positive mining:
                # subtle/near-threshold motion is exactly where recall drops in
                # deployment, so up-weight it; easy positives get lower weight.
                if policy == 'hard_negative':
                    base = 1.0
                elif ratio < 0.5:
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

            file_weights.append(base)

        if not file_weights:
            continue

        file_weights = np.asarray(file_weights, dtype=np.float32)
        file_mean = float(np.mean(file_weights))
        if file_mean > 1e-6:
            file_weights /= file_mean
        weights.extend(file_weights.tolist())

    return np.array(weights, dtype=np.float32)


# ============================================================================
# Model Training
# ============================================================================

class ClippedStandardScaler:
    """Clip heavy tails before applying standard z-score normalization."""

    def __init__(self, lower_percentile=1.0, upper_percentile=99.0):
        self.lower_percentile = float(lower_percentile)
        self.upper_percentile = float(upper_percentile)
        self.lower_bounds_ = None
        self.upper_bounds_ = None
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        self.lower_bounds_ = np.percentile(X, self.lower_percentile, axis=0)
        self.upper_bounds_ = np.percentile(X, self.upper_percentile, axis=0)
        clipped = np.clip(X, self.lower_bounds_, self.upper_bounds_)
        self.mean_ = clipped.mean(axis=0)
        self.scale_ = clipped.std(axis=0)
        self.scale_[self.scale_ < 1e-6] = 1.0
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        clipped = np.clip(X, self.lower_bounds_, self.upper_bounds_)
        return (clipped - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def build_preprocessor(mode=DEFAULT_SCALER_MODE, clip_percentiles=DEFAULT_CLIP_PERCENTILES):
    """Build the feature normalization object used in CV and final training."""
    from sklearn.preprocessing import RobustScaler, StandardScaler

    if mode == 'standard':
        return StandardScaler()
    if mode == 'robust':
        return RobustScaler()
    if mode == 'clipped_standard':
        return ClippedStandardScaler(*clip_percentiles)
    raise ValueError(f"Unsupported scaler mode: {mode}")


def get_preprocessor_arrays(preprocessor):
    """Extract center/scale arrays for export across scaler implementations."""
    center = getattr(preprocessor, 'mean_', None)
    if center is None:
        center = getattr(preprocessor, 'center_', None)
    scale = getattr(preprocessor, 'scale_', None)
    if center is None or scale is None:
        raise AttributeError("Preprocessor must expose center/scale arrays for export")

    center = np.asarray(center, dtype=np.float32)
    scale = np.asarray(scale, dtype=np.float32)
    scale[scale < 1e-6] = 1.0
    return center, scale


def slice_sample_context(sample_context, indices):
    """Slice aligned metadata dicts with NumPy indices."""
    if sample_context is None:
        return None
    return {
        key: np.asarray(values)[indices]
        for key, values in sample_context.items()
    }


def build_block_mask(sample_context, stride=1, group_key=DEFAULT_BLOCK_GROUP_KEY):
    """Subsample validation windows to reduce overlap optimism during scoring."""
    if sample_context is None:
        return None

    first_key = next(iter(sample_context), None)
    n_samples = len(sample_context[first_key]) if first_key is not None else 0
    if stride <= 1 or n_samples == 0:
        return np.ones(n_samples, dtype=bool)

    mask = np.zeros(n_samples, dtype=bool)
    group_values = sample_context.get(group_key)
    if group_values is None:
        mask[::stride] = True
        return mask

    counters = {}
    for idx, raw_group in enumerate(group_values):
        group = str(raw_group)
        count = counters.get(group, 0)
        if count % stride == 0:
            mask[idx] = True
        counters[group] = count + 1
    return mask


def evaluate_probabilities(y_true, y_prob, threshold=0.5):
    """Evaluate predicted probabilities with the deployment-equivalent threshold."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob > threshold).astype(int).flatten()
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0.0
    fp_rate = fp / (fp + tn) * 100 if (fp + tn) > 0 else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) * 100 if (2 * tp + fp + fn) > 0 else 0.0

    return {
        'recall': recall,
        'precision': precision,
        'fp_rate': fp_rate,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
    }


def build_group_report(y_true, y_prob, group_values):
    """Compute per-group metrics and worst-group summaries."""
    if group_values is None:
        return None

    group_values = np.asarray(group_values)
    rows = []
    for group_name in sorted({str(v) for v in group_values}):
        if not group_name or group_name == 'unknown-environment':
            continue
        mask = (group_values == group_name)
        if not np.any(mask):
            continue
        metrics = evaluate_probabilities(y_true[mask], y_prob[mask])
        rows.append({
            'group': group_name,
            'samples': int(np.sum(mask)),
            'positives': int(np.sum(y_true[mask] == 1)),
            'negatives': int(np.sum(y_true[mask] == 0)),
            **metrics,
        })

    if not rows:
        return None

    recall_rows = [r for r in rows if r['positives'] > 0]
    fp_rows = [r for r in rows if r['negatives'] > 0]
    rows_by_recall = sorted(recall_rows or rows, key=lambda r: (r['recall'], r['fp_rate'], -r['samples']))
    rows_by_fp = sorted(fp_rows or rows, key=lambda r: (-r['fp_rate'], r['recall'], -r['samples']))
    return {
        'rows': rows,
        'worst_recall': rows_by_recall[0],
        'worst_fp_rate': rows_by_fp[0],
        'count': len(rows),
    }


def load_exported_ml_weights():
    """Load the currently exported MicroPython ML weights module."""
    import importlib.util

    weights_path = SRC_DIR / 'ml_weights.py'
    if not weights_path.exists():
        raise FileNotFoundError(f"Exported ML weights not found: {weights_path}")
    spec = importlib.util.spec_from_file_location("exported_ml_weights", weights_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def predict_exported_probabilities_from_weights(weights_module, X_raw):
    """Vectorized inference matching src/python/micro_espectre/ml_detector.py for exported weights."""
    X_raw = np.asarray(X_raw, dtype=np.float32)
    center = np.asarray(weights_module.FEATURE_MEAN, dtype=np.float32)
    scale = np.asarray(weights_module.FEATURE_SCALE, dtype=np.float32)
    scale[scale < 1e-6] = 1.0

    activations = (X_raw - center) / scale
    weights = [np.asarray(w, dtype=np.float32) for w in weights_module.WEIGHTS]
    biases = [np.asarray(b, dtype=np.float32) for b in weights_module.BIASES]
    for layer_idx, (layer_weights, layer_biases) in enumerate(zip(weights, biases)):
        activations = activations @ layer_weights + layer_biases
        if layer_idx != len(weights) - 1:
            activations = np.maximum(activations, 0.0)

    logits = activations.reshape(-1) / float(DEFAULT_ML_TEMPERATURE)
    logits = np.clip(logits, -20.0, 20.0)
    return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)


def apply_gain_stress_to_features(X, feature_names, scale):
    """Scale only feature dimensions that move linearly with amplitude gain."""
    X_scaled = np.asarray(X, dtype=np.float32).copy()
    sensitive_indices = [
        idx for idx, name in enumerate(feature_names)
        if name in GAIN_SENSITIVE_FEATURES
    ]
    if sensitive_indices:
        X_scaled[:, sensitive_indices] *= np.float32(scale)
    return X_scaled, sensitive_indices


def evaluate_gain_stress_gate(environment_filter=None, excluded_chips=None,
                              scales=DEFAULT_GAIN_STRESS_SCALES):
    """Evaluate current exported model under artificial gain scaling."""
    environment_filter = parse_environment_filter(environment_filter)
    excluded_chips = parse_chip_filter(excluded_chips)
    scales = parse_gain_stress_scales(scales)

    weights_module = load_exported_ml_weights()
    feature_names = list(getattr(weights_module, 'FEATURE_NAMES', TRAINING_FEATURES))
    all_packets, stats = load_all_data(
        environment_filter=environment_filter,
        excluded_chips=excluded_chips,
        allowed_labels=BINARY_TRAINING_LABELS,
    )
    if not all_packets:
        raise RuntimeError("No empty/static_presence/motion packets found for gain stress gate")

    X, y, actual_feature_names, sample_context = extract_features(
        all_packets,
        feature_names=feature_names,
    )
    if list(actual_feature_names) != feature_names:
        raise RuntimeError(
            "Extracted feature order does not match exported model: "
            f"{actual_feature_names} != {feature_names}"
        )

    scaled_features = [
        name for name in feature_names if name in GAIN_SENSITIVE_FEATURES
    ]
    results = {
        'feature_names': feature_names,
        'scaled_features': scaled_features,
        'invariant_features': [
            name for name in feature_names if name not in scaled_features
        ],
        'stats': stats,
        'samples': int(len(X)),
        'scales': {},
    }

    for gain_scale in scales:
        X_stressed, sensitive_indices = apply_gain_stress_to_features(
            X,
            feature_names,
            gain_scale,
        )
        y_prob = predict_exported_probabilities_from_weights(weights_module, X_stressed)
        scale_result = {
            'scale': float(gain_scale),
            'sensitive_indices': [int(idx) for idx in sensitive_indices],
            'overall': evaluate_probabilities(y, y_prob),
            'group_reports': {},
        }
        for group_key in DEFAULT_REPORT_GROUP_KEYS:
            report = build_group_report(y, y_prob, sample_context.get(group_key))
            if report is not None:
                scale_result['group_reports'][group_key] = report
        results['scales'][float(gain_scale)] = scale_result
    return results


def print_gain_stress_summary(results):
    """Print a compact gain stress report."""
    stats = results.get('stats', {})
    print("\n" + "=" * 70)
    print("  EXPORTED ML GAIN-STRESS GATE")
    print("=" * 70)
    print(f"Samples: {results['samples']}")
    print(f"Chips: {', '.join(stats.get('chips', []))}")
    if stats.get('environment_groups'):
        print(f"Environments: {', '.join(stats['environment_groups'])}")
    print(f"Scaled features: {', '.join(results['scaled_features']) or 'none'}")
    print(f"Invariant features: {', '.join(results['invariant_features']) or 'none'}")
    print()
    print(
        "  scale | recall  precision  FP rate      F1 | "
        "worst chip recall        worst chip FP"
    )
    print("  " + "-" * 86)
    for scale in sorted(results['scales']):
        row = results['scales'][scale]
        overall = row['overall']
        chip_report = row['group_reports'].get('chip', {})
        worst_recall = chip_report.get('worst_recall', {})
        worst_fp = chip_report.get('worst_fp_rate', {})
        recall_label = (
            f"{worst_recall.get('group', 'n/a')} {worst_recall.get('recall', 0.0):5.1f}%"
        )
        fp_label = (
            f"{worst_fp.get('group', 'n/a')} {worst_fp.get('fp_rate', 0.0):5.1f}%"
        )
        print(
            f"  {scale:5.2f} | "
            f"{overall['recall']:6.1f}% "
            f"{overall['precision']:9.1f}% "
            f"{overall['fp_rate']:7.1f}% "
            f"{overall['f1']:7.1f}% | "
            f"{recall_label:24} {fp_label}"
        )

    for group_key in ('environment_group', 'session_group', 'source_file'):
        print(f"\nWorst {group_key} by scale:")
        for scale in sorted(results['scales']):
            report = results['scales'][scale]['group_reports'].get(group_key)
            if not report:
                continue
            worst_recall = report['worst_recall']
            worst_fp = report['worst_fp_rate']
            print(
                f"  {scale:5.2f}: "
                f"R {worst_recall['group']}={worst_recall['recall']:.1f}% | "
                f"FP {worst_fp['group']}={worst_fp['fp_rate']:.1f}%"
            )


def build_gain_feature_matrix(X_raw, raw_feature_names, feature_set):
    """Build raw, relative, or hybrid matrices from the production raw feature set."""
    feature_set = str(feature_set).strip().lower()
    if feature_set not in GAIN_FEATURE_EXPERIMENT_SETS:
        raise ValueError(
            f"Unsupported gain feature set {feature_set!r}; "
            f"expected one of {GAIN_FEATURE_EXPERIMENT_SETS}"
        )

    X_raw = np.asarray(X_raw, dtype=np.float32)
    raw_feature_names = list(raw_feature_names)
    columns = {name: X_raw[:, idx] for idx, name in enumerate(raw_feature_names)}
    mean = np.maximum(np.abs(columns['turb_mean']), np.float32(1e-6))
    mad = np.maximum(columns['turb_mad'], np.float32(1e-6))
    window_norm = mean * np.float32(max(SEG_WINDOW_SIZE - 1, 1))
    relative_columns = {
        'turb_std_over_mean': columns['turb_std'] / mean,
        'turb_max_over_mean': columns['turb_max'] / mean,
        'turb_min_over_mean': columns['turb_min'] / mean,
        'turb_iqr_over_mean': columns['turb_iqr'] / mean,
        'turb_mad_over_mean': columns['turb_mad'] / mean,
        'waveform_length_over_mean': columns['waveform_length'] / window_norm,
        'turb_range_over_mean': (columns['turb_max'] - columns['turb_min']) / mean,
        'turb_peak_over_mad': (columns['turb_max'] - columns['turb_mean']) / mad,
        'turb_skewness': columns['turb_skewness'],
        'turb_autocorr': columns['turb_autocorr'],
    }

    if feature_set == 'raw':
        return X_raw.copy(), list(raw_feature_names)

    relative_names = list(GAIN_RELATIVE_FEATURES) + list(GAIN_INVARIANT_FEATURES)
    relative_matrix = np.column_stack(
        [relative_columns[name] for name in relative_names]
    ).astype(np.float32)
    if feature_set == 'relative':
        return relative_matrix, relative_names
    if feature_set == 'relative_plus_range':
        names = relative_names + ['turb_range_over_mean']
        matrix = np.column_stack(
            [relative_columns[name] for name in names]
        ).astype(np.float32)
        return matrix, names
    if feature_set == 'relative_plus_peak':
        names = relative_names + ['turb_peak_over_mad']
        matrix = np.column_stack(
            [relative_columns[name] for name in names]
        ).astype(np.float32)
        return matrix, names

    hybrid_names = list(raw_feature_names) + list(GAIN_RELATIVE_FEATURES)
    hybrid_matrix = np.column_stack(
        [X_raw] + [relative_columns[name] for name in GAIN_RELATIVE_FEATURES]
    ).astype(np.float32)
    return hybrid_matrix, hybrid_names


def evaluate_model_gain_stress(model, scaler, X_raw, y, feature_names, sample_context,
                               scales=DEFAULT_GAIN_STRESS_SCALES):
    """Evaluate an in-memory trained model under artificial gain scaling."""
    scales = parse_gain_stress_scales(scales)
    results = {}
    for gain_scale in scales:
        X_stressed, sensitive_indices = apply_gain_stress_to_features(
            X_raw,
            feature_names,
            gain_scale,
        )
        X_scaled = scaler.transform(X_stressed)
        y_prob = predict_tempered_probabilities(model, X_scaled)
        scale_result = {
            'scale': float(gain_scale),
            'sensitive_indices': [int(idx) for idx in sensitive_indices],
            'overall': evaluate_probabilities(y, y_prob),
            'group_reports': {},
        }
        for group_key in DEFAULT_REPORT_GROUP_KEYS:
            report = build_group_report(y, y_prob, sample_context.get(group_key))
            if report is not None:
                scale_result['group_reports'][group_key] = report
        results[float(gain_scale)] = scale_result
    return results


def gain_stress_rank_key(result):
    """Rank feature experiments by stressed FP robustness, then nominal quality."""
    stressed_scales = [scale for scale in result['gain_stress'] if abs(scale - 1.0) > 1e-9]
    upper_scales = [scale for scale in stressed_scales if scale > 1.0]
    primary_scales = upper_scales or stressed_scales or [1.0]
    max_stressed_fp = max(
        result['gain_stress'][scale]['overall']['fp_rate']
        for scale in primary_scales
    )
    max_stressed_worst_chip_fp = max(
        result['gain_stress'][scale]['group_reports'].get('chip', {}).get(
            'worst_fp_rate',
            {},
        ).get('fp_rate', 0.0)
        for scale in primary_scales
    )
    min_stressed_recall = min(
        result['gain_stress'][scale]['overall']['recall']
        for scale in primary_scales
    )
    nominal = result['gain_stress'].get(1.0)
    nominal_f1 = nominal['overall']['f1'] if nominal else 0.0
    return (
        max_stressed_fp,
        max_stressed_worst_chip_fp,
        -min_stressed_recall,
        -nominal_f1,
        -result['cv']['oof_f1'],
    )


def print_gain_feature_experiment_summary(results, scales):
    """Print raw/relative/hybrid comparison table."""
    print("\n" + "=" * 78)
    print("  GAIN-FEATURE SET EXPERIMENT")
    print("=" * 78)
    print(
        "  set      feat  OOF_F1  nomFP  nomF1 | "
        + " | ".join(f"{scale:g}x FP/worstFP" for scale in scales if abs(scale - 1.0) > 1e-9)
    )
    print("  " + "-" * 112)
    for result in sorted(results, key=gain_stress_rank_key):
        nominal = result['gain_stress'].get(1.0)
        parts = []
        for scale in scales:
            if abs(scale - 1.0) <= 1e-9:
                continue
            row = result['gain_stress'][float(scale)]
            chip_report = row['group_reports'].get('chip', {})
            worst_fp = chip_report.get('worst_fp_rate', {}).get('fp_rate', 0.0)
            parts.append(f"{row['overall']['fp_rate']:5.1f}/{worst_fp:5.1f}")
        print(
            f"  {result['name']:<8} "
            f"{len(result['feature_names']):>4} "
            f"{result['cv']['oof_f1']:>7.1f} "
            f"{nominal['overall']['fp_rate']:>6.1f} "
            f"{nominal['overall']['f1']:>6.1f} | "
            + " | ".join(parts)
        )

    winner = min(results, key=gain_stress_rank_key)
    print(
        f"\nBest by gain-stress ranking: {winner['name']} "
        f"({len(winner['feature_names'])} features)"
    )


def experiment_gain_feature_sets(seed=None, feature_sets=None,
                                 scaler_mode=DEFAULT_SCALER_MODE,
                                 batch_size=DEFAULT_BATCH_SIZE,
                                 fp_weight=DEFAULT_FP_WEIGHT,
                                 hidden_layers=None,
                                 environment_filter=None,
                                 excluded_chips=None,
                                 scales=DEFAULT_GAIN_STRESS_SCALES,
                                 use_cache=True):
    """Compare raw, relative, and hybrid feature sets under gain stress."""
    total_start = perf_counter()
    if hidden_layers is None:
        hidden_layers = list(DEFAULT_HIDDEN_LAYERS)
    if feature_sets is None:
        feature_sets = list(GAIN_FEATURE_EXPERIMENT_SETS)
    feature_sets = [str(item).strip().lower() for item in feature_sets if str(item).strip()]
    scales = parse_gain_stress_scales(scales)
    environment_filter = parse_environment_filter(environment_filter)
    excluded_chips = parse_chip_filter(excluded_chips)

    try:
        ensure_torch_available()
        torch_device_label = describe_torch_device()
        seed = resolve_training_seed(seed)
        set_global_determinism(seed, torch_module=torch)
    except ImportError as exc:
        print(f"Error: Missing dependency - {exc}")
        return 1, seed, None
    except (RuntimeError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1, seed, None

    print("\n" + "=" * 78)
    print("  GAIN-FEATURE EXPERIMENT")
    print("=" * 78)
    print(f"Feature sets: {', '.join(feature_sets)}")
    print(f"Scales: {', '.join(f'{scale:g}' for scale in scales)}")
    print(f"Scaler: {scaler_mode}")
    print(f"Batch size: {batch_size}")
    print(f"Torch device: {torch_device_label}")
    print(f"FP weight: {fp_weight}")
    if environment_filter is not None:
        print(f"Environment filter: {', '.join(sorted(environment_filter))}")
    if excluded_chips is not None:
        print(f"Excluded chips: {', '.join(sorted(excluded_chips))}")

    print("\nLoading raw feature matrix...")
    matrix, _ = load_training_matrix(
        environment_filter=environment_filter,
        excluded_chips=excluded_chips,
        feature_names=RAW_FEATURES,
        use_cache=use_cache,
    )
    stats = matrix['stats']
    if not stats['chips']:
        print("Error: No empty/static_presence/motion data found")
        return 1, seed, None
    print(f"  Chips: {', '.join(stats['chips'])}")
    if stats.get('environment_groups'):
        print(f"  Environments: {', '.join(stats['environment_groups'])}")
    print(f"  Total packets: {stats['total']}")

    X_raw_base = matrix['X']
    y = matrix['y']
    raw_feature_names = matrix['feature_names']
    sample_context = matrix['sample_context']
    sample_weights = matrix['sample_weights']
    eval_groups = sample_context[DEFAULT_PRIMARY_GROUP_KEY]
    print(f"  Samples: {len(X_raw_base)}")
    print(f"  Raw features: {', '.join(raw_feature_names)}")
    print(f"  Class balance: IDLE={np.sum(y == 0)}, MOTION={np.sum(y == 1)}")

    results = []
    for idx, feature_set in enumerate(feature_sets):
        print(f"\n== {feature_set} ==")
        X_candidate, candidate_names = build_gain_feature_matrix(
            X_raw_base,
            raw_feature_names,
            feature_set,
        )
        print(f"  Features ({len(candidate_names)}): {', '.join(candidate_names)}")
        with suppress_stderr():
            cv = cross_validate(
                X_candidate,
                y,
                hidden_layers=hidden_layers,
                n_folds=DEFAULT_CV_FOLDS,
                max_epochs=DEFAULT_MAX_EPOCHS,
                fp_weight=fp_weight,
                sample_weight=sample_weights,
                groups=eval_groups,
                sample_context=sample_context,
                scaler_mode=scaler_mode,
                batch_size=batch_size,
                block_stride=SEG_WINDOW_SIZE,
                block_group_key=DEFAULT_BLOCK_GROUP_KEY,
                report_group_keys=DEFAULT_REPORT_GROUP_KEYS,
                seed=derive_seed(seed, idx),
            )

        scaler = build_preprocessor(scaler_mode)
        X_scaled = scaler.fit_transform(X_candidate)
        with suppress_stderr():
            model = train_model(
                X_scaled,
                y,
                hidden_layers=hidden_layers,
                max_epochs=DEFAULT_MAX_EPOCHS,
                fp_weight=fp_weight,
                sample_weight=sample_weights,
                batch_size=batch_size,
                seed=derive_seed(seed, 10_000, idx),
            )
        stress = evaluate_model_gain_stress(
            model,
            scaler,
            X_candidate,
            y,
            candidate_names,
            sample_context,
            scales=scales,
        )
        nominal = stress.get(1.0)
        max_upper_fp = max(
            stress[scale]['overall']['fp_rate']
            for scale in stress
            if scale > 1.0
        ) if any(scale > 1.0 for scale in stress) else nominal['overall']['fp_rate']
        print(
            f"  CV OOF F1={cv['oof_f1']:.1f}% | "
            f"nominal FP={nominal['overall']['fp_rate']:.1f}% "
            f"F1={nominal['overall']['f1']:.1f}% | "
            f"max upper-scale FP={max_upper_fp:.1f}%"
        )
        results.append({
            'name': feature_set,
            'feature_names': candidate_names,
            'cv': slim_cv_result(cv),
            'gain_stress': stress,
        })

    print_gain_feature_experiment_summary(results, scales)
    print(f"\nTotal experiment time: {format_duration(perf_counter() - total_start)}")
    return 0, seed, results


def build_candidate_key(cv_results):
    """Ranking key for seeds/architectures under the robust evaluation protocol."""
    group_reports = cv_results.get('group_reports', {})
    session_report = group_reports.get('session_group') or {}
    chip_report = group_reports.get('chip') or {}

    worst_session_recall = session_report.get('worst_recall', {}).get('recall', 0.0)
    worst_session_fp = session_report.get('worst_fp_rate', {}).get('fp_rate', 100.0)
    worst_chip_recall = chip_report.get('worst_recall', {}).get('recall', 0.0)

    return (
        worst_session_recall,
        worst_chip_recall,
        -worst_session_fp,
        cv_results.get('oof_f1', 0.0),
        cv_results.get('f1_mean', 0.0),
    )


def build_model(hidden_layers=None, num_features=12, use_dropout=True, dropout_rate=0.2,
                seed=None):
    """
    Build a PyTorch MLP model.

    Dropout layers are added during training for regularization but are
    automatically disabled during inference (and don't affect exported weights).
    
    Args:
        hidden_layers: List of hidden layer sizes
        num_features: Number of input features
        use_dropout: Whether to add dropout layers (for training only)
        dropout_rate: Dropout rate (0.0-1.0)
        seed: Optional base seed for deterministic initializers
    
    Returns:
        TorchMLP instance
    """
    ensure_torch_available()
    return TorchMLP(
        num_features=num_features,
        hidden_layers=hidden_layers,
        use_dropout=use_dropout,
        dropout_rate=dropout_rate,
        seed=seed,
    )


def _compute_weighted_bce(logits, targets, sample_weights=None):
    """Binary cross-entropy on logits with optional per-sample weights."""
    losses = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction='none',
    )
    if sample_weights is not None:
        losses = losses * sample_weights
    return losses.mean()


def train_model(X, y, hidden_layers=None, max_epochs=DEFAULT_MAX_EPOCHS, use_dropout=True,
                class_weight=None, fp_weight=DEFAULT_FP_WEIGHT, sample_weight=None,
                batch_size=DEFAULT_BATCH_SIZE, verbose=0, seed=None):
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
        batch_size: Mini-batch size for SGD/Adam updates
        verbose: Training verbosity
        seed: Optional base seed for deterministic training
    
    Returns:
        Trained TorchMLP model
    """
    torch_mod = ensure_torch_available()

    if hidden_layers is None:
        hidden_layers = list(DEFAULT_HIDDEN_LAYERS)
    
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

    # Merge class weights into sample_weight when both are requested so the
    # optimizer sees a single per-sample weighting term.
    if sample_weight is not None and class_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=np.float32).copy()
        class_multiplier = np.where(np.asarray(y) == 1, class_weight[1], class_weight[0])
        sample_weight *= class_multiplier.astype(np.float32)
        class_weight = None
    
    # Determine number of features from input shape
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    num_features = X.shape[1] if hasattr(X, 'shape') else len(X[0])
    set_global_determinism(seed, torch_module=torch_mod)
    device = resolve_torch_device(torch_module=torch_mod)
    model = build_model(
        hidden_layers=hidden_layers,
        num_features=num_features,
        use_dropout=use_dropout,
        seed=seed,
    ).to(device)
    
    # Keep a stratified validation split instead of relying on implicit slicing.
    from sklearn.model_selection import train_test_split as _val_split
    split_kwargs = dict(test_size=0.1, random_state=42, stratify=np.asarray(y))
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=np.float32)
        X_t, X_v, y_t, y_v, sw_t, sw_v = _val_split(
            X, y, sample_weight, **split_kwargs
        )
    else:
        X_t, X_v, y_t, y_v = _val_split(X, y, **split_kwargs)
        sw_t, sw_v = None, None

    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=DEFAULT_LR_PATIENCE,
        min_lr=1e-6,
    )

    X_t_tensor = torch.from_numpy(np.asarray(X_t, dtype=np.float32)).to(device)
    y_t_tensor = torch.from_numpy(np.asarray(y_t, dtype=np.float32)).view(-1, 1).to(device)
    X_v_tensor = torch.from_numpy(np.asarray(X_v, dtype=np.float32)).to(device)
    y_v_tensor = torch.from_numpy(np.asarray(y_v, dtype=np.float32)).view(-1, 1).to(device)
    sw_t_tensor = (
        None
        if sw_t is None
        else torch.from_numpy(np.asarray(sw_t, dtype=np.float32)).view(-1, 1).to(device)
    )
    sw_v_tensor = (
        None
        if sw_v is None
        else torch.from_numpy(np.asarray(sw_v, dtype=np.float32)).view(-1, 1).to(device)
    )

    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    batch_size = max(1, int(batch_size))

    for epoch in range(int(max_epochs)):
        model.train()
        for start in range(0, len(X_t_tensor), batch_size):
            stop = start + batch_size
            batch_x = X_t_tensor[start:stop]
            batch_y = y_t_tensor[start:stop]
            batch_weights = None
            if sw_t_tensor is not None:
                batch_weights = sw_t_tensor[start:stop].clone()
            if class_weight is not None:
                class_multiplier = torch.where(
                    batch_y > 0.5,
                    float(class_weight[1]),
                    float(class_weight[0]),
                )
                batch_weights = class_multiplier if batch_weights is None else batch_weights * class_multiplier

            optimizer.zero_grad()
            logits = model.forward_logits(batch_x)
            loss = _compute_weighted_bce(logits, batch_y, sample_weights=batch_weights)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model.forward_logits(X_v_tensor)
            val_loss = _compute_weighted_bce(
                val_logits,
                y_v_tensor,
                sample_weights=sw_v_tensor,
            ).item()
        scheduler.step(val_loss)

        if verbose:
            print(
                f"    epoch {epoch + 1:03d}/{max_epochs}: "
                f"val_loss={val_loss:.6f} lr={optimizer.param_groups[0]['lr']:.2e}"
            )

        if val_loss < (best_val_loss - 1e-4):
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= DEFAULT_EARLY_STOP_PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()

    return model


def predict_probabilities(model, X):
    """
    Return a flat probability vector for binary classification.
    """
    logits = predict_logits(model, X)
    return 1.0 / (1.0 + np.exp(-logits))


def predict_tempered_probabilities(model, X, temperature=DEFAULT_ML_TEMPERATURE):
    """
    Return probabilities after applying the same post-logit temperature scaling
    used by Python/C++ runtime inference.
    """
    X = np.asarray(X, dtype=np.float32)
    if temperature == 1.0:
        return predict_probabilities(model, X)
    scaled_logits = predict_logits(model, X) / float(temperature)
    probabilities = np.empty_like(scaled_logits, dtype=np.float32)
    probabilities[scaled_logits < -20.0] = 0.0
    probabilities[scaled_logits > 20.0] = 1.0
    mask = (scaled_logits >= -20.0) & (scaled_logits <= 20.0)
    probabilities[mask] = 1.0 / (1.0 + np.exp(-scaled_logits[mask]))
    return probabilities


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a model on test data and return metrics dict.
    
    Args:
        model: Trained PyTorch model
        X_test: Test features (normalized)
        y_test: Test labels
    
    Returns:
        dict: Metrics (recall, precision, fp_rate, f1, tp, fp, tn, fn)
    """
    probabilities = predict_probabilities(model, X_test)
    return evaluate_probabilities(y_test, probabilities)


def cross_validate(X, y, hidden_layers=None, n_folds=DEFAULT_CV_FOLDS, max_epochs=DEFAULT_MAX_EPOCHS,
                   fp_weight=DEFAULT_FP_WEIGHT, sample_weight=None, groups=None,
                   sample_context=None, scaler_mode=DEFAULT_SCALER_MODE,
                   batch_size=DEFAULT_BATCH_SIZE, block_stride=1,
                   block_group_key=DEFAULT_BLOCK_GROUP_KEY,
                   report_group_keys=DEFAULT_REPORT_GROUP_KEYS, seed=None):
    """
    Perform grouped cross-validation with de-overlapped scoring.

    Args:
        X: Feature matrix (NOT normalized - scaler fit per fold)
        y: Labels
        hidden_layers: List of hidden layer sizes
        n_folds: Number of CV folds
        max_epochs: Maximum training epochs per fold
        fp_weight: Multiplier for class 0 weight (>1.0 penalizes FP more)
        sample_weight: Optional per-sample weights aligned with X/y
        groups: Optional split-group labels per sample
        sample_context: Optional aligned metadata for reporting/blocking
        scaler_mode: Feature normalization mode
        batch_size: Mini-batch size used for fold training
        block_stride: Subsampling stride applied at scoring time
        block_group_key: Group key used for block subsampling
        report_group_keys: Extra group reports to compute from OOF predictions
        seed: Optional base seed for deterministic per-fold training

    Returns:
        dict: Mean and std of each metric across folds
    """
    if hidden_layers is None:
        hidden_layers = list(DEFAULT_HIDDEN_LAYERS)

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
    oof_prob = np.full(len(y), np.nan, dtype=np.float32)
    scored_mask = np.zeros(len(y), dtype=bool)
    fold_timings = []
    cv_start = perf_counter()

    for fold, (train_idx, val_idx) in enumerate(split_iter):
        fold_start = perf_counter()
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        sw_train_fold = sample_weight[train_idx] if sample_weight is not None else None

        # Fit normalization only on the training fold
        preprocess_start = perf_counter()
        scaler = build_preprocessor(scaler_mode)
        X_train_scaled = scaler.fit_transform(X_train_fold)
        X_val_scaled = scaler.transform(X_val_fold)
        preprocess_elapsed = perf_counter() - preprocess_start

        train_predict_start = perf_counter()
        fold_seed = derive_seed(seed, fold)
        with suppress_stderr():
            model = train_model(X_train_scaled, y_train_fold,
                                hidden_layers=hidden_layers, max_epochs=max_epochs,
                                fp_weight=fp_weight, sample_weight=sw_train_fold,
                                batch_size=batch_size, seed=fold_seed)
            val_prob = predict_probabilities(model, X_val_scaled)
        train_predict_elapsed = perf_counter() - train_predict_start

        oof_prob[val_idx] = val_prob
        scoring_start = perf_counter()
        val_context = slice_sample_context(sample_context, val_idx)
        local_mask = build_block_mask(
            val_context,
            stride=block_stride,
            group_key=block_group_key,
        )
        if local_mask is None:
            local_mask = np.ones(len(val_idx), dtype=bool)

        scored_idx = val_idx[local_mask]
        scored_mask[scored_idx] = True
        metrics = evaluate_probabilities(y_val_fold[local_mask], val_prob[local_mask])
        fold_metrics.append(metrics)
        scoring_elapsed = perf_counter() - scoring_start
        fold_elapsed = perf_counter() - fold_start
        fold_timings.append(fold_elapsed)
        print(
            f"  Fold {fold + 1}/{effective_folds} timing: "
            f"preprocess={format_duration(preprocess_elapsed)}, "
            f"train+predict={format_duration(train_predict_elapsed)}, "
            f"score={format_duration(scoring_elapsed)}, "
            f"total={format_duration(fold_elapsed)}"
        )

    # Aggregate
    result = {}
    for key in fold_metrics[0]:
        values = [m[key] for m in fold_metrics]
        result[f'{key}_mean'] = np.mean(values)
        result[f'{key}_std'] = np.std(values)

    scored_idx = np.flatnonzero(scored_mask)
    oof_metrics = evaluate_probabilities(y[scored_idx], oof_prob[scored_idx])
    for key, value in oof_metrics.items():
        result[f'oof_{key}'] = value

    result['n_folds'] = len(fold_metrics)
    result['scored_samples'] = int(len(scored_idx))
    result['dense_samples'] = int(np.sum(~np.isnan(oof_prob)))
    result['scaler_mode'] = scaler_mode
    result['timings'] = {
        'fold_seconds': fold_timings,
        'total_seconds': perf_counter() - cv_start,
    }

    if sample_context is not None and report_group_keys:
        scored_context = slice_sample_context(sample_context, scored_idx)
        group_reports = {}
        for group_key in report_group_keys:
            report = build_group_report(
                y[scored_idx],
                oof_prob[scored_idx],
                scored_context.get(group_key),
            )
            if report is not None:
                group_reports[group_key] = report
        result['group_reports'] = group_reports

    return result


def get_model_architecture(model):
    """Return the layer sizes of a dense MLP as [input, ..., output]."""
    weights = extract_model_weights(model)
    if not weights:
        return []

    layer_sizes = [int(weights[0].shape[0])]
    for idx in range(0, len(weights), 2):
        layer_sizes.append(int(weights[idx].shape[1]))
    return layer_sizes


def export_micropython(model, scaler, output_path, seed=None,
                       feature_names=None, scaler_mode=DEFAULT_SCALER_MODE):
    """
    Export model weights to MicroPython code.
    
    Generates ml_weights.py with network weights only.
    The inference functions are in ml_detector.py (not auto-generated).
    
    Args:
        model: Trained PyTorch model
        scaler: Fitted preprocessing object exposing center/scale arrays
        output_path: Output file path
        seed: Random seed used for training (or None if not set)
        feature_names: Ordered feature names expected by the model
        scaler_mode: Normalization mode used during training
    
    Returns:
        Size of generated code
    """
    from datetime import datetime
    weights = extract_model_weights(model)
    center, scale = get_preprocessor_arrays(scaler)
    architecture = get_model_architecture(model)
    if feature_names is None:
        feature_names = list(TRAINING_FEATURES)
    
    seed_info = f"Seed: {seed}"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    architecture_text = ' -> '.join(map(str, architecture))
    architecture_csv = ', '.join(str(x) for x in architecture)
    hidden_csv = ', '.join(str(x) for x in architecture[1:-1])
    feature_csv = ', '.join(repr(name) for name in feature_names)
    center_csv = ', '.join(f'{x:.9g}' for x in center)
    scale_csv = ', '.join(f'{x:.9g}' for x in scale)
    
    # Build code - weights only
    code = f'''"""
Micro-ESPectre - ML Model Weights

Auto-generated neural network weights for motion detection.
Architecture: {architecture_text}
Normalization: {scaler_mode}
Trained: {timestamp}
{seed_info}

This file is auto-generated by 10_train_ml_model.py.
DO NOT EDIT - your changes will be overwritten!

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

# Model metadata
MODEL_LAYER_SIZES = [{architecture_csv}]
MODEL_HIDDEN_LAYERS = [{hidden_csv}]
ML_NUM_FEATURES = {architecture[0]}
ML_NUM_LAYERS = {len(architecture) - 1}
NORMALIZATION_MODE = "{scaler_mode}"
FEATURE_NAMES = [{feature_csv}]

# Feature normalization
FEATURE_MEAN = [{center_csv}]
FEATURE_SCALE = [{scale_csv}]

'''
    
    # Add weights for each layer
    weight_names = []
    bias_names = []
    for i in range(0, len(weights), 2):
        W = weights[i]
        b = weights[i + 1]
        layer_num = i // 2 + 1
        in_size, out_size = W.shape
        
        activation = 'Sigmoid' if i == len(weights) - 2 else 'ReLU'
        code += f'# Layer {layer_num}: {in_size} -> {out_size} ({activation})\n'
        code += f'W{layer_num} = [\n'
        for row in W:
            code += '    [' + ', '.join(f'{x:.9g}' for x in row) + '],\n'
        code += ']\n'
        code += f'B{layer_num} = [' + ', '.join(f'{x:.9g}' for x in b) + ']\n\n'
        weight_names.append(f'W{layer_num}')
        bias_names.append(f'B{layer_num}')

    code += f'WEIGHTS = [{", ".join(weight_names)}]\n'
    code += f'BIASES = [{", ".join(bias_names)}]\n'
    
    with open(output_path, 'w') as f:
        f.write(code)
    
    return len(code)


def export_cpp_weights(model, scaler, output_path, seed=None,
                       feature_names=None, scaler_mode=DEFAULT_SCALER_MODE):
    """
    Export model weights to C++ header for ESPHome.
    
    Generates ml_weights.h with constexpr weights.
    
    Args:
        model: Trained PyTorch model
        scaler: Fitted preprocessing object exposing center/scale arrays
        output_path: Output file path
        seed: Random seed used for training (or None if not set)
        feature_names: Ordered feature names expected by the model
        scaler_mode: Normalization mode used during training
    
    Returns:
        Size of generated code
    """
    from datetime import datetime
    weights = extract_model_weights(model)
    architecture = get_model_architecture(model)
    arch = ' -> '.join(map(str, architecture))
    center, scale = get_preprocessor_arrays(scaler)
    if feature_names is None:
        feature_names = list(TRAINING_FEATURES)
    
    seed_info = f"Seed: {seed}"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    architecture_csv = ', '.join(str(x) for x in architecture)
    center_csv = ', '.join(f'{x:.9g}f' for x in center)
    scale_csv = ', '.join(f'{x:.9g}f' for x in scale)
    
    code = f'''/*
 * ESPectre - ML Model Weights
 * 
 * Auto-generated neural network weights for motion detection.
 * Architecture: {arch}
 * Normalization: {scaler_mode}
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

// Model metadata
constexpr uint8_t ML_MODEL_NUM_LAYERS = {len(architecture) - 1};
constexpr uint8_t ML_MODEL_INPUT_SIZE = {architecture[0]};
constexpr uint8_t ML_MAX_LAYER_WIDTH = {max(architecture[1:])};
constexpr uint8_t ML_MODEL_LAYER_SIZES[{len(architecture)}] = {{{architecture_csv}}};
constexpr char ML_NORMALIZATION_MODE[] = "{scaler_mode}";

// Feature normalization
constexpr float ML_FEATURE_MEAN[{len(center)}] = {{{center_csv}}};
constexpr float ML_FEATURE_SCALE[{len(scale)}] = {{{scale_csv}}};

'''
    
    # Add weights for each layer
    weight_names = []
    bias_names = []
    input_sizes = []
    output_sizes = []
    for i in range(0, len(weights), 2):
        W = weights[i]
        b = weights[i + 1]
        layer_num = i // 2 + 1
        in_size, out_size = W.shape
        
        activation = 'Sigmoid' if i == len(weights) - 2 else 'ReLU'
        code += f'// Layer {layer_num}: {in_size} -> {out_size} ({activation})\n'
        flat_weights = W.reshape(-1)
        code += f'constexpr float ML_W{layer_num}[{len(flat_weights)}] = {{' \
                + ', '.join(f'{x:.9g}f' for x in flat_weights) + '};\n'
        code += f'constexpr float ML_B{layer_num}[{out_size}] = {{{", ".join(f"{x:.9g}f" for x in b)}}};\n\n'
        weight_names.append(f'ML_W{layer_num}')
        bias_names.append(f'ML_B{layer_num}')
        input_sizes.append(str(in_size))
        output_sizes.append(str(out_size))

    code += (
        f'constexpr uint8_t ML_MODEL_LAYER_INPUT_SIZES[ML_MODEL_NUM_LAYERS] = '
        f'{{{", ".join(input_sizes)}}};\n'
    )
    code += (
        f'constexpr uint8_t ML_MODEL_LAYER_OUTPUT_SIZES[ML_MODEL_NUM_LAYERS] = '
        f'{{{", ".join(output_sizes)}}};\n'
    )
    code += (
        f'constexpr const float* ML_MODEL_WEIGHTS[ML_MODEL_NUM_LAYERS] = '
        f'{{{", ".join(weight_names)}}};\n'
    )
    code += (
        f'constexpr const float* ML_MODEL_BIASES[ML_MODEL_NUM_LAYERS] = '
        f'{{{", ".join(bias_names)}}};\n\n'
    )
    
    code += '''}  // namespace espectre
}  // namespace esphome
'''
    
    with open(output_path, 'w') as f:
        f.write(code)
    
    return len(code)


def export_test_data(model, scaler, X_test_raw, y_test, output_path, sample_context=None):
    """
    Export test data for validation across Python and C++.
    
    Generates ml_test_data.npz with RAW features (not normalized) and expected outputs.
    This allows testing the full pipeline including normalization.
    
    Args:
        model: Trained PyTorch model
        scaler: Fitted preprocessing object used for normalization
        X_test_raw: Test features (NOT normalized, raw values)
        y_test: Test labels
        output_path: Output file path
        sample_context: Optional aligned metadata to save alongside the samples
    
    Returns:
        Number of test samples
    """
    # Normalize for prediction
    X_test_scaled = scaler.transform(X_test_raw)
    predictions = predict_tempered_probabilities(model, X_test_scaled)
    
    # Save RAW features (not normalized) so tests can verify full pipeline
    payload = {
        'features': X_test_raw.astype(np.float32),
        'labels': y_test.astype(np.int32),
        'expected_outputs': predictions.astype(np.float32),
    }
    if sample_context is not None:
        if 'source_file' in sample_context:
            payload['source_files'] = np.asarray(sample_context['source_file'])
        if 'session_group' in sample_context:
            payload['session_groups'] = np.asarray(sample_context['session_group'])

    np.savez(output_path, **payload)
    
    return len(X_test_raw)


# ============================================================================
# Feature Importance (Correlation)
# ============================================================================

def calculate_correlation_importance(feature_names=None, use_cache=True):
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
    
    matrix, _ = load_training_matrix(
        feature_names=feature_names,
        use_cache=use_cache,
    )
    stats = matrix['stats']
    print(f"  Loaded {stats['total']} packets")
    if stats.get('cv_norm_files'):
        print(f"  Files using CV normalization: {len(stats['cv_norm_files'])}")

    X = matrix['X']
    y = matrix['y']
    actual_features = matrix['feature_names']
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
    from features import DEFAULT_FEATURES
    
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
        model: Trained PyTorch model
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
    explainer = shap.Explainer(
        lambda values: predict_probabilities(model, values).reshape(-1, 1),
        background,
        algorithm='permutation',
    )
    
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

def run_ablation_study(X, y, feature_names, sample_context=None, sample_weight=None,
                       hidden_layers=None, fp_weight=DEFAULT_FP_WEIGHT,
                       scaler_mode=DEFAULT_SCALER_MODE,
                       batch_size=DEFAULT_BATCH_SIZE):
    """
    Run ablation study: train model removing one feature at a time.
    
    This helps identify which features are truly important by measuring
    the impact of removing each one. Features whose removal improves or
    doesn't affect F1 are candidates for elimination.
    
    Args:
        X: Feature matrix (NOT normalized - scaler fit per fold)
        y: Labels
        feature_names: List of feature names
        sample_context: Optional aligned metadata for grouped CV
        sample_weight: Optional per-sample weights
        hidden_layers: Model architecture
        fp_weight: FP penalty weight
        scaler_mode: Feature normalization mode
        batch_size: Mini-batch size used during fold training
    
    Returns:
        list: Results for each ablation experiment
    """
    print("\n" + "="*80)
    print("                         ABLATION STUDY")
    print("="*80 + "\n")
    print("Training models with one feature removed at a time to measure impact...\n")

    if hidden_layers is None:
        hidden_layers = list(DEFAULT_HIDDEN_LAYERS)

    groups = None
    if sample_context is not None:
        groups = sample_context.get(DEFAULT_PRIMARY_GROUP_KEY)

    results = []

    # Baseline (all features)
    print(f"[1/{len(feature_names)+1}] Baseline (all {len(feature_names)} features)...")
    with suppress_stderr():
        static_presence_cv = cross_validate(
            X, y,
            hidden_layers=hidden_layers,
            n_folds=DEFAULT_CV_FOLDS,
            max_epochs=DEFAULT_MAX_EPOCHS,
            fp_weight=fp_weight,
            sample_weight=sample_weight,
            groups=groups,
            sample_context=sample_context,
            scaler_mode=scaler_mode,
            batch_size=batch_size,
            block_stride=SEG_WINDOW_SIZE,
        )
    static_presence_f1 = static_presence_cv['f1_mean']
    results.append({
        'removed': 'None (baseline)',
        'n_features': len(feature_names),
        'f1_mean': static_presence_f1,
        'f1_std': static_presence_cv['f1_std'],
        'oof_f1': static_presence_cv['oof_f1'],
        'recall_mean': static_presence_cv['recall_mean'],
        'fp_rate_mean': static_presence_cv['fp_rate_mean'],
        'delta_f1': 0.0,
    })
    print(
        f"    F1: {static_presence_f1:.2f}% (+/- {static_presence_cv['f1_std']:.2f}%), "
        f"blocked OOF={static_presence_cv['oof_f1']:.2f}%\n"
    )

    # Remove each feature one at a time
    for i, feature_name in enumerate(feature_names):
        print(f"[{i+2}/{len(feature_names)+1}] Removing '{feature_name}'...")

        # Create X without this feature
        X_ablated = np.delete(X, i, axis=1)

        with suppress_stderr():
            cv = cross_validate(
                X_ablated, y,
                hidden_layers=hidden_layers,
                n_folds=DEFAULT_CV_FOLDS,
                max_epochs=DEFAULT_MAX_EPOCHS,
                fp_weight=fp_weight,
                sample_weight=sample_weight,
                groups=groups,
                sample_context=sample_context,
                scaler_mode=scaler_mode,
                batch_size=batch_size,
                block_stride=SEG_WINDOW_SIZE,
            )

        f1 = cv['f1_mean']
        delta = f1 - static_presence_f1

        results.append({
            'removed': feature_name,
            'n_features': len(feature_names) - 1,
            'f1_mean': f1,
            'f1_std': cv['f1_std'],
            'oof_f1': cv['oof_f1'],
            'recall_mean': cv['recall_mean'],
            'fp_rate_mean': cv['fp_rate_mean'],
            'delta_f1': delta,
        })

        direction = "↑" if delta > 0.1 else "↓" if delta < -0.1 else "≈"
        print(
            f"    F1: {f1:.2f}% ({direction} {delta:+.2f}%), "
            f"blocked OOF={cv['oof_f1']:.2f}%\n"
        )

    # Print summary table
    print("\n" + "="*85)
    print("                           ABLATION SUMMARY")
    print("="*85 + "\n")
    
    # Sort by delta (worst impact first = most important features)
    sorted_results = sorted(results[1:], key=lambda r: r['delta_f1'])
    
    print(f"{'Removed Feature':<24} {'F1 (CV)':>14} {'OOF F1':>10} {'Delta':>10} {'Recall':>10} {'FP Rate':>10} {'Note':<12}")
    print("-"*85)
    
    # Print baseline first
    bl = results[0]
    print(f"{'None (baseline)':<24} {bl['f1_mean']:>8.2f}% +/-{bl['f1_std']:.1f} "
          f"{bl['oof_f1']:>9.2f}% {'---':>10} {bl['recall_mean']:>9.1f}% {bl['fp_rate_mean']:>9.1f}%")
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
              f"{r['oof_f1']:>9.2f}% {delta_str:>10} {r['recall_mean']:>9.1f}% {r['fp_rate_mean']:>9.1f}% {note:<12}")
    
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

def print_cv_summary(cv_results, title="Primary grouped CV"):
    """Print the robust evaluation summary used for model selection."""
    print(f"\n{title}:")
    print(f"  Fold recall:    {cv_results['recall_mean']:.1f}% (+/- {cv_results['recall_std']:.1f}%)")
    print(f"  Fold precision: {cv_results['precision_mean']:.1f}% (+/- {cv_results['precision_std']:.1f}%)")
    print(f"  Fold FP rate:   {cv_results['fp_rate_mean']:.1f}% (+/- {cv_results['fp_rate_std']:.1f}%)")
    print(f"  Fold F1:        {cv_results['f1_mean']:.1f}% (+/- {cv_results['f1_std']:.1f}%)")
    print(f"  Blocked OOF F1: {cv_results['oof_f1']:.1f}%")
    print(f"  Scored windows: {cv_results['scored_samples']} / {cv_results['dense_samples']}")

    group_reports = cv_results.get('group_reports', {})
    for group_key in DEFAULT_REPORT_GROUP_KEYS:
        report = group_reports.get(group_key)
        if not report:
            continue
        worst_recall = report['worst_recall']
        worst_fp = report['worst_fp_rate']
        print(
            f"  Worst {group_key} recall: "
            f"{worst_recall['group']} -> R={worst_recall['recall']:.1f}% "
            f"FP={worst_recall['fp_rate']:.1f}% (n={worst_recall['samples']})"
        )
        if worst_fp['group'] != worst_recall['group']:
            print(
                f"  Worst {group_key} FP:     "
                f"{worst_fp['group']} -> FP={worst_fp['fp_rate']:.1f}% "
                f"R={worst_fp['recall']:.1f}% (n={worst_fp['samples']})"
            )


def _format_confusion_matrix_rows(confusion_matrix, class_names):
    """Render a compact confusion matrix table."""
    confusion_matrix = np.asarray(confusion_matrix, dtype=np.int32)
    header = "pred -> " + " ".join(f"{name:>16}" for name in class_names)
    rows = [header]
    for idx, class_name in enumerate(class_names):
        values = " ".join(f"{int(value):>16d}" for value in confusion_matrix[idx])
        rows.append(f"actual {class_name:<8} {values}")
    return rows


def select_regression_subset_indices(sample_context, max_samples=2048, block_stride=SEG_WINDOW_SIZE):
    """Pick a deterministic subset for inference-regression artifacts."""
    if sample_context is None:
        return np.arange(0, max_samples)

    mask = build_block_mask(
        sample_context,
        stride=block_stride,
        group_key=DEFAULT_BLOCK_GROUP_KEY,
    )
    indices = np.flatnonzero(mask) if mask is not None else np.arange(len(next(iter(sample_context.values()))))
    if len(indices) == 0:
        return indices
    if len(indices) > max_samples:
        sampled = np.linspace(0, len(indices) - 1, num=max_samples, dtype=int)
        indices = indices[sampled]
    return indices


def read_exported_seed():
    """Read the seed embedded in generated weight files."""
    for path in (SRC_DIR / 'ml_weights.py', CPP_DIR / 'ml_weights.h'):
        if not path.exists():
            continue
        try:
            with open(path, 'r', encoding='utf-8') as f:
                contents = f.read()
        except OSError:
            continue
        match = re.search(r'Seed:\s*(\d+)', contents)
        if match:
            return int(match.group(1))
    return None

def show_info():
    """Show dataset information."""
    print("\n" + "="*60)
    print("              DATASET INFORMATION")
    print("="*60 + "\n")
    
    # Load dataset info
    dataset_info = load_dataset_info()
    
    print("Labels defined in dataset_info.json:")
    for label, info in dataset_info.get('labels', {}).items():
        label_type = "MOTION" if label == 'motion' else "IDLE"
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
    print(f"Session groups: {len(stats.get('session_groups', []))}")
    print(f"Named environments: {len(stats.get('environment_groups', []))}")
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


def train_all(fp_weight=DEFAULT_FP_WEIGHT, seed=None, feature_names=None,
              feature_importance=False, ablation=False, shap_samples=200,
              hidden_layers=None, scaler_mode=DEFAULT_SCALER_MODE,
              batch_size=DEFAULT_BATCH_SIZE, export_artifacts=True,
              environment_filter=None, excluded_chips=None,
              positive_chip_boost=None,
              sample_weight_mode=DEFAULT_SAMPLE_WEIGHT_MODE,
              use_cache=True):
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
        hidden_layers: Hidden layer widths. None uses DEFAULT_HIDDEN_LAYERS.
        scaler_mode: Feature normalization mode.
        batch_size: Mini-batch size used for training and CV.
        export_artifacts: If False, stop after robust CV evaluation.
        environment_filter: Optional environment name(s) to keep.
        excluded_chips: Optional chip name(s) to exclude.
        positive_chip_boost: Optional {CHIP: factor} boost applied to motion
                             samples after feature extraction.
        sample_weight_mode: Base sample weighting policy.
        use_cache: If True, reuse cached feature matrix and base sample weights.

    Returns:
        tuple[int, int | None, dict | None]:
            (exit_code, used_seed, evaluation_summary)
            - exit_code: 0 on success, non-zero on failure
            - used_seed: seed used for training (None only on early dependency errors)
            - evaluation_summary: CV report used for model selection
    """
    total_start = perf_counter()
    environment_filter = parse_environment_filter(environment_filter)
    excluded_chips = parse_chip_filter(excluded_chips)
    positive_chip_boost = parse_positive_chip_boost(positive_chip_boost)
    sample_weight_mode = normalize_sample_weight_mode(sample_weight_mode)
    if hidden_layers is None:
        hidden_layers = list(DEFAULT_HIDDEN_LAYERS)
    if feature_names is None:
        feature_names = DEFAULT_FEATURES.copy()
    feature_names = list(feature_names)
    
    print("\n" + "="*60)
    print("           ML MOTION DETECTOR TRAINING")
    print("="*60 + "\n")
    print(f"Fixed subcarriers: {list(DEFAULT_SUBCARRIERS)}\n")
    
    # Check dependencies and initialize deterministic training state.
    try:
        ensure_torch_available()
        torch_device_label = describe_torch_device()
        seed = resolve_training_seed(seed, trailing_newline=True)
        set_global_determinism(seed, torch_module=torch)
    except ImportError as e:
        print(f"Error: Missing dependency - {e}")
        print("Install with: pip install torch scikit-learn")
        return 1, None, None
    except (RuntimeError, ValueError) as e:
        print(f"Error: {e}")
        return 1, seed, None
    
    # Load or build the feature matrix used by training and CV.
    print("Loading training matrix...")
    matrix, _all_packets = load_training_matrix(
        environment_filter=environment_filter,
        excluded_chips=excluded_chips,
        feature_names=feature_names if feature_names is not None else DEFAULT_FEATURES.copy(),
        sample_weight_mode=sample_weight_mode,
        use_cache=use_cache,
    )
    X = matrix['X']
    y = matrix['y']
    actual_feature_names = matrix['feature_names']
    sample_context = matrix['sample_context']
    sample_weights = matrix['sample_weights']
    stats = matrix['stats']
    
    if not stats['chips']:
        print("Error: No datasets found in data/")
        print("Collect data using: ./espectre collect --label static_presence --duration 60")
        return 1, seed, None
    
    print(f"  Chips: {', '.join(stats['chips'])}")
    if environment_filter is not None:
        print(f"  Environment filter: {', '.join(sorted(environment_filter))}")
    if stats.get('excluded_chips'):
        print(f"  Excluded chips: {', '.join(stats['excluded_chips'])}")
    if stats.get('excluded_environments'):
        print(f"  Excluded environments: {', '.join(stats['excluded_environments'])}")
    if stats.get('cv_norm_files'):
        print(f"  Files using CV normalization: {len(stats['cv_norm_files'])}")
    print(f"  Session groups: {len(stats.get('session_groups', []))}")
    if stats.get('environment_groups'):
        print(f"  Named environments: {len(stats['environment_groups'])}")
    for label, count in sorted(stats['labels'].items()):
        print(f"  {label}: {count} packets")
    print(f"  Total: {stats['total']} packets")
    
    print(f"Architecture: {' -> '.join(map(str, [len(feature_names)] + hidden_layers + [1]))}")
    print(f"Scaler: {scaler_mode}")
    print(f"Batch size: {batch_size}\n")
    print(f"Torch device: {torch_device_label}\n")
    
    print(f"  Samples: {len(X)}")
    print(f"  Features: {len(actual_feature_names)}")
    print(f"  Feature set: {', '.join(actual_feature_names)}")
    n_idle = np.sum(y == 0)
    n_motion = np.sum(y == 1)
    print(f"  Class balance: IDLE={n_idle}, MOTION={n_motion}")
    if n_idle > 0 and n_motion > 0:
        ratio = max(n_idle, n_motion) / min(n_idle, n_motion)
        print(f"  Imbalance ratio: {ratio:.1f}:1")

    eval_groups = sample_context[DEFAULT_PRIMARY_GROUP_KEY]
    unique_eval_groups = len(set(eval_groups))
    print(f"  Primary eval groups ({DEFAULT_PRIMARY_GROUP_KEY}): {unique_eval_groups}")
    print(f"  Evaluation block stride: {SEG_WINDOW_SIZE} windows per source file")

    boosted_weights, boost_summary = apply_positive_chip_boost(
        sample_weights,
        sample_context,
        y,
        positive_chip_boost,
    )
    sample_weights = boosted_weights
    if len(sample_weights) != len(X):
        print(
            f"Error: sample weights mismatch (weights={len(sample_weights)}, samples={len(X)})."
        )
        return 1, seed, None
    print(f"  Weight mode: {sample_weight_mode}")
    print(
        f"  Weight stats: min={float(np.min(sample_weights)):.3f}, "
        f"max={float(np.max(sample_weights)):.3f}, "
        f"mean={float(np.mean(sample_weights)):.3f}"
    )
    if positive_chip_boost is not None:
        applied = [
            f"{chip}x{info['factor']:.2f} ({info['affected']} motion windows)"
            for chip, info in boost_summary.items()
        ]
        print(f"  Positive chip boost: {', '.join(applied) if applied else 'none'}")
    
    # Run ablation study if requested
    if ablation:
        run_ablation_study(
            X, y, actual_feature_names,
            sample_context=sample_context,
            sample_weight=sample_weights,
            hidden_layers=hidden_layers,
            fp_weight=fp_weight,
            scaler_mode=scaler_mode,
            batch_size=batch_size,
        )
        return 0, seed, None

    if fp_weight != 1.0:
        print(f"\nFP weight: {fp_weight}x (penalizing false positives)")
    print(
        f"\n{min(DEFAULT_CV_FOLDS, unique_eval_groups)}-fold grouped CV by "
        f"{DEFAULT_PRIMARY_GROUP_KEY}..."
    )
    cv_start = perf_counter()
    with suppress_stderr():
        cv_results = cross_validate(
            X, y,
            hidden_layers=hidden_layers,
            n_folds=DEFAULT_CV_FOLDS,
            max_epochs=DEFAULT_MAX_EPOCHS,
            fp_weight=fp_weight,
            sample_weight=sample_weights,
            groups=eval_groups,
            sample_context=sample_context,
            scaler_mode=scaler_mode,
            batch_size=batch_size,
            block_stride=SEG_WINDOW_SIZE,
            block_group_key=DEFAULT_BLOCK_GROUP_KEY,
            report_group_keys=DEFAULT_REPORT_GROUP_KEYS,
            seed=seed,
        )
    cv_elapsed = perf_counter() - cv_start
    print(f"\nCV total time: {format_duration(cv_elapsed)}")

    print_cv_summary(cv_results)

    if not export_artifacts:
        return 0, seed, cv_results

    regression_indices = select_regression_subset_indices(
        sample_context,
        max_samples=2048,
        block_stride=SEG_WINDOW_SIZE,
    )

    # Train final model on full dataset for production export
    print("\nTraining final model on full dataset...")
    final_train_start = perf_counter()
    scaler = build_preprocessor(scaler_mode)
    X_scaled = scaler.fit_transform(X)
    
    with suppress_stderr():
        model = train_model(
            X_scaled, y,
            hidden_layers=hidden_layers,
            max_epochs=DEFAULT_MAX_EPOCHS,
            fp_weight=fp_weight,
            sample_weight=sample_weights,
            batch_size=batch_size,
            seed=derive_seed(seed, 10_000),
        )
    print(f"  Final training time: {format_duration(perf_counter() - final_train_start)}")
    
    # Calculate SHAP feature importance if requested
    if feature_importance:
        importance = calculate_shap_importance(model, X_scaled, actual_feature_names, 
                                               n_samples=shap_samples)
        if importance:
            print_feature_importance(importance)
    
    # Export models
    print("\nExporting model artifacts...")
    export_start = perf_counter()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # MicroPython weights
    mp_path = SRC_DIR / 'ml_weights.py'
    mp_size = export_micropython(
        model, scaler, mp_path,
        seed=seed,
        feature_names=actual_feature_names,
        scaler_mode=scaler_mode,
    )
    print(f"  MicroPython weights: {mp_path.name} ({mp_size/1024:.1f} KB)")
    
    # C++ weights for ESPHome
    cpp_path = CPP_DIR / 'ml_weights.h'
    cpp_size = export_cpp_weights(
        model, scaler, cpp_path,
        seed=seed,
        feature_names=actual_feature_names,
        scaler_mode=scaler_mode,
    )
    print(f"  C++ weights: {cpp_path.name} ({cpp_size/1024:.1f} KB)")

    # Test data for validation (save deterministic regression subset)
    with suppress_stderr():
        GENERATED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        test_data_path = GENERATED_DATA_DIR / 'ml_test_data.npz'
        n_test = export_test_data(
            model,
            scaler,
            X[regression_indices],
            y[regression_indices],
            test_data_path,
            sample_context=slice_sample_context(sample_context, regression_indices),
        )
    print(f"  Test data: {test_data_path.name} ({n_test} blocked samples)")
    print(f"  Export time: {format_duration(perf_counter() - export_start)}")
    
    print("\n" + "="*60)
    print("                    DONE!")
    print("="*60)
    print(
        f"\nModel trained with blocked grouped CV F1={cv_results['oof_f1']:.1f}% "
        f"(fold mean {cv_results['f1_mean']:.1f}% +/- {cv_results['f1_std']:.1f}%)"
    )
    print(f"\nGenerated files:")
    print(f"  - {mp_path} (MicroPython)")
    print(f"  - {cpp_path} (C++ ESPHome)")
    print(f"  - {test_data_path} (test data for validation)")
    print(f"\nTotal runtime: {format_duration(perf_counter() - total_start)}")
    print()
    
    return 0, seed, cv_results


def summarize_gate(by_chip):
    """Aggregate per-chip gate metrics."""
    rows = list(by_chip.values())
    if not rows:
        return None
    return {
        'by_chip': by_chip,
        'pass_count': int(sum(1 for row in rows if row['recall'] > 95.0 and row['fp_rate'] < 5.0)),
        'mean_recall': float(np.mean([row['recall'] for row in rows])),
        'worst_chip_recall': float(np.min([row['recall'] for row in rows])),
        'mean_fp_rate': float(np.mean([row['fp_rate'] for row in rows])),
        'max_fp_rate': float(np.max([row['fp_rate'] for row in rows])),
        'mean_f1': float(np.mean([row['f1'] for row in rows])),
        'worst_chip_f1': float(np.min([row['f1'] for row in rows])),
        'total_fp': int(sum(row['fp'] for row in rows)),
        'total_fn': int(sum(row['fn'] for row in rows)),
    }


class StreamingEvaluator:
    """Evaluate a trained model with the runtime-equivalent feature path."""

    def __init__(self, model, scaler, feature_names):
        self.feature_names = list(feature_names)
        self.center, self.scale = get_preprocessor_arrays(scaler)
        raw_weights = extract_model_weights(model)
        self.layers = []
        for idx in range(0, len(raw_weights), 2):
            weights = raw_weights[idx]
            biases = raw_weights[idx + 1]
            is_output = idx == len(raw_weights) - 2
            self.layers.append((weights, biases, is_output))
        self.context = SegmentationContext(
            window_size=SEG_WINDOW_SIZE,
            threshold=1.0,
            enable_hampel=True,
            hampel_window=HAMPEL_WINDOW,
            hampel_threshold=HAMPEL_THRESHOLD,
        )
        self.context.use_cv_normalization = False
        self.current_amplitudes = None

    def _predict_probability(self, features):
        activations = (np.asarray(features, dtype=np.float32) - self.center) / self.scale
        for weights, biases, is_output in self.layers:
            activations = activations @ weights + biases
            if not is_output:
                activations = activations.clip(min=0.0)

        logit = float(activations.reshape(-1)[0]) / float(DEFAULT_ML_TEMPERATURE)
        if logit < -20.0:
            return 0.0
        if logit > 20.0:
            return 1.0
        return 1.0 / (1.0 + np.exp(-logit))

    def process_packet(self, csi_data):
        turbulence, amplitudes = self.context.calculate_spatial_turbulence(
            csi_data,
            DEFAULT_SUBCARRIERS,
            return_amplitudes=True,
        )
        self.current_amplitudes = amplitudes
        self.context.add_turbulence(turbulence)
        if self.context.buffer_count < self.context.window_size:
            return None

        idx = self.context.buffer_index
        turb_list = (
            self.context.turbulence_buffer[idx:]
            + self.context.turbulence_buffer[:idx]
        )
        features = extract_features_by_name(
            turb_list,
            len(turb_list),
            amplitudes=self.current_amplitudes,
            feature_names=self.feature_names,
        )
        return float(self._predict_probability(features))


def evaluate_split(model, scaler, feature_names, static_presence_packets, motion_packets, threshold=0.5):
    """Evaluate a split with the same windowing path used at runtime."""
    evaluator = StreamingEvaluator(model, scaler, feature_names)

    warmup = SEG_WINDOW_SIZE
    static_presence_eval_count = max(len(static_presence_packets) - warmup, 0)
    motion_eval_count = max(len(motion_packets) - warmup, 0)
    static_presence_motion_packets = 0
    motion_with_motion = 0
    motion_without_motion = 0

    for i, pkt in enumerate(static_presence_packets):
        prob = evaluator.process_packet(pkt['csi_data'])
        if i >= warmup and prob is not None and prob > threshold:
            static_presence_motion_packets += 1

    for i, pkt in enumerate(motion_packets):
        prob = evaluator.process_packet(pkt['csi_data'])
        if i >= warmup and prob is not None:
            if prob > threshold:
                motion_with_motion += 1
            else:
                motion_without_motion += 1

    tp = motion_with_motion
    fn = motion_without_motion
    fp = static_presence_motion_packets
    tn = max(static_presence_eval_count - static_presence_motion_packets, 0)
    recall = tp / (tp + fn) * 100.0 if (tp + fn) else 0.0
    precision = tp / (tp + fp) * 100.0 if (tp + fp) else 0.0
    fp_rate = fp / static_presence_eval_count * 100.0 if static_presence_eval_count else 0.0
    f1 = (
        2 * (precision / 100.0) * (recall / 100.0) / ((precision + recall) / 100.0) * 100.0
        if (precision + recall)
        else 0.0
    )
    return {
        'recall': float(recall),
        'precision': float(precision),
        'fp_rate': float(fp_rate),
        'f1': float(f1),
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn),
        'static_presence_eval_count': int(static_presence_eval_count),
        'motion_eval_count': int(motion_eval_count),
    }


def evaluate_paired_gate(model, scaler, feature_names, threshold=0.5, chips=None):
    """Evaluate a candidate on the paired validation datasets."""
    chips = tuple(chips or DEFAULT_PAIRED_GATE_CHIPS)
    by_chip = {}
    for chip in chips:
        try:
            static_presence_path, motion_path, _ = find_static_presence_motion_dataset(chip=chip, num_sc=64)
        except FileNotFoundError:
            continue
        static_presence_packets = load_npz_as_packets(static_presence_path)[300:]
        motion_packets = load_npz_as_packets(motion_path)
        by_chip[chip] = evaluate_split(
            model,
            scaler,
            feature_names,
            static_presence_packets,
            motion_packets,
            threshold=threshold,
        )
    return summarize_gate(by_chip)


def evaluate_long_gate(model, scaler, feature_names, threshold=0.5, chips=None):
    """Evaluate a candidate on the curated long recordings."""
    from conftest import get_available_long_test_datasets

    chips = tuple(chips or DEFAULT_LONG_GATE_CHIPS)
    by_chip = {}
    for _, static_presence_packets, motion_packets, _, chip, _ in get_available_long_test_datasets(
        chips=chips
    ):
        by_chip[chip] = evaluate_split(
            model,
            scaler,
            feature_names,
            static_presence_packets,
            motion_packets,
            threshold=threshold,
        )
    return summarize_gate(by_chip)


def _parse_long_recording_metrics(output):
    """Parse the stable long-recording ML summary table emitted by pytest."""
    pattern = re.compile(
        r"\|\s*([A-Za-z0-9_-]+)\s*\|"
        r"\s*([0-9.]+)%\s*\|"
        r"\s*([0-9.]+)%\s*\|"
        r"\s*([0-9.]+)%\s*\|"
        r"\s*([0-9.]+)%\s*\|"
        r"\s*(\d+)\s*\|"
    )
    rows = []
    for match in pattern.finditer(output):
        chip, recall, precision, fp_rate, f1, fp_count = match.groups()
        rows.append({
            'chip': chip,
            'recall': float(recall),
            'precision': float(precision),
            'fp_rate': float(fp_rate),
            'f1': float(f1),
            'fp_count': int(fp_count),
        })

    if not rows:
        return None

    return {
        'rows': rows,
        'by_chip': {r['chip']: r for r in rows},
        'pass_count': int(sum(1 for r in rows if r['recall'] > 95.0 and r['fp_rate'] < 5.0)),
        'mean_recall': float(np.mean([r['recall'] for r in rows])),
        'worst_chip_recall': float(np.min([r['recall'] for r in rows])),
        'mean_fp_rate': float(np.mean([r['fp_rate'] for r in rows])),
        'max_fp_rate': float(np.max([r['fp_rate'] for r in rows])),
        'mean_f1': float(np.mean([r['f1'] for r in rows])),
        'worst_chip_f1': float(np.min([r['f1'] for r in rows])),
        'total_fp': int(sum(r['fp_count'] for r in rows)),
    }


@dataclass
class ExportedMLGateResult:
    """Combined verification result for exported ML artifacts."""

    paired_returncode: int
    paired_output: str
    long_metrics: dict | None
    long_output: str

    @property
    def available(self):
        return self.long_metrics is not None

    @property
    def passed(self):
        return self.paired_returncode == 0 and self.available


def _run_ml_performance_tests():
    """
    Run the long-recording ML gate and parse per-chip metrics.

    Returns:
        tuple: (metrics_dict_or_none, raw_output)
    """
    project_root = repo_root()
    cmd = [
        sys.executable,
        '-m',
        'pytest',
        PYTEST_LONG_ML_GATE,
        '-v',
        '-s',
    ]
    result = subprocess.run(
        cmd,
        cwd=project_root,
        capture_output=True,
        text=True,
    )
    output = (result.stdout or "") + "\n" + (result.stderr or "")
    return _parse_long_recording_metrics(output), output


def _run_ml_paired_tests():
    """Run the paired ML regression suite and return (returncode, output)."""
    project_root = repo_root()
    cmd = [
        sys.executable,
        '-m',
        'pytest',
        PYTEST_PAIRED_ML_GATE,
        '-v',
        '-s',
    ]
    result = subprocess.run(
        cmd,
        cwd=project_root,
        capture_output=True,
        text=True,
    )
    output = (result.stdout or "") + "\n" + (result.stderr or "")
    return int(result.returncode), output


def run_exported_ml_gates():
    """Run all required gates for already-exported ML artifacts."""
    paired_rc, paired_output = _run_ml_paired_tests()
    long_metrics, long_output = _run_ml_performance_tests()
    return ExportedMLGateResult(
        paired_returncode=paired_rc,
        paired_output=paired_output,
        long_metrics=long_metrics,
        long_output=long_output,
    )


def _real_ml_gate_key(real_metrics):
    """Ranking key for ML-only real-data gate results."""
    if real_metrics is None:
        return None
    return (
        real_metrics['mean_f1'],
        real_metrics['worst_chip_f1'],
        -real_metrics['total_fp'],
        real_metrics['mean_recall'],
        real_metrics['pass_count'],
        -real_metrics['max_fp_rate'],
    )


def _combined_candidate_key(cv_metrics, real_metrics=None):
    """
    Final selection key.

    Stage 1: CV robust metrics filter obvious weak candidates.
    Stage 2: ML-only real-data gate decides final promotion using deploy-like data.
    """
    cv_key = build_candidate_key(cv_metrics)
    real_key = _real_ml_gate_key(real_metrics)
    if real_key is None:
        return cv_key
    return real_key + cv_key


def _format_long_ml_summary(real_metrics):
    """Build a short one-line summary for ML-only real-data gate metrics."""
    if real_metrics is None:
        return "long_gate=unavailable"
    total = len(real_metrics.get('rows', []))
    return (
        f"long_gate={total} "
        f"mean_f1={real_metrics['mean_f1']:.1f}% "
        f"worst_f1={real_metrics['worst_chip_f1']:.1f}% "
        f"total_fp={real_metrics['total_fp']} "
        f"mean_recall={real_metrics['mean_recall']:.1f}%"
    )


def _format_exported_gate_summary(gate):
    """Build a short one-line summary for exported-artifact verification."""
    if gate is None:
        return "exported_gates=not_run"
    paired = "paired=pass" if gate.paired_returncode == 0 else f"paired=fail({gate.paired_returncode})"
    return f"{paired} {_format_long_ml_summary(gate.long_metrics)}"


def _candidate_beats_baseline(candidate_cv, candidate_gate, static_presence_cv, static_presence_gate):
    """Compare candidate vs baseline only when exported gates are available."""
    if candidate_gate is None or static_presence_gate is None:
        return False
    if not candidate_gate.passed or not static_presence_gate.passed:
        return False
    candidate_long = candidate_gate.long_metrics or {}
    baseline_long = static_presence_gate.long_metrics or {}
    if (
        candidate_long.get('total_fp', 0)
        > baseline_long.get('total_fp', 0) + MAX_PROMOTION_TOTAL_FP_INCREASE
    ):
        return False
    if (
        candidate_long.get('max_fp_rate', 0.0)
        > baseline_long.get('max_fp_rate', 0.0) + MAX_PROMOTION_FP_RATE_INCREASE
    ):
        return False
    return _combined_candidate_key(candidate_cv, candidate_gate.long_metrics) > _combined_candidate_key(
        static_presence_cv,
        static_presence_gate.long_metrics,
    )


def _model_artifact_paths():
    """Return paths of generated model artifacts."""
    return [
        SRC_DIR / 'ml_weights.py',
        CPP_DIR / 'ml_weights.h',
        GENERATED_DATA_DIR / 'ml_test_data.npz',
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


def train_until_improvement(max_trials, fp_weight=DEFAULT_FP_WEIGHT, feature_names=None,
                            hidden_layers=None, scaler_mode=DEFAULT_SCALER_MODE,
                            batch_size=DEFAULT_BATCH_SIZE, environment_filter=None,
                            excluded_chips=None, positive_chip_boost=None,
                            sample_weight_mode=DEFAULT_SAMPLE_WEIGHT_MODE,
                            use_cache=True):
    """
    Train repeatedly with auto-generated seeds until the promoted candidate improves.

    Baseline is recomputed using the seed embedded in the current exported model.
    Promotion uses a two-stage decision:
      1) grouped CV prefilter
      2) ML-only real-data gate on exported artifacts
    """
    if max_trials < 1:
        print("Error: --seed-search-until-improvement must be >= 1")
        return 1

    if feature_names is None:
        feature_names = DEFAULT_FEATURES
    if hidden_layers is None:
        hidden_layers = list(DEFAULT_HIDDEN_LAYERS)
    excluded_chips = parse_chip_filter(excluded_chips)
    positive_chip_boost = parse_positive_chip_boost(positive_chip_boost)
    sample_weight_mode = normalize_sample_weight_mode(sample_weight_mode)
    try:
        ensure_torch_available()
        torch_device_label = describe_torch_device()
    except ImportError as exc:
        print(f"Error: Missing dependency - {exc}")
        return 1
    except (RuntimeError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    print("\n" + "=" * 70)
    print("  SEED SEARCH (loop until improvement)")
    print("=" * 70)
    print(f"Max trials: {max_trials}")
    print(f"FP weight: {fp_weight}")
    print(f"Scaler: {scaler_mode}")
    print(f"Batch size: {batch_size}")
    print(f"Sample weight mode: {sample_weight_mode}")
    print(f"Torch device: {torch_device_label}")
    if environment_filter is not None:
        print(f"Environment filter: {', '.join(sorted(parse_environment_filter(environment_filter)))}")
    if excluded_chips is not None:
        print(f"Excluded chips: {', '.join(sorted(excluded_chips))}")
    if positive_chip_boost is not None:
        print(
            "Positive chip boost: "
            + ', '.join(f"{chip}={factor:.2f}" for chip, factor in sorted(positive_chip_boost.items()))
        )

    static_presence_seed = read_exported_seed()
    if static_presence_seed is None:
        static_presence_seed = 42
        print("\nWarning: current exported seed not found, using 42 as baseline seed")

    print(f"\nEvaluating current model baseline with seed {static_presence_seed}...")
    static_presence_rc, _, static_presence_metrics = train_all(
        fp_weight=fp_weight,
        seed=static_presence_seed,
        feature_names=feature_names,
        feature_importance=False,
        ablation=False,
        shap_samples=200,
        hidden_layers=hidden_layers,
        scaler_mode=scaler_mode,
        batch_size=batch_size,
        export_artifacts=False,
        environment_filter=environment_filter,
        excluded_chips=excluded_chips,
        positive_chip_boost=positive_chip_boost,
        sample_weight_mode=sample_weight_mode,
        use_cache=use_cache,
    )
    if static_presence_rc != 0 or static_presence_metrics is None:
        print("Error: unable to evaluate current model baseline")
        return 1

    static_presence_session = static_presence_metrics.get('group_reports', {}).get('session_group', {}).get('worst_recall', {})
    static_presence_chip = static_presence_metrics.get('group_reports', {}).get('chip', {}).get('worst_recall', {})
    print(
        f"Baseline: session_min_recall={static_presence_session.get('recall', 0.0):.1f}% "
        f"chip_min_recall={static_presence_chip.get('recall', 0.0):.1f}% "
        f"blocked_oof_f1={static_presence_metrics['oof_f1']:.1f}%"
    )
    static_presence_gate = run_exported_ml_gates()
    print(f"Baseline exported ML gates: {_format_exported_gate_summary(static_presence_gate)}")
    if not static_presence_gate.passed:
        print("Error: baseline exported ML gates are required for seed-search promotion")
        if static_presence_gate.paired_returncode != 0 and static_presence_gate.paired_output.strip():
            print(static_presence_gate.paired_output.strip())
        if static_presence_gate.long_metrics is None and static_presence_gate.long_output.strip():
            print(static_presence_gate.long_output.strip())
        return 1

    backup_dir, saved_files = _backup_artifacts()
    print(f"Artifacts backup: {backup_dir}")

    trial_summaries = []
    improved = False
    improved_seed = None
    improved_metrics = None
    improved_gate = None

    for idx in range(1, max_trials + 1):
        print(f"\n[{idx}/{max_trials}] Training with auto-generated seed")
        train_rc, used_seed, metrics = train_all(
            fp_weight=fp_weight,
            seed=None,
            feature_names=feature_names,
            feature_importance=False,
            ablation=False,
            shap_samples=200,
            hidden_layers=hidden_layers,
            scaler_mode=scaler_mode,
            batch_size=batch_size,
            export_artifacts=False,
            environment_filter=environment_filter,
            excluded_chips=excluded_chips,
            positive_chip_boost=positive_chip_boost,
            sample_weight_mode=sample_weight_mode,
            use_cache=use_cache,
        )
        if train_rc != 0 or metrics is None:
            print(f"  Training failed (exit={train_rc})")
            continue

        session_summary = metrics.get('group_reports', {}).get('session_group', {}).get('worst_recall', {})
        fp_summary = metrics.get('group_reports', {}).get('session_group', {}).get('worst_fp_rate', {})
        print(
            f"  Result: session_min_recall={session_summary.get('recall', 0.0):.1f}% "
            f"session_max_fp={fp_summary.get('fp_rate', 0.0):.1f}% "
            f"blocked_oof_f1={metrics['oof_f1']:.1f}%"
        )

        if build_candidate_key(metrics) <= build_candidate_key(static_presence_metrics):
            trial_summaries.append((used_seed, metrics, None, 'cv_rejected'))
            print("  CV filter: rejected before real-data ML gate")
            continue

        print("  CV filter: passed, exporting candidate for real-data ML gate...")
        export_rc, _, final_metrics = train_all(
            fp_weight=fp_weight,
            seed=used_seed,
            feature_names=feature_names,
            feature_importance=False,
            ablation=False,
            shap_samples=200,
            hidden_layers=hidden_layers,
            scaler_mode=scaler_mode,
            batch_size=batch_size,
            export_artifacts=True,
            environment_filter=environment_filter,
            excluded_chips=excluded_chips,
            positive_chip_boost=positive_chip_boost,
            sample_weight_mode=sample_weight_mode,
            use_cache=use_cache,
        )
        if export_rc != 0 or final_metrics is None:
            print("  Candidate export failed, restoring previous artifacts")
            _restore_artifacts(saved_files)
            trial_summaries.append((used_seed, metrics, None, 'export_failed'))
            continue

        candidate_gate = run_exported_ml_gates()
        print(f"  Exported ML gates: {_format_exported_gate_summary(candidate_gate)}")
        if not candidate_gate.passed:
            if candidate_gate.paired_returncode != 0 and candidate_gate.paired_output.strip():
                print(candidate_gate.paired_output.strip())
            if candidate_gate.long_metrics is None and candidate_gate.long_output.strip():
                print(candidate_gate.long_output.strip())

        trial_summaries.append((used_seed, final_metrics, candidate_gate, 'exported_gates'))
        if _candidate_beats_baseline(final_metrics, candidate_gate, static_presence_metrics, static_presence_gate):
            improved = True
            improved_seed = used_seed
            improved_metrics = final_metrics
            improved_gate = candidate_gate
            print("  Improvement found after real-data ML gate: stopping search")
            break

        print("  Real-data ML gate rejected candidate, restoring previous artifacts")
        _restore_artifacts(saved_files)

    print("\n" + "=" * 70)
    print("  UNTIL-IMPROVEMENT SUMMARY")
    print("=" * 70)
    for seed, metrics, gate, status in trial_summaries:
        session_summary = metrics.get('group_reports', {}).get('session_group', {}).get('worst_recall', {})
        fp_summary = metrics.get('group_reports', {}).get('session_group', {}).get('worst_fp_rate', {})
        print(
            f"  seed={seed} | sessionMinR={session_summary.get('recall', 0.0):.1f}% "
            f"sessionMaxFP={fp_summary.get('fp_rate', 0.0):.1f}% "
            f"blockedOOF={metrics['oof_f1']:.1f}% | {status} | "
            f"{_format_exported_gate_summary(gate)}"
        )

    if improved:
        print(
            f"\nSelected seed: {improved_seed} "
            f"(blocked_oof_f1={improved_metrics['oof_f1']:.1f}%, "
            f"{_format_exported_gate_summary(improved_gate)})"
        )
        return 0

    print("\nNo improvement found within max trials; current artifacts remain unchanged")
    return 1


def write_json_results(path, payload):
    """Write a JSON experiment payload."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def slim_cv_result(cv_results):
    """Keep only the CV fields needed by experiment payloads."""
    session_report = cv_results.get('group_reports', {}).get('session_group', {})
    chip_report = cv_results.get('group_reports', {}).get('chip', {})
    return {
        'f1_mean': float(cv_results['f1_mean']),
        'f1_std': float(cv_results['f1_std']),
        'oof_f1': float(cv_results['oof_f1']),
        'recall_mean': float(cv_results['recall_mean']),
        'fp_rate_mean': float(cv_results['fp_rate_mean']),
        'worst_session_recall': float(session_report.get('worst_recall', {}).get('recall', 0.0)),
        'worst_session_fp_rate': float(session_report.get('worst_fp_rate', {}).get('fp_rate', 0.0)),
        'worst_chip_recall': float(chip_report.get('worst_recall', {}).get('recall', 0.0)),
        'candidate_key': list(build_candidate_key(cv_results)),
    }


def architecture_stats(input_dim, hidden_layers):
    """Return parameter, size, and FLOP estimates for an MLP."""
    layer_sizes = [input_dim] + list(hidden_layers) + [1]
    n_params = 0
    flops = 0
    for idx in range(len(layer_sizes) - 1):
        n_params += layer_sizes[idx] * layer_sizes[idx + 1]
        n_params += layer_sizes[idx + 1]
        flops += layer_sizes[idx] * layer_sizes[idx + 1]
    return {
        'layer_sizes': layer_sizes,
        'params': int(n_params),
        'weight_kb': float(n_params * 4 / 1024),
        'flops': int(flops),
    }


def architecture_campaign_rank_key(result):
    """Sort key for single-run architecture candidates (lower is better)."""
    return (
        result['long']['max_fp_rate'],
        result['long']['total_fp'],
        -result['long']['pass_count'],
        -result['long']['worst_chip_f1'],
        -result['paired']['pass_count'],
        result['paired']['max_fp_rate'],
        -result['paired']['worst_chip_f1'],
        -result['cv']['oof_f1'],
        -result['cv']['f1_mean'],
        result['params'],
    )


def aggregate_architecture_runs(name, runs):
    """Aggregate multi-seed runs for one architecture."""
    template = runs[0]
    return {
        'name': name,
        'layers': list(template['layers']),
        'architecture': template['architecture'],
        'params': int(template['params']),
        'weight_kb': float(template['weight_kb']),
        'flops': int(template['flops']),
        'seeds': [int(run['seed']) for run in runs],
        'median_long_max_fp_rate': float(np.median([run['long']['max_fp_rate'] for run in runs])),
        'median_long_total_fp': float(np.median([run['long']['total_fp'] for run in runs])),
        'median_long_pass_count': float(np.median([run['long']['pass_count'] for run in runs])),
        'median_long_worst_chip_f1': float(np.median([run['long']['worst_chip_f1'] for run in runs])),
        'worst_long_max_fp_rate': float(np.max([run['long']['max_fp_rate'] for run in runs])),
        'median_paired_pass_count': float(np.median([run['paired']['pass_count'] for run in runs])),
        'median_paired_max_fp_rate': float(np.median([run['paired']['max_fp_rate'] for run in runs])),
        'median_paired_worst_chip_f1': float(np.median([run['paired']['worst_chip_f1'] for run in runs])),
        'median_oof_f1': float(np.median([run['cv']['oof_f1'] for run in runs])),
        'best_single_run': min(runs, key=architecture_campaign_rank_key),
        'runs': runs,
    }


def aggregate_architecture_rank_key(summary):
    """Sort key for aggregated architecture summaries (lower is better)."""
    return (
        summary['median_long_max_fp_rate'],
        summary['median_long_total_fp'],
        -summary['median_long_pass_count'],
        -summary['median_long_worst_chip_f1'],
        summary['worst_long_max_fp_rate'],
        -summary['median_paired_pass_count'],
        summary['median_paired_max_fp_rate'],
        -summary['median_paired_worst_chip_f1'],
        -summary['median_oof_f1'],
        summary['params'],
    )


def paired_non_regression(candidate, baseline):
    """Treat paired validation as a non-regression constraint."""
    return (
        candidate['median_paired_pass_count'] >= baseline['median_paired_pass_count']
        and candidate['median_paired_max_fp_rate'] <= baseline['median_paired_max_fp_rate'] + 1e-6
        and candidate['median_paired_worst_chip_f1'] >= baseline['median_paired_worst_chip_f1'] - 1.0
    )


def architecture_candidate_beats_baseline(candidate, baseline):
    """Promote only stable FP-first improvements that do not regress paired validation."""
    if candidate['name'] == baseline['name']:
        return True
    if not paired_non_regression(candidate, baseline):
        return False
    candidate_long = (
        candidate['median_long_max_fp_rate'],
        candidate['median_long_total_fp'],
        -candidate['median_long_pass_count'],
        -candidate['median_long_worst_chip_f1'],
        candidate['worst_long_max_fp_rate'],
    )
    static_presence_long = (
        baseline['median_long_max_fp_rate'],
        baseline['median_long_total_fp'],
        -baseline['median_long_pass_count'],
        -baseline['median_long_worst_chip_f1'],
        baseline['worst_long_max_fp_rate'],
    )
    if candidate_long >= static_presence_long:
        return False
    return aggregate_architecture_rank_key(candidate) < aggregate_architecture_rank_key(baseline)


def evaluate_architecture_candidate(name, hidden_layers, seed, dataset, scaler_mode, batch_size, fp_weight):
    """Train and evaluate one architecture on CV, paired gate, and long gate."""
    stats = architecture_stats(dataset['X'].shape[1], hidden_layers)
    print(f"\n== {name} | seed {seed} ==")
    print(
        f"Architecture: {' -> '.join(map(str, stats['layer_sizes']))} | "
        f"params={stats['params']} | weights={stats['weight_kb']:.1f} KB | flops={stats['flops']}"
    )

    with suppress_stderr():
        cv = cross_validate(
            dataset['X'],
            dataset['y'],
            hidden_layers=list(hidden_layers),
            n_folds=DEFAULT_CV_FOLDS,
            max_epochs=DEFAULT_MAX_EPOCHS,
            fp_weight=fp_weight,
            sample_weight=dataset['sample_weights'],
            groups=dataset['groups'],
            sample_context=dataset['sample_context'],
            scaler_mode=scaler_mode,
            batch_size=batch_size,
            block_stride=SEG_WINDOW_SIZE,
            block_group_key=DEFAULT_BLOCK_GROUP_KEY,
            report_group_keys=DEFAULT_REPORT_GROUP_KEYS,
            seed=seed,
        )

    scaler = build_preprocessor(scaler_mode)
    X_scaled = scaler.fit_transform(dataset['X'])
    with suppress_stderr():
        model = train_model(
            X_scaled,
            dataset['y'],
            hidden_layers=list(hidden_layers),
            max_epochs=DEFAULT_MAX_EPOCHS,
            fp_weight=fp_weight,
            sample_weight=dataset['sample_weights'],
            batch_size=batch_size,
            seed=derive_seed(seed, 10_000),
        )

    sample = X_scaled[:1].astype(np.float32)
    for _ in range(10):
        predict_probabilities(model, sample)
    n_bench = 1000
    bench_start = perf_counter()
    for _ in range(n_bench):
        predict_probabilities(model, sample)
    inference_us = (perf_counter() - bench_start) / n_bench * 1e6

    paired = evaluate_paired_gate(model, scaler, dataset['feature_names'])
    long_gate = evaluate_long_gate(model, scaler, dataset['feature_names'])
    result = {
        'name': name,
        'seed': int(seed),
        'layers': list(hidden_layers),
        'architecture': ' -> '.join(map(str, stats['layer_sizes'])),
        'params': int(stats['params']),
        'weight_kb': float(stats['weight_kb']),
        'flops': int(stats['flops']),
        'inference_us': float(inference_us),
        'cv': slim_cv_result(cv),
        'paired': paired,
        'long': long_gate,
    }
    print(
        f"{name} | OOF={result['cv']['oof_f1']:.1f}% | "
        f"paired pass={paired['pass_count']} maxFP={paired['max_fp_rate']:.1f}% | "
        f"long pass={long_gate['pass_count']} maxFP={long_gate['max_fp_rate']:.1f}% "
        f"totalFP={long_gate['total_fp']} worstF1={long_gate['worst_chip_f1']:.1f}% | "
        f"inf={inference_us:.1f} us"
    )
    return result


def experiment_architectures(scaler_mode=DEFAULT_SCALER_MODE,
                             batch_size=DEFAULT_BATCH_SIZE,
                             fp_weight=DEFAULT_FP_WEIGHT,
                             environment_filter=None,
                             excluded_chips=None,
                             architectures=None,
                             positive_chip_boost=None,
                             sample_weight_mode=DEFAULT_SAMPLE_WEIGHT_MODE,
                             output_path=DEFAULT_EXPERIMENT_OUTPUT,
                             promote_winner=False,
                             use_cache=True):
    """Run the FP-first architecture campaign and optionally promote a winner."""
    environment_filter = parse_environment_filter(environment_filter)
    excluded_chips = parse_chip_filter(excluded_chips)
    positive_chip_boost = parse_positive_chip_boost(positive_chip_boost)
    sample_weight_mode = normalize_sample_weight_mode(sample_weight_mode)
    architectures = normalize_architecture_specs(architectures or DEFAULT_ARCHITECTURE_SWEEP)

    static_presence_layers = tuple(DEFAULT_HIDDEN_LAYERS)
    if static_presence_layers not in {tuple(spec['layers']) for spec in architectures}:
        architectures.insert(0, {
            'name': f"Current default ({format_hidden_layers(DEFAULT_HIDDEN_LAYERS)})",
            'layers': list(DEFAULT_HIDDEN_LAYERS),
        })
    else:
        architectures = sorted(
            architectures,
            key=lambda spec: tuple(spec['layers']) != static_presence_layers,
        )
    static_presence_name = next(
        spec['name'] for spec in architectures if tuple(spec['layers']) == static_presence_layers
    )
    screening_seed = read_exported_seed() or DEFAULT_EXPERIMENT_SCREENING_SEED

    try:
        ensure_torch_available()
        torch_device_label = describe_torch_device()
    except ImportError as exc:
        print(f"Error: Missing dependency - {exc}")
        return 1
    except (RuntimeError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    print("\n" + "=" * 70)
    print("  FP-FIRST MLP ARCHITECTURE CAMPAIGN")
    print("=" * 70)
    print(f"Scaler: {scaler_mode}")
    print(f"Batch size: {batch_size}")
    print(f"Torch device: {torch_device_label}")
    print(f"FP weight: {fp_weight}")
    print(f"Sample weight mode: {sample_weight_mode}")
    print(f"Screening seed: {screening_seed}")
    print(
        "Architectures: "
        + ', '.join(f"{spec['name']} [{format_hidden_layers(spec['layers'])}]" for spec in architectures)
    )
    if environment_filter is not None:
        print(f"Environment filter: {', '.join(sorted(environment_filter))}")
    if excluded_chips is not None:
        print(f"Excluded chips: {', '.join(sorted(excluded_chips))}")
    if positive_chip_boost is not None:
        print(
            "Positive chip boost: "
            + ', '.join(f"{chip}={factor:.2f}" for chip, factor in sorted(positive_chip_boost.items()))
        )

    print("\nLoading training matrix...")
    matrix, _ = load_training_matrix(
        environment_filter=environment_filter,
        excluded_chips=excluded_chips,
        feature_names=TRAINING_FEATURES,
        sample_weight_mode=sample_weight_mode,
        use_cache=use_cache,
    )
    stats = matrix['stats']
    if not stats['chips']:
        print("Error: No datasets found in data/")
        return 1

    print(f"  Chips: {', '.join(stats['chips'])}")
    print(f"  Session groups: {len(stats.get('session_groups', []))}")
    print(f"  Total: {stats['total']} packets")

    X = matrix['X']
    y = matrix['y']
    feature_names = matrix['feature_names']
    sample_context = matrix['sample_context']
    sample_weights = matrix['sample_weights']
    sample_weights, boost_summary = apply_positive_chip_boost(
        sample_weights,
        sample_context,
        y,
        positive_chip_boost,
    )
    dataset = {
        'X': np.asarray(X, dtype=np.float32),
        'y': np.asarray(y, dtype=np.int8),
        'feature_names': list(feature_names),
        'sample_context': sample_context,
        'sample_weights': np.asarray(sample_weights, dtype=np.float32),
        'groups': sample_context[DEFAULT_PRIMARY_GROUP_KEY],
        'boost_summary': boost_summary,
    }
    print(f"  Samples: {len(dataset['X'])}")
    print(f"  Features: {len(dataset['feature_names'])}")
    print(f"  Class balance: IDLE={np.sum(dataset['y']==0)}, MOTION={np.sum(dataset['y']==1)}")

    results = {
        'config': {
            'scaler': scaler_mode,
            'batch_size': batch_size,
            'fp_weight': fp_weight,
            'environment': sorted(environment_filter) if environment_filter else None,
            'exclude_chip': sorted(excluded_chips) if excluded_chips else [],
            'positive_chip_boost': positive_chip_boost,
            'sample_weight_mode': sample_weight_mode,
            'screening_seed': screening_seed,
            'initial_seeds': list(DEFAULT_EXPERIMENT_INITIAL_SEEDS),
            'final_seeds': list(DEFAULT_EXPERIMENT_FINAL_SEEDS),
            'promote_winner': bool(promote_winner),
            'architectures': architectures,
            'feature_names': list(feature_names),
        },
        'screening': [],
        'seed_filter': [],
        'seed_finalists': [],
        'promotion': None,
    }

    print("\n== Single-seed screening ==")
    screening_results = []
    for spec in architectures:
        run = evaluate_architecture_candidate(
            spec['name'],
            spec['layers'],
            screening_seed,
            dataset,
            scaler_mode,
            batch_size,
            fp_weight,
        )
        screening_results.append(run)
        results['screening'] = screening_results
        write_json_results(output_path, results)

    challengers = [
        item for item in sorted(screening_results, key=architecture_campaign_rank_key)
        if item['name'] != static_presence_name
    ][:2]
    finalists = [static_presence_name] + [item['name'] for item in challengers]
    print(f"\nFinalists for 3-seed filter: {', '.join(finalists)}")

    specs_by_name = {spec['name']: spec for spec in architectures}

    print("\n== 3-seed robustness filter ==")
    seed_filter = []
    for name in finalists:
        spec = specs_by_name[name]
        runs = [
            evaluate_architecture_candidate(
                name,
                spec['layers'],
                seed,
                dataset,
                scaler_mode,
                batch_size,
                fp_weight,
            )
            for seed in DEFAULT_EXPERIMENT_INITIAL_SEEDS
        ]
        summary = aggregate_architecture_runs(name, runs)
        seed_filter.append(summary)
        results['seed_filter'] = seed_filter
        write_json_results(output_path, results)
        print(
            f"{name} | median long maxFP={summary['median_long_max_fp_rate']:.1f}% | "
            f"median totalFP={summary['median_long_total_fp']:.1f} | "
            f"median worstF1={summary['median_long_worst_chip_f1']:.1f}% | "
            f"median paired pass={summary['median_paired_pass_count']:.1f}"
        )

    static_presence_filter = next(item for item in seed_filter if item['name'] == static_presence_name)
    challenger_summaries = [
        item for item in sorted(seed_filter, key=aggregate_architecture_rank_key)
        if item['name'] != static_presence_name
    ]
    head_to_head = [static_presence_name]
    if challenger_summaries:
        head_to_head.append(challenger_summaries[0]['name'])
    print(f"\n5-seed head-to-head: {', '.join(head_to_head)}")

    print("\n== 5-seed final comparison ==")
    seed_finalists = []
    for name in head_to_head:
        spec = specs_by_name[name]
        runs = [
            evaluate_architecture_candidate(
                name,
                spec['layers'],
                seed,
                dataset,
                scaler_mode,
                batch_size,
                fp_weight,
            )
            for seed in DEFAULT_EXPERIMENT_FINAL_SEEDS
        ]
        summary = aggregate_architecture_runs(name, runs)
        seed_finalists.append(summary)
        results['seed_finalists'] = seed_finalists
        write_json_results(output_path, results)
        print(
            f"{name} | median long maxFP={summary['median_long_max_fp_rate']:.1f}% | "
            f"median totalFP={summary['median_long_total_fp']:.1f} | "
            f"median worstF1={summary['median_long_worst_chip_f1']:.1f}% | "
            f"median paired pass={summary['median_paired_pass_count']:.1f}"
        )

    seed_finalists = sorted(seed_finalists, key=aggregate_architecture_rank_key)
    static_presence_final = next(item for item in seed_finalists if item['name'] == static_presence_name)
    winner = seed_finalists[0]
    promote_candidate = (
        winner['name'] != static_presence_name
        and architecture_candidate_beats_baseline(winner, static_presence_final)
    )
    results['promotion'] = {
        'winner': winner['name'],
        'static_presence': static_presence_final['name'],
        'decision': f"promote {winner['name']}" if promote_candidate else f"keep {static_presence_name}",
        'clear_winner': bool(promote_candidate),
        'summary': winner,
        'static_presence_summary': static_presence_final,
        'output_path': str(output_path),
    }
    write_json_results(output_path, results)

    if not promote_candidate:
        print(f"\nDecision: keep {static_presence_name}")
        return 0

    print(f"\nDecision: {winner['name']} beats {static_presence_name} on FP-first ranking")
    if not promote_winner:
        print("Promotion disabled (--experiment-promote not set), leaving current artifacts unchanged")
        return 0

    print("\n== Exporting promoted architecture ==")
    backup_dir, saved_files = _backup_artifacts()
    spec = specs_by_name[winner['name']]
    export_rc, used_seed, export_metrics = train_all(
        fp_weight=fp_weight,
        seed=winner['best_single_run']['seed'],
        feature_names=TRAINING_FEATURES,
        feature_importance=False,
        ablation=False,
        shap_samples=200,
        hidden_layers=spec['layers'],
        scaler_mode=scaler_mode,
        batch_size=batch_size,
        export_artifacts=True,
        environment_filter=environment_filter,
        excluded_chips=excluded_chips,
        positive_chip_boost=positive_chip_boost,
        sample_weight_mode=sample_weight_mode,
        use_cache=use_cache,
    )
    if export_rc != 0 or export_metrics is None:
        _restore_artifacts(saved_files)
        results['promotion']['final_export'] = {
            'status': 'export_failed',
            'backup_dir': str(backup_dir),
        }
        write_json_results(output_path, results)
        print("Promotion export failed, restored previous artifacts")
        return 1

    final_gate = run_exported_ml_gates()
    if not final_gate.passed:
        _restore_artifacts(saved_files)
        results['promotion']['final_export'] = {
            'status': 'verification_failed',
            'seed': int(used_seed),
            'paired_returncode': int(final_gate.paired_returncode),
            'long_metrics': final_gate.long_metrics,
            'backup_dir': str(backup_dir),
        }
        write_json_results(output_path, results)
        print("Promotion verification failed, restored previous artifacts")
        if final_gate.paired_output.strip():
            print(final_gate.paired_output.strip())
        if final_gate.long_output.strip():
            print(final_gate.long_output.strip())
        return 1

    results['promotion']['final_export'] = {
        'status': 'promoted',
        'seed': int(used_seed),
        'paired_returncode': int(final_gate.paired_returncode),
        'long_metrics': final_gate.long_metrics,
        'backup_dir': str(backup_dir),
        'paired_output': final_gate.paired_output,
        'long_output': final_gate.long_output,
    }
    write_json_results(output_path, results)
    print(f"Promoted architecture: {winner['name']} (seed {used_seed})")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Train ML motion detection model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python tools/10_train_ml_model.py                    # Train with current production defaults
  python tools/10_train_ml_model.py --info             # Show dataset info
  python tools/10_train_ml_model.py --experiment       # Run the FP-first MLP topology campaign
  python tools/10_train_ml_model.py --experiment --experiment-promote
                                           # Promote the winner if it beats the baseline
  python tools/10_train_ml_model.py --experiment --experiment-architectures "16,8;24,12;32,16;24;24,12,6"
                                           # Custom shortlist for the topology campaign
  python tools/10_train_ml_model.py --hidden-layers 24,12
                                           # Train/export the previous 8 -> 24 -> 12 -> 1 candidate
  python tools/10_train_ml_model.py --feature-set robust_relative --ablation
                                           # Evaluate spike-robust percentile features without export
  python tools/10_train_ml_model.py --feature-set robust_relative --no-export
                                           # Run grouped CV for an experimental feature set
  python tools/10_train_ml_model.py --fp-weight 2.0    # Penalize FP 2x more
  python tools/10_train_ml_model.py --scaler clipped_standard
                                           # Robust clipping + z-score
  python tools/10_train_ml_model.py --batch-size 32     # Smaller batch size experiment
  python tools/10_train_ml_model.py --device cuda       # Force CUDA when available
  python tools/10_train_ml_model.py --device mps        # Force Apple GPU when available
  python tools/10_train_ml_model.py --no-cache          # Rebuild cached training matrix
  python tools/10_train_ml_model.py --seed 42          # Reproducible training
  python tools/10_train_ml_model.py --hidden-layers 24,12 --positive-chip-boost ESP32=1.2
                                           # Bias training slightly toward ESP32 motion recall
  python tools/10_train_ml_model.py --seed-search-until-improvement 20
                                           # Stop at first improvement (max 20 trials)
  python tools/10_train_ml_model.py --gain-stress-gate --environment bedroom
                                           # Diagnose exported model robustness to feature gain shifts
  python tools/10_train_ml_model.py --gain-feature-experiment --seed 721498330
                                           # Compare raw/relative/hybrid feature sets under gain stress
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
                       help='Run the FP-first MLP topology campaign')
    parser.add_argument('--experiment-promote', action='store_true',
                       help='When used with --experiment, export the winning topology if it beats the baseline')
    parser.add_argument('--experiment-output', type=Path, default=DEFAULT_EXPERIMENT_OUTPUT,
                       help='JSON output path for --experiment results '
                            f'(default: {DEFAULT_EXPERIMENT_OUTPUT})')
    parser.add_argument('--experiment-architectures', type=parse_architecture_sweep, default=None,
                       help='Semicolon-separated hidden-layer specs for --experiment, '
                            'e.g. "16,8;24,12;32,16;24;24,12,6"')
    parser.add_argument('--seed', type=int, default=None,
                       help='Use specific random seed for reproducible training')
    parser.add_argument('--seed-search-until-improvement', type=int, default=0, metavar='MAX_TRIALS',
                       help='Train with auto-generated seeds until first '
                            'improvement over current ML performance, with '
                            'at most MAX_TRIALS attempts')
    parser.add_argument('--gain-stress-gate', action='store_true',
                       help='Evaluate current exported ML artifacts under '
                            'artificial gain scaling without training/exporting')
    parser.add_argument('--gain-stress-scales', type=parse_gain_stress_scales,
                       default=DEFAULT_GAIN_STRESS_SCALES,
                       help='Comma-separated gain multipliers for --gain-stress-gate '
                            f'(default: {",".join(map(str, DEFAULT_GAIN_STRESS_SCALES))})')
    parser.add_argument('--gain-feature-experiment', action='store_true',
                       help='Compare raw, relative, and hybrid feature sets '
                            'with grouped CV and in-memory gain-stress evaluation')
    parser.add_argument('--gain-feature-sets', type=str,
                       default=','.join(GAIN_FEATURE_EXPERIMENT_SETS),
                       help='Comma-separated feature sets for --gain-feature-experiment '
                            f'(default: {",".join(GAIN_FEATURE_EXPERIMENT_SETS)})')
    parser.add_argument('--fp-weight', type=float, default=DEFAULT_FP_WEIGHT,
                       help='Multiplier for IDLE class weight to penalize false positives. '
                            f'Values >1.0 make the model more conservative (default: {DEFAULT_FP_WEIGHT:.1f})')
    parser.add_argument('--scaler', choices=['standard', 'robust', 'clipped_standard'],
                       default=DEFAULT_SCALER_MODE,
                       help='Feature normalization mode for training/evaluation')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
                       help='Mini-batch size for PyTorch training '
                            f'(default: {DEFAULT_BATCH_SIZE})')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'],
                       default=DEFAULT_TORCH_DEVICE,
                       help='PyTorch training device. CUDA and MPS are opt-in '
                            f'(default: {DEFAULT_TORCH_DEVICE})')
    parser.add_argument('--hidden-layers', type=parse_hidden_layers, default=None,
                       help='Comma-separated hidden layer widths for the MLP '
                            f'(default: {",".join(map(str, DEFAULT_HIDDEN_LAYERS))})')
    parser.add_argument('--feature-set', choices=sorted(FEATURE_SET_CHOICES), default='production',
                       help='Named binary-training feature set. Only production is export-compatible '
                            '(default: production)')
    parser.add_argument('--no-export', action='store_true',
                       help='Run training/CV analysis without exporting runtime artifacts')
    parser.add_argument('--no-cache', action='store_true',
                       help='Rebuild training features and sample weights instead of using the local cache')
    parser.add_argument('--environment', type=str, default=None,
                       help='Restrict training/evaluation to one or more named environments '
                            '(comma-separated, e.g. bedroom or bedroom,living_room)')
    parser.add_argument('--exclude-chip', type=str,
                       default=','.join(DEFAULT_EXCLUDED_CHIPS),
                       help='Exclude one or more chips from training/evaluation '
                            '(comma-separated, e.g. ESP32 or ESP32,S3; '
                            f'default: {",".join(DEFAULT_EXCLUDED_CHIPS)})')
    parser.add_argument('--positive-chip-boost', type=parse_positive_chip_boost, default=None,
                       help='Boost motion samples for specific chips, e.g. ESP32=1.2 or ESP32=1.2,S3=1.1')
    parser.add_argument('--sample-weight-mode', choices=SAMPLE_WEIGHT_MODES,
                       default=DEFAULT_SAMPLE_WEIGHT_MODE,
                       help='Base sample weighting policy. "none" keeps MVS out of training; '
                            '"mvs_global" uses the legacy global MVS score; '
                            '"mvs_gridsearch" uses per-file optimal_threshold_gridsearch; '
                            '"mvs_hard_negative" uses MVS only to promote hard idle windows '
                            f'(default: {DEFAULT_SAMPLE_WEIGHT_MODE})')
    parser.add_argument('--shap', type=int, nargs='?', const=200, default=None,
                       metavar='SAMPLES',
                       help='Calculate SHAP feature importance (default: 200 samples)')
    parser.add_argument('--correlation', action='store_true',
                       help='Calculate correlation of selected training features with motion label')
    parser.add_argument('--ablation', action='store_true',
                       help='Run ablation study (test removing each feature)')
    args = parser.parse_args()
    set_active_torch_device(args.device)
    selected_training_features = resolve_training_feature_set(args.feature_set)
    export_compatible_feature_set = args.feature_set == 'production'
    
    if args.info:
        show_info()
        return 0

    if args.gain_stress_gate:
        if args.experiment or args.experiment_promote:
            print("Error: --gain-stress-gate cannot be combined with experiment flows")
            return 1
        if args.seed_search_until_improvement > 0 or args.seed is not None:
            print("Error: --gain-stress-gate evaluates exported artifacts and cannot use --seed or seed-search")
            return 1
        if args.shap is not None or args.ablation or args.correlation:
            print("Error: --gain-stress-gate cannot be combined with --shap, --ablation, or --correlation")
            return 1
        if args.positive_chip_boost is not None:
            print("Error: --positive-chip-boost is a training option and cannot be used with --gain-stress-gate")
            return 1
        results = evaluate_gain_stress_gate(
            environment_filter=args.environment,
            excluded_chips=args.exclude_chip,
            scales=args.gain_stress_scales,
        )
        print_gain_stress_summary(results)
        return 0

    if args.gain_feature_experiment:
        if args.experiment or args.experiment_promote:
            print("Error: --gain-feature-experiment cannot be combined with experiment flows")
            return 1
        if args.seed_search_until_improvement > 0:
            print("Error: --gain-feature-experiment cannot be combined with seed-search")
            return 1
        if args.shap is not None or args.ablation or args.correlation:
            print("Error: --gain-feature-experiment cannot be combined with --shap, --ablation, or --correlation")
            return 1
        if args.positive_chip_boost is not None:
            print("Error: --positive-chip-boost is not supported by --gain-feature-experiment")
            return 1
        rc, _, _ = experiment_gain_feature_sets(
            seed=args.seed,
            feature_sets=args.gain_feature_sets.split(','),
            scaler_mode=args.scaler,
            batch_size=args.batch_size,
            fp_weight=args.fp_weight,
            hidden_layers=args.hidden_layers,
            environment_filter=args.environment,
            excluded_chips=args.exclude_chip,
            scales=args.gain_stress_scales,
            use_cache=not args.no_cache,
        )
        return rc

    if args.experiment:
        if not export_compatible_feature_set:
            print("Error: --experiment currently supports only --feature-set production")
            return 1
        return experiment_architectures(
            scaler_mode=args.scaler,
            batch_size=args.batch_size,
            fp_weight=args.fp_weight,
            environment_filter=args.environment,
            excluded_chips=args.exclude_chip,
            architectures=args.experiment_architectures,
            positive_chip_boost=args.positive_chip_boost,
            sample_weight_mode=args.sample_weight_mode,
            output_path=args.experiment_output,
            promote_winner=args.experiment_promote,
            use_cache=not args.no_cache,
        )
    
    if args.correlation:
        correlations = calculate_correlation_importance(
            feature_names=selected_training_features,
            use_cache=not args.no_cache,
        )
        if correlations:
            print_correlation_table(correlations, selected_training_features)
        return 0

    if args.seed_search_until_improvement > 0:
        if not export_compatible_feature_set:
            print("Error: seed search can export artifacts and currently supports only --feature-set production")
            return 1
        if args.seed is not None:
            print("Error: --seed and --seed-search-until-improvement are mutually exclusive")
            return 1
        if args.shap is not None or args.ablation:
            print("Error: --seed-search-until-improvement cannot be combined with --shap or --ablation")
            return 1
        return train_until_improvement(
            max_trials=args.seed_search_until_improvement,
            fp_weight=args.fp_weight,
            feature_names=selected_training_features,
            hidden_layers=args.hidden_layers,
            scaler_mode=args.scaler,
            batch_size=args.batch_size,
            environment_filter=args.environment,
            excluded_chips=args.exclude_chip,
            positive_chip_boost=args.positive_chip_boost,
            sample_weight_mode=args.sample_weight_mode,
            use_cache=not args.no_cache,
        )
    
    if (
        not export_compatible_feature_set
        and not args.no_export
        and not (args.ablation or args.correlation or args.shap is not None)
    ):
        print(
            "Error: non-production feature sets are analysis-only until the C++ runtime "
            "extractor is updated for export compatibility. Use --no-export, --ablation, "
            "--shap, or --correlation."
        )
        return 1

    train_rc, _, _ = train_all(
        fp_weight=args.fp_weight, 
        seed=args.seed,
        feature_names=selected_training_features,
        feature_importance=args.shap is not None,
        ablation=args.ablation,
        shap_samples=args.shap if args.shap is not None else 200,
        hidden_layers=args.hidden_layers,
        scaler_mode=args.scaler,
        batch_size=args.batch_size,
        environment_filter=args.environment,
        excluded_chips=args.exclude_chip,
        positive_chip_boost=args.positive_chip_boost,
        sample_weight_mode=args.sample_weight_mode,
        use_cache=not args.no_cache,
        export_artifacts=not args.no_export,
    )
    return train_rc


if __name__ == '__main__':
    exit(main())
