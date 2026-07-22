#!/usr/bin/env python3
"""
ESPectre - ML Training

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
    python tools/train_ml_model.py                    # Train and export if paired gate passes
    python tools/train_ml_model.py --no-export        # Evaluate without replacing runtime artifacts
    python tools/train_ml_model.py --info             # Show dataset info
    python tools/train_ml_model.py --experiment       # Run the FP-first MLP topology campaign
    python tools/train_ml_model.py --experiment --experiment-promote
                                                    # Promote the winner if it beats the baseline
    python tools/train_ml_model.py --fp-weight 2.0    # Penalize FP 2x more
    python tools/train_ml_model.py --scaler clipped_standard
                                                    # Robust clipping + z-score
    python tools/train_ml_model.py --batch-size 32
                                                    # Smaller batch size experiment
    python tools/train_ml_model.py --device cuda # Force CUDA when available
    python tools/train_ml_model.py --device mps  # Force Apple GPU when available
    python tools/train_ml_model.py --shap --no-export  # Grouped OOF SHAP (200 samples)
    python tools/train_ml_model.py --shap 500 --no-export
                                                    # Grouped OOF SHAP (500 samples)

Configuration:
  - TRAINING_FEATURES: Production Core-6 feature list

Note: turbulence normalization now follows the shared production path:
CV-normalized turbulence (`std/mean`) for every stream.

To compare ML with the moving-variance baseline, use:
    python tools/compare_detection_methods.py

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import os
import sys

import argparse
import copy
import hashlib
import importlib.util
import json
import numpy as np
import random
import re
import shutil
import tempfile
from pathlib import Path
from dataclasses import dataclass
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT_PATH = SCRIPT_DIR.parent
if str(REPO_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_PATH))

from tools.lib.bootstrap import setup_paths  # noqa: F401

from tools.lib.repo_paths import (
    cpp_core_dir,
    generated_data_dir,
    python_src_dir,
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


def generate_random_training_seed():
    """Return a fresh non-negative 31-bit training seed."""
    from numpy.random import SeedSequence
    return int(SeedSequence().entropy % (2**31))


def resolve_training_seed(seed=None, trailing_newline=False, prefer_exported=True):
    """
    Resolve and print the seed used for a training/evaluation run.

    Priority when ``seed`` is omitted:
      1. seed embedded in the current exported model (if prefer_exported)
      2. a freshly generated random seed
    """
    suffix = "\n" if trailing_newline else ""
    if seed is not None:
        seed = int(seed)
        print(f"Using provided seed: {seed}{suffix}")
        return seed

    if prefer_exported:
        exported = read_exported_seed()
        if exported is not None:
            print(f"Using exported model seed: {exported}{suffix}")
            return int(exported)

    seed = generate_random_training_seed()
    if prefer_exported:
        print(f"No exported model seed found; generated random seed: {seed}{suffix}")
    else:
        print(f"Generated random seed: {seed}{suffix}")
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

from tools.lib.csi_io import load_npz_as_packets
from tools.lib.dataset_metadata import DATA_DIR
from config import (
    DEFAULT_SUBCARRIERS,
    ENABLE_HAMPEL_FILTER,
    ENABLE_LOWPASS_FILTER,
    EVALUATION_INTERVAL,
    HAMPEL_THRESHOLD,
    HAMPEL_WINDOW,
    LOWPASS_CUTOFF,
    MOTION_OFF_HITS,
    MOTION_ON_HITS,
    SEG_WINDOW_SIZE,
)
from detector_interface import MotionState
from segmentation import SegmentationContext
from tools.lib.performance_report import (
    evaluate_idle_runtime_policy as evaluate_idle_runtime_policy_states,
)
from runtime_policy import RuntimeMotionPolicy, make_evaluation_cadence
from csi_features import (
    DEFAULT_FEATURES,
    L1_DELTA_LAG,
    L1DeltaTracker,
    extract_features_by_name,
)
from ml_detector import FEATURE_NAMES as EXPORTED_FEATURE_NAMES, MLDetector  # noqa: F401 (re-exported for tests)


def _needs_l1_series(feature_names):
    """Return whether any requested feature is computed from L1 deltas."""
    return any(str(name).startswith('l1_delta') for name in feature_names)

# ============================================================================
# Feature Selection
# ============================================================================
#
# Production MLP uses the feature set in src/features.DEFAULT_FEATURES.
# See ALGORITHMS.md "Feature Importance" for SHAP/correlation rankings.
# ============================================================================

TRAINING_FEATURES = DEFAULT_FEATURES
BINARY_TRAINING_LABELS = ('empty', 'static_presence', 'motion')
# Directories
GENERATED_DATA_DIR = generated_data_dir()
SRC_DIR = python_src_dir()
CPP_DIR = cpp_core_dir()

# Default training/evaluation configuration
DEFAULT_HIDDEN_LAYERS = [32, 16]
DEFAULT_FP_WEIGHT = 2.0
DEFAULT_SCALER_MODE = 'standard'
DEFAULT_BATCH_SIZE = 1024
DEFAULT_TORCH_DEVICE = 'cpu'
TRAINING_FEATURE_CACHE_VERSION = 8
# All chips included: MLDetector keeps the legacy variance-baseline CV normalization disabled, then
# extracts the exported raw/relative feature set from the same turbulence base.
DEFAULT_EXCLUDED_CHIPS = ()
DEFAULT_ARCHITECTURE_SWEEP = (
    {'name': 'Legacy (16-8)', 'layers': [16, 8]},
    {'name': 'Previous default (24-12)', 'layers': [24, 12]},
    {'name': 'Shallow (24)', 'layers': [24]},
    {'name': 'Current default (32-16)', 'layers': [32, 16]},
    {'name': 'Deep (24-12-6)', 'layers': [24, 12, 6]},
)
DEFAULT_EXPERIMENT_OUTPUT = GENERATED_DATA_DIR / 'mlp_architecture_experiment.json'
DEFAULT_FP_WEIGHT_EXPERIMENT_OUTPUT = GENERATED_DATA_DIR / 'mlp_fp_weight_experiment.json'
DEFAULT_ROBUSTNESS_EXPERIMENT_OUTPUT = GENERATED_DATA_DIR / 'ml_robustness_experiment.json'
DEFAULT_FP_WEIGHT_SWEEP = (1.0, 1.5, 2.0, 2.5, 3.0)
DEFAULT_EXPERIMENT_SCREENING_SEED = 20260519
DEFAULT_EXPERIMENT_INITIAL_SEEDS = (20260518, 20260519, 20260520)
DEFAULT_EXPERIMENT_FINAL_SEEDS = (20260518, 20260519, 20260520, 20260521, 20260522)
DEFAULT_PAIRED_GATE_CHIPS = ('C3', 'C5', 'C6', 'ESP32', 'S3')
DEFAULT_GAIN_STRESS_SCALES = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
GAIN_SENSITIVE_FEATURES = ()
DEFAULT_MAX_EPOCHS = 100
DEFAULT_EARLY_STOP_PATIENCE = 8
DEFAULT_LR_PATIENCE = 4
DEFAULT_CLIP_PERCENTILES = (1.0, 99.0)
ROBUSTNESS_SCREENING_SEED = 20260519
ROBUSTNESS_FILTER_SEEDS = (20260518, 20260519, 20260520)
ROBUSTNESS_FINAL_SEEDS = (20260518, 20260519, 20260520, 20260521, 20260522)
ROBUSTNESS_TARGET_RECALL = 95.0
ROBUSTNESS_TARGET_FP_RATE = 5.0
DATASET_ROLES = ('train', 'selection', 'holdout')
DEFAULT_TRAINING_ROLES = ('train',)
DEFAULT_PRIMARY_GROUP_KEY = 'lineage_group'
DEFAULT_BLOCK_GROUP_KEY = 'source_file'
DEFAULT_CV_FOLDS = 3
DEFAULT_SHAP_BACKGROUND_SAMPLES = 100
ROBUST_TAIL_GROUPS = 5
OOF_F1_EQUIVALENCE_MARGIN = 0.2
DEFAULT_REPORT_GROUP_KEYS = (
    'chip',
    'environment_group',
    'lineage_group',
    'session_group',
    'source_file',
)
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


def normalize_dataset_roles(roles, *, default=DEFAULT_TRAINING_ROLES):
    """Normalize dataset roles and reject unknown role names."""
    if roles is None:
        roles = default
    if isinstance(roles, str):
        roles = roles.split(',')
    normalized = {str(role).strip().lower() for role in roles if str(role).strip()}
    unknown = normalized.difference(DATASET_ROLES)
    if unknown:
        raise ValueError(f"Unsupported dataset role(s): {', '.join(sorted(unknown))}")
    return normalized




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


def parse_fp_weight_sweep(value):
    """Parse a comma-separated, positive FP-weight sweep."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        values = [float(item) for item in value]
    else:
        try:
            values = [float(item.strip()) for item in str(value).split(',') if item.strip()]
        except ValueError as exc:
            raise argparse.ArgumentTypeError("FP weights must be comma-separated numbers") from exc
    if not values or any(value <= 0.0 for value in values):
        raise argparse.ArgumentTypeError("FP weights must contain one or more positive values")
    return list(dict.fromkeys(values))


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


def _resolve_counterpart_name(label, entry, dataset_info=None):
    """Resolve the paired baseline/movement file from explicit metadata."""
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
    return None


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
    session_group = explicit_session or pair_id or f"file:{filename or 'unknown'}"
    lineage_group = str(file_info.get('_lineage_group') or session_group)
    dataset_role = str(file_info.get('dataset_role', 'train')).strip().lower() or 'train'

    return {
        'chip': chip,
        'collected_at': collected_at.isoformat() if collected_at else '',
        'day_group': day_group,
        'pair_id': pair_id or '',
        # Session grouping is the primary evaluation key. Use explicit session
        # metadata when available, otherwise fall back to the paired capture or file.
        'session_group': session_group,
        # Synthetic derivatives and their real source pair share one lineage so
        # grouped CV cannot train on one representation and validate on the other.
        'lineage_group': lineage_group,
        'dataset_role': dataset_role,
        'synthetic': bool(file_info.get('synthetic', False)),
        # Keep a dedicated environment field so future datasets can report
        # room/environment worst-groups without changing the training code again.
        'environment_group': explicit_environment or 'unknown-environment',
    }


def _fallback_file_context(filename, label, packet):
    """Create grouping metadata for files missing from dataset_info.json."""
    fallback = {
        'filename': filename,
        'chip': packet.get('chip', 'unknown'),
        'collected_at': packet.get('collected_at', ''),
        'dataset_role': packet.get('dataset_role', 'train'),
        'synthetic': packet.get('synthetic', False),
    }
    return _build_file_context(label, fallback)


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


def _npz_provenance_fields(label, file_info):
    """Read scalar lineage metadata missing from older dataset-info entries."""
    fields = {}
    filename = str(file_info.get('filename', ''))
    if not filename:
        return fields
    path = DATA_DIR / str(label) / filename
    if not path.exists():
        return fields
    try:
        with np.load(path, allow_pickle=False) as data:
            for key in ('source_dataset', 'generation_group', 'generation_mode', 'synthetic'):
                if key not in data.files:
                    continue
                value = np.asarray(data[key])
                if value.ndim == 0:
                    fields[key] = value.item()
    except (OSError, ValueError):
        return fields
    return fields


def get_file_metadata(dataset_info):
    """
    Get metadata for all files in dataset_info.json.

    Returns a dict mapping filename to metadata including normalization flags and
    grouping context used by training/evaluation.

    Args:
        dataset_info: Loaded dataset_info.json

    Returns:
        dict: {filename: {...}}
    """
    file_metadata = {}
    files_by_label = dataset_info.get('files', {})
    entries_by_filename = {}
    for label, file_list in files_by_label.items():
        for file_info in file_list:
            filename = file_info.get('filename', '')
            if filename:
                enriched = dict(file_info)
                for key, value in _npz_provenance_fields(label, enriched).items():
                    enriched.setdefault(key, value)
                entries_by_filename[str(filename)] = (str(label), enriched)

    for filename, (label, file_info) in entries_by_filename.items():
        base_context = _build_file_context(label, file_info, dataset_info=dataset_info)
        lineage_group = base_context['session_group']
        source_dataset = str(file_info.get('source_dataset', '')).strip()
        if source_dataset and source_dataset in entries_by_filename:
            source_label, source_info = entries_by_filename[source_dataset]
            source_context = _build_file_context(
                source_label,
                source_info,
                dataset_info=dataset_info,
            )
            lineage_group = source_context['session_group']
        elif file_info.get('synthetic') and file_info.get('generation_group'):
            lineage_group = f"synthetic:{file_info['generation_group']}"
        enriched = dict(file_info)
        enriched['_lineage_group'] = lineage_group
        file_metadata[filename] = _build_file_context(
            label,
            enriched,
            dataset_info=dataset_info,
        )
    return file_metadata


def load_all_data(environment_filter=None, excluded_chips=None,
                  allowed_labels=BINARY_TRAINING_LABELS,
                  require_sync_metadata=False,
                  dataset_roles=DEFAULT_TRAINING_ROLES):
    """
    Load all available CSI data from the data/ directory.

    Reads label from npz file metadata (not folder structure).
    Uses dataset_info.json only to determine if label is motion or idle.
    Uses the normalized turbulence pipeline and attaches grouping metadata
    (file, session/pair, environment when available).

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
    dataset_roles = normalize_dataset_roles(dataset_roles)
    all_packets = []
    stats = {
        'chips': set(),
        'labels': {},
        'total': 0,
        'files': [],
        'excluded_labels': set(),
        'excluded_chips': set(),
        'excluded_environments': set(),
        'excluded_missing_sync_metadata': set(),
        'excluded_dataset_roles': set(),
        'session_groups': set(),
        'lineage_groups': set(),
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

                dataset_role = meta.get('dataset_role', 'train')
                if dataset_role not in dataset_roles:
                    stats['excluded_dataset_roles'].add(dataset_role)
                    continue

                environment_group = meta.get('environment_group', 'unknown-environment')
                if environment_filter is not None and environment_group not in environment_filter:
                    stats['excluded_environments'].add(environment_group)
                    continue

                has_sync_metadata = any(
                    p.get('wifi_rx_start_ts_ns') is not None
                    or p.get('device_ticks_us') is not None
                    or p.get('wifi_rx_ts_us') is not None
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

                stats['session_groups'].add(meta.get('session_group', f"file:{npz_file.name}"))
                stats['lineage_groups'].add(meta.get('lineage_group', f"file:{npz_file.name}"))
                if environment_group != 'unknown-environment':
                    stats['environment_groups'].add(environment_group)

                # Add flags to each packet
                is_motion = is_motion_label(label, dataset_info)
                for idx, p in enumerate(packets):
                    p['is_motion'] = is_motion
                    p['label_name'] = label_lc
                    p['source_file'] = npz_file.name
                    p['packet_index'] = idx
                    p['chip'] = meta.get('chip', chip)
                    p['collected_at'] = meta.get('collected_at', '')
                    p['day_group'] = meta.get('day_group', 'unknown-day')
                    p['pair_id'] = meta.get('pair_id', '')
                    p['session_group'] = meta.get('session_group', f"file:{npz_file.name}")
                    p['lineage_group'] = meta.get('lineage_group', p['session_group'])
                    p['dataset_role'] = dataset_role
                    p['synthetic'] = bool(meta.get('synthetic', False))
                    p['environment_group'] = environment_group
                
                all_packets.extend(packets)
                stats['files'].append(npz_file.name)
                
            except Exception as e:
                print(f"  Warning: Could not load {npz_file.name}: {e}")
    
    stats['chips'] = sorted(stats['chips'])
    stats['excluded_labels'] = sorted(stats['excluded_labels'])
    stats['excluded_chips'] = sorted(stats['excluded_chips'])
    stats['excluded_environments'] = sorted(stats['excluded_environments'])
    stats['excluded_missing_sync_metadata'] = sorted(stats['excluded_missing_sync_metadata'])
    stats['excluded_dataset_roles'] = sorted(stats['excluded_dataset_roles'])
    stats['session_groups'] = sorted(stats['session_groups'])
    stats['lineage_groups'] = sorted(stats['lineage_groups'])
    stats['environment_groups'] = sorted(stats['environment_groups'])
    stats['sync_metadata_files'] = sorted(stats['sync_metadata_files'])
    return all_packets, stats


def _training_dataset_manifest(environment_filter=None,
                               excluded_chips=None,
                               allowed_labels=BINARY_TRAINING_LABELS,
                               dataset_roles=DEFAULT_TRAINING_ROLES):
    """Build the dataset fingerprint shared by feature and weight caches."""
    allowed_labels = sorted(normalize_allowed_labels(allowed_labels) or ())
    environment_filter = sorted(parse_environment_filter(environment_filter) or ())
    excluded_chips = sorted(parse_chip_filter(excluded_chips) or ())
    dataset_roles = sorted(normalize_dataset_roles(dataset_roles))
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
        'allowed_labels': allowed_labels,
        'environment_filter': environment_filter,
        'excluded_chips': excluded_chips,
        'dataset_roles': dataset_roles,
        'dataset_info': dataset_info,
        'files': files,
    }


def _feature_cache_manifest(feature_names, environment_filter=None,
                            excluded_chips=None,
                            allowed_labels=BINARY_TRAINING_LABELS,
                            dataset_roles=DEFAULT_TRAINING_ROLES,
                            window_size=SEG_WINDOW_SIZE,
                            enable_lowpass=ENABLE_LOWPASS_FILTER,
                            lowpass_cutoff=LOWPASS_CUTOFF,
                            enable_hampel=ENABLE_HAMPEL_FILTER,
                            hampel_window=HAMPEL_WINDOW,
                            hampel_threshold=HAMPEL_THRESHOLD,
                            packet_augmentation=None,
                            augmentation_seed=None):
    """Build a stable manifest for the cached feature matrix."""
    dataset_manifest = _training_dataset_manifest(
        environment_filter=environment_filter,
        excluded_chips=excluded_chips,
        allowed_labels=allowed_labels,
        dataset_roles=dataset_roles,
    )
    return {
        'version': TRAINING_FEATURE_CACHE_VERSION,
        'dataset': dataset_manifest,
        'feature_names': list(feature_names),
        'default_subcarriers': [int(sc) for sc in DEFAULT_SUBCARRIERS],
        'window_size': int(window_size),
        'enable_lowpass': bool(enable_lowpass),
        'lowpass_cutoff': float(lowpass_cutoff),
        'enable_hampel': bool(enable_hampel),
        'hampel_window': int(hampel_window),
        'hampel_threshold': float(hampel_threshold),
        'packet_augmentation': dict(sorted((packet_augmentation or {}).items())),
        'augmentation_seed': None if augmentation_seed is None else int(augmentation_seed),
    }


def _cache_path(prefix, manifest):
    """Return the on-disk path for a cache manifest."""
    payload = json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode('utf-8')
    digest = hashlib.sha256(payload).hexdigest()[:16]
    return GENERATED_DATA_DIR / f'{prefix}_{digest}.npz'


def _feature_cache_path(manifest):
    """Return the on-disk path for a feature-matrix cache."""
    return _cache_path('training_features', manifest)


def _load_feature_cache(cache_path, manifest):
    """Load cached feature matrix, labels, context, and stats."""
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
            X = data['X'].astype(np.float32, copy=False)
            y = data['y'].astype(np.int8, copy=False)
            if len(X) != len(y):
                raise ValueError(f"feature cache row mismatch (X={len(X)} y={len(y)})")
            for key, values in sample_context.items():
                if len(values) != len(y):
                    raise ValueError(
                        f"feature cache context mismatch for {key} "
                        f"(ctx={len(values)} y={len(y)})"
                    )
            return {
                'X': X,
                'y': y,
                'feature_names': data['feature_names'].astype(str).tolist(),
                'sample_context': sample_context,
                'stats': stats,
            }
    except Exception as exc:
        print(f"  Warning: ignoring invalid feature cache {cache_path.name}: {exc}")
        return None


def _save_feature_cache(cache_path, manifest, X, y, feature_names,
                        sample_context, stats):
    """Persist a feature-matrix cache for repeated local runs."""
    try:
        GENERATED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            'manifest_json': np.asarray(json.dumps(manifest, sort_keys=True)),
            'stats_json': np.asarray(json.dumps(stats, sort_keys=True)),
            'X': np.asarray(X, dtype=np.float32),
            'y': np.asarray(y, dtype=np.int8),
            'feature_names': np.asarray(feature_names, dtype=object),
            'context_keys': np.asarray(list(sample_context.keys()), dtype=object),
        }
        for key, values in sample_context.items():
            payload[f'ctx_{key}'] = np.asarray(values)
        np.savez(cache_path, **payload)
    except Exception as exc:
        print(f"  Warning: could not write feature cache {cache_path.name}: {exc}")


def _stable_text_seed(value):
    digest = hashlib.sha256(str(value).encode('utf-8')).digest()
    return int.from_bytes(digest[:4], byteorder='little') & 0x7FFFFFFF


def augment_csi_packets(packets, config, seed):
    """Return a deterministic packet-level augmented copy for training only."""
    if not config:
        return list(packets)
    gain_sigma = float(config.get('gain_sigma', 0.0))
    noise_sigma = float(config.get('noise_sigma', 0.0))
    packet_loss = float(config.get('packet_loss', 0.0))
    if min(gain_sigma, noise_sigma, packet_loss) < 0.0 or packet_loss >= 1.0:
        raise ValueError("invalid packet augmentation parameters")

    grouped = {}
    for packet in packets:
        grouped.setdefault(str(packet.get('source_file', '__single_stream__')), []).append(packet)
    augmented = []
    for source in sorted(grouped):
        rng = np.random.default_rng(derive_seed(seed, _stable_text_seed(source)))
        source_packets = grouped[source]
        sample_len = len(source_packets[0].get('csi_data', ())) if source_packets else 0
        subcarriers = max(1, sample_len // 2)
        if gain_sigma > 0.0:
            knots = rng.normal(0.0, gain_sigma, size=4)
            smooth_log_gain = np.interp(
                np.linspace(0.0, 1.0, subcarriers),
                np.linspace(0.0, 1.0, len(knots)),
                knots,
            )
            gains = np.exp(smooth_log_gain)
            gains /= np.mean(gains)
        else:
            gains = np.ones(subcarriers, dtype=np.float64)

        for packet in source_packets:
            if packet_loss > 0.0 and rng.random() < packet_loss:
                continue
            raw = np.asarray(packet['csi_data'], dtype=np.float64).copy()
            usable = min(len(raw) // 2, len(gains))
            for sc in range(usable):
                pair = slice(2 * sc, 2 * sc + 2)
                raw[pair] *= gains[sc]
                if noise_sigma > 0.0:
                    magnitude = max(1.0, float(np.linalg.norm(raw[pair])))
                    raw[pair] += rng.normal(0.0, noise_sigma * magnitude / np.sqrt(2.0), size=2)
            copied = dict(packet)
            copied['csi_data'] = np.clip(np.rint(raw), -128, 127).astype(np.int8)
            augmented.append(copied)
    return augmented


def load_training_matrix(environment_filter=None, excluded_chips=None,
                         feature_names=None, use_cache=True,
                         packet_augmentation=None, augmentation_seed=None,
                         dataset_roles=DEFAULT_TRAINING_ROLES):
    """Load or build the cached feature matrix used by training."""
    if feature_names is None:
        feature_names = DEFAULT_FEATURES.copy()
    feature_names = list(feature_names)

    feature_manifest = _feature_cache_manifest(
        feature_names,
        environment_filter=environment_filter,
        excluded_chips=excluded_chips,
        allowed_labels=BINARY_TRAINING_LABELS,
        dataset_roles=dataset_roles,
        packet_augmentation=packet_augmentation,
        augmentation_seed=augmentation_seed,
    )
    feature_cache_path = _feature_cache_path(feature_manifest)
    feature_matrix = None
    all_packets = None
    stats = None

    if use_cache:
        feature_matrix = _load_feature_cache(feature_cache_path, feature_manifest)
        if feature_matrix is not None:
            print(f"  Training feature cache: hit ({feature_cache_path.name})")
            stats = feature_matrix['stats']
        else:
            print(f"  Training feature cache: miss ({feature_cache_path.name})")
    else:
        print("  Training feature cache: disabled")

    if feature_matrix is None:
        load_start = perf_counter()
        all_packets, stats = load_all_data(
            environment_filter=environment_filter,
            excluded_chips=excluded_chips,
            dataset_roles=dataset_roles,
        )
        if packet_augmentation:
            all_packets = augment_csi_packets(all_packets, packet_augmentation, augmentation_seed)
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

        feature_matrix = {
            'X': np.asarray(X, dtype=np.float32),
            'y': np.asarray(y, dtype=np.int8),
            'feature_names': list(actual_feature_names),
            'sample_context': sample_context,
            'stats': stats,
        }
        if use_cache:
            _save_feature_cache(
                feature_cache_path,
                feature_manifest,
                feature_matrix['X'],
                feature_matrix['y'],
                feature_matrix['feature_names'],
                feature_matrix['sample_context'],
                feature_matrix['stats'],
            )
            print(f"  Training feature cache: wrote {feature_cache_path.name}")

    matrix = {
        'X': feature_matrix['X'],
        'y': feature_matrix['y'],
        'feature_names': feature_matrix['feature_names'],
        'sample_context': feature_matrix['sample_context'],
        'sample_weights': np.ones(len(feature_matrix['y']), dtype=np.float32),
        'stats': feature_matrix['stats'],
    }
    return matrix, all_packets


def extract_features(packets, window_size=SEG_WINDOW_SIZE,
                     feature_names=None,
                     enable_lowpass=ENABLE_LOWPASS_FILTER, lowpass_cutoff=LOWPASS_CUTOFF,
                     enable_hampel=ENABLE_HAMPEL_FILTER,
                     hampel_window=HAMPEL_WINDOW, hampel_threshold=HAMPEL_THRESHOLD,
                     ):
    """
    Extract features from CSI packets using sliding window.
    
    Uses SegmentationContext.add_turbulence() so the configured runtime filter
    chain matches the training pipeline, ensuring train/deploy alignment.
    
    Args:
        packets: List of CSI packets with 'csi_data' and 'label'
        window_size: Sliding window size (default: SEG_WINDOW_SIZE from config.py)
        feature_names: List of feature names to extract (default: DEFAULT_FEATURES)
        enable_lowpass: Enable low-pass filter on turbulence (default: config.py)
        lowpass_cutoff: Low-pass cutoff frequency in Hz (default: config.py)
        enable_hampel: Enable Hampel filtering on turbulence and L1-delta streams
                       (default: config.py)
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
        'lineage_group': [],
        'session_group': [],
        'environment_group': [],
        'pair_id': [],
        'day_group': [],
        'dataset_role': [],
        'synthetic': [],
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
            'lineage_group': file_packets[0].get('lineage_group', f"file:{source_file}"),
            'session_group': file_packets[0].get('session_group', f"file:{source_file}"),
            'environment_group': file_packets[0].get('environment_group', 'unknown-environment'),
            'pair_id': file_packets[0].get('pair_id', ''),
            'day_group': file_packets[0].get('day_group', 'unknown-day'),
            'dataset_role': file_packets[0].get('dataset_role', 'train'),
            'synthetic': bool(file_packets[0].get('synthetic', False)),
        }
        ctx = SegmentationContext(
            window_size=window_size,
            threshold=1.0,
            enable_lowpass=enable_lowpass,
            lowpass_cutoff=lowpass_cutoff,
            enable_hampel=enable_hampel,
            hampel_window=hampel_window,
            hampel_threshold=hampel_threshold,
        )
        needs_l1_series = _needs_l1_series(feature_names)
        l1_capacity = max(2, window_size - L1_DELTA_LAG)
        l1_tracker = (
            L1DeltaTracker(
                window_size=l1_capacity,
                lag=L1_DELTA_LAG,
                allocate_amplitude_buffer=False,
                enable_hampel=enable_hampel,
                hampel_window=hampel_window,
                hampel_threshold=hampel_threshold,
            )
            if needs_l1_series else None
        )
        l1_series = [0.0] * l1_capacity if needs_l1_series else None
        window_index = 0
        for pkt in file_packets:
            csi_data = pkt['csi_data']
            turb, amps = ctx.calculate_spatial_turbulence(
                csi_data,
                DEFAULT_SUBCARRIERS,
                return_amplitudes=True,
            )
            ctx.add_turbulence(turb)
            if l1_tracker is not None:
                l1_tracker.process_amplitudes(amps, len(amps))

            if ctx.buffer_count < window_size:
                continue

            # Reconstruct chronological order from circular buffer
            idx = ctx.buffer_index
            turb_list = ctx.turbulence_buffer[idx:] + ctx.turbulence_buffer[:idx]
            n = len(turb_list)

            l1_count = l1_tracker.copy_deltas_into(l1_series) if l1_tracker is not None else 0
            features = extract_features_by_name(
                turb_list, n,
                feature_names=feature_names,
                l1_series=l1_series,
                l1_series_count=l1_count,
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


class SessionBalancedRobustScaler:
    """Robust affine scaler fitted on equal-sized session/class strata."""

    def __init__(self, max_samples_per_stratum=2048):
        self.max_samples_per_stratum = int(max_samples_per_stratum)
        self.center_ = None
        self.scale_ = None
        self.selected_indices_ = None

    def fit(self, X, y=None, groups=None):
        X = np.asarray(X, dtype=np.float32)
        if y is None or groups is None:
            raise ValueError("session_balanced_robust requires labels and session groups")
        y = np.asarray(y)
        groups = np.asarray(groups).astype(str)
        if len(X) != len(y) or len(X) != len(groups):
            raise ValueError("scaler inputs must have matching rows")

        strata = {}
        for idx, key in enumerate(zip(groups.tolist(), y.tolist())):
            strata.setdefault((str(key[0]), int(key[1])), []).append(idx)
        non_empty = [indices for indices in strata.values() if indices]
        if not non_empty:
            raise ValueError("session_balanced_robust received no samples")
        per_stratum = min(self.max_samples_per_stratum, min(map(len, non_empty)))
        selected = []
        for key in sorted(strata):
            indices = np.asarray(strata[key], dtype=np.int64)
            if len(indices) <= per_stratum:
                selected.extend(indices.tolist())
            else:
                positions = np.linspace(0, len(indices) - 1, per_stratum, dtype=np.int64)
                selected.extend(indices[positions].tolist())
        self.selected_indices_ = np.asarray(selected, dtype=np.int64)
        balanced = X[self.selected_indices_]
        self.center_ = np.median(balanced, axis=0).astype(np.float32)
        q25, q75 = np.percentile(balanced, (25.0, 75.0), axis=0)
        self.scale_ = np.asarray(q75 - q25, dtype=np.float32)
        self.scale_[self.scale_ < 1e-6] = 1.0
        return self

    def transform(self, X):
        if self.center_ is None or self.scale_ is None:
            raise ValueError("session_balanced_robust scaler is not fitted")
        return (np.asarray(X, dtype=np.float32) - self.center_) / self.scale_

    def fit_transform(self, X, y=None, groups=None):
        return self.fit(X, y=y, groups=groups).transform(X)


def build_preprocessor(mode=DEFAULT_SCALER_MODE, clip_percentiles=DEFAULT_CLIP_PERCENTILES):
    """Build the feature normalization object used in CV and final training."""
    from sklearn.preprocessing import RobustScaler, StandardScaler

    if mode == 'standard':
        return StandardScaler()
    if mode == 'robust':
        return RobustScaler()
    if mode == 'session_balanced_robust':
        return SessionBalancedRobustScaler()
    if mode == 'clipped_standard':
        return ClippedStandardScaler(*clip_percentiles)
    raise ValueError(f"Unsupported scaler mode: {mode}")


def fit_preprocessor(preprocessor, X, y=None, sample_context=None):
    """Fit a scaler with fold-local metadata when the mode requires it."""
    if isinstance(preprocessor, SessionBalancedRobustScaler):
        groups = None if sample_context is None else sample_context.get('session_group')
        preprocessor.fit(X, y=y, groups=groups)
    elif not hasattr(preprocessor, 'fit') and hasattr(preprocessor, 'fit_transform'):
        # Preserve compatibility with lightweight test doubles and third-party
        # preprocessors that only expose fit_transform.
        preprocessor.fit_transform(X)
        if not hasattr(preprocessor, 'transform'):
            preprocessor.transform = lambda values: np.asarray(values)
    else:
        preprocessor.fit(X)
    return preprocessor


def apply_l1_feature_variant(X, feature_names, variant='core6'):
    """Apply host-side Core-6 L1 ratio candidates without mutating input rows."""
    X = np.asarray(X, dtype=np.float32)
    if variant in (None, 'core6'):
        return X.copy()
    supported = {'l1_std_relative', 'l1_waveform_relative', 'l1_both_relative'}
    if variant not in supported:
        raise ValueError(f"Unsupported L1 feature variant: {variant}")
    names = list(feature_names)
    required = ('l1_delta', 'l1_delta_std', 'l1_delta_waveform_length')
    missing = [name for name in required if name not in names]
    if missing:
        raise ValueError(f"L1 feature variant requires: {', '.join(missing)}")
    result = X.copy()
    mean_idx = names.index('l1_delta')
    denom = result[:, mean_idx] + 1e-3
    if variant in ('l1_std_relative', 'l1_both_relative'):
        std_idx = names.index('l1_delta_std')
        result[:, std_idx] = result[:, std_idx] / denom
    if variant in ('l1_waveform_relative', 'l1_both_relative'):
        waveform_idx = names.index('l1_delta_waveform_length')
        l1_count = max(1, int(SEG_WINDOW_SIZE - L1_DELTA_LAG) - 1)
        result[:, waveform_idx] = result[:, waveform_idx] / (l1_count * denom)
    return result


def normalized_feature_bounds(preprocessor, feature_names):
    """Return normalized bounds used to keep augmented Core-6 rows valid."""
    center, scale = get_preprocessor_arrays(preprocessor)
    lower = np.full(len(feature_names), -np.inf, dtype=np.float32)
    upper = np.full(len(feature_names), np.inf, dtype=np.float32)
    for name in ('turb_mad_over_mean', 'l1_delta', 'l1_delta_std', 'l1_delta_waveform_length'):
        if name in feature_names:
            idx = feature_names.index(name)
            lower[idx] = (0.0 - center[idx]) / scale[idx]
    if 'turb_autocorr' in feature_names:
        idx = feature_names.index('turb_autocorr')
        lower[idx] = (-1.0 - center[idx]) / scale[idx]
        upper[idx] = (1.0 - center[idx]) / scale[idx]
    return lower, upper


def augment_normalized_features(X, config, seed, bounds=None, apply_fraction=0.5):
    """Deterministically augment normalized training rows only."""
    X = np.asarray(X, dtype=np.float32)
    if not config:
        return X.copy()
    rng = np.random.default_rng(seed)
    result = X.copy()
    selected = rng.random(len(result)) < float(apply_fraction)
    if not np.any(selected):
        return result
    row_count, feature_count = int(np.sum(selected)), result.shape[1]
    noise_sigma = float(config.get('noise_sigma', 0.0))
    if noise_sigma > 0.0:
        result[selected] += rng.normal(0.0, noise_sigma, size=(row_count, feature_count)).astype(np.float32)
    jitter_sigma = float(config.get('jitter_sigma', 0.0))
    if jitter_sigma > 0.0:
        jitter = np.empty((row_count, feature_count), dtype=np.float32)
        jitter[:, :min(3, feature_count)] = rng.normal(0.0, jitter_sigma, size=(row_count, 1))
        if feature_count > 3:
            jitter[:, 3:] = rng.normal(0.0, jitter_sigma, size=(row_count, 1))
        result[selected] += jitter
    dropout_probability = float(config.get('dropout_probability', 0.0))
    if dropout_probability > 0.0:
        dropout = rng.random((row_count, feature_count)) < dropout_probability
        selected_rows = result[selected]
        selected_rows[dropout] = 0.0
        result[selected] = selected_rows
    if bounds is not None:
        lower, upper = bounds
        result = np.maximum(result, np.asarray(lower, dtype=np.float32))
        result = np.minimum(result, np.asarray(upper, dtype=np.float32))
    return result


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


def select_balanced_shap_indices(y, sample_context, max_samples, seed):
    """Select deterministic SHAP samples balanced by class, chip, and session."""
    y = np.asarray(y)
    max_samples = min(max(int(max_samples), 0), len(y))
    if max_samples == 0:
        return np.asarray([], dtype=np.int64)

    rng = np.random.default_rng(seed)
    chip_values = np.asarray(
        sample_context.get('chip', np.full(len(y), 'unknown-chip'))
        if sample_context is not None else np.full(len(y), 'unknown-chip')
    )
    session_values = np.asarray(
        sample_context.get('session_group', np.full(len(y), 'unknown-session'))
        if sample_context is not None else np.full(len(y), 'unknown-session')
    )

    buckets_by_label = {}
    for idx, label in enumerate(y):
        stratum = (str(chip_values[idx]), str(session_values[idx]))
        buckets_by_label.setdefault(int(label), {}).setdefault(stratum, []).append(idx)

    label_states = {}
    for label, buckets in buckets_by_label.items():
        keys = sorted(buckets)
        rng.shuffle(keys)
        shuffled_buckets = {}
        for key in keys:
            values = np.asarray(buckets[key], dtype=np.int64)
            rng.shuffle(values)
            shuffled_buckets[key] = values.tolist()
        label_states[label] = {
            'keys': keys,
            'buckets': shuffled_buckets,
            'cursor': 0,
        }

    labels = sorted(label_states)
    rng.shuffle(labels)
    selected = []
    while len(selected) < max_samples:
        progressed = False
        for label in labels:
            state = label_states[label]
            keys = state['keys']
            for _ in range(len(keys)):
                key = keys[state['cursor'] % len(keys)]
                state['cursor'] += 1
                bucket = state['buckets'][key]
                if bucket:
                    selected.append(bucket.pop())
                    progressed = True
                    break
            if len(selected) >= max_samples:
                break
        if not progressed:
            break

    return np.asarray(selected, dtype=np.int64)


def distribute_samples(total_samples, n_folds):
    """Distribute a requested sample count as evenly as possible across folds."""
    total_samples = max(int(total_samples), 0)
    n_folds = max(int(n_folds), 1)
    base, remainder = divmod(total_samples, n_folds)
    return [base + (fold < remainder) for fold in range(n_folds)]


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
    recall_tail = rows_by_recall[:min(ROBUST_TAIL_GROUPS, len(rows_by_recall))]
    fp_tail = rows_by_fp[:min(ROBUST_TAIL_GROUPS, len(rows_by_fp))]

    def tail_summary(tail_rows, metric, denominator):
        return {
            'value': float(np.mean([row[metric] for row in tail_rows])),
            'groups': [row['group'] for row in tail_rows],
            # One changed evaluation per group is the natural resolution of
            # these blocked rates; use its mean as the equivalence margin.
            'resolution': float(np.mean([
                100.0 / max(int(row[denominator]), 1)
                for row in tail_rows
            ])),
        }

    return {
        'rows': rows,
        'worst_recall': rows_by_recall[0],
        'worst_fp_rate': rows_by_fp[0],
        'tail_recall': tail_summary(recall_tail, 'recall', 'positives'),
        'tail_fp_rate': tail_summary(fp_tail, 'fp_rate', 'negatives'),
        'count': len(rows),
    }


def load_exported_ml_weights():
    """Load the currently exported MicroPython ML weights module."""
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

    logits = activations.reshape(-1)
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


def build_candidate_key(cv_results):
    """Ranking key for seeds/architectures under the robust evaluation protocol."""
    group_reports = cv_results.get('group_reports', {})
    if not group_reports and cv_results.get('candidate_key') is not None:
        return tuple(float(value) for value in cv_results['candidate_key'])
    # Real sessions lead ranking whenever the provenance split exists; the
    # synthetic stress metrics act only as regression guards in the comparison.
    session_report = (
        group_reports.get('real_session_group')
        or group_reports.get('session_group')
        or {}
    )
    chip_report = group_reports.get('chip') or {}

    worst_session_recall = session_report.get('worst_recall', {}).get('recall', 0.0)
    worst_session_fp = session_report.get('worst_fp_rate', {}).get('fp_rate', 100.0)
    worst_chip_recall = chip_report.get('worst_recall', {}).get('recall', 0.0)
    worst_chip_fp = chip_report.get('worst_fp_rate', {}).get('fp_rate', 100.0)
    tail_session_recall = session_report.get('tail_recall', {}).get(
        'value', worst_session_recall)
    tail_session_fp = session_report.get('tail_fp_rate', {}).get(
        'value', worst_session_fp)

    return (
        tail_session_recall,
        -tail_session_fp,
        worst_session_recall,
        worst_chip_recall,
        -worst_session_fp,
        -worst_chip_fp,
        cv_results.get('oof_f1', 0.0),
        cv_results.get('f1_mean', 0.0),
    )


def _row_resolution(row, denominator):
    """Return the percentage-point step represented by one changed outcome."""
    if not row:
        return 0.0
    count = int(row.get(denominator, 0))
    return 100.0 / count if count > 0 else 0.0


def compare_robust_cv(candidate, baseline):
    """Compare grouped OOF results with one-event equivalence margins."""
    candidate_reports = candidate.get('group_reports', {})
    baseline_reports = baseline.get('group_reports', {})
    # Real sessions decide movement; combined reports remain the fallback for
    # datasets that never produced a provenance split.
    candidate_session = (
        candidate_reports.get('real_session_group')
        or candidate_reports.get('session_group')
        or {}
    )
    baseline_session = (
        baseline_reports.get('real_session_group')
        or baseline_reports.get('session_group')
        or {}
    )
    candidate_synthetic = candidate_reports.get('synthetic_session_group') or {}
    baseline_synthetic = baseline_reports.get('synthetic_session_group') or {}
    candidate_chip = candidate_reports.get('chip') or {}
    baseline_chip = baseline_reports.get('chip') or {}

    checks = []

    def add_higher(label, candidate_value, baseline_value, margin, guard_only=False):
        delta = float(candidate_value) - float(baseline_value)
        checks.append({
            'label': label,
            'delta': delta,
            'margin': float(margin),
            'regressed': delta < -float(margin) - 1e-9,
            'improved': not guard_only and delta > float(margin) + 1e-9,
        })

    def add_lower(label, candidate_value, baseline_value, margin, guard_only=False):
        delta = float(candidate_value) - float(baseline_value)
        checks.append({
            'label': label,
            'delta': delta,
            'margin': float(margin),
            'regressed': delta > float(margin) + 1e-9,
            'improved': not guard_only and delta < -float(margin) - 1e-9,
        })

    def add_session_checks(candidate_report, baseline_report, scope,
                           guard_only=False):
        candidate_worst_recall = candidate_report.get('worst_recall', {})
        baseline_worst_recall = baseline_report.get('worst_recall', {})
        add_higher(
            f'CV worst-{scope} recall',
            candidate_worst_recall.get('recall', 0.0),
            baseline_worst_recall.get('recall', 0.0),
            max(
                _row_resolution(candidate_worst_recall, 'positives'),
                _row_resolution(baseline_worst_recall, 'positives'),
            ),
            guard_only=guard_only,
        )

        candidate_worst_fp = candidate_report.get('worst_fp_rate', {})
        baseline_worst_fp = baseline_report.get('worst_fp_rate', {})
        add_lower(
            f'CV worst-{scope} FP',
            candidate_worst_fp.get('fp_rate', 100.0),
            baseline_worst_fp.get('fp_rate', 100.0),
            max(
                _row_resolution(candidate_worst_fp, 'negatives'),
                _row_resolution(baseline_worst_fp, 'negatives'),
            ),
            guard_only=guard_only,
        )

        for key, label, higher_is_better in (
            ('tail_recall', f'CV tail-{scope} recall', True),
            ('tail_fp_rate', f'CV tail-{scope} FP', False),
        ):
            candidate_tail = candidate_report.get(key, {})
            baseline_tail = baseline_report.get(key, {})
            candidate_value = candidate_tail.get('value')
            baseline_value = baseline_tail.get('value')
            if candidate_value is None or baseline_value is None:
                continue
            margin = max(
                float(candidate_tail.get('resolution', 0.0)),
                float(baseline_tail.get('resolution', 0.0)),
            )
            if higher_is_better:
                add_higher(label, candidate_value, baseline_value, margin,
                           guard_only=guard_only)
            else:
                add_lower(label, candidate_value, baseline_value, margin,
                          guard_only=guard_only)

    add_session_checks(candidate_session, baseline_session, 'session')
    # Synthetic stress sessions may only block: a candidate cannot win by
    # improving synthetic derivatives, nor materially regress on them.
    if candidate_synthetic and baseline_synthetic:
        add_session_checks(
            candidate_synthetic,
            baseline_synthetic,
            'synthetic-session',
            guard_only=True,
        )

    for report_key, label, higher_is_better, denominator in (
        ('worst_recall', 'CV worst-chip recall', True, 'positives'),
        ('worst_fp_rate', 'CV worst-chip FP', False, 'negatives'),
    ):
        candidate_row = candidate_chip.get(report_key, {})
        baseline_row = baseline_chip.get(report_key, {})
        metric = 'recall' if higher_is_better else 'fp_rate'
        margin = max(
            _row_resolution(candidate_row, denominator),
            _row_resolution(baseline_row, denominator),
        )
        if higher_is_better:
            add_higher(label, candidate_row.get(metric, 0.0), baseline_row.get(metric, 0.0), margin)
        else:
            add_lower(label, candidate_row.get(metric, 100.0), baseline_row.get(metric, 100.0), margin)

    add_higher(
        'Blocked OOF F1',
        candidate.get('oof_f1', 0.0),
        baseline.get('oof_f1', 0.0),
        OOF_F1_EQUIVALENCE_MARGIN,
    )

    regressions = [check for check in checks if check['regressed']]
    improvements = [check for check in checks if check['improved']]
    return {
        'passed': not regressions and bool(improvements),
        'non_regression': not regressions,
        'material_improvement': bool(improvements),
        'checks': checks,
        'regressions': regressions,
        'improvements': improvements,
    }


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
                batch_size=DEFAULT_BATCH_SIZE, verbose=0, seed=None,
                feature_augmentation=None, feature_bounds=None):
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
        feature_augmentation: Optional normalized feature-space perturbation policy.
        feature_bounds: Optional normalized lower/upper bounds for augmented rows.
    
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
            if feature_augmentation:
                augmented = augment_normalized_features(
                    X_t[start:stop],
                    feature_augmentation,
                    derive_seed(seed, epoch, start),
                    bounds=feature_bounds,
                )
                batch_x = torch.from_numpy(augmented).to(device)
            else:
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


def predict_runtime_probabilities(model, X):
    """
    Return probabilities using the same post-logit mapping as Python/C++ runtime inference.
    """
    X = np.asarray(X, dtype=np.float32)
    logits = predict_logits(model, X)
    probabilities = np.empty_like(logits, dtype=np.float32)
    probabilities[logits < -20.0] = 0.0
    probabilities[logits > 20.0] = 1.0
    mask = (logits >= -20.0) & (logits <= 20.0)
    probabilities[mask] = 1.0 / (1.0 + np.exp(-logits[mask]))
    return probabilities


def cross_validate(X, y, hidden_layers=None, n_folds=DEFAULT_CV_FOLDS, max_epochs=DEFAULT_MAX_EPOCHS,
                   fp_weight=DEFAULT_FP_WEIGHT, sample_weight=None, groups=None,
                   sample_context=None, scaler_mode=DEFAULT_SCALER_MODE,
                   batch_size=DEFAULT_BATCH_SIZE, block_stride=1,
                   block_group_key=DEFAULT_BLOCK_GROUP_KEY,
                   report_group_keys=DEFAULT_REPORT_GROUP_KEYS, seed=None,
                   shap_samples=0, shap_feature_names=None, shap_seed=None,
                   feature_augmentation=None, X_aug=None, y_aug=None, groups_aug=None):
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
        shap_samples: Total held-out samples to explain across all folds
        shap_feature_names: Feature names aligned with the columns in X
        shap_seed: Optional deterministic seed for SHAP sampling
        feature_augmentation: Optional train-time normalized feature perturbation
        X_aug: Optional packet-augmented feature matrix (train-only)
        y_aug: Labels aligned with X_aug
        groups_aug: Split-group labels aligned with X_aug

    Returns:
        dict: Mean and std of each metric across folds
    """
    if hidden_layers is None:
        hidden_layers = list(DEFAULT_HIDDEN_LAYERS)
    feature_augmentation = dict(feature_augmentation or {})
    feature_names = list(shap_feature_names) if shap_feature_names else None

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
    shap_abs_sum = np.zeros(X.shape[1], dtype=np.float64)
    shap_count = 0
    shap_module = None
    fold_shap_counts = distribute_samples(shap_samples, effective_folds)
    if shap_samples > 0:
        try:
            import shap as shap_module
        except ImportError:
            print("Error: SHAP not installed. Run: pip install shap")

    for fold, (train_idx, val_idx) in enumerate(split_iter):
        fold_start = perf_counter()
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        sw_train_fold = sample_weight[train_idx] if sample_weight is not None else None

        # Fit normalization only on the training fold
        preprocess_start = perf_counter()
        scaler = build_preprocessor(scaler_mode)
        train_context = slice_sample_context(sample_context, train_idx)
        fit_preprocessor(
            scaler, X_train_fold, y=y_train_fold, sample_context=train_context)
        X_train_scaled = scaler.transform(X_train_fold)
        X_val_scaled = scaler.transform(X_val_fold)
        if groups is not None:
            train_groups = np.asarray(groups)[train_idx]
        else:
            train_groups = np.arange(len(train_idx))
        X_train_scaled, y_train_fold, sw_train_fold = _append_augmented_training_rows(
            X_train_scaled,
            y_train_fold,
            scaler,
            X_aug,
            y_aug,
            groups_aug,
            train_groups,
            sample_weight=sw_train_fold,
        )
        feature_bounds = None
        if feature_augmentation and feature_names is not None:
            feature_bounds = normalized_feature_bounds(scaler, feature_names)
        preprocess_elapsed = perf_counter() - preprocess_start

        train_predict_start = perf_counter()
        fold_seed = derive_seed(seed, fold)
        with suppress_stderr():
            model = train_model(X_train_scaled, y_train_fold,
                                hidden_layers=hidden_layers, max_epochs=max_epochs,
                                fp_weight=fp_weight, sample_weight=sw_train_fold,
                                batch_size=batch_size, seed=fold_seed,
                                feature_augmentation=feature_augmentation or None,
                                feature_bounds=feature_bounds)
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

        requested_fold_samples = fold_shap_counts[fold]
        if shap_module is not None and requested_fold_samples > 0:
            train_context = slice_sample_context(sample_context, train_idx)
            background_idx = select_balanced_shap_indices(
                y_train_fold,
                train_context,
                DEFAULT_SHAP_BACKGROUND_SAMPLES,
                derive_seed(shap_seed, fold, 1),
            )
            scored_local_idx = np.flatnonzero(local_mask)
            scored_context = slice_sample_context(val_context, scored_local_idx)
            explain_scored_idx = select_balanced_shap_indices(
                y_val_fold[scored_local_idx],
                scored_context,
                requested_fold_samples,
                derive_seed(shap_seed, fold, 2),
            )
            explain_local_idx = scored_local_idx[explain_scored_idx]
            shap_values = calculate_shap_values(
                model,
                X_train_scaled[background_idx],
                X_val_scaled[explain_local_idx],
                shap_module=shap_module,
                seed=derive_seed(shap_seed, fold, 3),
            )
            if shap_values is not None:
                shap_abs_sum += np.sum(np.abs(shap_values), axis=0)
                shap_count += len(shap_values)
                print(
                    f"  Fold {fold + 1}/{effective_folds} SHAP: "
                    f"explained {len(shap_values)} held-out samples"
                )
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
    if shap_count > 0:
        feature_names = shap_feature_names or [f'feature_{idx}' for idx in range(X.shape[1])]
        mean_abs_shap = shap_abs_sum / shap_count
        result['shap_importance'] = dict(sorted(
            ((name, float(value)) for name, value in zip(feature_names, mean_abs_shap)),
            key=lambda item: item[1],
            reverse=True,
        ))
        result['shap_samples'] = shap_count

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
        # Provenance-split session reports: synthetic derivatives may stress the
        # model, but they must not mask (or fake) movement in the real-session
        # worst/tail metrics that lead promotion.
        synthetic_flags = np.asarray(
            scored_context.get('synthetic', ()), dtype=bool)
        session_values = scored_context.get('session_group')
        if session_values is not None and synthetic_flags.size and synthetic_flags.any():
            for provenance_key, provenance_mask in (
                ('real_session_group', ~synthetic_flags),
                ('synthetic_session_group', synthetic_flags),
            ):
                provenance_idx = np.flatnonzero(provenance_mask)
                if provenance_idx.size == 0:
                    continue
                report = build_group_report(
                    y[scored_idx][provenance_idx],
                    oof_prob[scored_idx][provenance_idx],
                    np.asarray(session_values)[provenance_idx],
                )
                if report is not None:
                    group_reports[provenance_key] = report
        result['group_reports'] = group_reports

    return result


def leave_one_group_out_validation(group_key, unit, detail_group_key,
                                   skip_values=(), fp_weight=DEFAULT_FP_WEIGHT,
                                   seed=None, feature_names=None, hidden_layers=None,
                                   scaler_mode=DEFAULT_SCALER_MODE,
                                   batch_size=DEFAULT_BATCH_SIZE,
                                   excluded_chips=None, block_stride=SEG_WINDOW_SIZE,
                                   use_cache=True, augment=False):
    """Leave-one-group-out generalization check over a sample-context grouping.

    For each value of ``group_key`` (a room, a chip, ...), train on all other
    values and evaluate on the held-out one. This measures how well the detector
    transfers to a group it never saw during training. Grouped CV can still mix
    groups across folds, so it tends to be optimistic about cross-group
    generalization; this routine removes that leakage by making the group the
    split boundary.

    Args:
        group_key: Sample-context key defining the held-out unit (e.g.
            ``environment_group`` or ``chip``).
        unit: Human-readable singular noun for the group (e.g. ``environment``).
        detail_group_key: Secondary sample-context key reported as the worst
            sub-group inside each held-out fold (e.g. ``chip`` for rooms).
        skip_values: Group values treated as missing metadata and ignored.
        augment: If True, apply the robustness-winner train-time augmentation.

    This is a diagnostic only: it never trains a promotable model or exports
    runtime artifacts. Held-out scoring reuses the same block subsampling as
    grouped CV so the numbers stay comparable to the trainer's own report.
    """
    if hidden_layers is None:
        hidden_layers = list(DEFAULT_HIDDEN_LAYERS)
    if feature_names is None:
        feature_names = DEFAULT_FEATURES.copy()
    feature_names = list(feature_names)
    excluded_chips = parse_chip_filter(excluded_chips)
    skip_values = {str(v) for v in skip_values}
    feature_augmentation, packet_augmentation = resolve_training_augmentation(augment)

    try:
        ensure_torch_available()
        torch_device_label = describe_torch_device()
        seed = resolve_training_seed(seed, trailing_newline=True)
        set_global_determinism(seed, torch_module=torch)
    except ImportError as exc:
        print(f"Error: Missing dependency - {exc}")
        print("Install with: pip install torch scikit-learn")
        return 1
    except (RuntimeError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    print("\n" + "=" * 70)
    print(f"  LEAVE-ONE-{unit.upper()}-OUT GENERALIZATION CHECK")
    print("=" * 70)
    print(f"FP weight: {fp_weight}")
    print(f"Scaler: {scaler_mode}")
    print(f"Batch size: {batch_size}")
    print(f"Architecture: {' -> '.join(map(str, [len(feature_names)] + hidden_layers + [1]))}")
    print(
        "Augmentation: "
        f"{format_augmentation_config(feature_augmentation, packet_augmentation)}"
    )
    print(f"Torch device: {torch_device_label}")
    if excluded_chips is not None:
        print(f"Excluded chips: {', '.join(sorted(excluded_chips))}")

    print("\nLoading training matrix...")
    matrix, _all_packets = load_training_matrix(
        environment_filter=None,
        excluded_chips=excluded_chips,
        feature_names=feature_names,
        use_cache=use_cache,
    )
    X = matrix['X']
    y = matrix['y']
    sample_context = matrix['sample_context']
    X_aug = y_aug = groups_aug = None
    if packet_augmentation:
        print("Loading packet-augmented training matrix...")
        aug_matrix, _ = load_training_matrix(
            environment_filter=None,
            excluded_chips=excluded_chips,
            feature_names=feature_names,
            use_cache=use_cache,
            packet_augmentation=packet_augmentation,
            augmentation_seed=seed,
        )
        X_aug = aug_matrix['X']
        y_aug = aug_matrix['y']
        groups_aug = aug_matrix['sample_context'].get(group_key)

    group_values = sample_context.get(group_key)
    if group_values is None or len(group_values) == 0:
        print(f"Error: no {unit} metadata available for cross-{unit} CV")
        return 1
    group_values = np.asarray([str(v) for v in group_values])

    groups = sorted(
        name for name in set(group_values.tolist())
        if name and name not in skip_values
    )
    if len(groups) < 2:
        print(
            f"Error: need at least 2 named {unit} groups, found {len(groups)}: "
            f"{', '.join(groups) or 'none'}"
        )
        return 1

    skipped_count = int(np.sum(np.isin(group_values, list(skip_values)))) if skip_values else 0
    print(f"  Named {unit} groups: {', '.join(groups)}")
    if skipped_count:
        print(f"  Windows without {unit} metadata (ignored): {skipped_count}")

    label_values = np.asarray(sample_context.get(
        'label_name', np.full(len(y), 'unknown')))

    fold_rows = []
    for held_out in groups:
        test_mask = group_values == held_out
        train_mask = np.isin(group_values, groups) & ~test_mask

        n_train = int(np.sum(train_mask))
        n_test = int(np.sum(test_mask))
        test_pos = int(np.sum(y[test_mask] == 1))
        test_neg = n_test - test_pos
        train_groups = [name for name in groups if name != held_out]

        print("\n" + "-" * 70)
        print(f"Held-out {unit}: {held_out}")
        print(f"  Train on: {', '.join(train_groups)} ({n_train} windows)")
        print(f"  Test on:  {held_out} ({n_test} windows: {test_pos} motion, {test_neg} idle)")

        if test_pos == 0 or test_neg == 0:
            print(f"  Skipped: held-out {unit} lacks both motion and idle windows")
            continue
        if n_train == 0:
            print(f"  Skipped: no training windows for the remaining {unit} groups")
            continue

        scaler = build_preprocessor(scaler_mode)
        train_context = slice_sample_context(sample_context, np.flatnonzero(train_mask))
        fit_preprocessor(
            scaler, X[train_mask], y=y[train_mask], sample_context=train_context)
        X_train_scaled = scaler.transform(X[train_mask])
        X_test_scaled = scaler.transform(X[test_mask])
        X_train_scaled, y_train_fold, _ = _append_augmented_training_rows(
            X_train_scaled,
            y[train_mask],
            scaler,
            X_aug,
            y_aug,
            groups_aug,
            train_groups,
        )
        feature_bounds = None
        if feature_augmentation:
            feature_bounds = normalized_feature_bounds(scaler, feature_names)

        fold_seed = derive_seed(seed, _stable_text_seed(held_out))
        with suppress_stderr():
            model = train_model(
                X_train_scaled, y_train_fold,
                hidden_layers=hidden_layers,
                fp_weight=fp_weight,
                batch_size=batch_size,
                seed=fold_seed,
                feature_augmentation=feature_augmentation or None,
                feature_bounds=feature_bounds,
            )
            test_prob = predict_probabilities(model, X_test_scaled)

        test_context = slice_sample_context(sample_context, np.flatnonzero(test_mask))
        block_mask = build_block_mask(
            test_context, stride=block_stride, group_key=DEFAULT_BLOCK_GROUP_KEY)
        if block_mask is None:
            block_mask = np.ones(n_test, dtype=bool)

        y_test = y[test_mask]
        metrics = evaluate_probabilities(y_test[block_mask], test_prob[block_mask])
        detail_values = test_context.get(detail_group_key)
        detail_report = build_group_report(
            y_test[block_mask], test_prob[block_mask],
            np.asarray(detail_values)[block_mask] if detail_values is not None else None,
        )

        # False-positive breakdown by idle sub-type on the held-out group.
        test_labels = label_values[test_mask][block_mask]
        idle_breakdown = {}
        for idle_label in ('empty', 'static_presence'):
            idle_mask = test_labels == idle_label
            n_idle = int(np.sum(idle_mask))
            if n_idle == 0:
                continue
            idle_metrics = evaluate_probabilities(
                y_test[block_mask][idle_mask], test_prob[block_mask][idle_mask])
            idle_breakdown[idle_label] = (n_idle, idle_metrics['fp_rate'])

        worst_detail = detail_report.get('worst_recall') if detail_report else None
        detail_noun = detail_group_key.replace('_group', '')
        print(
            f"  Recall={metrics['recall']:.1f}%  FP={metrics['fp_rate']:.1f}%  "
            f"Precision={metrics['precision']:.1f}%  F1={metrics['f1']:.1f}%  "
            f"(scored {int(np.sum(block_mask))} windows)"
        )
        for idle_label, (n_idle, fp_rate) in idle_breakdown.items():
            print(f"    {idle_label} FP: {fp_rate:.1f}% ({n_idle} windows)")
        if worst_detail:
            print(
                f"    worst {detail_noun} recall: {worst_detail['group']} "
                f"{worst_detail['recall']:.1f}% (FP {worst_detail['fp_rate']:.1f}%)"
            )

        fold_rows.append({
            'group': held_out,
            'train_windows': n_train,
            'test_windows': int(np.sum(block_mask)),
            'test_motion': test_pos,
            'test_idle': test_neg,
            **metrics,
            'idle_breakdown': idle_breakdown,
            'worst_detail': worst_detail,
        })

    if not fold_rows:
        print(f"\nNo {unit} could be evaluated as a held-out fold.")
        return 1

    print("\n" + "=" * 70)
    print(f"  SUMMARY (each row: model never saw that {unit} during training)")
    print("=" * 70)
    header = f"{unit:<16}{'recall':>9}{'fp':>8}{'prec':>8}{'f1':>8}{'test_win':>10}"
    print(header)
    print("-" * len(header))
    for row in fold_rows:
        print(
            f"{row['group']:<16}{row['recall']:>8.1f}%{row['fp_rate']:>7.1f}%"
            f"{row['precision']:>7.1f}%{row['f1']:>7.1f}%{row['test_windows']:>10}"
        )
    macro_recall = float(np.mean([r['recall'] for r in fold_rows]))
    macro_fp = float(np.mean([r['fp_rate'] for r in fold_rows]))
    macro_f1 = float(np.mean([r['f1'] for r in fold_rows]))
    worst_recall = min(fold_rows, key=lambda r: r['recall'])
    worst_fp = max(fold_rows, key=lambda r: r['fp_rate'])
    print("-" * len(header))
    print(
        f"{'macro-average':<16}{macro_recall:>8.1f}%{macro_fp:>7.1f}%"
        f"{'':>7}{macro_f1:>7.1f}%"
    )
    print(f"\nWorst held-out recall: {worst_recall['group']} {worst_recall['recall']:.1f}%")
    print(f"Worst held-out FP rate: {worst_fp['group']} {worst_fp['fp_rate']:.1f}%")
    print("\nRuntime artifacts unchanged (diagnostic run).")
    return 0


def cross_environment_validation(**kwargs):
    """Leave-one-environment-out generalization check (train on other rooms)."""
    return leave_one_group_out_validation(
        group_key='environment_group',
        unit='environment',
        detail_group_key='chip',
        skip_values=('unknown-environment',),
        **kwargs,
    )


def cross_chip_validation(**kwargs):
    """Leave-one-chip-out generalization check (train on other chips)."""
    return leave_one_group_out_validation(
        group_key='chip',
        unit='chip',
        detail_group_key='environment_group',
        skip_values=('', 'unknown', 'UNKNOWN', 'unknown-chip'),
        **kwargs,
    )


ROBUSTNESS_FEATURE_AUGMENTATIONS = (
    ('noise_002', {'noise_sigma': 0.02}),
    ('noise_005', {'noise_sigma': 0.05}),
    ('noise_010', {'noise_sigma': 0.10}),
    ('jitter_005', {'jitter_sigma': 0.05}),
    ('jitter_010', {'jitter_sigma': 0.10}),
    ('dropout_002', {'dropout_probability': 0.02}),
    ('dropout_005', {'dropout_probability': 0.05}),
    ('combined_moderate', {
        'noise_sigma': 0.05,
        'jitter_sigma': 0.05,
        'dropout_probability': 0.02,
    }),
)
ROBUSTNESS_PACKET_AUGMENTATIONS = (
    ('gain_005', {'gain_sigma': 0.05}),
    ('gain_010', {'gain_sigma': 0.10}),
    ('amplitude_noise_001', {'noise_sigma': 0.01}),
    ('amplitude_noise_003', {'noise_sigma': 0.03}),
    ('packet_loss_005', {'packet_loss': 0.05}),
    ('packet_loss_010', {'packet_loss': 0.10}),
    ('packet_combined_moderate', {
        'gain_sigma': 0.05,
        'noise_sigma': 0.01,
        'packet_loss': 0.05,
    }),
)
# Production shortcut for --augment: Core-6 robustness campaign winner
# (feature jitter 0.10 + moderate packet combined augmentation).
ROBUSTNESS_WINNER_NAME = (
    'baseline_standard__feature_jitter_010__packet_packet_combined_moderate'
)
ROBUSTNESS_WINNER_FEATURE_AUGMENTATION = {'jitter_sigma': 0.10}
ROBUSTNESS_WINNER_PACKET_AUGMENTATION = {
    'gain_sigma': 0.05,
    'noise_sigma': 0.01,
    'packet_loss': 0.05,
}


def resolve_training_augmentation(augment):
    """Return (feature_augmentation, packet_augmentation) for production training."""
    if not augment:
        return {}, {}
    return (
        dict(ROBUSTNESS_WINNER_FEATURE_AUGMENTATION),
        dict(ROBUSTNESS_WINNER_PACKET_AUGMENTATION),
    )


def format_augmentation_config(feature_augmentation=None, packet_augmentation=None):
    """Compact one-line description of an active training augmentation recipe."""
    feature_augmentation = dict(feature_augmentation or {})
    packet_augmentation = dict(packet_augmentation or {})
    if not feature_augmentation and not packet_augmentation:
        return 'none'
    parts = [f"recipe={ROBUSTNESS_WINNER_NAME}"]
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


def _robustness_candidate(name, scaler='standard', l1_variant='core6',
                          feature_augmentation=None, packet_augmentation=None):
    return {
        'name': str(name),
        'scaler': str(scaler),
        'l1_variant': str(l1_variant),
        'feature_augmentation': dict(feature_augmentation or {}),
        'packet_augmentation': dict(packet_augmentation or {}),
    }


def robustness_run_rank_key(run):
    """Rank one complete robustness run; lower is better."""
    folds = run['folds']
    pass_count = sum(
        row['recall'] > ROBUSTNESS_TARGET_RECALL
        and row['fp_rate'] < ROBUSTNESS_TARGET_FP_RATE
        for row in folds
    )
    return (
        -pass_count,
        max(row['fp_rate'] for row in folds),
        -min(row['recall'] for row in folds),
        -float(np.mean([row['f1'] for row in folds])),
        run['candidate']['name'],
    )


def aggregate_robustness_runs(candidate, runs):
    """Aggregate fold metrics across seeds for one robustness candidate."""
    by_fold = {}
    for run in runs:
        for row in run['folds']:
            by_fold.setdefault(row['fold'], []).append(row)
    fold_summaries = []
    for fold_name in sorted(by_fold):
        rows = by_fold[fold_name]
        median_recall = float(np.median([row['recall'] for row in rows]))
        median_fp = float(np.median([row['fp_rate'] for row in rows]))
        median_f1 = float(np.median([row['f1'] for row in rows]))
        seed_passes = sum(
            row['recall'] > ROBUSTNESS_TARGET_RECALL
            and row['fp_rate'] < ROBUSTNESS_TARGET_FP_RATE
            for row in rows
        )
        fold_summaries.append({
            'fold': fold_name,
            'median_recall': median_recall,
            'median_fp_rate': median_fp,
            'median_f1': median_f1,
            'seed_passes': int(seed_passes),
            'seed_count': len(rows),
            'median_pass': bool(
                median_recall > ROBUSTNESS_TARGET_RECALL
                and median_fp < ROBUSTNESS_TARGET_FP_RATE
            ),
        })
    return {
        'candidate': candidate,
        'seeds': [int(run['seed']) for run in runs],
        'folds': fold_summaries,
        'holdout_count': len(fold_summaries),
        'median_pass_count': sum(row['median_pass'] for row in fold_summaries),
        'worst_median_fp_rate': max(row['median_fp_rate'] for row in fold_summaries),
        'worst_median_recall': min(row['median_recall'] for row in fold_summaries),
        'macro_median_f1': float(np.mean([row['median_f1'] for row in fold_summaries])),
        'runs': runs,
    }


def robustness_summary_rank_key(summary):
    return (
        -summary['median_pass_count'],
        summary['worst_median_fp_rate'],
        -summary['worst_median_recall'],
        -summary['macro_median_f1'],
        summary['candidate']['name'],
    )


def evaluate_robustness_candidate(candidate, seed, matrix, augmented_matrix=None,
                                  hidden_layers=None, fp_weight=DEFAULT_FP_WEIGHT,
                                  batch_size=DEFAULT_BATCH_SIZE):
    """Evaluate one candidate across every environment and chip holdout."""
    hidden_layers = list(hidden_layers or DEFAULT_HIDDEN_LAYERS)
    feature_names = list(matrix['feature_names'])
    X = apply_l1_feature_variant(matrix['X'], feature_names, candidate['l1_variant'])
    y = np.asarray(matrix['y'])
    context = matrix['sample_context']
    X_aug = y_aug = context_aug = None
    if augmented_matrix is not None:
        X_aug = apply_l1_feature_variant(
            augmented_matrix['X'], feature_names, candidate['l1_variant'])
        y_aug = np.asarray(augmented_matrix['y'])
        context_aug = augmented_matrix['sample_context']

    dimensions = (
        ('environment_group', 'environment', {'unknown-environment'}, 'chip'),
        ('chip', 'chip', {'', 'unknown', 'UNKNOWN', 'unknown-chip'}, 'environment_group'),
    )
    folds = []
    expected_folds = []
    run_started = perf_counter()
    for group_key, unit, skipped, detail_key in dimensions:
        group_values = np.asarray(context[group_key]).astype(str)
        groups = sorted(set(group_values.tolist()) - skipped)
        for held_out in groups:
            expected_folds.append(f'{unit}:{held_out}')
            fold_started = perf_counter()
            test_mask = group_values == held_out
            train_mask = np.isin(group_values, groups) & ~test_mask
            if not np.any(train_mask) or len(set(y[test_mask].tolist())) < 2:
                continue
            train_context = slice_sample_context(context, np.flatnonzero(train_mask))
            scaler = build_preprocessor(candidate['scaler'])
            fit_preprocessor(scaler, X[train_mask], y=y[train_mask], sample_context=train_context)
            X_train = scaler.transform(X[train_mask])
            y_train = y[train_mask]
            if X_aug is not None:
                aug_groups = np.asarray(context_aug[group_key]).astype(str)
                aug_mask = np.isin(aug_groups, groups) & (aug_groups != held_out)
                if np.any(aug_mask):
                    X_train = np.concatenate((X_train, scaler.transform(X_aug[aug_mask])), axis=0)
                    y_train = np.concatenate((y_train, y_aug[aug_mask]), axis=0)
            X_test = scaler.transform(X[test_mask])
            bounds = normalized_feature_bounds(scaler, feature_names)
            fold_seed = derive_seed(seed, _stable_text_seed(f'{group_key}:{held_out}'))
            with suppress_stderr():
                model = train_model(
                    X_train,
                    y_train,
                    hidden_layers=hidden_layers,
                    fp_weight=fp_weight,
                    batch_size=batch_size,
                    seed=fold_seed,
                    feature_augmentation=candidate['feature_augmentation'],
                    feature_bounds=bounds,
                )
                probabilities = predict_probabilities(model, X_test)

            test_context = slice_sample_context(context, np.flatnonzero(test_mask))
            block_mask = build_block_mask(
                test_context, stride=SEG_WINDOW_SIZE, group_key=DEFAULT_BLOCK_GROUP_KEY)
            if block_mask is None:
                block_mask = np.ones(int(np.sum(test_mask)), dtype=bool)
            y_test = y[test_mask][block_mask]
            scored_prob = probabilities[block_mask]
            metrics = evaluate_probabilities(y_test, scored_prob)
            detail_values = np.asarray(test_context[detail_key])[block_mask]
            detail_report = build_group_report(y_test, scored_prob, detail_values)
            labels = np.asarray(test_context['label_name'])[block_mask]
            idle_breakdown = {}
            for label in ('empty', 'static_presence'):
                label_mask = labels == label
                if np.any(label_mask):
                    label_metrics = evaluate_probabilities(
                        y_test[label_mask], scored_prob[label_mask])
                    idle_breakdown[label] = {
                        'samples': int(np.sum(label_mask)),
                        'fp_rate': float(label_metrics['fp_rate']),
                    }
            folds.append({
                'fold': f'{unit}:{held_out}',
                'dimension': unit,
                'group': held_out,
                'train_windows': int(len(y_train)),
                'test_windows': int(np.sum(block_mask)),
                **{key: float(value) if isinstance(value, (float, np.floating)) else int(value)
                   for key, value in metrics.items()},
                'idle_breakdown': idle_breakdown,
                'worst_detail_recall': None if detail_report is None else detail_report['worst_recall'],
                'worst_detail_fp_rate': None if detail_report is None else detail_report['worst_fp_rate'],
                'seconds': float(perf_counter() - fold_started),
            })
    completed_folds = sorted(row['fold'] for row in folds)
    if completed_folds != sorted(expected_folds):
        missing = sorted(set(expected_folds) - set(completed_folds))
        raise RuntimeError(
            f"robustness candidate completed {len(folds)}/{len(expected_folds)} holdouts; "
            f"missing: {', '.join(missing) or 'none'}"
        )
    result = {
        'candidate': candidate,
        'seed': int(seed),
        'folds': folds,
        'holdout_count': len(folds),
        'seconds': float(perf_counter() - run_started),
    }
    result['rank_key'] = list(robustness_run_rank_key(result))
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

This file is auto-generated by train_ml_model.py.
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


# Canonical C++ feature ids, mirroring the MLFeatureId enum in
# src/cpp/core/csi_features.h. Keep the numeric values in sync. Only features
# with a real C++ extractor entry can be exported to firmware.
CPP_FEATURE_IDS = {
    'turb_skewness': 5,
    'turb_autocorr': 6,
    'turb_mad_over_mean': 13,
    'l1_delta': 17,
    'l1_delta_std': 18,
    'l1_delta_waveform_length': 23,
}


def resolve_cpp_feature_ids(feature_names):
    """Map feature names to their C++ ids, raising on any unsupported feature."""
    ids = []
    for name in feature_names:
        if name not in CPP_FEATURE_IDS:
            raise ValueError(
                f"feature {name!r} has no C++ extractor id; add it to "
                f"CPP_FEATURE_IDS and the MLFeatureId enum in csi_features.h "
                f"before exporting a model that uses it"
            )
        ids.append(CPP_FEATURE_IDS[name])
    return ids


def export_cpp_weights(model, scaler, output_path, seed=None,
                       feature_names=None, scaler_mode=DEFAULT_SCALER_MODE):
    """
    Export model weights to the shared C++ header.
    
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

    def cpp_float(value):
        """Render a numeric literal with a valid C++ float suffix."""
        text = f'{float(value):.9g}'
        if 'e' not in text and 'E' not in text and '.' not in text:
            text += '.0'
        return text + 'f'

    weights = extract_model_weights(model)
    architecture = get_model_architecture(model)
    arch = ' -> '.join(map(str, architecture))
    center, scale = get_preprocessor_arrays(scaler)
    if feature_names is None:
        feature_names = list(TRAINING_FEATURES)
    
    feature_ids = resolve_cpp_feature_ids(feature_names)

    seed_info = f"Seed: {seed}"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    architecture_csv = ', '.join(str(x) for x in architecture)
    center_csv = ', '.join(cpp_float(x) for x in center)
    scale_csv = ', '.join(cpp_float(x) for x in scale)
    feature_ids_csv = ', '.join(str(i) for i in feature_ids)
    feature_names_comment = ', '.join(feature_names)
    
    code = f'''/*
 * ESPectre - ML Model Weights
 * 
 * Auto-generated neural network weights for motion detection.
 * Architecture: {arch}
 * Normalization: {scaler_mode}
 * Trained: {timestamp}
 * {seed_info}
 * 
 * This file is auto-generated by train_ml_model.py.
 * DO NOT EDIT - your changes will be overwritten!
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#pragma once

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

// Feature identity (MLFeatureId in csi_features.h), one per model input slot.
// Order: {feature_names_comment}
constexpr uint8_t ML_FEATURE_IDS[{len(feature_ids)}] = {{{feature_ids_csv}}};

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
                + ', '.join(cpp_float(x) for x in flat_weights) + '};\n'
        code += f'constexpr float ML_B{layer_num}[{out_size}] = {{{", ".join(cpp_float(x) for x in b)}}};\n\n'
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
    predictions = predict_runtime_probabilities(model, X_test_scaled)
    
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


def print_correlation_table(correlations, current_features=None):
    """Print correlation results in a nice table."""
    from csi_features import DEFAULT_FEATURES
    
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

def calculate_shap_values(model, background, X_explain, shap_module=None, seed=None):
    """Calculate SHAP values for explicitly separated background and explain sets."""
    if shap_module is None:
        try:
            import shap as shap_module
        except ImportError:
            print("Error: SHAP not installed. Run: pip install shap")
            return None

    if len(background) == 0 or len(X_explain) == 0:
        return None

    explainer = shap_module.Explainer(
        lambda values: predict_probabilities(model, values).reshape(-1, 1),
        background,
        algorithm='permutation',
        seed=seed,
    )

    with suppress_stderr():
        shap_values = explainer(X_explain).values

    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.asarray(shap_values)
    while shap_values.ndim > 2 and shap_values.shape[-1] == 1:
        shap_values = np.squeeze(shap_values, axis=-1)
    if shap_values.ndim == 1:
        shap_values = shap_values.reshape(len(X_explain), -1)
    return shap_values


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
    provenance_keys = tuple(
        key for key in ('real_session_group', 'synthetic_session_group')
        if key in group_reports
    )
    for group_key in DEFAULT_REPORT_GROUP_KEYS + provenance_keys:
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
        if group_key in ('lineage_group', 'session_group') + provenance_keys:
            tail_recall = report.get('tail_recall', {})
            tail_fp = report.get('tail_fp_rate', {})
            print(
                f"  Worst-{len(tail_recall.get('groups', []))} {group_key} mean: "
                f"R={tail_recall.get('value', 0.0):.1f}% "
                f"FP={tail_fp.get('value', 0.0):.1f}%"
            )


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
    
    print("\nSummary:")
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
              evaluate_deployment=False,
              deployment_roles=('selection', 'holdout'),
              allow_legacy_gate_fallback=True,
              environment_filter=None, excluded_chips=None,
              positive_chip_boost=None,
              use_cache=True, augment=False):
    """
    Train models with all available data.
    
    Args:
        fp_weight: Multiplier for class 0 (IDLE) weight. Values >1.0 penalize
                   false positives more, producing a more conservative model.
        seed: Optional random seed for reproducible training. If None, a random
              seed is generated and saved for reproducibility.
        feature_names: List of feature names to use. If None, uses DEFAULT_FEATURES.
        feature_importance: If True, calculate grouped out-of-fold SHAP importance.
        ablation: If True, run ablation study instead of training.
        hidden_layers: Hidden layer widths. None uses DEFAULT_HIDDEN_LAYERS.
        scaler_mode: Feature normalization mode.
        batch_size: Mini-batch size used for training and CV.
        export_artifacts: If False, leave runtime artifacts unchanged.
        evaluate_deployment: Train the final in-memory model and run the paired
                             gate even when artifacts are not exported.
        deployment_roles: Dataset roles allowed in the deployment replay.
        allow_legacy_gate_fallback: Use the latest real train pair when no
                                    role-isolated replay is configured.
        environment_filter: Optional environment name(s) to keep.
        excluded_chips: Optional chip name(s) to exclude.
        positive_chip_boost: Optional {CHIP: factor} boost applied to motion
                             samples after feature extraction.
        use_cache: If True, reuse the cached feature matrix.
        augment: If True, apply the Core-6 robustness-winner train-time
                 augmentation recipe (feature jitter + packet combined).

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
    feature_augmentation, packet_augmentation = resolve_training_augmentation(augment)
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
        use_cache=use_cache,
    )
    X = matrix['X']
    y = matrix['y']
    actual_feature_names = matrix['feature_names']
    sample_context = matrix['sample_context']
    sample_weights = matrix['sample_weights']
    stats = matrix['stats']
    X_aug = y_aug = groups_aug = None
    if packet_augmentation:
        print("Loading packet-augmented training matrix...")
        aug_matrix, _ = load_training_matrix(
            environment_filter=environment_filter,
            excluded_chips=excluded_chips,
            feature_names=feature_names,
            use_cache=use_cache,
            packet_augmentation=packet_augmentation,
            augmentation_seed=seed,
        )
        X_aug = aug_matrix['X']
        y_aug = aug_matrix['y']
        groups_aug = aug_matrix['sample_context'].get(DEFAULT_PRIMARY_GROUP_KEY)
    
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
    print(f"  Session groups: {len(stats.get('session_groups', []))}")
    print(f"  Lineage groups: {len(stats.get('lineage_groups', []))}")
    if stats.get('excluded_dataset_roles'):
        print(
            "  Reserved roles excluded from training: "
            + ', '.join(stats['excluded_dataset_roles'])
        )
    if stats.get('environment_groups'):
        print(f"  Named environments: {len(stats['environment_groups'])}")
    for label, count in sorted(stats['labels'].items()):
        print(f"  {label}: {count} packets")
    print(f"  Total: {stats['total']} packets")
    
    print(f"Architecture: {' -> '.join(map(str, [len(feature_names)] + hidden_layers + [1]))}")
    print(f"Scaler: {scaler_mode}")
    print(f"Batch size: {batch_size}")
    print(
        "Augmentation: "
        f"{format_augmentation_config(feature_augmentation, packet_augmentation)}"
    )
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
    if X_aug is not None:
        print(f"  Packet-augmented samples: {len(X_aug)} (train-only)")

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
        print(
            "\nCV-only ablation is a screening diagnostic. "
            "Validate any finalist with --ablation-feature before making a feature decision."
        )
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
            shap_samples=shap_samples if feature_importance else 0,
            shap_feature_names=actual_feature_names,
            shap_seed=derive_seed(seed, 20_000),
            feature_augmentation=feature_augmentation or None,
            X_aug=X_aug,
            y_aug=y_aug,
            groups_aug=groups_aug,
        )
    cv_elapsed = perf_counter() - cv_start
    print(f"\nCV total time: {format_duration(cv_elapsed)}")

    print_cv_summary(cv_results)
    if feature_importance and cv_results.get('shap_importance'):
        print(
            f"\nGrouped out-of-fold SHAP used "
            f"{cv_results['shap_samples']} balanced held-out samples."
        )
        print_feature_importance(
            cv_results['shap_importance'],
            title="Feature Importance (grouped out-of-fold SHAP)",
        )

    if not export_artifacts and not evaluate_deployment:
        return 0, seed, cv_results

    # Train final model on full dataset for production export
    print("\nTraining final model on full dataset...")
    final_train_start = perf_counter()
    scaler = build_preprocessor(scaler_mode)
    fit_preprocessor(scaler, X, y=y, sample_context=sample_context)
    X_scaled = scaler.transform(X)
    y_final = y
    sw_final = sample_weights
    X_scaled, y_final, sw_final = _append_augmented_training_rows(
        X_scaled,
        y_final,
        scaler,
        X_aug,
        y_aug,
        groups_aug,
        eval_groups,
        sample_weight=sw_final,
    )
    feature_bounds = None
    if feature_augmentation:
        feature_bounds = normalized_feature_bounds(scaler, actual_feature_names)

    with suppress_stderr():
        model = train_model(
            X_scaled, y_final,
            hidden_layers=hidden_layers,
            max_epochs=DEFAULT_MAX_EPOCHS,
            fp_weight=fp_weight,
            sample_weight=sw_final,
            batch_size=batch_size,
            seed=derive_seed(seed, 10_000),
            feature_augmentation=feature_augmentation or None,
            feature_bounds=feature_bounds,
        )
    print(f"  Final training time: {format_duration(perf_counter() - final_train_start)}")

    if evaluate_deployment:
        print("\nEvaluating in-memory candidate on deployment safety recordings...")
        paired_gate = evaluate_paired_gate(
            model,
            scaler,
            actual_feature_names,
            roles=deployment_roles,
            allow_legacy_fallback=allow_legacy_gate_fallback,
        )
        quiet_gate = evaluate_quiet_gate(
            model,
            scaler,
            actual_feature_names,
            roles=deployment_roles,
        )
        cv_results['paired'] = paired_gate
        cv_results['quiet'] = quiet_gate
        print(
            f"  Paired: pass={paired_gate['pass_count']} "
            f"maxFP={paired_gate['max_fp_rate']:.2f}% "
            f"worstRecall={paired_gate['worst_chip_recall']:.2f}% "
            f"alarms={paired_gate.get('total_effective_alarms', 0)}"
        )
        if quiet_gate is None:
            print("  Quiet holdout: not configured")
        else:
            print(
                f"  Quiet holdout: {'pass' if quiet_gate['passed'] else 'fail'} "
                f"maxFP={quiet_gate['max_fp_rate']:.2f}% "
                f"alarms={quiet_gate['total_effective_alarms']}"
            )
        paired_total = len(paired_gate.get('by_chip', {}))
        if export_artifacts and (
            paired_total == 0
            or paired_gate['pass_count'] < paired_total
            or (quiet_gate is not None and not quiet_gate['passed'])
        ):
            print("Error: deployment safety gate failed; runtime artifacts were not exported")
            return 1, seed, cv_results
        if export_artifacts:
            try:
                baseline_paired = evaluate_exported_paired_gate(
                    roles=deployment_roles,
                    allow_legacy_fallback=allow_legacy_gate_fallback,
                )
            except (FileNotFoundError, ImportError, AttributeError) as exc:
                baseline_paired = None
                print(f"  Exported baseline unavailable ({exc}); using absolute paired gate")
            if baseline_paired is not None:
                print(
                    f"  Baseline paired: pass={baseline_paired['pass_count']} "
                    f"maxFP={baseline_paired['max_fp_rate']:.2f}% "
                    f"worstRecall={baseline_paired['worst_chip_recall']:.2f}%"
                )
                if not paired_result_non_regression(paired_gate, baseline_paired):
                    print(
                        "Error: candidate regresses the paired deployment gate; "
                        "runtime artifacts were not exported"
                    )
                    return 1, seed, cv_results

    if not export_artifacts:
        print("\nArtifacts unchanged (--no-export).")
        return 0, seed, cv_results

    regression_indices = select_regression_subset_indices(
        sample_context,
        max_samples=2048,
        block_stride=SEG_WINDOW_SIZE,
    )
    
    # Export models
    print("\nExporting model artifacts...")
    export_start = perf_counter()

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
    print("\nGenerated files:")
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
        'pass_count': int(sum(
            1 for row in rows
            if row['recall'] > ROBUSTNESS_TARGET_RECALL
            and row['fp_rate'] < ROBUSTNESS_TARGET_FP_RATE
            and row.get('effective_alarms', 0) == 0
        )),
        'mean_recall': float(np.mean([row['recall'] for row in rows])),
        'worst_chip_recall': float(np.min([row['recall'] for row in rows])),
        'mean_fp_rate': float(np.mean([row['fp_rate'] for row in rows])),
        'max_fp_rate': float(np.max([row['fp_rate'] for row in rows])),
        'mean_f1': float(np.mean([row['f1'] for row in rows])),
        'worst_chip_f1': float(np.min([row['f1'] for row in rows])),
        'total_fp': int(sum(row['fp'] for row in rows)),
        'total_fn': int(sum(row['fn'] for row in rows)),
        'total_effective_alarms': int(sum(row.get('effective_alarms', 0) for row in rows)),
        'max_effective_alarms': int(max(row.get('effective_alarms', 0) for row in rows)),
    }


def evaluate_idle_runtime_policy(probabilities, threshold=0.5):
    """Evaluate deploy-time cadence and hit filtering on an IDLE score stream."""
    return evaluate_idle_runtime_policy_states(np.asarray(probabilities) > threshold)


def evaluate_runtime_policy_evaluations(raw_motion_states):
    """Apply production hit filtering to states already sampled at eval ticks."""
    policy = RuntimeMotionPolicy(EVALUATION_INTERVAL, MOTION_ON_HITS, MOTION_OFF_HITS)
    effective_alarms = 0
    false_motion_evaluations = 0
    for raw_motion in raw_motion_states:
        raw_state = MotionState.MOTION if raw_motion else MotionState.IDLE
        effective_state, changed = policy.apply_state(raw_state)
        if changed and effective_state == MotionState.MOTION:
            effective_alarms += 1
        if effective_state == MotionState.MOTION:
            false_motion_evaluations += 1
    return {
        'effective_alarms': effective_alarms,
        'false_motion_evaluations': false_motion_evaluations,
    }


def _layer_arrays_from_model(model):
    """Return [(weights, biases, is_output), ...] arrays from a torch model."""
    raw_weights = extract_model_weights(model)
    layers = []
    for idx in range(0, len(raw_weights), 2):
        weights = raw_weights[idx]
        biases = raw_weights[idx + 1]
        is_output = idx == len(raw_weights) - 2
        layers.append((weights, biases, is_output))
    return layers


def _batch_predict_probabilities(features, center, scale, layers):
    """Vectorized forward pass matching the runtime inference math."""
    features = np.asarray(features, dtype=np.float32)
    if features.size == 0:
        return np.zeros(0, dtype=np.float64)
    activations = (features - center) / scale
    for weights, biases, is_output in layers:
        activations = activations @ weights + biases
        if not is_output:
            activations = activations.clip(min=0.0)

    logits = activations.reshape(-1).astype(np.float64)
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))
    probabilities[logits < -20.0] = 0.0
    probabilities[logits > 20.0] = 1.0
    return probabilities


class StreamingFeatureExtractor:
    """Compute runtime-equivalent feature vectors from a CSI packet stream."""

    def __init__(self, feature_names):
        self.feature_names = list(feature_names)
        self.context = SegmentationContext(
            window_size=SEG_WINDOW_SIZE,
            threshold=1.0,
            enable_lowpass=ENABLE_LOWPASS_FILTER,
            lowpass_cutoff=LOWPASS_CUTOFF,
            enable_hampel=ENABLE_HAMPEL_FILTER,
            hampel_window=HAMPEL_WINDOW,
            hampel_threshold=HAMPEL_THRESHOLD,
        )
        # L1 features share the Hampel-filtered delta stream used by training
        # and both runtimes; skip the tracker when the model has no L1 inputs.
        self.needs_l1_series = _needs_l1_series(self.feature_names)
        l1_capacity = max(2, SEG_WINDOW_SIZE - L1_DELTA_LAG)
        self.l1_tracker = (
            L1DeltaTracker(
                window_size=l1_capacity,
                lag=L1_DELTA_LAG,
                allocate_amplitude_buffer=False,
                enable_hampel=ENABLE_HAMPEL_FILTER,
                hampel_window=HAMPEL_WINDOW,
                hampel_threshold=HAMPEL_THRESHOLD,
            )
            if self.needs_l1_series else None
        )
        self.l1_series = [0.0] * l1_capacity if self.needs_l1_series else None

    def process_packet(self, csi_data):
        turbulence, amplitudes = self.context.calculate_spatial_turbulence(
            csi_data,
            DEFAULT_SUBCARRIERS,
            return_amplitudes=True,
        )
        self.context.add_turbulence(turbulence)
        if self.l1_tracker is not None:
            self.l1_tracker.process_amplitudes(amplitudes, len(amplitudes))
        if self.context.buffer_count < self.context.window_size:
            return None

        idx = self.context.buffer_index
        turb_list = (
            self.context.turbulence_buffer[idx:]
            + self.context.turbulence_buffer[:idx]
        )
        l1_count = (
            self.l1_tracker.copy_deltas_into(self.l1_series)
            if self.l1_tracker is not None else 0
        )
        return extract_features_by_name(
            turb_list,
            len(turb_list),
            feature_names=self.feature_names,
            l1_series=self.l1_series,
            l1_series_count=l1_count,
        )


class StreamingEvaluator:
    """Evaluate a trained model with the runtime-equivalent feature path."""

    def __init__(self, model, scaler, feature_names):
        self.extractor = StreamingFeatureExtractor(feature_names)
        self.center, self.scale = get_preprocessor_arrays(scaler)
        self.layers = _layer_arrays_from_model(model)

    def process_packet(self, csi_data):
        features = self.extractor.process_packet(csi_data)
        if features is None:
            return None
        probabilities = _batch_predict_probabilities(
            np.asarray(features, dtype=np.float32).reshape(1, -1),
            self.center,
            self.scale,
            self.layers,
        )
        return float(probabilities[0])


class ArrayStreamingEvaluator:
    """Runtime-equivalent evaluator backed by exported weight arrays."""

    def __init__(self, center, scale, layers, feature_names):
        self.extractor = StreamingFeatureExtractor(feature_names)
        self.center = center
        self.scale = scale
        self.layers = layers

    def process_packet(self, csi_data):
        features = self.extractor.process_packet(csi_data)
        if features is None:
            return None
        probabilities = _batch_predict_probabilities(
            np.asarray(features, dtype=np.float32).reshape(1, -1),
            self.center,
            self.scale,
            self.layers,
        )
        return float(probabilities[0])


def packet_csi_data(packet):
    """Return CSI data from a packet dictionary or a compact matrix row."""
    return packet['csi_data'] if isinstance(packet, dict) else packet


def evaluate_streaming_split(evaluator, static_presence_packets, motion_packets, threshold=0.5):
    """Evaluate a paired split at the production evaluation cadence."""
    warmup = SEG_WINDOW_SIZE
    static_presence_eval_count = 0
    motion_eval_count = 0
    static_presence_motion_packets = 0
    static_presence_motion_states = []
    motion_with_motion = 0
    motion_without_motion = 0

    cadence = make_evaluation_cadence(EVALUATION_INTERVAL)
    for i, pkt in enumerate(static_presence_packets):
        prob = evaluator.process_packet(packet_csi_data(pkt))
        if not cadence.note_evaluation_tick():
            continue
        if i < warmup or prob is None:
            continue
        static_presence_eval_count += 1
        is_motion = prob > threshold
        static_presence_motion_states.append(is_motion)
        if is_motion:
            static_presence_motion_packets += 1

    cadence = make_evaluation_cadence(EVALUATION_INTERVAL)
    for i, pkt in enumerate(motion_packets):
        prob = evaluator.process_packet(packet_csi_data(pkt))
        if not cadence.note_evaluation_tick():
            continue
        if i < warmup or prob is None:
            continue
        motion_eval_count += 1
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
        **evaluate_runtime_policy_evaluations(static_presence_motion_states),
    }


def evaluate_split(model, scaler, feature_names, static_presence_packets, motion_packets, threshold=0.5):
    """Evaluate a split with the same windowing path used at runtime."""
    evaluator = StreamingEvaluator(model, scaler, feature_names)
    return evaluate_streaming_split(
        evaluator,
        static_presence_packets,
        motion_packets,
        threshold=threshold,
    )


def evaluate_array_split(center, scale, layers, feature_names,
                         static_presence_packets, motion_packets, threshold=0.5):
    """Evaluate a split directly from exported runtime arrays."""
    evaluator = ArrayStreamingEvaluator(center, scale, layers, feature_names)
    return evaluate_streaming_split(
        evaluator,
        static_presence_packets,
        motion_packets,
        threshold=threshold,
    )


# Process-local cache for immutable paired validation NPZ streams.
_PAIRED_PACKET_CACHE = {}


def _load_npz_packets_cached(path):
    """Load NPZ packets once per process for repeated paired-gate evaluations."""
    key = str(Path(path).resolve())
    cached = _PAIRED_PACKET_CACHE.get(key)
    if cached is not None:
        return cached
    packets = load_npz_as_packets(path)
    _PAIRED_PACKET_CACHE[key] = packets
    return packets


def _iter_paired_chip_packets(chips=None, roles=('selection',),
                              allow_legacy_fallback=True):
    """Yield role-isolated real pairs, or the legacy latest pair as fallback."""
    dataset_info = load_dataset_info()
    files = dataset_info.get('files', {})
    roles = normalize_dataset_roles(roles, default=('selection',))
    motion_by_name = {
        str(entry.get('filename', '')): entry
        for entry in files.get('motion', [])
    }
    for chip in tuple(chips or DEFAULT_PAIRED_GATE_CHIPS):
        role_pairs = []
        for static_entry in files.get('static_presence', []):
            if str(static_entry.get('chip', '')).upper() != chip:
                continue
            if int(static_entry.get('subcarriers', 0) or 0) != 64:
                continue
            if bool(static_entry.get('synthetic')):
                continue
            role = str(static_entry.get('dataset_role', 'train')).lower()
            if role not in roles:
                continue
            motion_name = str(static_entry.get('optimal_pair_motion_file', ''))
            motion_entry = motion_by_name.get(motion_name)
            if motion_entry is None or bool(motion_entry.get('synthetic')):
                continue
            if str(motion_entry.get('dataset_role', 'train')).lower() != role:
                continue
            static_path = DATA_DIR / 'static_presence' / str(static_entry['filename'])
            motion_path = DATA_DIR / 'motion' / motion_name
            if static_path.exists() and motion_path.exists():
                role_pairs.append((role, static_path, motion_path))
        if role_pairs:
            for role, static_path, motion_path in sorted(role_pairs):
                key = f"{chip}:{role}:{static_path.name}"
                yield (
                    key,
                    _load_npz_packets_cached(static_path),
                    _load_npz_packets_cached(motion_path),
                )
            continue
        if not allow_legacy_fallback:
            continue

        # Backward-compatible fallback for repositories that have not assigned
        # roles yet. Keep it explicitly real and train-only so a newly generated
        # synthetic derivative can never become the deployment replay by being
        # the latest timestamped pair.
        legacy_pairs = []
        for static_entry in files.get('static_presence', []):
            if str(static_entry.get('chip', '')).upper() != chip:
                continue
            if int(static_entry.get('subcarriers', 0) or 0) != 64:
                continue
            if bool(static_entry.get('synthetic')):
                continue
            if str(static_entry.get('dataset_role', 'train')).lower() != 'train':
                continue
            motion_name = str(static_entry.get('optimal_pair_motion_file', ''))
            motion_entry = motion_by_name.get(motion_name)
            if motion_entry is None or bool(motion_entry.get('synthetic')):
                continue
            if str(motion_entry.get('dataset_role', 'train')).lower() != 'train':
                continue
            static_path = DATA_DIR / 'static_presence' / str(static_entry.get('filename', ''))
            motion_path = DATA_DIR / 'motion' / motion_name
            if static_path.exists() and motion_path.exists():
                sort_key = (
                    str(static_entry.get('collected_at', '')),
                    static_path.name,
                )
                legacy_pairs.append((sort_key, static_path, motion_path))
        if not legacy_pairs:
            continue
        _, static_path, motion_path = max(legacy_pairs, key=lambda row: row[0])
        yield (
            chip,
            _load_npz_packets_cached(static_path),
            _load_npz_packets_cached(motion_path),
        )


def evaluate_paired_gate(model, scaler, feature_names, threshold=0.5, chips=None,
                         roles=('selection',), allow_legacy_fallback=True):
    """Evaluate a candidate on the paired validation datasets."""
    by_chip = {}
    for chip, static_presence_packets, motion_packets in _iter_paired_chip_packets(
        chips,
        roles=roles,
        allow_legacy_fallback=allow_legacy_fallback,
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


def _load_exported_model_arrays():
    """Load exported MicroPython weights as inference-ready arrays."""
    module = load_exported_ml_weights()
    center = np.asarray(module.FEATURE_MEAN, dtype=np.float64)
    scale = np.asarray(module.FEATURE_SCALE, dtype=np.float64)
    layers = []
    for idx, (weights, biases) in enumerate(zip(module.WEIGHTS, module.BIASES)):
        layers.append((
            np.asarray(weights, dtype=np.float32),
            np.asarray(biases, dtype=np.float32),
            idx == len(module.WEIGHTS) - 1,
        ))
    return list(module.FEATURE_NAMES), center, scale, layers


def evaluate_exported_paired_gate(threshold=0.5, chips=None,
                                  roles=('selection', 'holdout'),
                                  allow_legacy_fallback=True):
    """Evaluate exported runtime arrays on the paired validation datasets."""
    feature_names, center, scale, layers = _load_exported_model_arrays()
    by_chip = {}
    for chip, static_presence_packets, motion_packets in _iter_paired_chip_packets(
        chips,
        roles=roles,
        allow_legacy_fallback=allow_legacy_fallback,
    ):
        by_chip[chip] = evaluate_array_split(
            center,
            scale,
            layers,
            feature_names,
            static_presence_packets,
            motion_packets,
            threshold=threshold,
        )
    return summarize_gate(by_chip)


def _iter_quiet_gate_packets(roles=('selection', 'holdout')):
    """Yield real empty recordings explicitly reserved for selection/holdout."""
    dataset_info = load_dataset_info()
    roles = normalize_dataset_roles(roles, default=('selection', 'holdout'))
    for entry in dataset_info.get('files', {}).get('empty', []):
        role = str(entry.get('dataset_role', 'train')).lower()
        if role not in roles or bool(entry.get('synthetic')):
            continue
        path = DATA_DIR / 'empty' / str(entry.get('filename', ''))
        if path.exists():
            chip = str(entry.get('chip', 'unknown')).upper()
            yield f"{chip}:{role}:{path.name}", _load_npz_packets_cached(path)


def evaluate_idle_streaming(evaluator, packets, threshold=0.5):
    """Evaluate one quiet stream at production cadence and hit filtering."""
    raw_states = []
    cadence = make_evaluation_cadence(EVALUATION_INTERVAL)
    for index, packet in enumerate(packets):
        probability = evaluator.process_packet(packet_csi_data(packet))
        if not cadence.note_evaluation_tick():
            continue
        if index < SEG_WINDOW_SIZE or probability is None:
            continue
        raw_states.append(probability > threshold)
    fp = int(sum(raw_states))
    count = len(raw_states)
    return {
        'fp': fp,
        'evaluations': count,
        'fp_rate': fp / count * 100.0 if count else 0.0,
        **evaluate_runtime_policy_evaluations(raw_states),
    }


def summarize_quiet_gate(by_dataset):
    """Aggregate explicitly reserved empty-room safety replays."""
    if not by_dataset:
        return None
    rows = list(by_dataset.values())
    return {
        'by_dataset': by_dataset,
        'max_fp_rate': float(max(row['fp_rate'] for row in rows)),
        'total_effective_alarms': int(sum(row['effective_alarms'] for row in rows)),
        'max_effective_alarms': int(max(row['effective_alarms'] for row in rows)),
        'passed': all(
            row['fp_rate'] < ROBUSTNESS_TARGET_FP_RATE
            and row['effective_alarms'] == 0
            for row in rows
        ),
    }


def evaluate_quiet_gate(model, scaler, feature_names, threshold=0.5,
                        roles=('selection', 'holdout')):
    """Evaluate an in-memory candidate on reserved real empty recordings."""
    by_dataset = {}
    for key, packets in _iter_quiet_gate_packets(roles=roles):
        by_dataset[key] = evaluate_idle_streaming(
            StreamingEvaluator(model, scaler, feature_names),
            packets,
            threshold=threshold,
        )
    return summarize_quiet_gate(by_dataset)


def evaluate_exported_quiet_gate(threshold=0.5, roles=('selection', 'holdout')):
    """Evaluate exported arrays on reserved real empty recordings."""
    feature_names, center, scale, layers = _load_exported_model_arrays()
    by_dataset = {}
    for key, packets in _iter_quiet_gate_packets(roles=roles):
        by_dataset[key] = evaluate_idle_streaming(
            ArrayStreamingEvaluator(center, scale, layers, feature_names),
            packets,
            threshold=threshold,
        )
    return summarize_quiet_gate(by_dataset)


@dataclass
class ExportedMLGateResult:
    """Verification result for exported ML artifacts (paired gate)."""

    paired_returncode: int
    paired_output: str
    paired_metrics: dict | None = None
    quiet_metrics: dict | None = None

    @property
    def available(self):
        return self.paired_metrics is not None or self.quiet_metrics is not None

    @property
    def passed(self):
        return (
            self.available
            and (self.paired_metrics is None or self.paired_returncode == 0)
            and (self.quiet_metrics is None or self.quiet_metrics.get('passed', False))
        )


def run_exported_ml_gates(roles=('selection', 'holdout'),
                          allow_legacy_fallback=True):
    """Run paired and explicitly reserved quiet gates for exported artifacts."""
    paired_metrics = evaluate_exported_paired_gate(
        roles=roles,
        allow_legacy_fallback=allow_legacy_fallback,
    )
    quiet_metrics = evaluate_exported_quiet_gate(roles=roles)
    paired_total = len(paired_metrics.get('by_chip', {})) if paired_metrics else 0
    paired_rc = 0 if paired_total and paired_metrics['pass_count'] == paired_total else 1
    return ExportedMLGateResult(
        paired_returncode=paired_rc,
        paired_output="",
        paired_metrics=paired_metrics,
        quiet_metrics=quiet_metrics,
    )


def _paired_gate_key(paired_metrics):
    """Ranking key for paired real-data gate results."""
    if paired_metrics is None:
        return None
    return (
        paired_metrics.get('pass_count', 0),
        -paired_metrics.get('total_effective_alarms', float('inf')),
        -paired_metrics.get('max_fp_rate', float('inf')),
        paired_metrics.get('worst_chip_recall', -float('inf')),
        paired_metrics.get('worst_chip_f1', -float('inf')),
        paired_metrics.get('mean_f1', -float('inf')),
        paired_metrics.get('mean_recall', -float('inf')),
    )


def _combined_candidate_key(cv_metrics, paired_metrics=None):
    """
    Final selection key.

    Grouped OOF robustness leads ranking after deployment safety passes.
    """
    cv_key = build_candidate_key(cv_metrics)
    paired_key = _paired_gate_key(paired_metrics)
    if paired_key is None:
        return cv_key
    return cv_key + paired_key


def _format_exported_gate_summary(gate):
    """Build a short one-line summary for exported-artifact verification."""
    if gate is None:
        return "exported_gates=not_run"
    metrics = gate.paired_metrics
    if metrics is None:
        summary = "paired=not_configured"
    else:
        paired = (
            "paired=pass"
            if gate.paired_returncode == 0
            else f"paired=fail({gate.paired_returncode})"
        )
        summary = (
            f"{paired} maxFP={metrics.get('max_fp_rate', 0.0):.2f}% "
            f"worstRecall={metrics.get('worst_chip_recall', 0.0):.2f}% "
            f"worstF1={metrics.get('worst_chip_f1', 0.0):.2f}% "
            f"alarms={metrics.get('total_effective_alarms', 0)}"
        )
    if gate.quiet_metrics is None:
        return summary + " quiet=not_configured"
    return (
        summary
        + f" quietMaxFP={gate.quiet_metrics.get('max_fp_rate', 0.0):.2f}%"
        + f" quietAlarms={gate.quiet_metrics.get('total_effective_alarms', 0)}"
    )


def _candidate_beats_baseline(candidate_cv, candidate_gate, static_presence_cv, static_presence_gate):
    """Require deployment safety plus robust grouped-CV improvement."""
    if candidate_gate is None or static_presence_gate is None:
        return False
    if not candidate_gate.passed or not static_presence_gate.passed:
        return False
    if (
        candidate_gate.paired_metrics is not None
        and static_presence_gate.paired_metrics is not None
        and not paired_result_non_regression(
            candidate_gate.paired_metrics,
            static_presence_gate.paired_metrics,
        )
    ):
        return False
    return compare_robust_cv(candidate_cv, static_presence_cv)['passed']


def _format_candidate_comparison(candidate_cv, baseline_cv):
    """Format material CV deltas and equivalence decisions for seed search."""
    comparison = compare_robust_cv(candidate_cv, baseline_cv)
    parts = []
    for check in comparison['checks']:
        state = 'regression' if check['regressed'] else 'improvement' if check['improved'] else 'tie'
        parts.append(
            f"{check['label']} {check['delta']:+.2f}pp "
            f"(margin {check['margin']:.2f}, {state})"
        )
    return comparison, '; '.join(parts)


def _search_candidate_key(cv_metrics, gate=None):
    """Ranking key for broken-baseline seed search fallback."""
    gate_passed = 1 if gate is not None and gate.passed else 0
    paired_passed = 1 if gate is not None and gate.paired_returncode == 0 else 0
    paired_key = _paired_gate_key(gate.paired_metrics if gate is not None else None)
    if paired_key is None:
        paired_key = (-float('inf'),) * 7
    return (
        gate_passed,
        paired_passed,
    ) + tuple(paired_key) + build_candidate_key(cv_metrics)


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
                            use_cache=True, augment=False):
    """
    Train all requested seeds and keep the strongest robust improvement.

    Baseline is recomputed using the seed embedded in the current exported model.
    Deployment replays are safety gates. Grouped OOF worst/tail metrics lead
    ranking with one-event equivalence margins.

    When the current exported baseline fails the paired gate, the command still
    evaluates all MAX_TRIALS candidates, but only a candidate that restores the
    deployment safety gate can replace it.
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
    feature_augmentation, packet_augmentation = resolve_training_augmentation(augment)
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
    print("  SEED SEARCH (evaluate all candidates)")
    print("=" * 70)
    print(f"Max trials: {max_trials}")
    print(f"FP weight: {fp_weight}")
    print(f"Scaler: {scaler_mode}")
    print(f"Batch size: {batch_size}")
    print(
        "Augmentation: "
        f"{format_augmentation_config(feature_augmentation, packet_augmentation)}"
    )
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

    train_kwargs = {
        'fp_weight': fp_weight,
        'feature_names': feature_names,
        'feature_importance': False,
        'ablation': False,
        'hidden_layers': hidden_layers,
        'scaler_mode': scaler_mode,
        'batch_size': batch_size,
        'environment_filter': environment_filter,
        'excluded_chips': excluded_chips,
        'positive_chip_boost': positive_chip_boost,
        'use_cache': use_cache,
        'augment': bool(augment),
        # Candidate selection may reuse selection recordings, but the holdout
        # stays sealed until exactly one winner has been chosen.
        'deployment_roles': ('selection',),
        'allow_legacy_gate_fallback': True,
    }

    print(f"\nEvaluating current model baseline with seed {static_presence_seed}...")
    static_presence_rc, _, static_presence_metrics = train_all(
        seed=static_presence_seed,
        export_artifacts=False,
        **train_kwargs,
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
    static_presence_gate = run_exported_ml_gates(roles=('selection',))
    print(f"Baseline exported ML gates: {_format_exported_gate_summary(static_presence_gate)}")
    baseline_holdout_gate = run_exported_ml_gates(
        roles=('holdout',),
        allow_legacy_fallback=False,
    )
    if baseline_holdout_gate.available:
        print(
            "Reserved baseline holdout captured for final non-regression: "
            f"{_format_exported_gate_summary(baseline_holdout_gate)}"
        )
    else:
        print("Reserved holdout: not configured")
    broken_baseline_mode = not static_presence_gate.passed
    if broken_baseline_mode:
        print(
            "Warning: baseline paired gate failed; "
            "running all trials and ranking candidates against the broken baseline"
        )

    backup_dir, saved_files = _backup_artifacts()
    print(f"Artifacts backup: {backup_dir}")

    trial_summaries = []
    improved = False
    improved_seed = None
    improved_metrics = None
    improved_gate = None
    best_candidate_backup_dir = None
    best_candidate_saved_files = None
    best_search_key = _search_candidate_key(static_presence_metrics, static_presence_gate)

    for idx in range(1, max_trials + 1):
        trial_seed = generate_random_training_seed()
        print(f"\n[{idx}/{max_trials}] Training with auto-generated seed {trial_seed}")
        # One train_all per trial: CV once, then final fit + export for the paired gate.
        # Pass an explicit random seed so resolve_training_seed does not reuse the
        # currently exported model seed on every trial.
        export_rc, used_seed, final_metrics = train_all(
            seed=trial_seed,
            export_artifacts=True,
            **train_kwargs,
        )
        if export_rc != 0 or final_metrics is None:
            print(f"  Candidate train/export failed (exit={export_rc})")
            _restore_artifacts(saved_files)
            trial_summaries.append((used_seed, final_metrics or {}, None, 'export_failed'))
            continue

        session_summary = final_metrics.get('group_reports', {}).get('session_group', {}).get('worst_recall', {})
        fp_summary = final_metrics.get('group_reports', {}).get('session_group', {}).get('worst_fp_rate', {})
        print(
            f"  Result: session_min_recall={session_summary.get('recall', 0.0):.1f}% "
            f"session_max_fp={fp_summary.get('fp_rate', 0.0):.1f}% "
            f"blocked_oof_f1={final_metrics['oof_f1']:.1f}%"
        )

        candidate_gate = run_exported_ml_gates(roles=('selection',))
        print(f"  Exported ML gates: {_format_exported_gate_summary(candidate_gate)}")
        if not candidate_gate.passed and candidate_gate.paired_output.strip():
            print(candidate_gate.paired_output.strip())

        if broken_baseline_mode:
            status = 'ranked_rejected'
            candidate_search_key = _search_candidate_key(final_metrics, candidate_gate)
            if candidate_gate.passed and candidate_search_key > best_search_key:
                improved = True
                improved_seed = used_seed
                improved_metrics = final_metrics
                improved_gate = candidate_gate
                best_search_key = candidate_search_key
                if best_candidate_backup_dir is not None:
                    shutil.rmtree(best_candidate_backup_dir, ignore_errors=True)
                best_candidate_backup_dir, best_candidate_saved_files = _backup_artifacts()
                status = 'ranked_best'
                print("  Broken baseline mode: current best candidate updated")
            elif not candidate_gate.passed:
                print("  Broken baseline mode: candidate still fails deployment safety")
            else:
                print("  Broken baseline mode: candidate did not beat current best")
            trial_summaries.append((used_seed, final_metrics, candidate_gate, status))
            _restore_artifacts(saved_files)
            continue

        comparison, comparison_text = _format_candidate_comparison(
            final_metrics,
            static_presence_metrics,
        )
        print(f"  Robust CV comparison: {comparison_text}")
        status = 'robust_rejected'
        if _candidate_beats_baseline(
            final_metrics,
            candidate_gate,
            static_presence_metrics,
            static_presence_gate,
        ):
            candidate_key = _combined_candidate_key(
                final_metrics,
                candidate_gate.paired_metrics,
            )
            if not improved or candidate_key > best_search_key:
                improved = True
                improved_seed = used_seed
                improved_metrics = final_metrics
                improved_gate = candidate_gate
                best_search_key = candidate_key
                if best_candidate_backup_dir is not None:
                    shutil.rmtree(best_candidate_backup_dir, ignore_errors=True)
                best_candidate_backup_dir, best_candidate_saved_files = _backup_artifacts()
                status = 'robust_best'
                print("  Robust improvement: current best candidate updated")
            else:
                status = 'robust_eligible'
                print("  Robust improvement: eligible, but not the current best")
        elif not candidate_gate.passed:
            print("  Deployment safety gate rejected candidate")
        elif not paired_result_non_regression(
            candidate_gate.paired_metrics,
            static_presence_gate.paired_metrics,
        ):
            print("  Per-recording paired non-regression rejected candidate")
        elif comparison['regressions']:
            print("  Robust CV rejected candidate due to material regression")
        else:
            print("  Robust CV found no material improvement")
        trial_summaries.append((used_seed, final_metrics, candidate_gate, status))
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
            f"blockedOOF={metrics.get('oof_f1', 0.0):.1f}% | {status} | "
            f"{_format_exported_gate_summary(gate)}"
        )

    if improved:
        if best_candidate_saved_files is not None:
            _restore_artifacts(best_candidate_saved_files)
            final_holdout_gate = run_exported_ml_gates(
                roles=('holdout',),
                allow_legacy_fallback=False,
            )
            if final_holdout_gate.available:
                print(
                    "Final reserved holdout: "
                    f"{_format_exported_gate_summary(final_holdout_gate)}"
                )
                holdout_non_regression = (
                    baseline_holdout_gate.paired_metrics is None
                    or final_holdout_gate.paired_metrics is None
                    or paired_result_non_regression(
                        final_holdout_gate.paired_metrics,
                        baseline_holdout_gate.paired_metrics,
                    )
                )
                if not final_holdout_gate.passed or not holdout_non_regression:
                    _restore_artifacts(saved_files)
                    print(
                        "Final reserved holdout rejected the selected candidate; "
                        "current artifacts were restored"
                    )
                    return 1
            print(
                f"\nSelected seed after full robust ranking: {improved_seed} "
                f"(blocked_oof_f1={improved_metrics['oof_f1']:.1f}%, "
                f"{_format_exported_gate_summary(improved_gate)})"
            )
            return 0

    if broken_baseline_mode:
        print("\nNo candidate beat the current broken baseline; current artifacts remain unchanged")
        return 1

    print("\nNo improvement found within max trials; current artifacts remain unchanged")
    return 1


def write_json_results(path, payload):
    """Write a JSON experiment payload."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_value))


def _json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def run_robustness_experiment(output_path=DEFAULT_ROBUSTNESS_EXPERIMENT_OUTPUT,
                              use_cache=True, hidden_layers=None,
                              fp_weight=DEFAULT_FP_WEIGHT,
                              batch_size=DEFAULT_BATCH_SIZE,
                              augmentation_only=False):
    """Run the non-destructive staged Core-6 robustness campaign."""
    ensure_torch_available()
    hidden_layers = list(hidden_layers or DEFAULT_HIDDEN_LAYERS)
    print("\n" + "=" * 70)
    print("  CORE-6 ROBUSTNESS CAMPAIGN")
    print("=" * 70)
    print("Artifacts: unchanged")
    print(f"Output: {output_path}")
    print(f"Architecture: {' -> '.join(map(str, [len(TRAINING_FEATURES)] + hidden_layers + [1]))}")

    matrix, _ = load_training_matrix(
        feature_names=TRAINING_FEATURES,
        use_cache=use_cache,
    )
    payload = {
        'config': {
            'screening_seed': ROBUSTNESS_SCREENING_SEED,
            'filter_seeds': list(ROBUSTNESS_FILTER_SEEDS),
            'final_seeds': list(ROBUSTNESS_FINAL_SEEDS),
            'target_recall': ROBUSTNESS_TARGET_RECALL,
            'target_fp_rate': ROBUSTNESS_TARGET_FP_RATE,
            'hidden_layers': hidden_layers,
            'batch_size': int(batch_size),
            'fp_weight': float(fp_weight),
            'feature_names': list(TRAINING_FEATURES),
            'promote': False,
            'augmentation_only': bool(augmentation_only),
        },
        'stages': [],
        'filter': [],
        'final': [],
        'decision': None,
    }
    run_cache = {}

    def candidate_key(candidate):
        return json.dumps(candidate, sort_keys=True, separators=(',', ':'))

    def evaluate(candidate, seed):
        key = (candidate_key(candidate), int(seed))
        if key in run_cache:
            return run_cache[key]
        print(f"\n== {candidate['name']} | seed {seed} ==")
        augmented = None
        if candidate['packet_augmentation']:
            augmented, _ = load_training_matrix(
                feature_names=TRAINING_FEATURES,
                use_cache=use_cache,
                packet_augmentation=candidate['packet_augmentation'],
                augmentation_seed=seed,
            )
        run = evaluate_robustness_candidate(
            candidate,
            seed,
            matrix,
            augmented_matrix=augmented,
            hidden_layers=hidden_layers,
            fp_weight=fp_weight,
            batch_size=batch_size,
        )
        run_cache[key] = run
        rank = run['rank_key']
        print(
            f"  holdout passes={-rank[0]}/{run['holdout_count']} | worst FP={rank[1]:.1f}% | "
            f"worst recall={-rank[2]:.1f}%"
        )
        return run

    def screen_stage(name, candidates):
        stage = {'name': name, 'runs': []}
        payload['stages'].append(stage)
        for candidate in candidates:
            stage['runs'].append(evaluate(candidate, ROBUSTNESS_SCREENING_SEED))
            write_json_results(output_path, payload)
        return sorted(stage['runs'], key=robustness_run_rank_key)

    baseline = _robustness_candidate('baseline_standard')
    if augmentation_only:
        scaler_runs = screen_stage('baseline', [baseline])
        l1_runs = []
        best_pre_augmentation = baseline
    else:
        scaler_candidates = [
            baseline,
            _robustness_candidate('scaler_robust', scaler='robust'),
            _robustness_candidate(
                'scaler_session_balanced_robust', scaler='session_balanced_robust'),
        ]
        scaler_runs = screen_stage('scalers', scaler_candidates)
        best_scaler = scaler_runs[0]['candidate']

        l1_candidates = []
        for variant in ('l1_std_relative', 'l1_waveform_relative', 'l1_both_relative'):
            l1_candidates.append(_robustness_candidate(
                f"{best_scaler['name']}__{variant}",
                scaler=best_scaler['scaler'],
                l1_variant=variant,
            ))
        l1_runs = screen_stage('l1_normalization', l1_candidates)
        best_pre_augmentation = min(
            [next(run for run in scaler_runs if run['candidate'] == best_scaler)] + l1_runs,
            key=robustness_run_rank_key,
        )['candidate']

    feature_candidates = [
        _robustness_candidate(
            f"{best_pre_augmentation['name']}__feature_{suffix}",
            scaler=best_pre_augmentation['scaler'],
            l1_variant=best_pre_augmentation['l1_variant'],
            feature_augmentation=config,
        )
        for suffix, config in ROBUSTNESS_FEATURE_AUGMENTATIONS
    ]
    feature_runs = screen_stage('feature_augmentation', feature_candidates)
    pre_packet_run = evaluate(best_pre_augmentation, ROBUSTNESS_SCREENING_SEED)
    top_feature_candidates = [
        run['candidate']
        for run in sorted([pre_packet_run] + feature_runs, key=robustness_run_rank_key)[:2]
    ]

    packet_candidates = []
    for parent in top_feature_candidates:
        for suffix, config in ROBUSTNESS_PACKET_AUGMENTATIONS:
            packet_candidates.append(_robustness_candidate(
                f"{parent['name']}__packet_{suffix}",
                scaler=parent['scaler'],
                l1_variant=parent['l1_variant'],
                feature_augmentation=parent['feature_augmentation'],
                packet_augmentation=config,
            ))
    packet_runs = screen_stage('packet_augmentation', packet_candidates)

    all_screening = scaler_runs + l1_runs + feature_runs + packet_runs
    unique_screening = {}
    for run in all_screening:
        unique_screening[candidate_key(run['candidate'])] = run
    challengers = [
        run for run in sorted(unique_screening.values(), key=robustness_run_rank_key)
        if run['candidate']['name'] != baseline['name']
    ][:2]
    filter_candidates = [baseline] + [run['candidate'] for run in challengers]
    filter_summaries = []
    for candidate in filter_candidates:
        runs = [evaluate(candidate, seed) for seed in ROBUSTNESS_FILTER_SEEDS]
        summary = aggregate_robustness_runs(candidate, runs)
        filter_summaries.append(summary)
        payload['filter'] = filter_summaries
        write_json_results(output_path, payload)

    baseline_filter = next(
        item for item in filter_summaries if item['candidate']['name'] == baseline['name'])
    best_challenger = min(
        (item for item in filter_summaries if item['candidate']['name'] != baseline['name']),
        key=robustness_summary_rank_key,
    )
    final_candidates = [baseline, best_challenger['candidate']]
    final_summaries = []
    for candidate in final_candidates:
        runs = [evaluate(candidate, seed) for seed in ROBUSTNESS_FINAL_SEEDS]
        summary = aggregate_robustness_runs(candidate, runs)
        final_summaries.append(summary)
        payload['final'] = final_summaries
        write_json_results(output_path, payload)

    baseline_final = next(
        item for item in final_summaries if item['candidate']['name'] == baseline['name'])
    challenger_final = next(
        item for item in final_summaries if item['candidate']['name'] != baseline['name'])
    all_medians_pass = (
        challenger_final['median_pass_count'] == challenger_final['holdout_count'])
    four_of_five_pass = all(row['seed_passes'] >= 4 for row in challenger_final['folds'])
    non_regression = (
        challenger_final['median_pass_count'] >= baseline_final['median_pass_count'])
    generalization_qualified = bool(all_medians_pass and four_of_five_pass and non_regression)
    payload['decision'] = {
        'winner': challenger_final['candidate']['name']
            if robustness_summary_rank_key(challenger_final) < robustness_summary_rank_key(baseline_final)
            else baseline_final['candidate']['name'],
        'generalization_qualified': generalization_qualified,
        'all_holdout_medians_pass': bool(all_medians_pass),
        'four_of_five_per_holdout': bool(four_of_five_pass),
        'baseline_non_regression': bool(non_regression),
        'deployment_validation': 'required',
        'required_checks': ['paired_gate', 'gain_stress_gate', 'long_recordings', 'python_cpp_parity'],
        'artifacts_changed': False,
    }
    payload['completed_at'] = datetime.now().astimezone().isoformat()
    write_json_results(output_path, payload)
    print("\n" + "=" * 70)
    print("  ROBUSTNESS CAMPAIGN DECISION")
    print("=" * 70)
    print(f"Winner: {payload['decision']['winner']}")
    print(f"Generalization qualified: {generalization_qualified}")
    print("Deployment gates remain required; runtime artifacts unchanged.")
    return payload


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
    """Sort safely passing runs by robust grouped-CV performance."""
    robust_key = tuple(-value for value in build_candidate_key(result['cv']))
    return (
        -result['paired']['pass_count'],
        result['paired'].get('total_effective_alarms', 0),
        *robust_key,
        result['paired']['max_fp_rate'],
        -result['paired']['worst_chip_recall'],
        -result['paired']['worst_chip_f1'],
        result['params'],
    )


def aggregate_architecture_runs(name, runs):
    """Aggregate multi-seed runs for one architecture."""
    template = runs[0]
    candidate_keys = np.asarray(
        [build_candidate_key(run['cv']) for run in runs],
        dtype=np.float64,
    )
    return {
        'name': name,
        'layers': list(template['layers']),
        'architecture': template['architecture'],
        'params': int(template['params']),
        'weight_kb': float(template['weight_kb']),
        'flops': int(template['flops']),
        'seeds': [int(run['seed']) for run in runs],
        'median_paired_pass_count': float(np.median([run['paired']['pass_count'] for run in runs])),
        'median_paired_effective_alarms': float(np.median([
            run['paired'].get('total_effective_alarms', 0) for run in runs
        ])),
        'median_paired_max_fp_rate': float(np.median([run['paired']['max_fp_rate'] for run in runs])),
        'median_paired_worst_chip_recall': float(np.median([
            run['paired']['worst_chip_recall'] for run in runs
        ])),
        'median_paired_worst_chip_f1': float(np.median([run['paired']['worst_chip_f1'] for run in runs])),
        'median_oof_f1': float(np.median([run['cv']['oof_f1'] for run in runs])),
        'median_cv_candidate_key': [
            float(value) for value in np.median(candidate_keys, axis=0)
        ],
        'best_single_run': min(runs, key=architecture_campaign_rank_key),
        'runs': runs,
    }


def aggregate_architecture_rank_key(summary):
    """Sort key for aggregated architecture summaries (lower is better)."""
    robust_key = tuple(-value for value in summary.get(
        'median_cv_candidate_key',
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, summary['median_oof_f1'], 0.0),
    ))
    return (
        -summary['median_paired_pass_count'],
        summary.get('median_paired_effective_alarms', 0.0),
        *robust_key,
        summary['median_paired_max_fp_rate'],
        -summary['median_paired_worst_chip_recall'],
        -summary['median_paired_worst_chip_f1'],
        summary['params'],
    )


def paired_non_regression(candidate, baseline):
    """Treat paired validation as a non-regression constraint."""
    return (
        candidate['median_paired_pass_count'] >= baseline['median_paired_pass_count']
        and candidate['median_paired_max_fp_rate'] <= baseline['median_paired_max_fp_rate'] + 1e-6
        and candidate['median_paired_worst_chip_recall']
        >= baseline['median_paired_worst_chip_recall'] - 0.25
        and candidate['median_paired_worst_chip_f1']
        >= baseline['median_paired_worst_chip_f1'] - 0.25
    )


def architecture_candidate_beats_baseline(candidate, baseline):
    """Promote only stable paired improvements that do not regress validation."""
    if candidate['name'] == baseline['name']:
        return True
    if not paired_non_regression(candidate, baseline):
        return False
    return aggregate_architecture_rank_key(candidate) < aggregate_architecture_rank_key(baseline)


def evaluate_architecture_candidate(name, hidden_layers, seed, dataset, scaler_mode, batch_size, fp_weight):
    """Train and evaluate one architecture on CV and the paired gate."""
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
    fit_preprocessor(
        scaler,
        dataset['X'],
        y=dataset['y'],
        sample_context=dataset['sample_context'],
    )
    X_scaled = scaler.transform(dataset['X'])
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
    result = {
        'name': name,
        'seed': int(seed),
        'fp_weight': float(fp_weight),
        'layers': list(hidden_layers),
        'architecture': ' -> '.join(map(str, stats['layer_sizes'])),
        'params': int(stats['params']),
        'weight_kb': float(stats['weight_kb']),
        'flops': int(stats['flops']),
        'inference_us': float(inference_us),
        'cv': slim_cv_result(cv),
        'paired': paired,
    }
    print(
        f"{name} | OOF={result['cv']['oof_f1']:.1f}% | "
        f"paired pass={paired['pass_count']} maxFP={paired['max_fp_rate']:.1f}% "
        f"worstRecall={paired['worst_chip_recall']:.1f}% "
        f"worstF1={paired['worst_chip_f1']:.1f}% | "
        f"inf={inference_us:.1f} us"
    )
    return result


def build_feature_ablation_dataset(dataset, feature_name):
    """Return a dataset view with one named feature removed."""
    feature_names = list(dataset['feature_names'])
    if feature_name not in feature_names:
        raise ValueError(
            f"Unknown ablation feature '{feature_name}'. "
            f"Available features: {', '.join(feature_names)}"
        )
    feature_idx = feature_names.index(feature_name)
    candidate = dict(dataset)
    candidate['X'] = np.delete(dataset['X'], feature_idx, axis=1)
    candidate['feature_names'] = [
        name for idx, name in enumerate(feature_names) if idx != feature_idx
    ]
    return candidate


def _print_feature_ablation_comparison(baseline, candidate):
    """Print the CV and paired real-data deltas for a targeted feature ablation."""
    rows = (
        ('Blocked OOF F1', baseline['cv']['oof_f1'], candidate['cv']['oof_f1'], '%'),
        ('Fold F1', baseline['cv']['f1_mean'], candidate['cv']['f1_mean'], '%'),
        ('Fold recall', baseline['cv']['recall_mean'], candidate['cv']['recall_mean'], '%'),
        ('Fold FP rate', baseline['cv']['fp_rate_mean'], candidate['cv']['fp_rate_mean'], '%'),
        (
            'Worst-session recall',
            baseline['cv']['worst_session_recall'],
            candidate['cv']['worst_session_recall'],
            '%',
        ),
        (
            'Worst-session FP rate',
            baseline['cv']['worst_session_fp_rate'],
            candidate['cv']['worst_session_fp_rate'],
            '%',
        ),
        ('Paired mean F1', baseline['paired']['mean_f1'], candidate['paired']['mean_f1'], '%'),
        ('Paired worst-chip F1', baseline['paired']['worst_chip_f1'], candidate['paired']['worst_chip_f1'], '%'),
        ('Paired max FP rate', baseline['paired']['max_fp_rate'], candidate['paired']['max_fp_rate'], '%'),
        (
            'Paired worst-chip recall',
            baseline['paired']['worst_chip_recall'],
            candidate['paired']['worst_chip_recall'],
            '%',
        ),
    )
    print("\n" + "=" * 82)
    print("  TARGETED FEATURE ABLATION COMPARISON")
    print("=" * 82)
    print(f"{'Metric':<29} {'Core-6':>14} {'Candidate':>14} {'Delta':>14}")
    print("-" * 82)
    for label, baseline_value, candidate_value, suffix in rows:
        delta = candidate_value - baseline_value
        if suffix:
            print(
                f"{label:<29} {baseline_value:>13.2f}{suffix} "
                f"{candidate_value:>13.2f}{suffix} {delta:>+13.2f}{suffix}"
            )
        else:
            print(
                f"{label:<29} {int(baseline_value):>14d} "
                f"{int(candidate_value):>14d} {int(delta):>+14d}"
            )
    print("-" * 82)


def _feature_ablation_rank_key(result):
    """Rank one targeted ablation result with paired metrics first."""
    cv = result['cv']
    return _paired_gate_key(result['paired']) + (
        cv['worst_session_recall'],
        cv['worst_chip_recall'],
        -cv['worst_session_fp_rate'],
        cv['oof_f1'],
        cv['f1_mean'],
    )


def paired_result_non_regression(candidate, baseline, tolerance=0.25):
    """Preserve each paired replay within one changed-evaluation margin."""
    if candidate['pass_count'] != baseline['pass_count']:
        return candidate['pass_count'] > baseline['pass_count']
    candidate_rows = candidate.get('by_chip') or {}
    baseline_rows = baseline.get('by_chip') or {}
    shared_keys = sorted(set(candidate_rows).intersection(baseline_rows))
    if shared_keys and len(shared_keys) == len(candidate_rows) == len(baseline_rows):
        for key in shared_keys:
            candidate_row = candidate_rows[key]
            baseline_row = baseline_rows[key]
            fp_margin = max(
                100.0 / max(int(candidate_row.get('static_presence_eval_count', 0)), 1),
                100.0 / max(int(baseline_row.get('static_presence_eval_count', 0)), 1),
            )
            recall_margin = max(
                100.0 / max(int(candidate_row.get('motion_eval_count', 0)), 1),
                100.0 / max(int(baseline_row.get('motion_eval_count', 0)), 1),
            )
            if candidate_row.get('fp_rate', 100.0) > baseline_row.get('fp_rate', 100.0) + fp_margin + 1e-9:
                return False
            if candidate_row.get('recall', 0.0) < baseline_row.get('recall', 0.0) - recall_margin - 1e-9:
                return False
            if candidate_row.get('effective_alarms', 0) > baseline_row.get('effective_alarms', 0):
                return False
        return True
    return (
        candidate['max_fp_rate'] <= baseline['max_fp_rate'] + tolerance
        and candidate['worst_chip_recall'] >= baseline['worst_chip_recall'] - tolerance
        and candidate['worst_chip_f1'] >= baseline['worst_chip_f1'] - tolerance
        and candidate.get('total_effective_alarms', 0)
        <= baseline.get('total_effective_alarms', 0)
    )


def deployment_candidate_beats_baseline(candidate, baseline):
    """Compare single-run candidates with safety first and robust CV leading."""
    if not paired_result_non_regression(candidate['paired'], baseline['paired']):
        return False
    return compare_robust_cv(candidate['cv'], baseline['cv'])['passed']


def experiment_feature_ablation(feature_name, seed=None,
                                scaler_mode=DEFAULT_SCALER_MODE,
                                batch_size=DEFAULT_BATCH_SIZE,
                                fp_weight=DEFAULT_FP_WEIGHT,
                                environment_filter=None,
                                excluded_chips=None,
                                positive_chip_boost=None,
                                use_cache=True):
    """Compare Core-6 against one feature removal without exporting artifacts."""
    environment_filter = parse_environment_filter(environment_filter)
    excluded_chips = parse_chip_filter(excluded_chips)
    positive_chip_boost = parse_positive_chip_boost(positive_chip_boost)
    try:
        ensure_torch_available()
        seed = resolve_training_seed(seed, trailing_newline=True)
        set_global_determinism(seed, torch_module=torch)
    except (ImportError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    print("\n" + "=" * 70)
    print("  TARGETED FEATURE ABLATION")
    print("=" * 70)
    print(f"Removed feature: {feature_name}")
    print(f"Seed: {seed}")
    print("Artifacts: unchanged")

    matrix, _ = load_training_matrix(
        environment_filter=environment_filter,
        excluded_chips=excluded_chips,
        feature_names=TRAINING_FEATURES,
        use_cache=use_cache,
    )
    if not matrix['stats']['chips']:
        print("Error: No datasets found in data/")
        return 1

    sample_weights, _ = apply_positive_chip_boost(
        matrix['sample_weights'],
        matrix['sample_context'],
        matrix['y'],
        positive_chip_boost,
    )
    baseline_dataset = {
        'X': np.asarray(matrix['X'], dtype=np.float32),
        'y': np.asarray(matrix['y'], dtype=np.int8),
        'feature_names': list(matrix['feature_names']),
        'sample_context': matrix['sample_context'],
        'sample_weights': np.asarray(sample_weights, dtype=np.float32),
        'groups': matrix['sample_context'][DEFAULT_PRIMARY_GROUP_KEY],
    }
    try:
        candidate_dataset = build_feature_ablation_dataset(
            baseline_dataset,
            feature_name,
        )
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1

    baseline = evaluate_architecture_candidate(
        'Core-6 baseline',
        DEFAULT_HIDDEN_LAYERS,
        seed,
        baseline_dataset,
        scaler_mode,
        batch_size,
        fp_weight,
    )
    candidate = evaluate_architecture_candidate(
        f"Drop {feature_name}",
        DEFAULT_HIDDEN_LAYERS,
        seed,
        candidate_dataset,
        scaler_mode,
        batch_size,
        fp_weight,
    )
    _print_feature_ablation_comparison(baseline, candidate)
    if deployment_candidate_beats_baseline(candidate, baseline):
        print("Paired-first result: candidate ranks above the Core-6 baseline for this seed.")
    else:
        print("Paired-first result: candidate does not beat the Core-6 baseline for this seed.")
    return 0


def experiment_architectures(scaler_mode=DEFAULT_SCALER_MODE,
                             batch_size=DEFAULT_BATCH_SIZE,
                             fp_weight=DEFAULT_FP_WEIGHT,
                             environment_filter=None,
                             excluded_chips=None,
                             architectures=None,
                             positive_chip_boost=None,
                             output_path=DEFAULT_EXPERIMENT_OUTPUT,
                             promote_winner=False,
                             use_cache=True):
    """Run the FP-first architecture campaign and optionally promote a winner."""
    environment_filter = parse_environment_filter(environment_filter)
    excluded_chips = parse_chip_filter(excluded_chips)
    positive_chip_boost = parse_positive_chip_boost(positive_chip_boost)
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
            f"{name} | median paired pass={summary['median_paired_pass_count']:.1f} | "
            f"median maxFP={summary['median_paired_max_fp_rate']:.1f}% | "
            f"median worstRecall={summary['median_paired_worst_chip_recall']:.1f}% | "
            f"median worstF1={summary['median_paired_worst_chip_f1']:.1f}% | "
            f"median OOF={summary['median_oof_f1']:.1f}%"
        )

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
            f"{name} | median paired pass={summary['median_paired_pass_count']:.1f} | "
            f"median maxFP={summary['median_paired_max_fp_rate']:.1f}% | "
            f"median worstRecall={summary['median_paired_worst_chip_recall']:.1f}% | "
            f"median worstF1={summary['median_paired_worst_chip_f1']:.1f}% | "
            f"median OOF={summary['median_oof_f1']:.1f}%"
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

    print(f"\nDecision: {winner['name']} beats {static_presence_name} on paired ranking")
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
        hidden_layers=spec['layers'],
        scaler_mode=scaler_mode,
        batch_size=batch_size,
        export_artifacts=True,
        environment_filter=environment_filter,
        excluded_chips=excluded_chips,
        positive_chip_boost=positive_chip_boost,
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
            'backup_dir': str(backup_dir),
        }
        write_json_results(output_path, results)
        print("Promotion verification failed, restored previous artifacts")
        if final_gate.paired_output.strip():
            print(final_gate.paired_output.strip())
        return 1

    results['promotion']['final_export'] = {
        'status': 'promoted',
        'seed': int(used_seed),
        'paired_returncode': int(final_gate.paired_returncode),
        'backup_dir': str(backup_dir),
        'paired_output': final_gate.paired_output,
    }
    write_json_results(output_path, results)
    print(f"Promoted architecture: {winner['name']} (seed {used_seed})")
    return 0


def experiment_fp_weights(fp_weights=None, scaler_mode=DEFAULT_SCALER_MODE,
                          batch_size=DEFAULT_BATCH_SIZE, hidden_layers=None,
                          environment_filter=None, excluded_chips=None,
                          positive_chip_boost=None,
                          output_path=DEFAULT_FP_WEIGHT_EXPERIMENT_OUTPUT,
                          promote_winner=False, use_cache=True):
    """Run a gated, multi-seed FP-weight campaign."""
    weights = parse_fp_weight_sweep(fp_weights or DEFAULT_FP_WEIGHT_SWEEP)
    if DEFAULT_FP_WEIGHT not in weights:
        weights.insert(0, DEFAULT_FP_WEIGHT)
    else:
        weights = [DEFAULT_FP_WEIGHT] + [value for value in weights if value != DEFAULT_FP_WEIGHT]
    hidden_layers = list(hidden_layers or DEFAULT_HIDDEN_LAYERS)
    environment_filter = parse_environment_filter(environment_filter)
    excluded_chips = parse_chip_filter(excluded_chips)
    positive_chip_boost = parse_positive_chip_boost(positive_chip_boost)
    screening_seed = read_exported_seed() or DEFAULT_EXPERIMENT_SCREENING_SEED

    try:
        ensure_torch_available()
        torch_device_label = describe_torch_device()
    except (ImportError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    print("\n" + "=" * 70)
    print("  FP-WEIGHT CAMPAIGN")
    print("=" * 70)
    print(f"Architecture: {format_hidden_layers(hidden_layers)}")
    print(f"Scaler: {scaler_mode}")
    print(f"Batch size: {batch_size}")
    print(f"Torch device: {torch_device_label}")
    print(f"Screening seed: {screening_seed}")
    print(f"FP weights: {', '.join(map(str, weights))}")
    print("Artifacts: unchanged during evaluation")

    matrix, _ = load_training_matrix(
        environment_filter=environment_filter,
        excluded_chips=excluded_chips,
        feature_names=TRAINING_FEATURES,
        use_cache=use_cache,
    )
    if not matrix['stats']['chips']:
        print("Error: No datasets found in data/")
        return 1
    sample_weights, _ = apply_positive_chip_boost(
        matrix['sample_weights'], matrix['sample_context'], matrix['y'], positive_chip_boost,
    )
    dataset = {
        'X': np.asarray(matrix['X'], dtype=np.float32),
        'y': np.asarray(matrix['y'], dtype=np.int8),
        'feature_names': list(matrix['feature_names']),
        'sample_context': matrix['sample_context'],
        'sample_weights': np.asarray(sample_weights, dtype=np.float32),
        'groups': matrix['sample_context'][DEFAULT_PRIMARY_GROUP_KEY],
    }

    def evaluate(weight, seed):
        return evaluate_architecture_candidate(
            f"fp_weight={weight:g}", hidden_layers, seed, dataset,
            scaler_mode, batch_size, weight,
        )

    results = {
        'config': {
            'weights': weights,
            'baseline_weight': DEFAULT_FP_WEIGHT,
            'hidden_layers': hidden_layers,
            'scaler': scaler_mode,
            'batch_size': batch_size,
            'screening_seed': screening_seed,
            'initial_seeds': list(DEFAULT_EXPERIMENT_INITIAL_SEEDS),
            'final_seeds': list(DEFAULT_EXPERIMENT_FINAL_SEEDS),
            'promote_winner': bool(promote_winner),
        },
        'screening': [],
        'seed_filter': [],
        'seed_finalists': [],
        'promotion': None,
    }

    print("\n== Single-seed screening ==")
    for weight in weights:
        results['screening'].append(evaluate(weight, screening_seed))
        write_json_results(output_path, results)
    baseline_name = f"fp_weight={DEFAULT_FP_WEIGHT:g}"
    challengers = [
        run for run in sorted(results['screening'], key=architecture_campaign_rank_key)
        if run['name'] != baseline_name
    ][:2]
    finalist_weights = [DEFAULT_FP_WEIGHT] + [run['fp_weight'] for run in challengers]

    print("\n== 3-seed robustness filter ==")
    for weight in finalist_weights:
        runs = [evaluate(weight, seed) for seed in DEFAULT_EXPERIMENT_INITIAL_SEEDS]
        summary = aggregate_architecture_runs(f"fp_weight={weight:g}", runs)
        summary['fp_weight'] = float(weight)
        results['seed_filter'].append(summary)
        write_json_results(output_path, results)

    challengers = [
        item for item in sorted(results['seed_filter'], key=aggregate_architecture_rank_key)
        if item['name'] != baseline_name
    ]
    head_to_head = [DEFAULT_FP_WEIGHT]
    if challengers:
        head_to_head.append(challengers[0]['fp_weight'])

    print("\n== 5-seed final comparison ==")
    for weight in head_to_head:
        runs = [evaluate(weight, seed) for seed in DEFAULT_EXPERIMENT_FINAL_SEEDS]
        summary = aggregate_architecture_runs(f"fp_weight={weight:g}", runs)
        summary['fp_weight'] = float(weight)
        results['seed_finalists'].append(summary)
        write_json_results(output_path, results)

    finalists = sorted(results['seed_finalists'], key=aggregate_architecture_rank_key)
    baseline = next(item for item in finalists if item['name'] == baseline_name)
    winner = finalists[0]
    promote_candidate = (
        winner['name'] != baseline_name
        and architecture_candidate_beats_baseline(winner, baseline)
    )
    results['promotion'] = {
        'winner': winner['name'],
        'baseline': baseline_name,
        'decision': f"promote {winner['name']}" if promote_candidate else f"keep {baseline_name}",
        'clear_winner': bool(promote_candidate),
        'summary': winner,
        'baseline_summary': baseline,
    }
    write_json_results(output_path, results)
    print(f"\nDecision: {results['promotion']['decision']}")
    if not promote_candidate or not promote_winner:
        if promote_candidate:
            print("Promotion disabled (--experiment-promote not set); artifacts unchanged")
        return 0

    backup_dir, saved_files = _backup_artifacts()
    export_rc, used_seed, _ = train_all(
        fp_weight=winner['fp_weight'],
        seed=winner['best_single_run']['seed'],
        feature_names=TRAINING_FEATURES,
        hidden_layers=hidden_layers,
        scaler_mode=scaler_mode,
        batch_size=batch_size,
        export_artifacts=True,
        evaluate_deployment=True,
        environment_filter=environment_filter,
        excluded_chips=excluded_chips,
        positive_chip_boost=positive_chip_boost,
        use_cache=use_cache,
    )
    final_gate = run_exported_ml_gates() if export_rc == 0 else None
    if final_gate is None or not final_gate.passed:
        _restore_artifacts(saved_files)
        results['promotion']['final_export'] = {'status': 'verification_failed'}
        write_json_results(output_path, results)
        print(f"Promotion failed; previous artifacts restored from {backup_dir}")
        return 1
    results['promotion']['final_export'] = {
        'status': 'promoted',
        'seed': int(used_seed),
        'fp_weight': float(winner['fp_weight']),
    }
    write_json_results(output_path, results)
    print(f"Promoted fp_weight={winner['fp_weight']:g} (seed {used_seed})")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Train ML motion detection model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python tools/train_ml_model.py                    # Train and export if the paired gate passes
  python tools/train_ml_model.py --no-export        # Evaluate without replacing runtime artifacts
  python tools/train_ml_model.py --info             # Show dataset info
  python tools/train_ml_model.py --experiment       # Run the FP-first MLP topology campaign
  python tools/train_ml_model.py --experiment --experiment-promote
                                           # Promote the winner if it beats the baseline
  python tools/train_ml_model.py --experiment-fp-weights "1,1.5,2,2.5,3"
                                           # Gated multi-seed FP-weight campaign
  python tools/train_ml_model.py --experiment --experiment-architectures "16,8;24,12;32,16;24;24,12,6"
                                           # Custom shortlist for the topology campaign
  python tools/train_ml_model.py --hidden-layers 24,12
                                           # Evaluate the previous 8 -> 24 -> 12 -> 1 candidate
  python tools/train_ml_model.py --fp-weight 2.0    # Penalize FP 2x more
  python tools/train_ml_model.py --scaler clipped_standard
                                           # Robust clipping + z-score
  python tools/train_ml_model.py --batch-size 32     # Smaller batch size experiment
  python tools/train_ml_model.py --device cuda       # Force CUDA when available
  python tools/train_ml_model.py --device mps        # Force Apple GPU when available
  python tools/train_ml_model.py --no-cache          # Rebuild cached training matrix
  python tools/train_ml_model.py --seed 42          # Reproducible training
  python tools/train_ml_model.py --hidden-layers 24,12 --positive-chip-boost ESP32=1.2
                                           # Bias training slightly toward ESP32 motion recall
  python tools/train_ml_model.py --seed-search-until-improvement 20
                                           # Rank all 20 seeds by robust worst/tail CV
  python tools/train_ml_model.py --augment # Train with the robustness-winner augmentation recipe
  python tools/train_ml_model.py --augment --seed-search-until-improvement 10
                                           # Seed-search using the same train-time augmentation
  python tools/train_ml_model.py --gain-stress-gate --environment bedroom
                                           # Diagnose exported model robustness to feature gain shifts
  python tools/train_ml_model.py --shap --no-export  # Grouped OOF SHAP (200 samples)
  python tools/train_ml_model.py --shap 500 --no-export
                                           # Grouped OOF SHAP (500 samples)
  python tools/train_ml_model.py --ablation-feature turb_skewness --seed 1386543369
                                           # Targeted CV and real-data feature ablation

Configuration (edit at top of this file):
  TRAINING_FEATURES = [...]   # Feature list to use

To compare ML with the moving-variance baseline, use:
  python tools/compare_detection_methods.py
'''
    )
    parser.add_argument('--info', action='store_true', 
                       help='Show dataset information')
    parser.add_argument('--experiment', action='store_true',
                       help='Run the FP-first MLP topology campaign')
    parser.add_argument('--experiment-promote', action='store_true',
                       help='With an experiment campaign, export its winner if it beats the baseline')
    parser.add_argument('--experiment-output', type=Path, default=DEFAULT_EXPERIMENT_OUTPUT,
                       help='JSON output path for --experiment results '
                            f'(default: {DEFAULT_EXPERIMENT_OUTPUT})')
    parser.add_argument('--experiment-architectures', type=parse_architecture_sweep, default=None,
                       help='Semicolon-separated hidden-layer specs for --experiment, '
                            'e.g. "16,8;24,12;32,16;24;24,12,6"')
    parser.add_argument('--experiment-fp-weights', type=parse_fp_weight_sweep, default=None,
                       metavar='WEIGHTS',
                       help='Run a gated multi-seed campaign over comma-separated FP weights')
    parser.add_argument('--experiment-robustness', action='store_true',
                       help='Run the non-destructive staged Core-6 robustness campaign')
    parser.add_argument('--robustness-augmentation-only', action='store_true',
                       help='With --experiment-robustness, keep standard Core-6 '
                            'normalization and evaluate only augmentation candidates')
    parser.add_argument('--robustness-experiment-output', type=Path,
                       default=DEFAULT_ROBUSTNESS_EXPERIMENT_OUTPUT,
                       help='JSON output path for --experiment-robustness '
                            f'(default: {DEFAULT_ROBUSTNESS_EXPERIMENT_OUTPUT})')
    parser.add_argument('--fp-weight-experiment-output', type=Path,
                       default=DEFAULT_FP_WEIGHT_EXPERIMENT_OUTPUT,
                       help='JSON output path for --experiment-fp-weights '
                            f'(default: {DEFAULT_FP_WEIGHT_EXPERIMENT_OUTPUT})')
    parser.add_argument('--seed', type=int, default=None,
                       help='Training seed. When omitted, reuse the seed embedded '
                            'in the current exported model when available; '
                            'otherwise generate a random seed. '
                            '--seed-search-until-improvement always samples fresh seeds')
    parser.add_argument('--augment', action='store_true',
                       help='Apply the Core-6 robustness-winner train-time '
                            'augmentation recipe (feature jitter 0.10 + moderate '
                            'packet gain/noise/loss). Inference stays unaugmented')
    parser.add_argument('--seed-search-until-improvement', type=int, default=0, metavar='MAX_TRIALS',
                       help='Evaluate MAX_TRIALS auto-generated seeds, require '
                            'deployment safety and per-recording non-regression, '
                            'then keep the strongest material worst/tail grouped-CV '
                            'improvement. A reserved holdout, when configured, is '
                            'opened only for the selected winner')
    parser.add_argument('--gain-stress-gate', action='store_true',
                       help='Evaluate current exported ML artifacts under '
                            'artificial gain scaling without training/exporting')
    parser.add_argument('--gain-stress-scales', type=parse_gain_stress_scales,
                       default=DEFAULT_GAIN_STRESS_SCALES,
                       help='Comma-separated gain multipliers for --gain-stress-gate '
                            f'(default: {",".join(map(str, DEFAULT_GAIN_STRESS_SCALES))})')
    parser.add_argument('--fp-weight', type=float, default=DEFAULT_FP_WEIGHT,
                       help='Multiplier for IDLE class weight to penalize false positives. '
                            f'Values >1.0 make the model more conservative (default: {DEFAULT_FP_WEIGHT:.1f})')
    parser.add_argument('--scaler', choices=[
                           'standard', 'robust', 'session_balanced_robust', 'clipped_standard'],
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
    parser.add_argument('--no-export', action='store_true',
                       help='Leave runtime artifacts unchanged (CV-only for normal training; '
                            'also use with --shap / --ablation diagnostics)')
    parser.add_argument('--no-cache', action='store_true',
                       help='Rebuild the training feature matrix instead of using the local cache')
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
    parser.add_argument('--shap', type=int, nargs='?', const=200, default=None,
                       metavar='SAMPLES',
                       help='Calculate grouped out-of-fold SHAP importance '
                            '(default: 200 balanced held-out samples)')
    parser.add_argument('--correlation', action='store_true',
                       help='Calculate correlation of selected training features with motion label')
    parser.add_argument('--ablation', action='store_true',
                       help='Run ablation study (test removing each feature)')
    parser.add_argument('--ablation-feature', type=str, default=None,
                       help='Compare Core-6 against one named feature removal using grouped CV '
                            'and paired validation without exporting artifacts')
    parser.add_argument('--cross-environment', action='store_true',
                       help='Leave-one-environment-out generalization check: train on all '
                            'other named environments and evaluate on the held-out room. '
                            'Diagnostic only; does not train a promotable model or export artifacts')
    parser.add_argument('--cross-chip', action='store_true',
                       help='Leave-one-chip-out generalization check: train on all other chips '
                            'and evaluate on the held-out chip. '
                            'Diagnostic only; does not train a promotable model or export artifacts')
    args = parser.parse_args()
    set_active_torch_device(args.device)
    selected_training_features = list(DEFAULT_FEATURES)
    
    if args.info:
        show_info()
        return 0

    if args.experiment_promote and not (
        args.experiment or args.experiment_fp_weights is not None
    ):
        print("Error: --experiment-promote requires an experiment campaign")
        return 1
    experiment_count = sum((
        bool(args.experiment),
        args.experiment_fp_weights is not None,
        bool(args.experiment_robustness),
    ))
    if experiment_count > 1:
        print("Error: experiment campaigns are mutually exclusive")
        return 1

    if args.robustness_augmentation_only and not args.experiment_robustness:
        print("Error: --robustness-augmentation-only requires --experiment-robustness")
        return 1
    if args.augment and (
        args.experiment
        or args.experiment_fp_weights is not None
        or args.experiment_robustness
        or args.gain_stress_gate
        or args.ablation_feature
        or args.ablation
        or args.shap is not None
        or args.correlation
    ):
        print(
            "Error: --augment applies only to production training, seed search, "
            "and cross-environment/cross-chip diagnostics"
        )
        return 1

    if args.experiment_robustness:
        if args.experiment_promote:
            print("Error: --experiment-robustness never promotes artifacts")
            return 1
        if args.seed is not None or args.seed_search_until_improvement > 0:
            print("Error: --experiment-robustness uses its fixed 1/3/5-seed schedule")
            return 1
        if args.environment is not None or parse_chip_filter(args.exclude_chip) is not None:
            print("Error: --experiment-robustness requires all environments and chips")
            return 1
        if args.positive_chip_boost is not None:
            print("Error: --experiment-robustness keeps sample weighting fixed")
            return 1
        run_robustness_experiment(
            output_path=args.robustness_experiment_output,
            use_cache=not args.no_cache,
            hidden_layers=args.hidden_layers,
            fp_weight=args.fp_weight,
            batch_size=args.batch_size,
            augmentation_only=args.robustness_augmentation_only,
        )
        return 0

    if args.gain_stress_gate:
        if (args.experiment or args.experiment_fp_weights is not None
                or args.experiment_robustness or args.experiment_promote):
            print("Error: --gain-stress-gate cannot be combined with experiment flows")
            return 1
        if args.seed_search_until_improvement > 0 or args.seed is not None:
            print("Error: --gain-stress-gate evaluates exported artifacts and cannot use --seed or seed-search")
            return 1
        if args.shap is not None or args.ablation or args.ablation_feature or args.correlation:
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

    if args.cross_environment or args.cross_chip:
        mode = '--cross-environment' if args.cross_environment else '--cross-chip'
        if args.cross_environment and args.cross_chip:
            print("Error: --cross-environment and --cross-chip are mutually exclusive")
            return 1
        if (args.experiment or args.experiment_fp_weights is not None
                or args.experiment_robustness or args.experiment_promote):
            print(f"Error: {mode} cannot be combined with experiment flows")
            return 1
        if args.shap is not None or args.ablation or args.ablation_feature or args.correlation:
            print(f"Error: {mode} cannot be combined with --shap, --ablation, or --correlation")
            return 1
        if args.seed_search_until_improvement > 0:
            print(f"Error: {mode} cannot be combined with seed search")
            return 1
        if args.environment is not None:
            print(f"Error: {mode} holds out one group at a time and "
                  "cannot be combined with --environment")
            return 1
        cross_validation = (
            cross_environment_validation if args.cross_environment else cross_chip_validation
        )
        return cross_validation(
            fp_weight=args.fp_weight,
            seed=args.seed,
            feature_names=selected_training_features,
            hidden_layers=args.hidden_layers,
            scaler_mode=args.scaler,
            batch_size=args.batch_size,
            excluded_chips=args.exclude_chip,
            use_cache=not args.no_cache,
            augment=args.augment,
        )

    if args.experiment:
        return experiment_architectures(
            scaler_mode=args.scaler,
            batch_size=args.batch_size,
            fp_weight=args.fp_weight,
            environment_filter=args.environment,
            excluded_chips=args.exclude_chip,
            architectures=args.experiment_architectures,
            positive_chip_boost=args.positive_chip_boost,
            output_path=args.experiment_output,
            promote_winner=args.experiment_promote,
            use_cache=not args.no_cache,
        )

    if args.experiment_fp_weights is not None:
        return experiment_fp_weights(
            fp_weights=args.experiment_fp_weights,
            scaler_mode=args.scaler,
            batch_size=args.batch_size,
            hidden_layers=args.hidden_layers,
            environment_filter=args.environment,
            excluded_chips=args.exclude_chip,
            positive_chip_boost=args.positive_chip_boost,
            output_path=args.fp_weight_experiment_output,
            promote_winner=args.experiment_promote,
            use_cache=not args.no_cache,
        )

    if args.ablation_feature:
        if args.ablation:
            print("Error: --ablation and --ablation-feature are mutually exclusive")
            return 1
        return experiment_feature_ablation(
            feature_name=args.ablation_feature,
            seed=args.seed,
            scaler_mode=args.scaler,
            batch_size=args.batch_size,
            fp_weight=args.fp_weight,
            environment_filter=args.environment,
            excluded_chips=args.exclude_chip,
            positive_chip_boost=args.positive_chip_boost,
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
        if args.seed is not None:
            print("Error: --seed and --seed-search-until-improvement are mutually exclusive")
            return 1
        if args.shap is not None or args.ablation or args.ablation_feature:
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
            use_cache=not args.no_cache,
            augment=args.augment,
        )

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
        use_cache=not args.no_cache,
        augment=args.augment,
        export_artifacts=(
            not args.no_export
            and args.shap is None
            and not args.ablation
        ),
        evaluate_deployment=(
            not args.no_export
            and args.shap is None
            and not args.ablation
        ),
    )
    return train_rc


if __name__ == '__main__':
    exit(main())
