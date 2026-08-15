#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
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
    python tools/train_ml_model.py --fp-weight 1.75   # Penalize FP 1.75x more
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
  - TRAINING_FEATURES: Production ML feature list

Note: turbulence normalization now follows the shared production path:
CV-normalized turbulence (`std/mean`) for every stream.

To compare ML with Lightweight and RSSI baselines, use:
    python tools/compare_detection_methods.py

Author: Francesco Pace <francesco.pace@gmail.com>
"""

import os
import sys

import argparse
import ast
import copy
import hashlib
import importlib.util
import inspect
import json
import numpy as np
import random
import re
import shutil
import tempfile
import textwrap
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from dataclasses import dataclass
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT_PATH = SCRIPT_DIR.parent
if str(REPO_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_PATH))

from tools.lib.bootstrap import setup_paths  # noqa: F401
from tools.lib.dataset_metadata import (
    admitted_dataset_role,
    dataset_role,
    measure_packet_interval_us,
    paired_dataset_role,
    resolve_entry_path,
)

from tools.lib.repo_paths import (
    cpp_core_dir,
    generated_data_dir,
    python_src_dir,
)
from tools.lib import npz_cache
from tools.lib.atomic_io import atomic_savez, atomic_write_set, atomic_write_text
from tools.lib.timing_quality import summarize_capture_timing
from contextlib import contextmanager, nullcontext
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

from tools.lib.csi_io import load_npz_packet_view
from tools.lib.dataset_metadata import DATA_DIR
from config import (
    DEFAULT_SUBCARRIERS,
    ENABLE_HAMPEL_FILTER,
    ENABLE_LOWPASS_FILTER,
    EVALUATION_INTERVAL_MS,
    HAMPEL_THRESHOLD,
    HAMPEL_WINDOW,
    LOWPASS_CUTOFF,
    MOTION_OFF_HITS,
    MOTION_ON_HITS,
    SEGMENTATION_WINDOW_SIZE_MS,
)
from detector_interface import MotionState
from runtime_policy import (
    RuntimeMotionPolicy,
    derive_detector_timing,
    nominal_packet_interval_us,
)
from temporal_csi_sampler import (
    TemporalCsiSampler,
    minimum_valid_slots,
    temporal_window_slots,
)
from segmentation import SegmentationContext
from tools.lib.performance_report import (
    STRESS_TARGET_FP_RATE,
    STRESS_TARGET_RECALL,
    build_ml_replay_rows,
    load_or_compute_ml_replay_rows,
    timing_cadence_for_window,
)
from tools.lib.temporal_replay import (
    iter_temporal_admissions,
    packet_timestamp_us,
    target_pps_for_packets,
)
from csi_features import (
    AGGREGATED_TURBULENCE_FEATURES,
    ALL_FEATURES,
    DEFAULT_FEATURES,
    L1_DELTA_LAG,
    L1_TRACKER_FEATURES,
    L1DeltaTracker,
    TURB_IQR_AGGREGATION_WIDTH,
    calc_autocorrelation,
    calc_zero_crossing_rate,
    extract_features_by_name,
)

DEFAULT_WINDOW_PACKETS_AT_NOMINAL_RATE = temporal_window_slots(
    100,
    SEGMENTATION_WINDOW_SIZE_MS,
)
from tools.lib.candidate_features import (
    CANDIDATE_FEATURES,
    assemble_feature_vector,
    candidate_feature_cache_identity,
    candidate_values,
    needs_aggregated_turbulence,
    needs_amplitude_profiles,
    needs_channel_coherence,
    needs_channel_shape,
    needs_channel_shape_trajectory,
    needs_l1_series as needs_candidate_l1_series,
    needs_phase_residual,
    needs_subband_coherence,
    split_feature_names,
)
from tools.lib.host_feature_trackers import (
    AmplitudeProfileTracker,
    CHANNEL_SHAPE_BIN_US,
    ChannelCoherenceTracker,
    ChannelShapeTrajectoryTracker,
    ChannelShapeTracker,
    PhaseResidualTracker,
)
from high_accuracy_detector import FEATURE_NAMES as EXPORTED_FEATURE_NAMES, HighAccuracyDetector  # noqa: F401 (re-exported for tests)


def _needs_l1_tracker(feature_names):
    """Return whether any requested feature needs the L1-delta tracker."""
    return (
        any(name in L1_TRACKER_FEATURES for name in feature_names)
        or needs_candidate_l1_series(feature_names)
    )


def _needs_l1_series(feature_names):
    """Return whether any requested feature reads the rebuilt L1-delta series.

    The lag ratio needs the tracker but not the series, so the two questions
    are asked separately; mirrors MLFeatureSource in csi_features.h.
    """
    return needs_candidate_l1_series(feature_names)


def _production_tracker_feature_kwargs(
    feature_names,
    shape_trajectory_tracker=None,
):
    """Return preprocessed production-only tracker values for extraction."""
    kwargs = {}
    if shape_trajectory_tracker is not None:
        innovation, excess, spread = (
            shape_trajectory_tracker.trajectory_features_with_spread()
        )
        if 'chan_shape_spread_subband' in feature_names:
            kwargs['chan_shape_spread_subband'] = spread
        if 'chan_shape_coherent_innovation_energy' in feature_names:
            kwargs['chan_shape_coherent_innovation_energy'] = innovation
        if 'chan_shape_excess_path' in feature_names:
            kwargs['chan_shape_excess_path'] = excess
    return kwargs

# ============================================================================
# Feature Selection
# ============================================================================
#
# Production MLP uses the promoted Subband 7F feature set in
# src/python/micro_espectre/csi_features.py DEFAULT_FEATURES.
# See ALGORITHMS.md "Feature Importance" for SHAP/correlation rankings.
# ============================================================================

TRAINING_FEATURES = DEFAULT_FEATURES
ACTIVE_TRAJECTORY_BIN_US = CHANNEL_SHAPE_BIN_US
DEFAULT_AGGREGATED_CANDIDATE_WIDTH = TURB_IQR_AGGREGATION_WIDTH
FIXED_PACKET_AUGMENTATION_SEEDS = (20260807, 20260808)
# Single-view diagnostics retain the first promoted seed. Production training
# uses both fixed views through training_packet_augmentation_seeds().
FIXED_PACKET_AUGMENTATION_SEED = FIXED_PACKET_AUGMENTATION_SEEDS[0]


def selectable_features():
    """Names `--features` accepts.

    Host-side candidates widen the selectable set without touching the
    production surface the two runtimes share; the export guard still rejects
    them because they have no C++ extractor id. Resolved per call so tests can
    substitute the production surface.
    """
    return tuple(ALL_FEATURES) + tuple(CANDIDATE_FEATURES)


def set_active_trajectory_bin_ms(value):
    """Select the host-side trajectory bin used by read-only experiments."""
    global ACTIVE_TRAJECTORY_BIN_US
    milliseconds = int(value)
    if milliseconds < 1:
        raise ValueError("trajectory bin must be at least 1 ms")
    ACTIVE_TRAJECTORY_BIN_US = milliseconds * 1000


@contextmanager
def canonical_trajectory_bin():
    """Temporarily restore the production bin for exported-model baselines."""
    global ACTIVE_TRAJECTORY_BIN_US
    previous = ACTIVE_TRAJECTORY_BIN_US
    ACTIVE_TRAJECTORY_BIN_US = CHANNEL_SHAPE_BIN_US
    try:
        yield
    finally:
        ACTIVE_TRAJECTORY_BIN_US = previous
BINARY_TRAINING_LABELS = ('empty', 'static_presence', 'motion')
# Directories
GENERATED_DATA_DIR = generated_data_dir()
SRC_DIR = python_src_dir()
CPP_DIR = cpp_core_dir()

# Default training/evaluation configuration
DEFAULT_HIDDEN_LAYERS = [24, 12]
DEFAULT_FP_WEIGHT = 1.75
DEFAULT_SCALER_MODE = 'standard'
DEFAULT_BATCH_SIZE = 1024
DEFAULT_TORCH_DEVICE = 'cpu'
# All chips included: HighAccuracyDetector keeps the legacy variance-baseline CV normalization disabled, then
# extracts the exported raw/relative feature set from the same turbulence base.
DEFAULT_EXCLUDED_CHIPS = ()
DEFAULT_ARCHITECTURE_SWEEP = (
    {'name': 'Legacy (16-8)', 'layers': [16, 8]},
    {'name': 'Promoted default (24-12)', 'layers': [24, 12]},
    {'name': 'Shallow (24)', 'layers': [24]},
    {'name': 'Wider reference (32-16)', 'layers': [32, 16]},
    {'name': 'Deep (24-12-6)', 'layers': [24, 12, 6]},
)
DEFAULT_EXPERIMENT_OUTPUT = GENERATED_DATA_DIR / 'mlp_architecture_experiment.json'
DEFAULT_FP_WEIGHT_EXPERIMENT_OUTPUT = GENERATED_DATA_DIR / 'mlp_fp_weight_experiment.json'
DEFAULT_SEED_SEARCH_OUTPUT = GENERATED_DATA_DIR / 'mlp_seed_search.json'
DEFAULT_FP_WEIGHT_SWEEP = (1.0, 1.5, 1.75, 2.0, 2.5, 3.0)
DEFAULT_EXPERIMENT_SCREENING_SEED = 20260519
DEFAULT_EXPERIMENT_INITIAL_SEEDS = (20260518, 20260519, 20260520)
DEFAULT_EXPERIMENT_FINAL_SEEDS = (20260518, 20260519, 20260520, 20260521, 20260522)
DEFAULT_PAIRED_GATE_CHIPS = ('C3', 'C5', 'C6', 'ESP32', 'S3')
DEFAULT_GAIN_STRESS_SCALES = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
GAIN_SENSITIVE_FEATURES = ()
DEFAULT_GATE_TARGET_RECALL = 95.0
DEFAULT_GATE_TARGET_FP_RATE = 5.0
DEFAULT_MAX_EPOCHS = 100
DEFAULT_EARLY_STOP_PATIENCE = 8
DEFAULT_LR_PATIENCE = 4
DEFAULT_CLIP_PERCENTILES = (1.0, 99.0)
DATASET_ROLES = ('train', 'selection', 'holdout', 'exclude')
DEFAULT_TRAINING_ROLES = ('train',)
DEFAULT_PRIMARY_GROUP_KEY = 'lineage_group'
DEFAULT_BLOCK_GROUP_KEY = 'source_file'
DEFAULT_CV_FOLDS = 3
DEFAULT_SHAP_BACKGROUND_SAMPLES = 100
ROBUST_TAIL_GROUPS = 5
OOF_F1_EQUIVALENCE_MARGIN = 0.2
DEFAULT_TIMING_QUALITY_POLICY = "keep"
DEFAULT_TIMING_WARN_WEIGHT = 0.5
TRAINING_SAMPLE_CONTRACT = "stream_dense"
TIMING_QUALITY_POLICIES = (
    "keep",
    "exclude-fail",
    "downweight-warn",
    "exclude-fail-downweight-warn",
)
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


def parse_timing_quality_policy(value):
    """Normalize one timing-quality policy name."""
    if value is None:
        return DEFAULT_TIMING_QUALITY_POLICY
    policy = str(value).strip().lower()
    if policy not in TIMING_QUALITY_POLICIES:
        raise argparse.ArgumentTypeError(
            "timing-quality policy must be one of: "
            + ", ".join(TIMING_QUALITY_POLICIES)
        )
    return policy


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


TRAINING_AUGMENT_COMPONENT_ORDER = (
    "base",
    "drift",
    "burst-loss",
)
DEFAULT_TRAINING_AUGMENT_COMPONENTS = TRAINING_AUGMENT_COMPONENT_ORDER


def parse_augmentation_components(value):
    """Normalize one augmentation component set, keeping CLI compatibility."""
    if value in (None, False):
        return tuple()
    if value is True:
        return DEFAULT_TRAINING_AUGMENT_COMPONENTS
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return tuple()
        parts = text.split(",")
    elif isinstance(value, (list, tuple, set)):
        parts = value
    else:
        parts = str(value).split(",")

    normalized = []
    seen = set()
    for part in parts:
        name = str(part).strip().lower()
        if not name:
            continue
        if name not in TRAINING_AUGMENT_COMPONENT_ORDER:
            raise argparse.ArgumentTypeError(
                "augment components must be chosen from: "
                + ", ".join(TRAINING_AUGMENT_COMPONENT_ORDER)
            )
        if name in seen:
            continue
        seen.add(name)
        normalized.append(name)
    if not normalized:
        return tuple()
    ordered = [
        name for name in TRAINING_AUGMENT_COMPONENT_ORDER
        if name in seen
    ]
    return tuple(ordered)


def format_augmentation_components(components):
    """Return a stable, user-facing component list."""
    normalized = parse_augmentation_components(components)
    return "none" if not normalized else ",".join(normalized)


def timing_policy_excludes_status(status, policy):
    """Return True when one timing-quality status is filtered out."""
    return str(policy) in ("exclude-fail", "exclude-fail-downweight-warn") and str(status) == "FAIL"


def timing_policy_weight(status, policy, warn_weight=DEFAULT_TIMING_WARN_WEIGHT):
    """Return one per-window multiplier derived from timing provenance."""
    if (
        str(policy) in ("downweight-warn", "exclude-fail-downweight-warn")
        and str(status) == "WARN"
    ):
        return float(warn_weight)
    return 1.0


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
    role = dataset_role(file_info)

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
        'dataset_role': role,
        'synthetic': bool(file_info.get('synthetic', False)),
        'long_recording': bool(file_info.get('long_recording', False)),
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
        'dataset_role': dataset_role(packet),
        'synthetic': packet.get('synthetic', False),
        'long_recording': packet.get('long_recording', False),
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


@lru_cache(maxsize=1)
def _training_source_metadata_parameters():
    """Return the cache identity for lightweight training-source admission."""
    return {
        'contract': 'ml_training_source_metadata_v1',
        'implementation_sha256': _implementation_source_digest(
            _build_training_source_metadata,
        ),
        'sources': {
            'csi_io': npz_cache.source_manifest(SCRIPT_DIR / 'lib' / 'csi_io.py'),
            'dataset_metadata': npz_cache.source_manifest(
                SCRIPT_DIR / 'lib' / 'dataset_metadata.py'
            ),
            'timing_quality': npz_cache.source_manifest(
                SCRIPT_DIR / 'lib' / 'timing_quality.py'
            ),
            'python_config': npz_cache.source_manifest(
                python_src_dir() / 'config.py'
            ),
            'python_runtime_policy': npz_cache.source_manifest(
                python_src_dir() / 'runtime_policy.py'
            ),
        },
    }


def _build_training_source_metadata(npz_file):
    """Materialize only the admission metadata reused across training runs."""
    packets = load_npz_packet_view(npz_file)
    if not packets:
        raise RuntimeError(
            f"{Path(npz_file).name} has no HT20/HT-LTF/64-SC sensing packets "
            "after format filtering"
        )
    first_packet = packets[0]
    return {
        'label': str(first_packet.get('label', Path(npz_file).parent.name)),
        'chip': str(first_packet.get('chip', 'unknown')).upper(),
        'packet_count': len(packets),
        'has_sync_metadata': any(
            packet.get('wifi_rx_start_ts_ns') is not None
            or packet.get('device_ticks_us') is not None
            or packet.get('wifi_rx_ts_us') is not None
            for packet in packets
        ),
        'timing_summary': summarize_capture_timing(packets),
        'fallback_context': {
            'chip': first_packet.get('chip', 'unknown'),
            'collected_at': first_packet.get('collected_at', ''),
            'dataset_role': dataset_role(first_packet),
            'synthetic': bool(first_packet.get('synthetic', False)),
            'long_recording': bool(first_packet.get('long_recording', False)),
        },
    }


def _load_or_compute_training_source_metadata(npz_file):
    """Load cached source admission metadata without retaining packet rows."""
    parameters = _training_source_metadata_parameters()
    cached = npz_cache.load_ml_training_source_metadata_artifact(
        npz_file,
        parameters=parameters,
    )
    if cached is not None:
        return cached
    with npz_cache.artifact_build_lock(
        npz_file,
        artifact_name='ml_training_source_metadata',
        artifact_version=npz_cache.ML_TRAINING_SOURCE_METADATA_ARTIFACT_VERSION,
        parameters=parameters,
    ):
        cached = npz_cache.load_ml_training_source_metadata_artifact(
            npz_file,
            parameters=parameters,
        )
        if cached is not None:
            return cached
        metadata = _build_training_source_metadata(npz_file)
        npz_cache.save_ml_training_source_metadata_artifact(
            npz_file,
            parameters=parameters,
            metadata=metadata,
        )
        persisted = npz_cache.load_ml_training_source_metadata_artifact(
            npz_file,
            parameters=parameters,
        )
        return persisted if persisted is not None else metadata


def _load_training_file_records(environment_filter=None, excluded_chips=None,
                                allowed_labels=BINARY_TRAINING_LABELS,
                                require_sync_metadata=False,
                                dataset_roles=DEFAULT_TRAINING_ROLES,
                                timing_quality_policy=DEFAULT_TIMING_QUALITY_POLICY,
                                timing_warn_weight=DEFAULT_TIMING_WARN_WEIGHT):
    """Return filtered per-file training records plus aggregate stats."""
    environment_filter = parse_environment_filter(environment_filter)
    excluded_chips = parse_chip_filter(excluded_chips)
    allowed_labels = normalize_allowed_labels(allowed_labels)
    dataset_roles = normalize_dataset_roles(dataset_roles)
    timing_quality_policy = parse_timing_quality_policy(timing_quality_policy)
    timing_warn_weight = float(timing_warn_weight)
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
        'excluded_long_recordings': set(),
        'excluded_timing_quality': set(),
        'session_groups': set(),
        'lineage_groups': set(),
        'environment_groups': set(),
        'sync_metadata_files': set(),
        'timing_quality_counts': {
            'clean': 0,
            'degraded': 0,
            'poor': 0,
            'unknown': 0,
        },
    }
    records = []

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
                source_metadata = _load_or_compute_training_source_metadata(
                    npz_file
                )
                
                # Get label from the shared packet view metadata.
                label = source_metadata.get('label', subdir.name)
                
                label_lc = str(label).lower()
                if allowed_labels is not None and label_lc not in allowed_labels:
                    stats['excluded_labels'].add(label_lc)
                    continue

                # Get chip
                chip = str(source_metadata.get('chip', 'unknown')).upper()
                if excluded_chips is not None and chip in excluded_chips:
                    stats['excluded_chips'].add(chip)
                    continue
                
                # Get file-specific metadata
                meta = file_metadata.get(npz_file.name)
                if meta is None:
                    meta = _fallback_file_context(
                        npz_file.name,
                        label_lc,
                        source_metadata.get('fallback_context', {}),
                    )

                if label_lc == 'empty' and bool(meta.get('long_recording', False)):
                    stats['excluded_long_recordings'].add(npz_file.name)
                    continue

                role = dataset_role(meta)
                if role not in dataset_roles:
                    stats['excluded_dataset_roles'].add(role)
                    continue

                environment_group = meta.get('environment_group', 'unknown-environment')
                if environment_filter is not None and environment_group not in environment_filter:
                    stats['excluded_environments'].add(environment_group)
                    continue

                has_sync_metadata = bool(
                    source_metadata.get('has_sync_metadata', False)
                )
                if require_sync_metadata and not has_sync_metadata:
                    stats['excluded_missing_sync_metadata'].add(npz_file.name)
                    continue
                if has_sync_metadata:
                    stats['sync_metadata_files'].add(npz_file.name)

                timing_summary = source_metadata['timing_summary']
                timing_status = str(timing_summary['quality_status'])
                timing_bucket = str(timing_summary['quality_bucket'])
                if timing_policy_excludes_status(timing_status, timing_quality_policy):
                    stats['excluded_timing_quality'].add(npz_file.name)
                    continue
                stats['timing_quality_counts'][timing_bucket] += 1

                # Track stats after all active filters
                if label not in stats['labels']:
                    stats['labels'][label] = 0
                packet_count = int(source_metadata['packet_count'])
                stats['labels'][label] += packet_count
                stats['total'] += packet_count
                stats['chips'].add(chip)

                stats['session_groups'].add(meta.get('session_group', f"file:{npz_file.name}"))
                stats['lineage_groups'].add(meta.get('lineage_group', f"file:{npz_file.name}"))
                if environment_group != 'unknown-environment':
                    stats['environment_groups'].add(environment_group)

                is_motion = is_motion_label(label, dataset_info)
                stats['files'].append(npz_file.name)
                records.append({
                    'path': npz_file,
                    'packets_loader': (
                        lambda current_path=npz_file: load_npz_packet_view(
                            current_path
                        )
                    ),
                    'label_name': label_lc,
                    'is_motion': is_motion,
                    'chip': meta.get('chip', chip),
                    'collected_at': meta.get('collected_at', ''),
                    'day_group': meta.get('day_group', 'unknown-day'),
                    'pair_id': meta.get('pair_id', ''),
                    'session_group': meta.get('session_group', f"file:{npz_file.name}"),
                    'lineage_group': meta.get('lineage_group', meta.get('session_group', f"file:{npz_file.name}")),
                    'dataset_role': role,
                    'synthetic': bool(meta.get('synthetic', False)),
                    'long_recording': bool(meta.get('long_recording', False)),
                    'environment_group': environment_group,
                    'timing_quality_status': timing_status,
                    'timing_quality_bucket': timing_bucket,
                    'timing_summary': timing_summary,
                    'timing_weight': timing_policy_weight(
                        timing_status,
                        timing_quality_policy,
                        warn_weight=timing_warn_weight,
                    ),
                })

            except RuntimeError:
                # Sensing-contract violations must stop training explicitly;
                # a silently skipped file would hide a contaminated dataset.
                raise
            except Exception as e:
                print(f"  Warning: Could not load {npz_file.name}: {e}")
    
    stats['chips'] = sorted(stats['chips'])
    stats['excluded_labels'] = sorted(stats['excluded_labels'])
    stats['excluded_chips'] = sorted(stats['excluded_chips'])
    stats['excluded_environments'] = sorted(stats['excluded_environments'])
    stats['excluded_missing_sync_metadata'] = sorted(stats['excluded_missing_sync_metadata'])
    stats['excluded_dataset_roles'] = sorted(stats['excluded_dataset_roles'])
    stats['excluded_long_recordings'] = sorted(stats['excluded_long_recordings'])
    stats['excluded_timing_quality'] = sorted(stats['excluded_timing_quality'])
    stats['session_groups'] = sorted(stats['session_groups'])
    stats['lineage_groups'] = sorted(stats['lineage_groups'])
    stats['environment_groups'] = sorted(stats['environment_groups'])
    stats['sync_metadata_files'] = sorted(stats['sync_metadata_files'])
    return records, stats


def load_all_data(environment_filter=None, excluded_chips=None,
                  allowed_labels=BINARY_TRAINING_LABELS,
                  require_sync_metadata=False,
                  dataset_roles=DEFAULT_TRAINING_ROLES,
                  timing_quality_policy=DEFAULT_TIMING_QUALITY_POLICY,
                  timing_warn_weight=DEFAULT_TIMING_WARN_WEIGHT):
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
    records, stats = _load_training_file_records(
        environment_filter=environment_filter,
        excluded_chips=excluded_chips,
        allowed_labels=allowed_labels,
        require_sync_metadata=require_sync_metadata,
        dataset_roles=dataset_roles,
        timing_quality_policy=timing_quality_policy,
        timing_warn_weight=timing_warn_weight,
    )
    all_packets = []
    for record in records:
        for idx, packet in enumerate(_ensure_record_packets(record)):
            enriched = dict(packet)
            enriched['is_motion'] = record['is_motion']
            enriched['label_name'] = record['label_name']
            enriched['source_file'] = record['path'].name
            enriched['packet_index'] = idx
            enriched['chip'] = record['chip']
            enriched['collected_at'] = record['collected_at']
            enriched['day_group'] = record['day_group']
            enriched['pair_id'] = record['pair_id']
            enriched['session_group'] = record['session_group']
            enriched['lineage_group'] = record['lineage_group']
            enriched['dataset_role'] = record['dataset_role']
            enriched['synthetic'] = record['synthetic']
            enriched['long_recording'] = record['long_recording']
            enriched['environment_group'] = record['environment_group']
            enriched['timing_quality_status'] = record['timing_quality_status']
            enriched['timing_quality_bucket'] = record['timing_quality_bucket']
            all_packets.append(enriched)
    return all_packets, stats

# Sample-context columns that are not label strings.
CONTEXT_INT_KEYS = ('packet_index', 'window_index', 'reset_index')
CONTEXT_BOOL_KEYS = ('synthetic',)
def _prepare_feature_packets_for_record(record, packet_augmentation=None, augmentation_seed=None):
    """Return one per-file packet list ready for feature extraction."""
    packets = []
    for idx, packet in enumerate(_ensure_record_packets(record)):
        copied = dict(packet)
        copied['source_file'] = record['path'].name
        copied['packet_index'] = idx
        packets.append(copied)
    if packet_augmentation:
        return augment_csi_packets(packets, packet_augmentation, augmentation_seed)
    return packets


@lru_cache(maxsize=None)
def _implementation_source_digest(*objects):
    """Hash only the implementation objects that affect one cached stream."""
    digest = hashlib.sha256()
    for implementation in objects:
        source = inspect.getsource(implementation)
        digest.update(implementation.__qualname__.encode('utf-8'))
        digest.update(b'\0')
        digest.update(source.encode('utf-8'))
        digest.update(b'\0')
    return digest.hexdigest()


@lru_cache(maxsize=64)
def _packet_augmentation_stream_provenance_cached(config_json, augmentation_seed):
    """Build one immutable-by-convention packet-transform provenance value."""
    return {
        'transform': 'training_packet_augmentation_v2',
        'config': json.loads(config_json),
        'seed': int(augmentation_seed),
        'implementation_sha256': _implementation_source_digest(
            _prepare_feature_packets_for_record,
            _ensure_record_packets,
            _normalize_augmentation_range,
            _estimate_packet_rate_pps,
            _resample_stable_packet_rate,
            _smoothed_iq_profile,
            _add_scaled_iq_noise,
            _stable_text_seed,
            derive_seed,
            augment_csi_packets,
        ),
        'timing_quality': npz_cache.source_manifest(
            SCRIPT_DIR / 'lib' / 'timing_quality.py'
        ),
    }


def _packet_augmentation_stream_provenance(packet_augmentation, augmentation_seed):
    """Return a stable cache identity for one deterministic packet transform."""
    if not packet_augmentation or augmentation_seed is None:
        return None
    config_json = json.dumps(
        dict(packet_augmentation),
        sort_keys=True,
        separators=(',', ':'),
    )
    return copy.deepcopy(
        _packet_augmentation_stream_provenance_cached(
            config_json,
            int(augmentation_seed),
        )
    )


@lru_cache(maxsize=64)
def _host_feature_base_stream_provenance(feature_names, trajectory_bin_us):
    """Build shared host provenance once per ordered feature schema."""
    feature_identities = {
        name: _host_feature_cache_identity(name)
        for name in feature_names
    }
    for identity in feature_identities.values():
        if identity.get('provider') == 'channel_shape_trajectory':
            identity['trajectory_bin_us'] = int(trajectory_bin_us)
    return {
        'transform': 'host_feature_rows_v4',
        'feature_names': list(feature_names),
        'row_stream': {
            'contract': 'host_feature_row_spine_v1',
            'timing_sources': {
                'runtime_policy': npz_cache.source_manifest(
                    python_src_dir() / 'runtime_policy.py'
                ),
                'temporal_csi_sampler': npz_cache.source_manifest(
                    python_src_dir() / 'temporal_csi_sampler.py'
                ),
                'config': npz_cache.source_manifest(
                    python_src_dir() / 'config.py'
                ),
                'row_builder_sha256': _implementation_source_digest(
                    build_host_feature_rows,
                    timing_cadence_for_window,
                    iter_temporal_admissions,
                    TemporalCsiSampler.admit,
                    StreamingFeatureExtractor.process_packet,
                    StreamingFeatureExtractor.advance_missing_slots,
                    StreamingFeatureExtractor._ordered_series,
                ),
            },
        },
        'feature_identities': feature_identities,
    }


def _host_feature_stream_provenance(feature_names, *,
                                    packet_augmentation=None,
                                    augmentation_seed=None):
    """Return an isolated copy of memoized granular host provenance."""
    names = tuple(str(name) for name in feature_names)
    provenance = copy.deepcopy(_host_feature_base_stream_provenance(
        names,
        int(ACTIVE_TRAJECTORY_BIN_US),
    ))
    packet_provenance = _packet_augmentation_stream_provenance(
        packet_augmentation,
        augmentation_seed,
    )
    if packet_provenance is not None:
        provenance['packet_augmentation'] = packet_provenance
    return provenance


PRODUCTION_FEATURE_PROVIDER_VERSIONS = {
    'turbulence_window': 2,
    'aggregated_turbulence_window': 2,
    'l1_delta_tracker': 1,
    'channel_shape_trajectory': 2,
}


def _production_feature_provider(feature_name):
    """Return the independently versioned provider of one runtime feature."""
    if feature_name == 'turb_iqr_over_mean_aggr':
        return 'aggregated_turbulence_window'
    if feature_name in {'turb_autocorr', 'turb_zcr'}:
        return 'turbulence_window'
    if feature_name == 'l1_delta_lag_ratio':
        return 'l1_delta_tracker'
    if feature_name in {
        'chan_shape_spread_subband',
        'chan_shape_coherent_innovation_energy',
        'chan_shape_excess_path',
    }:
        return 'channel_shape_trajectory'
    raise ValueError(f'Unknown production feature: {feature_name}')


@lru_cache(maxsize=None)
def _named_feature_branch_digest(function, feature_name):
    """Hash only branches that explicitly implement one named feature."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    matching = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
            continue
        operands = (node.test.left, *node.test.comparators)
        if not any(
            isinstance(item, ast.Name) and item.id == 'name'
            for item in operands
        ):
            continue
        if not any(
            isinstance(item, ast.Constant) and item.value == feature_name
            for item in operands
        ):
            continue
        matching.append(
            ast.dump(
                ast.Module(body=node.body, type_ignores=[]),
                annotate_fields=True,
                include_attributes=False,
            )
        )
    payload = '\n'.join(matching)
    if not payload:
        raise ValueError(f'No extraction branch found for {feature_name}')
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


@lru_cache(maxsize=None)
def _production_provider_digest(provider):
    """Hash shared provider behavior without hashing unrelated siblings."""
    if provider == 'turbulence_window':
        return _implementation_source_digest(
            calc_autocorrelation,
            calc_zero_crossing_rate,
        )
    if provider == 'aggregated_turbulence_window':
        return hashlib.sha256(
            f'aggregation-width:{TURB_IQR_AGGREGATION_WIDTH}'.encode('utf-8')
        ).hexdigest()
    if provider == 'l1_delta_tracker':
        return _implementation_source_digest(
            L1DeltaTracker.__init__,
            L1DeltaTracker.process_amplitudes,
            L1DeltaTracker.delta_lag_ratio,
            L1DeltaTracker.reset,
        )
    if provider == 'channel_shape_trajectory':
        return _implementation_source_digest(
            ChannelShapeTrajectoryTracker.__init__,
            ChannelShapeTrajectoryTracker.reset,
            ChannelShapeTrajectoryTracker.process_packet,
            ChannelShapeTrajectoryTracker._binned_path,
            ChannelShapeTrajectoryTracker.trajectory_features_with_spread,
        )
    raise ValueError(f'Unknown production provider: {provider}')


@lru_cache(maxsize=None)
def _host_feature_cache_identity_cached(name):
    """Build one memoized host-feature identity."""
    if name in CANDIDATE_FEATURES:
        return candidate_feature_cache_identity(name)
    provider = _production_feature_provider(name)
    return {
        'feature_name': name,
        'provider': provider,
        'provider_version': PRODUCTION_FEATURE_PROVIDER_VERSIONS[provider],
        'provider_sha256': _production_provider_digest(provider),
        'formula_sha256': _named_feature_branch_digest(
            extract_features_by_name,
            name,
        ),
    }


def _host_feature_cache_identity(feature_name):
    """Return an isolated copy of one memoized host-feature identity."""
    return dict(_host_feature_cache_identity_cached(str(feature_name)))


def _host_row_stream_identity(stream_provenance):
    """Strip the requested schema while retaining stream-transform identity."""
    provenance = dict(stream_provenance or {})
    return {
        'transform': provenance.get('transform', 'host_feature_rows_v3'),
        'row_stream': provenance.get('row_stream', {}),
        'packet_augmentation': provenance.get('packet_augmentation'),
    }


def training_packet_augmentation_seed(packet_augmentation):
    """Return the primary fixed seed used by single-view stress diagnostics."""
    if not packet_augmentation:
        return None
    return FIXED_PACKET_AUGMENTATION_SEED


def training_packet_augmentation_seeds(packet_augmentation):
    """Return the promoted fixed augmentation views used for model training."""
    if not packet_augmentation:
        return tuple()
    return FIXED_PACKET_AUGMENTATION_SEEDS


def _mix_packet_augmentation_replay_rows(view_rows):
    """Return one constant-size deterministic mix of augmented replay views.

    View ``i`` contributes row positions congruent to ``i`` modulo the number
    of views. The rule is local to each source capture, so adding or removing a
    different capture cannot change an existing file's assignments.
    """
    rows_by_view = list(view_rows)
    if not rows_by_view:
        raise ValueError("at least one packet-augmentation view is required")
    feature_names = list(rows_by_view[0]['feature_names'])
    row_keys = ('X', 'packet_index', 'evaluation_index', 'reset_index', 'evaluation_due')
    selected = {key: [] for key in row_keys}
    view_count = len(rows_by_view)
    for view_index, rows in enumerate(rows_by_view):
        if list(rows['feature_names']) != feature_names:
            raise ValueError("packet-augmentation views have different feature schemas")
        row_count = int(len(rows['X']))
        mask = np.arange(row_count, dtype=np.int64) % view_count == view_index
        for key in row_keys:
            default_dtype = bool if key == 'evaluation_due' else np.int32
            if key == 'X':
                values = np.asarray(rows[key], dtype=np.float32)
            else:
                values = np.asarray(rows.get(key, np.empty(0)), dtype=default_dtype)
            if len(values) != row_count:
                raise ValueError(f"packet-augmentation row field {key} has inconsistent length")
            selected[key].append(values[mask])
    return {
        'X': np.concatenate(selected['X'], axis=0).astype(np.float32, copy=False),
        'feature_names': feature_names,
        'packet_index': np.concatenate(selected['packet_index']).astype(np.int32, copy=False),
        'evaluation_index': np.concatenate(selected['evaluation_index']).astype(np.int32, copy=False),
        'reset_index': np.concatenate(selected['reset_index']).astype(np.int32, copy=False),
        'evaluation_due': np.concatenate(selected['evaluation_due']).astype(bool, copy=False),
    }


def _concatenate_packet_augmentation_replay_rows(view_rows):
    """Concatenate packet-augmentation views already selected by row position."""
    rows_by_view = list(view_rows)
    if not rows_by_view:
        raise ValueError("at least one packet-augmentation view is required")
    feature_names = list(rows_by_view[0]['feature_names'])
    row_keys = ('X', 'packet_index', 'evaluation_index', 'reset_index', 'evaluation_due')
    combined = {key: [] for key in row_keys}
    for rows in rows_by_view:
        if list(rows['feature_names']) != feature_names:
            raise ValueError("packet-augmentation views have different feature schemas")
        row_count = int(len(rows['X']))
        for key in row_keys:
            default_dtype = bool if key == 'evaluation_due' else np.int32
            values = np.asarray(
                rows[key] if key == 'X' else rows.get(key, np.empty(0)),
                dtype=np.float32 if key == 'X' else default_dtype,
            )
            if len(values) != row_count:
                raise ValueError(
                    f"packet-augmentation row field {key} has inconsistent length"
                )
            combined[key].append(values)
    return {
        'X': np.concatenate(combined['X'], axis=0).astype(np.float32, copy=False),
        'feature_names': feature_names,
        'packet_index': np.concatenate(combined['packet_index']).astype(np.int32, copy=False),
        'evaluation_index': np.concatenate(combined['evaluation_index']).astype(np.int32, copy=False),
        'reset_index': np.concatenate(combined['reset_index']).astype(np.int32, copy=False),
        'evaluation_due': np.concatenate(combined['evaluation_due']).astype(bool, copy=False),
    }


def _packet_augmentation_mix_stream_provenance(packet_augmentation,
                                                augmentation_seeds,
                                                feature_names,
                                                use_runtime_cache):
    """Return the complete identity for the promoted mixed-view row cache."""
    seeds = tuple(int(seed) for seed in augmentation_seeds)
    if not packet_augmentation or not seeds:
        return None
    if use_runtime_cache:
        views = [
            _packet_augmentation_stream_provenance(packet_augmentation, seed)
            for seed in seeds
        ]
    else:
        views = [
            _host_feature_stream_provenance(
                feature_names,
                packet_augmentation=packet_augmentation,
                augmentation_seed=seed,
            )
            for seed in seeds
        ]
    return {
        'transform': 'training_packet_augmentation_mix_v1',
        'views': views,
        'selection': {
            'scope': 'source_file',
            'rule': 'row_position_modulo_view_count',
            'offsets': list(range(len(seeds))),
        },
        'implementation_sha256': _implementation_source_digest(
            _mix_packet_augmentation_replay_rows,
            _concatenate_packet_augmentation_replay_rows,
        ),
    }


def _load_or_compute_packet_augmentation_mix_rows(record, *,
                                                   packet_augmentation,
                                                   augmentation_seeds,
                                                   feature_names,
                                                   use_cache,
                                                   use_runtime_cache):
    """Load or build one cached deterministic mix for a source capture."""
    seeds = tuple(int(seed) for seed in augmentation_seeds)
    if len(seeds) < 2:
        raise ValueError("mixed packet augmentation requires at least two seeds")
    mix_provenance = _packet_augmentation_mix_stream_provenance(
        packet_augmentation,
        seeds,
        feature_names,
        use_runtime_cache,
    )
    parameters = npz_cache.ml_training_augmentation_row_parameters(
        selected_subcarriers=DEFAULT_SUBCARRIERS,
        feature_names=feature_names,
        stream_provenance=mix_provenance,
    )
    if use_cache:
        cached = npz_cache.load_ml_training_augmentation_row_artifact(
            record['path'],
            parameters=parameters,
        )
        if cached is not None:
            cached['cache_hit'] = True
            return cached
    lock_context = (
        npz_cache.artifact_build_lock(
            record['path'],
            artifact_name='ml_training_augmentation_rows',
            artifact_version=npz_cache.ML_TRAINING_AUGMENTATION_ROW_ARTIFACT_VERSION,
            parameters=parameters,
        )
        if use_cache
        else nullcontext()
    )
    with lock_context:
        if use_cache:
            cached = npz_cache.load_ml_training_augmentation_row_artifact(
                record['path'],
                parameters=parameters,
            )
            if cached is not None:
                cached['cache_hit'] = True
                return cached
        views = []
        for view_index, seed in enumerate(seeds):
            packets_factory = lambda current_record=record, current_seed=seed: (
                _prepare_feature_packets_for_record(
                    current_record,
                    packet_augmentation=packet_augmentation,
                    augmentation_seed=current_seed,
                )
            )
            if use_runtime_cache:
                rows = load_or_compute_ml_replay_rows(
                    record['path'],
                    packets_factory=packets_factory,
                    selected_subcarriers=DEFAULT_SUBCARRIERS,
                    window_size=None,
                    feature_names=feature_names,
                    sample_contract=TRAINING_SAMPLE_CONTRACT,
                    use_cache=use_cache,
                    cache_write=False,
                    stream_provenance=_packet_augmentation_stream_provenance(
                        packet_augmentation,
                        seed,
                    ),
                    row_stride=len(seeds),
                    row_offset=view_index,
                )
            else:
                rows = load_or_compute_host_feature_rows(
                    record['path'],
                    packets_factory=packets_factory,
                    feature_names=feature_names,
                    sample_contract=TRAINING_SAMPLE_CONTRACT,
                    use_cache=use_cache,
                    cache_write=True,
                    stream_provenance=_host_feature_stream_provenance(
                        feature_names,
                        packet_augmentation=packet_augmentation,
                        augmentation_seed=seed,
                    ),
                )
                rows = _select_host_feature_rows(rows, len(seeds), view_index)
            views.append(rows)

        mixed = _concatenate_packet_augmentation_replay_rows(views)
        if use_cache:
            npz_cache.save_ml_training_augmentation_row_artifact(
                record['path'],
                parameters=parameters,
                rows=mixed,
            )
    mixed['cache_hit'] = False
    return mixed


def _normalize_augmentation_range(value, *, name, integer=False, minimum=0.0):
    """Return one validated inclusive numeric (min, max) range."""
    if isinstance(value, np.ndarray):
        parts = value.tolist()
    elif isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        parts = [item.strip() for item in str(value).split(',') if str(item).strip()]
    if len(parts) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    try:
        first = int(parts[0]) if integer else float(parts[0])
        second = int(parts[1]) if integer else float(parts[1])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain valid numeric values") from exc
    if first < minimum or second < minimum or second < first:
        raise ValueError(f"{name} must satisfy {minimum} <= min <= max")
    return first, second


def _estimate_packet_rate_pps(packets):
    """Estimate effective source throughput through the shared timing policy."""
    summary = summarize_capture_timing(packets)
    return max(1e-9, float(summary["packet_rate_pps"]))


def _resample_stable_packet_rate(packets, rate_scale):
    """Return a lower-rate stable stream without modelling the drop as loss."""
    scale = float(rate_scale)
    if scale >= 1.0 or len(packets) < 2:
        return [dict(packet) for packet in packets]

    source_rate_pps = _estimate_packet_rate_pps(packets)
    target_rate_pps = max(80.0, source_rate_pps * scale)
    if target_rate_pps >= source_rate_pps:
        return [dict(packet) for packet in packets]

    stride = source_rate_pps / target_rate_pps
    interval_us = max(1, int(round(1_000_000.0 / target_rate_pps)))
    selected = []
    cursor = 0.0
    next_seq_num = None
    next_device_ticks_us = None
    next_wifi_rx_ts_us = None
    next_wifi_rx_start_ts_ns = None
    while True:
        source_index = int(round(cursor))
        if source_index >= len(packets):
            break
        packet = dict(packets[source_index])
        if next_seq_num is None:
            next_seq_num = int(packet.get('seq_num', packet.get('stream_seq_num', 0)) or 0)
        else:
            next_seq_num += 1
        packet['seq_num'] = next_seq_num
        packet['stream_seq_num'] = next_seq_num
        if packet.get('device_ticks_us') is not None:
            if next_device_ticks_us is None:
                next_device_ticks_us = int(packet['device_ticks_us'])
            else:
                next_device_ticks_us += interval_us
            packet['device_ticks_us'] = next_device_ticks_us
        if packet.get('wifi_rx_ts_us') is not None:
            if next_wifi_rx_ts_us is None:
                next_wifi_rx_ts_us = int(packet['wifi_rx_ts_us'])
            else:
                next_wifi_rx_ts_us = (next_wifi_rx_ts_us + interval_us) % (1 << 32)
            packet['wifi_rx_ts_us'] = next_wifi_rx_ts_us
        if packet.get('wifi_rx_start_ts_ns') is not None:
            if next_wifi_rx_start_ts_ns is None:
                next_wifi_rx_start_ts_ns = int(packet['wifi_rx_start_ts_ns'])
            else:
                next_wifi_rx_start_ts_ns += interval_us * 1000
            packet['wifi_rx_start_ts_ns'] = next_wifi_rx_start_ts_ns
        selected.append(packet)
        cursor += stride
    return selected


def _smoothed_iq_profile(rng, usable_subcarriers):
    """Return one smooth, unit-RMS per-tone I/Q perturbation template."""
    profile = rng.normal(0.0, 1.0, size=(int(usable_subcarriers), 2))
    if usable_subcarriers > 1:
        kernel = np.asarray([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float64)
        kernel /= np.sum(kernel)
        for axis in range(profile.shape[1]):
            profile[:, axis] = np.convolve(profile[:, axis], kernel, mode='same')
    rms = float(np.sqrt(np.mean(np.square(profile))))
    if rms <= 0.0:
        return profile
    return profile / rms


def _constant_object_array(value, length):
    """Return one object array filled with one repeated value."""
    array = np.empty(int(length), dtype=object)
    array[:] = value
    return array


def _build_sample_context_for_replay_rows(record, rows):
    """Return per-row sample context for one canonical replay-row payload."""
    count = int(len(rows['X']))
    return {
        'chip': _constant_object_array(str(record['chip']).upper(), count),
        'source_file': _constant_object_array(record['path'].name, count),
        'lineage_group': _constant_object_array(record['lineage_group'], count),
        'session_group': _constant_object_array(record['session_group'], count),
        'environment_group': _constant_object_array(record['environment_group'], count),
        'pair_id': _constant_object_array(record['pair_id'], count),
        'day_group': _constant_object_array(record['day_group'], count),
        'dataset_role': _constant_object_array(record['dataset_role'], count),
        'timing_quality_status': _constant_object_array(record['timing_quality_status'], count),
        'timing_quality_bucket': _constant_object_array(record['timing_quality_bucket'], count),
        'synthetic': np.full(count, bool(record['synthetic']), dtype=bool),
        'label_name': _constant_object_array(record['label_name'], count),
        'packet_index': np.asarray(rows['packet_index'], dtype=np.int32),
        'window_index': np.asarray(rows['evaluation_index'], dtype=np.int32),
        'reset_index': np.asarray(rows['reset_index'], dtype=np.int32),
    }


def _ensure_record_packets(record):
    """Materialize record packets only when a cache miss needs them."""
    packets = record.get('packets')
    if packets is not None:
        return packets
    loader = record.get('packets_loader')
    if callable(loader):
        packets = loader()
    else:
        packets = _load_npz_packets_cached(record['path'])
    record['packets'] = packets
    return packets


def _stable_text_seed(value):
    digest = hashlib.sha256(str(value).encode('utf-8')).digest()
    return int.from_bytes(digest[:4], byteorder='little') & 0x7FFFFFFF


def _add_scaled_iq_noise(raw, usable, noise_sigma, rng):
    """Add per-tone relative Gaussian noise with one vectorized RNG draw."""
    if usable <= 0 or noise_sigma <= 0.0:
        return
    tone_view = raw[:2 * usable].reshape(usable, 2)
    magnitudes = np.maximum(1.0, np.linalg.norm(tone_view, axis=1))
    noise_scale = noise_sigma * magnitudes[:, None] / np.sqrt(2.0)
    tone_view += rng.normal(0.0, noise_scale, size=(usable, 2))


def augment_csi_packets(packets, config, seed):
    """Return a deterministic packet-level augmented copy for training only."""
    if not config:
        return list(packets)
    noise_sigma = float(config.get('noise_sigma', 0.0))
    packet_loss = float(config.get('packet_loss', 0.0))
    stutter_probability = float(config.get('stutter_probability', 0.0))
    drift_sigma = float(config.get('drift_sigma', 0.0))
    drift_episode_count = int(config.get('drift_episode_count', 0))
    drift_duration_seconds = _normalize_augmentation_range(
        config.get('drift_duration_seconds', (20.0, 60.0)),
        name='drift_duration_seconds',
    )
    burst_loss_starts_per_minute = float(config.get('burst_loss_starts_per_minute', 0.0))
    burst_length_packets = _normalize_augmentation_range(
        config.get('burst_length_packets', (2, 6)),
        name='burst_length_packets',
        integer=True,
        minimum=1,
    )
    packet_rate_scale = _normalize_augmentation_range(
        config.get('packet_rate_scale', (1.0, 1.0)),
        name='packet_rate_scale',
        minimum=0.0,
    )
    if (
        min(
            noise_sigma,
            packet_loss,
            stutter_probability,
            drift_sigma,
            burst_loss_starts_per_minute,
        ) < 0.0
        or packet_loss >= 1.0
        or stutter_probability > 1.0
        or drift_episode_count < 0
        or packet_rate_scale[0] <= 0.0
        or packet_rate_scale[1] > 1.0
    ):
        raise ValueError("invalid packet augmentation parameters")

    grouped = {}
    for packet in packets:
        grouped.setdefault(str(packet.get('source_file', '__single_stream__')), []).append(packet)
    augmented = []
    for source in sorted(grouped):
        rng = np.random.default_rng(derive_seed(seed, _stable_text_seed(source)))
        source_packets = grouped[source]
        rate_scale = float(rng.uniform(*packet_rate_scale))
        source_packets = _resample_stable_packet_rate(source_packets, rate_scale)
        packet_rate_pps = _estimate_packet_rate_pps(source_packets)
        burst_start_probability = min(
            1.0,
            burst_loss_starts_per_minute / 60.0 / max(packet_rate_pps, 1e-9),
        )
        min_burst_packets, max_burst_packets = burst_length_packets
        drift_episodes = []
        if drift_sigma > 0.0 and drift_episode_count > 0 and source_packets:
            example_raw = np.asarray(source_packets[0]['csi_data'], dtype=np.float64)
            usable = len(example_raw) // 2
            if usable > 0:
                source_matrix = np.asarray(
                    [np.asarray(packet['csi_data'], dtype=np.float64)[:2 * usable] for packet in source_packets],
                    dtype=np.float64,
                ).reshape(len(source_packets), usable, 2)
                tone_scale = np.median(np.linalg.norm(source_matrix, axis=2), axis=0)
                tone_scale = np.maximum(tone_scale, 1.0)[:, None] / np.sqrt(2.0)
                min_seconds, max_seconds = drift_duration_seconds
                for _ in range(drift_episode_count):
                    duration_seconds = rng.uniform(min_seconds, max_seconds)
                    duration_packets = int(round(duration_seconds * packet_rate_pps))
                    duration_packets = max(2, min(duration_packets, len(source_packets)))
                    max_start = max(0, len(source_packets) - duration_packets)
                    start_index = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
                    template = _smoothed_iq_profile(rng, usable) * tone_scale
                    amplitude = float(rng.normal(0.0, drift_sigma))
                    drift_episodes.append((start_index, start_index + duration_packets, amplitude, template))
        previous_emitted = None
        burst_skip_remaining = 0
        for packet_index, packet in enumerate(source_packets):
            if burst_skip_remaining > 0:
                burst_skip_remaining -= 1
                continue
            if burst_start_probability > 0.0 and rng.random() < burst_start_probability:
                burst_skip_remaining = int(rng.integers(min_burst_packets, max_burst_packets + 1))
                if burst_skip_remaining > 0:
                    burst_skip_remaining -= 1
                    continue
            if packet_loss > 0.0 and rng.random() < packet_loss:
                continue
            raw = np.asarray(packet['csi_data'], dtype=np.float64)
            stuttered = False
            if (
                stutter_probability > 0.0
                and previous_emitted is not None
                and rng.random() < stutter_probability
            ):
                raw = previous_emitted.copy()
                stuttered = True
            else:
                raw = raw.copy()
            usable = len(raw) // 2
            if usable > 0 and drift_episodes and not stuttered:
                tone_view = raw[:2 * usable].reshape(usable, 2)
                for start_index, stop_index, amplitude, template in drift_episodes:
                    if start_index <= packet_index < stop_index:
                        phase = np.pi * float((packet_index - start_index) + 1) / float(
                            (stop_index - start_index) + 1
                        )
                        tone_view += np.sin(phase) * amplitude * template[:usable]
            _add_scaled_iq_noise(raw, usable, noise_sigma, rng)
            copied = dict(packet)
            emitted = np.clip(np.rint(raw), -128, 127).astype(np.int8)
            copied['csi_data'] = emitted
            augmented.append(copied)
            previous_emitted = emitted.astype(np.float64)
    return augmented


def load_training_matrix(environment_filter=None, excluded_chips=None,
                         feature_names=None, use_cache=True,
                         packet_augmentation=None, augmentation_seed=None,
                         augmentation_seeds=None,
                         dataset_roles=DEFAULT_TRAINING_ROLES,
                         timing_quality_policy=DEFAULT_TIMING_QUALITY_POLICY,
                         timing_warn_weight=DEFAULT_TIMING_WARN_WEIGHT):
    """Load the canonical reset-aware streaming feature matrix used by training."""
    if augmentation_seed is not None and augmentation_seeds is not None:
        raise ValueError("pass augmentation_seed or augmentation_seeds, not both")
    resolved_augmentation_seeds = (
        tuple(int(seed) for seed in augmentation_seeds)
        if augmentation_seeds is not None
        else tuple()
    )
    if len(set(resolved_augmentation_seeds)) != len(resolved_augmentation_seeds):
        raise ValueError("augmentation_seeds must not contain duplicates")
    if (
        packet_augmentation
        and augmentation_seed is None
        and not resolved_augmentation_seeds
    ):
        raise ValueError("packet augmentation requires a deterministic seed")
    if feature_names is None:
        feature_names = DEFAULT_FEATURES.copy()
    feature_names = list(feature_names)
    use_runtime_cache = _feature_rows_use_runtime_cache(feature_names)
    if use_cache and use_runtime_cache:
        print("  Training feature cache: enabled (canonical time-aware replay rows)")
    elif use_cache:
        print("  Training feature cache: enabled (host-side replay rows)")
    else:
        reason = []
        if not _feature_names_support_replay_rows(feature_names):
            reason.append("host-only feature extraction")
        detail = f" ({', '.join(reason)})" if reason else ""
        print(f"  Training feature cache: disabled{detail}")

    load_start = perf_counter()
    records, stats = _load_training_file_records(
        environment_filter=environment_filter,
        excluded_chips=excluded_chips,
        dataset_roles=dataset_roles,
        timing_quality_policy=timing_quality_policy,
        timing_warn_weight=timing_warn_weight,
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
        }, None

    print("\nExtracting features...")
    features_start = perf_counter()
    X_parts = []
    y_parts = []
    weight_parts = []
    context_parts = {
        'chip': [],
        'source_file': [],
        'lineage_group': [],
        'session_group': [],
        'environment_group': [],
        'pair_id': [],
        'day_group': [],
        'dataset_role': [],
        'timing_quality_status': [],
        'timing_quality_bucket': [],
        'synthetic': [],
        'label_name': [],
        'packet_index': [],
        'window_index': [],
        'reset_index': [],
    }
    cache_hits = 0
    cache_misses = 0
    actual_feature_names = feature_names

    for record in records:
        if packet_augmentation and len(resolved_augmentation_seeds) > 1:
            replay_rows = _load_or_compute_packet_augmentation_mix_rows(
                record,
                packet_augmentation=packet_augmentation,
                augmentation_seeds=resolved_augmentation_seeds,
                feature_names=feature_names,
                use_cache=use_cache,
                use_runtime_cache=use_runtime_cache,
            )
            cache_hit = bool(replay_rows.get('cache_hit', False))
        elif use_runtime_cache and packet_augmentation:
            resolved_seed = (
                resolved_augmentation_seeds[0]
                if resolved_augmentation_seeds
                else augmentation_seed
            )
            stream_provenance = _packet_augmentation_stream_provenance(
                packet_augmentation,
                resolved_seed,
            )
            replay_rows = load_or_compute_ml_replay_rows(
                record['path'],
                packets_factory=lambda current_record=record: (
                    _prepare_feature_packets_for_record(
                        current_record,
                        packet_augmentation=packet_augmentation,
                        augmentation_seed=resolved_seed,
                    )
                ),
                selected_subcarriers=DEFAULT_SUBCARRIERS,
                window_size=None,
                feature_names=feature_names,
                sample_contract=TRAINING_SAMPLE_CONTRACT,
                use_cache=use_cache and stream_provenance is not None,
                stream_provenance=stream_provenance,
            )
            cache_hit = bool(replay_rows.get('cache_hit', False))
        elif use_runtime_cache:
            replay_rows = load_or_compute_ml_replay_rows(
                record['path'],
                selected_subcarriers=DEFAULT_SUBCARRIERS,
                window_size=None,
                feature_names=feature_names,
                use_cache=use_cache,
                sample_contract=TRAINING_SAMPLE_CONTRACT,
            )
            cache_hit = bool(replay_rows.get('cache_hit', False))
        else:
            resolved_seed = (
                resolved_augmentation_seeds[0]
                if resolved_augmentation_seeds
                else augmentation_seed
            )
            stream_provenance = _host_feature_stream_provenance(
                feature_names,
                packet_augmentation=packet_augmentation,
                augmentation_seed=resolved_seed,
            )
            replay_rows = load_or_compute_host_feature_rows(
                record['path'],
                packets_factory=lambda current_record=record: (
                    _prepare_feature_packets_for_record(
                        current_record,
                        packet_augmentation=packet_augmentation,
                        augmentation_seed=resolved_seed,
                    )
                ),
                feature_names=feature_names,
                sample_contract=TRAINING_SAMPLE_CONTRACT,
                use_cache=use_cache,
                stream_provenance=stream_provenance,
            )
            cache_hit = bool(replay_rows.get('cache_hit', False))
        file_matrix = {
            'X': np.asarray(replay_rows['X'], dtype=np.float32),
            'feature_names': list(replay_rows['feature_names']),
        }
        if cache_hit:
            cache_hits += 1
        else:
            cache_misses += 1
        X_file = np.asarray(file_matrix['X'], dtype=np.float32)
        actual_feature_names = list(file_matrix['feature_names'])
        if len(X_file) == 0:
            continue
        X_parts.append(X_file)
        y_parts.append(
            np.full(len(X_file), 1 if record['is_motion'] else 0, dtype=np.int8)
        )
        weight_parts.append(
            np.full(
                len(X_file),
                float(record.get('timing_weight', 1.0)),
                dtype=np.float32,
            )
        )
        sample_context = _build_sample_context_for_replay_rows(record, replay_rows)
        for key, values in sample_context.items():
            context_parts[key].append(values)

    print(
        f"  Feature cache files: {cache_hits} hit(s), {cache_misses} miss(es)"
    )
    print(f"  Feature extraction time: {format_duration(perf_counter() - features_start)}")

    if X_parts:
        X = np.concatenate(X_parts, axis=0)
        y = np.concatenate(y_parts, axis=0)
        sample_context = {
            key: np.concatenate(parts, axis=0)
            for key, parts in context_parts.items()
            if parts
        }
        sample_weights = np.concatenate(weight_parts, axis=0).astype(np.float32, copy=False)
        mean_weight = float(np.mean(sample_weights))
        if mean_weight > 1e-6:
            sample_weights /= np.float32(mean_weight)
    else:
        X = np.empty((0, len(actual_feature_names)), dtype=np.float32)
        y = np.asarray([], dtype=np.int8)
        sample_context = {
            'chip': np.empty(0, dtype=object),
            'source_file': np.empty(0, dtype=object),
            'lineage_group': np.empty(0, dtype=object),
            'session_group': np.empty(0, dtype=object),
            'environment_group': np.empty(0, dtype=object),
            'pair_id': np.empty(0, dtype=object),
            'day_group': np.empty(0, dtype=object),
            'dataset_role': np.empty(0, dtype=object),
            'timing_quality_status': np.empty(0, dtype=object),
            'timing_quality_bucket': np.empty(0, dtype=object),
            'synthetic': np.empty(0, dtype=bool),
            'label_name': np.empty(0, dtype=object),
            'packet_index': np.empty(0, dtype=np.int32),
            'window_index': np.empty(0, dtype=np.int32),
            'reset_index': np.empty(0, dtype=np.int32),
        }
        sample_weights = np.asarray([], dtype=np.float32)

    matrix = {
        'X': X,
        'y': y,
        'feature_names': actual_feature_names,
        'sample_context': sample_context,
        'sample_weights': sample_weights,
        'stats': stats,
    }
    return matrix, None


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


def normalized_feature_bounds(preprocessor, feature_names):
    """Return normalized bounds used to keep augmented feature rows valid."""
    center, scale = get_preprocessor_arrays(preprocessor)
    lower = np.full(len(feature_names), -np.inf, dtype=np.float32)
    upper = np.full(len(feature_names), np.inf, dtype=np.float32)
    for name in (
        'turb_mad_over_mean',
        'turb_iqr_over_mean',
        'turb_p95_over_mean',
        'turb_mad_over_mean_aggr',
        'turb_iqr_over_mean_aggr',
        'turb_p95_over_mean_aggr',
    ):
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


def exported_weight_matrices(weights_module):
    """Return exported matrices in the host [input][output] convention."""
    if hasattr(weights_module, 'WEIGHTS_T'):
        return [
            np.asarray(layer, dtype=np.float32).T
            for layer in weights_module.WEIGHTS_T
        ]
    return [
        np.asarray(layer, dtype=np.float32)
        for layer in weights_module.WEIGHTS
    ]


def predict_exported_probabilities_from_weights(weights_module, X_raw):
    """Vectorized inference matching src/python/micro_espectre/high_accuracy_detector.py for exported weights."""
    center = np.asarray(weights_module.FEATURE_MEAN, dtype=np.float32)
    scale = np.asarray(weights_module.FEATURE_SCALE, dtype=np.float32)
    scale[scale < 1e-6] = 1.0
    matrices = exported_weight_matrices(weights_module)
    layers = [
        (
            layer_weights,
            np.asarray(layer_biases, dtype=np.float32),
            layer_index == len(matrices) - 1,
        )
        for layer_index, (layer_weights, layer_biases) in enumerate(
            zip(matrices, weights_module.BIASES)
        )
    ]
    return predict_probabilities_from_arrays(X_raw, center, scale, layers)


def predict_probabilities_from_arrays(features, center, scale, layers):
    """Run the shared exported/runtime-array inference implementation."""
    features = np.asarray(features, dtype=np.float32)
    if features.size == 0:
        return np.zeros(0, dtype=np.float32)
    center = np.asarray(center, dtype=np.float32)
    scale = np.asarray(scale, dtype=np.float32).copy()
    scale[scale < 1e-6] = 1.0
    activations = (features - center) / scale
    for weights, biases, is_output in layers:
        activations = (
            activations @ np.asarray(weights, dtype=np.float32)
            + np.asarray(biases, dtype=np.float32)
        )
        if not is_output:
            activations = np.maximum(activations, 0.0)

    logits = activations.reshape(-1).astype(np.float32, copy=False)
    probabilities = np.empty(logits.shape, dtype=np.float32)
    probabilities[logits < -20.0] = 0.0
    probabilities[logits > 20.0] = 1.0
    mask = (logits >= -20.0) & (logits <= 20.0)
    probabilities[mask] = 1.0 / (1.0 + np.exp(-logits[mask]))
    return probabilities


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
    matrix, _ = load_training_matrix(
        environment_filter=environment_filter,
        excluded_chips=excluded_chips,
        feature_names=feature_names,
    )
    X = np.asarray(matrix['X'], dtype=np.float32)
    y = np.asarray(matrix['y'], dtype=np.int8)
    actual_feature_names = list(matrix['feature_names'])
    sample_context = matrix['sample_context']
    stats = matrix['stats']
    if len(X) == 0:
        raise RuntimeError("No empty/static_presence/motion packets found for gain stress gate")

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


def evaluate_candidate_gain_stress(model, scaler, feature_names, *,
                                   environment_filter=None, excluded_chips=None,
                                   dataset_roles=('train', 'selection', 'holdout'),
                                   scales=DEFAULT_GAIN_STRESS_SCALES):
    """Evaluate one in-memory candidate under artificial gain scaling."""
    environment_filter = parse_environment_filter(environment_filter)
    excluded_chips = parse_chip_filter(excluded_chips)
    scales = parse_gain_stress_scales(scales)
    center, scale = get_preprocessor_arrays(scaler)
    layers = _layer_arrays_from_model(model)
    matrix, _ = load_training_matrix(
        environment_filter=environment_filter,
        excluded_chips=excluded_chips,
        feature_names=feature_names,
        use_cache=True,
        dataset_roles=dataset_roles,
    )
    X = np.asarray(matrix['X'], dtype=np.float32)
    y = np.asarray(matrix['y'], dtype=np.int8)
    actual_feature_names = list(matrix['feature_names'])
    sample_context = matrix['sample_context']
    stats = matrix['stats']
    if len(X) == 0:
        raise RuntimeError("No empty/static_presence/motion packets found for gain stress gate")
    if list(actual_feature_names) != list(feature_names):
        raise RuntimeError(
            "Extracted feature order does not match trained model: "
            f"{actual_feature_names} != {feature_names}"
        )

    scaled_features = [
        name for name in feature_names if name in GAIN_SENSITIVE_FEATURES
    ]
    results = {
        'feature_names': list(feature_names),
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
        y_prob = _batch_predict_probabilities(X_stressed, center, scale, layers)
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


def print_gain_stress_summary(results, title="EXPORTED ML GAIN-STRESS GATE"):
    """Print a compact gain stress report."""
    stats = results.get('stats', {})
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)
    print(f"Samples: {results['samples']}")
    print(f"Chips: {', '.join(stats.get('chips', []))}")
    if stats.get('environment_groups'):
        print(f"Environments: {', '.join(stats['environment_groups'])}")
    print(f"Scaled features: {', '.join(results['scaled_features']) or 'none'}")
    print(f"Invariant features: {', '.join(results['invariant_features']) or 'none'}")
    if not results['scaled_features']:
        print("Note: current export has no gain-sensitive feature dimensions; this")
        print("gate is informational and acts as a regression guard against")
        print("reintroducing them.")
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
    Return probabilities through the deployment-equivalent logit mapping.
    """
    return predict_runtime_probabilities(model, X)


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
        # SHAP must describe clean, held-out deployment windows while the model
        # retains the promoted train-time augmentation recipe. Keep an explicit
        # view of the clean fold before packet-augmented rows are appended.
        X_train_clean_scaled = X_train_scaled
        y_train_clean = y_train_fold
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
                y_train_clean,
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
                X_train_clean_scaled[background_idx],
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
                                   excluded_chips=None,
                                   block_stride=DEFAULT_WINDOW_PACKETS_AT_NOMINAL_RATE,
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
        augment: Optional augmentation component set for the held-out runs.

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
    augment_components, feature_augmentation, packet_augmentation = resolve_training_augmentation(augment)

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
        f"{format_augmentation_config(feature_augmentation, packet_augmentation, components=augment_components)}"
    )
    if packet_augmentation:
        print(
            "Packet augmentation seeds: "
            + ", ".join(str(seed) for seed in FIXED_PACKET_AUGMENTATION_SEEDS)
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
            augmentation_seeds=training_packet_augmentation_seeds(
                packet_augmentation
            ),
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


# Production shortcut for --augment: feature jitter plus a stable lower-rate
# stream and moderate packet noise, loss, and stutter.
ROBUSTNESS_WINNER_NAME = (
    'baseline_standard__feature_jitter_010__packet_rate_noise_loss_stutter_moderate'
)
ROBUSTNESS_WINNER_FEATURE_AUGMENTATION = {'jitter_sigma': 0.10}
ROBUSTNESS_WINNER_PACKET_AUGMENTATION = {
    'noise_sigma': 0.01,
    'packet_loss': 0.05,
    'stutter_probability': 0.08,
    'packet_rate_scale': (0.8, 1.0),
}
CORRELATED_DRIFT_PACKET_AUGMENTATION = {
    'drift_sigma': 0.035,
    'drift_episode_count': 1,
    'drift_duration_seconds': (20.0, 60.0),
}
BURST_LOSS_PACKET_AUGMENTATION = {
    'burst_loss_starts_per_minute': 0.6,
    'burst_length_packets': (2, 6),
}
TRAINING_AUGMENT_COMPONENTS = {
    'base': {
        'feature': dict(ROBUSTNESS_WINNER_FEATURE_AUGMENTATION),
        'packet': dict(ROBUSTNESS_WINNER_PACKET_AUGMENTATION),
    },
    'drift': {
        'feature': {},
        'packet': dict(CORRELATED_DRIFT_PACKET_AUGMENTATION),
    },
    'burst-loss': {
        'feature': {},
        'packet': dict(BURST_LOSS_PACKET_AUGMENTATION),
    },
}


def _merge_augmentation_dicts(*configs):
    """Merge augmentation config fragments into one plain dict."""
    merged = {}
    for config in configs:
        for key, value in dict(config or {}).items():
            merged[key] = tuple(value) if isinstance(value, (list, tuple)) else value
    return merged


def resolve_training_augmentation(augment):
    """Return active components plus merged augmentation configs."""
    components = parse_augmentation_components(augment)
    if not components:
        return tuple(), {}, {}
    feature_parts = [TRAINING_AUGMENT_COMPONENTS[name]['feature'] for name in components]
    packet_parts = [TRAINING_AUGMENT_COMPONENTS[name]['packet'] for name in components]
    return (
        components,
        _merge_augmentation_dicts(*feature_parts),
        _merge_augmentation_dicts(*packet_parts),
    )


def format_augmentation_config(feature_augmentation=None, packet_augmentation=None, *, components=None):
    """Compact one-line description of an active training augmentation recipe."""
    feature_augmentation = dict(feature_augmentation or {})
    packet_augmentation = dict(packet_augmentation or {})
    if not feature_augmentation and not packet_augmentation:
        return 'none'
    normalized_components = parse_augmentation_components(components)
    parts = []
    if normalized_components:
        parts.append(f"components={'+'.join(normalized_components)}")
    elif feature_augmentation == ROBUSTNESS_WINNER_FEATURE_AUGMENTATION and packet_augmentation == ROBUSTNESS_WINNER_PACKET_AUGMENTATION:
        parts.append(f"recipe={ROBUSTNESS_WINNER_NAME}")
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


def get_model_architecture(model):
    """Return the layer sizes of a dense MLP as [input, ..., output]."""
    weights = extract_model_weights(model)
    if not weights:
        return []

    layer_sizes = [int(weights[0].shape[0])]
    for idx in range(0, len(weights), 2):
        layer_sizes.append(int(weights[idx].shape[1]))
    return layer_sizes


def render_micropython_weights(weights, center, scale, architecture, seed=None,
                               feature_names=None,
                               scaler_mode=DEFAULT_SCALER_MODE,
                               trained_at=None):
    """Render inference-ready MicroPython weights without a runtime transpose."""
    from datetime import datetime
    if feature_names is None:
        feature_names = list(TRAINING_FEATURES)
    seed_info = f"Seed: {seed}"
    timestamp = trained_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    architecture_text = ' -> '.join(map(str, architecture))
    architecture_csv = ', '.join(str(x) for x in architecture)
    hidden_csv = ', '.join(str(x) for x in architecture[1:-1])
    feature_csv = ', '.join(repr(name) for name in feature_names)
    center_csv = ', '.join(f'{x:.9g}' for x in center)
    scale_csv = ', '.join(f'{x:.9g}' for x in scale)
    
    # Build code - weights only
    code = f'''# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""
Micro-ESPectre - ML Model Weights

Auto-generated neural network weights for motion detection.
Architecture: {architecture_text}
Normalization: {scaler_mode}
Trained: {timestamp}
{seed_info}

This file is auto-generated by train_ml_model.py.
DO NOT EDIT - your changes will be overwritten!

Author: Francesco Pace <francesco.pace@gmail.com>
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
    
    # Store each matrix as [output][input], matching the inference loop. This
    # avoids retaining a second forest of nested lists while transposing at
    # import time on memory-constrained MicroPython targets.
    weight_names = []
    bias_names = []
    for i in range(0, len(weights), 2):
        W = weights[i]
        b = weights[i + 1]
        layer_num = i // 2 + 1
        in_size, out_size = W.shape
        
        activation = 'Sigmoid' if i == len(weights) - 2 else 'ReLU'
        code += f'# Layer {layer_num}: {in_size} -> {out_size} ({activation})\n'
        code += f'WT{layer_num} = [\n'
        for row in W.T:
            code += '    [' + ', '.join(f'{x:.9g}' for x in row) + '],\n'
        code += ']\n'
        code += f'B{layer_num} = [' + ', '.join(f'{x:.9g}' for x in b) + ']\n\n'
        weight_names.append(f'WT{layer_num}')
        bias_names.append(f'B{layer_num}')

    code += f'WEIGHTS_T = [{", ".join(weight_names)}]\n'
    code += f'BIASES = [{", ".join(bias_names)}]\n'
    return code


def export_micropython(model, scaler, output_path, seed=None,
                       feature_names=None, scaler_mode=DEFAULT_SCALER_MODE,
                       trained_at=None):
    """
    Export model weights to MicroPython code.

    Generates ml_weights.py with inference-ready transposed network weights.
    The inference functions are in high_accuracy_detector.py (not auto-generated).
    """
    weights = extract_model_weights(model)
    center, scale = get_preprocessor_arrays(scaler)
    architecture = get_model_architecture(model)
    code = render_micropython_weights(
        weights,
        center,
        scale,
        architecture,
        seed=seed,
        feature_names=feature_names,
        scaler_mode=scaler_mode,
        trained_at=trained_at,
    )
    atomic_write_text(output_path, code)
    return len(code)


# Canonical C++ feature ids, mirroring the MLFeatureId enum in
# src/cpp/core/csi_features.h. Keep the numeric values in sync. Only features
# with a real C++ extractor entry can be exported to firmware.
CPP_FEATURE_IDS = {
    'turb_autocorr': 6,
    'turb_zcr': 14,
    'l1_delta_lag_ratio': 25,
    'turb_iqr_over_mean_aggr': 45,
    'chan_shape_coherent_innovation_energy': 46,
    'chan_shape_excess_path': 47,
    'chan_shape_spread_subband': 48,
}
def resolve_cpp_feature_ids(feature_names):
    """Map feature names to their published C++ extractor ids."""
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
                       feature_names=None, scaler_mode=DEFAULT_SCALER_MODE,
                       trained_at=None):
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
        trained_at: Optional training timestamp preserved during metadata-only regeneration
    
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
    timestamp = trained_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
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
    
    atomic_write_text(output_path, code)
    
    return len(code)


def export_test_data(model, scaler, X_test_raw, y_test, output_path):
    """
    Export test data for validation across Python and C++.

    Generates ml_test_data.npz with RAW features (not normalized) and expected outputs.
    This allows testing the full pipeline including normalization.

    The artifact is committed, so it carries only what the host and C++
    regression suites read: raw features, labels, and expected outputs. It
    holds no object arrays and stays loadable with ``allow_pickle=False``.

    Args:
        model: Trained PyTorch model
        scaler: Fitted preprocessing object used for normalization
        X_test_raw: Test features (NOT normalized, raw values)
        y_test: Test labels
        output_path: Output file path

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
    atomic_savez(output_path, payload)
    
    return len(X_test_raw)


# ============================================================================
# Feature Importance (Correlation)
# ============================================================================

def calculate_correlation_importance(feature_names=None, use_cache=True):
    """
    Calculate correlation of selected training features with motion label.
    
    This is a fast alternative to SHAP for initial feature screening.
    Reuses the canonical time-aware training matrix.
    
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

    print_candidate_redundancy(X, actual_features)

    return sorted_corr


def print_candidate_redundancy(X, feature_names, baseline_features=None):
    """Report how much of each candidate the production set already explains.

    A candidate earns its place by what it adds, not by how well it separates on
    its own, so the screening question is redundancy: its strongest pairwise
    correlation against the production members, and the share of its variance a
    least-squares fit on all of them removes.
    """
    if baseline_features is None:
        baseline_features = DEFAULT_FEATURES
    names = list(feature_names)
    baseline_index = [i for i, name in enumerate(names) if name in baseline_features]
    candidate_index = [i for i, name in enumerate(names) if name not in baseline_features]
    if not candidate_index or not baseline_index:
        return

    design = np.column_stack(
        [X[:, baseline_index], np.ones(len(X), dtype=X.dtype)]
    )
    print("\n" + "=" * 74)
    print("  Candidate Redundancy Against The Production Set")
    print("=" * 74)
    print(f"{'Candidate':<22} {'max |r| vs production':>22} {'closest':>16} {'R2':>8}")
    print("-" * 74)
    for i in candidate_index:
        values = X[:, i]
        strongest, closest = 0.0, "-"
        for j in baseline_index:
            if values.std() < 1e-12 or X[:, j].std() < 1e-12:
                continue
            r = abs(float(np.corrcoef(values, X[:, j])[0, 1]))
            if r > strongest:
                strongest, closest = r, names[j]
        coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
        residual = values - design @ coefficients
        variance = float(values.var())
        r_squared = 1.0 - float(residual.var()) / variance if variance > 0 else 0.0
        print(f"{names[i]:<22} {strongest:>22.4f} {closest:>16} {r_squared:>8.4f}")
    print("-" * 74)
    print("  Lower is better: a candidate the production set can reconstruct "
          "adds nothing.")


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
            block_stride=DEFAULT_WINDOW_PACKETS_AT_NOMINAL_RATE,
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
                block_stride=DEFAULT_WINDOW_PACKETS_AT_NOMINAL_RATE,
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


def select_regression_subset_indices(
    sample_context,
    max_samples=2048,
    block_stride=DEFAULT_WINDOW_PACKETS_AT_NOMINAL_RATE,
):
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
              force_export=False,
              environment_filter=None, excluded_chips=None,
              positive_chip_boost=None,
              use_cache=True, augment=False,
              timing_quality_policy=DEFAULT_TIMING_QUALITY_POLICY,
              timing_warn_weight=DEFAULT_TIMING_WARN_WEIGHT):
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
        force_export: Export runtime artifacts even when the deployment
                      safety gates fail or regress. Gates still run and their
                      results are printed; the bypass is reported loudly.
        environment_filter: Optional environment name(s) to keep.
        excluded_chips: Optional chip name(s) to exclude.
        positive_chip_boost: Optional {CHIP: factor} boost applied to motion
                             samples after feature extraction.
        use_cache: If True, reuse the cached feature matrix.
        augment: Optional augmentation component set. ``base`` keeps the
                 current validated recipe, while ``drift`` and ``burst-loss``
                 add their named components.
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
    augment_components, feature_augmentation, packet_augmentation = resolve_training_augmentation(augment)
    if hidden_layers is None:
        hidden_layers = list(DEFAULT_HIDDEN_LAYERS)
    if feature_names is None:
        feature_names = DEFAULT_FEATURES.copy()
    feature_names = list(feature_names)
    if export_artifacts:
        unsupported = [name for name in feature_names if name not in CPP_FEATURE_IDS]
        if unsupported:
            print(
                "Error: runtime export requires a C++ extractor id for every "
                f"feature; missing: {', '.join(unsupported)}"
            )
            return 1, seed, None
    
    print("\n" + "="*60)
    print("           ML MOTION DETECTOR TRAINING")
    print("="*60 + "\n")
    
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
        timing_quality_policy=timing_quality_policy,
        timing_warn_weight=timing_warn_weight,
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
            augmentation_seeds=training_packet_augmentation_seeds(
                packet_augmentation
            ),
            timing_quality_policy=timing_quality_policy,
            timing_warn_weight=timing_warn_weight,
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
    if stats.get('excluded_long_recordings'):
        print(
            "  Long-recording empty replays excluded from training: "
            + str(len(stats['excluded_long_recordings']))
        )
    if stats.get('environment_groups'):
        print(f"  Named environments: {len(stats['environment_groups'])}")
    timing_counts = stats.get('timing_quality_counts', {})
    print(
        "  Timing provenance: "
        f"clean={timing_counts.get('clean', 0)}, "
        f"degraded={timing_counts.get('degraded', 0)}, "
        f"poor={timing_counts.get('poor', 0)}, "
        f"unknown={timing_counts.get('unknown', 0)}"
    )
    if stats.get('excluded_timing_quality'):
        print(
            "  Poor-timing files excluded: "
            + str(len(stats['excluded_timing_quality']))
        )
    for label, count in sorted(stats['labels'].items()):
        print(f"  {label}: {count} packets")
    print(f"  Total: {stats['total']} packets")
    
    print(f"Architecture: {' -> '.join(map(str, [len(feature_names)] + hidden_layers + [1]))}")
    print(f"Scaler: {scaler_mode}")
    print(f"Batch size: {batch_size}")
    print(f"Training sample contract: {TRAINING_SAMPLE_CONTRACT} (only supported contract)")
    print(
        "Augmentation: "
        f"{format_augmentation_config(feature_augmentation, packet_augmentation, components=augment_components)}"
    )
    if packet_augmentation:
        print(
            "Packet augmentation seeds: "
            + ", ".join(str(seed) for seed in FIXED_PACKET_AUGMENTATION_SEEDS)
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
    print(
        "  Evaluation block stride: "
        f"{DEFAULT_WINDOW_PACKETS_AT_NOMINAL_RATE} windows per source file"
    )

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
    if timing_quality_policy != DEFAULT_TIMING_QUALITY_POLICY:
        print(
            "  Timing quality policy: "
            f"{timing_quality_policy}"
            + (
                f" (warn weight={float(timing_warn_weight):.2f})"
                if "downweight-warn" in str(timing_quality_policy)
                else ""
            )
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
            block_stride=DEFAULT_WINDOW_PACKETS_AT_NOMINAL_RATE,
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
        gate_progress = lambda message: print(f"  {message}")
        paired_gate = evaluate_paired_gate(
            model,
            scaler,
            actual_feature_names,
            roles=deployment_roles,
            allow_legacy_fallback=allow_legacy_gate_fallback,
            progress=gate_progress,
        )
        quiet_gate = evaluate_quiet_gate(
            model,
            scaler,
            actual_feature_names,
            roles=deployment_roles,
            progress=gate_progress,
        )
        gain_stress = evaluate_candidate_gain_stress(
            model,
            scaler,
            actual_feature_names,
            environment_filter=environment_filter,
            excluded_chips=excluded_chips,
            dataset_roles=deployment_roles,
        )
        cv_results['paired'] = paired_gate
        cv_results['quiet'] = quiet_gate
        cv_results['gain_stress'] = gain_stress
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
        print_gain_stress_summary(gain_stress, title="IN-MEMORY ML GAIN-STRESS GATE")
        paired_total = len(paired_gate.get('by_chip', {}))
        if export_artifacts and (
            paired_total == 0
            or paired_gate['pass_count'] < paired_total
            or (quiet_gate is not None and not quiet_gate['passed'])
        ):
            if force_export:
                print(
                    "WARNING: deployment safety gate FAILED; exporting anyway "
                    "because --force-promote bypasses the promotion rules"
                )
            else:
                print("Error: deployment safety gate failed; runtime artifacts were not exported")
                return 1, seed, cv_results
        try:
            baseline_paired = evaluate_exported_paired_gate(
                roles=deployment_roles,
                allow_legacy_fallback=allow_legacy_gate_fallback,
            )
        except (FileNotFoundError, ImportError, AttributeError) as exc:
            baseline_paired = None
            print(f"  Exported baseline unavailable ({exc}); using absolute paired gate")
        if baseline_paired is not None:
            cv_results['baseline_paired'] = baseline_paired
            print(
                f"  Baseline paired: pass={baseline_paired['pass_count']} "
                f"maxFP={baseline_paired['max_fp_rate']:.2f}% "
                f"worstRecall={baseline_paired['worst_chip_recall']:.2f}%"
            )
            paired_failures = paired_non_regression_failures(
                paired_gate, baseline_paired)
            cv_results['paired_non_regression_failures'] = paired_failures
            if paired_failures:
                label = (
                    "Blocked by per-recording non-regression on:"
                    if export_artifacts
                    else "Per-recording non-regression failures:"
                )
                print(f"  {label}")
                print(format_non_regression_failures(paired_failures, indent='    '))
                if export_artifacts and force_export:
                    print(
                        "WARNING: candidate regresses the paired deployment "
                        "gate; exporting anyway because --force-promote "
                        "bypasses the promotion rules"
                    )
                elif export_artifacts:
                    print(
                        "Error: candidate regresses the paired deployment gate; "
                        "runtime artifacts were not exported"
                    )
                    return 1, seed, cv_results

    if not export_artifacts:
        print("\nArtifacts unchanged.")
        return 0, seed, cv_results

    regression_indices = select_regression_subset_indices(
        sample_context,
        max_samples=2048,
        block_stride=DEFAULT_WINDOW_PACKETS_AT_NOMINAL_RATE,
    )
    
    # Export models
    print("\nExporting model artifacts...")
    export_start = perf_counter()

    mp_path = SRC_DIR / 'ml_weights.py'
    cpp_path = CPP_DIR / 'ml_weights.h'
    test_data_path = GENERATED_DATA_DIR / 'ml_test_data.npz'
    with tempfile.TemporaryDirectory(prefix='espectre_model_export_') as staging:
        staging_dir = Path(staging)
        staged_mp_path = staging_dir / mp_path.name
        staged_cpp_path = staging_dir / cpp_path.name
        staged_test_data_path = staging_dir / test_data_path.name
        mp_size = export_micropython(
            model, scaler, staged_mp_path,
            seed=seed,
            feature_names=actual_feature_names,
            scaler_mode=scaler_mode,
        )
        cpp_size = export_cpp_weights(
            model, scaler, staged_cpp_path,
            seed=seed,
            feature_names=actual_feature_names,
            scaler_mode=scaler_mode,
        )
        with suppress_stderr():
            n_test = export_test_data(
                model,
                scaler,
                X[regression_indices],
                y[regression_indices],
                staged_test_data_path,
            )
        atomic_write_set({
            mp_path: staged_mp_path.read_bytes(),
            cpp_path: staged_cpp_path.read_bytes(),
            test_data_path: staged_test_data_path.read_bytes(),
        })

    print(f"  MicroPython weights: {mp_path.name} ({mp_size/1024:.1f} KB)")
    print(f"  C++ weights: {cpp_path.name} ({cpp_size/1024:.1f} KB)")
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


# Absolute alarm budget for one static-presence replay. One sustained
# micro-motion of the present person is genuine motion, not model noise: the
# 2026-07-23 diagnosis located a ~1 s coherent event (four consecutive
# evaluations at p>0.94, ~1m51s into the C3 selection capture) that the
# exported baseline and every seed-search candidate detect identically.
# Quiet `empty` replays keep a zero-alarm requirement, and the per-recording
# non-regression checks still forbid exceeding the exported baseline's alarms
# on any individual replay.
PAIRED_ALARM_BUDGET = 1


def _gate_row_passes(row):
    """Per-replay pass criterion under the link-class policy.

    Normal-link replays keep the strict production bar, allowing at most
    ``PAIRED_ALARM_BUDGET`` runtime-filtered alarms for real micro-motion of
    the present person. Real weak-link (`low_rssi`) replays are stress
    diagnostics: they use the relaxed stress targets, and their alarms are
    reported but bounded only by the per-recording non-regression checks.
    """
    if row.get('low_rssi'):
        return (
            row['recall'] > STRESS_TARGET_RECALL
            and row['fp_rate'] < STRESS_TARGET_FP_RATE
        )
    return (
        row['recall'] > DEFAULT_GATE_TARGET_RECALL
        and row['fp_rate'] < DEFAULT_GATE_TARGET_FP_RATE
        and row.get('effective_alarms', 0) <= PAIRED_ALARM_BUDGET
    )


def summarize_gate(by_chip):
    """Aggregate per-chip gate metrics."""
    rows = list(by_chip.values())
    if not rows:
        return None
    return {
        'by_chip': by_chip,
        'pass_count': int(sum(1 for row in rows if _gate_row_passes(row))),
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


def evaluate_runtime_policy_evaluations(raw_motion_states):
    """Apply production hit filtering to states already sampled at eval ticks."""
    policy = RuntimeMotionPolicy(
        evaluation_interval_ms=EVALUATION_INTERVAL_MS,
        motion_on_hits=MOTION_ON_HITS,
        motion_off_hits=MOTION_OFF_HITS,
    )
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
    """Compatibility wrapper around the shared runtime-array inference."""
    return predict_probabilities_from_arrays(features, center, scale, layers)


class StreamingFeatureExtractor:
    """Compute runtime-equivalent feature vectors from a CSI packet stream."""

    def __init__(
        self,
        feature_names,
        window_packets=DEFAULT_WINDOW_PACKETS_AT_NOMINAL_RATE,
        packet_interval_us=None,
    ):
        self.feature_names = list(feature_names)
        self.window_packets = max(1, int(window_packets))
        self.packet_interval_us = max(
            1,
            int(
                packet_interval_us
                if packet_interval_us is not None
                else nominal_packet_interval_us(self.window_packets)
            ),
        )
        self.trajectory_elapsed_us = 0
        self.trajectory_packet_count = 0
        self.context = SegmentationContext(
            window_size=self.window_packets,
            enable_lowpass=ENABLE_LOWPASS_FILTER,
            lowpass_cutoff=LOWPASS_CUTOFF,
            enable_hampel=ENABLE_HAMPEL_FILTER,
            hampel_window=HAMPEL_WINDOW,
            hampel_threshold=HAMPEL_THRESHOLD,
        )
        # L1 features share the Hampel-filtered delta stream used by training
        # and both runtimes; skip the tracker when the model has no L1 inputs.
        self.needs_l1_tracker = _needs_l1_tracker(self.feature_names)
        self.needs_l1_series = _needs_l1_series(self.feature_names)
        l1_capacity = max(2, self.window_packets - L1_DELTA_LAG)
        self.l1_tracker = (
            L1DeltaTracker(
                window_size=l1_capacity,
                lag=L1_DELTA_LAG,
                allocate_amplitude_buffer=False,
                enable_hampel=ENABLE_HAMPEL_FILTER,
                hampel_window=HAMPEL_WINDOW,
                hampel_threshold=HAMPEL_THRESHOLD,
            )
            if self.needs_l1_tracker else None
        )
        self.l1_series = [0.0] * l1_capacity if self.needs_l1_series else None
        # Candidate features run through the same streaming path as production
        # ones, so the replay gates measure a candidate the way a runtime would.
        self.production_names, self.candidate_names = split_feature_names(
            self.feature_names
        )
        self.aggregated_context = (
            SegmentationContext(
                window_size=self.window_packets,
                enable_lowpass=ENABLE_LOWPASS_FILTER,
                lowpass_cutoff=LOWPASS_CUTOFF,
                enable_hampel=ENABLE_HAMPEL_FILTER,
                hampel_window=HAMPEL_WINDOW,
                hampel_threshold=HAMPEL_THRESHOLD,
                adjacent_aggregation_width=TURB_IQR_AGGREGATION_WIDTH,
            )
            if (
                any(
                    name in AGGREGATED_TURBULENCE_FEATURES
                    for name in self.feature_names
                )
                or needs_aggregated_turbulence(self.feature_names)
            ) else None
        )
        self.coherence_tracker = (
            ChannelCoherenceTracker(
                window_size=l1_capacity,
                lag=L1_DELTA_LAG,
                track_subbands=needs_subband_coherence(self.feature_names),
            )
            if needs_channel_coherence(self.feature_names) else None
        )
        self.phase_tracker = (
            PhaseResidualTracker(window_size=l1_capacity, lag=L1_DELTA_LAG)
            if needs_phase_residual(self.feature_names) else None
        )
        self.shape_tracker = (
            ChannelShapeTracker(
                window_size=l1_capacity,
                lag=L1_DELTA_LAG,
                feature_names=self.feature_names,
            )
            if needs_channel_shape(self.feature_names) else None
        )
        self.shape_trajectory_tracker = (
            ChannelShapeTrajectoryTracker(
                window_duration_us=SEGMENTATION_WINDOW_SIZE_MS * 1000,
                bin_us=ACTIVE_TRAJECTORY_BIN_US,
                track_subband_rank_gap=(
                    'chan_shape_subband_rank_gap' in self.feature_names
                ),
                track_subband_kendall_lag_excess=(
                    'chan_shape_subband_kendall_lag_excess'
                    in self.feature_names
                ),
            )
            if needs_channel_shape_trajectory(self.feature_names) else None
        )
        self.amplitude_profile_tracker = (
            AmplitudeProfileTracker(window_size=self.window_packets)
            if needs_amplitude_profiles(self.feature_names) else None
        )

    @staticmethod
    def _ordered_series(context):
        """Return chronological turbulence values and missing-slot validity."""
        count = context.buffer_count
        if count < context.window_size:
            return (
                list(context.turbulence_buffer[:count]),
                list(context.validity_buffer[:count]),
            )
        index = context.buffer_index
        return (
            list(context.turbulence_buffer[index:])
            + list(context.turbulence_buffer[:index]),
            list(context.validity_buffer[index:])
            + list(context.validity_buffer[:index]),
        )

    def _trajectory_timestamp_us(self, packet, timestamp_us=None):
        if timestamp_us is not None:
            return int(timestamp_us)
        resolved = packet_timestamp_us(
            packet,
            fallback_index=self.trajectory_packet_count,
            fallback_interval_us=self.packet_interval_us,
        ) if packet is not None else None
        if resolved is not None:
            self.trajectory_packet_count += 1
            return int(resolved)
        if self.trajectory_packet_count == 0:
            self.trajectory_packet_count = 1
            return 0
        self.trajectory_elapsed_us += self.packet_interval_us
        self.trajectory_packet_count += 1
        return self.trajectory_elapsed_us

    def advance_missing_slots(self, count):
        """Preserve temporal holes in every tracker that owns slot state."""
        missing = max(0, int(count))
        for _ in range(missing):
            self.context.add_missing_slot()
            if self.aggregated_context is not None:
                self.aggregated_context.add_missing_slot()
        if self.l1_tracker is not None:
            self.l1_tracker.advance_missing_slots(missing)
        for tracker in (
            self.coherence_tracker,
            self.phase_tracker,
            self.shape_tracker,
            self.amplitude_profile_tracker,
        ):
            advance = getattr(tracker, 'advance_missing_slots', None)
            if callable(advance):
                advance(missing)

    def is_ready(self, minimum_valid_samples=None):
        """Match the production detector readiness contract for host rows."""
        minimum = (
            self.window_packets
            if minimum_valid_samples is None
            else max(1, min(int(minimum_valid_samples), self.window_packets))
        )
        if self.context.buffer_count < self.window_packets:
            return False
        if self.context.valid_count < minimum:
            return False
        if (
            self.aggregated_context is not None
            and (
                self.aggregated_context.buffer_count < self.window_packets
                or self.aggregated_context.valid_count < minimum
            )
        ):
            return False
        if (
            self.l1_tracker is not None
            and self.window_packets > L1_DELTA_LAG
            and self.l1_tracker.count == 0
        ):
            return False
        return True

    def process_packet(self, csi_data, packet=None, timestamp_us=None):
        turbulence, amplitudes = self.context.calculate_spatial_turbulence(
            csi_data,
            DEFAULT_SUBCARRIERS,
            return_amplitudes=True,
        )
        self.context.add_turbulence(turbulence)
        aggregated_turbulence = None
        aggregated_amplitudes = None
        if self.aggregated_context is not None:
            if self.amplitude_profile_tracker is not None:
                (
                    aggregated_turbulence,
                    aggregated_amplitudes,
                ) = self.aggregated_context.calculate_spatial_turbulence(
                    csi_data,
                    DEFAULT_SUBCARRIERS,
                    return_amplitudes=True,
                )
            else:
                aggregated_turbulence = (
                    self.aggregated_context.calculate_spatial_turbulence(
                        csi_data,
                        DEFAULT_SUBCARRIERS,
                    )
                )
            self.aggregated_context.add_turbulence(aggregated_turbulence)
        if self.amplitude_profile_tracker is not None:
            self.amplitude_profile_tracker.process_amplitudes(
                amplitudes,
                aggregated_amplitudes,
            )
        if self.l1_tracker is not None:
            self.l1_tracker.process_amplitudes(amplitudes, len(amplitudes))
        if self.coherence_tracker is not None:
            self.coherence_tracker.process_packet(csi_data)
        if self.phase_tracker is not None:
            self.phase_tracker.process_packet(csi_data)
        if self.shape_tracker is not None:
            self.shape_tracker.process_packet(csi_data)
        if self.shape_trajectory_tracker is not None:
            self.shape_trajectory_tracker.process_packet(
                csi_data,
                self._trajectory_timestamp_us(packet, timestamp_us),
            )
        if self.context.buffer_count < self.context.window_size:
            return None

        turb_list, turb_validity = self._ordered_series(self.context)
        aggregated_turb_list = None
        aggregated_validity = None
        if self.aggregated_context is not None:
            aggregated_turb_list, aggregated_validity = self._ordered_series(
                self.aggregated_context
            )
        l1_count = (
            self.l1_tracker.copy_deltas_into(self.l1_series)
            if self.l1_series is not None else 0
        )
        features = extract_features_by_name(
            turb_list,
            len(turb_list),
            feature_names=self.production_names,
            aggregated_turbulence_buffer=aggregated_turb_list,
            aggregated_turbulence_count=(
                len(aggregated_turb_list)
                if aggregated_turb_list is not None else None
            ),
            l1_delta_lag_ratio=(
                self.l1_tracker.delta_lag_ratio()
                if (
                    self.l1_tracker is not None
                    and 'l1_delta_lag_ratio' in self.production_names
                )
                else None
            ),
            turbulence_validity=turb_validity,
            aggregated_turbulence_validity=aggregated_validity,
            **_production_tracker_feature_kwargs(
                self.production_names,
                self.shape_trajectory_tracker,
            ),
        )
        if not self.candidate_names:
            return features
        return assemble_feature_vector(
            self.feature_names,
            self.production_names,
            features,
            candidate_values(
                self.candidate_names,
                self.coherence_tracker,
                turbulence_series=turb_list,
                aggregated_turbulence_series=aggregated_turb_list,
                phase_tracker=self.phase_tracker,
                shape_tracker=self.shape_tracker,
                shape_trajectory_tracker=self.shape_trajectory_tracker,
                amplitude_profile_tracker=self.amplitude_profile_tracker,
                l1_series=(
                    self.l1_series[:l1_count]
                    if self.l1_series is not None else None
                ),
            ),
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
    """Return CSI data from a packet mapping or a compact matrix row."""
    return packet['csi_data'] if isinstance(packet, Mapping) else packet


def _normalize_feature_row_contract(sample_contract):
    """Normalize one local feature-row sampling contract."""
    contract = str(sample_contract).strip().lower()
    if contract not in {'replay_tick', 'stream_dense'}:
        raise ValueError(f"Unsupported ML sample contract: {sample_contract!r}")
    return contract


def _feature_rows_use_runtime_cache(feature_names):
    """Return True when canonical runtime replay rows can serve this request."""
    return (
        ACTIVE_TRAJECTORY_BIN_US == CHANNEL_SHAPE_BIN_US
        and _feature_names_support_replay_rows(feature_names)
    )


def _empty_feature_rows(feature_names):
    """Return one empty feature-row payload for the requested schema."""
    resolved = [str(name) for name in feature_names]
    return {
        'X': np.empty((0, len(resolved)), dtype=np.float32),
        'feature_names': resolved,
        'packet_index': np.empty(0, dtype=np.int32),
        'evaluation_index': np.empty(0, dtype=np.int32),
        'reset_index': np.empty(0, dtype=np.int32),
        'evaluation_due': np.empty(0, dtype=bool),
    }


def _normalize_host_row_selection(row_stride, row_offset):
    """Validate an optional dense host-row modulo selection."""
    if row_stride is None:
        if int(row_offset) != 0:
            raise ValueError("row_offset requires row_stride")
        return None, 0
    stride = int(row_stride)
    offset = int(row_offset)
    if stride < 1 or offset < 0 or offset >= stride:
        raise ValueError("row selection requires 0 <= row_offset < row_stride")
    return stride, offset


def _select_host_feature_rows(rows, row_stride, row_offset):
    """Select cached dense host rows by position."""
    stride, offset = _normalize_host_row_selection(row_stride, row_offset)
    if stride is None:
        return rows
    row_count = len(np.asarray(rows.get('packet_index', ())))
    mask = np.arange(row_count, dtype=np.int64) % stride == offset
    selected = {
        'X': np.asarray(rows['X'], dtype=np.float32)[mask],
        'feature_names': list(rows['feature_names']),
        'packet_index': np.asarray(rows['packet_index'], dtype=np.int32)[mask],
        'evaluation_index': np.asarray(rows['evaluation_index'], dtype=np.int32)[mask],
        'reset_index': np.asarray(rows['reset_index'], dtype=np.int32)[mask],
        'evaluation_due': np.asarray(rows['evaluation_due'], dtype=bool)[mask],
    }
    if 'cache_hit' in rows:
        selected['cache_hit'] = bool(rows['cache_hit'])
    return selected


def build_host_feature_rows(packets, feature_names, *,
                            sample_contract='replay_tick',
                            row_stride=None, row_offset=0):
    """Build reset-aware feature rows through the host streaming path."""
    requested_feature_names = [str(name) for name in feature_names]
    normalized_contract = _normalize_feature_row_contract(sample_contract)
    stride, offset = _normalize_host_row_selection(row_stride, row_offset)
    if stride is not None and normalized_contract != 'stream_dense':
        raise ValueError("dense row selection requires sample_contract='stream_dense'")
    if not packets:
        return _empty_feature_rows(requested_feature_names)

    interval_us = measure_packet_interval_us(packets)
    target_pps = target_pps_for_packets(packets, interval_us)
    window_packets = temporal_window_slots(
        target_pps,
        SEGMENTATION_WINDOW_SIZE_MS,
    )
    _, cadence = timing_cadence_for_window(window_packets, interval_us)
    extractor = StreamingFeatureExtractor(
        requested_feature_names,
        window_packets,
        packet_interval_us=interval_us,
    )
    minimum_samples = minimum_valid_slots(window_packets)
    packets_since_reset = 0
    reset_index = 0
    evaluation_index = 0
    row_features = []
    packet_index_values = []
    evaluation_index_values = []
    reset_index_values = []
    evaluation_due_values = []

    for admission in iter_temporal_admissions(
        packets,
        target_pps=target_pps,
        window_size_ms=SEGMENTATION_WINDOW_SIZE_MS,
        fallback_interval_us=interval_us,
    ):
        packet_index = admission.packet_index
        packet = admission.packet
        if admission.reset_required:
            extractor = StreamingFeatureExtractor(
                requested_feature_names,
                window_packets,
                packet_interval_us=interval_us,
            )
            cadence.reset()
            reset_index += 1
            packets_since_reset = 0
        elif admission.missing_slots_before:
            extractor.advance_missing_slots(admission.missing_slots_before)
        cadence.note_packet(elapsed_us=admission.coverage_us)
        should_evaluate = cadence.should_evaluate()
        if should_evaluate:
            cadence.after_evaluation()
        values = extractor.process_packet(
            packet_csi_data(packet),
            packet=packet,
            timestamp_us=admission.timestamp_us,
        )
        packets_since_reset = admission.slot_index + 1
        if (
            values is None
            or packets_since_reset < window_packets
            or not extractor.is_ready(minimum_samples)
        ):
            continue
        dense_row_index = evaluation_index
        evaluation_index += 1
        if stride is not None and dense_row_index % stride != offset:
            continue
        row_features.append(np.asarray(values, dtype=np.float32))
        packet_index_values.append(int(packet_index))
        evaluation_index_values.append(int(dense_row_index))
        reset_index_values.append(int(reset_index))
        evaluation_due_values.append(bool(should_evaluate))

    if not row_features:
        return _empty_feature_rows(requested_feature_names)

    dense_rows = {
        'X': np.vstack(row_features).astype(np.float32, copy=False),
        'feature_names': requested_feature_names,
        'packet_index': np.asarray(packet_index_values, dtype=np.int32),
        'evaluation_index': np.asarray(evaluation_index_values, dtype=np.int32),
        'reset_index': np.asarray(reset_index_values, dtype=np.int32),
        'evaluation_due': np.asarray(evaluation_due_values, dtype=bool),
    }
    row_mask = (
        dense_rows['evaluation_due']
        if normalized_contract == 'replay_tick'
        else np.ones(len(dense_rows['packet_index']), dtype=bool)
    )
    projected = {
        'X': dense_rows['X'][row_mask],
        'feature_names': requested_feature_names,
        'packet_index': dense_rows['packet_index'][row_mask],
        'evaluation_index': dense_rows['evaluation_index'][row_mask],
        'reset_index': dense_rows['reset_index'][row_mask],
        'evaluation_due': dense_rows['evaluation_due'][row_mask],
    }
    if normalized_contract == 'replay_tick':
        projected['evaluation_index'] = np.arange(
            len(projected['packet_index']),
            dtype=np.int32,
        )
    return projected


def load_or_compute_host_feature_rows(source_path, *,
                                      packets=None,
                                      packets_factory=None,
                                      feature_names=(),
                                      sample_contract='replay_tick',
                                      use_cache=True,
                                      cache_write=True,
                                      stream_provenance=None,
                                      row_stride=None,
                                      row_offset=0):
    """Load or assemble host rows from independently cached feature columns."""
    if packets is not None and packets_factory is not None:
        raise ValueError("pass packets or packets_factory, not both")
    requested_feature_names = [str(name) for name in feature_names]
    normalized_contract = _normalize_feature_row_contract(sample_contract)
    stride, offset = _normalize_host_row_selection(row_stride, row_offset)
    if stride is not None and normalized_contract != 'stream_dense':
        raise ValueError("dense row selection requires sample_contract='stream_dense'")
    if stride is not None and use_cache and cache_write:
        raise ValueError("selected rows cannot be written under a full-row cache key")
    resolved_provenance = stream_provenance or _host_feature_stream_provenance(
        requested_feature_names
    )
    stream_identity = _host_row_stream_identity(resolved_provenance)
    spine_parameters = {
        'contract': 'host_feature_row_spine_v1',
        'selected_subcarriers': [int(value) for value in DEFAULT_SUBCARRIERS],
        'stream': stream_identity,
    }

    def feature_parameters(name):
        identities = resolved_provenance.get('feature_identities', {})
        identity = identities.get(name) or _host_feature_cache_identity(name)
        return {
            'contract': 'host_feature_column_v1',
            'spine': spine_parameters,
            'feature': identity,
        }

    def load_cached_parts():
        spine = npz_cache.load_host_feature_row_spine_artifact(
            source_path,
            parameters=spine_parameters,
        )
        if spine is None:
            return None, {}
        row_count = len(spine['packet_index'])
        columns = {}
        for name in requested_feature_names:
            column = npz_cache.load_host_feature_column_artifact(
                source_path,
                parameters=feature_parameters(name),
            )
            if column is not None and len(column) == row_count:
                columns[name] = column
        return spine, columns

    def assemble(spine, columns, *, cache_hit):
        row_count = len(spine['packet_index'])
        matrix = (
            np.column_stack([columns[name] for name in requested_feature_names])
            .astype(np.float32, copy=False)
            if requested_feature_names
            else np.empty((row_count, 0), dtype=np.float32)
        )
        rows = {
            'X': matrix,
            'feature_names': list(requested_feature_names),
            **spine,
            'cache_hit': bool(cache_hit),
        }
        if normalized_contract == 'stream_dense':
            return _select_host_feature_rows(rows, stride, offset)
        row_mask = np.asarray(rows['evaluation_due'], dtype=bool)
        return {
            'X': matrix[row_mask],
            'feature_names': list(requested_feature_names),
            'packet_index': np.asarray(rows['packet_index'], dtype=np.int32)[row_mask],
            'evaluation_index': np.arange(int(np.sum(row_mask)), dtype=np.int32),
            'reset_index': np.asarray(rows['reset_index'], dtype=np.int32)[row_mask],
            'evaluation_due': np.asarray(rows['evaluation_due'], dtype=bool)[row_mask],
            'cache_hit': bool(cache_hit),
        }

    spine, columns = load_cached_parts() if use_cache else (None, {})
    missing = [name for name in requested_feature_names if name not in columns]
    if spine is not None and not missing:
        return assemble(spine, columns, cache_hit=True)
    if stride is not None:
        if packets is not None:
            packet_stream = packets
        elif packets_factory is not None:
            packet_stream = packets_factory()
        else:
            packet_stream = load_npz_packet_view(Path(source_path))
        rows = build_host_feature_rows(
            packet_stream,
            requested_feature_names,
            sample_contract='stream_dense',
            row_stride=stride,
            row_offset=offset,
        )
        rows['cache_hit'] = False
        return rows

    build_lock_parameters = {
        'contract': 'host_feature_column_build_v1',
        'spine': spine_parameters,
    }
    lock_context = (
        npz_cache.artifact_build_lock(
            source_path,
            artifact_name='host_feature_columns',
            artifact_version=npz_cache.HOST_FEATURE_COLUMN_ARTIFACT_VERSION,
            parameters=build_lock_parameters,
        )
        if use_cache and cache_write and missing
        else nullcontext()
    )
    with lock_context:
        if use_cache:
            spine, columns = load_cached_parts()
            missing = [name for name in requested_feature_names if name not in columns]
            if spine is not None and not missing:
                return assemble(spine, columns, cache_hit=True)

        if packets is not None:
            packet_stream = packets
        elif packets_factory is not None:
            packet_stream = packets_factory()
        else:
            packet_stream = load_npz_packet_view(Path(source_path))
        computed_names = missing or requested_feature_names
        computed = build_host_feature_rows(
            packet_stream,
            computed_names,
            sample_contract='stream_dense',
            row_stride=stride,
            row_offset=offset,
        )
        computed_spine = {
            key: np.asarray(computed[key])
            for key in (
                'packet_index',
                'evaluation_index',
                'reset_index',
                'evaluation_due',
            )
        }
        if spine is not None:
            for key, values in computed_spine.items():
                if not np.array_equal(values, np.asarray(spine[key])):
                    raise RuntimeError(
                        f"cached host feature row spine diverged for {key}"
                    )
        else:
            spine = computed_spine
            if use_cache and cache_write:
                npz_cache.save_host_feature_row_spine_artifact(
                    source_path,
                    parameters=spine_parameters,
                    rows=spine,
                )
        computed_matrix = np.asarray(computed['X'], dtype=np.float32)
        for index, name in enumerate(computed_names):
            column = computed_matrix[:, index]
            columns[name] = column
            if use_cache and cache_write:
                npz_cache.save_host_feature_column_artifact(
                    source_path,
                    parameters=feature_parameters(name),
                    values=column,
                )
        return assemble(spine, columns, cache_hit=False)


def _feature_names_support_replay_rows(feature_names):
    """Return True when a feature set can use canonical runtime replay rows."""
    return all(str(name) in EXPORTED_FEATURE_NAMES for name in feature_names)


def _evaluate_replay_row_probabilities(center, scale, layers, rows):
    """Predict one already-evaluated replay-row stream."""
    X = np.asarray(rows.get('X', np.empty((0, 0), dtype=np.float32)), dtype=np.float32)
    if X.size == 0:
        return np.zeros(0, dtype=np.float64), 0
    probabilities = _batch_predict_probabilities(X, center, scale, layers)
    return probabilities, int(len(X))


def _evaluate_replay_row_split(center, scale, layers,
                               static_presence_rows, motion_rows, threshold=0.5):
    """Evaluate paired replay metrics from canonical time-aware replay rows."""
    static_probs, static_eval_count = _evaluate_replay_row_probabilities(
        center, scale, layers, static_presence_rows
    )
    motion_probs, motion_eval_count = _evaluate_replay_row_probabilities(
        center, scale, layers, motion_rows
    )
    static_presence_motion_states = static_probs > threshold
    motion_states = motion_probs > threshold
    static_presence_motion_packets = int(np.sum(static_presence_motion_states))
    motion_with_motion = int(np.sum(motion_states))
    motion_without_motion = int(motion_eval_count - motion_with_motion)
    tp = motion_with_motion
    fn = motion_without_motion
    fp = static_presence_motion_packets
    tn = max(static_eval_count - static_presence_motion_packets, 0)
    recall = tp / (tp + fn) * 100.0 if (tp + fn) else 0.0
    precision = tp / (tp + fp) * 100.0 if (tp + fp) else 0.0
    fp_rate = fp / static_eval_count * 100.0 if static_eval_count else 0.0
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
        'static_presence_eval_count': int(static_eval_count),
        'motion_eval_count': int(motion_eval_count),
        **evaluate_runtime_policy_evaluations(static_presence_motion_states),
    }


def _evaluate_replay_row_idle(center, scale, layers, rows, threshold=0.5):
    """Evaluate quiet replay metrics from canonical time-aware replay rows."""
    probabilities, count = _evaluate_replay_row_probabilities(center, scale, layers, rows)
    raw_states = probabilities > threshold
    fp = int(np.sum(raw_states))
    return {
        'fp': fp,
        'evaluations': int(count),
        'fp_rate': fp / count * 100.0 if count else 0.0,
        **evaluate_runtime_policy_evaluations(raw_states),
    }


def evaluate_split(model, scaler, feature_names, static_presence_packets,
                   motion_packets, threshold=0.5):
    """Evaluate a split through reset-aware production-time replay ticks."""
    center, scale = get_preprocessor_arrays(scaler)
    layers = _layer_arrays_from_model(model)
    if _feature_rows_use_runtime_cache(feature_names):
        static_rows = build_ml_replay_rows(
            static_presence_packets,
            DEFAULT_SUBCARRIERS,
            None,
            feature_names,
            sample_contract="replay_tick",
        )
        motion_rows = build_ml_replay_rows(
            motion_packets,
            DEFAULT_SUBCARRIERS,
            None,
            feature_names,
            sample_contract="replay_tick",
        )
    else:
        static_rows = build_host_feature_rows(
            static_presence_packets,
            feature_names,
            sample_contract="replay_tick",
        )
        motion_rows = build_host_feature_rows(
            motion_packets,
            feature_names,
            sample_contract="replay_tick",
        )
    return _evaluate_replay_row_split(
        center,
        scale,
        layers,
        static_rows,
        motion_rows,
        threshold=threshold,
    )


def evaluate_array_split(center, scale, layers, feature_names,
                         static_presence_packets, motion_packets,
                         threshold=0.5):
    """Evaluate exported arrays through reset-aware production-time replay ticks."""
    if _feature_rows_use_runtime_cache(feature_names):
        static_rows = build_ml_replay_rows(
            static_presence_packets,
            DEFAULT_SUBCARRIERS,
            None,
            feature_names,
            sample_contract="replay_tick",
        )
        motion_rows = build_ml_replay_rows(
            motion_packets,
            DEFAULT_SUBCARRIERS,
            None,
            feature_names,
            sample_contract="replay_tick",
        )
    else:
        static_rows = build_host_feature_rows(
            static_presence_packets,
            feature_names,
            sample_contract="replay_tick",
        )
        motion_rows = build_host_feature_rows(
            motion_packets,
            feature_names,
            sample_contract="replay_tick",
        )
    return _evaluate_replay_row_split(
        center,
        scale,
        layers,
        static_rows,
        motion_rows,
        threshold=threshold,
    )


def _load_npz_packets_cached(path):
    """Load NPZ packets through the shared packet-view cache."""
    packets = load_npz_packet_view(Path(path))
    if not packets:
        raise RuntimeError(
            f"{Path(path).name} has no HT20/HT-LTF/64-SC sensing packets after format filtering"
        )
    return packets


def _load_gate_feature_rows(path, label_name, feature_names, *,
                            dataset_info=None, file_metadata=None,
                            use_cache=True):
    """Load one replay file through the canonical time-aware row cache."""
    del label_name, dataset_info, file_metadata
    path = Path(path)
    if _feature_rows_use_runtime_cache(feature_names):
        return load_or_compute_ml_replay_rows(
            path,
            selected_subcarriers=DEFAULT_SUBCARRIERS,
            window_size=None,
            feature_names=feature_names,
            use_cache=use_cache,
            sample_contract="replay_tick",
        )
    rows = load_or_compute_host_feature_rows(
        path,
        feature_names=feature_names,
        sample_contract="replay_tick",
        use_cache=use_cache,
        stream_provenance=_host_feature_stream_provenance(
            feature_names,
        ),
    )
    return rows


def _evaluate_cached_feature_stream(center, scale, layers, rows):
    """Return probabilities for one canonical runtime-tick replay stream."""
    X = np.asarray(rows.get('X', np.empty((0, 0), dtype=np.float32)), dtype=np.float32)
    if X.size == 0:
        return np.zeros(0, dtype=np.float64), 0
    probabilities = _batch_predict_probabilities(X, center, scale, layers)
    return probabilities, int(len(X))


def evaluate_cached_feature_split(center, scale, layers,
                                  static_presence_rows, motion_rows, threshold=0.5):
    """Evaluate a paired split from canonical runtime-tick feature rows."""
    return _evaluate_replay_row_split(
        center,
        scale,
        layers,
        static_presence_rows,
        motion_rows,
        threshold=threshold,
    )


def evaluate_cached_idle_stream(center, scale, layers, rows, threshold=0.5):
    """Evaluate one quiet replay from canonical runtime-tick feature rows."""
    return _evaluate_replay_row_idle(
        center,
        scale,
        layers,
        rows,
        threshold=threshold,
    )


def evaluate_cached_array_split(center, scale, layers, feature_names,
                                static_presence_path, motion_path, threshold=0.5,
                                *, dataset_info=None, file_metadata=None,
                                use_cache=True):
    """Evaluate one paired replay directly from cached per-window features."""
    static_presence_rows = _load_gate_feature_rows(
        static_presence_path,
        'static_presence',
        feature_names,
        dataset_info=dataset_info,
        file_metadata=file_metadata,
        use_cache=use_cache,
    )
    motion_rows = _load_gate_feature_rows(
        motion_path,
        'motion',
        feature_names,
        dataset_info=dataset_info,
        file_metadata=file_metadata,
        use_cache=use_cache,
    )
    return evaluate_cached_feature_split(
        center,
        scale,
        layers,
        static_presence_rows,
        motion_rows,
        threshold=threshold,
    )


def evaluate_cached_idle_array(center, scale, layers, feature_names, path,
                               threshold=0.5, *, dataset_info=None,
                               file_metadata=None, use_cache=True):
    """Evaluate one quiet replay directly from cached per-window features."""
    rows = _load_gate_feature_rows(
        path,
        'empty',
        feature_names,
        dataset_info=dataset_info,
        file_metadata=file_metadata,
        use_cache=use_cache,
    )
    return evaluate_cached_idle_stream(
        center,
        scale,
        layers,
        rows,
        threshold=threshold,
    )


def _iter_paired_chip_replays(chips=None, roles=('selection',),
                              allow_legacy_fallback=True):
    """Yield role-isolated real pair replay paths, or one legacy train fallback."""
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
            motion_name = str(static_entry.get('optimal_pair_motion_file', ''))
            motion_entry = motion_by_name.get(motion_name)
            if motion_entry is None or bool(motion_entry.get('synthetic')):
                continue
            role = paired_dataset_role(
                static_entry,
                motion_entry,
                admitted_roles=roles,
            )
            if role is None:
                continue
            static_path = resolve_entry_path('static_presence', static_entry)
            motion_path = resolve_entry_path('motion', motion_entry)
            if static_path.exists() and motion_path.exists():
                low_rssi = bool(static_entry.get('low_rssi')) or bool(motion_entry.get('low_rssi'))
                role_pairs.append((role, static_path, motion_path, low_rssi))
        if role_pairs:
            for role, static_path, motion_path, low_rssi in sorted(role_pairs):
                key = f"{chip}:{role}:{static_path.name}"
                yield (key, static_path, motion_path, low_rssi)
            continue
        if not allow_legacy_fallback:
            continue

        # When no reserved pair exists, use only an explicitly admitted real
        # train pair. Missing roles default to exclude, and a newly generated
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
            motion_name = str(static_entry.get('optimal_pair_motion_file', ''))
            motion_entry = motion_by_name.get(motion_name)
            if motion_entry is None or bool(motion_entry.get('synthetic')):
                continue
            if paired_dataset_role(
                static_entry,
                motion_entry,
                admitted_roles=('train',),
            ) != 'train':
                continue
            static_path = resolve_entry_path('static_presence', static_entry)
            motion_path = resolve_entry_path('motion', motion_entry)
            if static_path.exists() and motion_path.exists():
                sort_key = (
                    str(static_entry.get('collected_at', '')),
                    static_path.name,
                )
                low_rssi = bool(static_entry.get('low_rssi')) or bool(motion_entry.get('low_rssi'))
                legacy_pairs.append((sort_key, static_path, motion_path, low_rssi))
        if not legacy_pairs:
            continue
        _, static_path, motion_path, low_rssi = max(legacy_pairs, key=lambda row: row[0])
        yield (chip, static_path, motion_path, low_rssi)


def _iter_paired_chip_packets(chips=None, roles=('selection',),
                              allow_legacy_fallback=True):
    """Yield role-isolated real pairs, or the legacy latest pair as fallback."""
    for chip, static_path, motion_path, low_rssi in _iter_paired_chip_replays(
        chips,
        roles=roles,
        allow_legacy_fallback=allow_legacy_fallback,
    ):
        yield (
            chip,
            _load_npz_packets_cached(static_path),
            _load_npz_packets_cached(motion_path),
            low_rssi,
        )


def evaluate_paired_gate(model, scaler, feature_names, threshold=0.5, chips=None,
                         roles=('selection',), allow_legacy_fallback=True,
                         progress=None, use_cached_features=True,
                         use_cache=True):
    """Evaluate a candidate on the paired validation datasets."""
    pairs = list(_iter_paired_chip_replays(
        chips,
        roles=roles,
        allow_legacy_fallback=allow_legacy_fallback,
    ))
    center = scale = layers = dataset_info = file_metadata = None
    if use_cached_features:
        center, scale = get_preprocessor_arrays(scaler)
        layers = _layer_arrays_from_model(model)
        dataset_info = load_dataset_info()
        file_metadata = get_file_metadata(dataset_info)
    if progress is not None:
        progress(
            f"Paired gate: evaluating {len(pairs)} chip replay(s)"
        )
    by_chip = {}
    gate_start = perf_counter()
    for index, (chip, static_path, motion_path, low_rssi) in enumerate(
        pairs,
        start=1,
    ):
        step_start = perf_counter()
        if use_cached_features:
            row = evaluate_cached_array_split(
                center,
                scale,
                layers,
                feature_names,
                static_path,
                motion_path,
                threshold=threshold,
                dataset_info=dataset_info,
                file_metadata=file_metadata,
                use_cache=use_cache,
            )
        else:
            row = evaluate_split(
                model,
                scaler,
                feature_names,
                _load_npz_packets_cached(static_path),
                _load_npz_packets_cached(motion_path),
                threshold=threshold,
            )
        row['low_rssi'] = low_rssi
        by_chip[chip] = row
        if progress is not None:
            progress(
                f"Paired gate {index}/{len(pairs)} {chip}: "
                f"R={row['recall']:.2f}% FP={row['fp_rate']:.2f}% "
                f"alarms={row.get('effective_alarms', 0)} "
                f"in {format_duration(perf_counter() - step_start)}"
            )
    summary = summarize_gate(by_chip)
    if progress is not None and summary is not None:
        progress(
            f"Paired gate complete in {format_duration(perf_counter() - gate_start)}: "
            f"pass={summary['pass_count']} maxFP={summary['max_fp_rate']:.2f}% "
            f"worstRecall={summary['worst_chip_recall']:.2f}% "
            f"alarms={summary.get('total_effective_alarms', 0)}"
        )
    return summary


def _load_exported_model_arrays():
    """Load exported MicroPython weights as inference-ready arrays."""
    module = load_exported_ml_weights()
    center = np.asarray(module.FEATURE_MEAN, dtype=np.float32)
    scale = np.asarray(module.FEATURE_SCALE, dtype=np.float32)
    matrices = exported_weight_matrices(module)
    layers = []
    for idx, (weights, biases) in enumerate(zip(matrices, module.BIASES)):
        layers.append((
            weights,
            np.asarray(biases, dtype=np.float32),
            idx == len(matrices) - 1,
        ))
    return list(module.FEATURE_NAMES), center, scale, layers


def _evaluate_exported_paired_gate_at_active_bin(
        threshold=0.5, chips=None, roles=('selection', 'holdout'),
        allow_legacy_fallback=True, use_cached_features=True, use_cache=True):
    """Evaluate exported runtime arrays on the paired validation datasets."""
    feature_names, center, scale, layers = _load_exported_model_arrays()
    by_chip = {}
    dataset_info = file_metadata = None
    if use_cached_features:
        dataset_info = load_dataset_info()
        file_metadata = get_file_metadata(dataset_info)
    for chip, static_path, motion_path, low_rssi in _iter_paired_chip_replays(
        chips,
        roles=roles,
        allow_legacy_fallback=allow_legacy_fallback,
    ):
        if use_cached_features:
            row = evaluate_cached_array_split(
                center,
                scale,
                layers,
                feature_names,
                static_path,
                motion_path,
                threshold=threshold,
                dataset_info=dataset_info,
                file_metadata=file_metadata,
                use_cache=use_cache,
            )
        else:
            row = evaluate_array_split(
                center,
                scale,
                layers,
                feature_names,
                _load_npz_packets_cached(static_path),
                _load_npz_packets_cached(motion_path),
                threshold=threshold,
            )
        row['low_rssi'] = low_rssi
        by_chip[chip] = row
    return summarize_gate(by_chip)


def evaluate_exported_paired_gate(threshold=0.5, chips=None,
                                  roles=('selection', 'holdout'),
                                  allow_legacy_fallback=True,
                                  use_cached_features=True,
                                  use_cache=True):
    """Evaluate exported arrays with their canonical production trajectory bin."""
    with canonical_trajectory_bin():
        return _evaluate_exported_paired_gate_at_active_bin(
            threshold=threshold,
            chips=chips,
            roles=roles,
            allow_legacy_fallback=allow_legacy_fallback,
            use_cached_features=use_cached_features,
            use_cache=use_cache,
        )


def _iter_quiet_gate_replays(roles=('selection', 'holdout')):
    """Yield real empty replay paths explicitly reserved for selection/holdout."""
    dataset_info = load_dataset_info()
    roles = normalize_dataset_roles(roles, default=('selection', 'holdout'))
    for entry in dataset_info.get('files', {}).get('empty', []):
        role = admitted_dataset_role(entry, admitted_roles=roles)
        if role is None or bool(entry.get('synthetic')):
            continue
        path = resolve_entry_path('empty', entry)
        if path.exists():
            chip = str(entry.get('chip', 'unknown')).upper()
            yield f"{chip}:{role}:{path.name}", path


def _iter_quiet_gate_packets(roles=('selection', 'holdout')):
    """Yield real empty recordings explicitly reserved for selection/holdout."""
    for key, path in _iter_quiet_gate_replays(roles=roles):
        yield key, _load_npz_packets_cached(path)


def evaluate_idle_streaming(evaluator, packets, threshold=0.5):
    """Evaluate one quiet stream at production cadence and hit filtering."""
    if not isinstance(evaluator, (ArrayStreamingEvaluator, StreamingEvaluator)):
        raise TypeError("Time-aware idle evaluation requires a streaming ML evaluator")
    feature_names = evaluator.extractor.feature_names
    if _feature_rows_use_runtime_cache(feature_names):
        rows = build_ml_replay_rows(
            packets,
            DEFAULT_SUBCARRIERS,
            None,
            feature_names,
            sample_contract="replay_tick",
        )
    else:
        rows = build_host_feature_rows(
            packets,
            feature_names,
            sample_contract="replay_tick",
        )
    return _evaluate_replay_row_idle(
        evaluator.center,
        evaluator.scale,
        evaluator.layers,
        rows,
        threshold=threshold,
    )


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
            row['fp_rate'] < DEFAULT_GATE_TARGET_FP_RATE
            and row['effective_alarms'] == 0
            for row in rows
        ),
    }


def evaluate_quiet_gate(model, scaler, feature_names, threshold=0.5,
                        roles=('selection', 'holdout'), progress=None,
                        use_cached_features=True, use_cache=True):
    """Evaluate an in-memory candidate on reserved real empty recordings."""
    datasets = list(_iter_quiet_gate_replays(roles=roles))
    center = scale = layers = dataset_info = file_metadata = None
    if use_cached_features:
        center, scale = get_preprocessor_arrays(scaler)
        layers = _layer_arrays_from_model(model)
        dataset_info = load_dataset_info()
        file_metadata = get_file_metadata(dataset_info)
    if progress is not None:
        progress(
            f"Quiet gate: evaluating {len(datasets)} reserved empty replay(s)"
        )
    by_dataset = {}
    gate_start = perf_counter()
    for index, (key, path) in enumerate(datasets, start=1):
        step_start = perf_counter()
        if use_cached_features:
            by_dataset[key] = evaluate_cached_idle_array(
                center,
                scale,
                layers,
                feature_names,
                path,
                threshold=threshold,
                dataset_info=dataset_info,
                file_metadata=file_metadata,
                use_cache=use_cache,
            )
        else:
            by_dataset[key] = evaluate_idle_streaming(
                StreamingEvaluator(model, scaler, feature_names),
                _load_npz_packets_cached(path),
                threshold=threshold,
            )
        row = by_dataset[key]
        if progress is not None:
            progress(
                f"Quiet gate {index}/{len(datasets)} {key}: "
                f"FP={row['fp_rate']:.2f}% alarms={row['effective_alarms']} "
                f"in {format_duration(perf_counter() - step_start)}"
            )
    summary = summarize_quiet_gate(by_dataset)
    if progress is not None and summary is not None:
        progress(
            f"Quiet gate complete in {format_duration(perf_counter() - gate_start)}: "
            f"{'pass' if summary['passed'] else 'fail'} "
            f"maxFP={summary['max_fp_rate']:.2f}% "
            f"alarms={summary['total_effective_alarms']}"
        )
    return summary


def _evaluate_exported_quiet_gate_at_active_bin(
        threshold=0.5, roles=('selection', 'holdout'),
        use_cached_features=True, use_cache=True):
    """Evaluate exported arrays on reserved real empty recordings."""
    feature_names, center, scale, layers = _load_exported_model_arrays()
    by_dataset = {}
    dataset_info = file_metadata = None
    if use_cached_features:
        dataset_info = load_dataset_info()
        file_metadata = get_file_metadata(dataset_info)
    for key, path in _iter_quiet_gate_replays(roles=roles):
        if use_cached_features:
            by_dataset[key] = evaluate_cached_idle_array(
                center,
                scale,
                layers,
                feature_names,
                path,
                threshold=threshold,
                dataset_info=dataset_info,
                file_metadata=file_metadata,
                use_cache=use_cache,
            )
        else:
            by_dataset[key] = evaluate_idle_streaming(
                ArrayStreamingEvaluator(center, scale, layers, feature_names),
                _load_npz_packets_cached(path),
                threshold=threshold,
            )
    return summarize_quiet_gate(by_dataset)


def evaluate_exported_quiet_gate(threshold=0.5, roles=('selection', 'holdout'),
                                 use_cached_features=True, use_cache=True):
    """Evaluate exported arrays with their canonical production trajectory bin."""
    with canonical_trajectory_bin():
        return _evaluate_exported_quiet_gate_at_active_bin(
            threshold=threshold,
            roles=roles,
            use_cached_features=use_cached_features,
            use_cache=use_cache,
        )


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


def in_memory_gate_result(training_metrics):
    """Adapt one in-memory training result to the shared gate summary type."""
    paired_metrics = training_metrics.get('paired') if training_metrics else None
    quiet_metrics = training_metrics.get('quiet') if training_metrics else None
    paired_total = len(paired_metrics.get('by_chip', {})) if paired_metrics else 0
    paired_rc = (
        0
        if paired_total and paired_metrics.get('pass_count', 0) == paired_total
        else 1
    )
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
        paired_metrics.get('worst_chip_recall', -float('inf')),
        paired_metrics.get('worst_chip_f1', -float('inf')),
        -paired_metrics.get('max_fp_rate', float('inf')),
        -paired_metrics.get('total_effective_alarms', float('inf')),
        paired_metrics.get('mean_f1', -float('inf')),
        paired_metrics.get('mean_recall', -float('inf')),
    )


def _combined_candidate_key(cv_metrics, paired_metrics=None):
    """
    Final selection key.

    After deployment safety passes, paired replay recall leads ranking and
    grouped OOF robustness breaks ties between similarly safe candidates.
    """
    cv_key = build_candidate_key(cv_metrics)
    paired_key = _paired_gate_key(paired_metrics)
    if paired_key is None:
        return cv_key
    return paired_key + cv_key


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
            saved_files.append((path, backup_dir / rel_name, True))
        else:
            saved_files.append((path, None, False))
    return backup_dir, saved_files


def _restore_artifacts(saved_files):
    """Restore model artifacts from backup copies."""
    for original, backup, existed in saved_files:
        if existed and backup is not None and backup.exists():
            original.parent.mkdir(parents=True, exist_ok=True)
            temporary = original.parent / f".{original.name}.restore.{os.getpid()}"
            shutil.copy2(backup, temporary)
            os.replace(temporary, original)
        elif not existed:
            original.unlink(missing_ok=True)


def train_until_improvement(max_trials, fp_weight=DEFAULT_FP_WEIGHT, feature_names=None,
                            hidden_layers=None, scaler_mode=DEFAULT_SCALER_MODE,
                            batch_size=DEFAULT_BATCH_SIZE, environment_filter=None,
                            excluded_chips=None, positive_chip_boost=None,
                            use_cache=True, augment=False,
                            timing_quality_policy=DEFAULT_TIMING_QUALITY_POLICY,
                            timing_warn_weight=DEFAULT_TIMING_WARN_WEIGHT,
                            search_output_path=DEFAULT_SEED_SEARCH_OUTPUT,
                            export_artifacts=True):
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
    host_only_search = any(name not in CPP_FEATURE_IDS for name in feature_names)
    in_memory_search = host_only_search or not export_artifacts
    excluded_chips = parse_chip_filter(excluded_chips)
    positive_chip_boost = parse_positive_chip_boost(positive_chip_boost)
    augment_components, feature_augmentation, packet_augmentation = resolve_training_augmentation(augment)
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
    if in_memory_search:
        reason = "host-side candidate features" if host_only_search else "--no-export"
        print(f"Artifact mode: in-memory only ({reason})")
    print(
        "Augmentation: "
        f"{format_augmentation_config(feature_augmentation, packet_augmentation, components=augment_components)}"
    )
    if packet_augmentation:
        print(
            "Packet augmentation seeds: "
            + ", ".join(str(seed) for seed in FIXED_PACKET_AUGMENTATION_SEEDS)
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
        'augment': augment_components,
        'timing_quality_policy': timing_quality_policy,
        'timing_warn_weight': timing_warn_weight,
        # Candidate selection may reuse selection recordings, but the holdout
        # stays sealed until exactly one winner has been chosen.
        'deployment_roles': ('selection',),
        'allow_legacy_gate_fallback': True,
    }

    baseline_train_kwargs = dict(train_kwargs)
    baseline_train_kwargs['feature_names'] = list(DEFAULT_FEATURES)
    print(f"\nEvaluating current model baseline with seed {static_presence_seed}...")
    static_presence_rc, _, static_presence_metrics = train_all(
        seed=static_presence_seed,
        export_artifacts=False,
        **baseline_train_kwargs,
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

    search_results = {
        'config': {
            'max_trials': max_trials,
            'fp_weight': fp_weight,
            'feature_names': list(feature_names),
            'hidden_layers': list(hidden_layers),
            'scaler_mode': scaler_mode,
            'batch_size': batch_size,
            'augment': bool(parse_augmentation_components(augment)),
            'augmentation': format_augmentation_config(
                feature_augmentation,
                packet_augmentation,
                components=augment_components,
            ),
            'timing_quality_policy': timing_quality_policy,
            'timing_warn_weight': float(timing_warn_weight),
            'training_sample_contract': TRAINING_SAMPLE_CONTRACT,
            'export_artifacts': not in_memory_search,
            'environment_filter': environment_filter,
            'excluded_chips': sorted(excluded_chips) if excluded_chips else None,
            'started_at': datetime.now().isoformat(timespec='seconds'),
        },
        'baseline': {
            'seed': static_presence_seed,
            'feature_names': list(DEFAULT_FEATURES),
            'oof_f1': static_presence_metrics.get('oof_f1'),
            'session_min_recall': static_presence_session.get('recall'),
            'selection_paired_metrics': static_presence_gate.paired_metrics,
            'selection_quiet_metrics': static_presence_gate.quiet_metrics,
            'holdout_paired_metrics': baseline_holdout_gate.paired_metrics,
        },
        'trials': [],
        'final_holdout': None,
    }

    def _write_search_results():
        """Persist after every trial: a search that crashes at trial 9 of 10
        still leaves the per-replay rows behind."""
        if search_output_path is None:
            return
        write_json_results(Path(search_output_path), search_results)

    def _record_trial(trial_seed_value, status, cv_metrics, gate):
        """Record one evaluated trial in both the summary and the results file.

        Every branch that evaluates a candidate goes through here, including the
        broken-baseline ranking path: a search that promotes nothing still has to
        leave its per-replay rows behind, and that is the run whose rows matter
        most.
        """
        trial_summaries.append((trial_seed_value, cv_metrics, gate, status))
        session_summary = cv_metrics.get('group_reports', {}).get('session_group', {}).get('worst_recall', {})
        fp_summary = cv_metrics.get('group_reports', {}).get('session_group', {}).get('worst_fp_rate', {})
        search_results['trials'].append({
            'seed': trial_seed_value,
            'status': status,
            'oof_f1': cv_metrics.get('oof_f1'),
            'session_min_recall': session_summary.get('recall'),
            'session_max_fp_rate': fp_summary.get('fp_rate'),
            'cv': cv_metrics.get('cv'),
            'paired_passed': bool(gate.passed) if gate else None,
            'paired_metrics': gate.paired_metrics if gate else None,
            'quiet_metrics': gate.quiet_metrics if gate else None,
            'non_regression_failures': (
                paired_non_regression_failures(
                    gate.paired_metrics,
                    static_presence_gate.paired_metrics)
                if gate is not None
                and gate.paired_metrics is not None
                and static_presence_gate.paired_metrics is not None
                else []
            ),
        })
        _write_search_results()

    _write_search_results()

    if in_memory_search:
        backup_dir, saved_files = None, []
        print("Artifacts: unchanged throughout in-memory seed search")
    else:
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
        # One train_all per trial: CV once, then final fit for the paired gate.
        # Pass an explicit random seed so resolve_training_seed does not reuse the
        # currently exported model seed on every trial.
        export_rc, used_seed, final_metrics = train_all(
            seed=trial_seed,
            export_artifacts=not in_memory_search,
            evaluate_deployment=in_memory_search,
            **train_kwargs,
        )
        if export_rc != 0 or final_metrics is None:
            print(f"  Candidate training failed (exit={export_rc})")
            _restore_artifacts(saved_files)
            failure_status = 'training_failed' if in_memory_search else 'export_failed'
            trial_summaries.append((used_seed, final_metrics or {}, None, failure_status))
            search_results['trials'].append({
                'seed': used_seed,
                'status': failure_status,
                'returncode': export_rc,
            })
            _write_search_results()
            continue

        session_summary = final_metrics.get('group_reports', {}).get('session_group', {}).get('worst_recall', {})
        fp_summary = final_metrics.get('group_reports', {}).get('session_group', {}).get('worst_fp_rate', {})
        print(
            f"  Result: session_min_recall={session_summary.get('recall', 0.0):.1f}% "
            f"session_max_fp={fp_summary.get('fp_rate', 0.0):.1f}% "
            f"blocked_oof_f1={final_metrics['oof_f1']:.1f}%"
        )

        candidate_gate = (
            in_memory_gate_result(final_metrics)
            if in_memory_search
            else run_exported_ml_gates(roles=('selection',))
        )
        gate_kind = "In-memory" if in_memory_search else "Exported"
        print(f"  {gate_kind} ML gates: {_format_exported_gate_summary(candidate_gate)}")
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
                if not in_memory_search:
                    best_candidate_backup_dir, best_candidate_saved_files = _backup_artifacts()
                status = 'ranked_best'
                print("  Broken baseline mode: current best candidate updated")
            elif not candidate_gate.passed:
                print("  Broken baseline mode: candidate still fails deployment safety")
            else:
                print("  Broken baseline mode: candidate did not beat current best")
            _record_trial(used_seed, status, final_metrics, candidate_gate)
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
                if not in_memory_search:
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
            print(format_non_regression_failures(paired_non_regression_failures(
                candidate_gate.paired_metrics, static_presence_gate.paired_metrics)))
        elif comparison['regressions']:
            print("  Robust CV rejected candidate due to material regression")
        else:
            print("  Robust CV found no material improvement")
        _record_trial(used_seed, status, final_metrics, candidate_gate)
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

    if improved and in_memory_search:
        holdout_kwargs = dict(train_kwargs)
        holdout_kwargs['deployment_roles'] = ('holdout',)
        holdout_kwargs['allow_legacy_gate_fallback'] = False
        holdout_rc, _, holdout_metrics = train_all(
            seed=improved_seed,
            export_artifacts=False,
            evaluate_deployment=True,
            **holdout_kwargs,
        )
        if holdout_rc != 0 or holdout_metrics is None:
            print("Final reserved holdout evaluation failed")
            return 1
        final_holdout_gate = in_memory_gate_result(holdout_metrics)
        if final_holdout_gate.available:
            print(
                "Final reserved holdout: "
                f"{_format_exported_gate_summary(final_holdout_gate)}"
            )
            holdout_failures = (
                []
                if (baseline_holdout_gate.paired_metrics is None
                    or final_holdout_gate.paired_metrics is None)
                else paired_non_regression_failures(
                    final_holdout_gate.paired_metrics,
                    baseline_holdout_gate.paired_metrics,
                )
            )
            search_results['final_holdout'] = {
                'seed': improved_seed,
                'passed': bool(final_holdout_gate.passed),
                'paired_metrics': final_holdout_gate.paired_metrics,
                'quiet_metrics': final_holdout_gate.quiet_metrics,
                'non_regression_failures': holdout_failures,
            }
            _write_search_results()
            if not final_holdout_gate.passed or holdout_failures:
                if holdout_failures:
                    print("Blocked by per-recording non-regression on:")
                    print(format_non_regression_failures(holdout_failures))
                print("Final reserved holdout rejected the selected candidate")
                return 1
        search_results['selected_seed'] = improved_seed
        _write_search_results()
        print(
            f"\nSelected research seed after full robust ranking: {improved_seed} "
            f"(blocked_oof_f1={improved_metrics['oof_f1']:.1f}%, "
            f"{_format_exported_gate_summary(improved_gate)})"
        )
        print("Runtime artifacts unchanged; rerun the selected seed explicitly to export it")
        if search_output_path is not None:
            print(f"Seed search results: {search_output_path}")
        return 0

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
                holdout_failures = (
                    []
                    if (baseline_holdout_gate.paired_metrics is None
                        or final_holdout_gate.paired_metrics is None)
                    else paired_non_regression_failures(
                        final_holdout_gate.paired_metrics,
                        baseline_holdout_gate.paired_metrics,
                    )
                )
                search_results['final_holdout'] = {
                    'seed': improved_seed,
                    'passed': bool(final_holdout_gate.passed),
                    'paired_metrics': final_holdout_gate.paired_metrics,
                    'quiet_metrics': final_holdout_gate.quiet_metrics,
                    'non_regression_failures': holdout_failures,
                }
                _write_search_results()
                if not final_holdout_gate.passed or holdout_failures:
                    _restore_artifacts(saved_files)
                    if holdout_failures:
                        print("Blocked by per-recording non-regression on:")
                        print(format_non_regression_failures(holdout_failures))
                    print(
                        "Final reserved holdout rejected the selected candidate; "
                        "current artifacts were restored"
                    )
                    return 1
            search_results['selected_seed'] = improved_seed
            _write_search_results()
            print(
                f"\nSelected seed after full robust ranking: {improved_seed} "
                f"(blocked_oof_f1={improved_metrics['oof_f1']:.1f}%, "
                f"{_format_exported_gate_summary(improved_gate)})"
            )
            if search_output_path is not None:
                print(f"Seed search results: {search_output_path}")
            return 0

    search_results['selected_seed'] = None
    _write_search_results()
    if search_output_path is not None:
        print(f"\nSeed search results: {search_output_path}")
    if broken_baseline_mode:
        print("No candidate beat the current broken baseline; current artifacts remain unchanged")
        return 1

    print("No improvement found within max trials; current artifacts remain unchanged")
    return 1


def write_json_results(path, payload):
    """Write a JSON experiment payload."""
    atomic_write_text(
        path,
        json.dumps(payload, indent=2, default=_json_value) + '\n',
    )


def _json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


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


def evaluate_architecture_candidate(
    name,
    hidden_layers,
    seed,
    dataset,
    scaler_mode,
    batch_size,
    fp_weight,
    feature_augmentation=None,
):
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
            block_stride=DEFAULT_WINDOW_PACKETS_AT_NOMINAL_RATE,
            block_group_key=DEFAULT_BLOCK_GROUP_KEY,
            report_group_keys=DEFAULT_REPORT_GROUP_KEYS,
            seed=seed,
            shap_feature_names=dataset['feature_names'],
            feature_augmentation=feature_augmentation,
            X_aug=dataset.get('X_aug'),
            y_aug=dataset.get('y_aug'),
            groups_aug=dataset.get('groups_aug'),
        )

    scaler = build_preprocessor(scaler_mode)
    fit_preprocessor(
        scaler,
        dataset['X'],
        y=dataset['y'],
        sample_context=dataset['sample_context'],
    )
    X_scaled = scaler.transform(dataset['X'])
    y_train = dataset['y']
    sample_weight = dataset['sample_weights']
    X_scaled, y_train, sample_weight = _append_augmented_training_rows(
        X_scaled,
        y_train,
        scaler,
        dataset.get('X_aug'),
        dataset.get('y_aug'),
        dataset.get('groups_aug'),
        dataset['groups'],
        sample_weight=sample_weight,
    )
    feature_bounds = (
        normalized_feature_bounds(scaler, dataset['feature_names'])
        if feature_augmentation else None
    )
    with suppress_stderr():
        model = train_model(
            X_scaled,
            y_train,
            hidden_layers=list(hidden_layers),
            max_epochs=DEFAULT_MAX_EPOCHS,
            fp_weight=fp_weight,
            sample_weight=sample_weight,
            batch_size=batch_size,
            seed=derive_seed(seed, 10_000),
            feature_augmentation=feature_augmentation,
            feature_bounds=feature_bounds,
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
    """Return a dataset view with one or more ``+``-joined features removed."""
    feature_names = list(dataset['feature_names'])
    removed_features = [
        name.strip() for name in str(feature_name).split('+') if name.strip()
    ]
    unknown = [name for name in removed_features if name not in feature_names]
    if not removed_features or unknown:
        raise ValueError(
            f"Unknown ablation feature(s) '{', '.join(unknown or removed_features)}'. "
            f"Available features: {', '.join(feature_names)}"
        )
    removed_indices = [feature_names.index(name) for name in removed_features]
    candidate = dict(dataset)
    candidate['X'] = np.delete(dataset['X'], removed_indices, axis=1)
    if dataset.get('X_aug') is not None:
        candidate['X_aug'] = np.delete(
            dataset['X_aug'], removed_indices, axis=1
        )
    candidate['feature_names'] = [
        name for idx, name in enumerate(feature_names)
        if idx not in removed_indices
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
    print(f"{'Metric':<29} {'Baseline':>14} {'Candidate':>14} {'Delta':>14}")
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


def _non_regression_failure(replay, metric, candidate_value, baseline_value,
                            margin=0.0, eval_count=None):
    """Describe one blocking comparison, in the units the gate reasons about."""
    entry = {
        'replay': replay,
        'metric': metric,
        'candidate': candidate_value,
        'baseline': baseline_value,
        'margin': margin,
    }
    if eval_count:
        # Percentages hide how small these differences are; the gate margin is
        # one evaluation, so report the evaluation count that produced them.
        entry['eval_count'] = int(eval_count)
        entry['candidate_evaluations'] = round(candidate_value * eval_count / 100.0)
        entry['baseline_evaluations'] = round(baseline_value * eval_count / 100.0)
    return entry


# Per-recording non-regression margins, in evaluations, set from measured
# seed-to-seed dispersion: a margin below the noise floor rejects candidates
# over weight initialisation rather than over behaviour a user would notice.
#
# Measured with tools/analyze_seed_dispersion.py across fifteen seeds of one
# feature set on one corpus: false-positive evaluations on the hardest
# normal-link replay ranged over four, while recall did not move at all and
# effective alarms never moved on any replay. FP carries one evaluation of
# headroom because a maximum over fifteen samples understates the range;
# recall keeps the original single evaluation, since nothing measured asks for
# more. Effective alarms stay at zero margin deliberately: that ratchet has
# already caught a candidate whose aggregates looked strictly better.
FP_SEED_NOISE_EVALUATIONS = 5
RECALL_SEED_NOISE_EVALUATIONS = 1


def paired_non_regression_failures(candidate, baseline, tolerance=0.25):
    """List every comparison blocking a candidate; empty when nothing does.

    Reported rather than merely counted, because a rejection that does not say
    which replay moved cannot be argued with.
    """
    if candidate['pass_count'] != baseline['pass_count']:
        if candidate['pass_count'] > baseline['pass_count']:
            return []
        return [_non_regression_failure(
            '<paired>', 'pass_count', candidate['pass_count'], baseline['pass_count'])]
    candidate_rows = candidate.get('by_chip') or {}
    baseline_rows = baseline.get('by_chip') or {}
    shared_keys = sorted(set(candidate_rows).intersection(baseline_rows))
    failures = []
    if shared_keys and len(shared_keys) == len(candidate_rows) == len(baseline_rows):
        for key in shared_keys:
            candidate_row = candidate_rows[key]
            baseline_row = baseline_rows[key]
            if candidate_row.get('effective_alarms', 0) > baseline_row.get('effective_alarms', 0):
                failures.append(_non_regression_failure(
                    key, 'effective_alarms',
                    candidate_row.get('effective_alarms', 0),
                    baseline_row.get('effective_alarms', 0)))
                continue
            if candidate_row.get('low_rssi') or baseline_row.get('low_rssi'):
                # Weak-link replays are stress diagnostics: at -75/-77 dBm
                # recall and FP jitter by whole events between equally healthy
                # models, so within the absolute stress targets only the alarm
                # count ratchets against the baseline.
                continue
            fp_evals = max(
                int(candidate_row.get('static_presence_eval_count', 0)),
                int(baseline_row.get('static_presence_eval_count', 0)),
            )
            recall_evals = max(
                int(candidate_row.get('motion_eval_count', 0)),
                int(baseline_row.get('motion_eval_count', 0)),
            )
            fp_margin = FP_SEED_NOISE_EVALUATIONS * 100.0 / max(fp_evals, 1)
            recall_margin = RECALL_SEED_NOISE_EVALUATIONS * 100.0 / max(recall_evals, 1)
            if candidate_row.get('fp_rate', 100.0) > baseline_row.get('fp_rate', 100.0) + fp_margin + 1e-9:
                failures.append(_non_regression_failure(
                    key, 'fp_rate', candidate_row.get('fp_rate', 100.0),
                    baseline_row.get('fp_rate', 100.0), fp_margin, fp_evals))
            if candidate_row.get('recall', 0.0) < baseline_row.get('recall', 0.0) - recall_margin - 1e-9:
                failures.append(_non_regression_failure(
                    key, 'recall', candidate_row.get('recall', 0.0),
                    baseline_row.get('recall', 0.0), recall_margin, recall_evals))
        return failures
    aggregate_checks = (
        ('max_fp_rate', candidate['max_fp_rate'], baseline['max_fp_rate'] + tolerance,
         candidate['max_fp_rate'] <= baseline['max_fp_rate'] + tolerance),
        ('worst_chip_recall', candidate['worst_chip_recall'], baseline['worst_chip_recall'] - tolerance,
         candidate['worst_chip_recall'] >= baseline['worst_chip_recall'] - tolerance),
        ('worst_chip_f1', candidate['worst_chip_f1'], baseline['worst_chip_f1'] - tolerance,
         candidate['worst_chip_f1'] >= baseline['worst_chip_f1'] - tolerance),
        ('total_effective_alarms', candidate.get('total_effective_alarms', 0),
         baseline.get('total_effective_alarms', 0),
         candidate.get('total_effective_alarms', 0) <= baseline.get('total_effective_alarms', 0)),
    )
    for metric, candidate_value, limit, ok in aggregate_checks:
        if not ok:
            failures.append(_non_regression_failure(
                '<aggregate>', metric, candidate_value, limit, tolerance))
    return failures


def paired_result_non_regression(candidate, baseline, tolerance=0.25):
    """Preserve each paired replay within its measured seed-noise margin."""
    return not paired_non_regression_failures(candidate, baseline, tolerance)


def format_non_regression_failures(failures, indent='    '):
    """Render blocking comparisons as one readable line each."""
    lines = []
    for item in failures:
        detail = (
            f"{item['candidate']:.4g} vs {item['baseline']:.4g} "
            f"(margin {item['margin']:.4g})"
        )
        if 'candidate_evaluations' in item:
            detail += (
                f" = {item['candidate_evaluations']} vs "
                f"{item['baseline_evaluations']} of {item['eval_count']} evaluations"
            )
        lines.append(f"{indent}{item['replay']} | {item['metric']}: {detail}")
    return '\n'.join(lines)


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
                                use_cache=True,
                                augment=False,
                                timing_quality_policy=DEFAULT_TIMING_QUALITY_POLICY,
                                timing_warn_weight=DEFAULT_TIMING_WARN_WEIGHT):
    """Compare the production baseline against feature removals without exporting."""
    environment_filter = parse_environment_filter(environment_filter)
    excluded_chips = parse_chip_filter(excluded_chips)
    positive_chip_boost = parse_positive_chip_boost(positive_chip_boost)
    removed_features = [
        name.strip() for name in str(feature_name).split(',') if name.strip()
    ]
    if not removed_features:
        print("Error: --ablation-feature requires at least one feature name")
        return 1
    if len(set(removed_features)) != len(removed_features):
        print("Error: --ablation-feature contains duplicate names")
        return 1
    augment_components, feature_augmentation, packet_augmentation = (
        resolve_training_augmentation(augment)
    )
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
    print(f"Removed features: {', '.join(removed_features)}")
    print(f"Seed: {seed}")
    print(
        "Augmentation: "
        + format_augmentation_config(
            feature_augmentation,
            packet_augmentation,
            components=augment_components,
        )
    )
    print("Artifacts: unchanged")

    matrix, _ = load_training_matrix(
        environment_filter=environment_filter,
        excluded_chips=excluded_chips,
        feature_names=TRAINING_FEATURES,
        use_cache=use_cache,
        timing_quality_policy=timing_quality_policy,
        timing_warn_weight=timing_warn_weight,
    )
    if not matrix['stats']['chips']:
        print("Error: No datasets found in data/")
        return 1

    augmented_matrix = None
    if packet_augmentation:
        print("Loading packet-augmented training matrix...")
        augmented_matrix, _ = load_training_matrix(
            environment_filter=environment_filter,
            excluded_chips=excluded_chips,
            feature_names=TRAINING_FEATURES,
            use_cache=use_cache,
            packet_augmentation=packet_augmentation,
            augmentation_seeds=training_packet_augmentation_seeds(
                packet_augmentation
            ),
            timing_quality_policy=timing_quality_policy,
            timing_warn_weight=timing_warn_weight,
        )

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
    if augmented_matrix is not None:
        baseline_dataset.update({
            'X_aug': np.asarray(augmented_matrix['X'], dtype=np.float32),
            'y_aug': np.asarray(augmented_matrix['y'], dtype=np.int8),
            'groups_aug': augmented_matrix['sample_context'][
                DEFAULT_PRIMARY_GROUP_KEY
            ],
        })

    for removed_feature in removed_features:
        requested = [
            name.strip() for name in removed_feature.split('+') if name.strip()
        ]
        unknown = [
            name for name in requested
            if name not in baseline_dataset['feature_names']
        ]
        if not requested or unknown:
            print(
                "Error: unknown ablation feature(s) '"
                + ', '.join(unknown or requested)
                + "'. "
                "Available features: "
                + ', '.join(baseline_dataset['feature_names'])
            )
            return 1

    baseline = evaluate_architecture_candidate(
        'production baseline',
        DEFAULT_HIDDEN_LAYERS,
        seed,
        baseline_dataset,
        scaler_mode,
        batch_size,
        fp_weight,
        feature_augmentation=feature_augmentation or None,
    )
    for removed_feature in removed_features:
        candidate_dataset = build_feature_ablation_dataset(
            baseline_dataset,
            removed_feature,
        )
        candidate = evaluate_architecture_candidate(
            f"Drop {removed_feature}",
            DEFAULT_HIDDEN_LAYERS,
            seed,
            candidate_dataset,
            scaler_mode,
            batch_size,
            fp_weight,
            feature_augmentation=feature_augmentation or None,
        )
        _print_feature_ablation_comparison(baseline, candidate)
        if deployment_candidate_beats_baseline(candidate, baseline):
            print(
                "Paired-first result: candidate ranks above the production "
                "baseline for this seed."
            )
        else:
            print(
                "Paired-first result: candidate does not beat the production "
                "baseline for this seed."
            )
    return 0


def experiment_architectures(scaler_mode=DEFAULT_SCALER_MODE,
                             batch_size=DEFAULT_BATCH_SIZE,
                             fp_weight=DEFAULT_FP_WEIGHT,
                             feature_names=None,
                             environment_filter=None,
                             excluded_chips=None,
                             architectures=None,
                             positive_chip_boost=None,
                             output_path=DEFAULT_EXPERIMENT_OUTPUT,
                             use_cache=True,
                             timing_quality_policy=DEFAULT_TIMING_QUALITY_POLICY,
                             timing_warn_weight=DEFAULT_TIMING_WARN_WEIGHT):
    """Run the FP-first architecture campaign without changing artifacts."""
    environment_filter = parse_environment_filter(environment_filter)
    excluded_chips = parse_chip_filter(excluded_chips)
    positive_chip_boost = parse_positive_chip_boost(positive_chip_boost)
    feature_names = list(feature_names or TRAINING_FEATURES)
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
    print(f"Feature set: {', '.join(feature_names)}")
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
        feature_names=feature_names,
        use_cache=use_cache,
        timing_quality_policy=timing_quality_policy,
        timing_warn_weight=timing_warn_weight,
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
            'timing_quality_policy': timing_quality_policy,
            'timing_warn_weight': float(timing_warn_weight),
            'training_sample_contract': TRAINING_SAMPLE_CONTRACT,
            'screening_seed': screening_seed,
            'initial_seeds': list(DEFAULT_EXPERIMENT_INITIAL_SEEDS),
            'final_seeds': list(DEFAULT_EXPERIMENT_FINAL_SEEDS),
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
    return 0


def experiment_fp_weights(fp_weights=None, scaler_mode=DEFAULT_SCALER_MODE,
                          batch_size=DEFAULT_BATCH_SIZE, hidden_layers=None,
                          feature_names=None,
                          environment_filter=None, excluded_chips=None,
                          positive_chip_boost=None,
                          output_path=DEFAULT_FP_WEIGHT_EXPERIMENT_OUTPUT,
                          use_cache=True,
                          timing_quality_policy=DEFAULT_TIMING_QUALITY_POLICY,
                          timing_warn_weight=DEFAULT_TIMING_WARN_WEIGHT):
    """Run a gated, multi-seed FP-weight campaign."""
    weights = parse_fp_weight_sweep(fp_weights or DEFAULT_FP_WEIGHT_SWEEP)
    if DEFAULT_FP_WEIGHT not in weights:
        weights.insert(0, DEFAULT_FP_WEIGHT)
    else:
        weights = [DEFAULT_FP_WEIGHT] + [value for value in weights if value != DEFAULT_FP_WEIGHT]
    hidden_layers = list(hidden_layers or DEFAULT_HIDDEN_LAYERS)
    feature_names = list(feature_names or TRAINING_FEATURES)
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
    print(f"Feature set: {', '.join(feature_names)}")
    print("Artifacts: unchanged during evaluation")

    matrix, _ = load_training_matrix(
        environment_filter=environment_filter,
        excluded_chips=excluded_chips,
        feature_names=feature_names,
        use_cache=use_cache,
        timing_quality_policy=timing_quality_policy,
        timing_warn_weight=timing_warn_weight,
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
            'feature_names': list(feature_names),
            'timing_quality_policy': timing_quality_policy,
            'timing_warn_weight': float(timing_warn_weight),
            'training_sample_contract': TRAINING_SAMPLE_CONTRACT,
            'screening_seed': screening_seed,
            'initial_seeds': list(DEFAULT_EXPERIMENT_INITIAL_SEEDS),
            'final_seeds': list(DEFAULT_EXPERIMENT_FINAL_SEEDS),
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
    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Train ML motion detection model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=''
    )
    parser.add_argument('--info', action='store_true', 
                       help='Show dataset information')
    parser.add_argument('--experiment', action='store_true',
                       help='Run the FP-first MLP topology campaign')
    parser.add_argument('--experiment-output', type=Path, default=DEFAULT_EXPERIMENT_OUTPUT,
                       help='JSON output path for --experiment results '
                            f'(default: {DEFAULT_EXPERIMENT_OUTPUT})')
    parser.add_argument('--experiment-architectures', type=parse_architecture_sweep, default=None,
                       help='Semicolon-separated hidden-layer specs for --experiment, '
                            'e.g. "16,8;24,12;32,16;24;24,12,6"')
    parser.add_argument('--experiment-fp-weights', type=parse_fp_weight_sweep, default=None,
                       metavar='WEIGHTS',
                       help='Run a gated multi-seed campaign over comma-separated FP weights')
    parser.add_argument('--fp-weight-experiment-output', type=Path,
                       default=DEFAULT_FP_WEIGHT_EXPERIMENT_OUTPUT,
                       help='JSON output path for --experiment-fp-weights '
                            f'(default: {DEFAULT_FP_WEIGHT_EXPERIMENT_OUTPUT})')
    parser.add_argument('--seed', type=int, default=None,
                       help='Training seed. When omitted, reuse the seed embedded '
                            'in the current exported model when available; '
                            'otherwise generate a random seed. '
                            '--seed-search-until-improvement always samples fresh seeds')
    parser.add_argument('--augment', nargs='?',
                       const=','.join(DEFAULT_TRAINING_AUGMENT_COMPONENTS),
                       type=parse_augmentation_components, default=None,
                       metavar='COMPONENTS',
                       help='Apply one or more train-time augmentation components. '
                            '--augment with no value enables base, drift, and '
                            'burst-loss. '
                            'Supported comma-separated components: '
                            'base, drift, burst-loss. Inference stays unaugmented')
    parser.add_argument('--seed-search-until-improvement', type=int, default=0, metavar='MAX_TRIALS',
                       help='Evaluate MAX_TRIALS auto-generated seeds, require '
                            'deployment safety and per-recording non-regression, '
                            'then keep the strongest material worst/tail grouped-CV '
                            'improvement. A reserved holdout, when configured, is '
                            'opened only for the selected winner. Host-side '
                            'candidate searches run in memory without exporting '
                            'runtime artifacts; pass --no-export to use the same '
                            'mode for runtime-supported feature sets')
    parser.add_argument('--seed-search-output', type=Path, default=DEFAULT_SEED_SEARCH_OUTPUT,
                       help='JSON output path for --seed-search-until-improvement, '
                            'holding the per-replay rows and the exact comparisons '
                            'that blocked each candidate. Written after every trial. '
                            f'(default: {DEFAULT_SEED_SEARCH_OUTPUT})')
    parser.add_argument('--gain-stress-gate', action='store_true',
                       help='Evaluate current exported ML artifacts under '
                            'artificial gain scaling without training/exporting')
    parser.add_argument('--gain-stress-scales', type=parse_gain_stress_scales,
                       default=DEFAULT_GAIN_STRESS_SCALES,
                       help='Comma-separated gain multipliers for --gain-stress-gate '
                            f'(default: {",".join(map(str, DEFAULT_GAIN_STRESS_SCALES))})')
    parser.add_argument('--fp-weight', type=float, default=DEFAULT_FP_WEIGHT,
                       help='Multiplier for IDLE class weight to penalize false positives. '
                            f'Values >1.0 make the model more conservative (default: {DEFAULT_FP_WEIGHT:g})')
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
    parser.add_argument('--features', type=str, default=None, metavar='NAME1,NAME2,...',
                       help='Comma-separated feature set for training/evaluation '
                            'experiments (default: promoted Subband 7F production set). Host-side '
                            'candidates from tools/lib/candidate_features.py are '
                            'selectable too; they have no C++ extractor id, so they '
                            'require --no-export or an evaluation-only flow until '
                            'they are promoted and added to CPP_FEATURE_IDS')
    parser.add_argument('--trajectory-bin-ms', type=int,
                       default=CHANNEL_SHAPE_BIN_US // 1000,
                       metavar='MS',
                       help='Host-side trajectory-bin experiment in milliseconds '
                            f'(default: {CHANNEL_SHAPE_BIN_US // 1000}; '
                            'non-default values cannot export runtime artifacts)')
    parser.add_argument('--evaluate-gates', action='store_true',
                       help='Run the deployment replay gates and report them without '
                            'exporting runtime artifacts. This is how a host-side '
                            'candidate feature is measured against the paired and '
                            'quiet gates, since a candidate cannot be exported')
    parser.add_argument('--no-export', action='store_true',
                       help='Leave runtime artifacts unchanged (CV-only for normal training; '
                            'also use with --shap / --ablation-feature diagnostics)')
    parser.add_argument('--force-promote', action='store_true',
                       help='Export runtime artifacts even when the deployment '
                            'safety gates fail or regress. Gates still run and '
                            'report; use only for a deliberate, explicit '
                            'baseline reset with a fixed --seed')
    parser.add_argument('--no-cache', action='store_true',
                       help='Rebuild the training feature matrix instead of using the local cache')
    parser.add_argument('--timing-quality-policy', type=parse_timing_quality_policy,
                       default=DEFAULT_TIMING_QUALITY_POLICY,
                       help='Apply conservative timing-provenance controls before fitting. '
                            'keep: metadata only; exclude-fail: drop poor-timing files; '
                            'downweight-warn: reduce degraded files; '
                            'exclude-fail-downweight-warn: combine both')
    parser.add_argument('--timing-warn-weight', type=float,
                       default=DEFAULT_TIMING_WARN_WEIGHT,
                       help='Per-window weight for degraded timing files when the policy '
                            f'includes downweight-warn (default: {DEFAULT_TIMING_WARN_WEIGHT:g})')
    parser.add_argument('--environment', type=str, default=None,
                       help='Restrict training/evaluation to one or more named environments '
                            '(comma-separated, e.g. bedroom or bedroom,living_room)')
    parser.add_argument('--exclude-chip', type=str,
                       default=','.join(DEFAULT_EXCLUDED_CHIPS),
                       help='Exclude one or more chips from training/evaluation '
                            '(comma-separated, e.g. ESP32 or ESP32,S3; '
                            f'default: {",".join(DEFAULT_EXCLUDED_CHIPS)})')
    parser.add_argument('--shap', type=int, nargs='?', const=200, default=None,
                       metavar='SAMPLES',
                       help='Calculate grouped out-of-fold SHAP importance '
                            '(default: 200 balanced held-out samples)')
    parser.add_argument('--correlation', action='store_true',
                       help='Calculate correlation of selected training features with motion label')
    parser.add_argument('--ablation-feature', type=str, default=None,
                       help='Compare the production baseline against one or more '
                            'comma-separated independent removals; join names '
                            'with + for one joint removal. Uses grouped CV and '
                            'paired validation without exporting artifacts')
    parser.add_argument('--cross-environment', action='store_true',
                       help='Leave-one-environment-out generalization check: train on all '
                            'other named environments and evaluate on the held-out room. '
                            'Diagnostic only; does not train a promotable model or export artifacts')
    parser.add_argument('--cross-chip', action='store_true',
                       help='Leave-one-chip-out generalization check: train on all other chips '
                            'and evaluate on the held-out chip. '
                            'Diagnostic only; does not train a promotable model or export artifacts')
    args = parser.parse_args()
    try:
        set_active_trajectory_bin_ms(args.trajectory_bin_ms)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1
    if args.timing_warn_weight <= 0.0 or args.timing_warn_weight > 1.0:
        print("Error: --timing-warn-weight must be in the range (0.0, 1.0]")
        return 1
    set_active_torch_device(args.device)
    selected_training_features = list(TRAINING_FEATURES)
    if args.features is not None:
        selected_training_features = [
            name.strip() for name in args.features.split(',') if name.strip()
        ]
        if not selected_training_features:
            print("Error: --features requires at least one feature name")
            return 1
        unknown = [
            name for name in selected_training_features
            if name not in selectable_features()
        ]
        if unknown:
            print(
                f"Error: unknown feature(s): {', '.join(unknown)}. "
                f"Available: {', '.join(ALL_FEATURES)}"
                + (
                    f" (host-side candidates: {', '.join(CANDIDATE_FEATURES)})"
                    if CANDIDATE_FEATURES else ""
                )
            )
            return 1
        host_only = [
            name for name in selected_training_features
            if name not in EXPORTED_FEATURE_NAMES
        ]
        if len(set(selected_training_features)) != len(selected_training_features):
            print("Error: --features contains duplicate names")
            return 1
        # Plain training exports runtime artifacts. Host-side seed searches use
        # in-memory gates and remain export-free until the feature is promoted.
        will_export = (
            args.seed_search_until_improvement == 0
            and not (
                args.no_export
                or args.evaluate_gates
                or args.shap is not None
                or args.ablation_feature
                or args.correlation
                or args.cross_environment
                or args.cross_chip
                or args.gain_stress_gate
                or args.experiment
                or args.experiment_fp_weights is not None
                or args.info
            )
        )
        unsupported = [
            name for name in selected_training_features
            if name not in CPP_FEATURE_IDS
        ]
        if will_export and unsupported:
            print(
                "Error: feature(s) without a C++ extractor id cannot be "
                f"exported: {', '.join(unsupported)}. Use --no-export or "
                "--evaluate-gates until they are promoted"
            )
            return 1
        if host_only:
            print(
                "Host-side-only features enabled: "
                + ', '.join(host_only)
            )
        print(f"Selected features ({len(selected_training_features)}): "
              + ', '.join(selected_training_features))

    if args.info:
        show_info()
        return 0

    if (
        ACTIVE_TRAJECTORY_BIN_US != CHANNEL_SHAPE_BIN_US
        and not (
            args.no_export
            or args.evaluate_gates
            or args.shap is not None
            or args.ablation_feature
            or args.correlation
            or args.cross_environment
            or args.cross_chip
            or args.experiment
            or args.experiment_fp_weights is not None
        )
    ):
        print(
            "Error: a non-default --trajectory-bin-ms is experimental and "
            "requires a read-only flow such as --no-export or --evaluate-gates"
        )
        return 1
    if ACTIVE_TRAJECTORY_BIN_US != CHANNEL_SHAPE_BIN_US:
        print(
            "Trajectory bin experiment: "
            f"{ACTIVE_TRAJECTORY_BIN_US / 1000:g} ms (runtime artifacts unchanged)"
        )

    experiment_count = sum((
        bool(args.experiment),
        args.experiment_fp_weights is not None,
    ))
    if experiment_count > 1:
        print("Error: experiment campaigns are mutually exclusive")
        return 1

    if args.force_promote:
        if args.no_export:
            print("Error: --force-promote and --no-export are mutually exclusive")
            return 1
        if args.seed is None:
            print("Error: --force-promote requires an explicit --seed so the "
                  "bypassed candidate is deliberate and reproducible")
            return 1
        if (args.seed_search_until_improvement > 0
                or args.experiment
                or args.experiment_fp_weights is not None
                or args.gain_stress_gate
                or args.cross_environment
                or args.cross_chip
                or args.shap is not None
                or args.ablation_feature
                or args.correlation):
            print("Error: --force-promote applies only to a plain single-seed "
                  "training run")
            return 1
    if args.augment and (
        args.experiment
        or args.experiment_fp_weights is not None
        or args.gain_stress_gate
        or args.correlation
    ):
        print(
            "Error: --augment applies only to production training, seed search, "
            "grouped OOF SHAP, targeted ablation, and "
            "cross-environment/cross-chip diagnostics"
        )
        return 1

    if args.gain_stress_gate:
        if args.experiment or args.experiment_fp_weights is not None:
            print("Error: --gain-stress-gate cannot be combined with experiment flows")
            return 1
        if args.seed_search_until_improvement > 0 or args.seed is not None:
            print("Error: --gain-stress-gate evaluates exported artifacts and cannot use --seed or seed-search")
            return 1
        if args.shap is not None or args.ablation_feature or args.correlation:
            print("Error: --gain-stress-gate cannot be combined with --shap, --ablation-feature, or --correlation")
            return 1
        if any(name not in EXPORTED_FEATURE_NAMES for name in selected_training_features):
            print("Error: --gain-stress-gate evaluates exported artifacts and cannot use host-only --features")
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
        if args.experiment or args.experiment_fp_weights is not None:
            print(f"Error: {mode} cannot be combined with experiment flows")
            return 1
        if args.shap is not None or args.ablation_feature or args.correlation:
            print(f"Error: {mode} cannot be combined with --shap, --ablation-feature, or --correlation")
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
            feature_names=selected_training_features,
            environment_filter=args.environment,
            excluded_chips=args.exclude_chip,
            architectures=args.experiment_architectures,
            positive_chip_boost=None,
            output_path=args.experiment_output,
            use_cache=not args.no_cache,
            timing_quality_policy=args.timing_quality_policy,
            timing_warn_weight=args.timing_warn_weight,
        )

    if args.experiment_fp_weights is not None:
        return experiment_fp_weights(
            fp_weights=args.experiment_fp_weights,
            scaler_mode=args.scaler,
            batch_size=args.batch_size,
            hidden_layers=args.hidden_layers,
            feature_names=selected_training_features,
            environment_filter=args.environment,
            excluded_chips=args.exclude_chip,
            positive_chip_boost=None,
            output_path=args.fp_weight_experiment_output,
            use_cache=not args.no_cache,
            timing_quality_policy=args.timing_quality_policy,
            timing_warn_weight=args.timing_warn_weight,
        )

    if args.ablation_feature:
        return experiment_feature_ablation(
            feature_name=args.ablation_feature,
            seed=args.seed,
            scaler_mode=args.scaler,
            batch_size=args.batch_size,
            fp_weight=args.fp_weight,
            environment_filter=args.environment,
            excluded_chips=args.exclude_chip,
            positive_chip_boost=None,
            use_cache=not args.no_cache,
            augment=args.augment,
            timing_quality_policy=args.timing_quality_policy,
            timing_warn_weight=args.timing_warn_weight,
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
        if args.shap is not None or args.ablation_feature:
            print("Error: --seed-search-until-improvement cannot be combined with --shap or --ablation-feature")
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
            positive_chip_boost=None,
            use_cache=not args.no_cache,
            augment=args.augment,
            timing_quality_policy=args.timing_quality_policy,
            timing_warn_weight=args.timing_warn_weight,
            search_output_path=args.seed_search_output,
            export_artifacts=not args.no_export,
        )

    train_rc, _, _ = train_all(
        fp_weight=args.fp_weight, 
        seed=args.seed,
        feature_names=selected_training_features,
        feature_importance=args.shap is not None,
        ablation=False,
        shap_samples=args.shap if args.shap is not None else 200,
        hidden_layers=args.hidden_layers,
        scaler_mode=args.scaler,
        batch_size=args.batch_size,
        environment_filter=args.environment,
        excluded_chips=args.exclude_chip,
        positive_chip_boost=None,
        use_cache=not args.no_cache,
        augment=args.augment,
        timing_quality_policy=args.timing_quality_policy,
        timing_warn_weight=args.timing_warn_weight,
        export_artifacts=(
            not args.no_export
            and not args.evaluate_gates
            and args.shap is None
        ),
        evaluate_deployment=(
            (args.evaluate_gates or not args.no_export)
            and args.shap is None
        ),
        force_export=args.force_promote,
    )
    return train_rc


if __name__ == '__main__':
    exit(main())
