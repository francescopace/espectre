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
    python tools/train_ml_model.py                    # Train and export if the deployment gates pass
    python tools/train_ml_model.py --no-export        # Evaluate without replacing runtime artifacts
    python tools/train_ml_model.py --info             # Show dataset info
    python tools/train_ml_model.py --fp-weight 1.75   # Penalize FP 1.75x more
    python tools/train_ml_model.py --scaler clipped_standard --no-export
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

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT_PATH = SCRIPT_DIR.parent
if str(REPO_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_PATH))

from tools.lib.bootstrap import setup_paths
from tools.lib.ml_training import feature_cache as feature_cache_module

setup_paths()

import argparse
from tools.lib.csi_features import (
    ALL_FEATURES,
)
from tools.lib.candidate_features import (
    CANDIDATE_FEATURES,
)
from tools.lib.host_feature_trackers import (
    CHANNEL_SHAPE_BIN_US,
)
from tools.lib.high_accuracy_detector import (
    FEATURE_NAMES as EXPORTED_FEATURE_NAMES,
)

from tools.lib.ml_training.augmentation import (
    DEFAULT_TRAINING_AUGMENT_COMPONENTS,
    parse_augmentation_components,
)

from tools.lib.ml_training.dataset import (
    DEFAULT_EXCLUDED_CHIPS,
    DEFAULT_TIMING_QUALITY_POLICY,
    DEFAULT_TIMING_WARN_WEIGHT,
    parse_timing_quality_policy,
)

from tools.lib.ml_training.evaluation import (
    DEFAULT_GAIN_STRESS_SCALES,
    evaluate_gain_stress_gate,
    parse_gain_stress_scales,
    print_gain_stress_summary,
)

from tools.lib.ml_training.export import (
    CPP_FEATURE_IDS,
    DEFAULT_HIDDEN_LAYERS,
    DEFAULT_TORCH_DEVICE,
    set_active_torch_device,
)

from tools.lib.ml_training.feature_cache import (
    TRAINING_FEATURES,
    selectable_features,
    set_active_trajectory_bin_ms,
)

from tools.lib.ml_training.preprocessing import (
    DEFAULT_SCALER_MODE,
)

from tools.lib.ml_training.training import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_FP_WEIGHT,
    DEFAULT_SEED_SEARCH_OUTPUT,
    DEFAULT_CV_REPEATS,
    calculate_correlation_importance,
    cross_chip_validation,
    cross_environment_validation,
    parse_hidden_layers,
    print_correlation_table,
    set_cv_repeats,
    show_info,
    train_all,
    train_until_improvement,
)

def main():
    parser = argparse.ArgumentParser(
        description='Train ML motion detection model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=''
    )
    parser.add_argument('--info', action='store_true', 
                       help='Show dataset information')
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
                            'base, drift, burst-loss. '
                            'Inference stays unaugmented')
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
                           'standard', 'robust', 'session_balanced_robust',
                           'clipped_standard', 'log1p_standard'],
                       default=DEFAULT_SCALER_MODE,
                       help='Feature normalization mode; log1p_standard compresses '
                            'right-skewed non-negative features before z-scoring; '
                            'clipped_standard supports host-side CV only')
    parser.add_argument('--cv-repeats', type=int, default=DEFAULT_CV_REPEATS,
                       metavar='R',
                       help='Average out-of-fold probabilities over R grouped fold '
                            f'assignments (default: {DEFAULT_CV_REPEATS})')
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
                            'experiments (default: promoted Subband 8F production set). Host-side '
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
                       help='Run the final selection and holdout deployment replay '
                            'gates without exporting runtime artifacts. Use only after '
                            'the candidate and seed are fixed; repeated use opens the '
                            'reserved holdout')
    parser.add_argument('--evaluate-selection', action='store_true',
                       help='Run only the selection deployment replay gates without '
                            'exporting runtime artifacts. Use while comparing candidates '
                            'so the holdout remains sealed')
    parser.add_argument('--no-export', action='store_true',
                       help='Leave runtime artifacts unchanged (CV-only for normal training; '
                            'also use with --shap diagnostics)')
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
    parser.add_argument('--cross-environment', action='store_true',
                       help='Leave-one-environment-out generalization check: train on all '
                            'other named environments and evaluate on the held-out room. '
                            'Diagnostic only; does not train a promotable model or export artifacts')
    parser.add_argument('--cross-chip', action='store_true',
                       help='Leave-one-chip-out generalization check: train on all other chips '
                            'and evaluate on the held-out chip. '
                            'Diagnostic only; does not train a promotable model or export artifacts')
    args = parser.parse_args()
    if args.evaluate_gates and args.evaluate_selection:
        print("Error: --evaluate-gates and --evaluate-selection are mutually exclusive")
        return 1
    # Each mode replaces the plain training run, so at most one may be active.
    modes = [
        flag for flag, active in (
            ('--info', args.info),
            ('--gain-stress-gate', args.gain_stress_gate),
            ('--cross-environment', args.cross_environment),
            ('--cross-chip', args.cross_chip),
            ('--correlation', args.correlation),
            ('--seed-search-until-improvement', args.seed_search_until_improvement > 0),
            ('--shap', args.shap is not None),
        ) if active
    ]
    if len(modes) > 1:
        print(f"Error: {', '.join(modes[:-1])} and {modes[-1]} are mutually exclusive")
        return 1
    if modes and (args.evaluate_gates or args.evaluate_selection):
        flag = '--evaluate-gates' if args.evaluate_gates else '--evaluate-selection'
        print(f"Error: {flag} applies only to a single training run, not {modes[0]}")
        return 1
    if modes and args.force_promote:
        print(f"Error: --force-promote applies only to a single training run, not {modes[0]}")
        return 1
    # Every mode except seed search leaves runtime artifacts unchanged.
    read_only = (
        args.no_export
        or args.evaluate_gates
        or args.evaluate_selection
        or (bool(modes) and modes != ['--seed-search-until-improvement'])
    )
    try:
        set_active_trajectory_bin_ms(args.trajectory_bin_ms)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1
    if args.timing_warn_weight <= 0.0 or args.timing_warn_weight > 1.0:
        print("Error: --timing-warn-weight must be in the range (0.0, 1.0]")
        return 1
    set_active_torch_device(args.device)
    try:
        set_cv_repeats(args.cv_repeats)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1
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
        will_export = not read_only and args.seed_search_until_improvement == 0
        unsupported = [
            name for name in selected_training_features
            if name not in CPP_FEATURE_IDS
        ]
        if will_export and unsupported:
            print(
                "Error: feature(s) without a C++ extractor id cannot be "
                f"exported: {', '.join(unsupported)}. Use --no-export or "
                "--evaluate-selection until they are promoted"
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

    if feature_cache_module.ACTIVE_TRAJECTORY_BIN_US != CHANNEL_SHAPE_BIN_US and not read_only:
        print(
            "Error: a non-default --trajectory-bin-ms is experimental and "
            "requires a read-only flow such as --no-export or --evaluate-selection"
        )
        return 1
    if feature_cache_module.ACTIVE_TRAJECTORY_BIN_US != CHANNEL_SHAPE_BIN_US:
        print(
            "Trajectory bin experiment: "
            f"{feature_cache_module.ACTIVE_TRAJECTORY_BIN_US / 1000:g} ms "
            "(runtime artifacts unchanged)"
        )

    if args.force_promote:
        if read_only:
            print("Error: --force-promote cannot be combined with --no-export, "
                  "--evaluate-gates, or --evaluate-selection")
            return 1
        if args.seed is None:
            print("Error: --force-promote requires an explicit --seed so the "
                  "bypassed candidate is deliberate and reproducible")
            return 1
    if args.augment and (args.gain_stress_gate or args.correlation):
        print(
            "Error: --augment applies only to production training, seed search, "
            "grouped OOF SHAP, and "
            "cross-environment/cross-chip diagnostics"
        )
        return 1

    if args.gain_stress_gate:
        if args.seed is not None:
            print("Error: --gain-stress-gate evaluates exported artifacts and cannot use --seed")
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
        return train_until_improvement(
            max_trials=args.seed_search_until_improvement,
            fp_weight=args.fp_weight,
            feature_names=selected_training_features,
            hidden_layers=args.hidden_layers,
            scaler_mode=args.scaler,
            batch_size=args.batch_size,
            environment_filter=args.environment,
            excluded_chips=args.exclude_chip,
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
        shap_samples=args.shap if args.shap is not None else 200,
        hidden_layers=args.hidden_layers,
        scaler_mode=args.scaler,
        batch_size=args.batch_size,
        environment_filter=args.environment,
        excluded_chips=args.exclude_chip,
        use_cache=not args.no_cache,
        augment=args.augment,
        timing_quality_policy=args.timing_quality_policy,
        timing_warn_weight=args.timing_warn_weight,
        export_artifacts=(
            not args.no_export
            and not args.evaluate_gates
            and not args.evaluate_selection
            and args.shap is None
        ),
        evaluate_deployment=(
            (args.evaluate_gates or args.evaluate_selection or not args.no_export)
            and args.shap is None
        ),
        deployment_roles=(
            ('selection',)
            if args.evaluate_selection
            else ('selection', 'holdout')
        ),
        force_export=args.force_promote,
    )
    return train_rc

if __name__ == '__main__':
    raise SystemExit(main())
