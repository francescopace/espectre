# SPDX-License-Identifier: GPL-3.0-only
# Commercial licensing available under separate agreement; see LICENSING.md.
"""ESPectre dataset quality validation command-line entrypoint."""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.lib import dataset_metadata  # noqa: E402
from tools.lib.dataset_quality import core
from tools.lib.dataset_quality.rendering import (
    _report_evaluation_view_is_current,
)
from tools.lib.dataset_quality.runner import (
    run_validation,
)

def main():
    parser = argparse.ArgumentParser(
        description="ESPectre Dataset Quality Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python validate_dataset_quality.py              # Full validation (auto report + metadata refresh)
  python validate_dataset_quality.py --chip C6    # Validate C6 only
  python validate_dataset_quality.py --data-dir data/untracked/example --preserve-pairs
  python validate_dataset_quality.py --data-dir data/untracked/example --diagnostic-all-phy
  python validate_dataset_quality.py --no-cache   # Bypass persisted validation artifacts
  python validate_dataset_quality.py --no-report  # Skip markdown report
        """
    )
    parser.add_argument('--chip', type=str, default=None,
                       help='Filter by chip type (e.g., C6, S3, C3, ESP32)')
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=core.DATA_DIR,
        help='Dataset root containing dataset_info.json and label directories',
    )
    parser.add_argument(
        '--report-output',
        type=Path,
        default=None,
        help='Report path (default: <data-dir>/auto_generated/DATASET_QUALITY_CHECK.md)',
    )
    parser.add_argument(
        '--preserve-pairs',
        action='store_true',
        help='Keep explicit catalog pairs instead of refreshing them by timestamp',
    )
    parser.add_argument(
        '--diagnostic-all-phy',
        action='store_true',
        help=(
            'Evaluate all explicit PHY rows after still reporting violations of '
            'the supported HT20/HT-LTF sensing contract'
        ),
    )
    parser.add_argument('--no-report', action='store_true',
                       help='Skip writing DATASET_QUALITY_CHECK.md')
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Bypass persisted time-aware ML rows for one run',
    )
    parser.add_argument(
        '--check-current',
        action='store_true',
        help='Exit successfully only when the report matches its current inputs',
    )

    args = parser.parse_args()
    core.configure_dataset_paths(args.data_dir, args.report_output)
    core.configure_validation_mode(diagnostic_all_phy=args.diagnostic_all_phy)

    if args.check_current:
        if dataset_metadata.generated_report_is_current(
            core.REPORT_OUTPUT,
            core.DATASET_INFO,
            input_paths=core._report_input_paths(),
        ) and _report_evaluation_view_is_current(chip_filter=args.chip):
            print(f"Current: {core.REPORT_OUTPUT}")
            sys.exit(0)
        print(
            f"Stale or missing: {core.REPORT_OUTPUT}; regenerate it from current inputs",
            file=sys.stderr,
        )
        sys.exit(1)

    exit_code = run_validation(
        chip_filter=args.chip,
        generate_report=not args.no_report,
        use_cache=not args.no_cache,
        refresh_pair_metadata=not args.preserve_pairs,
        diagnostic_all_phy=args.diagnostic_all_phy,
    )
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
