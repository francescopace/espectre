#!/usr/bin/env python3
"""
Plot I/Q Constellation Diagrams for CSI Subcarriers

Visualizes the constellation diagrams (I/Q plots) for the fixed production
subcarriers, comparing static presence (stable) vs motion (dispersed) patterns.
Uses a limited number of contiguous packets to avoid overcrowding.

Usage:
    python tools/8_plot_constellation.py              # Use C6 dataset
    python tools/8_plot_constellation.py --chip S3    # Use S3 dataset
    python tools/8_plot_constellation.py --chip S3 --packets 500 --offset 100
Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import numpy as np
import argparse
import matplotlib.pyplot as plt
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.lib.csi_io import load_static_presence_and_motion
from tools.lib.dataset_metadata import resolve_explicit_pair, select_dataset_interactively
from tools.lib.ui import show_plot_window
from config import DEFAULT_SUBCARRIERS

def extract_iq_data(packets, subcarriers, num_packets=500, offset=100):
    """
    Extract I/Q data for the requested plotting subcarriers from packets.
    
    Args:
        packets: List of CSI packets
        subcarriers: Subcarrier indices to extract for plotting
        num_packets: Number of contiguous packets to use
        offset: Starting packet index
    
    Returns:
        dict: {subcarrier_idx: {'I': [...], 'Q': [...]}}
    """
    end_idx = min(offset + num_packets, len(packets))
    selected_packets = packets[offset:end_idx]
    
    iq_data = {sc: {'I': [], 'Q': []} for sc in subcarriers}
    
    for pkt in selected_packets:
        csi_data = pkt['csi_data']
        for sc_idx in subcarriers:
            # Espressif CSI format: [Imaginary, Real, ...] per subcarrier
            q_idx = sc_idx * 2      # Imaginary first
            i_idx = sc_idx * 2 + 1  # Real second
            if i_idx < len(csi_data):
                I = float(csi_data[i_idx])
                Q = float(csi_data[q_idx])
                iq_data[sc_idx]['I'].append(I)
                iq_data[sc_idx]['Q'].append(Q)
    
    return iq_data

def plot_constellation_comparison(static_presence_packets, motion_packets, 
                                 subcarriers, num_packets=500, offset=100,
                                 total_subcarriers=64):
    """
    Plot I/Q constellation diagrams comparing static presence and motion.
    
    Creates a 2x2 grid:
    - Top row: All subcarriers (static presence vs motion)
    - Bottom row: Only the fixed production subcarriers (static presence vs motion)
    
    Args:
        static_presence_packets: List of static-presence packets
        motion_packets: List of motion packets
        subcarriers: Subcarrier indices to plot
        num_packets: Number of contiguous packets to use
        offset: Starting packet index
        total_subcarriers: Total number of subcarriers in the dataset (64 for HT20 mode)
    """
    # Extract I/Q data for all subcarriers (top row)
    print(f"Extracting I/Q data for {num_packets} packets (offset={offset})...")
    all_subcarriers = list(range(total_subcarriers))
    static_presence_iq_all = extract_iq_data(static_presence_packets, all_subcarriers, num_packets, offset)
    motion_iq_all = extract_iq_data(motion_packets, all_subcarriers, num_packets, offset)
    
    # Extract I/Q data for the fixed production subcarriers (bottom row)
    static_presence_iq = extract_iq_data(static_presence_packets, subcarriers, num_packets, offset)
    motion_iq = extract_iq_data(motion_packets, subcarriers, num_packets, offset)
    
    # Create color map for subcarriers
    colors = plt.cm.tab20(np.linspace(0, 1, len(subcarriers)))
    
    # Create figure with 2x2 layout
    fig = plt.figure(figsize=(20, 12))
    
    # Main title with subcarrier info
    sc_range = f"SC [{subcarriers[0]}-{subcarriers[-1]}]" if len(subcarriers) > 1 else f"SC {subcarriers[0]}"
    fig.suptitle(f'I/Q Constellation Diagrams - {total_subcarriers} SC Dataset - {sc_range}\n{num_packets} Packets (offset={offset})', 
                 fontsize=14, fontweight='bold')
    
    # Maximize window
    try:
        mng = plt.get_current_fig_manager()
        if hasattr(mng, 'window'):
            if hasattr(mng.window, 'showMaximized'):
                mng.window.showMaximized()
            elif hasattr(mng.window, 'state'):
                mng.window.state('zoomed')
        elif hasattr(mng, 'full_screen_toggle'):
            mng.full_screen_toggle()
    except Exception:
        pass
    
    # ========================================================================
    # TOP LEFT: Baseline - ALL 64 Subcarriers
    # ========================================================================
    ax1 = plt.subplot(2, 2, 1)
    for sc_idx in all_subcarriers:
        I_vals = static_presence_iq_all[sc_idx]['I']
        Q_vals = static_presence_iq_all[sc_idx]['Q']
        ax1.scatter(I_vals, Q_vals, color='gray', alpha=0.3, s=10)
    
    ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax1.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax1.set_xlabel('I (In-phase)', fontsize=11)
    ax1.set_ylabel('Q (Quadrature)', fontsize=11)
    ax1.set_title(f'Baseline - All {total_subcarriers} Subcarriers', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')
    
    # ========================================================================
    # TOP RIGHT: Movement - ALL Subcarriers
    # ========================================================================
    ax2 = plt.subplot(2, 2, 2)
    for sc_idx in all_subcarriers:
        I_vals = motion_iq_all[sc_idx]['I']
        Q_vals = motion_iq_all[sc_idx]['Q']
        ax2.scatter(I_vals, Q_vals, color='gray', alpha=0.3, s=10)
    
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax2.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax2.set_xlabel('I (In-phase)', fontsize=11)
    ax2.set_ylabel('Q (Quadrature)', fontsize=11)
    ax2.set_title(f'Movement - All {total_subcarriers} Subcarriers', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', adjustable='box')
    
    # ========================================================================
    # BOTTOM LEFT: Baseline - SELECTED Subcarriers Only
    # ========================================================================
    ax3 = plt.subplot(2, 2, 3)
    for i, sc_idx in enumerate(subcarriers):
        I_vals = static_presence_iq[sc_idx]['I']
        Q_vals = static_presence_iq[sc_idx]['Q']
        ax3.scatter(I_vals, Q_vals, color=colors[i], alpha=0.7, s=30, 
                   label=f'SC {sc_idx}', edgecolors='black', linewidth=0.5)
    
    ax3.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax3.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax3.set_xlabel('I (In-phase)', fontsize=11)
    ax3.set_ylabel('Q (Quadrature)', fontsize=11)
    ax3.set_title('Baseline - Selected Band', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=9, ncol=3, loc='upper right')
    ax3.set_aspect('equal', adjustable='box')
    
    # ========================================================================
    # BOTTOM RIGHT: Movement - SELECTED Subcarriers Only
    # ========================================================================
    ax4 = plt.subplot(2, 2, 4)
    for i, sc_idx in enumerate(subcarriers):
        I_vals = motion_iq[sc_idx]['I']
        Q_vals = motion_iq[sc_idx]['Q']
        ax4.scatter(I_vals, Q_vals, color=colors[i], alpha=0.7, s=30, 
                   label=f'SC {sc_idx}', edgecolors='black', linewidth=0.5)
    
    ax4.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax4.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax4.set_xlabel('I (In-phase)', fontsize=11)
    ax4.set_ylabel('Q (Quadrature)', fontsize=11)
    ax4.set_title('Movement - Selected Band', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=9, ncol=3, loc='upper right')
    ax4.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    # Print statistics
    print("\n" + "="*80)
    print("  CONSTELLATION STATISTICS (Selected Subcarriers)")
    print("="*80)
    print(f"\nBaseline (packets {offset} to {offset + num_packets}):")
    for sc_idx in subcarriers[:min(4, len(subcarriers))]:  # Show stats for first 4
        I_vals = np.array(static_presence_iq[sc_idx]['I'])
        Q_vals = np.array(static_presence_iq[sc_idx]['Q'])
        I_std = np.std(I_vals)
        Q_std = np.std(Q_vals)
        print(f"  SC {sc_idx:2d}: I_std={I_std:6.2f}, Q_std={Q_std:6.2f}, "
              f"I_range=[{I_vals.min():6.1f}, {I_vals.max():6.1f}], "
              f"Q_range=[{Q_vals.min():6.1f}, {Q_vals.max():6.1f}]")
    
    print(f"\nMovement (packets {offset} to {offset + num_packets}):")
    for sc_idx in subcarriers[:min(4, len(subcarriers))]:
        I_vals = np.array(motion_iq[sc_idx]['I'])
        Q_vals = np.array(motion_iq[sc_idx]['Q'])
        I_std = np.std(I_vals)
        Q_std = np.std(Q_vals)
        print(f"  SC {sc_idx:2d}: I_std={I_std:6.2f}, Q_std={Q_std:6.2f}, "
              f"I_range=[{I_vals.min():6.1f}, {I_vals.max():6.1f}], "
              f"Q_range=[{Q_vals.min():6.1f}, {Q_vals.max():6.1f}]")
    
    print("\n" + "="*80 + "\n")
    
    show_plot_window(plt)

def plot_single_subcarrier_grid(static_presence_packets, motion_packets, 
                                subcarriers, num_packets=500, offset=100,
                                total_subcarriers=64):
    """
    Plot individual constellation diagrams for each subcarrier in a grid
    
    Creates a grid of subplots, one for each subcarrier, showing static
    presence and motion overlaid with different colors.
    
    Args:
        static_presence_packets: List of static-presence packets
        motion_packets: List of motion packets
        subcarriers: Subcarrier indices to plot
        num_packets: Number of contiguous packets to use
        offset: Starting packet index
        total_subcarriers: Total number of subcarriers in the dataset (unused, for API consistency)
    """
    # Extract I/Q data
    print(f"Extracting I/Q data for {num_packets} packets (offset={offset})...")
    static_presence_iq = extract_iq_data(static_presence_packets, subcarriers, num_packets, offset)
    motion_iq = extract_iq_data(motion_packets, subcarriers, num_packets, offset)
    
    # Determine grid size
    n_subcarriers = len(subcarriers)
    n_cols = min(4, n_subcarriers)
    n_rows = (n_subcarriers + n_cols - 1) // n_cols
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 12))
    fig.suptitle(f'Individual Subcarrier Constellations - {num_packets} Packets (offset={offset})', 
                 fontsize=14, fontweight='bold')
    
    # Maximize window
    try:
        mng = plt.get_current_fig_manager()
        if hasattr(mng, 'window'):
            if hasattr(mng.window, 'showMaximized'):
                mng.window.showMaximized()
            elif hasattr(mng.window, 'state'):
                mng.window.state('zoomed')
        elif hasattr(mng, 'full_screen_toggle'):
            mng.full_screen_toggle()
    except Exception:
        pass
    
    # Flatten axes array for easier iteration
    if n_subcarriers == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Plot each subcarrier
    for idx, sc_idx in enumerate(subcarriers):
        ax = axes[idx]
        
        # Plot baseline (blue)
        I_base = static_presence_iq[sc_idx]['I']
        Q_base = static_presence_iq[sc_idx]['Q']
        ax.scatter(I_base, Q_base, color='blue', alpha=0.5, s=20, label='Baseline')
        
        # Plot movement (red)
        I_move = motion_iq[sc_idx]['I']
        Q_move = motion_iq[sc_idx]['Q']
        ax.scatter(I_move, Q_move, color='red', alpha=0.5, s=20, label='Movement')
        
        # Formatting
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
        ax.set_xlabel('I', fontsize=9)
        ax.set_ylabel('Q', fontsize=9)
        ax.set_title(f'Subcarrier {sc_idx}', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_aspect('equal', adjustable='box')
    
    # Hide unused subplots
    for idx in range(n_subcarriers, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    show_plot_window(plt)

def main():
    parser = argparse.ArgumentParser(
        description='Plot I/Q constellation diagrams for CSI subcarriers',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot using default C6 dataset
  python tools/8_plot_constellation.py
  
  # Plot using S3 dataset
  python tools/8_plot_constellation.py --chip S3
  
  # Plot with more packets
  python tools/8_plot_constellation.py --chip C6 --packets 800 --offset 100
  
  # Use grid layout (one subplot per subcarrier)
  python tools/8_plot_constellation.py --chip S3 --grid
        """
    )
    
    raw_args = __import__('sys').argv[1:]
    chip_explicit = '--chip' in raw_args
    parser.add_argument('--chip', type=str, default='C6',
                       help='Chip type to use: C6, S3, etc. (default: C6)')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Dataset filename, stem, or dataset id; pair is resolved from metadata')
    parser.add_argument('--interactive', action='store_true',
                       help='Choose the dataset interactively from dataset_info.json')
    parser.add_argument('--packets', type=int, default=500,
                       help='Number of contiguous packets to plot (default: 500)')
    parser.add_argument('--offset', type=int, default=100,
                       help='Starting packet index (default: 100)')
    parser.add_argument('--grid', action='store_true',
                       help='Use grid layout (one subplot per subcarrier)')
    
    args = parser.parse_args()
    
    # Find dataset files dynamically
    chip = args.chip.upper()
    try:
        chip_filter = chip if chip_explicit and not args.dataset else (None if args.dataset else chip)
        if args.interactive:
            selected = select_dataset_interactively(
                chip=chip if chip_explicit else None,
                num_sc=64,
                require_pair=True,
                prompt='Select dataset for constellation plotting',
            )
            pair = resolve_explicit_pair(dataset=selected.path.name, num_sc=64)
        else:
            pair = resolve_explicit_pair(dataset=args.dataset, chip=chip_filter, num_sc=64)
        static_presence_file = pair.static_presence.path
        motion_file = pair.motion.path
        chip_name = pair.chip
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print(f"\nCollect data using: ./espectre collect --label static_presence --duration 10")
        return
    
    # Always use the fixed production subcarriers.
    subcarriers = DEFAULT_SUBCARRIERS
    
    print("")
    print("╔═══════════════════════════════════════════════════════╗")
    print("║        I/Q Constellation Diagram Plotter              ║")
    print("╚═══════════════════════════════════════════════════════╝")
    print(f"\nConfiguration:")
    print(f"  Chip: {chip_name}")
    print(f"  Packets: {args.packets}")
    print(f"  Offset: {args.offset}")
    print(f"  Subcarriers: {subcarriers}")
    print(f"  Layout: {'Grid' if args.grid else 'Comparison'}")
    
    # Load data
    print(f"\nLoading data...")
    print(f"  Static presence: {static_presence_file.name}")
    print(f"  Motion:          {motion_file.name}")
    try:
        static_presence_packets, motion_packets = load_static_presence_and_motion(
            static_presence_file=static_presence_file,
            motion_file=motion_file,
            chip=chip_name,
            dataset=args.dataset,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    print(f"   Static presence: {len(static_presence_packets)} packets")
    print(f"   Motion:          {len(motion_packets)} packets")
    
    # Validate subcarrier indices (HT20 mode = 64 subcarriers)
    num_subcarriers = 64
    invalid_subcarriers = [sc for sc in subcarriers if sc < 0 or sc >= num_subcarriers]
    if invalid_subcarriers:
        print(f"\nError: Invalid subcarrier indices for {chip_name} dataset: {invalid_subcarriers}")
        print(f"       Valid range: 0-{num_subcarriers - 1}")
        return
    
    # Validate offset and packet count
    max_packets = min(len(static_presence_packets), len(motion_packets))
    if args.offset >= max_packets:
        print(f"\nError: Offset {args.offset} exceeds available packets ({max_packets})")
        return
    
    available_packets = max_packets - args.offset
    if args.packets > available_packets:
        print(f"\nWarning: Requested {args.packets} packets, but only {available_packets} available")
        print(f"         Using {available_packets} packets instead")
        args.packets = available_packets
    
    # Generate plots
    print(f"\nGenerating constellation plots...")
    
    if args.grid:
        plot_single_subcarrier_grid(static_presence_packets, motion_packets, 
                                    subcarriers, args.packets, args.offset,
                                    total_subcarriers=num_subcarriers)
    else:
        plot_constellation_comparison(static_presence_packets, motion_packets, 
                                     subcarriers, args.packets, args.offset,
                                     total_subcarriers=num_subcarriers)
    
    print("Done!\n")

if __name__ == '__main__':
    main()
