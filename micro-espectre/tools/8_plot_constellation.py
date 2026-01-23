#!/usr/bin/env python3
"""
Plot I/Q Constellation Diagrams for CSI Subcarriers

Visualizes the constellation diagrams (I/Q plots) for selected subcarriers,
comparing baseline (stable) vs movement (dispersed) patterns.
Uses a limited number of contiguous packets to avoid overcrowding.

Usage:
    python tools/8_plot_constellation.py              # Use C6 dataset
    python tools/8_plot_constellation.py --chip S3    # Use S3 dataset
    python tools/8_plot_constellation.py --chip S3 --packets 50
    python tools/8_plot_constellation.py --subcarriers 11,12,13,14

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import sys
import numpy as np
import argparse
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from csi_utils import load_baseline_and_movement, find_dataset, DEFAULT_SUBCARRIERS

# Alias for backward compatibility
SELECTED_SUBCARRIERS = DEFAULT_SUBCARRIERS

def extract_iq_data(packets, subcarriers, num_packets=50, offset=0):
    """
    Extract I/Q data for selected subcarriers from packets
    
    Args:
        packets: List of CSI packets
        subcarriers: List of subcarrier indices to extract
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

def plot_constellation_comparison(baseline_packets, movement_packets, 
                                 subcarriers, num_packets=50, offset=0,
                                 total_subcarriers=64):
    """
    Plot I/Q constellation diagrams comparing baseline and movement
    
    Creates a 2x2 grid:
    - Top row: All subcarriers (baseline vs movement)
    - Bottom row: Only selected subcarriers (baseline vs movement)
    
    Args:
        baseline_packets: List of baseline packets
        movement_packets: List of movement packets
        subcarriers: List of subcarrier indices to plot
        num_packets: Number of contiguous packets to use
        offset: Starting packet index
        total_subcarriers: Total number of subcarriers in the dataset (64 for HT20 mode)
    """
    # Extract I/Q data for all subcarriers (top row)
    print(f"Extracting I/Q data for {num_packets} packets (offset={offset})...")
    all_subcarriers = list(range(total_subcarriers))
    baseline_iq_all = extract_iq_data(baseline_packets, all_subcarriers, num_packets, offset)
    movement_iq_all = extract_iq_data(movement_packets, all_subcarriers, num_packets, offset)
    
    # Extract I/Q data for selected subcarriers (bottom row)
    baseline_iq = extract_iq_data(baseline_packets, subcarriers, num_packets, offset)
    movement_iq = extract_iq_data(movement_packets, subcarriers, num_packets, offset)
    
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
        I_vals = baseline_iq_all[sc_idx]['I']
        Q_vals = baseline_iq_all[sc_idx]['Q']
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
        I_vals = movement_iq_all[sc_idx]['I']
        Q_vals = movement_iq_all[sc_idx]['Q']
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
        I_vals = baseline_iq[sc_idx]['I']
        Q_vals = baseline_iq[sc_idx]['Q']
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
        I_vals = movement_iq[sc_idx]['I']
        Q_vals = movement_iq[sc_idx]['Q']
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
        I_vals = np.array(baseline_iq[sc_idx]['I'])
        Q_vals = np.array(baseline_iq[sc_idx]['Q'])
        I_std = np.std(I_vals)
        Q_std = np.std(Q_vals)
        print(f"  SC {sc_idx:2d}: I_std={I_std:6.2f}, Q_std={Q_std:6.2f}, "
              f"I_range=[{I_vals.min():6.1f}, {I_vals.max():6.1f}], "
              f"Q_range=[{Q_vals.min():6.1f}, {Q_vals.max():6.1f}]")
    
    print(f"\nMovement (packets {offset} to {offset + num_packets}):")
    for sc_idx in subcarriers[:min(4, len(subcarriers))]:
        I_vals = np.array(movement_iq[sc_idx]['I'])
        Q_vals = np.array(movement_iq[sc_idx]['Q'])
        I_std = np.std(I_vals)
        Q_std = np.std(Q_vals)
        print(f"  SC {sc_idx:2d}: I_std={I_std:6.2f}, Q_std={Q_std:6.2f}, "
              f"I_range=[{I_vals.min():6.1f}, {I_vals.max():6.1f}], "
              f"Q_range=[{Q_vals.min():6.1f}, {Q_vals.max():6.1f}]")
    
    print("\n" + "="*80 + "\n")
    
    plt.show()

def plot_single_subcarrier_grid(baseline_packets, movement_packets, 
                                subcarriers, num_packets=50, offset=0,
                                total_subcarriers=64):
    """
    Plot individual constellation diagrams for each subcarrier in a grid
    
    Creates a grid of subplots, one for each subcarrier, showing baseline
    and movement overlaid with different colors.
    
    Args:
        baseline_packets: List of baseline packets
        movement_packets: List of movement packets
        subcarriers: List of subcarrier indices to plot
        num_packets: Number of contiguous packets to use
        offset: Starting packet index
        total_subcarriers: Total number of subcarriers in the dataset (unused, for API consistency)
    """
    # Extract I/Q data
    print(f"Extracting I/Q data for {num_packets} packets (offset={offset})...")
    baseline_iq = extract_iq_data(baseline_packets, subcarriers, num_packets, offset)
    movement_iq = extract_iq_data(movement_packets, subcarriers, num_packets, offset)
    
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
        I_base = baseline_iq[sc_idx]['I']
        Q_base = baseline_iq[sc_idx]['Q']
        ax.scatter(I_base, Q_base, color='blue', alpha=0.5, s=20, label='Baseline')
        
        # Plot movement (red)
        I_move = movement_iq[sc_idx]['I']
        Q_move = movement_iq[sc_idx]['Q']
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
    plt.show()

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
  python tools/8_plot_constellation.py --chip C6 --packets 200
  
  # Plot specific subcarriers
  python tools/8_plot_constellation.py --subcarriers 11,12,13,14
  
  # Use grid layout (one subplot per subcarrier)
  python tools/8_plot_constellation.py --chip S3 --grid
        """
    )
    
    parser.add_argument('--chip', type=str, default='C6',
                       help='Chip type to use: C6, S3, etc. (default: C6)')
    parser.add_argument('--packets', type=int, default=50,
                       help='Number of contiguous packets to plot (default: 50)')
    parser.add_argument('--offset', type=int, default=0,
                       help='Starting packet index (default: 0)')
    parser.add_argument('--subcarriers', type=str, default=None,
                       help='Comma-separated list of subcarrier indices (default: central subcarriers)')
    parser.add_argument('--grid', action='store_true',
                       help='Use grid layout (one subplot per subcarrier)')
    
    args = parser.parse_args()
    
    # Find dataset files dynamically
    chip = args.chip.upper()
    try:
        baseline_file, movement_file, chip_name = find_dataset(chip=chip)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print(f"\nCollect data using: ./me collect --label baseline --duration 10")
        return
    
    # Determine subcarriers to use
    if args.subcarriers:
        subcarriers = [int(x.strip()) for x in args.subcarriers.split(',')]
    else:
        subcarriers = SELECTED_SUBCARRIERS
    
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
    print(f"  Baseline: {baseline_file.name}")
    print(f"  Movement: {movement_file.name}")
    try:
        baseline_packets, movement_packets = load_baseline_and_movement(
            baseline_file=baseline_file,
            movement_file=movement_file
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    print(f"   Baseline: {len(baseline_packets)} packets")
    print(f"   Movement: {len(movement_packets)} packets")
    
    # Validate subcarrier indices (HT20 mode = 64 subcarriers)
    num_subcarriers = 64
    invalid_subcarriers = [sc for sc in subcarriers if sc < 0 or sc >= num_subcarriers]
    if invalid_subcarriers:
        print(f"\nError: Invalid subcarrier indices for {chip_name} dataset: {invalid_subcarriers}")
        print(f"       Valid range: 0-{num_subcarriers - 1}")
        return
    
    # Validate offset and packet count
    max_packets = min(len(baseline_packets), len(movement_packets))
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
        plot_single_subcarrier_grid(baseline_packets, movement_packets, 
                                    subcarriers, args.packets, args.offset,
                                    total_subcarriers=num_subcarriers)
    else:
        plot_constellation_comparison(baseline_packets, movement_packets, 
                                     subcarriers, args.packets, args.offset,
                                     total_subcarriers=num_subcarriers)
    
    print("Done!\n")

if __name__ == '__main__':
    main()
