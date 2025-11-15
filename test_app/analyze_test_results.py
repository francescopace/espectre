#!/usr/bin/env python3
"""
ESPectre - Test Results Analyzer

Parses JSON output from test suite and generates visualizations:
- ROC Curves
- Precision-Recall Curves
- Confusion Matrices
- Feature Importance
- Performance Reports

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def parse_test_output(log_file: Path) -> Dict:
    """Parse test output and extract JSON data."""
    results = {
        'performance_suite': None,
        'threshold_optimization': None,
        'temporal_robustness': None,
        'home_assistant': None,
        'segmentation': None
    }
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Extract JSON blocks
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    
    for match in re.finditer(json_pattern, content, re.DOTALL):
        try:
            data = json.loads(match.group())
            test_name = data.get('test_name', '')
            
            if 'performance_suite' in test_name:
                results['performance_suite'] = data
            elif 'threshold_optimization' in test_name:
                results['threshold_optimization'] = data
            elif 'temporal_robustness' in test_name:
                results['temporal_robustness'] = data
            elif 'home_assistant' in test_name:
                results['home_assistant'] = data
            elif 'segmentation' in test_name:
                results['segmentation'] = data
        except json.JSONDecodeError:
            continue
    
    return results


def plot_roc_curve(data: Dict, output_dir: Path):
    """Generate ROC curve plot."""
    if not data or 'roc_curve' not in data:
        print("⚠️  No ROC curve data found")
        return
    
    roc_curve = data['roc_curve']
    tpr = [point['tpr'] for point in roc_curve]
    fpr = [point['fpr'] for point in roc_curve]
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC={data["auc"]:.3f})')
    plt.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random Classifier')
    
    # Mark optimal point
    optimal_metrics = data['optimal_metrics']
    plt.plot(optimal_metrics['fp_rate'], optimal_metrics['recall'], 
             'go', markersize=12, label=f'Optimal (Recall={optimal_metrics["recall"]:.2f})')
    
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate (Recall)', fontsize=12)
    plt.title(f'ROC Curve - {data["feature"]}', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    output_file = output_dir / 'roc_curve.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ ROC curve saved to {output_file}")


def plot_precision_recall_curve(data: Dict, output_dir: Path):
    """Generate Precision-Recall curve plot."""
    if not data or 'roc_curve' not in data:
        print("⚠️  No precision-recall data found")
        return
    
    roc_curve = data['roc_curve']
    precision = [point['precision'] for point in roc_curve]
    recall = [point['tpr'] for point in roc_curve]
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, 'b-', linewidth=2, label='Precision-Recall Curve')
    
    # Mark optimal point
    optimal_metrics = data['optimal_metrics']
    plt.plot(optimal_metrics['recall'], optimal_metrics['precision'], 
             'go', markersize=12, label=f'Optimal (F1={optimal_metrics["f1_score"]:.2f})')
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'Precision-Recall Curve - {data["feature"]}', fontsize=14, fontweight='bold')
    plt.legend(loc='lower left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    output_file = output_dir / 'precision_recall_curve.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Precision-Recall curve saved to {output_file}")


def plot_confusion_matrix(data: Dict, output_dir: Path):
    """Generate confusion matrix heatmap (segmentation-based)."""
    if not data:
        print("⚠️  No confusion matrix data found")
        return
    
    # Check if this is segmentation-based data (new architecture)
    if 'segmentation' in data:
        seg = data['segmentation']
        baseline_samples = data['baseline_samples']
        movement_samples = data['movement_samples']
        
        tn = int((1 - seg['fp_rate']) * baseline_samples)
        fp = int(seg['fp_rate'] * baseline_samples)
        tp = int(seg['recall'] * movement_samples)
        fn = int((1 - seg['recall']) * movement_samples)
        
        title = f'Confusion Matrix - Segmentation\n' \
                f'Recall: {seg["recall"]*100:.1f}% | FP Rate: {seg["fp_rate"]*100:.1f}%'
    
    # Fallback to old feature-based data
    elif 'best_feature' in data:
        best = data['best_feature']
        baseline_samples = data['baseline_samples']
        movement_samples = data['movement_samples']
        
        tn = int((1 - best['fp_rate']) * baseline_samples)
        fp = int(best['fp_rate'] * baseline_samples)
        tp = int(best['recall'] * movement_samples)
        fn = int((1 - best['recall']) * movement_samples)
        
        title = f'Confusion Matrix - {best["name"]}\n' \
                f'Recall: {best["recall"]*100:.1f}% | FP Rate: {best["fp_rate"]*100:.1f}%'
    else:
        print("⚠️  No confusion matrix data found")
        return
    
    cm = np.array([[tn, fp], [fn, tp]])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted IDLE', 'Predicted MOTION'],
                yticklabels=['Actual IDLE', 'Actual MOTION'],
                cbar_kws={'label': 'Count'})
    
    plt.title(title, fontsize=14, fontweight='bold')
    
    output_file = output_dir / 'confusion_matrix.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Confusion matrix saved to {output_file}")


def plot_temporal_scenarios(data: Dict, output_dir: Path):
    """Generate temporal robustness scenario comparison."""
    if not data or 'scenarios' not in data:
        print("⚠️  No temporal scenario data found")
        return
    
    scenarios = data['scenarios']
    names = [s['name'] for s in scenarios]
    detection_rates = [s['detection_rate'] * 100 for s in scenarios]
    fp_rates = [s['false_alarm_rate'] * 100 for s in scenarios]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Detection rates
    colors = ['green' if rate >= 90 else 'orange' if rate >= 80 else 'red' 
              for rate in detection_rates]
    ax1.barh(names, detection_rates, color=colors, alpha=0.7)
    ax1.axvline(x=90, color='red', linestyle='--', linewidth=2, label='90% Target')
    ax1.set_xlabel('Detection Rate (%)', fontsize=12)
    ax1.set_title('Detection Rate by Scenario', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='x')
    
    # False alarm rates
    colors = ['green' if rate <= 10 else 'orange' if rate <= 20 else 'red' 
              for rate in fp_rates]
    ax2.barh(names, fp_rates, color=colors, alpha=0.7)
    ax2.axvline(x=10, color='red', linestyle='--', linewidth=2, label='10% Limit')
    ax2.set_xlabel('False Alarm Rate (%)', fontsize=12)
    ax2.set_title('False Alarm Rate by Scenario', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    output_file = output_dir / 'temporal_scenarios.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Temporal scenarios plot saved to {output_file}")


def plot_segmentation_analysis(data: Dict, output_dir: Path):
    """Generate segmentation visualization (4-panel plot)."""
    if not data or 'baseline_turbulence' not in data:
        print("⚠️  No segmentation data found")
        return
    
    baseline_turb = np.array(data['baseline_turbulence'])
    movement_turb = np.array(data['movement_turbulence'])
    threshold = data['threshold']
    mean_var = data['mean_variance']
    std_var = data['std_variance']
    window_size = data.get('window_size', 30)
    
    baseline_segments = data.get('baseline_segments', [])
    movement_segments = data.get('movement_segments', [])
    
    # Calculate moving variance (replicate Python algorithm)
    def calculate_moving_variance(signal, window):
        if len(signal) < window:
            return np.zeros(len(signal))
        
        moving_var = np.zeros(len(signal))
        for i in range(window - 1, len(signal)):
            window_data = signal[i - window + 1 : i + 1]
            moving_var[i] = np.var(window_data)
        
        return moving_var
    
    baseline_moving_var = calculate_moving_variance(baseline_turb, window_size)
    movement_moving_var = calculate_moving_variance(movement_turb, window_size)
    
    # Create 4-panel plot
    fig, axes = plt.subplots(4, 1, figsize=(14, 10))
    fig.suptitle('Segmentation Analysis - Moving Variance Segmentation (MVS)', 
                 fontsize=14, fontweight='bold')
    
    # Time axis (in seconds @ 20Hz)
    time_baseline = np.arange(len(baseline_turb)) / 20.0
    time_movement = np.arange(len(movement_turb)) / 20.0
    
    # Plot 1: Baseline Turbulence
    axes[0].plot(time_baseline, baseline_turb, 'b-', alpha=0.7, linewidth=0.8, 
                 label='Spatial Turbulence')
    axes[0].set_ylabel('Turbulence (std)', fontsize=10)
    axes[0].set_title('Baseline - Spatial Turbulence Signal', fontsize=11, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper right')
    
    # Plot 2: Baseline Moving Variance
    axes[1].plot(time_baseline, baseline_moving_var, 'g-', alpha=0.7, linewidth=0.8, 
                 label='Moving Variance')
    axes[1].axhline(y=threshold, color='r', linestyle='--', linewidth=2, 
                    label=f'Threshold = {threshold:.2f}')
    axes[1].axhline(y=mean_var, color='orange', linestyle=':', linewidth=1.5, 
                    label=f'Mean = {mean_var:.2f}')
    axes[1].axhline(y=mean_var + std_var, color='purple', linestyle=':', linewidth=1, 
                    alpha=0.5, label='Mean + 1sigma')
    axes[1].axhline(y=mean_var - std_var, color='purple', linestyle=':', linewidth=1, alpha=0.5)
    
    # Highlight detected segments (if any)
    for seg in baseline_segments:
        start = seg['start']
        length = seg['length']
        axes[1].axvspan(start/20.0, (start+length)/20.0, alpha=0.3, color='red')
    
    axes[1].set_ylabel('Variance', fontsize=10)
    axes[1].set_title(f'Baseline - Moving Variance (Window={window_size}, K=2.5)', 
                      fontsize=11, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper right', fontsize=9)
    
    # Plot 3: Movement Turbulence
    axes[2].plot(time_movement, movement_turb, 'b-', alpha=0.7, linewidth=0.8, 
                 label='Spatial Turbulence')
    axes[2].set_ylabel('Turbulence (std)', fontsize=10)
    axes[2].set_title('Movement - Spatial Turbulence Signal', fontsize=11, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc='upper right')
    
    # Plot 4: Movement Moving Variance
    axes[3].plot(time_movement, movement_moving_var, 'g-', alpha=0.7, linewidth=0.8, 
                 label='Moving Variance')
    axes[3].axhline(y=threshold, color='r', linestyle='--', linewidth=2, 
                    label=f'Threshold = {threshold:.2f}')
    
    # Highlight detected segments
    for seg in movement_segments:
        start = seg['start']
        length = seg['length']
        axes[3].axvspan(start/20.0, (start+length)/20.0, alpha=0.3, color='green')
    
    axes[3].set_xlabel('Time (seconds)', fontsize=10)
    axes[3].set_ylabel('Variance', fontsize=10)
    axes[3].set_title(f'Movement - Moving Variance with Detected Segments ({len(movement_segments)} segments)', 
                      fontsize=11, fontweight='bold')
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    output_file = output_dir / 'segmentation_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Segmentation analysis saved to {output_file}")


def plot_home_assistant_summary(data: Dict, output_dir: Path):
    """Generate Home Assistant integration summary (segment-based)."""
    if not data or 'metrics' not in data:
        print("⚠️  No Home Assistant data found")
        return
    
    metrics = data['metrics']
    targets = data['targets_met']
    
    # Check if this is segment-based data (new architecture)
    is_segment_based = 'baseline_segments' in metrics and 'movement_segments' in metrics
    
    if is_segment_based:
        # New segment-based visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # Segments detected
        baseline_segs = metrics['baseline_segments']
        movement_segs = metrics['movement_segments']
        
        ax1.bar(['Baseline', 'Movement'], [baseline_segs, movement_segs], 
                color=['green' if baseline_segs == 0 else 'red', 
                       'green' if movement_segs > 0 else 'red'], alpha=0.7)
        ax1.set_ylabel('Segments Detected', fontsize=11)
        ax1.set_title(f'Segment Detection\nBaseline: {baseline_segs} | Movement: {movement_segs}', 
                      fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Detection latency
        latency = metrics.get('detection_latency_seconds', 0)
        color = 'green' if 0 < latency <= 2 else 'orange' if latency <= 5 else 'red'
        ax2.barh(['Latency'], [latency], color=color, alpha=0.7)
        ax2.axvline(x=2, color='green', linestyle='--', linewidth=2, label='2s target')
        ax2.axvline(x=5, color='red', linestyle='--', linewidth=2, label='5s limit')
        ax2.set_xlabel('Seconds', fontsize=11)
        ax2.set_title(f'Detection Latency: {latency:.2f}s', 
                      fontsize=12, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3, axis='x')
        
        # State transitions
        transitions = metrics['state_transitions']
        color = 'green' if 2 <= transitions <= 10 else 'orange'
        ax3.bar(['Transitions'], [transitions], color=color, alpha=0.7)
        ax3.axhline(y=2, color='green', linestyle='--', linewidth=1, alpha=0.5)
        ax3.axhline(y=10, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax3.set_ylabel('Count', fontsize=11)
        trans_status = '[STABLE]' if targets.get("stable_transitions", False) else '[UNSTABLE]'
        ax3.set_title(f'State Transitions: {transitions} {trans_status}', 
                      fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Summary status
        ax4.axis('off')
        status_text = "READY" if targets.get("motion_detected") and targets.get("no_false_alarms") else "NEEDS REVIEW"
        ax4.text(0.5, 0.7, status_text, fontsize=32, ha='center', va='center',
                 color='green' if "READY" in status_text else 'orange', fontweight='bold')
        
        checks = []
        checks.append(f"Motion: {movement_segs} segments" if movement_segs > 0 else "No motion")
        checks.append(f"No FP: {baseline_segs} segments" if baseline_segs == 0 else f"FP: {baseline_segs}")
        checks.append(f"Stable: {transitions}" if 2 <= transitions <= 10 else f"Unstable: {transitions}")
        checks.append(f"Fast: {latency:.1f}s" if 0 < latency <= 5 else f"Slow: {latency:.1f}s")
        
        ax4.text(0.5, 0.4, '\n'.join(checks), fontsize=11, ha='center', va='center',
                 family='monospace')
    
    else:
        # Legacy packet-based visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # Recall gauge
        recall = metrics['recall'] * 100
        ax1.barh(['Recall'], [recall], color='green' if recall >= 90 else 'orange', alpha=0.7)
        ax1.axvline(x=90, color='red', linestyle='--', linewidth=2)
        ax1.set_xlim([0, 100])
        ax1.set_xlabel('Percentage (%)', fontsize=11)
        status_text = '[TARGET MET]' if targets.get("recall_90", False) else '[BELOW TARGET]'
        ax1.set_title(f'Recall: {recall:.1f}% {status_text}', 
                      fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # FP per hour
        fp_hour = metrics.get('fp_per_hour', 0)
        color = 'green' if 1 <= fp_hour <= 5 else 'orange' if fp_hour < 1 else 'red'
        ax2.bar(['FP/hour'], [fp_hour], color=color, alpha=0.7)
        ax2.axhline(y=1, color='green', linestyle='--', linewidth=1, alpha=0.5)
        ax2.axhline(y=5, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax2.set_ylabel('False Positives per Hour', fontsize=11)
        fp_status = '[TARGET MET]' if targets.get("fp_1_5_per_hour", False) else '[OUT OF RANGE]'
        ax2.set_title(f'FP Rate: {fp_hour:.1f}/hour {fp_status}', 
                      fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Metrics comparison
        metric_names = ['Recall', 'Precision', 'F1-Score']
        metric_values = [metrics['recall'] * 100, metrics.get('precision', 0) * 100, metrics.get('f1_score', 0) * 100]
        ax3.bar(metric_names, metric_values, color=['green', 'blue', 'purple'], alpha=0.7)
        ax3.set_ylabel('Percentage (%)', fontsize=11)
        ax3.set_ylim([0, 100])
        ax3.set_title('Performance Metrics', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # State transitions
        transitions = metrics['state_transitions']
        color = 'green' if 2 <= transitions <= 6 else 'orange'
        ax4.bar(['Transitions'], [transitions], color=color, alpha=0.7)
        ax4.axhline(y=2, color='green', linestyle='--', linewidth=1, alpha=0.5)
        ax4.axhline(y=6, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax4.set_ylabel('Count', fontsize=11)
        trans_status = '[STABLE]' if targets.get("stable_transitions", False) else '[UNSTABLE]'
        ax4.set_title(f'State Transitions: {transitions} {trans_status}', 
                      fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Home Assistant Integration Assessment', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_file = output_dir / 'home_assistant_summary.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Home Assistant summary saved to {output_file}")


def generate_report(results: Dict, output_dir: Path):
    """Generate comprehensive HTML report (segmentation-focused)."""
    
    # Detect architecture type
    is_segmentation_based = (results.get('performance_suite') and 
                             results['performance_suite'].get('architecture') == 'segmentation_based')
    
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>ESPectre Performance Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .architecture-note {{ background: #e3f2fd; padding: 15px; border-left: 4px solid #2196f3; margin: 20px 0; }}
        .metric {{ display: inline-block; margin: 10px; padding: 15px; background: #ecf0f1; border-radius: 5px; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #2980b9; }}
        .metric-label {{ font-size: 14px; color: #7f8c8d; }}
        .pass {{ color: #27ae60; }}
        .warn {{ color: #f39c12; }}
        .fail {{ color: #e74c3c; }}
        img {{ max-width: 100%; height: auto; margin: 20px 0; border: 1px solid #ddd; border-radius: 5px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #3498db; color: white; }}
        tr:hover {{ background-color: #f5f5f5; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🛜 ESPectre Performance Report 👻</h1>
        <p><strong>Generated:</strong> {Path(output_dir).name}</p>
        <p><strong>Architecture:</strong> {'Segmentation-Based' if is_segmentation_based else 'Feature-Based (Legacy)'}</p>
        <p><strong>Focus:</strong> Home Assistant Security/Presence Detection</p>
"""
    
    if is_segmentation_based:
        html += """
        <div class="architecture-note">
            <strong>📌 New Architecture:</strong> CSI Packet → Segmentation (always) → 
            IF MOTION && features_enabled: Extract Features + Publish<br>
            <strong>Accuracy based on:</strong> Segmentation performance (Moving Variance Segmentation - MVS)
        </div>
"""
    
    # 1. SEGMENTATION ANALYSIS (PRIMARY - if available)
    if results['segmentation']:
        seg = results['segmentation']
        html += f"""
        <h2>🎯 1. Segmentation Analysis (PRIMARY)</h2>
        <p><strong>Algorithm:</strong> Moving Variance Segmentation (MVS)</p>
        <p><strong>Threshold:</strong> {seg['threshold']:.4f}</p>
        <p><strong>Window Size:</strong> {seg.get('window_size', 30)} packets</p>
        <p><strong>Detected Segments:</strong> {len(seg.get('movement_segments', []))}</p>
        <img src="segmentation_analysis.png" alt="Segmentation Analysis">
"""
    
    # 2. SEGMENTATION PERFORMANCE METRICS (if segmentation-based)
    if is_segmentation_based and results['performance_suite']:
        perf = results['performance_suite']
        seg_metrics = perf.get('segmentation', {})
        if seg_metrics:
            segments_detected = seg_metrics.get('segments_detected', 0)
            # Determine if segmentation is working well based on segments, not packet-level recall
            segments_ok = segments_detected >= 10
            fp_rate_ok = seg_metrics['fp_rate'] <= 0.05
            
            html += f"""
        <h2>📊 2. Segmentation Performance Metrics</h2>
        <div style="padding: 15px; background: {'#d4edda' if segments_ok and fp_rate_ok else '#fff3cd'}; border-radius: 5px; margin: 20px 0;">
            <h3 style="margin-top: 0;">{'✅ Segmentation Working Correctly' if segments_ok and fp_rate_ok else '⚠️ Needs Review'}</h3>
            <p><strong>Key Metric:</strong> Segments Detected = <span style="font-size: 24px; color: {'#27ae60' if segments_ok else '#f39c12'};">{segments_detected}</span> 
               {'✅ (target: >10)' if segments_ok else '⚠️ (target: >10)'}</p>
            <p><em>Note: Packet-level recall of {seg_metrics['recall']*100:.1f}% is NORMAL. 
               Segmentation detects motion bursts ({segments_detected} segments), not every packet.</em></p>
        </div>
        <div class="metric">
            <div class="metric-label">Segments Detected</div>
            <div class="metric-value {'pass' if segments_ok else 'warn'}">{segments_detected}</div>
        </div>
        <div class="metric">
            <div class="metric-label">FP Rate</div>
            <div class="metric-value {'pass' if seg_metrics['fp_rate'] <= 0.05 else 'warn'}">{seg_metrics['fp_rate']*100:.1f}%</div>
        </div>
        <div class="metric">
            <div class="metric-label">Precision</div>
            <div class="metric-value">{seg_metrics['precision']*100:.1f}%</div>
        </div>
        <div class="metric">
            <div class="metric-label">Packet Recall</div>
            <div class="metric-value">{seg_metrics['recall']*100:.1f}%</div>
        </div>
        <img src="confusion_matrix.png" alt="Confusion Matrix - Segmentation">
        <p><em>Confusion Matrix shows packet-level classification. 
           The {segments_detected} segments detected indicate successful motion burst detection.</em></p>
"""
   
    # 3. FEATURE RANKING (SECONDARY - for features_enabled mode)
    if results['threshold_optimization']:
        opt = results['threshold_optimization']
        html += f"""
        <h2>📈 4. Feature Ranking (SECONDARY - for features_enabled mode)</h2>
        <p><em>Note: Features extracted only when segmentation detects motion</em></p>
        <p><strong>Best Feature:</strong> {opt['feature']}</p>
        <p><strong>AUC:</strong> {opt['auc']:.3f}</p>
        <p><strong>Optimal Threshold:</strong> {opt['optimal_threshold']:.4f}</p>
        <img src="roc_curve.png" alt="ROC Curve">
        <img src="precision_recall_curve.png" alt="Precision-Recall Curve">
"""
    
    # Legacy: Feature Performance (if not segmentation-based)
    if not is_segmentation_based and results['performance_suite']:
        perf = results['performance_suite']
        if 'best_feature' in perf:
            html += f"""
        <h2>Feature Performance (Legacy)</h2>
        <p><strong>Best Feature:</strong> {perf['best_feature']['name']}</p>
        <img src="confusion_matrix.png" alt="Confusion Matrix">
"""
        
    html += """
    </div>
</body>
</html>
"""
    
    report_file = output_dir / 'report.html'
    with open(report_file, 'w') as f:
        f.write(html)
    
    print(f"✅ HTML report saved to {report_file}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_test_results.py <test_output.log>")
        sys.exit(1)
    
    log_file = Path(sys.argv[1])
    if not log_file.exists():
        print(f"❌ Error: File not found: {log_file}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path('test_results')
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n📊 Analyzing test results from: {log_file}\n")
    
    # Parse results
    results = parse_test_output(log_file)
    
    # Generate plots
    print("Generating visualizations...")
    plot_roc_curve(results['threshold_optimization'], output_dir)
    plot_precision_recall_curve(results['threshold_optimization'], output_dir)
    plot_confusion_matrix(results['performance_suite'], output_dir)
    # Note: temporal_robustness test has been removed in new architecture
    # plot_temporal_scenarios(results['temporal_robustness'], output_dir)
    plot_segmentation_analysis(results['segmentation'], output_dir)
    
    # Generate report
    print("\nGenerating HTML report...")
    generate_report(results, output_dir)
    
    print(f"\n✅ Analysis complete! Results saved to: {output_dir}/")
    print(f"📄 View the complete report in your browser:\n")
    print(f"   open {output_dir}/report.html\n")


if __name__ == '__main__':
    main()
