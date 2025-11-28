"""
NBVI (Normalized Baseline Variability Index) Calibrator

Automatic subcarrier selection using percentile-based baseline detection.
Implements NBVI Weighted α=0.3 algorithm for optimal subcarrier selection.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import math


class NBVICalibrator:
    """
    Automatic NBVI calibrator with percentile-based baseline detection
    
    Collects CSI packets at boot and automatically selects optimal subcarriers
    using NBVI Weighted α=0.3 algorithm with percentile-based detection.
    """
    
    def __init__(self, buffer_size=1000, percentile=10, alpha=0.3, min_spacing=3):
        """
        Initialize NBVI calibrator
        
        Args:
            buffer_size: Number of packets to collect (default: 1000)
            percentile: Percentile for baseline detection (default: 10)
            alpha: NBVI weighting factor (default: 0.3)
            min_spacing: Minimum spacing between subcarriers (default: 3)
        """
        self.buffer_size = buffer_size
        self.percentile = percentile
        self.alpha = alpha
        self.min_spacing = min_spacing
        self.noise_gate_percentile = 10
        
        # Buffer: Store only magnitudes (64 per packet)
        # Using list of lists: [[mag0, mag1, ..., mag63], ...]
        self.buffer = []
        
    def add_packet(self, csi_data):
        """
        Add CSI packet to calibration buffer
        
        Args:
            csi_data: CSI data array (128 bytes: 64 subcarriers × 2 I/Q)
        
        Returns:
            int: Current buffer size (progress indicator)
        """
        if len(self.buffer) >= self.buffer_size:
            return self.buffer_size
        
        # Extract magnitudes for all 64 subcarriers
        magnitudes = []
        for sc in range(64):
            i_idx = sc * 2
            q_idx = sc * 2 + 1
            
            if q_idx < len(csi_data):
                I = csi_data[i_idx]
                Q = csi_data[q_idx]
                # Calculate magnitude and store as int16 to save memory
                mag = int(math.sqrt(I*I + Q*Q))
                magnitudes.append(mag)
            else:
                magnitudes.append(0)
        
        self.buffer.append(magnitudes)
        return len(self.buffer)
    
    def _calculate_variance_two_pass(self, values):
        """Two-pass variance algorithm (numerically stable)"""
        if not values:
            return 0.0
        
        n = len(values)
        if n < 2:
            return 0.0
        
        # First pass: calculate mean
        mean = sum(values) / n
        
        # Second pass: calculate variance
        sum_sq_diff = sum((x - mean) ** 2 for x in values)
        variance = sum_sq_diff / n
        
        return variance
    
    def _find_baseline_window_percentile(self, current_band, window_size=100, step=50):
        """
        Find baseline window using percentile-based detection
        
        NO absolute threshold - adapts automatically to environment
        
        Args:
            current_band: Current subcarrier band (for variance calculation)
            window_size: Size of analysis window (default: 100 packets)
            step: Step size for sliding window (default: 50 packets)
        
        Returns:
            list: Best baseline window, or None if not found
        """
        if len(self.buffer) < window_size:
            return None
        
        # Analyze sliding windows
        window_variances = []
        
        for i in range(0, len(self.buffer) - window_size, step):
            window = self.buffer[i:i+window_size]
            
            # Calculate spatial turbulence for this window
            turbulences = []
            for packet_mags in window:
                # Extract magnitudes for current band
                band_mags = [packet_mags[sc] for sc in current_band if sc < len(packet_mags)]
                if band_mags:
                    # Spatial turbulence = std of subcarrier magnitudes
                    mean_mag = sum(band_mags) / len(band_mags)
                    variance = sum((m - mean_mag) ** 2 for m in band_mags) / len(band_mags)
                    std = math.sqrt(variance) if variance > 0 else 0.0
                    turbulences.append(std)
            
            # Calculate variance of turbulence (moving variance)
            if turbulences:
                turb_variance = self._calculate_variance_two_pass(turbulences)
                window_variances.append({
                    'start': i,
                    'variance': turb_variance,
                    'window': window
                })
        
        if not window_variances:
            return None
        
        # Calculate percentile threshold (adaptive!)
        variances = [w['variance'] for w in window_variances]
        p_threshold = self._percentile(variances, self.percentile)
        
        # Find windows below percentile
        baseline_candidates = [w for w in window_variances if w['variance'] <= p_threshold]
        
        if not baseline_candidates:
            return None
        
        # Use window with minimum variance
        best_window = min(baseline_candidates, key=lambda x: x['variance'])
        
        print(f"NBVI: Baseline window found")
        print(f"  Variance: {best_window['variance']:.4f}")
        print(f"  p{self.percentile} threshold: {p_threshold:.4f} (adaptive)")
        print(f"  Windows analyzed: {len(window_variances)}")
        print(f"  Baseline candidates: {len(baseline_candidates)}")
        
        return best_window['window']
    
    def _percentile(self, values, p):
        """Calculate percentile (simple implementation)"""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        k = (n - 1) * p / 100.0
        f = int(k)
        c = f + 1
        
        if c >= n:
            return sorted_values[-1]
        
        # Linear interpolation
        d0 = sorted_values[f] * (c - k)
        d1 = sorted_values[c] * (k - f)
        return d0 + d1
    
    def _calculate_nbvi_weighted(self, magnitudes):
        """
        Calculate NBVI Weighted α=0.3
        
        NBVI = 0.3 × (σ/μ²) + 0.7 × (σ/μ)
        
        Args:
            magnitudes: List of magnitude values for a subcarrier
        
        Returns:
            dict: {'nbvi': value, 'mean': μ, 'std': σ}
        """
        if not magnitudes:
            return {'nbvi': float('inf'), 'mean': 0.0, 'std': 0.0}
        
        mean = sum(magnitudes) / len(magnitudes)
        
        if mean < 1e-6:
            return {'nbvi': float('inf'), 'mean': mean, 'std': 0.0}
        
        # Calculate std
        variance = sum((m - mean) ** 2 for m in magnitudes) / len(magnitudes)
        std = math.sqrt(variance) if variance > 0 else 0.0
        
        # NBVI Weighted α=0.3
        cv = std / mean
        nbvi_energy = std / (mean * mean)
        nbvi_weighted = self.alpha * nbvi_energy + (1 - self.alpha) * cv
        
        return {
            'nbvi': nbvi_weighted,
            'mean': mean,
            'std': std
        }
    
    def _apply_noise_gate(self, subcarrier_metrics):
        """
        Apply Noise Gate: exclude weak subcarriers
        
        Args:
            subcarrier_metrics: List of dicts with 'subcarrier' and 'mean' keys
        
        Returns:
            list: Filtered metrics (strong subcarriers only)
        """
        means = [m['mean'] for m in subcarrier_metrics]
        threshold = self._percentile(means, self.noise_gate_percentile)
        
        filtered = [m for m in subcarrier_metrics if m['mean'] >= threshold]
        
        print(f"NBVI: Noise Gate excluded {len(subcarrier_metrics) - len(filtered)} weak subcarriers")
        print(f"  Threshold: {threshold:.2f} (p{self.noise_gate_percentile})")
        
        return filtered
    
    def _select_with_spacing(self, sorted_metrics, k=12):
        """
        Select subcarriers with spectral de-correlation
        
        Strategy:
        - Top 5: Always include (highest priority)
        - Remaining 7: Select with minimum spacing Δf≥min_spacing
        
        Args:
            sorted_metrics: Subcarriers sorted by NBVI (ascending)
            k: Number to select (default: 12)
        
        Returns:
            list: Selected subcarrier indices
        """
        # Phase 1: Top 5 absolute best
        top_5 = [m['subcarrier'] for m in sorted_metrics[:5]]
        selected = top_5[:]
        
        # Phase 2: Remaining 7 with spacing
        for candidate in sorted_metrics[5:]:
            if len(selected) >= k:
                break
            
            sc = candidate['subcarrier']
            
            # Check spacing with already selected
            min_dist = min(abs(sc - s) for s in selected)
            
            if min_dist >= self.min_spacing:
                selected.append(sc)
        
        # If not enough, add best remaining regardless of spacing
        if len(selected) < k:
            for candidate in sorted_metrics:
                if len(selected) >= k:
                    break
                sc = candidate['subcarrier']
                if sc not in selected:
                    selected.append(sc)
        
        selected.sort()
        
        print(f"NBVI: Selected {len(selected)} subcarriers")
        print(f"  Top 5: {top_5}")
        print(f"  With spacing Δf≥{self.min_spacing}: {selected[5:]}")
        
        return selected
    
    def calibrate(self, current_band, window_size=None, step=None):
        """
        Calibrate using NBVI Weighted α=0.3 with percentile-based detection
        
        Args:
            current_band: Current subcarrier band (for baseline detection)
            window_size: Window size for baseline detection (default: from config)
            step: Step size for sliding window (default: from config)
        
        Returns:
            list: Selected subcarrier band (12 subcarriers), or None if failed
        """
        # Import config here to get current values
        import src.config as config
        
        # Use config values if not provided
        if window_size is None:
            window_size = config.NBVI_WINDOW_SIZE
        if step is None:
            step = config.NBVI_WINDOW_STEP
        
        print("\nNBVI: Starting calibration...")
        print(f"  Window size: {window_size} packets")
        print(f"  Step size: {step} packets")
        
        # Step 1: Find baseline window using percentile
        baseline_window = self._find_baseline_window_percentile(current_band, window_size, step)
        
        if baseline_window is None:
            print("NBVI: Failed to find baseline window")
            return None
        
        print(f"NBVI: Using {len(baseline_window)} packets for calibration")
        
        # Step 2: Calculate NBVI for all 64 subcarriers
        all_metrics = []
        
        for sc in range(64):
            # Extract magnitude series for this subcarrier
            magnitudes = [packet_mags[sc] for packet_mags in baseline_window]
            
            # Calculate NBVI
            metrics = self._calculate_nbvi_weighted(magnitudes)
            metrics['subcarrier'] = sc
            all_metrics.append(metrics)
        
        # Step 3: Apply Noise Gate
        filtered_metrics = self._apply_noise_gate(all_metrics)
        
        if len(filtered_metrics) < 12:
            print(f"NBVI: Not enough subcarriers after Noise Gate ({len(filtered_metrics)} < 12)")
            return None
        
        # Step 4: Sort by NBVI (ascending - lower is better)
        sorted_metrics = sorted(filtered_metrics, key=lambda x: x['nbvi'])
        
        # Step 5: Select with spectral spacing
        selected_band = self._select_with_spacing(sorted_metrics, k=12)
        
        if len(selected_band) != 12:
            print(f"NBVI: Invalid band size ({len(selected_band)} != 12)")
            return None
        
        # Calculate average metrics for selected band
        selected_metrics = [m for m in all_metrics if m['subcarrier'] in selected_band]
        avg_nbvi = sum(m['nbvi'] for m in selected_metrics) / len(selected_metrics)
        avg_mean = sum(m['mean'] for m in selected_metrics) / len(selected_metrics)
        
        print(f"NBVI: Calibration successful!")
        print(f"  Band: {selected_band}")
        print(f"  Average NBVI: {avg_nbvi:.6f}")
        print(f"  Average magnitude: {avg_mean:.2f}")
        
        return selected_band
