"""
Micro-ESPectre - Configuration Module Tests

Tests for src/config.py to verify configuration constants are properly defined.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


class TestConfigConstants:
    """Test that all required configuration constants are defined"""
    
    def test_wifi_config_exists(self):
        """Test WiFi configuration constants exist"""
        import config
        
        assert hasattr(config, 'WIFI_SSID')
        assert hasattr(config, 'WIFI_PASSWORD')
        assert isinstance(config.WIFI_SSID, str)
        assert isinstance(config.WIFI_PASSWORD, str)
    
    def test_mqtt_config_exists(self):
        """Test MQTT configuration constants exist"""
        import config
        
        assert hasattr(config, 'MQTT_BROKER')
        assert hasattr(config, 'MQTT_PORT')
        assert hasattr(config, 'MQTT_CLIENT_ID')
        assert hasattr(config, 'MQTT_TOPIC')
        assert hasattr(config, 'MQTT_USERNAME')
        assert hasattr(config, 'MQTT_PASSWORD')
        
        assert isinstance(config.MQTT_PORT, int)
        assert config.MQTT_PORT > 0
    
    def test_traffic_generator_config(self):
        """Test traffic generator configuration"""
        import config
        
        assert hasattr(config, 'TRAFFIC_GENERATOR_RATE')
        assert isinstance(config.TRAFFIC_GENERATOR_RATE, int)
        assert config.TRAFFIC_GENERATOR_RATE >= 0
    
    def test_csi_config(self):
        """Test CSI configuration"""
        import config
        
        assert hasattr(config, 'CSI_BUFFER_SIZE')
        assert isinstance(config.CSI_BUFFER_SIZE, int)
        assert config.CSI_BUFFER_SIZE > 0
    
    def test_normalization_config(self):
        """Test normalization configuration"""
        import config
        
        # Normalization is always enabled, only NORMALIZATION_SCALE is configurable
        assert hasattr(config, 'NORMALIZATION_SCALE')
        
        assert isinstance(config.NORMALIZATION_SCALE, (int, float))
        assert config.NORMALIZATION_SCALE > 0
    
    def test_nbvi_config(self):
        """Test NBVI calibration configuration"""
        import config
        
        assert hasattr(config, 'NBVI_BUFFER_SIZE')
        assert hasattr(config, 'NBVI_WINDOW_SIZE')
        assert hasattr(config, 'NBVI_WINDOW_STEP')
        assert hasattr(config, 'NBVI_PERCENTILE')
        assert hasattr(config, 'NBVI_ALPHA')
        assert hasattr(config, 'NBVI_MIN_SPACING')
        assert hasattr(config, 'NBVI_NOISE_GATE_PERCENTILE')
        
        assert isinstance(config.NBVI_BUFFER_SIZE, int)
        assert isinstance(config.NBVI_WINDOW_SIZE, int)
        assert isinstance(config.NBVI_ALPHA, (int, float))
        assert 0 <= config.NBVI_ALPHA <= 1
    
    def test_segmentation_config(self):
        """Test segmentation configuration"""
        import config
        
        assert hasattr(config, 'SEG_WINDOW_SIZE')
        assert hasattr(config, 'SEG_THRESHOLD')
        
        assert isinstance(config.SEG_WINDOW_SIZE, int)
        assert isinstance(config.SEG_THRESHOLD, (int, float))
        assert config.SEG_WINDOW_SIZE > 0
        assert config.SEG_THRESHOLD >= 0
    
    def test_lowpass_filter_config(self):
        """Test low-pass filter configuration"""
        import config
        
        assert hasattr(config, 'ENABLE_LOWPASS_FILTER')
        assert hasattr(config, 'LOWPASS_CUTOFF')
        
        assert isinstance(config.ENABLE_LOWPASS_FILTER, bool)
        assert isinstance(config.LOWPASS_CUTOFF, (int, float))
        assert config.LOWPASS_CUTOFF > 0
    
    def test_hampel_filter_config(self):
        """Test Hampel filter configuration"""
        import config
        
        assert hasattr(config, 'ENABLE_HAMPEL_FILTER')
        assert hasattr(config, 'HAMPEL_WINDOW')
        assert hasattr(config, 'HAMPEL_THRESHOLD')
        
        assert isinstance(config.ENABLE_HAMPEL_FILTER, bool)
        assert isinstance(config.HAMPEL_WINDOW, int)
        assert isinstance(config.HAMPEL_THRESHOLD, (int, float))
        assert config.HAMPEL_WINDOW > 0
        assert config.HAMPEL_THRESHOLD > 0
    
    def test_features_config(self):
        """Test feature extraction configuration"""
        import config
        
        assert hasattr(config, 'ENABLE_FEATURES')
        assert isinstance(config.ENABLE_FEATURES, bool)


class TestConfigDefaultValues:
    """Test that configuration has sensible default values"""
    
    def test_default_traffic_rate(self):
        """Test default traffic generator rate is reasonable"""
        import config
        
        # Should be between 10 and 1000 Hz
        assert 10 <= config.TRAFFIC_GENERATOR_RATE <= 1000
    
    def test_default_segmentation_window(self):
        """Test default segmentation window is reasonable"""
        import config
        
        # Should be between 10 and 200
        assert 10 <= config.SEG_WINDOW_SIZE <= 200
    
    def test_default_threshold(self):
        """Test default threshold is reasonable"""
        import config
        
        # Should be between 0.1 and 10
        assert 0.1 <= config.SEG_THRESHOLD <= 10
    
    def test_default_nbvi_parameters(self):
        """Test default NBVI parameters are reasonable"""
        import config
        
        assert config.NBVI_BUFFER_SIZE >= 100
        assert config.NBVI_WINDOW_SIZE >= 10
        assert 0 < config.NBVI_ALPHA < 1
        assert 0 < config.NBVI_PERCENTILE < 100
    
    def test_mqtt_port_standard(self):
        """Test MQTT port is standard"""
        import config
        
        # Should be 1883 (standard) or 8883 (TLS)
        assert config.MQTT_PORT in [1883, 8883]

