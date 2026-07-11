"""
Micro-ESPectre - Configuration Module Tests

Tests for src/config.py to verify configuration constants are properly defined.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import pytest
import importlib.util

from tools.lib.repo_paths import python_src_dir

# Load src/config.py directly using importlib to avoid conflicts with tools/config.py
SRC_CONFIG_PATH = python_src_dir() / 'config.py'


def load_src_config():
    """Load src/config.py directly, bypassing sys.path"""
    spec = importlib.util.spec_from_file_location("src_config", SRC_CONFIG_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestConfigConstants:
    """Test that all required configuration constants are defined"""
    
    def test_wifi_config_exists(self):
        """Test WiFi configuration constants exist"""
        config = load_src_config()
        
        assert hasattr(config, 'WIFI_SSID')
        assert hasattr(config, 'WIFI_PASSWORD')
        assert isinstance(config.WIFI_SSID, str)
        assert isinstance(config.WIFI_PASSWORD, str)
    
    def test_mqtt_config_exists(self):
        """Test MQTT configuration constants exist"""
        config = load_src_config()
        
        assert hasattr(config, 'MQTT_BROKER')
        assert hasattr(config, 'MQTT_PORT')
        assert hasattr(config, 'MQTT_CLIENT_ID')
        assert hasattr(config, 'MQTT_TOPIC_PREFIX')
        assert hasattr(config, 'MQTT_USERNAME')
        assert hasattr(config, 'MQTT_PASSWORD')
        
        assert isinstance(config.MQTT_PORT, int)
        assert config.MQTT_PORT > 0
    
    def test_traffic_generator_config(self):
        """Test traffic generator configuration"""
        config = load_src_config()
        
        assert hasattr(config, 'TRAFFIC_GENERATOR_RATE')
        assert hasattr(config, 'PUBLISH_INTERVAL')
        assert hasattr(config, 'EVALUATION_INTERVAL')
        assert hasattr(config, 'MOTION_ON_HITS')
        assert hasattr(config, 'MOTION_OFF_HITS')
        assert isinstance(config.TRAFFIC_GENERATOR_RATE, int)
        assert isinstance(config.PUBLISH_INTERVAL, int)
        assert isinstance(config.EVALUATION_INTERVAL, int)
        assert isinstance(config.MOTION_ON_HITS, int)
        assert isinstance(config.MOTION_OFF_HITS, int)
        assert config.TRAFFIC_GENERATOR_RATE >= 0
    
    def test_csi_config(self):
        """Test CSI configuration"""
        config = load_src_config()
        
        assert hasattr(config, 'CSI_BUFFER_SIZE')
        assert isinstance(config.CSI_BUFFER_SIZE, int)
        assert config.CSI_BUFFER_SIZE > 0
    
    def test_calibration_config(self):
        """Test band calibration configuration"""
        config = load_src_config()
        
        assert hasattr(config, 'CALIBRATION_BUFFER_SIZE')
        
        assert isinstance(config.CALIBRATION_BUFFER_SIZE, int)
        assert config.CALIBRATION_BUFFER_SIZE >= 100
    
    def test_segmentation_config(self):
        """Test segmentation configuration"""
        config = load_src_config()
        
        assert hasattr(config, 'SEG_WINDOW_SIZE')
        
        assert isinstance(config.SEG_WINDOW_SIZE, int)
        assert config.SEG_WINDOW_SIZE > 0
        # Note: threshold is calculated adaptively (max x 1.3), not in config
    
    def test_lowpass_filter_config(self):
        """Test low-pass filter configuration"""
        config = load_src_config()
        
        assert hasattr(config, 'ENABLE_LOWPASS_FILTER')
        assert hasattr(config, 'LOWPASS_CUTOFF')
        
        assert isinstance(config.ENABLE_LOWPASS_FILTER, bool)
        assert isinstance(config.LOWPASS_CUTOFF, (int, float))
        assert config.LOWPASS_CUTOFF > 0
    
    def test_hampel_filter_config(self):
        """Test Hampel filter configuration"""
        config = load_src_config()
        
        assert hasattr(config, 'ENABLE_HAMPEL_FILTER')
        assert hasattr(config, 'HAMPEL_WINDOW')
        assert hasattr(config, 'HAMPEL_THRESHOLD')
        
        assert isinstance(config.ENABLE_HAMPEL_FILTER, bool)
        assert isinstance(config.HAMPEL_WINDOW, int)
        assert isinstance(config.HAMPEL_THRESHOLD, (int, float))
        assert config.HAMPEL_WINDOW > 0
        assert config.HAMPEL_THRESHOLD > 0
    
class TestConfigDefaultValues:
    """Test that configuration has sensible default values"""
    
    def test_default_traffic_rate(self):
        """Test default traffic generator rate is reasonable"""
        config = load_src_config()
        
        # Should be between 10 and 1000 Hz
        assert 10 <= config.TRAFFIC_GENERATOR_RATE <= 1000
        assert 1 <= config.PUBLISH_INTERVAL <= 1000
        assert 1 <= config.EVALUATION_INTERVAL <= 1000
        assert config.MOTION_ON_HITS >= 1
        assert config.MOTION_OFF_HITS >= 1
    
    def test_default_segmentation_window(self):
        """Test default segmentation window is reasonable"""
        config = load_src_config()
        
        assert config.SEG_WINDOW_SIZE > 0
    
    def test_default_calibration_parameters(self):
        """Test default calibration parameters are reasonable"""
        config = load_src_config()
        
        assert config.CALIBRATION_BUFFER_SIZE >= 100
    
    def test_mqtt_port_standard(self):
        """Test MQTT port is standard"""
        config = load_src_config()
        
        # Should be 1883 (standard) or 8883 (TLS)
        assert config.MQTT_PORT in [1883, 8883]
