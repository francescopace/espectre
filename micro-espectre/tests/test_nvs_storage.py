"""
Micro-ESPectre - NVS Storage Unit Tests

Tests for NVSStorage class in src/nvs_storage.py.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""

import pytest
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import tempfile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from nvs_storage import NVSStorage


@pytest.fixture
def temp_config_file():
    """Create a temporary config file for testing"""
    # Create a temp directory
    temp_dir = tempfile.mkdtemp()
    original_file = NVSStorage.CONFIG_FILE
    NVSStorage.CONFIG_FILE = os.path.join(temp_dir, "test_config.json")
    
    yield NVSStorage.CONFIG_FILE
    
    # Cleanup
    try:
        os.remove(NVSStorage.CONFIG_FILE)
    except:
        pass
    try:
        os.rmdir(temp_dir)
    except:
        pass
    NVSStorage.CONFIG_FILE = original_file


@pytest.fixture
def nvs(temp_config_file):
    """Create NVSStorage instance with temp file"""
    return NVSStorage()


class TestNVSStorageInit:
    """Test NVSStorage initialization"""
    
    def test_init(self, nvs):
        """Test NVS initialization"""
        assert nvs.config == {}


class TestNVSStorageLoad:
    """Test NVSStorage load functionality"""
    
    def test_load_existing_config(self, nvs, temp_config_file):
        """Test loading existing configuration"""
        # Create config file
        config = {"key": "value", "number": 42}
        with open(temp_config_file, 'w') as f:
            json.dump(config, f)
        
        result = nvs.load()
        
        assert result is not None
        assert result['key'] == 'value'
        assert result['number'] == 42
    
    def test_load_missing_file(self, nvs):
        """Test loading when file doesn't exist"""
        result = nvs.load()
        
        assert result is None
    
    def test_load_invalid_json(self, nvs, temp_config_file):
        """Test loading invalid JSON file"""
        with open(temp_config_file, 'w') as f:
            f.write("invalid json {{{")
        
        result = nvs.load()
        
        assert result is None


class TestNVSStorageSave:
    """Test NVSStorage save functionality"""
    
    def test_save_config(self, nvs, temp_config_file):
        """Test saving configuration"""
        config = {"threshold": 1.5, "window": 100}
        
        result = nvs.save(config)
        
        assert result is True
        
        # Verify file contents
        with open(temp_config_file, 'r') as f:
            saved = json.load(f)
        
        assert saved['threshold'] == 1.5
        assert saved['window'] == 100
    
    def test_save_error_handling(self, nvs):
        """Test error handling when save fails"""
        # Set invalid path
        original = NVSStorage.CONFIG_FILE
        NVSStorage.CONFIG_FILE = "/nonexistent/path/config.json"
        
        result = nvs.save({"key": "value"})
        
        assert result is False
        
        NVSStorage.CONFIG_FILE = original


class TestNVSStorageExists:
    """Test NVSStorage exists functionality"""
    
    def test_exists_true(self, nvs, temp_config_file):
        """Test exists when file exists"""
        with open(temp_config_file, 'w') as f:
            f.write("{}")
        
        result = nvs.exists()
        
        assert result is True
    
    def test_exists_false(self, nvs, temp_config_file):
        """Test exists when file doesn't exist"""
        # Ensure file doesn't exist
        try:
            os.remove(temp_config_file)
        except:
            pass
        
        result = nvs.exists()
        
        assert result is False


class TestNVSStorageErase:
    """Test NVSStorage erase functionality"""
    
    def test_erase_existing(self, nvs, temp_config_file):
        """Test erasing existing file"""
        with open(temp_config_file, 'w') as f:
            f.write("{}")
        
        result = nvs.erase()
        
        assert result is True
        assert not os.path.exists(temp_config_file)
    
    def test_erase_nonexistent(self, nvs, temp_config_file):
        """Test erasing non-existent file"""
        try:
            os.remove(temp_config_file)
        except:
            pass
        
        result = nvs.erase()
        
        assert result is True  # Should succeed even if file doesn't exist


class TestNVSStorageSaveFullConfig:
    """Test NVSStorage save_full_config functionality"""
    
    def test_save_full_config(self, nvs, temp_config_file):
        """Test saving full configuration from segmentation"""
        # Mock segmentation object
        seg = MagicMock()
        seg.threshold = 2.0
        seg.window_size = 75
        seg.normalization_scale = 1.5
        
        result = nvs.save_full_config(seg)
        
        assert result is True
        
        # Verify saved data
        with open(temp_config_file, 'r') as f:
            saved = json.load(f)
        
        assert saved['segmentation']['threshold'] == 2.0
        assert saved['segmentation']['window_size'] == 75
        assert saved['segmentation']['normalization_scale'] == 1.5


class TestNVSStorageLoadAndApply:
    """Test NVSStorage load_and_apply functionality"""
    
    def test_load_and_apply(self, nvs, temp_config_file):
        """Test loading and applying configuration"""
        # Create config file
        config = {
            "segmentation": {
                "threshold": 2.5,
                "window_size": 100,
                "normalization_scale": 1.2
            }
        }
        with open(temp_config_file, 'w') as f:
            json.dump(config, f)
        
        # Mock segmentation object
        seg = MagicMock()
        seg.threshold = 1.0
        seg.window_size = 50
        seg.normalization_scale = 1.0
        
        result = nvs.load_and_apply(seg)
        
        assert result is not None
        assert seg.threshold == 2.5
        assert seg.window_size == 100
        assert seg.normalization_scale == 1.2
        # Buffer should be reset
        assert seg.turbulence_buffer == [0.0] * 100
        assert seg.buffer_index == 0
        assert seg.buffer_count == 0
    
    def test_load_and_apply_missing_file(self, nvs, temp_config_file):
        """Test load_and_apply when file doesn't exist"""
        seg = MagicMock()
        seg.threshold = 1.0
        
        result = nvs.load_and_apply(seg)
        
        assert result is None
        # Segmentation should not be modified
        assert seg.threshold == 1.0
    
    def test_load_and_apply_partial_config(self, nvs, temp_config_file):
        """Test load_and_apply with partial configuration"""
        # Create config with only threshold
        config = {
            "segmentation": {
                "threshold": 3.0
            }
        }
        with open(temp_config_file, 'w') as f:
            json.dump(config, f)
        
        seg = MagicMock()
        seg.threshold = 1.0
        seg.window_size = 50
        seg.normalization_scale = 1.0
        
        result = nvs.load_and_apply(seg)
        
        assert result is not None
        assert seg.threshold == 3.0
        # Other values should use defaults from seg
    
    def test_load_and_apply_no_segmentation_key(self, nvs, temp_config_file):
        """Test load_and_apply with config missing segmentation key"""
        config = {"other_key": "value"}
        with open(temp_config_file, 'w') as f:
            json.dump(config, f)
        
        seg = MagicMock()
        seg.threshold = 1.0
        
        result = nvs.load_and_apply(seg)
        
        assert result is not None
        # Segmentation should not be modified
        assert seg.threshold == 1.0

