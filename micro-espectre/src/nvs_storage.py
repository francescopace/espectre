"""
Micro-ESPectre - Configuration Persistence

Uses ESP32 NVS (Non-Volatile Storage) for configuration persistence.
Manages saving and loading of system configuration across reboots.

Author: Francesco Pace <francesco.pace@gmail.com>
License: GPLv3
"""
import json


class NVSStorage:
    """Configuration persistence using JSON file"""
    
    CONFIG_FILE = "espectre_config.json"
    
    def __init__(self):
        """Initialize NVS storage"""
        self.config = {}
        
    def load(self):
        """
        Load configuration from file
        
        Returns:
            dict: Configuration data, or None if not found
        """
        try:
            with open(self.CONFIG_FILE, 'r') as f:
                self.config = json.load(f)
            print(f"Loaded configuration from {self.CONFIG_FILE}")
            return self.config
        except OSError:
            # File doesn't exist
            print(f"No saved configuration found. Using defaults")
            return None
        except Exception as e:
            print(f"Error loading configuration: {e}")
            return None
    
    def save(self, config_data):
        """
        Save configuration to file
        
        Args:
            config_data: Dictionary with configuration to save
            
        Returns:
            bool: True if saved successfully
        """
        try:
            with open(self.CONFIG_FILE, 'w') as f:
                json.dump(config_data, f)
            print(f"Configuration saved to {self.CONFIG_FILE}")
            return True
        except Exception as e:
            print(f"Error saving configuration: {e}")
            return False
    
    def exists(self):
        """
        Check if configuration file exists
        
        Returns:
            bool: True if configuration exists
        """
        try:
            with open(self.CONFIG_FILE, 'r') as f:
                pass
            return True
        except OSError:
            return False
    
    def erase(self):
        """
        Erase configuration file (factory reset)
        
        Returns:
            bool: True if erased successfully
        """
        try:
            import os
            os.remove(self.CONFIG_FILE)
            print(f"Configuration file {self.CONFIG_FILE} erased")
            return True
        except OSError:
            # File doesn't exist - that's ok
            return True
        except Exception as e:
            print(f"Error erasing configuration: {e}")
            return False
    
    def save_full_config(self, seg):
        """
        Save complete configuration
        
        Args:
            seg: SegmentationContext instance
        """
        config_data = {
            "segmentation": {
                "threshold": seg.threshold,
                "window_size": seg.window_size,
                "normalization_scale": seg.normalization_scale
            }
        }
                
        return self.save(config_data)
    
    def load_and_apply(self, seg):
        """
        Load configuration and apply to segmentation
        
        Args:
            seg: SegmentationContext instance
            
        Returns:
            dict: Loaded configuration, or None if not found
        """
        config_data = self.load()
        if not config_data:
            return None
        
        # Apply segmentation parameters
        if "segmentation" in config_data:
            seg_cfg = config_data["segmentation"]
            seg.threshold = seg_cfg.get("threshold", seg.threshold)
            seg.window_size = seg_cfg.get("window_size", seg.window_size)
            seg.normalization_scale = seg_cfg.get("normalization_scale", seg.normalization_scale)
            
            # Reset buffer if window size changed
            seg.turbulence_buffer = [0.0] * seg.window_size
            seg.buffer_index = 0
            seg.buffer_count = 0
            
            #print(f"Segmentation config loaded: threshold={seg.threshold:.2f}, window={seg.window_size}, norm_scale={seg.normalization_scale:.3f}")
        
        return config_data
