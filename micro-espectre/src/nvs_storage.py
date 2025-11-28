"""
Micro-ESPectre - Configuration Persistence
Simulates NVS storage using JSON file

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
            print(f"💾 Loaded configuration from {self.CONFIG_FILE}")
            return self.config
        except OSError:
            # File doesn't exist
            print(f"💾 No saved configuration found")
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
            print(f"💾 Configuration saved to {self.CONFIG_FILE}")
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
            print(f"💾 Configuration file {self.CONFIG_FILE} erased")
            return True
        except OSError:
            # File doesn't exist - that's ok
            return True
        except Exception as e:
            print(f"Error erasing configuration: {e}")
            return False
    
    def save_segmentation_config(self, seg):
        """
        Save segmentation configuration
        
        Args:
            seg: SegmentationContext instance
        """
        config_data = {
            "segmentation": {
                "threshold": seg.threshold,
                "window_size": seg.window_size
            }
        }
        
        # Merge with existing config if any
        existing = self.load()
        if existing:
            existing.update(config_data)
            config_data = existing
        
        return self.save(config_data)
    
    def save_full_config(self, seg, config_module, traffic_gen=None):
        """
        Save complete configuration
        
        Args:
            seg: SegmentationContext instance
            config_module: Configuration module
            traffic_gen: TrafficGenerator instance (optional)
        """
        config_data = {
            "segmentation": {
                "threshold": seg.threshold,
                "window_size": seg.window_size
            },
            "subcarriers": {
                "indices": config_module.SELECTED_SUBCARRIERS
            },
            "options": {
                "smart_publishing": config_module.SMART_PUBLISHING
            }
        }
        
        # Add traffic generator rate if available
        if traffic_gen:
            config_data["traffic_generator"] = {
                "rate": traffic_gen.get_rate()
            }
        
        return self.save(config_data)
    
    def load_and_apply(self, seg, config_module):
        """
        Load configuration and apply to segmentation and config
        
        Args:
            seg: SegmentationContext instance
            config_module: Configuration module
            
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
            
            # Reset buffer if window size changed
            seg.turbulence_buffer = [0.0] * seg.window_size
            seg.buffer_index = 0
            seg.buffer_count = 0
            
            print(f"📍 Segmentation config loaded: threshold={seg.threshold:.2f}, window={seg.window_size}")
        
        # Apply subcarrier selection
        if "subcarriers" in config_data:
            config_module.SELECTED_SUBCARRIERS = config_data["subcarriers"]["indices"]
            print(f"📡 Subcarrier selection loaded: {config_module.SELECTED_SUBCARRIERS}")
        
        # Apply options
        if "options" in config_data:
            config_module.SMART_PUBLISHING = config_data["options"].get("smart_publishing", config_module.SMART_PUBLISHING)
            print(f"📡 Smart publishing: {config_module.SMART_PUBLISHING}")
        
        return config_data
