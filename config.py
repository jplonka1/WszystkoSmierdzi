import json
import os
from typing import Optional


class SimpleConfig:
    """Simple config object with dot notation access."""
    
    def __init__(self, config_dict: dict):
        # Convert nested dicts to SimpleConfig objects for dot notation
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, SimpleConfig(value))
            else:
                setattr(self, key, value)

def load_config(config_path: Optional[str] = None) -> SimpleConfig:
    """Load configuration from JSON file or use default."""
    
    # Use default config file if not specified
    if config_path is None:
        config_path = "config_default.json"
    
    # Load JSON config
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return SimpleConfig(config_dict)
    else:
        raise FileNotFoundError(f"Config file not found: {config_path}")


# For backward compatibility with existing code
ExperimentConfig = SimpleConfig
