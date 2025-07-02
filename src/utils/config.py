import yaml
import os

# Configuration constants
DATA_DIR = "data"
MODELS_DIR = "models"
LOGS_DIR = "logs"
PREDICTIONS_DIR = "predictions"
LOG_LEVEL = "INFO"

class Config:
    def __init__(self, config_file):
        with open(config_file, "r") as f:
            self._config = yaml.safe_load(f)

    def get(self, key, default=None):
        return self._config.get(key, default)

    def set(self, key, value):
        self._config[key] = value

    def save(self, config_file):
        with open(config_file, "w") as f:
            yaml.safe_dump(self._config, f, default_flow_style=False)

def read_config(config_file):
    """Read configuration from YAML file"""
    with open(config_file, "r") as f:
        return yaml.safe_load(f)

def get_config_value(config, key_path, default=None):
    """Get nested configuration value using dot notation"""
    keys = key_path.split('.')
    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value
