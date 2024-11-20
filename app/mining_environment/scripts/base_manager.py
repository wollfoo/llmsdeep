# base_manager.py
import json
from pathlib import Path
import logging

class BaseManager:
    def __init__(self, config_path: Path, model_path: Path, logger: logging.Logger):
        self.config_path = config_path
        self.model_path = model_path
        self.logger = logger
        self.config = self.load_config()
        self.validate_config(self.config)
        self.model, self.device = self.load_model(self.model_path)

    def load_config(self):
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            self.logger.info(f"Loaded configuration from {self.config_path}")
            return config
        except FileNotFoundError:
            self.logger.error(f"Configuration file not found: {self.config_path}")
            raise
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error in {self.config_path}: {e}")
            raise

    def validate_config(self, config):
        required_keys = [
            "resource_allocation",
            "temperature_limits",
            "power_limits",
            "monitoring_parameters",
            "optimization_parameters",
            "cloak_strategies",
            "process_priority_map",
            "ai_driven_monitoring"
        ]
        for key in required_keys:
            if key not in config:
                self.logger.error(f"Missing configuration key: {key}")
                raise KeyError(f"Missing configuration key: {key}")
        # Add more detailed checks if necessary

    def load_model(self, model_path: Path):
        if not model_path.exists():
            self.logger.error(f"AI model not found at: {model_path}")
            raise FileNotFoundError(f"AI model not found at: {model_path}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            model = torch.load(model_path, map_location=device)
            model.eval()
            self.logger.info(f"Loaded AI model from {model_path}")
            return model, device
        except Exception as e:
            self.logger.error(f"Error loading AI model from {model_path}: {e}")
            raise e
