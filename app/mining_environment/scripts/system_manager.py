# system_manager.py

import os
import sys
import json
from pathlib import Path
from time import sleep
from typing import Dict, Any
from resource_manager import ResourceManager
from anomaly_detector import AnomalyDetector  # Assumes anomaly_detector.py defines AnomalyDetector
from logging_config import setup_logging

# Define configuration directories
CONFIG_DIR = Path(os.getenv('CONFIG_DIR', '/app/mining_environment/config'))
MODELS_DIR = Path(os.getenv('MODELS_DIR', '/app/mining_environment/models'))
LOGS_DIR = Path(os.getenv('LOGS_DIR', '/app/mining_environment/logs'))

# Paths to AI models
RESOURCE_OPTIMIZATION_MODEL_PATH = MODELS_DIR / "resource_optimization_model.pt"
ANOMALY_CLOAKING_MODEL_PATH = MODELS_DIR / "anomaly_cloaking_model.pt"

# Setup loggers
system_logger = setup_logging('system_manager', LOGS_DIR / 'system_manager.log', 'INFO')
resource_logger = setup_logging('resource_manager', LOGS_DIR / 'resource_manager.log', 'INFO')


class SystemManager:
    """
    Class to combine ResourceManager and AnomalyDetector, ensuring both operate synchronously
    without conflicts when accessing shared resources.
    """
    def __init__(self, config: Dict[str, Any], system_logger: logging.Logger, resource_logger: logging.Logger):
        self.config = config
        self.system_logger = system_logger
        self.resource_manager = ResourceManager(config, RESOURCE_OPTIMIZATION_MODEL_PATH, resource_logger)
        self.anomaly_detector = AnomalyDetector(config, ANOMALY_CLOAKING_MODEL_PATH, resource_logger)
        
        # Gán ResourceManager cho AnomalyDetector
        self.anomaly_detector.set_resource_manager(self.resource_manager)

        self.system_logger.info("SystemManager initialized successfully.")

    def start(self):
        """
        Start system components.
        """
        self.system_logger.info("Starting SystemManager...")
        try:
            self.resource_manager.start()
            self.anomaly_detector.start()
            self.system_logger.info("SystemManager started successfully.")
        except Exception as e:
            self.system_logger.error(f"Error starting SystemManager: {e}")
            self.stop()  # Ensure entire system is stopped if error occurs
            raise

    def stop(self):
        """
        Stop system components.
        """
        self.system_logger.info("Stopping SystemManager...")
        try:
            self.resource_manager.stop()
            self.anomaly_detector.stop()
            self.system_logger.info("SystemManager stopped successfully.")
        except Exception as e:
            self.system_logger.error(f"Error stopping SystemManager: {e}")
            raise


def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load configuration from a JSON file.

    Args:
        config_path (Path): Path to the configuration file.

    Returns:
        Dict[str, Any]: Loaded configuration.
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        system_logger.info(f"Loaded configuration from {config_path}")
        return config
    except FileNotFoundError:
        system_logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        system_logger.error(f"JSON syntax error in configuration file {config_path}: {e}")
        sys.exit(1)


def start():
    """
    Start the entire system.
    """
    # Load configurations
    resource_config_path = CONFIG_DIR / "resource_config.json"
    process_config_path = CONFIG_DIR / "process_config.json"

    resource_config = load_config(resource_config_path)
    process_config = load_config(process_config_path)

    # Merge configurations
    config = {**resource_config, **process_config}

    # Check existence of AI models
    if not RESOURCE_OPTIMIZATION_MODEL_PATH.exists():
        system_logger.error(f"AI model not found at: {RESOURCE_OPTIMIZATION_MODEL_PATH}")
        sys.exit(1)
    if not ANOMALY_CLOAKING_MODEL_PATH.exists():
        system_logger.error(f"AI model not found at: {ANOMALY_CLOAKING_MODEL_PATH}")
        sys.exit(1)

    # Initialize SystemManager with configuration and loggers
    system_manager = SystemManager(config, system_logger, resource_logger)

    # Start SystemManager
    try:
        system_manager.start()

        # Log system running status
        system_logger.info("SystemManager is running. Press Ctrl+C to stop.")

        # Keep the system running continuously
        while True:
            sleep(1)
    except KeyboardInterrupt:
        system_logger.info("Received stop signal from user. Stopping SystemManager...")
        system_manager.stop()
    except Exception as e:
        system_logger.error(f"Unexpected error in SystemManager: {e}")
        system_manager.stop()
        sys.exit(1)


if __name__ == "__main__":
    # Ensure the script is run with root privileges
    if os.geteuid() != 0:
        print("Script must be run as root.")
        sys.exit(1)

    start()
