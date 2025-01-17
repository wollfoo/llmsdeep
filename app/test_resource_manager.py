# test_resource_manager.py

import asyncio
from mining_environment.scripts.resource_manager import ResourceManager
from mining_environment.scripts.system_manager import load_config
from mining_environment.scripts.system_manager import ConfigModel
from mining_environment.scripts.auxiliary_modules.event_bus import EventBus
import logging
import json

async def main():
    # Setup logger
    logger = logging.getLogger('test_resource_manager')
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Load config
    config_path = "/app/mining_environment/config/resource_config.json"
    config = await load_config(config_path)

    # Initialize EventBus
    event_bus = EventBus()

    # Initialize ResourceManager
    try:
        resource_manager = ResourceManager(config, event_bus, logger)
        await resource_manager.start()
        logger.info("ResourceManager khởi động thành công.")
    except Exception as e:
        logger.error(f"Failed to initialize ResourceManager: {e}")

if __name__ == "__main__":
    asyncio.run(main())
