import logging
from mining_environment.scripts.cloak_strategies import CloakStrategyFactory  # Bổ sung import

def test_apply_cloak_strategy():
    config = {'throttle_percentage': 20}
    logger = logging.getLogger('test')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())

    # Mock ResourceManager
    class MockResourceManager:
        def __init__(self, config, logger):
            self.config = config
            self.logger = logger

        def is_gpu_initialized(self):
            return True

        def execute_adjustments(self, adjustments, process):
            logger.info(f"Executing adjustments: {adjustments} for process {process.name} (PID: {process.pid})")

        def apply_cloak_strategy(self, strategy_name, process):
            try:
                # Log việc tạo strategy
                self.logger.debug(f"Tạo strategy '{strategy_name}' cho {process.name} (PID={process.pid})")

                # Tạo chiến lược từ factory
                strategy = CloakStrategyFactory.create_strategy(
                    strategy_name,
                    self.config,
                    self.logger,
                    self.is_gpu_initialized()
                )

                # Kiểm tra chiến lược có hợp lệ không
                if not strategy:
                    self.logger.error(f"Failed to create strategy '{strategy_name}'. Strategy is None.")
                    return
                if not callable(getattr(strategy, 'apply', None)):
                    self.logger.error(f"Invalid strategy: {strategy.__class__.__name__} does not implement a callable 'apply' method.")
                    return

                # Áp dụng chiến lược
                adjustments = strategy.apply(process)
                if adjustments:
                    self.logger.info(f"Áp dụng '{strategy_name}' => {adjustments} cho {process.name} (PID={process.pid}).")
                    self.execute_adjustments(adjustments, process)
                else:
                    self.logger.warning(f"Không có điều chỉnh nào được trả về từ strategy '{strategy_name}' cho tiến trình {process.name} (PID={process.pid}).")
            except Exception as e:
                self.logger.error(
                    f"Lỗi khi áp dụng cloaking '{strategy_name}' cho {process.name} (PID={process.pid}): {e}"
                )

    resource_manager = MockResourceManager(config, logger)

    # Test with valid strategy
    process = type('FakeProcess', (), {'pid': 123, 'name': 'test-process'})
    resource_manager.apply_cloak_strategy('gpu', process)

    # Test with invalid strategy
    resource_manager.apply_cloak_strategy('invalid', process)

if __name__ == "__main__":
    test_apply_cloak_strategy()
