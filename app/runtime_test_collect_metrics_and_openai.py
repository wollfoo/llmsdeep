import os
import json
import logging
from mining_environment.scripts.resource_manager import ResourceManager
from mining_environment.scripts.azure_clients import AzureOpenAIClient

def main():
    logger = logging.getLogger("runtime_test")
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logger.info("=== BẮT ĐẦU KIỂM THỬ RUNTIME: ResourceManager & AzureOpenAIClient ===")

    # 1) Đọc file test_config.json
    try:
        with open("/app/mining_environment/config/test_config.json", "r") as f:
            resource_manager_config = json.load(f)
        logger.info("Đã load config từ test_config.json")
    except Exception as e:
        logger.error(f"Không thể đọc test_config.json: {e}")
        return

    # 2) Tạo ResourceManager
    try:
        resource_manager = ResourceManager(config=resource_manager_config, logger=logger)
        logger.info("ResourceManager đã khởi tạo thành công.")
    except KeyError as ke:
        logger.error(f"Thiếu key trong config: {ke}")
        return
    except Exception as e:
        logger.error(f"Lỗi khởi tạo ResourceManager: {e}")
        return

    # 3) Tìm tiến trình (theo chuỗi 'python', 'nvidia-smi', etc.)
    resource_manager.discover_mining_processes()
    num_found = len(resource_manager.mining_processes)
    logger.info(f"Tìm được {num_found} tiến trình khai thác.")

    # 4) Thu thập metrics
    all_metrics = resource_manager.collect_all_metrics()
    logger.info(f"Dữ liệu metrics thu thập:\n{all_metrics}")

    # 5) Khởi tạo AzureOpenAIClient (nếu có API key)
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        logger.warning("Chưa thiết lập biến môi trường OPENAI_API_KEY => Không thể gọi AzureOpenAIClient.")
        return
    try:
        openai_client = AzureOpenAIClient(logger, resource_manager_config)
        logger.info("AzureOpenAIClient đã khởi tạo thành công.")
    except Exception as e:
        logger.error(f"Lỗi khởi tạo AzureOpenAIClient: {e}")
        return

    # 6) Gọi get_optimization_suggestions
    suggestions = openai_client.get_optimization_suggestions(all_metrics)
    logger.info("=== Kết quả gợi ý tối ưu từ AzureOpenAIClient ===")
    logger.info(f"Số phần tử gợi ý: {len(suggestions)}")
    logger.info(f"Danh sách gợi ý: {suggestions}")

    logger.info("=== KẾT THÚC KIỂM THỬ RUNTIME ===")


if __name__ == "__main__":
    main()
