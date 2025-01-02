import logging
from mining_environment.scripts.azure_clients import AzureTrafficAnalyticsClient


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Thiết lập logger cơ bản.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    console_handler = logging.StreamHandler()
    console_format = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    console_handler.setFormatter(console_format)

    logger.addHandler(console_handler)
    return logger


def main():
    # Thiết lập logger
    logger = setup_logger("TestTrafficAnalytics")

    # Khởi tạo AzureTrafficAnalyticsClient
    try:
        azure_traffic_client = AzureTrafficAnalyticsClient(logger)
        logger.info("Đã khởi tạo AzureTrafficAnalyticsClient thành công.")
    except Exception as e:
        logger.error(f"Lỗi khi khởi tạo AzureTrafficAnalyticsClient: {e}", exc_info=True)
        return

    # Lấy danh sách Workspace IDs
    try:
        workspace_ids = azure_traffic_client.get_traffic_workspace_ids()
        if workspace_ids:
            logger.info(f"Danh sách Workspace IDs: {workspace_ids}")
        else:
            logger.warning("Không tìm thấy Workspace nào.")
            return
    except Exception as e:
        logger.error(f"Lỗi khi lấy danh sách Workspace: {e}", exc_info=True)
        return

    # Truy vấn dữ liệu Traffic Analytics
    try:
        query = """
        AzureNetworkAnalytics_CL
        | summarize ConnectionCount = count() by DestinationIP_s, DestinationPort, bin(TimeGenerated, 1m)
        | where ConnectionCount > 100
        """

        logger.info("Bắt đầu truy vấn Traffic Analytics...")
        # Không cần chỉ định timespan, lớp sẽ tự động xác định
        traffic_data = azure_traffic_client.get_traffic_data(query=query)

        if traffic_data:
            logger.info(f"Đã lấy được {len(traffic_data)} bảng dữ liệu từ Traffic Analytics.")

            # Log thêm chi tiết bảng đầu tiên
            first_table = traffic_data[0] if traffic_data else None
            if first_table:
                logger.info(f"Các cột trong bảng đầu tiên: {first_table.columns}")
                logger.info(f"Số dòng trong bảng đầu tiên: {len(first_table.rows)}")

            # Phân tích dữ liệu bất thường
            logger.info("Bắt đầu phân tích bất thường trong dữ liệu...")
            azure_traffic_client.analyze_traffic_anomalies(traffic_data)
        else:
            logger.warning("Không có dữ liệu trả về từ Traffic Analytics.")
    except Exception as e:
        logger.error(f"Lỗi khi truy vấn hoặc phân tích Traffic Analytics: {e}", exc_info=True)


if __name__ == "__main__":
    main()
