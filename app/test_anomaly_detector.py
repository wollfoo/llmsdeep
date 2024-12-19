import os
import json
import logging
from azure.ai.anomalydetector import AnomalyDetectorClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.anomalydetector.models import UnivariateDetectionOptions, TimeSeriesPoint

# Thiết lập logging
logging.basicConfig(level=logging.INFO)

def load_config(config_path='mining_environment/config/resource_config.json'):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def test_anomaly_detector(api_base, api_key):
    try:
        # Khởi tạo client
        client = AnomalyDetectorClient(
            endpoint=api_base,
            credential=AzureKeyCredential(api_key)
        )
        logging.info("Khởi tạo AnomalyDetectorClient thành công.")

        # Chuẩn bị dữ liệu series (ít nhất 12 điểm dữ liệu)
        series = [
            TimeSeriesPoint(timestamp="2023-01-01", value=100),
            TimeSeriesPoint(timestamp="2023-01-02", value=102),
            TimeSeriesPoint(timestamp="2023-01-03", value=101),
            TimeSeriesPoint(timestamp="2023-01-04", value=105),
            TimeSeriesPoint(timestamp="2023-01-05", value=107),
            TimeSeriesPoint(timestamp="2023-01-06", value=110),
            TimeSeriesPoint(timestamp="2023-01-07", value=108),
            TimeSeriesPoint(timestamp="2023-01-08", value=112),
            TimeSeriesPoint(timestamp="2023-01-09", value=115),
            TimeSeriesPoint(timestamp="2023-01-10", value=117),
            TimeSeriesPoint(timestamp="2023-01-11", value=120),  # Bổ sung
            TimeSeriesPoint(timestamp="2023-01-12", value=125)   # Bổ sung
        ]
        granularity = 'daily'  # hoặc 'hourly', 'minutely' tùy vào dữ liệu

        # Tạo đối tượng UnivariateDetectionOptions bao gồm cả series và granularity
        options = UnivariateDetectionOptions(
            series=series,
            granularity=granularity,
            max_anomaly_ratio=0.25,
            sensitivity=50
        )

        # Gọi phương thức detect_univariate_entire_series với options duy nhất
        result = client.detect_univariate_entire_series(options=options)

        logging.info("Yêu cầu Detect Univariate Entire Series Anomaly thành công.")
        logging.info(f"Kết quả phát hiện bất thường: {result.is_anomaly}")

    except Exception as e:
        logging.error(f"Lỗi khi kiểm tra Anomaly Detector: {e}")

if __name__ == "__main__":
    config = load_config()
    azure_config = config.get('azure_anomaly_detector', {})
    api_base = azure_config.get('api_base')
    api_key = azure_config.get('api_key')

    if api_key.startswith('${') and api_key.endswith('}'):
        env_var = api_key[2:-1]
        api_key = os.getenv(env_var)

    if not api_base or not api_key:
        logging.error("Thiếu api_base hoặc api_key trong cấu hình.")
        exit(1)

    test_anomaly_detector(api_base, api_key)
