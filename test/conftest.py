import pytest
import os
from pathlib import Path
from app.mining_environment.scripts.resource_manager import ResourceManager
from app.mining_environment.scripts.temperature_monitor import TemperatureMonitor


# ==== FIXTURES CHUNG ====

@pytest.fixture(scope="session")
def test_config():
    """
    Fixture cung cấp cấu hình kiểm thử mock.
    Fixture này có phạm vi session (tồn tại xuyên suốt tất cả kiểm thử).
    """
    return {
        "resource_allocation": {
            "cpu": {"cpu_shares": 1024, "cpu_quota": 50000},
            "ram": {"max_allocation_mb": 2048},
            "disk_io": {"read_limit_mbps": 50, "write_limit_mbps": 50},
        },
        "log_directory": "/tmp/logs",  # Chỉ định thư mục logs tạm thời
    }


@pytest.fixture(scope="module")
def mock_resource_manager(test_config):
    """
    Fixture khởi tạo ResourceManager với cấu hình mock.
    """
    resource_manager = ResourceManager(
        config=test_config,
        optimization_model_path="path/to/mock/model.pt",
        logger=None,  # Không sử dụng logger thật trong kiểm thử
    )
    yield resource_manager
    resource_manager.stop()  # Đảm bảo dừng ResourceManager sau kiểm thử


@pytest.fixture(scope="function")
def temperature_monitor_instance():
    """
    Fixture khởi tạo một đối tượng TemperatureMonitor cho mỗi bài kiểm thử.
    """
    temperature_monitor = TemperatureMonitor()
    yield temperature_monitor
    temperature_monitor.shutdown()  # Đảm bảo shutdown sau kiểm thử


# ==== CÁC FIXTURE HỖ TRỢ DOCKER ====

@pytest.fixture(scope="session")
def docker_image():
    """
    Fixture build Docker image kiểm thử từ Dockerfile.
    """
    import subprocess
    image_name = "test-image"

    # Build Docker image
    result = subprocess.run(
        ["docker", "build", "-t", image_name, "."],
        cwd="../app",  # Chỉ định thư mục Dockerfile
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to build Docker image: {result.stderr.decode()}")
    yield image_name

    # Cleanup: Xóa Docker image sau khi kiểm thử
    subprocess.run(["docker", "rmi", image_name])


@pytest.fixture(scope="function")
def docker_container(docker_image):
    """
    Fixture khởi chạy Docker container kiểm thử.
    """
    import subprocess
    container_name = "test-container"

    # Run Docker container
    result = subprocess.run(
        ["docker", "run", "--rm", "--name", container_name, docker_image],
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to run Docker container: {result.stderr.decode()}")

    yield container_name

    # Cleanup: Dừng container nếu cần (nếu không sử dụng `--rm`)
    subprocess.run(["docker", "stop", container_name])


# ==== FIXTURE CHO TESTING CỤ THỂ ====

@pytest.fixture(scope="function")
def mock_temperature_data():
    """
    Fixture tạo mock dữ liệu nhiệt độ cho kiểm thử TemperatureMonitor.
    """
    return {
        "cpu_temperature": 50.0,
        "gpu_temperatures": [45.0, 47.5],
    }


@pytest.fixture(scope="function")
def create_temp_file():
    """
    Fixture tạo một file tạm thời, sau đó tự động xóa khi kết thúc kiểm thử.
    """
    temp_file_path = Path("/tmp/temp_test_file.txt")
    with open(temp_file_path, "w") as f:
        f.write("This is a temporary test file.")
    yield temp_file_path

    # Cleanup: Xóa file sau kiểm thử
    if temp_file_path.exists():
        temp_file_path.unlink()


@pytest.fixture(scope="function")
def mock_log_directory():
    """
    Fixture tạo thư mục logs tạm thời để kiểm thử.
    """
    log_dir = Path("/tmp/mock_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    yield log_dir

    # Cleanup: Xóa thư mục logs sau kiểm thử
    if log_dir.exists():
        for log_file in log_dir.iterdir():
            log_file.unlink()
        log_dir.rmdir()
