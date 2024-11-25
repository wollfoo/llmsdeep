import subprocess
import pytest
import os

# Đường dẫn Dockerfile và các tệp liên quan
DOCKERFILE_PATH = "./Dockerfile"
REQUIREMENTS_FILE_PATH = "./requirements.txt"
APP_DIR = "./app"

@pytest.fixture(scope="module")
def build_docker_image():
    """
    Fixture để xây dựng image Docker từ Dockerfile.
    """
    image_name = "llmsdeep_test_image"
    build_command = ["docker", "build", "-t", image_name, "-f", DOCKERFILE_PATH, "."]
    subprocess.run(build_command, check=True)
    yield image_name
    # Xóa image sau khi kiểm thử
    subprocess.run(["docker", "rmi", "-f", image_name], check=True)


def test_dockerfile_exists():
    """
    Kiểm tra xem Dockerfile có tồn tại không.
    """
    assert os.path.exists(DOCKERFILE_PATH), "Dockerfile không tồn tại"


def test_requirements_exists():
    """
    Kiểm tra xem requirements.txt có tồn tại không.
    """
    assert os.path.exists(REQUIREMENTS_FILE_PATH), "requirements.txt không tồn tại"


def test_app_directory_exists():
    """
    Kiểm tra xem thư mục app có tồn tại không.
    """
    assert os.path.exists(APP_DIR), "Thư mục app không tồn tại"


def test_build_docker_image(build_docker_image):
    """
    Kiểm tra quá trình build Docker image.
    """
    assert build_docker_image is not None, "Build Docker image thất bại"


def test_install_python_dependencies(build_docker_image):
    """
    Kiểm tra cài đặt các phụ thuộc Python.
    """
    container_name = "test_container"
    try:
        run_command = [
            "docker", "run", "--rm", "--name", container_name, build_docker_image, 
            "pip", "freeze"
        ]
        result = subprocess.run(run_command, capture_output=True, text=True, check=True)
        output = result.stdout
        # Kiểm tra một số phụ thuộc cụ thể
        dependencies = ["torch", "tensorflow", "psutil"]
        for dep in dependencies:
            assert dep in output, f"{dep} không được cài đặt"
    finally:
        # Đảm bảo container được xóa
        subprocess.run(["docker", "rm", "-f", container_name], check=False)


def test_environment_variables(build_docker_image):
    """
    Kiểm tra xem các biến môi trường đã được thiết lập chưa.
    """
    container_name = "test_container"
    try:
        run_command = [
            "docker", "run", "--rm", "--name", container_name, build_docker_image, 
            "env"
        ]
        result = subprocess.run(run_command, capture_output=True, text=True, check=True)
        output = result.stdout
        env_vars = [
            "AZURE_SUBSCRIPTION_ID",
            "AZURE_CLIENT_ID",
            "AZURE_CLIENT_SECRET",
            "AZURE_TENANT_ID",
        ]
        for var in env_vars:
            assert var in output, f"Biến môi trường {var} không được thiết lập"
    finally:
        subprocess.run(["docker", "rm", "-f", container_name], check=False)


def test_entrypoint(build_docker_image):
    """
    Kiểm tra xem entrypoint có hoạt động đúng không.
    """
    container_name = "test_container"
    try:
        run_command = [
            "docker", "run", "--rm", "--name", container_name, build_docker_image
        ]
        result = subprocess.run(run_command, capture_output=True, text=True, check=False)
        assert result.returncode == 0, "Entrypoint không hoạt động đúng"
    finally:
        subprocess.run(["docker", "rm", "-f", container_name], check=False)
