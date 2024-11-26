# test/conftest.py

import pytest
import os

@pytest.fixture(scope="session", autouse=True)
def set_testing_env():
    os.environ["TESTING"] = "1"
