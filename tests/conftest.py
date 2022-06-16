# tests directory-specific settings - this file is run automatically by pytest before any tests are run

import sys
import pytest
from os.path import abspath, dirname, join
import torch
import warnings

# allow having multiple repository checkouts and not needing to remember to rerun
# 'pip install -e .[dev]' when switching between checkouts and running tests.
git_repo_path = abspath(join(dirname(dirname(__file__)), "src"))
sys.path.insert(1, git_repo_path)


def pytest_addoption(parser):
    parser.addoption("--torch_ver", default=None, type=str)
    parser.addoption("--cuda_ver", default=None, type=str)


@pytest.fixture(scope="session", autouse=True)
def check_environment(pytestconfig):
    expected_torch_version = pytestconfig.getoption("torch_ver")
    expected_cuda_version = pytestconfig.getoption("cuda_ver")
    torch_version = '.'.join(torch.__version__.split('.')[:2])
    cuda_version = torch.version.cuda
    if expected_torch_version is None:
        warnings.warn(
            "Running test without verifying torch version, please provide an expected torch version with --torch_ver"
        )
    elif expected_torch_version != torch_version:
        pytest.exit(
            f"expected torch version {expected_torch_version} did not match found torch version {torch_version}",
            returncode=2)
    if expected_cuda_version is None:
        warnings.warn(
            "Running test without verifying cuda version, please provide an expected cuda version with --cuda_ver"
        )
    elif expected_cuda_version != cuda_version:
        pytest.exit(
            f"expected cuda version {expected_cuda_version} did not match found cuda version {cuda_version}",
            returncode=2)
