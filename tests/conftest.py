# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# tests directory-specific settings - this file is run automatically by pytest before any tests are run

import sys
import pytest
import os
from os.path import abspath, dirname, join
import torch
import warnings

# Set this environment variable for the T5 inference unittest(s) (e.g. google/t5-v1_1-small)
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# allow having multiple repository checkouts and not needing to remember to rerun
# 'pip install -e .[dev]' when switching between checkouts and running tests.
git_repo_path = abspath(join(dirname(dirname(__file__)), "src"))
sys.path.insert(1, git_repo_path)


def pytest_configure(config):
    config.option.color = "yes"
    config.option.durations = 0
    config.option.durations_min = 1
    config.option.verbose = True


def pytest_addoption(parser):
    parser.addoption("--torch_ver", default=None, type=str)
    parser.addoption("--cuda_ver", default=None, type=str)


def validate_version(expected, found):
    version_depth = expected.count('.') + 1
    found = '.'.join(found.split('.')[:version_depth])
    return found == expected


@pytest.fixture(scope="session", autouse=True)
def check_environment(pytestconfig):
    expected_torch_version = pytestconfig.getoption("torch_ver")
    expected_cuda_version = pytestconfig.getoption("cuda_ver")
    if expected_torch_version is None:
        warnings.warn(
            "Running test without verifying torch version, please provide an expected torch version with --torch_ver")
    elif not validate_version(expected_torch_version, torch.__version__):
        pytest.exit(
            f"expected torch version {expected_torch_version} did not match found torch version {torch.__version__}",
            returncode=2)
    if expected_cuda_version is None:
        warnings.warn(
            "Running test without verifying cuda version, please provide an expected cuda version with --cuda_ver")
    elif not validate_version(expected_cuda_version, torch.version.cuda):
        pytest.exit(
            f"expected cuda version {expected_cuda_version} did not match found cuda version {torch.version.cuda}",
            returncode=2)


# Override of pytest "runtest" for DistributedTest class
# This hook is run before the default pytest_runtest_call
@pytest.hookimpl(tryfirst=True)
def pytest_runtest_call(item):
    # We want to use our own launching function for distributed tests
    if getattr(item.cls, "is_dist_test", False):
        dist_test_class = item.cls()
        dist_test_class(item._request)
        item.runtest = lambda: True  # Dummy function so test is not run twice


def write_to_log_with_lock(log_file_path: str, header: str, msg: str):
    import fcntl
    with open(log_file_path, 'a+') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(f"{header} {msg}\n")
            f.flush()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


dist_test_class = None


# We allow DistributedTest to reuse distributed environments. When the last
# test for a class is run, we want to make sure those distributed environments
# are destroyed.
def pytest_runtest_teardown(item, nextitem):
    RUNNING_TEST_LOG_FILE = os.environ.get("RUNNING_TEST_LOG_FILE", "/tmp/running_test.log")

    global dist_test_class
    # Last test might not have .cls. So we record the pool_cache here
    if item.cls is not None:
        dist_test_class = item.cls()

    def get_xdist_worker_id():
        xdist_worker = os.environ.get('PYTEST_XDIST_WORKER', None)
        if xdist_worker is not None:
            xdist_worker_id = xdist_worker.replace('gw', '')
            return int(xdist_worker_id)
        return None

    if RUNNING_TEST_LOG_FILE:
        reuse_dist_env = getattr(item.cls, "reuse_dist_env", False)
        write_to_log_with_lock(RUNNING_TEST_LOG_FILE, f"pytest_runtest_teardown,xdist={get_xdist_worker_id()}",
                               f"reuse_dist_env={reuse_dist_env} nextitem={nextitem}")

    if not nextitem and dist_test_class is not None and dist_test_class._pool_cache is not None:
        for num_procs, pool in dist_test_class._pool_cache.items():
            write_to_log_with_lock(RUNNING_TEST_LOG_FILE, f"pytest_runtest_teardown,xdist={get_xdist_worker_id()}",
                                   f"closing pool num_procs={num_procs} nextitem={nextitem}")
            dist_test_class._close_pool(pool, num_procs, force=True)


@pytest.hookimpl(tryfirst=True)
def pytest_fixture_setup(fixturedef, request):
    if getattr(fixturedef.func, "is_dist_fixture", False):
        dist_fixture_class = fixturedef.func()
        dist_fixture_class(request)
