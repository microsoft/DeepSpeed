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
from deepspeed.accelerator import get_accelerator

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
    parser.addoption("--device_ver", default=None, type=str)


def validate_version(expected, found):
    if found == "ALL_SUPPORT":
        return True
    version_depth = expected.count('.') + 1
    found = '.'.join(found.split('.')[:version_depth])
    return found == expected


@pytest.fixture(scope="session", autouse=True)
def check_environment(pytestconfig):
    expected_torch_version = pytestconfig.getoption("torch_ver")
    expected_device_version = pytestconfig.getoption("device_ver")
    found_device_version = get_accelerator().get_found_device_version()
    if expected_torch_version is None:
        warnings.warn(
            "Running test without verifying torch version, please provide an expected torch version with --torch_ver")
    elif not validate_version(expected_torch_version, torch.__version__):
        pytest.exit(
            f"expected torch version {expected_torch_version} did not match found torch version {torch.__version__}",
            returncode=2)
    if expected_device_version is None:
        warnings.warn("Running test without verifying device version, "
                      "please provide an expected device version with --device_ver")
    elif not validate_version(expected_device_version, found_device_version):
        pytest.exit(
            f"expected device version {expected_device_version} "
            f"did not match found device version {found_device_version}",
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


# We allow DistributedTest to reuse distributed environments. When the last
# test for a class is run, we want to make sure those distributed environments
# are destroyed.
def pytest_runtest_teardown(item, nextitem):
    if getattr(item.cls, "reuse_dist_env", False) and not nextitem:
        dist_test_class = item.cls()
        for num_procs, pool in dist_test_class._pool_cache.items():
            dist_test_class._close_pool(pool, num_procs, force=True)


@pytest.hookimpl(tryfirst=True)
def pytest_fixture_setup(fixturedef, request):
    if getattr(fixturedef.func, "is_dist_fixture", False):
        dist_fixture_class = fixturedef.func()
        dist_fixture_class(request)
