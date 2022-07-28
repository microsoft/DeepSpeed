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
            "Running test without verifying torch version, please provide an expected torch version with --torch_ver"
        )
    elif not validate_version(expected_torch_version, torch.__version__):
        pytest.exit(
            f"expected torch version {expected_torch_version} did not match found torch version {torch.__version__}",
            returncode=2)
    if expected_cuda_version is None:
        warnings.warn(
            "Running test without verifying cuda version, please provide an expected cuda version with --cuda_ver"
        )
    elif not validate_version(expected_cuda_version, torch.version.cuda):
        pytest.exit(
            f"expected cuda version {expected_cuda_version} did not match found cuda version {torch.version.cuda}",
            returncode=2)


''' Override of pytest "runtest" for DistributedTest class '''


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_call(item):
    for val in dir(item._request):
        print(val.upper(), getattr(item._request, val), "\n\n")
    print(list(item._request.keywords.items()))
    # We want to use our own launching function for distributed tests
    if getattr(item.cls, "dist_test", False):
        dist_test_class = item.cls()
        dist_test_class._run_test(item._request)
        item.runtest = lambda: True  # Dummy function so test is not run twice


'''
in_dist_test = False
@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    global in_dist_test
    result = outcome.get_result()
    # DistributedTest runs the test in the setup phase. After it runs, we want
    # to skip running the test again with pytest "call" phase. To do that, we
    # skip and add the reason as "dist-test-pass"
    if (call.when == 'setup') and (result.outcome
                                   == 'skipped') and (call.excinfo.value.msg
                                                      == 'dist-test-pass'):
        in_dist_test = True
        result.outcome = 'passed'
    # Because we manually set the setup outcome to "passed", we will need to
    # catch the upcoming failure when pytest tries to run the test again
    if (call.when == 'call') and in_dist_test:
        result.outcome = 'passed'
        in_dist_test = False
'''
