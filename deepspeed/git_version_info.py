try:
    #  This is populated by setup.py
    from .git_version_info_installed import *
except ModuleNotFoundError:
    # Will be missing from checkouts that haven't been installed (e.g., readthedocs)
    version = '0.3.0+[none]'
    git_hash = '[none]'
    git_branch = '[none]'

    from .ops.op_builder import ALL_OPS
    installed_ops = dict.fromkeys(ALL_OPS.keys(), False)
    compatible_ops = dict.fromkeys(ALL_OPS.keys(), False)
