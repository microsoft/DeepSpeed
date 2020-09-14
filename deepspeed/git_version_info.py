try:
    #  This is populated by setup.py
    from .git_version_info_installed import *
except ModuleNotFoundError:
    # Will be missing from checkouts that haven't been installed (e.g., readthedocs)
    from .git_version_info_template import *
