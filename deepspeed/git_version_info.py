try:
    from .git_version_info_installed import *
except ImportError:
    from .git_version_info_sample import *
