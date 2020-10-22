try:
    #  This is populated by setup.py
    from .git_version_info_installed import *
except ModuleNotFoundError:
    # Will be missing from checkouts that haven't been installed (e.g., readthedocs)
    version = open('version.txt', 'r').read().strip()
    git_hash = '[none]'
    git_branch = '[none]'
    installed_ops = {
        'cpu_adam_op': False,
        'fused_adam_op': False,
        'fused_lamb_op': False,
        'sparse_attn_op': False,
        'transformer_op': False,
        'stochastic_transformer_op': False,
        'utils_op': False
    }
    compatible_ops = {
        'cpu_adam_op': False,
        'fused_adam_op': False,
        'fused_lamb_op': False,
        'sparse_attn_op': False,
        'transformer_op': False,
        'stochastic_transformer_op': False,
        'utils_op': False
    }
