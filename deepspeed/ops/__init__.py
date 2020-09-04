from ..git_version_info import installed_ops
from . import lamb
from . import transformer
if installed_ops['sparse-attn']:
    from . import sparse_attention
if installed_ops['cpu-adam']:
    from . import adam
