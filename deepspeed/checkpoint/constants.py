'''
    Various symbolic constants used for model checkpointing
'''

#########################################
# Optimizer checkpoint keys
#########################################
OPTIMIZER_STATE_DICT = "optimizer_state_dict"
FP32_GROUPS = "fp32_groups"
FP32_FLAT_GROUPS = 'fp32_flat_groups'

BASE_OPTIMIZER_STATE = 'base_optimizer_state'
SINGLE_PARTITION_OF_FP32_GROUPS = "single_partition_of_fp32_groups"

PARTITION_COUNT = 'partition_count'
ZERO_STAGE = 'zero_stage'

#########################################
# Module checkpoint keys
#########################################
PARAM_SHAPES = 'param_shapes'
BUFFER_NAMES = 'buffer_names'

#########################################
# Checkpoint naming constants
#########################################
MODEL_FILE_PREFIX = 'mp_rank_'
ZERO_FILE_PREFIX = 'zero_pp_rank_'
OPTIM_FILE_SUFFIX = '_optim_states.pt'
MODEL_FILE_SUFFIX = '_model_states.pt'

#########################################
# Checkpoint utility keys
#########################################
EMBEDDING_LAYER_INDEX = 0
FINAL_LAYER_NORM_INDEX = -1
ARGS_KEY = 'args'
CHECKPOINT_INFO_KEY = 'checkpoint_info'
ITERATION_KEY = 'iteration'
SEQUENTIAL_LAYERS = [
    'input_layernorm.weight',
    'input_layernorm.bias',
    'self_attention.dense.bias',
    'post_attention_layernorm.weight',
    'post_attention_layernorm.bias',
    'mlp.dense_4h_to_h.bias',
    'position_embeddings.weight'
]

LAYER_CONCAT_DIM = {'self_attention.dense.weight': 1, 'mlp.dense_4h_to_h.weight': 1}
