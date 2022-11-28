from .replace_module import revert_transformer_layer, ReplaceWithTensorSlicing, GroupQuantizer, generic_injection
# if we want to use the refactored/new code, we need to import the new code
from .replace_layer import replace_transformer_layer
# else: we need to import the old code
#from .replace_module import replace_transformer_layer
from .module_quantize import quantize_transformer_layer
from .replace_policy import HFBertLayerPolicy
from .layers import LinearAllreduce, LinearLayer, EmbeddingLayer, Normalize
from .policy import DSPolicy
