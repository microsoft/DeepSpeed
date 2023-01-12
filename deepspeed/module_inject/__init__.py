from .replace_module import revert_transformer_layer, ReplaceWithTensorSlicing, GroupQuantizer, generic_injection
from .replace_module import replace_transformer_layer
from .module_quantize import quantize_transformer_layer
from .replace_policy import HFBertLayerPolicy
from .layers import LinearAllreduce, LinearLayer, EmbeddingLayer, Normalize
from .policy import DSPolicy
