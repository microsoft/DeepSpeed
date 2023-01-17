from .replace_module import replace_transformer_layer, revert_transformer_layer, ReplaceWithTensorSlicing, GroupQuantizer
from .module_quantize import quantize_transformer_layer
from .layers import LinearAllreduce, LinearLayer, EmbeddingLayer, Normalize
from .replace_policy import DSPolicy, HFBertLayerPolicy, \
    HFGPTNEOLayerPolicy, \
    GPTNEOXLayerPolicy, \
    HFGPTJLayerPolicy, \
    MegatronLayerPolicy, \
    HFGPT2LayerPolicy, \
    BLOOMLayerPolicy, \
    HFOPTLayerPolicy, \
    HFCLIPLayerPolicy, \
    HFDistilBertLayerPolicy
