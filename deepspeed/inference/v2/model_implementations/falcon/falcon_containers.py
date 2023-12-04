# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# Create a container object to save model-specific tensors using the policy file above.

from ...model_implementations.common_parameters import *
from ...model_implementations.layer_container_base import LayerContainer
'''
 # HF Falcon model looks like this:

FalconForCausalLM(
  (transformer): FalconModel(
    (word_embeddings): Embedding(65024, 4544)
    (h): ModuleList(
      (0-31): 32 x FalconDecoderLayer(
        (self_attention): FalconAttention(
          (maybe_rotary): FalconRotaryEmbedding()
          (query_key_value): FalconLinear(in_features=4544, out_features=4672, bias=False)
          (dense): FalconLinear(in_features=4544, out_features=4544, bias=False)
          (attention_dropout): Dropout(p=0.0, inplace=False)
        )
        (mlp): FalconMLP(
          (dense_h_to_4h): FalconLinear(in_features=4544, out_features=18176, bias=False)
          (act): GELU(approximate='none')
          (dense_4h_to_h): FalconLinear(in_features=18176, out_features=4544, bias=False)
        )
        (input_layernorm): LayerNorm((4544,), eps=1e-05, elementwise_affine=True)
      )
    )
    (ln_f): LayerNorm((4544,), eps=1e-05, elementwise_affine=True)
  )
  (lm_head): Linear(in_features=4544, out_features=65024, bias=False)
)
'''


class FalconTransformerContainer(LayerContainer):
    """
        Transformer layer container for the Falcon model.
    """
    qkv_w: FusedQKVParameter
    attn_out_w: AttentionOutputParameter
    mlp_1_w: MLP1Parameter
    mlp_2_w: MLP2Parameter
    input_layernorm_gamma: NormParameter
    input_layernorm_beta: NormParameter

    PARAM_MAPPING = {
        "self_attention.query_key_value.weight": "qkv_w.params",
        "self_attention.dense.weight": "attn_out_w.params",
        "mlp.dense_h_to_4h.weight": "mlp_1_w.params",
        "mlp.dense_4h_to_h.weight": "mlp_2_w.params",
        "input_layernorm.weight": "input_layernorm_gamma.params",
        "input_layernorm.bias": "input_layernorm_beta.params",
    }


class FalconNonTransformerContainer(LayerContainer):
    """
        Non-Transformer layer container for the Falcon model.
    """
    word_emb: EmbeddingParameter
    word_unembed: UnembedParameter
    final_norm_gamma: NormParameter
    final_norm_beta: NormParameter

    PARAM_MAPPING = {
        "transformer.word_embeddings.weight": "word_emb.params",
        "transformer.ln_f.weight": "final_norm_gamma.params",
        "transformer.ln_f.bias": "final_norm_beta.params",
        "lm_head.weight": "word_unembed.params",
    }
