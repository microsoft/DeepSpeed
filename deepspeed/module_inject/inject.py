import copy
import torch
from deepspeed.ops.transformer import DeepSpeedTransformerLayer, DeepSpeedTransformerConfig


def module_inject(layer_obj, model, config, micro_batch_size, max_seq_length, seed):
    for name, child in model.named_children():
        if isinstance(child, layer_obj):
            print('REPLACING BertLayer')

            cuda_config = DeepSpeedTransformerConfig(
                batch_size=micro_batch_size,
                max_seq_length=max_seq_length,
                hidden_size=config.hidden_size,
                heads=config.num_attention_heads,
                attn_dropout_ratio=config.attention_probs_dropout_prob,
                hidden_dropout_ratio=config.hidden_dropout_prob,
                num_hidden_layers=config.num_hidden_layers,
                initializer_range=config.initializer_range,
                seed=seed,
                fp16=True,
                pre_layer_norm=True)

            new_module = DeepSpeedTransformerLayer(cuda_config)

            #TODO: copy relevant state from child -> new module

            setattr(model, name, copy.deepcopy(new_module))

        else:
            module_inject(layer_obj,
                          child,
                          config,
                          micro_batch_size,
                          max_seq_length,
                          seed)

    return model


def test_hi():
    from turing.nvidia_modelingpreln import BertConfig as BertConfigPreLN
    from turing.nvidia_modelingpreln import BertForQuestionAnswering as BertForQuestionAnsweringPreLN
    from turing.nvidia_modelingpreln import BertLayer
    bert_model_config = {
        "vocab_size_or_config_json_file": 119547,
        "hidden_size": 1024,
        "num_hidden_layers": 1,
        "num_attention_heads": 16,
        "intermediate_size": 4096,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_position_embeddings": 512,
        "type_vocab_size": 2,
        "initializer_range": 0.02
    }
    bert_config = BertConfigPreLN(**bert_model_config)
    base_model = BertForQuestionAnsweringPreLN(bert_config, args=None)

    #base_model = LinearStack()

    test_model = copy.deepcopy(base_model)
    test_model = module_inject(BertLayer, test_model, bert_config, 4, 384, 1234)

    print('BASE', base_model)
    print('TEST', test_model)

    #base_model.eval()
    #test_model.eval()

    #test_input = torch.rand(1, base_model.input_dim)

    #base_output = base_model(test_input)
    #test_output = test_model(test_input)
    #
    #assert torch.allclose(base_output, test_output, atol=3e-8)
