parser_policy_map = dict( 
    codegen=dict(CodeGenBlock=("mlp.fc_out", "attn.out_proj")),
    blenderbot=dict(BlenderbotEncoderLayer=(".fc2", "self_attn.out_proj", ), BlenderbotDecoderLayer=(".fc2", "encoder_attn.out_proj", "self_attn.out_proj",)),
    electra=dict(ElectraLayer=("output.dense")),
    roberta=dict(RobertaLayer=("output.dense")),
    t5=dict(T5Block=("SelfAttention.o", "EncDecAttention.o", "DenseReluDense.wo")),
    albert=dict(AlbertLayer=("attention.dense", "ffn_output")),
    bart=dict(BartEncoderLayer=("self_attn.out_proj", "fc2")),
    deberta=dict(DebertaLayer=("output.dense")),
    deberta_v2=dict(DebertaV2Layer=("output.dense")),
    wav2vec2=dict(Wav2Vec2EncoderLayer=("attention.out_proj", "feed_forward.output_dense")),
)
