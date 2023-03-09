_gpt2_small = dict(
    seq_length=1024,
    vocab_size=50257,
    hidden_size=768,
    num_heads=12,
    depth=12,
    numel=124439808,
    checkpoint=True,
    evaluation="ppl",
)

_gpt2_xl = dict(
    seq_length=1024,
    vocab_size=50257,
    hidden_size=1600,
    num_heads=25,
    depth=48,
    numel=1557611200,
    checkpoint=True,
    evaluation="ppl",
)

_gpt2_10b = dict(
    seq_length=1024,
    vocab_size=50257,
    hidden_size=4096,
    num_heads=16,
    depth=50,
    numel=10279047168,
    checkpoint=True,
    evaluation="ppl",
)

_gpt2_4b = dict(
    seq_length=1024,
    vocab_size=50257,
    hidden_size=2304,
    num_heads=16,
    depth=64,
    checkpoint=True,
    evaluation="ppl",
)

_gpt2_2b = dict(
    seq_length=1024,
    vocab_size=50257,
    hidden_size=2048,
    num_heads=16,
    depth=40,
    checkpoint=True,
    evaluation="ppl",
)

_gpt2_3b = dict(
    seq_length=1024,
    vocab_size=50257,
    hidden_size=2560,
    num_heads=40,
    depth=24,
    checkpoint=True,
    evaluation="ppl",
)

_gpt2_6b = dict(
    seq_length=1024,
    vocab_size=50257,
    hidden_size=4096,
    num_heads=16,
    depth=30,
    checkpoint=True,
    evaluation="ppl",
)

_gpt2_8b = dict(
    seq_length=1024,
    vocab_size=50257,
    hidden_size=4096,
    num_heads=16,
    depth=40,
    checkpoint=True,
    evaluation="ppl",
)

_gpt2_12b = dict(
    seq_length=1024,
    vocab_size=50257,
    hidden_size=4096,
    num_heads=16,
    depth=60,
    checkpoint=True,
    evaluation="ppl",
)

_gpt2_15b = dict(
    seq_length=1024,
    vocab_size=50257,
    hidden_size=4096,
    num_heads=16,
    depth=78,
    checkpoint=True,
    evaluation="ppl",
)

_gpt2_18b = dict(
    seq_length=1024,
    vocab_size=50257,
    hidden_size=4096,
    num_heads=16,
    depth=90,
    checkpoint=True,
    evaluation="ppl",
)

_gpt2_20b = dict(
    seq_length=1024,
    vocab_size=50257,
    hidden_size=8192,
    num_heads=16,
    depth=25,
    checkpoint=True,
    evaluation="ppl",
)

_gpt2_24b = dict(
    seq_length=1024,
    vocab_size=50257,
    hidden_size=8192,
    num_heads=16,
    depth=30,
    checkpoint=True,
    evaluation="ppl",
)

_gpt2_28b = dict(
    seq_length=1024,
    vocab_size=50257,
    hidden_size=8192,
    num_heads=16,
    depth=35,
    checkpoint=True,
    evaluation="ppl",
)

_gpt2_32b = dict(
    seq_length=1024,
    vocab_size=50257,
    hidden_size=8192,
    num_heads=16,
    depth=40,
    checkpoint=True,
    evaluation="ppl",
)

_gpt2_36b = dict(
    seq_length=1024,
    vocab_size=50257,
    hidden_size=8192,
    num_heads=16,
    depth=45,
    checkpoint=True,
    evaluation="ppl",
)

_gpt2_40b = dict(
    seq_length=1024,
    vocab_size=50257,
    hidden_size=8192,
    num_heads=16,
    depth=50,
    checkpoint=True,
    evaluation="ppl",
)

gpt2_configs = {
    "gpt2": _gpt2_small,
    "gpt2-small": _gpt2_small,
    "gpt2-xl": _gpt2_xl,
    "gpt2-10b": _gpt2_10b,
    "gpt2-4b": _gpt2_4b,
    "gpt2-6b": _gpt2_6b,
    "gpt2-8b": _gpt2_8b,
    "gpt2-2b": _gpt2_2b,
    "gpt2-3b": _gpt2_3b,
    "gpt2-12b": _gpt2_12b,
    "gpt2-15b": _gpt2_15b,
    "gpt2-18b": _gpt2_18b,
    "gpt2-20b": _gpt2_20b,
    "gpt2-40b": _gpt2_40b,
    "gpt2-24b": _gpt2_24b,
    "gpt2-28b": _gpt2_28b,
    "gpt2-32b": _gpt2_32b,
    "gpt2-36b": _gpt2_36b,
}
