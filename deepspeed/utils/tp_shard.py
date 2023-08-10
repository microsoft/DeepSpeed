from deepspeed import comm as dist
global num_kv_heads

def set_num_kv_heads(num):
    global num_kv_heads
    num_kv_heads = num

def get_shard_size(total_size, mp_size):
    global num_kv_heads
    # When we have num_kv_heads defined, uneven division is possible, otherwise enforce even division
    if num_kv_heads != None:
        my_slices = num_kv_heads // mp_size + (1 if dist.get_rank() < (num_kv_heads % mp_size) else 0)
        return total_size // num_kv_heads * my_slices
    else:
        if total_size % mp_size == 0:
            return total_size // mp_size
        else:
            assert False, f"Number of attention heads ({total_size}) must be divisible by mp_size ({mp_size})"
