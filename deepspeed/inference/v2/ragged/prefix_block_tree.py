from typing import Dict, List, Set
import hashlib
from collections import defaultdict

import torch


def token_ids_to_hash(token_ids: torch.Tensor):
    # Convert the tensor to bytes
    tensor_bytes = token_ids.numpy().tobytes()
    hash_obj = hashlib.sha256()
    # Update the hash object with the bytes
    hash_obj.update(tensor_bytes)
    # Get the hexadecimal digest of the hash
    return hash_obj.hexdigest()


class PrefixBlockMap():

    def __init__(self, block_size: int):
        self.tokens_to_blocks: Dict[str, List[int]] = {}
        self.blocks_to_tokens: Dict[Set[int], str] = {}
        self.block_size: int = block_size
    
    def lookup(self, tokens: torch.Tensor) -> torch.Tensor:
        n_blocks = len(tokens) // self.block_size
        cached_blocks = torch.tensor([], dtype=torch.int32)
        for i in range(n_blocks):
            chunk = tokens[:(i+1)*self.block_size]
            hash = token_ids_to_hash(chunk)
            if hash in self.tokens_to_blocks:
                cached_blocks = self.tokens_to_blocks[hash]
            else:
                break
        return cached_blocks

    def extend(self, tokens: torch.Tensor, new_block_ids: List[int]) -> None:
        n_blocks = len(tokens) // self.block_size
        for i in range(n_blocks):
            chunk = tokens[:(i+1)*self.block_size]
            hash = token_ids_to_hash(chunk)
            if hash not in self.tokens_to_blocks:
                self.tokens_to_blocks[hash] = new_block_ids[:i+1]
                self.blocks_to_tokens[frozenset(new_block_ids[:i+1])] = hash

    def delete(self, block_ids: List[int]) -> None:
        blocks_set = frozenset(block_ids)
        for used_blocks, hash in self.blocks_to_tokens.items():
            # check intersection
            if blocks_set & used_blocks:
                del self.tokens_to_blocks[hash]
                del self.blocks_to_tokens[used_blocks]

