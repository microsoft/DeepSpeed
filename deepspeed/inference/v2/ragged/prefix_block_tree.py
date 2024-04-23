from typing import Any, Dict, Optional, List
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


class PrefixBlockNode:
    def __init__(self, token_ids: torch.Tensor, block_id: int, children: Dict):
        self.token_ids = token_ids
        self.block_id = block_id
        self.children = children
        self.hash = token_ids_to_hash(token_ids) 
        self.ref_count = 1

    def add_child(self, hash, node):
        self.children[hash] = node

    def get_child(self, prefix):
        return self.children.get(prefix)

    def inc_ref_count(self):
        self.ref_count += 1

    def dec_ref_count(self):
        self.ref_count -= 1

    def __repr__(self):
        return f"PrefixBlockNode(token_ids={self.token_ids.shape}, block_id={self.block_id}, children={self.children}, ref_count={self.ref_count})"


class PrefixBlockTree():

    def __init__(self, block_size: int):
        self.root: PrefixBlockNode = PrefixBlockNode(token_ids=torch.tensor([], dtype=torch.int32), block_id=-1, children={})
    
        # Mapping from uid to token_ids.
        self.prefix_nodes: Dict[int, List[PrefixBlockNode]] = defaultdict(list)
        self.tokens: Dict[int, torch.Tensor] = defaultdict(lambda: torch.tensor([], dtype=torch.int32))
        self.block_size: int = block_size

    def lookup(self, tokens: torch.Tensor, increment_ref=False, decrement_ref=False) -> List[PrefixBlockNode]:
        assert not (increment_ref and decrement_ref), 'increment_ref and decrement_ref cannot be set to True at the same time'

        chunks = torch.split(tokens, self.block_size)
        current_node = self.root
        path = []
        for chunk in chunks:
            hash = token_ids_to_hash(chunk)
            # print(f"lookup chunk={chunk.shape} hash={hash}")
            if hash in current_node.children:
                current_node = current_node.children[hash]
                path.append(current_node)
                if increment_ref:
                    current_node.inc_ref_count()
            else:
                break
        return path

    def allocate(self, tokens: torch.Tensor) -> List[int]:
        path = self.lookup(tokens)
        if len(path) == 0:
            return torch.tensor([], dtype=torch.int32)
        return torch.cat([node.block_id.unsqueeze(0) for node in path])

    def extend(self, uid: int, tokens: torch.Tensor, new_block_ids: List[int]) -> None:
        path = self.prefix_nodes[uid]
        self.tokens[uid] = torch.cat([self.tokens[uid], tokens])

        n_full_blocks = len(self.tokens[uid]) // self.block_size
        new_full_blocks = n_full_blocks - len(path)

        if new_full_blocks == 0:
            return

        chunks = torch.split(tokens, self.block_size)[len(path):len(path) + new_full_blocks]
        if len(path) == 0:
            current_node = self.root
        else:
            current_node = path[-1]

        for chunk, block_id in zip(chunks, new_block_ids):
            hash = token_ids_to_hash(chunk)
            assert hash not in current_node.children, 'Chunk already exists in the tree'

            new_node = PrefixBlockNode(token_ids=chunk, block_id=block_id, children={})
            current_node.add_child(hash, new_node)

            path.append(current_node)
            current_node = current_node.children[hash]
            # current_node.inc_ref_count()

            # print(f"adding chunk to tree: hash={hash} current_node={current_node}")

    def delete(self, prefix_ids: torch.Tensor) -> None:
        self.lookup(prefix_ids, decrement_ref=True)
