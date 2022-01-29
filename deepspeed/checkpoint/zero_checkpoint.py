import torch
from .reshape_utils import (basic_folder_validation,
                            get_files,
                            get_files_with_prefix,
                            merge_state,
                            ZERO_FILE_PREFIX)


class ZeROCheckpoint(object):
    def __init__(self, dir):
        basic_folder_validation(dir)
        self.dir = dir
        self.file_list = get_files_with_prefix(get_files(dir), ZERO_FILE_PREFIX)
        self.num_files = len(self.file_list)

    def get_files_for_global_rank(self, world_size, global_rank):
        assert global_rank < world_size, f'Expected global_rank {global_rank} to be less than world size {world_size}'
        if world_size == self.num_files:
            return [self.file_list[global_rank]]
        elif world_size < self.num_files:
            assert self.num_files % world_size == 0, \
                f'Expected world size {world_size} that can divide number of zero files {self.num_files}'
            num_files_per_rank = self.num_files // world_size
            starting_index = global_rank * num_files_per_rank
            return self.file_list[starting_index:(starting_index + num_files_per_rank)]
        else:
            assert world_size % self.num_files == 0, \
                f'Expected world size {world_size} that is multiple of number of zero files {self.num_files}'
            num_ranks_per_file = world_size // self.num_files
            return [self.file_list[global_rank // num_ranks_per_file]]

    def get_state_for_global_rank(self, world_size, global_rank, keys_to_ignore=[]):
        rank_file_list = self.get_files_for_global_rank(world_size, global_rank)
        assert len(rank_file_list) > 0, f'Expected global_rank files count {len(rank_file_list)} > 0'
        rank_state = None
        for ckpt_file in rank_file_list:
            sd = torch.load(ckpt_file, map_location=torch.device('cpu'))
            for key in keys_to_ignore:
                sd.pop(key, None)
            if rank_state is None:
                rank_state = sd
            else:
                rank_state = merge_state(rank_state, sd)

        return rank_state
