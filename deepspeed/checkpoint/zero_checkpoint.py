import torch
from .reshape_utils import (basic_folder_validation,
                            get_files,
                            get_files_with_prefix,
                            ZERO_FILE_PREFIX,
                            merge_state)

from .reshape_3d_utils import (model_3d_desc, get_model_3d_descriptor)


class ZeROCheckpoint(object):
    def __init__(self, dir):
        basic_folder_validation(dir)
        self.dir = dir
        self.file_list = get_files_with_prefix(get_files(dir), ZERO_FILE_PREFIX)
        self.num_files = len(self.file_list)

        self.src_3d = get_model_3d_descriptor(dir)
        self.target_3d = model_3d_desc(pp_degree=self.src_3d.pp_degree,
                                       tp_degree=self.src_3d.tp_degree,
                                       dp_degree=self.src_3d.dp_degree)
        self._3d_file_map = self.src_3d.reshape(self.target_3d)

    def get_file_indices_for_rank(self, pp_index, tp_index, dp_index):
        assert dp_index < len(self._3d_file_map), f'DP index {dp_index} >= DP degree {len(self._3d_file_map)}'
        dp_2d_map = self._3d_file_map[dp_index]
        return dp_2d_map.get_data(pp_index, tp_index)

    def get_files_for_rank(self, pp_index, tp_index, dp_index):
        file_idx_list = self.get_file_indices_for_rank(pp_index, tp_index, dp_index)
        return [self.file_list[idx] for idx in file_idx_list]

    def get_state_for_rank(self, pp_index, tp_index, dp_index):
        state_file_list = self.get_files_for_rank(pp_index, tp_index, dp_index)
        merged_sd = None
        for state_file in state_file_list:
            sd = torch.load(state_file, map_location=torch.device('cpu'))
            if merged_sd is None:
                merged_sd = sd
            else:
                merged_sd = merge_state(merged_sd, sd)

        return merged_sd

    def print_3d_index_map(self, tag=None):
        if tag:
            print(f'3D index map: {tag}')
        for dp_index, _2d_map in enumerate(self._3d_file_map):
            _2d_map.print_data(f'dp = {dp_index}')

    def print_3d_file_map(self, tag=None):
        if tag:
            print(f'3D file map: {tag}')
        for dp_index, _2d_map in enumerate(self._3d_file_map):
            for pp_index in _2d_map.pp_degree:
                for tp_index in _2d_map.tp_degree:
                    file_index_list = _2d_map.get_data(pp_index, tp_index)
                    file_list = [self.file_list[idx] for idx in file_index_list]
                    print(f'{pp_index}, {tp_index}, {dp_index} => {file_list}')

    def reshape(self, target_3d_desc: model_3d_desc):
        self.target_3d = target_3d_desc
        self._3d_file_map = self.src_3d.reshape(self.target_3d)
