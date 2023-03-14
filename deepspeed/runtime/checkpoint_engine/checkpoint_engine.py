# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os


class CheckpointEngine(object):

    # init checkpoint engine for save/load
    def __init__(self, config_params=None):
        pass

    def create(self, save_dir, tag):
        # create checkpoint on give tag for save/load.
        pass

    def makedirs(self, path, exist_ok=False):
        os.makedirs(path, exist_ok=exist_ok)

    def save(self, state_dict, path: str):
        pass

    def commit(self, tag):
        # to tell checkpoint services if all files are ready.
        pass

    def open(self, load_dir, tag=None):
        # The open() function can be used by the checkpoint engine to find
        # and prepare a checkpoint for reading. The caller must specify
        # a directory in load_dir and a checkpoint name in tag. If
        # tag == None or "latest", the checkpoint engine loads the most
        # recent checkpoint that it can find. Otherwise, the checkpoint
        # engine attempts to load the checkpoint named in tag.
        #
        # open() returns the tag value of the checkpoint that it actually
        # loaded or None if it fails to find a checkpoint.
        pass

    def load(self, path: str, map_location=None):
        # Reads and returns data from a checkpoint file.
        # Must be called between open() and close().
        pass

    def close(self, tag):
        # Must be called after loading all checkpoint files.
        # Can be used by checkpoint engine to free resources it
        # may have allocated during open().
        pass
