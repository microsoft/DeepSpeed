# Checkpoint Engine


The `CheckpointEngine` was designed to modularize the checkpoint serialization. In this way, we can simply replace/refine the checkpoint serialization methods.

### Interface for `CheckpointEngine`

Basically, for checkpoint management(save/load by deepspeed with the given tag), the `CheckpointEngine` will:

	1. Make preliminaries ready by calling `create(tag)`. For `torch`, we can just log some extra info as `torch` can directly call `save/load` without other preparation.

	2. After the `create(tag)`, deepspeed can call `save/load` to persist files into disk/memory/etc.

	3. When all the files for a tag are ready, deepspeed engine will call `commit()` to tell the checkpoint engine current checkpoint is complete. For original torch, it also plays the role of logger.


```python
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

```
