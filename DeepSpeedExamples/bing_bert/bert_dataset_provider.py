class BertDatasetProviderInterface:
    def get_shard(self, index, shuffle=True):
        raise NotImplementedError

    def release_shard(self, index):
        raise NotImplementedError

    def prefetch_shard(self, index):
        raise NotImplementedError

    def get_batch(self, batch_iter):
        raise NotImplementedError

    def prefetch_batch(self):
        raise NotImplementedError
