import threading
import queue
import time


class AsyncWorker(threading.Thread):
    def __init__(self, dataloaders, dataset_picker):
        threading.Thread.__init__(self)
        self.req_queue = queue.Queue()
        self.ret_queue = queue.Queue()
        self.dataloaders = dataloaders
        self.dataset_picker = dataset_picker
        self.prefetch_idx = 3
        for i in range(self.prefetch_idx):
            self.req_queue.put(dataset_picker[i])

    def run(self):
        while True:
            dataset_type = self.req_queue.get(block=True)
            if dataset_type is None:
                break
            batch = next(self.dataloaders[dataset_type])
            self.req_queue.task_done()
            self.ret_queue.put(batch)

    def get(self):
        batch = self.ret_queue.get()
        self.ret_queue.task_done()
        return batch

    def prefetch(self):
        if self.prefetch_idx < len(self.dataset_picker):
            self.req_queue.put(self.dataset_picker[self.prefetch_idx])
            self.prefetch_idx += 1

    def stop(self):
        self.req_queue.put(None)
