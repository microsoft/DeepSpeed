'''
Copyright 2020 The Microsoft DeepSpeed Team
'''

import torch.distributed as dist


class NcclBackend(object):
    def __init__(self, group, size, rank):
        self.world_group = dist.new_group(ranks=range(dist.get_world_size()))
        self.rank = dist.get_rank(group=self.world_group)
        self.size = dist.get_world_size(group=self.world_group)

    def my_igather(self, rank, size, group, sendbuf, recvbuf, root):
        req = []
        if rank == root:
            for idx in range(size):
                if idx != rank:
                    req.append(dist.irecv(recvbuf[idx], src=idx, group=group))
                else:
                    recvbuf[rank] = sendbuf
        else:
            req.append(dist.isend(sendbuf, group=group, dst=root))
        return req

    def my_gather(self, rank, size, group, sendbuf, recvbuf, root):
        if rank == root:
            for idx in range(size):
                if idx != rank:
                    dist.recv(recvbuf[idx], src=idx, group=group)
                else:
                    recvbuf[rank] = sendbuf
        else:
            dist.send(sendbuf, group=group, dst=root)

    def compressed_allreduce(self,
                             buffer_m: torch.tensor,
                             worker_error,
                             server_error,
                             local_rank):

        all_start_time = time.time()
        original_size = buffer_m.numel()
        cupy.cuda.Device(local_rank).use()

        if torch.numel(buffer_m) != torch.numel(worker_error):
            empty_tensor = torch.zeros(torch.numel(worker_error) - torch.numel(buffer_m),
                                       device=buffer_m.device)
            buffer_m = torch.cat([buffer_m, empty_tensor])

        buffer_m.add_(worker_error)
        worker_scale = torch.norm(buffer_m) / np.sqrt(torch.numel(buffer_m))
        sign_buffer_m = buffer_m.sign().add_(1).bool()
        sign_buffer_m = sign_buffer_m.float()
        sign_buffer_m.add_(-0.5).mul_(2.0)
        worker_error.set_((buffer_m - worker_scale * sign_buffer_m))
        sign_buffer_m = None

        compensated_buffer_m = buffer_m
        compensated_buffer_m.sign_()
        compensated_buffer_m = compensated_buffer_m.add_(1).bool()

        cupy_worker_scale = self.torch2cupy(worker_scale)
        cupy_compensated_buffer_m = self.torch2cupy(compensated_buffer_m)
        compensated_buffer_m = None

        cupy_sign_list_packed = self.compress_by_chunk(cupy_compensated_buffer_m,
                                                       self.size)
        cupy_compensated_buffer_m = None

        cupy_recvbuf_sign = cupy.zeros(
            [self.size,
             cupy_sign_list_packed[self.rank].size],
            dtype=cupy_sign_list_packed[0].dtype)
        cupy_recvbuf_scale = cupy.zeros([self.size, 1], dtype=cupy_worker_scale.dtype)

        sign_list_packed = [None] * self.size

        for idx in range(self.size):
            sign_list_packed[idx] = self.cupy2torch(cupy_sign_list_packed[idx])

        recvbuf_sign = self.cupy2torch(cupy_recvbuf_sign)

        worker_scale = self.cupy2torch(cupy_worker_scale)
        recvbuf_scale = self.cupy2torch(cupy_recvbuf_scale)

        # communication phase 1
        gather_start = time.time()
        requests = []
        for idx in range(self.size):
            requests += self.my_igather(self.rank,
                                        self.size,
                                        self.world_group,
                                        sign_list_packed[idx],
                                        recvbuf_sign,
                                        root=idx)
            requests += self.my_igather(self.rank,
                                        self.size,
                                        self.world_group,
                                        worker_scale,
                                        recvbuf_scale,
                                        root=idx)

        for i in range(len(requests)):
            requests[i].wait()

        gather_end = time.time()

        cupy_recvbuf_sign = self.torch2cupy(recvbuf_sign)
        cupy_recvbuf_scale = self.torch2cupy(recvbuf_scale)

        cupy_unpacked_sign = (cupy.unpackbits(cupy_recvbuf_sign.flatten())).reshape(
            self.size,
            -1)
        cupy_recvbuf_sign = None

        unpacked_sign = self.cupy2torch(cupy_unpacked_sign).float()
        cupy_unpacked_sign = None

        unpacked_sign = unpacked_sign.add_(-0.5).mul_(2.0)
        worker_scale = self.cupy2torch(cupy_recvbuf_scale).mul_(1 / self.size)

        compensated_server_m = unpacked_sign.mul_(worker_scale).sum(0)
        unpacked_sign = None

        compensated_server_m.add_(server_error)
        server_scale = torch.norm(compensated_server_m) / np.sqrt(
            compensated_server_m.numel())
        sign_server_m = compensated_server_m.sign().add_(1).bool()
        sign_server_m = sign_server_m.float()
        sign_server_m.add_(-0.5).mul_(2.0)
        server_error.set_(compensated_server_m - server_scale * sign_server_m)
        sign_server_m = None

        compensated_server_m.sign_()
        compensated_server_m = compensated_server_m.add_(1).bool()
        cupy_server_scale = self.torch2cupy(server_scale)
        cupy_compensated_server_m = self.torch2cupy(compensated_server_m)
        compensated_server_m = None

        cupy_server_sign_packed = self.compress_by_chunk(cupy_compensated_server_m, 1)

        cupy_recvbuf_sign_server = cupy.zeros(
            [self.size,
             cupy_server_sign_packed[0].size],
            dtype=cupy_sign_list_packed[0].dtype)

        server_sign_packed = [None] * 1
        recvbuf_sign_server = [None] * self.size

        for idx in range(self.size):
            recvbuf_sign_server[idx] = self.cupy2torch(cupy_recvbuf_sign_server[idx])

        server_sign_packed[0] = self.cupy2torch(cupy_server_sign_packed[0])

        server_scale = self.cupy2torch(cupy_server_scale)
        cupy_recvbuf_scale_server = cupy.zeros([self.size,
                                                1],
                                               dtype=cupy_worker_scale.dtype)

        recvbuf_scale_server = [None] * self.size
        for idx in range(self.size):
            recvbuf_scale_server[idx] = self.cupy2torch(cupy_recvbuf_scale_server[idx])

        # Communication Phase 2
        dist.all_gather(recvbuf_sign_server, server_sign_packed[0])
        dist.all_gather(recvbuf_scale_server, server_scale)

        # need to convert from a tensor list to a single tensor
        # dist.all_gather only provides a tensor list as the recv/output buffer
        recvbuf_sign_server = torch.stack(recvbuf_sign_server)

        cupy_recvbuf_sign_server = self.torch2cupy(recvbuf_sign_server)

        cupy_server_unpacked_sign = (cupy.unpackbits(
            cupy_recvbuf_sign_server.flatten())).reshape(self.size,
                                                         -1)
        cupy_recvbuf_sign_server = None

        server_unpacked_sign = self.cupy2torch(cupy_server_unpacked_sign)
        cupy_server_unpacked_sign = None

        server_unpacked_sign = server_unpacked_sign.float().add_(-0.5).mul_(2.0)
        server_scale = self.cupy2torch(cupy_recvbuf_scale_server)
        buffer_m = server_unpacked_sign.mul_(server_scale).flatten()[0:original_size]

        return buffer_m
