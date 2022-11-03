import torch


def orthogonalize(matrix: torch.Tensor, eps=torch.tensor(1e-16)):
    if torch.distributed.get_rank() == 0:
        print("orthogonalize, matrix.shape", matrix.shape)
        print("orthogonalize, matrix.dtype", matrix.dtype)
        print("orthogonalize, matrix.device", matrix.device)
    
    if matrix.shape[-1] == 1:
        matrix.div_(torch.maximum(matrix.norm(), eps))
    else:
        Q, R = torch.linalg.qr(matrix)
        if torch.distributed.get_rank() == 0:
            print("orthogonalize Q, matrix.shape", matrix.shape)
            print("orthogonalize Q, matrix.device", Q.device)
        matrix.copy_(Q)
        # matrix.copy_(torch.linalg.qr(matrix).Q)
