from deepspeed.ops import softmax_dropout
import torch
import time

mse = torch.nn.MSELoss()

cuda_rng_state = torch.cuda.get_rng_state()

g_cuda = torch.Generator(device='cuda')

soft = torch.nn.Softmax(dim=-1)
drop = torch.nn.Dropout(0.0)

torch.cuda.set_rng_state(cuda_rng_state)
for N in [512]:
    d = torch.randn(1024, N, N, requires_grad=True).half().cuda()
    rel_pos = torch.randn(16, 64, N, N, requires_grad=True).half().cuda()
    mask = torch.randint(2, (16, N), requires_grad=False).cuda()
    y = torch.randn(1024, N, N).half().cuda()
    d1 = d.clone()
    rel_pos1 = rel_pos.clone()
    for _ in range(1):
        d = d.view(16, 64, N, N)
        d = (d + rel_pos).masked_fill(
            (1-mask).unsqueeze(1).unsqueeze(2).bool(),
            float("-inf"),
        )
        d = d.view(1024, N, N)
        out = drop(soft(d))
    print(out[0])
   
    torch.cuda.set_rng_state(cuda_rng_state)
    g_cuda = torch.Generator(device='cuda')
    sd = softmax_dropout()

    for _ in range(1):
        out1 = sd(d1, rel_pos1, mask.bool(), 64, ratio=0.0, generator=g_cuda)
    print(out1[0])
    
    print(f"\nmean relative diff: {(out-out1).abs().float().sum()/out.numel()}\nmax relative diff: {(out-out1).abs().max()/out.view(-1)[out.argmax()]}\n")

#print(out1)
#print((out-out1).abs().sum())
