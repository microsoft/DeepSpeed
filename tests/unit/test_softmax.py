from deepspeed.ops import softmax_dropout
import torch
import time

mse = torch.nn.MSELoss()

cuda_rng_state = torch.cuda.get_rng_state()

g_cuda = torch.Generator(device='cuda')

soft = torch.nn.Softmax(dim=-1)
drop = torch.nn.Dropout(0.1)

torch.cuda.set_rng_state(cuda_rng_state)
for N in [40, 48, 64, 122, 260, 512]:
    d = torch.randn(1024, N, N, requires_grad=True).half().cuda()
    rel_pos = torch.randn(16, 64, N, N, requires_grad=True).half().cuda()
    mask = torch.randint(2, (16, N), dtype=torch.bool, requires_grad=False).cuda()
    y = torch.randn(1024, N, N).half().cuda()
    d1 = d.clone()
    rel_pos1 = rel_pos.clone()
    for _ in range(10):
        d = d.view(16, 64, N, N)
        d = (d + rel_pos).masked_fill(
            mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
            float("-inf"),
        )
        d = d.view(1024, N, N)
        out = drop(soft(d))
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(1000):
        d = d.view(16, 64, N, N)
        d = (d + rel_pos).masked_fill(
            mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
            float("-inf"),
        )
        d = d.view(1024, N, N)
        out = drop(soft(d))
    torch.cuda.synchronize()
    bs_fwd = (time.time() - t0)
    out.retain_grad()
    d.retain_grad()
    rel_pos.retain_grad()
    for i in range(10):
        d = torch.empty_like(d, requires_grad=True)
        d = d.view(16, 64, N, N)
        d = (d + rel_pos).masked_fill(
            mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
            float("-inf"),
        )
        d = d.view(1024, N, N)
        out = drop(soft(d))
        out.retain_grad()
        #rel_pos.retain_grad()
        #d.retain_grad()
        loss = mse(out, y)
        loss.backward()
    for _ in range(100):
        d = torch.empty_like(d, requires_grad=True)
        d = d.view(16, 64, N, N)
        d = (d + rel_pos).masked_fill(
            mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
            float("-inf"),
        )
        d = d.view(1024, N, N)
        out = drop(soft(d))
        out.retain_grad()
        d.retain_grad()
        rel_pos.retain_grad()
        loss = mse(out, y)
        loss.backward()

    torch.cuda.synchronize()
    bs_bwd = (time.time() - t0)
    #print(out)

    torch.cuda.set_rng_state(cuda_rng_state)
    g_cuda = torch.Generator(device='cuda')
    sd = softmax_dropout()

    for _ in range(10):
        out1 = sd(d1, rel_pos1, mask, 64, ratio=0.1, generator=g_cuda)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(1000):
        out1 = sd(d1, rel_pos1, mask, 64, ratio=0.1, generator=g_cuda)
    torch.cuda.synchronize()
    ds_fwd = ((time.time() - t0))
    out1.retain_grad()
    d1.retain_grad()
    for _ in range(10):
        out1 = sd(d1, rel_pos1, mask, 64, ratio=0.1, generator=g_cuda)
        out1.retain_grad()
        d1.retain_grad()
        rel_pos.retain_grad()
        loss = mse(out1, y)
        loss.backward()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100):
        out1 = sd(d1, rel_pos1, mask, 64, ratio=0.1, generator=g_cuda)
        out1.retain_grad()
        d1.retain_grad()
        rel_pos.retain_grad()
        loss = mse(out1, y)
        loss.backward()
    torch.cuda.synchronize()
    ds_bwd = ((time.time() - t0))
    print(f"{N}: {bs_fwd} -> {ds_fwd} \t {bs_bwd} -> {ds_bwd}")

#print(out1)
#print((out-out1).abs().sum())
