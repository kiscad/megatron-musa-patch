"""Pre-warm the TileLang gated-delta-rule kernel cache for a multi-rank run.

The TileLang GDN kernels JIT-compile on first use, keyed by head count and the
dense/unpadded specialization -- not by sequence length. When the first
training step is also the first use, every rank compiles the same kernels into
the shared ``~/.tilelang`` cache concurrently; those races have crashed runs on
MUSA (device error / SIGABRT). Compiling once in this single process makes the
subsequent multi-rank run hit the cache.

Run with the same interpreter and the same TP degree as the training (the
per-rank head count is ``linear_num_value_heads // tensor_parallel_size``):

    python examples/warm_gdn_tilelang.py --heads 8   # Qwen3.5-9B at TP=4

Large ``--seq`` is not needed: any sequence above one chunk exercises the same
kernels. This script touches no Megatron code and no distributed state.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heads", type=int, default=8, help="per-rank value-head count")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq", type=int, default=256, help="any length >= 128")
    parser.add_argument("--dim", type=int, default=128, help="key/value head dim (64 or 128)")
    args = parser.parse_args()

    from torch_kernels.attention import gated_delta_net

    device = "musa"
    generator = torch.Generator(device=device).manual_seed(0)

    def leaf(*shape, dtype):
        return torch.randn(*shape, generator=generator, device=device,
                           dtype=dtype, requires_grad=True)

    # The model L2-normalizes q/k before the kernel; unnormalized keys make
    # the delta rule diverge on synthetic inputs, so mirror the model here.
    q, k, v = (leaf(args.batch, args.seq, args.heads, args.dim, dtype=torch.bfloat16)
               for _ in range(3))
    q = F.normalize(q.float(), dim=-1).to(torch.bfloat16).detach().requires_grad_(True)
    k = F.normalize(k.float(), dim=-1).to(torch.bfloat16).detach().requires_grad_(True)
    g = (-torch.rand(args.batch, args.seq, args.heads, generator=generator,
                     device=device, dtype=torch.float32) * 2.0).requires_grad_(True)
    beta = torch.rand(args.batch, args.seq, args.heads, generator=generator,
                      device=device, dtype=torch.bfloat16, requires_grad=True)

    out, _ = gated_delta_net(q, k, v, g, beta, backend="tilelang")
    out.float().pow(2).mean().backward()
    torch.musa.synchronize()
    assert all(torch.isfinite(t.grad).all() for t in (q, k, v, g, beta))
    print(f"warm_gdn_tilelang: cached B={args.batch} S={args.seq} H={args.heads} "
          f"D={args.dim} (forward + backward)")


if __name__ == "__main__":
    main()
