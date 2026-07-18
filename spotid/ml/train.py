"""Train the spot embedding encoder on synthetic data.

Usage (CPU proof of concept):
    python -m spotid.ml.train --steps 1200 --out spotid/ml/checkpoints/encoder.pt

Scaling up on a GPU box: same command with --device cuda --width 64
--embed-dim 256 --steps 20000 --ids-per-batch 32.
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from .dataset import make_batch
from .model import SpotEncoder, supcon_loss


def _worker_batch(args):
    seed, n_ids, k_views, id_pool = args
    rng = np.random.default_rng(seed)
    return make_batch(rng, n_ids, k_views, id_pool)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--ids-per-batch", type=int, default=16)
    ap.add_argument("--views-per-id", type=int, default=4)
    ap.add_argument("--id-pool", type=int, default=4000,
                    help="number of distinct training identities")
    ap.add_argument("--embed-dim", type=int, default=128)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="spotid/ml/checkpoints/encoder.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model = SpotEncoder(args.embed_dim, args.width).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"encoder: {n_params/1e6:.2f}M params | device {device}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.steps, eta_min=args.lr * 0.03)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pool = ProcessPoolExecutor(max_workers=args.workers)
    spec = (args.ids_per_batch, args.views_per_id, args.id_pool)
    # Keep a few batches in flight so rendering overlaps with training.
    inflight = [pool.submit(_worker_batch, (args.seed * 777_000 + i, *spec))
                for i in range(args.workers + 1)]
    next_seed = args.seed * 777_000 + len(inflight)

    t0 = time.time()
    ema_loss = ema_acc = None
    for step in range(1, args.steps + 1):
        x_np, y_np = inflight.pop(0).result()
        inflight.append(pool.submit(_worker_batch, (next_seed, *spec)))
        next_seed += 1
        x = torch.from_numpy(x_np).to(device)
        y = torch.from_numpy(y_np).to(device)
        emb = model(x)
        loss = supcon_loss(emb, y, args.temperature)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()

        with torch.no_grad():
            sim = emb @ emb.T
            sim.fill_diagonal_(float("-inf"))
            nn_label = y[sim.argmax(dim=1)]
            acc = (nn_label == y).float().mean().item()
        ema_loss = loss.item() if ema_loss is None else \
            0.95 * ema_loss + 0.05 * loss.item()
        ema_acc = acc if ema_acc is None else 0.95 * ema_acc + 0.05 * acc
        if step % 25 == 0 or step == args.steps:
            rate = step / (time.time() - t0)
            print(f"step {step:5d}/{args.steps}  loss {ema_loss:.4f}  "
                  f"batch-NN acc {ema_acc:.3f}  {rate:.2f} it/s", flush=True)
        if step % 200 == 0 or step == args.steps:
            torch.save({"model": model.state_dict(),
                        "embed_dim": args.embed_dim, "width": args.width,
                        "step": step}, args.out)
    pool.shutdown()
    print(f"saved {args.out} ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
