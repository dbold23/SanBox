"""Compact CNN encoder mapping a spot patch to an L2-normalized embedding.

Sized to train usefully on CPU in minutes; scale ``width``/``embed_dim``
up when training on GPU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.GroupNorm(min(8, cout), cout),
        nn.SiLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.GroupNorm(min(8, cout), cout),
        nn.SiLU(inplace=True),
    )


class SpotEncoder(nn.Module):
    def __init__(self, embed_dim: int = 128, width: int = 32):
        super().__init__()
        w = width
        self.stem = nn.Sequential(
            nn.Conv2d(1, w, 5, stride=2, padding=2, bias=False),
            nn.GroupNorm(min(8, w), w),
            nn.SiLU(inplace=True),
        )
        self.stages = nn.ModuleList()
        chans = [w, 2 * w, 4 * w, 8 * w]
        cin = w
        for cout in chans:
            self.stages.append(nn.Sequential(
                _block(cin, cout),
                nn.MaxPool2d(2),
            ))
            cin = cout
        # Mean+std pooling and a LayerNorm keep initial embeddings spread
        # out; plain mean-pool + linear collapses at init (all inputs map
        # to nearly the same vector, cos ~ 0.99) and training stalls.
        self.post = nn.LayerNorm(2 * cin)
        self.head = nn.Linear(2 * cin, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        z = torch.cat([x.mean(dim=(2, 3)), x.std(dim=(2, 3))], dim=1)
        return F.normalize(self.head(self.post(z)), dim=1)


def supcon_loss(emb: torch.Tensor, labels: torch.Tensor,
                temperature: float = 0.1) -> torch.Tensor:
    """Supervised contrastive (multi-positive NT-Xent) loss on
    L2-normalized embeddings."""
    sim = emb @ emb.T / temperature
    n = emb.shape[0]
    eye = torch.eye(n, dtype=torch.bool, device=emb.device)
    sim = sim.masked_fill(eye, float("-inf"))
    pos = labels[:, None].eq(labels[None, :]) & ~eye
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    n_pos = pos.sum(dim=1).clamp(min=1)
    loss = -(log_prob.masked_fill(~pos, 0.0).sum(dim=1) / n_pos)
    return loss[pos.any(dim=1)].mean()
