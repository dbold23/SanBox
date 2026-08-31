"""Supervised metric fine-tuning for the Melops ablation -- the decisive run-2 arm.

Run 1 (see ``results/ANALYSIS.md``) measured zero-shot embeddings at 1-2
Rank-1 points, ~20x below the 15-point decision floor, so the head-vs-headless
verdict was INCONCLUSIVE. This module implements the fix the analysis calls
for: fine-tune a timm backbone with an ArcFace head on an identity-disjoint
60% of the corpus, then evaluate the same four-arm ablation on the held-out
40% through the UNCHANGED protocol.

Contract
--------
* ``split_identities(df, train_frac, seed)`` splits by **identity** (both
  flanks of one fish travel together -- an eval fish must be wholly unseen by
  training) with ZERO identity overlap, enforced by ``ProtocolViolation``.
  The eval frame is then fed to the existing
  ``protocol.one_shot_open_set_split`` unchanged; fine-tuning never sees an
  eval identity.
* Training classes are ``(identity, side)`` units, matching the protocol's
  enrollment unit: the two flanks of one fish are separate classes.
* ``FinetunedEmbedder`` loads a saved checkpoint and exposes the same
  ``embed(items) -> (n, d) L2-normalized float64`` interface as
  ``embedders.py``. The runner reaches it through the backbone spec
  ``finetuned:CHECKPOINT_PATH`` (``get_embedder_from_spec``; the factory in
  ``embedders.py`` recognizes the prefix via a guarded lazy import).
* Every stochastic step (split, torch init, batch order, augmentation) is
  seeded; CPU runs are bit-deterministic.

torch/timm are optional at import time (same convention as ``embedders.py``):
the module imports cleanly without them, and any function that needs them
raises a RuntimeError naming the exact pip line.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image

import embedders
import melops_data
from protocol import ProtocolViolation

try:  # optional deps, embedders.py convention: import-guarded, never required
    import timm
    import torch
    import torch.nn.functional as F
    from torch import nn
except ImportError:  # pragma: no cover - exercised only on torch-less machines
    timm = None
    torch = None
    F = None
    nn = None

DEFAULT_BACKBONE = "hf-hub:BVRA/MegaDescriptor-L-384"
FINETUNED_PREFIX = "finetuned:"
_CKPT_FORMAT = "melops-finetune-v1"

_TORCH_HINT = (
    "requires torch + timm. On the training box run:\n"
    "  pip install torch torchvision timm\n"
    "(this sandbox has them installed but huggingface.co egress-blocked, so "
    "pretrained weights cannot download here -- use pretrained=False tiny "
    "models for tests)."
)

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
_IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def _require_torch(who):
    if torch is None or timm is None:
        raise RuntimeError("%s %s" % (who, _TORCH_HINT))


def _resolve_device(device):
    """``None``/``"auto"`` -> cuda if available else cpu; else pass through."""
    if device in (None, "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _seed_everything(seed):
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():  # pragma: no cover - CPU-only sandbox
        torch.cuda.manual_seed_all(int(seed))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Identity-disjoint split
# ---------------------------------------------------------------------------


def load_train_identities(ckpt_path):
    """Identity strings a checkpoint was trained on, read from the checkpoint.

    Self-contained on purpose: run_ablation enforces the eval split from the
    checkpoint itself, so losing split_identities.json cannot silently turn
    an eval into a leaked one. Returns a set of identity strings.
    """
    import torch

    payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    classes = payload.get("classes") or []
    return {str(c[0]) for c in classes if isinstance(c, (list, tuple)) and len(c) > 0}


def split_identities(df, train_frac=0.6, seed=0):
    """Identity-disjoint train/eval split of a catalogue frame.

    Splits on ``identity`` alone -- both flanks (sides) of one fish travel to
    the same partition, so an eval identity is wholly unseen by training.
    ``train_frac`` of the unique identities (rounded, clamped so both sides
    are non-empty) go to train; the rest to eval. Deterministic under
    ``seed``. Zero identity overlap is enforced with ``ProtocolViolation``,
    not assumed.

    Returns ``(train_df, eval_df)``. The eval frame is what run 2 feeds to
    ``protocol.one_shot_open_set_split`` -- unchanged.
    """
    if "identity" not in df.columns:
        raise ProtocolViolation("input frame missing column 'identity'")
    if not (0.0 < float(train_frac) < 1.0):
        raise ValueError("train_frac must be in (0, 1), got %r" % (train_frac,))
    if df["identity"].isna().any():
        n_bad = int(df["identity"].isna().sum())
        raise ProtocolViolation(
            "%d rows have null identity; fix the metadata before splitting" % n_bad
        )
    # The same pixels under two rows would let training see eval imagery once
    # the rows land on opposite sides of the split -- and because the split
    # separates the frames, protocol.py's own duplicate guards can no longer
    # catch it downstream. Reject here, on the WHOLE frame.
    for col in ("path", "image_id"):
        if col in df.columns and df[col].duplicated().any():
            dups = df.loc[df[col].duplicated(), col].astype(str).tolist()[:3]
            raise ProtocolViolation(
                "duplicate %s values in catalogue (first: %r); the same image "
                "must not appear twice or it can leak across the split" % (col, dups)
            )
    identities = np.array(sorted(df["identity"].astype(str).unique()))
    if len(identities) < 2:
        raise ProtocolViolation(
            "need >= 2 identities to split, got %d" % len(identities)
        )
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(len(identities))
    n_train = int(round(float(train_frac) * len(identities)))
    n_train = min(max(n_train, 1), len(identities) - 1)
    train_ids = set(identities[perm[:n_train]].tolist())
    eval_ids = set(identities[perm[n_train:]].tolist())
    overlap = train_ids & eval_ids
    if overlap:  # unreachable by construction; enforced anyway (contract)
        raise ProtocolViolation(
            "identity leaked into both splits: %r" % sorted(overlap)[:3]
        )
    id_col = df["identity"].astype(str)
    train_df = df[id_col.isin(train_ids)].reset_index(drop=True)
    eval_df = df[id_col.isin(eval_ids)].reset_index(drop=True)
    got_overlap = set(train_df["identity"].astype(str)) & set(
        eval_df["identity"].astype(str)
    )
    if got_overlap:
        raise ProtocolViolation(
            "identity overlap after materialization: %r" % sorted(got_overlap)[:3]
        )
    return train_df, eval_df


# ---------------------------------------------------------------------------
# ArcFace head
# ---------------------------------------------------------------------------

if nn is not None:

    class ArcMarginProduct(nn.Module):
        """ArcFace additive angular margin logits.

        Standard formulation (Deng et al., "ArcFace: Additive Angular Margin
        Loss for Deep Face Recognition", CVPR 2019, arXiv:1801.07698): with
        L2-normalized feature x and L2-normalized class weights W_j,
        ``cos(theta_j) = <x, W_j>``; the target-class logit becomes
        ``cos(theta_y + m)`` (additive angular margin ``m``), every logit is
        scaled by ``s``, and cross-entropy is applied to the result. Defaults
        ``m=0.5``, ``s=30`` are the paper's values. The usual numerical guard
        applies: where ``theta + m`` would pass pi (cosine no longer
        monotonic), the margin falls back to ``cos(theta) - m*sin(m)``.
        """

        def __init__(self, in_features, n_classes, s=30.0, m=0.5):
            super().__init__()
            self.s = float(s)
            self.m = float(m)
            self.weight = nn.Parameter(torch.empty(int(n_classes), int(in_features)))
            nn.init.xavier_uniform_(self.weight)
            self.cos_m = float(np.cos(self.m))
            self.sin_m = float(np.sin(self.m))
            self.th = float(np.cos(np.pi - self.m))
            self.mm = float(np.sin(np.pi - self.m) * self.m)

        def forward(self, features, labels):
            cosine = F.linear(F.normalize(features), F.normalize(self.weight))
            sine = torch.sqrt(torch.clamp(1.0 - cosine * cosine, min=1e-7))
            phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
            one_hot = F.one_hot(labels, num_classes=self.weight.shape[0])
            one_hot = one_hot.to(dtype=cosine.dtype)
            return self.s * (one_hot * phi + (1.0 - one_hot) * cosine)


# ---------------------------------------------------------------------------
# Preprocessing / augmentation
# ---------------------------------------------------------------------------


def _preprocess(img, img_size):
    """PIL RGB -> (3, H, W) float64 CHW array, ImageNet-normalized.

    Matches ``embedders._TimmEmbedder`` exactly so fine-tuned and zero-shot
    arms see identical preprocessing.
    """
    img = img.resize((int(img_size), int(img_size)), Image.BILINEAR)
    arr = (np.asarray(img, dtype=np.float64) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
    return arr.transpose(2, 0, 1)


def _augment(img, rng, hflip=False):
    """Mild, fully seeded train-time augmentation on the PIL crop.

    Horizontal flip is OFF by default and must stay off for Melops-like data:
    left and right flanks are SEPARATE identity units in this protocol, and
    mirroring a left flank manufactures a fake right flank -- it would alias
    the two flank classes at training time.
    """
    angle = float(rng.uniform(-3.0, 3.0))
    tx = int(rng.integers(-2, 3))
    ty = int(rng.integers(-2, 3))
    img = img.rotate(angle, resample=Image.BILINEAR, translate=(tx, ty))
    if hflip and rng.random() < 0.5:  # deliberately not the default; see above
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    arr = np.asarray(img, dtype=np.float64) * float(rng.uniform(0.9, 1.1))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _class_units(train_df):
    """Sorted (identity, side) units and per-row integer labels."""
    pairs = list(
        zip(train_df["identity"].astype(str), train_df["side"].astype(str))
    )
    units = sorted(set(pairs))
    index = {u: i for i, u in enumerate(units)}
    labels = np.array([index[p] for p in pairs], dtype=np.int64)
    return units, labels


def _apply_freeze(model, freeze_backbone, unfreeze_last_n):
    """Freeze policy over the backbone's parameter tensors.

    ``freeze_backbone`` freezes everything; ``unfreeze_last_n`` freezes
    everything except the LAST N parameter tensors in the module's parameter
    order (a deliberately simple, architecture-agnostic proxy for "the last
    blocks"; timm modules list parameters input-to-output).
    """
    if not freeze_backbone and unfreeze_last_n is None:
        return
    params = list(model.parameters())
    for p in params:
        p.requires_grad = False
    if unfreeze_last_n is not None:
        n = int(unfreeze_last_n)
        if n < 0:
            raise ValueError("unfreeze_last_n must be >= 0, got %d" % n)
        if n > 0:
            for p in params[-n:]:
                p.requires_grad = True


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_finetune(
    root,
    out_dir,
    backbone=DEFAULT_BACKBONE,
    bbox="body",
    train_frac=0.6,
    epochs=10,
    batch_size=64,
    lr=1e-4,
    backbone_lr=None,
    img_size=384,
    freeze_backbone=False,
    unfreeze_last_n=None,
    pretrained=True,
    hflip=False,
    device=None,
    seed=0,
    arc_s=30.0,
    arc_m=0.5,
):
    """ArcFace fine-tuning over an identity-disjoint train split.

    Writes into ``out_dir``: ``checkpoint.pt`` (backbone + head state),
    ``train_log.json`` (config + per-epoch mean loss) and
    ``split_identities.json`` (the exact train/eval identity lists, so the
    eval run can prove disjointness later). ``backbone_lr`` defaults to
    ``lr * 0.1`` (head learns fast, unfrozen backbone slowly).

    ``epochs=0`` saves the untouched (random-init or pretrained) backbone --
    the tests use this as the same-seed, same-architecture baseline.

    Returns a dict with the artifact paths, the split frames and the log.
    """
    _require_torch("train_finetune")
    device = _resolve_device(device)
    _seed_everything(seed)

    df = melops_data.load_melops(root, bbox=bbox)
    train_df, eval_df = split_identities(df, train_frac=train_frac, seed=seed)
    units, labels = _class_units(train_df)

    model = timm.create_model(backbone, pretrained=bool(pretrained), num_classes=0)
    _apply_freeze(model, freeze_backbone, unfreeze_last_n)
    head = ArcMarginProduct(model.num_features, len(units), s=arc_s, m=arc_m)
    model.to(device)
    head.to(device)

    if backbone_lr is None:
        backbone_lr = float(lr) * 0.1
    trainable = [p for p in model.parameters() if p.requires_grad]
    groups = [{"params": list(head.parameters()), "lr": float(lr)}]
    if trainable:
        groups.append({"params": trainable, "lr": float(backbone_lr)})
    optimizer = torch.optim.Adam(groups)

    n = len(train_df)
    if n == 0:
        raise ProtocolViolation("empty train split")
    model.train()
    head.train()
    epoch_log = []
    t0 = time.time()
    for epoch in range(int(epochs)):
        rng = np.random.default_rng([int(seed), 1000 + epoch])
        order = rng.permutation(n)
        losses = []
        for start in range(0, n, int(batch_size)):
            idx = order[start : start + int(batch_size)]
            batch = []
            for i in idx:
                img = melops_data.load_crop(root, train_df.iloc[int(i)])
                img = _augment(img, rng, hflip=hflip)
                batch.append(_preprocess(img, img_size))
            x = torch.from_numpy(np.stack(batch)).float().to(device)
            y = torch.from_numpy(labels[idx]).to(device)
            logits = head(model(x), y)
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        epoch_log.append(
            {
                "epoch": epoch,
                "mean_loss": float(np.mean(losses)),
                "n_batches": len(losses),
            }
        )

    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "checkpoint.pt")
    config = {
        "backbone": backbone,
        "bbox": bbox,
        "train_frac": float(train_frac),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "backbone_lr": float(backbone_lr),
        "img_size": int(img_size),
        "freeze_backbone": bool(freeze_backbone),
        "unfreeze_last_n": None if unfreeze_last_n is None else int(unfreeze_last_n),
        "pretrained": bool(pretrained),
        "hflip": bool(hflip),
        "device": device,
        "seed": int(seed),
        "arc_s": float(arc_s),
        "arc_m": float(arc_m),
        "n_train_images": int(n),
        "n_train_classes": len(units),
    }
    model.eval()
    torch.save(
        {
            "format": _CKPT_FORMAT,
            "backbone": backbone,
            "img_size": int(img_size),
            "num_features": int(model.num_features),
            "backbone_state_dict": model.state_dict(),
            "head_state_dict": head.state_dict(),
            "classes": [list(u) for u in units],
            "config": config,
        },
        ckpt_path,
    )
    log_path = os.path.join(out_dir, "train_log.json")
    with open(log_path, "w") as f:
        json.dump(
            {"config": config, "epochs": epoch_log, "elapsed_s": round(time.time() - t0, 2)},
            f,
            indent=2,
        )
    split_path = os.path.join(out_dir, "split_identities.json")
    with open(split_path, "w") as f:
        json.dump(
            {
                "train_identities": sorted(set(train_df["identity"].astype(str))),
                "eval_identities": sorted(set(eval_df["identity"].astype(str))),
            },
            f,
            indent=2,
        )
    return {
        "checkpoint": ckpt_path,
        "train_log": log_path,
        "split": split_path,
        "train_df": train_df,
        "eval_df": eval_df,
        "epochs": epoch_log,
    }


# ---------------------------------------------------------------------------
# Evaluation bridge (the Embedder-protocol side)
# ---------------------------------------------------------------------------


class FinetunedEmbedder:
    """Loads a ``train_finetune`` checkpoint; same interface as embedders.py.

    ``embed(items)`` (PIL images or paths) returns an ``(n, d)`` float64
    array with L2-normalized rows, so ``protocol.evaluate`` and the runner
    consume it exactly like any built-in embedder.
    """

    def __init__(self, checkpoint_path, device=None, batch_size=8):
        _require_torch("FinetunedEmbedder")
        if not os.path.exists(checkpoint_path):
            raise ValueError("checkpoint %r does not exist" % (checkpoint_path,))
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if ckpt.get("format") != _CKPT_FORMAT:
            raise ValueError(
                "%r is not a %s checkpoint" % (checkpoint_path, _CKPT_FORMAT)
            )
        self.device = _resolve_device(device)
        self.batch_size = int(batch_size)
        self.img_size = int(ckpt["img_size"])
        self.backbone_name = ckpt["backbone"]
        self.model = timm.create_model(
            self.backbone_name, pretrained=False, num_classes=0
        )
        self.model.load_state_dict(ckpt["backbone_state_dict"])
        self.model.eval().to(self.device)

    def fit(self, items):
        pass

    def embed(self, items):
        out = []
        with torch.no_grad():
            for start in range(0, len(items), self.batch_size):
                batch = [
                    _preprocess(embedders._as_image(item), self.img_size)
                    for item in items[start : start + self.batch_size]
                ]
                tensor = torch.from_numpy(np.stack(batch)).float().to(self.device)
                out.append(self.model(tensor).cpu().numpy())
        return embedders._l2_normalize(np.concatenate(out).astype(np.float64))


def get_embedder_from_spec(spec, seed=0, device=None, batch_size=8):
    """Backbone-spec dispatch for the runner.

    ``finetuned:CHECKPOINT_PATH`` -> ``FinetunedEmbedder(CHECKPOINT_PATH)``;
    any other spec falls through to ``embedders.get_embedder`` unchanged, so
    the runner can treat every backbone string uniformly:

        python run_ablation.py ... --backbone finetuned:/path/to/checkpoint.pt
    """
    if spec.startswith(FINETUNED_PREFIX):
        path = spec[len(FINETUNED_PREFIX) :]
        return FinetunedEmbedder(path, device=device, batch_size=batch_size)
    return embedders.get_embedder(spec, seed=seed)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="corpus root directory")
    parser.add_argument("--out", required=True, help="output dir for checkpoint + logs")
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE,
                        help="timm model string (default: %s)" % DEFAULT_BACKBONE)
    parser.add_argument("--bbox", default="body", choices=melops_data.BBOX_PARTS,
                        help="crop the training images are taken from")
    parser.add_argument("--train-frac", type=float, default=0.6)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4, help="head learning rate")
    parser.add_argument("--backbone-lr", type=float, default=None,
                        help="unfrozen-backbone learning rate (default lr * 0.1)")
    parser.add_argument("--img-size", type=int, default=384)
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="train only the ArcFace head")
    parser.add_argument("--unfreeze-last-N", dest="unfreeze_last_n", type=int,
                        default=None,
                        help="freeze all but the last N backbone parameter tensors")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="random-init backbone (tests / egress-blocked boxes)")
    parser.add_argument("--hflip", action="store_true",
                        help="enable horizontal flip augmentation. OFF by default: "
                             "sides are separate identities; mirroring aliases flanks")
    parser.add_argument("--device", default=None, help="auto: cuda if available else cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    result = train_finetune(
        args.root,
        args.out,
        backbone=args.backbone,
        bbox=args.bbox,
        train_frac=args.train_frac,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        backbone_lr=args.backbone_lr,
        img_size=args.img_size,
        freeze_backbone=args.freeze_backbone,
        unfreeze_last_n=args.unfreeze_last_n,
        pretrained=not args.no_pretrained,
        hflip=args.hflip,
        device=args.device,
        seed=args.seed,
    )
    for entry in result["epochs"]:
        print("epoch %2d  mean ArcFace loss %.4f  (%d batches)"
              % (entry["epoch"], entry["mean_loss"], entry["n_batches"]))
    print("checkpoint: %s" % result["checkpoint"])
    print("train log:  %s" % result["train_log"])
    print("split:      %s" % result["split"])
    print("evaluate with: python run_ablation.py ... --backbone finetuned:%s"
          % result["checkpoint"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
