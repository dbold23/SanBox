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
import pandas as pd
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
    # Prefer the explicit full-train-split record (present from the
    # min_images_per_unit change onward): classes may be a strict subset of
    # the train split when singleton units were dropped from ArcFace, and
    # eval exclusion must cover the whole split.
    recorded = payload.get("train_identities")
    if recorded:
        return {str(i) for i in recorded}
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


def _filter_min_images_per_unit(train_df, min_images_per_unit):
    """Training-side filter: drop (identity, side) units with too few images.

    Singleton ArcFace classes teach nothing and bloat the head (~2.5
    images/identity means there are many). TRAINING-SIDE ONLY by contract:
    callers must keep using the unfiltered frame for ``split_identities.json``
    and the checkpoint's train-identity record, so eval exclusion still
    covers every identity in the train split, dropped units included.

    Returns ``(filtered_df, n_units_dropped, n_rows_dropped)``.
    """
    k = int(min_images_per_unit)
    if k <= 1:
        return train_df, 0, 0
    sizes = train_df.groupby(["identity", "side"])["image_id"].transform("size")
    kept = sizes >= k
    n_rows_dropped = int((~kept).sum())
    before = train_df.groupby(["identity", "side"]).ngroups
    filtered = train_df[kept].reset_index(drop=True)
    after = filtered.groupby(["identity", "side"]).ngroups if len(filtered) else 0
    if len(filtered) == 0:
        raise ProtocolViolation(
            "min_images_per_unit=%d removed every training row" % k
        )
    return filtered, before - after, n_rows_dropped


def _build_probe(train_fit_df, probe_units):
    """Fixed early-stopping probe, train-side only (run 3 Leg B).

    Deterministic: for up to ``probe_units`` sorted (identity, side) units
    with >= 2 images, gallery = the unit's earliest image, query = its
    latest. Never touches eval identities, so early stopping cannot select
    a checkpoint on the held-out split.
    Returns ``(gallery_rows, query_rows)`` as lists of catalogue rows;
    gallery_rows[i] and query_rows[i] belong to the same unit.
    """
    df = train_fit_df.copy()
    df["_date"] = pd.to_datetime(df["date"])
    gallery_rows, query_rows = [], []
    for _unit, group in df.groupby(["identity", "side"], sort=True):
        if len(group) < 2:
            continue
        group = group.sort_values(["_date", "image_id"])
        gallery_rows.append(group.iloc[0])
        query_rows.append(group.iloc[-1])
        if len(gallery_rows) >= int(probe_units):
            break
    if not gallery_rows:
        raise ProtocolViolation("no multi-image units available for the probe")
    return gallery_rows, query_rows


def _probe_rank1(model, probe, root, img_size, device, autocast_factory, pool):
    """Rank-1 of the probe queries against the probe gallery (cosine)."""
    gallery_rows, query_rows = probe
    model.eval()
    feats = []
    with torch.no_grad():
        for rows in (gallery_rows, query_rows):
            out = []
            for start in range(0, len(rows), 32):
                chunk = rows[start : start + 32]
                crops = list(pool.map(
                    lambda r: melops_data.load_crop(root, r), chunk))
                batch = np.stack([_preprocess(c, img_size) for c in crops])
                x = torch.from_numpy(batch).float().to(device)
                with autocast_factory():
                    f = model(x)
                out.append(f.float().cpu().numpy())
            m = np.concatenate(out).astype(np.float64)
            norms = np.linalg.norm(m, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            feats.append(m / norms)
    model.train()
    sims = feats[1] @ feats[0].T
    return float((sims.argmax(axis=1) == np.arange(len(query_rows))).mean())


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
    min_images_per_unit=1,
    bf16=False,
    grad_checkpointing=False,
    grad_accum=1,
    early_stop_patience=None,
    probe_units=500,
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
    # Training-side only: ArcFace classes and batches come from the filtered
    # frame; split_identities.json and the checkpoint's train-identity record
    # stay on the FULL train_df so eval exclusion is unaffected.
    train_fit_df, n_units_dropped, n_rows_dropped = _filter_min_images_per_unit(
        train_df, min_images_per_unit
    )
    if n_units_dropped:
        print(
            "min_images_per_unit=%d: dropped %d singleton-ish units (%d rows) "
            "from ArcFace classes; %d units remain. Eval exclusion still "
            "covers all %d train identities."
            % (
                int(min_images_per_unit),
                n_units_dropped,
                n_rows_dropped,
                train_fit_df.groupby(["identity", "side"]).ngroups,
                train_df["identity"].nunique(),
            )
        )
    units, labels = _class_units(train_fit_df)

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

    n = len(train_fit_df)
    if n == 0:
        raise ProtocolViolation("empty train split")

    if grad_checkpointing:
        try:
            model.set_grad_checkpointing(True)
            print("gradient checkpointing ON")
        except Exception as exc:  # timm models without the hook
            print("gradient checkpointing unavailable (%s); continuing without" % exc)
    if bf16:
        print("bf16 autocast ON")
    accum = max(1, int(grad_accum))
    if accum > 1:
        print("gradient accumulation x%d (effective batch %d)"
              % (accum, accum * int(batch_size)))

    probe = None
    if early_stop_patience is not None:
        probe = _build_probe(train_fit_df, probe_units)
        print("early stopping ON: patience %d on probe Rank-1 (%d probe units)"
              % (int(early_stop_patience), len(probe[0])))

    from concurrent.futures import ThreadPoolExecutor

    def _autocast():
        if bf16:
            return torch.autocast(device_type="cuda" if "cuda" in str(device) else "cpu",
                                  dtype=torch.bfloat16)
        import contextlib
        return contextlib.nullcontext()

    model.train()
    head.train()
    epoch_log = []
    best = {"rank1": -1.0, "epoch": None, "model": None, "head": None}
    bad_epochs = 0
    early_stopped = False
    t0 = time.time()
    pool = ThreadPoolExecutor(max_workers=16)
    for epoch in range(int(epochs)):
        rng = np.random.default_rng([int(seed), 1000 + epoch])
        order = rng.permutation(n)
        losses = []
        optimizer.zero_grad()
        pending = False
        for step_i, start in enumerate(range(0, n, int(batch_size))):
            idx = order[start : start + int(batch_size)]
            rows = [train_fit_df.iloc[int(i)] for i in idx]
            # loads consume no rng, so threading them preserves the exact
            # augmentation stream of the serial loop
            crops = list(pool.map(lambda r: melops_data.load_crop(root, r), rows))
            batch = [_preprocess(_augment(img, rng, hflip=hflip), img_size)
                     for img in crops]
            x = torch.from_numpy(np.stack(batch)).float().to(device)
            y = torch.from_numpy(labels[idx]).to(device)
            with _autocast():
                logits = head(model(x), y)
                loss = F.cross_entropy(logits, y)
            (loss / accum).backward()
            pending = True
            if (step_i + 1) % accum == 0:
                optimizer.step()
                optimizer.zero_grad()
                pending = False
            losses.append(float(loss.item()))
        if pending:
            optimizer.step()
            optimizer.zero_grad()
        entry = {
            "epoch": epoch,
            "mean_loss": float(np.mean(losses)),
            "n_batches": len(losses),
        }
        if probe is not None:
            r1 = _probe_rank1(model, probe, root, img_size, device, _autocast, pool)
            entry["probe_rank1"] = r1
            print("epoch %2d  mean ArcFace loss %.4f  probe Rank-1 %.4f"
                  % (epoch, entry["mean_loss"], r1), flush=True)
        else:
            print("epoch %2d  mean ArcFace loss %.4f" % (epoch, entry["mean_loss"]),
                  flush=True)
        epoch_log.append(entry)
        if probe is not None:
            if entry["probe_rank1"] > best["rank1"] + 1e-9:
                best = {
                    "rank1": entry["probe_rank1"],
                    "epoch": epoch,
                    "model": {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()},
                    "head": {k: v.detach().cpu().clone()
                             for k, v in head.state_dict().items()},
                }
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= int(early_stop_patience):
                    early_stopped = True
                    print("EARLY_STOP at epoch %d (best probe Rank-1 %.4f at epoch %d)"
                          % (epoch, best["rank1"], best["epoch"]), flush=True)
                    break
    pool.shutdown()
    if probe is not None and best["model"] is not None:
        model.load_state_dict(best["model"])
        head.load_state_dict(best["head"])
        print("restored best-probe weights (epoch %d, probe Rank-1 %.4f)"
              % (best["epoch"], best["rank1"]))

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
        "min_images_per_unit": int(min_images_per_unit),
        "n_units_dropped_min_images": int(n_units_dropped),
        "n_rows_dropped_min_images": int(n_rows_dropped),
        "bf16": bool(bf16),
        "grad_checkpointing": bool(grad_checkpointing),
        "grad_accum": int(grad_accum),
        "early_stop_patience": None if early_stop_patience is None else int(early_stop_patience),
        "probe_units": int(probe_units),
        "early_stopped": bool(early_stopped),
        "best_probe_epoch": best["epoch"],
        "best_probe_rank1": None if best["epoch"] is None else float(best["rank1"]),
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
            # Full train split, NOT just trained classes: units dropped by
            # min_images_per_unit must still be excluded from eval.
            "train_identities": sorted(set(train_df["identity"].astype(str))),
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
    parser.add_argument("--min-images-per-unit", dest="min_images_per_unit", type=int,
                        default=1,
                        help="drop (identity, side) units with fewer than this many "
                             "images from the ArcFace classes (training-side only; "
                             "eval exclusion still covers the full train split)")
    parser.add_argument("--bf16", action="store_true",
                        help="bfloat16 autocast for forward/loss (run 3 Leg B)")
    parser.add_argument("--grad-checkpointing", action="store_true",
                        help="timm gradient checkpointing to cut activation memory")
    parser.add_argument("--grad-accum", type=int, default=1,
                        help="optimizer step every N micro-batches "
                             "(effective batch = batch-size x N)")
    parser.add_argument("--early-stop-patience", dest="early_stop_patience", type=int,
                        default=None,
                        help="stop after N epochs without probe Rank-1 improvement; "
                             "--epochs becomes the maximum. Restores best-probe weights.")
    parser.add_argument("--probe-units", dest="probe_units", type=int, default=500,
                        help="max (identity, side) units in the early-stop probe "
                             "(train-side only, deterministic)")
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
        min_images_per_unit=args.min_images_per_unit,
        bf16=args.bf16,
        grad_checkpointing=args.grad_checkpointing,
        grad_accum=args.grad_accum,
        early_stop_patience=args.early_stop_patience,
        probe_units=args.probe_units,
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
