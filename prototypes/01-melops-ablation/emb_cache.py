"""npz-per-(backbone, arm) embedding cache for the Melops ablation.

Run 1 spent ~1000 s per arm per backbone re-embedding the same crops. This
module caches the embedding matrix for one (backbone, bbox_arm, image-id list)
triple in a single compressed ``.npz`` file so a re-run (or ``diagnose.py``)
skips embedding entirely on a hit.

Key semantics (deliberately strict -- a wrong hit is worse than a miss):

* The file name is derived from ``(backbone, arm, sha256(sorted image_ids))``,
  so the head / body / headless arms can never collide with each other, nor
  can two different backbones or two different image sets.
* The stored payload additionally records the backbone, the arm and the
  image ids IN THE EXACT ORDER the embeddings were computed. ``load`` returns
  the matrix only when backbone, arm and the full ordered id list all match
  the request; any mismatch -- including the same ids in a different order --
  returns ``None`` (a miss), never a reordered or partial hit.
* Embeddings are stored as float32 (halves disk, plenty for cosine retrieval)
  and re-L2-normalized on load, because float32 rounding alone can push row
  norms outside the protocol's 1e-6 normalization tolerance.

Known limitation, by design: the key is metadata (ids), not pixels. If the
images under a root are regenerated with different content but identical
image_ids (e.g. re-rendering a synthetic corpus with another seed into the
same directory), a stale hit is possible -- use one cache directory per
corpus, as ``run_ablation.py --emb-cache`` and ``diagnose.py --emb-cache``
document.
"""

from __future__ import annotations

import hashlib
import os
import re

import numpy as np

import melops_data

_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def _sanitize(name):
    return _SAFE.sub("-", str(name))


def _ids_list(image_ids):
    return [str(i) for i in image_ids]


def cache_key(backbone, arm, image_ids):
    """File-name key: (backbone, arm, sha256 of the ORDERED image-id list).

    Ordered, not sorted: two orderings of the same id set get two files.
    With a sorted hash they mapped to one file and callers with different
    frame orders permanently thrashed it (every load missed the ordered-id
    check, every save overwrote the other's entry).
    """
    ids = _ids_list(image_ids)
    digest = hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()[:20]
    return "%s__%s__%s" % (_sanitize(backbone), _sanitize(arm), digest)


def cache_path(cache_dir, backbone, arm, image_ids):
    return os.path.join(cache_dir, cache_key(backbone, arm, image_ids) + ".npz")


def save(cache_dir, backbone, arm, image_ids, embeddings):
    """Store ``embeddings`` (row-aligned with ``image_ids``) as float32 npz."""
    ids = _ids_list(image_ids)
    embeddings = np.asarray(embeddings)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(ids):
        raise ValueError(
            "embeddings must be (n_ids, d); got shape %r for %d ids"
            % (embeddings.shape, len(ids))
        )
    os.makedirs(cache_dir, exist_ok=True)
    path = cache_path(cache_dir, backbone, arm, ids)
    np.savez_compressed(
        path,
        backbone=np.asarray(str(backbone)),
        arm=np.asarray(str(arm)),
        image_ids=np.asarray(ids),
        embeddings=embeddings.astype(np.float32),
    )
    return path


def load(cache_dir, backbone, arm, image_ids):
    """Return the cached float64, re-L2-normalized matrix, or ``None``.

    ``None`` on: missing/unreadable file, backbone or arm mismatch, or any
    difference in the ORDERED image-id list (reordering is a miss -- the
    caller's frame order is authoritative and a silently reordered matrix
    would corrupt every downstream metric).
    """
    ids = _ids_list(image_ids)
    path = cache_path(cache_dir, backbone, arm, ids)
    if not os.path.exists(path):
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            stored_backbone = str(z["backbone"])
            stored_arm = str(z["arm"])
            stored_ids = [str(x) for x in z["image_ids"].tolist()]
            emb = np.asarray(z["embeddings"], dtype=np.float32)
    except Exception:
        return None  # corrupt or foreign file: treat as a miss
    if stored_backbone != str(backbone) or stored_arm != str(arm):
        return None
    if stored_ids != ids:
        return None  # includes the reordered-ids case
    if emb.ndim != 2 or emb.shape[0] != len(ids):
        return None
    return _canonicalize(emb)


def _canonicalize(emb):
    """Round-trip through the float32 storage precision and re-L2-normalize.

    Applied on EVERY path (cached or not), so uncached, cold-cached and
    warm-cached runs compute on bit-identical matrices. Without this, the
    first cached run would use the embedder's fresh float64 output while all
    later runs used the float32-roundtripped load, and rejection-curve
    thresholds drifted at ~1e-8 between runs.
    """
    emb = np.asarray(emb, dtype=np.float32).astype(np.float64)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return emb / norms


def embed_frame(embedder, root, frame, backbone, arm, cache_dir=None):
    """Embed the crops of ``frame``; with ``cache_dir``, hit skips embedding.

    On a miss the crops are loaded, embedded, canonicalized (float32
    round-trip + renormalize -- see ``_canonicalize``) and saved before being
    returned. ``cache_dir=None`` returns the same canonicalized matrix.
    """
    ids = _ids_list(frame["image_id"].tolist())
    if cache_dir is not None:
        cached = load(cache_dir, backbone, arm, ids)
        if cached is not None:
            return cached
    crops = [melops_data.load_crop(root, row) for _, row in frame.iterrows()]
    emb = _canonicalize(embedder.embed(crops))
    if cache_dir is not None:
        save(cache_dir, backbone, arm, ids, emb)
    return emb
