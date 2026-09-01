"""Ground-truth fixtures for the rig tests: a STRAIGHT capsule with box fins.

Module B never imports module A's mesh code -- it consumes the plain-numpy input
contract documented in ``rig.py``. This file builds a minimal instance of that
contract: an elliptical tube along +X (snout) to -X (tail) with eight box fins
labelled per vertex, plus the straight centerline and the fin insertion/tip table.
Everything is analytic, so every assertion in the tests has ground truth.
"""

from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

BODY_LENGTH = 2.0
BODY_RADIUS = 0.13

# fin name -> (x of base centre, outward direction, lobe length, base chord, tip chord)
FIN_SPEC = {
    "pectoral_left":  (0.42, (0.0, 1.0, -0.35), 0.28, 0.22, 0.06),
    "pectoral_right": (0.42, (0.0, -1.0, -0.35), 0.28, 0.22, 0.06),
    "pelvic_left":    (-0.33, (0.0, 1.0, -0.55), 0.13, 0.14, 0.05),
    "pelvic_right":   (-0.33, (0.0, -1.0, -0.55), 0.13, 0.14, 0.05),
    # single dorsal, far posterior, over the pelvics -- no second dorsal
    "dorsal":         (-0.30, (0.0, 0.0, 1.0), 0.24, 0.26, 0.08),
    "anal":           (-0.50, (0.0, 0.0, -1.0), 0.14, 0.16, 0.05),
    # strongly heterocercal caudal: long upper lobe, weak lower lobe
    "caudal_upper":   (-0.82, (-0.75, 0.0, 1.0), 0.46, 0.16, 0.05),
    "caudal_lower":   (-0.82, (-0.30, 0.0, -1.0), 0.20, 0.14, 0.05),
}


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def body_radius(u):
    """Elliptical cross-section semi-radii (ry, rz) at snout-fraction ``u`` in [0, 1]."""
    prof = 0.18 + 0.82 * np.sin(np.pi * np.clip(u, 0.0, 1.0) ** 0.75)
    ry = BODY_RADIUS * prof
    rz = 1.25 * ry
    return ry, rz


def _tube(n_stations=48, n_around=24):
    half = BODY_LENGTH / 2.0
    u = np.linspace(0.0, 1.0, n_stations)
    x = half - u * BODY_LENGTH
    phi = np.linspace(0.0, 2.0 * np.pi, n_around, endpoint=False)
    ry, rz = body_radius(u)
    verts = np.zeros((n_stations, n_around, 3))
    verts[:, :, 0] = x[:, None]
    verts[:, :, 1] = ry[:, None] * np.cos(phi)[None, :]
    verts[:, :, 2] = rz[:, None] * np.sin(phi)[None, :]
    uv = np.zeros((n_stations, n_around, 2))
    uv[:, :, 0] = (phi / (2.0 * np.pi))[None, :]
    uv[:, :, 1] = u[:, None]

    faces = []
    for i in range(n_stations - 1):
        for k in range(n_around):
            k2 = (k + 1) % n_around
            a = i * n_around + k
            b = i * n_around + k2
            c = (i + 1) * n_around + k2
            d = (i + 1) * n_around + k
            faces.append([a, b, c])
            faces.append([a, c, d])
    return verts.reshape(-1, 3), uv.reshape(-1, 2), np.asarray(faces, dtype=np.int64)


def _box_fin(base, direction, length, chord_base, chord_tip, thickness=0.012):
    d = _unit(direction)
    xaxis = np.array([1.0, 0.0, 0.0])
    nrm = _unit(np.cross(d, xaxis))
    tip = base + d * length
    verts = []
    for centre, chord in ((base, chord_base), (tip, chord_tip)):
        for sc in (-0.5, 0.5):
            for sn in (-0.5, 0.5):
                verts.append(centre + sc * chord * xaxis + sn * thickness * nrm)
    verts = np.asarray(verts)          # 0..3 base quad, 4..7 tip quad
    faces = np.asarray([
        [0, 1, 3], [0, 3, 2],          # base
        [4, 7, 5], [4, 6, 7],          # tip
        [0, 4, 5], [0, 5, 1],
        [2, 3, 7], [2, 7, 6],
        [0, 2, 6], [0, 6, 4],
        [1, 5, 7], [1, 7, 3],
    ], dtype=np.int64)
    uv = np.zeros((8, 2))
    uv[:, 0] = np.tile([0.0, 0.0, 1.0, 1.0], 2)
    uv[:4, 1] = 0.0
    uv[4:, 1] = 1.0
    return verts, uv, faces, tip


def straight_capsule(n_stations=48, n_around=24):
    """Build the fixture.

    Returns a dict with:
        ``vertices`` (N, 3), ``faces`` (F, 3) int64, ``uv`` (N, 2),
        ``labels`` (N,) object array ("body" or a fin name),
        ``centerline`` (M, 3) straight, snout first,
        ``fin_info`` {name: {"insertion", "tip"}}.
    """
    verts, uv, faces = _tube(n_stations, n_around)
    labels = np.array([("body")] * len(verts), dtype=object)
    fin_info = {}
    all_v = [verts]
    all_uv = [uv]
    all_f = [faces]
    all_lab = [labels]
    offset = len(verts)
    half = BODY_LENGTH / 2.0
    for name, (bx, direction, length, chord_b, chord_t) in FIN_SPEC.items():
        u = (half - bx) / BODY_LENGTH
        ry, rz = body_radius(u)
        d = _unit(direction)
        # push the base centre out to the body wall along the fin direction
        surf = np.array([bx, d[1] * ry, d[2] * rz])
        fv, fuv, ff, tip = _box_fin(surf, direction, length, chord_b, chord_t)
        all_v.append(fv)
        all_uv.append(fuv)
        all_f.append(ff + offset)
        all_lab.append(np.array([name] * len(fv), dtype=object))
        fin_info[name] = {"insertion": surf, "tip": tip}
        offset += len(fv)

    centerline = np.column_stack([
        np.linspace(half, -half, 64),
        np.zeros(64),
        np.zeros(64),
    ])
    return {
        "vertices": np.concatenate(all_v, axis=0),
        "uv": np.concatenate(all_uv, axis=0),
        "faces": np.concatenate(all_f, axis=0),
        "labels": np.concatenate(all_lab, axis=0),
        "centerline": centerline,
        "fin_info": fin_info,
    }


def as_trimesh(fixture, textured=True, seed=0):
    """Wrap the fixture in a ``trimesh.Trimesh`` with a UV-mapped checker texture."""
    import trimesh
    from PIL import Image

    mesh = trimesh.Trimesh(
        vertices=fixture["vertices"], faces=fixture["faces"], process=False
    )
    if textured:
        rng = np.random.default_rng(seed)
        size = 32
        checker = ((np.indices((size, size)).sum(axis=0) % 2) * 160 + 60).astype(np.uint8)
        rgb = np.stack([checker, np.roll(checker, 3, axis=0), checker.T], axis=-1)
        rgb = (rgb * 0.8 + rng.integers(0, 40, rgb.shape)).clip(0, 255).astype(np.uint8)
        image = Image.fromarray(rgb, mode="RGB")
        material = trimesh.visual.material.PBRMaterial(
            name="sevengill_skin", baseColorTexture=image, metallicFactor=0.0,
            roughnessFactor=0.7,
        )
        mesh.visual = trimesh.visual.TextureVisuals(uv=fixture["uv"], material=material)
    return mesh
