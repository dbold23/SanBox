"""End-to-end sevengill rigging CLI: any GLB in, a skinned + animated GLB out.

The pipeline is mesh-agnostic.  Nothing here knows that the input is a shark
except the anatomical priors inside :mod:`mesh3d`'s fin naming and the joint
names of the phase1b sevengill schema; every geometric step is "elongate body
with fins".

    load_mesh                 GLB/OBJ/PLY, UVs + material intact
    extract_centerline_3d     voxelise -> 3D EDT -> medial-weighted Dijkstra
    tube_frames               rotation-minimising frames along that centerline
    tube_coords               every vertex -> (s, r, phi) + station index
    debend                    re-embed on a straight axis == the REST POSE
    detect_fins               per-vertex label + insertion geometry (rest pose)
    build_skeleton            13 schema spine joints + 2 joints per fin
    compute_weights           LBS weights, <= 4 influences, rows summing to 1
    make_clip                 cruise / turn / escape / rest / glide
    write_skinned_glb         one GLB, Khronos-validated

Why de-bend at all
------------------
The real input is a photogrammetric scan of an animal that was *mid-turn*: its
rest pose is a strong lateral C.  Binding a skeleton to a C-shaped mesh bakes
that C into the bind pose, so every animation is a bend applied on top of an
existing bend.  Tube coordinates give the fix: (s, r, phi) against a
rotation-minimising frame field is prototype 02's canonical chart lifted to 3D,
and *keeping (r, phi) while replacing the centerline* straightens the animal
without touching topology, UVs, materials or the heterocercal tail (which hangs
off the end of the chart and is transported rigidly -- see ``mesh3d``).

``--keep-bent`` writes the scan pose back as an animation clip called
``as_scanned``: the extracted centerline is turned into per-joint rotations, so
the rig visibly reproduces the pose the animal was photographed in.  That is the
audit: if ``as_scanned`` does not look like the scan, the chart is wrong.

CLI
---
    python rig_sevengill.py --glb IN.glb --out OUT.glb
        [--motion cruise,turn,escape,rest,glide] [--fps 30] [--seconds 4]
        [--voxel-pitch AUTO] [--keep-bent] [--report DIR]
        [--n-stations 64] [--up X Y Z] [--core-radius-frac 0.17]
        [--sigma W] [--precaudal-fraction 0.78] [--seed 0] [--no-validate]

``--up`` is load-bearing on a real mesh: it seeds the frame field with the
animal's dorsal direction, and phi = 0 dorsal is what makes the L/R fin split
and the "purely lateral" de-bend correct.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import NamedTuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gltf_export  # noqa: E402
import mesh3d  # noqa: E402
import motion  # noqa: E402
import rig  # noqa: E402

__all__ = [
    "DEFAULT_MOTIONS",
    "AS_SCANNED_NAME",
    "RigResult",
    "run_pipeline",
    "solve_as_scanned",
    "clip_for_mode",
    "write_report",
    "main",
]

DEFAULT_MOTIONS = ("cruise",)
AS_SCANNED_NAME = "as_scanned"

# Ease-in of the as_scanned clip: frame 0 is the straight rest pose, the pose is
# reached at ``_AS_SCANNED_EASE_S`` and held to the end.  A static two-frame clip
# would show the same pose, but viewers that autoplay the first animation then
# give no visual evidence that the REST pose is straight.
_AS_SCANNED_EASE_S = 1.0
_AS_SCANNED_HOLD_S = 0.5


class RigResult(NamedTuple):
    """Everything :func:`run_pipeline` produced, for the demo, tests and report.

    mesh: the input mesh as loaded (the scanned, bent pose).
    centerline / centerline_info: :func:`mesh3d.extract_centerline_3d` output.
    frames: (tangents, normals, binormals) of the scanned centerline.
    straight_mesh / straight_centerline: the de-bent REST pose.
    detection: :class:`mesh3d.FinDetection` measured on the rest pose.
    skeleton / weights: the rig.
    clips: ``{name: motion.Clip}`` in write order.
    out_path: the written GLB, or None.
    issues: the Khronos validator's issues dict, or None when skipped.
    timings: ``{stage: seconds}``.
    """

    mesh: object
    centerline: np.ndarray
    centerline_info: dict
    frames: tuple
    straight_mesh: object
    straight_centerline: np.ndarray
    detection: object
    skeleton: object
    weights: np.ndarray
    clips: dict
    out_path: object
    issues: object
    timings: dict


# ---------------------------------------------------------------------------
# as_scanned: extracted centerline -> joint rotations
# ---------------------------------------------------------------------------
def _frame_matrices_at(centerline, frames, fractions):
    """World rotation of the tube frame at each arc-length fraction.

    Returns ``(M (K, 3, 3), q (K, 3))``: ``M[k]`` maps the canonical straight
    frame (T = -X, N = +Z, B = +Y) onto the frame of the bent centerline at
    ``fractions[k]``, and ``q[k]`` is the centerline point there.

    The tangent column is replaced by the CHORD direction to the next joint
    before orthonormalisation, because ``rig.forward_kinematics`` advances the
    chain along the rotated rest bone: a bone must point at the next joint, not
    along the tangent at its own end (a forward-Euler lag of half a segment).
    The last joint keeps the frame's own tangent.
    """
    cl = np.asarray(centerline, dtype=float)
    tangents, normals, binormals = [np.asarray(a, dtype=float) for a in frames]
    s = mesh3d.arc_length(cl)
    total = float(s[-1])
    targets = np.asarray(fractions, dtype=float) * total

    q = np.column_stack([np.interp(targets, s, cl[:, i]) for i in range(3)])
    tan = np.column_stack([np.interp(targets, s, tangents[:, i]) for i in range(3)])
    nrm = np.column_stack([np.interp(targets, s, normals[:, i]) for i in range(3)])

    # bone direction: chord to the next joint (last joint reuses its tangent)
    chord = np.diff(q, axis=0)
    lengths = np.linalg.norm(chord, axis=1, keepdims=True)
    good = lengths[:, 0] > 1e-12
    direction = tan.copy()
    direction[:-1][good] = chord[good] / lengths[good]

    # Gram-Schmidt: T = direction, N = the RMF normal made orthogonal to it.
    t_col = direction / np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-12)
    n_col = nrm - t_col * np.sum(nrm * t_col, axis=1, keepdims=True)
    bad = np.linalg.norm(n_col, axis=1) < 1e-9
    if np.any(bad):  # degenerate seed: any vector not parallel to T will do
        n_col[bad] = np.cross(t_col[bad], np.array([0.0, 0.0, 1.0]))
    n_col = n_col / np.maximum(np.linalg.norm(n_col, axis=1, keepdims=True), 1e-12)
    b_col = np.cross(t_col, n_col)

    frame = np.stack([t_col, n_col, b_col], axis=2)          # columns T, N, B
    # canonical straight frame, same column order: T = -X, N = +Z, B = +Y
    canon = np.column_stack([[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    return frame @ canon.T, q


def solve_as_scanned(skeleton, bent_centerline, frames=None, up=(0.0, 0.0, 1.0),
                     fps=30.0, ease_s=_AS_SCANNED_EASE_S, hold_s=_AS_SCANNED_HOLD_S,
                     name=AS_SCANNED_NAME, rest_centerline=None):
    """Joint rotations that bend the straight rig back onto the scanned centerline.

    The spine joints sit at known arc-length fractions of the rest centerline; the
    same fractions on the SCANNED centerline give a target position and a target
    frame per joint.  The cumulative world rotation of joint ``j`` is the rotation
    carrying the canonical straight frame onto that target frame, and the local
    rotation is ``M_parent^T M_j`` -- exactly the quantity ``rig.forward_kinematics``
    composes.  Fin joints stay at identity: a fin's pose in the scan is whatever the
    body carries it to, and the chart has no independent measurement of it.

    The clip eases from the rest pose (frame 0) to the scan pose over ``ease_s`` and
    holds it for ``hold_s``, so a viewer that autoplays shows both poses.  A root
    TRANSLATION channel carries the rig from the rest origin to where the scanned
    centerline actually is, so the clip lands on the scan in world space.

    Args:
        skeleton: ``rig.Skeleton`` built on the STRAIGHT rest centerline.
        bent_centerline: (M, 3) the extracted, scanned centerline.
        frames: its ``(T, N, B)``; recomputed from ``up`` when None.
        up: dorsal seed for the frames.
        rest_centerline: the polyline the skeleton was built on when it is not
            the canonical straight axis (a spine extended through a caudal lobe);
            None means every rest bone points along -X.
        fps, ease_s, hold_s, name: clip shape.

    Returns:
        ``(motion.Clip, info)``. ``info`` has ``rotmats`` (J, 3, 3) full-pose local
        rotations, ``translation`` (3,) the root world offset, ``joint_targets``
        (J_spine, 3) and ``joint_error`` (J_spine,) |posed - target| per spine joint.
    """
    cl = np.asarray(bent_centerline, dtype=float)
    if frames is None:
        frames = mesh3d.tube_frames(cl, up=up)

    spine_idx = np.asarray(skeleton.spine_indices, dtype=int)
    fractions = np.asarray(skeleton.fractions, dtype=float)[spine_idx]
    world, targets = _frame_matrices_at(cl, frames, fractions)
    if rest_centerline is not None:
        # The rest bones are not all along -X (the spine runs on into a
        # caudal lobe): the world rotation must carry the REST frame, not the
        # canonical one, onto the target.  W = M_target @ M_rest^T.
        rest_cl = np.asarray(rest_centerline, dtype=float)
        rest_world, _ = _frame_matrices_at(
            rest_cl, mesh3d.tube_frames(rest_cl, up=(0.0, 0.0, 1.0)), fractions)
        world = np.einsum("kab,kcb->kac", world, rest_world)

    n_joints = skeleton.num_joints
    rotmats = np.tile(np.eye(3), (n_joints, 1, 1))
    for k, j in enumerate(spine_idx):
        parent = int(skeleton.parents[int(j)])
        if parent == -1:
            rotmats[int(j)] = world[k]
        else:
            kp = int(np.flatnonzero(spine_idx == parent)[0])
            rotmats[int(j)] = world[kp].T @ world[k]

    translation = targets[0] - np.asarray(skeleton.joints, dtype=float)[int(spine_idx[0])]

    posed = rig.posed_joints(skeleton, rotmats)[spine_idx] + translation
    joint_error = np.linalg.norm(posed - targets, axis=1)

    # -- time base: ease then hold ---------------------------------------
    n_ease = max(1, int(round(float(ease_s) * float(fps))))
    n_hold = max(1, int(round(float(hold_s) * float(fps))))
    times = np.concatenate([
        np.linspace(0.0, float(ease_s), n_ease + 1),
        float(ease_s) + np.linspace(0.0, float(hold_s), n_hold + 1)[1:],
    ])
    u = np.clip(times / max(float(ease_s), 1e-9), 0.0, 1.0)
    alpha = u * u * (3.0 - 2.0 * u)                     # smoothstep

    axes, angles = _rotmats_to_axis_angle(rotmats)
    quats = np.empty((len(times), n_joints, 4), dtype=float)
    for i, a in enumerate(alpha):
        half = 0.5 * a * angles
        quats[i, :, :3] = axes * np.sin(half)[:, None]
        quats[i, :, 3] = np.cos(half)
    quats /= np.linalg.norm(quats, axis=-1, keepdims=True)

    rest_offsets = _rest_local_translations(skeleton)
    translations = np.tile(rest_offsets, (len(times), 1, 1))
    translations[:, int(spine_idx[0]), :] += alpha[:, None] * translation[None, :]

    clip = motion.Clip(
        name=name,
        times=times,
        quats=quats,
        joint_names=list(skeleton.names),
        fps=float(fps),
        loop=False,
        meta={
            "mode": AS_SCANNED_NAME,
            "translations": translations,
            "root_translation": translation,
            "joint_error": joint_error,
        },
    )
    info = {
        "rotmats": rotmats,
        "translation": translation,
        "translations": translations,
        "joint_targets": targets,
        "joint_error": joint_error,
    }
    return clip, info


def _rest_local_translations(skeleton):
    """(J, 3) node-local translations of the rest pose, matching ``gltf_export``."""
    joints = np.asarray(skeleton.joints, dtype=float)
    out = np.array(joints, copy=True)
    for j in range(skeleton.num_joints):
        parent = int(skeleton.parents[j])
        if parent != -1:
            out[j] = joints[j] - joints[parent]
    return out


def _rotmats_to_axis_angle(rotmats):
    """(J, 3, 3) -> ``(axes (J, 3) unit, angles (J,))`` in [0, pi]."""
    quats = rig.rotmat_to_quat(np.asarray(rotmats, dtype=float))
    quats = np.where(quats[:, 3:4] < 0.0, -quats, quats)       # shortest arc
    w = np.clip(quats[:, 3], -1.0, 1.0)
    angles = 2.0 * np.arccos(w)
    sin_half = np.sqrt(np.maximum(1.0 - w * w, 0.0))
    axes = np.zeros((len(quats), 3), dtype=float)
    small = sin_half < 1e-12
    axes[~small] = quats[~small, :3] / sin_half[~small, None]
    axes[small] = np.array([0.0, 0.0, 1.0])
    return axes, angles


def as_scanned_animation(clip):
    """``clip.to_animation()`` plus the translation channel stored in its meta."""
    anim = clip.to_animation()
    translations = clip.meta.get("translations")
    if translations is not None:
        anim["translations"] = translations
    return anim


# ---------------------------------------------------------------------------
# Motion clips
# ---------------------------------------------------------------------------
def clip_for_mode(mskel, mode, fps=30.0, seconds=4.0, seed=0):
    """One :class:`motion.Clip` for ``mode``, ``seconds`` long where that is legal.

    A looping mode needs a WHOLE number of tail-beat periods (``motion.make_clip``
    refuses to emit a loop that pops), so ``seconds`` is rounded to the nearest
    whole period, at least one.  ``escape`` is a one-shot transient whose length is
    set by its own stage durations and ignores ``seconds`` entirely.
    """
    cfg = motion.MODE_CONFIG[mode]
    if mode == "escape":
        return motion.make_clip(mskel, mode, fps=fps, seed=seed)
    params = motion.params_for_mode(mode)
    if cfg["loop"]:
        n_periods = max(1, int(round(float(seconds) * params.frequency_hz)))
        return motion.make_clip(mskel, mode, fps=fps, n_periods=n_periods, seed=seed)
    return motion.make_clip(mskel, mode, fps=fps, duration=float(seconds), seed=seed)


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------
#: A caudal island whose axial extent is at least this many times its radial
#: extent is a LOBE (the real heterocercal upper lobe: 0.19 m long, 0.05 m
#: tall), not a blade.  Driven as a fin about a root joint it tears against
#: the peduncle -- a mid-lobe root levers its front half against the body, a
#: base root shears its underside -- so a lobe is folded into the body and
#: rides the last two spine joints.
LOBE_ASPECT = 2.0


def fold_caudal_lobes(fin_info, labels, fins, vertices, centerline, force=False):
    """Drop caudal fins that are lobes from ``fin_info`` and relabel their
    vertices ``body``.  Returns ``(fin_info, labels, [(name, aspect), ...])``.
    ``force`` folds every caudal fin regardless of shape."""
    verts = np.asarray(vertices, dtype=float)
    labels = np.asarray(labels).astype(str).copy()
    out = dict(fin_info)
    folded = []
    for name in list(out):
        if not name.startswith("caudal") or name not in fins:
            continue
        members = np.asarray(fins[name]["vertex_indices"], dtype=int)
        if len(members) == 0:
            continue
        foot, tang = rig._axis_frame_at(np.asarray(fins[name]["insertion_centroid"], dtype=float),
                                        centerline)
        rel = verts[members] - foot
        axial = rel @ tang
        radial = np.linalg.norm(rel - np.outer(axial, tang), axis=1)
        aspect = float(np.ptp(axial)) / max(float(np.ptp(radial)), 1e-9)
        if force or aspect >= LOBE_ASPECT:
            out.pop(name)
            labels[members] = "body"
            folded.append((name, aspect))
    return out, labels, folded


def run_pipeline(
    glb,
    out=None,
    motions=DEFAULT_MOTIONS,
    fps=30.0,
    seconds=4.0,
    voxel_pitch=None,
    n_stations=64,
    up=(0.0, 0.0, 1.0),
    core_radius_frac=0.17,
    hook_turn_mult=3.0,
    caudal_lobes="fins",
    fin_blend_dist=None,
    sigma=None,
    fin_blend_rings=rig.DEFAULT_FIN_BLEND_RINGS,
    precaudal_fraction=rig.DEFAULT_PRECAUDAL_FRACTION,
    keep_bent=False,
    report=None,
    validate=True,
    seed=0,
    verbose=True,
):
    """Run load -> de-bend -> rig -> animate -> GLB.  Returns :class:`RigResult`.

    Args:
        glb: input mesh path (GLB/OBJ/PLY), or a ``trimesh.Trimesh`` already loaded.
        out: output GLB path; None runs the pipeline without writing.
        motions: sequence of :data:`motion.MODES` names.
        fps, seconds: clip frame rate and nominal length (see :func:`clip_for_mode`).
        voxel_pitch: None = ``max(extents)/128``.
        n_stations: chart resolution.
        up: the input mesh's DORSAL direction.  Load-bearing; see the module docstring.
        fin_blend_dist: fin-base blend width in metres (see
            ``rig.compute_weights``); None uses ``fin_blend_rings``.
        caudal_lobes: ``"fins"`` (default) keeps every caudal island as a driven
            fin, as the synthetic fixture expects; ``"auto"`` folds a caudal
            island that is a lobe (see ``LOBE_ASPECT``) into the body so it
            rides the last spine joints -- what a scanned heterocercal tail
            needs; ``"body"`` folds them all.
        core_radius_frac: thick-core threshold that keeps the medial path out of the
            fins.  Must sit between fin half-thickness and peduncle radius.
        sigma: Gaussian weight falloff width in world units; None = two-joint
            arc-length binding.
        fin_blend_rings: width in mesh edge rings of the fin-base weight ramp
            (``rig.compute_weights``); 1 restores the old hard seam.
        precaudal_fraction: precaudal length / total chart length.
        keep_bent: also emit the ``as_scanned`` clip.
        report: directory for the diagnostic dump, or None.
        validate: run the Khronos validator on the written GLB.
        seed: passed to ``motion.make_clip``.
        verbose: print a one-line summary per stage.
    """
    timings = {}
    log = (lambda msg: print(msg)) if verbose else (lambda msg: None)

    t0 = time.time()
    if hasattr(glb, "vertices"):
        mesh = glb
    else:
        mesh = mesh3d.load_mesh(glb, report=False)
    timings["load"] = time.time() - t0
    log("load: %d verts, %d faces, extents %s"
        % (len(mesh.vertices), len(mesh.faces), np.round(mesh.extents, 4)))

    t0 = time.time()
    centerline, info = mesh3d.extract_centerline_3d(
        mesh, voxel_pitch=voxel_pitch, n_stations=n_stations,
        core_radius_frac=core_radius_frac, seed=seed,
    )
    timings["centerline"] = time.time() - t0
    log("centerline: %d stations, length %.4f, pitch %.5f, head/tail width %.4f/%.4f"
        % (len(centerline), info["length"], info["pitch"],
           info["head_width"], info["tail_width"]))

    t0 = time.time()
    centerline, n_hook = mesh3d.trim_end_hooks(centerline, up=tuple(up), turn_mult=hook_turn_mult)
    if n_hook:
        info["length"] = float(mesh3d.arc_length(centerline)[-1])
        info["n_hook_stations"] = int(n_hook)
        log("end hook: %d station(s) that pitched into a fin trimmed; chart length now %.4f"
            % (n_hook, info["length"]))
    frames = mesh3d.tube_frames(centerline, up=tuple(up))
    coords = mesh3d.tube_coords(mesh, centerline, frames)
    straight_mesh, straight_centerline = mesh3d.debend(mesh, centerline, frames)
    timings["debend"] = time.time() - t0
    bend_sagitta = float(np.max(np.linalg.norm(
        centerline - _chord_projection(centerline), axis=1)))
    log("de-bend: sagitta of the scanned centerline %.4f (%.1f%% of length), "
        "rest extents %s"
        % (bend_sagitta, 100.0 * bend_sagitta / info["length"],
           np.round(straight_mesh.extents, 4)))

    t0 = time.time()
    straight_frames = mesh3d.canonical_frames(len(straight_centerline))
    straight_coords = mesh3d.tube_coords(straight_mesh, straight_centerline, straight_frames)
    detection = mesh3d.detect_fins(straight_mesh, straight_coords)
    timings["fins"] = time.time() - t0
    log("fins: %d islands -- %s" % (len(detection.fins), ", ".join(sorted(detection.fins))))

    t0 = time.time()
    fin_info = rig.fin_info_from_detection(
        detection.fins, straight_mesh.vertices, centerline=straight_centerline
    )
    labels = np.asarray(detection.labels).astype(str)
    folded = []
    if caudal_lobes == "body" or caudal_lobes == "auto":
        fin_info, labels, folded = fold_caudal_lobes(
            fin_info, labels, detection.fins, straight_mesh.vertices, straight_centerline,
            force=(caudal_lobes == "body"))
        for name, ratio in folded:
            log("caudal: %s is a lobe (axial/radial extent %.1f); it rides the last spine "
                "joints as body, no fin joints" % (name, ratio))
        if folded:
            detection = detection._replace(labels=labels)
    # The spine polyline the skeleton is built on.  When a caudal lobe was
    # folded into the body, the spine runs on from the chart's end to the
    # lobe's tip so the two caudal-axis joints sit ON the lobe (the schema's
    # own intent: "the vertebral axis turns up into the long upper lobe") and
    # the wave continues into it instead of the lobe swinging as one rigid
    # piece off the last joint.  The scanned polyline gets the same vertex's
    # scanned position so as_scanned stays consistent, and the precaudal
    # fraction is rescaled so the precaudal joint stays at the peduncle.
    spine_straight, spine_bent, spine_frames = straight_centerline, centerline, frames
    pf = precaudal_fraction
    rest_for_solve = None
    if folded:
        lobe = max(folded, key=lambda nr: len(detection.fins[nr[0]]["vertex_indices"]))[0]
        members = np.asarray(detection.fins[lobe]["vertex_indices"], dtype=int)
        tip = int(members[np.argmin(np.asarray(straight_mesh.vertices)[members, 0])])
        spine_straight = np.vstack([straight_centerline, np.asarray(straight_mesh.vertices)[tip]])
        spine_bent = np.vstack([centerline, np.asarray(mesh.vertices)[tip]])
        spine_frames = mesh3d.tube_frames(spine_bent, up=tuple(up))
        l_chart = float(mesh3d.arc_length(straight_centerline)[-1])
        l_ext = float(mesh3d.arc_length(spine_straight)[-1])
        pf = float(precaudal_fraction) * l_chart / l_ext
        rest_for_solve = spine_straight
        log("spine extended through %s to its tip: %.4f -> %.4f m, precaudal fraction %.3f -> %.3f"
            % (lobe, l_chart, l_ext, precaudal_fraction, pf))
    skeleton = rig.build_skeleton(spine_straight, fin_info, precaudal_fraction=pf)
    weights = rig.compute_weights(
        straight_mesh.vertices, labels, skeleton, sigma=sigma,
        faces=straight_mesh.faces, fin_blend_rings=fin_blend_rings,
        fin_blend_dist=fin_blend_dist,
    )
    timings["rig"] = time.time() - t0
    log("rig: %d joints (%d spine + %d fin), weights %s, max influences %d"
        % (skeleton.num_joints, rig.NUM_SPINE_JOINTS, 2 * len(skeleton.fins),
           weights.shape, int((weights > 0).sum(axis=1).max())))

    t0 = time.time()
    mskel = motion.MotionSkeleton.from_skeleton(skeleton, fps=fps)
    clips = {}
    for mode in motions:
        clips[mode] = clip_for_mode(mskel, mode, fps=fps, seconds=seconds, seed=seed)
        log("clip %-10s %d frames, %.2f s, loop=%s"
            % (mode, clips[mode].num_frames, clips[mode].duration_s, clips[mode].loop))
    as_scanned_info = None
    if keep_bent:
        clips[AS_SCANNED_NAME], as_scanned_info = solve_as_scanned(
            skeleton, spine_bent, spine_frames, up=tuple(up), fps=fps,
            rest_centerline=rest_for_solve,
        )
        log("clip %-10s %d frames, %.2f s, spine joint error max %.5f"
            % (AS_SCANNED_NAME, clips[AS_SCANNED_NAME].num_frames,
               clips[AS_SCANNED_NAME].duration_s,
               float(np.max(as_scanned_info["joint_error"]))))
    timings["motion"] = time.time() - t0

    issues = None
    out_path = None
    if out is not None:
        t0 = time.time()
        animations = [
            as_scanned_animation(c) if name == AS_SCANNED_NAME else c.to_animation()
            for name, c in clips.items()
        ]
        out_path = gltf_export.write_skinned_glb(
            straight_mesh, skeleton, weights, out, animations=animations
        )
        timings["write"] = time.time() - t0
        log("wrote %s (%.1f kB)" % (out_path, os.path.getsize(out_path) / 1024.0))
        if validate:
            t0 = time.time()
            issues = gltf_export.validate_glb(out_path)
            timings["validate"] = time.time() - t0
            log("validator: %d errors, %d warnings"
                % (issues["numErrors"], issues["numWarnings"]))

    result = RigResult(
        mesh=mesh,
        centerline=centerline,
        centerline_info=info,
        frames=frames,
        straight_mesh=straight_mesh,
        straight_centerline=straight_centerline,
        detection=detection,
        skeleton=skeleton,
        weights=weights,
        clips=clips,
        out_path=out_path,
        issues=issues,
        timings=timings,
    )
    if report is not None:
        t0 = time.time()
        write_report(report, result, as_scanned_info=as_scanned_info)
        timings["report"] = time.time() - t0
        log("report: %s" % report)
    return result


def _chord_projection(centerline):
    """Each centerline point projected onto the straight chord end-to-end."""
    cl = np.asarray(centerline, dtype=float)
    a, b = cl[0], cl[-1]
    axis = b - a
    length = np.linalg.norm(axis)
    if length <= 0:
        return np.tile(a, (len(cl), 1))
    u = axis / length
    return a[None, :] + np.clip((cl - a) @ u, 0.0, length)[:, None] * u[None, :]


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def write_report(directory, result, as_scanned_info=None, contact_strip=True):
    """Write ``centerline.json``, ``fins.json``, ``skeleton.json``, ``weights.json``
    and ``contact_strip.png`` into ``directory``.  Returns the list of paths."""
    os.makedirs(directory, exist_ok=True)
    paths = []

    info = dict(result.centerline_info)
    centerline_doc = {
        "n_stations": int(len(result.centerline)),
        "length": float(info["length"]),
        "voxel_pitch": float(info["pitch"]),
        "tau_thick_core": float(info["tau"]),
        "head_width": float(info["head_width"]),
        "tail_width": float(info["tail_width"]),
        "n_core_voxels": int(info["n_core_voxels"]),
        "sagitta": float(np.max(np.linalg.norm(
            result.centerline - _chord_projection(result.centerline), axis=1))),
        "scanned_centerline": _jsonable(result.centerline),
        "station_radius": _jsonable(info["radius"]),
        "straight_centerline": _jsonable(result.straight_centerline),
    }
    paths.append(_dump(directory, "centerline.json", centerline_doc))

    labels = np.asarray(result.detection.labels)
    names, counts = np.unique(labels, return_counts=True)
    total_length = float(info["length"])
    fins_doc = {
        "label_counts": {str(n): int(c) for n, c in zip(names, counts)},
        "n_vertices": int(len(labels)),
        "envelope": _jsonable(result.detection.envelope),
        "fins": {},
    }
    for name in sorted(result.detection.fins):
        fin = result.detection.fins[name]
        s_lo, s_hi = [float(v) for v in fin["s_range"]]
        fins_doc["fins"][name] = {
            "n_vertices": int(fin["n_vertices"]),
            "station_range": [int(v) for v in fin["station_range"]],
            "s_range": [s_lo, s_hi],
            "s_fraction_range": [s_lo / total_length, s_hi / total_length],
            "insertion_centroid": _jsonable(fin["insertion_centroid"]),
            "phi_centroid_deg": float(np.degrees(fin["phi_centroid"])),
        }
    paths.append(_dump(directory, "fins.json", fins_doc))

    sk = result.skeleton
    skeleton_doc = {
        "num_joints": int(sk.num_joints),
        "num_spine_joints": int(rig.NUM_SPINE_JOINTS),
        "names": list(sk.names),
        "parents": _jsonable(sk.parents),
        "kinds": list(sk.kinds),
        "fractions": _jsonable(sk.fractions),
        "joints": _jsonable(sk.joints),
        "fins": {k: [int(v[0]), int(v[1])] for k, v in sk.fins.items()},
        "fin_parents": {
            k: str(sk.names[int(sk.parents[int(v[0])])]) for k, v in sk.fins.items()
        },
        "clips": {
            name: {
                "frames": int(c.num_frames),
                "duration_s": float(c.duration_s),
                "fps": float(c.fps),
                "loop": bool(c.loop),
                "mode": str(c.meta.get("mode", name)),
            }
            for name, c in result.clips.items()
        },
    }
    if as_scanned_info is not None:
        skeleton_doc["as_scanned"] = {
            "root_translation": _jsonable(as_scanned_info["translation"]),
            "spine_joint_error": _jsonable(as_scanned_info["joint_error"]),
            "spine_joint_error_units": "world units (same units as 'joints'); "
                                       "divide by centerline.json 'length' for BL",
            "spine_joint_error_bl": _jsonable(
                np.asarray(as_scanned_info["joint_error"], dtype=float) / total_length
            ),
            "spine_joint_targets": _jsonable(as_scanned_info["joint_targets"]),
        }
    paths.append(_dump(directory, "skeleton.json", skeleton_doc))

    w = np.asarray(result.weights, dtype=float)
    nz = (w > 0).sum(axis=1)
    per_joint = w.sum(axis=0)
    order = np.argsort(-per_joint)
    weights_doc = {
        "shape": [int(w.shape[0]), int(w.shape[1])],
        "row_sum_min": float(w.sum(axis=1).min()),
        "row_sum_max": float(w.sum(axis=1).max()),
        "max_influences": int(nz.max()),
        "influences_histogram": {
            str(k): int((nz == k).sum()) for k in range(1, int(nz.max()) + 1)
        },
        "mean_influences": float(nz.mean()),
        "unweighted_joints": [
            str(sk.names[j]) for j in range(sk.num_joints) if per_joint[j] <= 0
        ],
        "vertex_mass_by_joint": {
            str(sk.names[int(j)]): float(per_joint[int(j)]) for j in order
        },
    }
    paths.append(_dump(directory, "weights.json", weights_doc))

    if contact_strip:
        paths.append(render_contact_strip(
            os.path.join(directory, "contact_strip.png"), result
        ))
    return paths


def _dump(directory, name, doc):
    path = os.path.join(directory, name)
    with open(path, "w") as fh:
        json.dump(doc, fh, indent=1, sort_keys=True)
    return path


# ---------------------------------------------------------------------------
# PIL contact strip (orthographic; there is no OpenGL in this environment)
# ---------------------------------------------------------------------------
_PANEL = (400, 200)
_BG = (250, 250, 248)
_INK = (40, 42, 46)


class _Projector(object):
    """World -> pixel for one orthographic panel with a FIXED world box.

    Sharing the box across panels is what makes the strip readable: the rest pose
    and the cruise frames are then drawn at the same scale and the amplitude of the
    body wave is a real, comparable quantity.
    """

    def __init__(self, box_lo, box_hi, axes, size, pad=0.05):
        self.h_ax, self.v_ax = axes
        self.size = size
        lo = np.array([box_lo[self.h_ax], box_lo[self.v_ax]], dtype=float)
        hi = np.array([box_hi[self.h_ax], box_hi[self.v_ax]], dtype=float)
        span = np.maximum(hi - lo, 1e-9)
        self.scale = (1.0 - 2.0 * pad) * min(size[0] / span[0], size[1] / span[1])
        self.origin = np.array(size, dtype=float) * 0.5 - 0.5 * (lo + hi) * self.scale

    def __call__(self, points):
        p = np.asarray(points, dtype=float).reshape(-1, 3)
        px = p[:, [self.h_ax, self.v_ax]] * self.scale + self.origin
        px[:, 1] = self.size[1] - px[:, 1]
        return px

    @property
    def depth_axis(self):
        return 3 - self.h_ax - self.v_ax


def _draw_mesh(draw, vertices, faces, proj, light=(0.4, 0.5, 0.75), tint=(0.72, 0.76, 0.80)):
    v = np.asarray(vertices, dtype=float)
    f = np.asarray(faces, dtype=np.int64)
    px = proj(v)
    tri = v[f]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    lam = np.abs(nrm @ (np.asarray(light, dtype=float) / np.linalg.norm(light)))
    shade = np.clip(0.28 + 0.72 * lam, 0.0, 1.0)
    depth = tri[:, :, proj.depth_axis].mean(axis=1)
    for k in np.argsort(depth):
        c = 255.0 * shade[k]
        draw.polygon([tuple(px[i]) for i in f[k]],
                     fill=(int(c * tint[0]), int(c * tint[1]), int(c * tint[2])))


def _panel(title, box_lo, box_hi, axes, size=_PANEL):
    from PIL import Image, ImageDraw
    img = Image.new("RGB", size, _BG)
    draw = ImageDraw.Draw(img)
    proj = _Projector(box_lo, box_hi, axes, size)
    return img, draw, proj, title


def _finish_panel(img, title):
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, img.size[0] - 1, img.size[1] - 1], outline=(200, 200, 196))
    draw.text((7, 6), title, fill=_INK)
    return img


def render_contact_strip(path, result, n_cruise_frames=6, motion_name=None):
    """Orthographic contact strip: scan + centerline, rest + skeleton, 6 posed frames.

    Row 1 -- the mesh AS SCANNED, top (XY) and side (XZ), with the extracted
    centerline drawn through it.  Row 2 -- the de-bent REST pose, same two views,
    with the skeleton overlaid (spine dark, fin bones warm).  Row 3 -- six evenly
    spaced frames of the first looping clip, skinned through ``rig.lbs``, top view,
    all at one shared scale so the travelling wave is legible.
    """
    from PIL import Image, ImageDraw

    if motion_name is None:
        for candidate in ("cruise",) + tuple(result.clips):
            if candidate in result.clips and candidate != AS_SCANNED_NAME:
                motion_name = candidate
                break
    clip = result.clips.get(motion_name) if motion_name else None

    bent_v = np.asarray(result.mesh.vertices, dtype=float)
    rest_v = np.asarray(result.straight_mesh.vertices, dtype=float)
    faces = np.asarray(result.straight_mesh.faces, dtype=np.int64)
    sk = result.skeleton

    posed = []
    posed_spines = []
    idx = np.zeros(0, dtype=int)
    if clip is not None:
        idx = np.linspace(0, clip.num_frames - 1, n_cruise_frames).round().astype(int)
        spine_idx = np.asarray(sk.spine_indices, dtype=int)
        for i in idx:
            rot = rig.quat_to_rotmat(clip.quats[i])
            posed.append(rig.lbs(rest_v, result.weights, sk, rot))
            posed_spines.append(rig.posed_joints(sk, rot)[spine_idx])

    bent_box = (bent_v.min(axis=0), bent_v.max(axis=0))
    rest_stack = [rest_v, sk.joints] + posed
    rest_box = (np.min([a.min(axis=0) for a in rest_stack], axis=0),
                np.max([a.max(axis=0) for a in rest_stack], axis=0))

    panels = []

    for axes, label in (((0, 1), "top XY"), ((0, 2), "side XZ")):
        img, draw, proj, _ = _panel("", bent_box[0], bent_box[1], axes)
        _draw_mesh(draw, bent_v, faces, proj)
        cl = proj(result.centerline)
        draw.line([tuple(p) for p in cl], fill=(196, 62, 48), width=3)
        for p in cl[::8]:
            draw.ellipse([p[0] - 2.5, p[1] - 2.5, p[0] + 2.5, p[1] + 2.5],
                         fill=(196, 62, 48))
        panels.append(("row1", _finish_panel(img, "as scanned (bent) + centerline, %s" % label)))

    for axes, label in (((0, 1), "top XY"), ((0, 2), "side XZ")):
        img, draw, proj, _ = _panel("", rest_box[0], rest_box[1], axes)
        _draw_mesh(draw, rest_v, faces, proj, tint=(0.80, 0.80, 0.78))
        jpx = proj(sk.joints)
        for j in range(sk.num_joints):
            p = int(sk.parents[j])
            if p < 0:
                continue
            colour = (28, 74, 138) if sk.kinds[j] == "spine" else (214, 122, 30)
            draw.line([tuple(jpx[p]), tuple(jpx[j])], fill=colour,
                      width=4 if sk.kinds[j] == "spine" else 2)
        for j in range(sk.num_joints):
            p = jpx[j]
            r = 3.5 if sk.kinds[j] == "spine" else 2.5
            draw.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=(16, 16, 20))
        panels.append(("row2", _finish_panel(img, "de-bent REST pose + skeleton, %s" % label)))

    strip_panel = (int(_PANEL[0] * 4 / max(n_cruise_frames, 1)), _PANEL[1])
    for k, verts in enumerate(posed):
        img, draw, proj, _ = _panel("", rest_box[0], rest_box[1], (0, 1), size=strip_panel)
        _draw_mesh(draw, verts, faces, proj, tint=(0.70, 0.78, 0.74))
        # the posed spine makes the travelling wave legible; the mesh alone at a
        # 0.11 BL tail amplitude reads as a rigid swing.
        spx = proj(posed_spines[k])
        draw.line([tuple(p) for p in spx], fill=(196, 62, 48), width=2)
        for p in spx:
            draw.ellipse([p[0] - 2, p[1] - 2, p[0] + 2, p[1] + 2], fill=(150, 30, 24))
        panels.append(("row3", _finish_panel(
            img, "%s f%03d  t=%.2fs" % (motion_name or "pose", int(idx[k]),
                                        float(clip.times[idx[k]])))))

    rows = [[img for tag, img in panels if tag == want] for want in ("row1", "row2", "row3")]
    rows = [r for r in rows if r]

    gap, margin, header = 8, 10, 26
    width = max(sum(im.size[0] for im in r) + gap * (len(r) - 1) for r in rows) + 2 * margin
    height = header + sum(max(im.size[1] for im in r) for r in rows) + gap * (len(rows) - 1) + 2 * margin
    canvas = Image.new("RGB", (width, height), (238, 238, 234))
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 8), "sevengill rig contact strip -- %d verts, %d joints, "
                           "chart length %.4f, %d fins"
              % (len(rest_v), sk.num_joints, result.centerline_info["length"],
                 len(sk.fins)), fill=_INK)
    y = header + margin
    for row in rows:
        x = margin
        for im in row:
            canvas.paste(im, (x, y))
            x += im.size[0] + gap
        y += max(im.size[1] for im in row) + gap
    canvas.save(str(path))
    return str(path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_pitch(value):
    if value is None or str(value).strip().upper() in ("AUTO", "NONE", ""):
        return None
    return float(value)


def build_parser():
    ap = argparse.ArgumentParser(
        prog="rig_sevengill.py",
        description="De-bend, rig and animate an elongate-body GLB (sevengill shark).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="example:\n"
               "  python rig_sevengill.py --glb scan.glb --out rigged.glb \\\n"
               "      --motion cruise,turn,escape --keep-bent --report out/report",
    )
    ap.add_argument("--glb", required=True, help="input mesh (GLB/OBJ/PLY)")
    ap.add_argument("--out", required=True, help="output skinned GLB")
    ap.add_argument("--motion", default=",".join(DEFAULT_MOTIONS),
                    help="comma-separated modes: %s" % ",".join(
                        m for m in motion.MODES if motion.MODE_CONFIG[m]["implemented"]))
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--seconds", type=float, default=4.0,
                    help="nominal clip length; looping modes round to whole tail beats")
    ap.add_argument("--voxel-pitch", default="AUTO",
                    help="voxel pitch in world units, or AUTO (= max extent / 128)")
    ap.add_argument("--keep-bent", action="store_true",
                    help="also emit the '%s' clip reproducing the input pose" % AS_SCANNED_NAME)
    ap.add_argument("--report", default=None, help="directory for the diagnostic dump")
    ap.add_argument("-n", "--n-stations", type=int, default=64)
    ap.add_argument("--up", type=float, nargs=3, default=(0.0, 0.0, 1.0),
                    help="DORSAL direction of the input mesh (default +Z)")
    ap.add_argument("--fin-blend-dist", type=float, default=None,
                    help="fin-base weight blend width in metres (default: --fin-blend-rings edge rings)")
    ap.add_argument("--caudal-lobes", choices=("fins", "auto", "body"), default="fins",
                    help="fins (default): every caudal island is a driven fin; auto: a caudal island "
                         "that is a lobe (axial >= %.0fx radial extent) rides the spine as body instead, "
                         "a blade keeps its fin joints; body: fold them all" % LOBE_ASPECT)
    ap.add_argument("--hook-turn-mult", type=float, default=3.0,
                    help="trim a terminal hook whose sagittal turn per station exceeds this "
                         "multiple of the body's median (and 5 deg); 0 disables")
    ap.add_argument("--core-radius-frac", type=float, default=0.17,
                    help="thick-core threshold keeping the medial path out of the fins")
    ap.add_argument("--sigma", type=float, default=None,
                    help="Gaussian skin-weight falloff width in world units")
    ap.add_argument("--fin-blend-rings", type=int, default=rig.DEFAULT_FIN_BLEND_RINGS,
                    help="mesh edge rings over which the fin-root weight ramps out "
                         "into the body (1 = hard seam)")
    ap.add_argument("--precaudal-fraction", type=float,
                    default=rig.DEFAULT_PRECAUDAL_FRACTION)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-validate", action="store_true")
    ap.add_argument("-q", "--quiet", action="store_true")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    modes = [m.strip() for m in args.motion.split(",") if m.strip()]
    unknown = [m for m in modes if m not in motion.MODE_CONFIG]
    if unknown:
        raise SystemExit("unknown motion mode(s): %s (known: %s)"
                         % (", ".join(unknown), ", ".join(motion.MODES)))
    return run_pipeline(
        args.glb,
        out=args.out,
        motions=modes,
        fps=args.fps,
        seconds=args.seconds,
        voxel_pitch=_parse_pitch(args.voxel_pitch),
        n_stations=args.n_stations,
        up=tuple(args.up),
        core_radius_frac=args.core_radius_frac,
        hook_turn_mult=args.hook_turn_mult,
        caudal_lobes=args.caudal_lobes,
        fin_blend_dist=args.fin_blend_dist,
        sigma=args.sigma,
        fin_blend_rings=args.fin_blend_rings,
        precaudal_fraction=args.precaudal_fraction,
        keep_bent=args.keep_bent,
        report=args.report,
        validate=not args.no_validate,
        seed=args.seed,
        verbose=not args.quiet,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
