"""End-to-end synthetic demo: build a bent sevengill, de-bend it, rig it, animate it.

The real input for this prototype is a textured Meshy-AI GLB of a live sevengill
whose rest pose is a strong lateral C.  That file is not available in this
session, so the demo manufactures the same problem with GROUND TRUTH attached:
:func:`synth.make_sevengill` builds a straight, textured, UV-mapped sevengill with
seven gill slits and the right fin set, and :func:`synth.bend` transports it onto
a ~120 degree C-curve through the same tube coordinates the pipeline will later
have to recover.  Every number the demo prints is therefore checkable against a
quantity we know exactly.

Run::

    python demo.py [--out DIR] [--seconds 4] [--fps 30]

Outputs (default ``demo/`` next to this file):

    sevengill_synthetic_bent.glb   the INPUT: the C-posed, textured mesh
    sevengill_rigged.glb           the OUTPUT: skinned, with cruise / turn /
                                   escape / as_scanned clips
    sevengill_rest.glb             the de-bent rest pose alone, for inspection
    report/                        centerline.json, fins.json, skeleton.json,
                                   weights.json, contact_strip.png

The printed round-trip RMS is the de-bent mesh against the straight mesh the
bend started from, after removing a rigid translation (the chart's origin is
arbitrary: centerline extraction trims a few percent off each end, so the
recovered chart is a slightly shorter tube than the true one).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gltf_export  # noqa: E402
import mesh3d  # noqa: E402
import rig  # noqa: E402
import rig_sevengill  # noqa: E402
import synth  # noqa: E402

DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo")
DEMO_MOTIONS = ("cruise", "turn", "escape")
TURN_DEG = 120.0
SEED = 0


def _rms_after_translation(a, b):
    """(rms, max) distance between two vertex sets with the mean offset removed."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    d = d - d.mean(axis=0, keepdims=True)
    n = np.linalg.norm(d, axis=1)
    return float(np.sqrt(np.mean(n ** 2))), float(n.max())


def run(out_dir=DEFAULT_OUT, seconds=4.0, fps=30.0, n_stations=64, seed=SEED,
        validate=True):
    """Build, bend, rig, animate and report.  Returns a dict of measured numbers."""
    os.makedirs(out_dir, exist_ok=True)
    report_dir = os.path.join(out_dir, "report")
    t_start = time.time()

    print("=" * 78)
    print("prototype 04 -- sevengill de-bend + rig + swim, synthetic end-to-end demo")
    print("=" * 78)

    # -- 1. ground truth ---------------------------------------------------
    straight_truth = synth.make_sevengill(seed=seed)
    truth_centerline = np.asarray(straight_truth.metadata["centerline"], dtype=float)
    total_length = float(mesh3d.arc_length(truth_centerline)[-1])
    truth_labels = np.asarray(straight_truth.metadata["vertex_labels"])
    bent, bend_info = synth.bend(
        straight_truth, synth.c_curve(total_length, TURN_DEG, n_stations)
    )
    print("\n[1] synthetic sevengill: %d verts, %d faces, %d fins, texture %s"
          % (len(bent.vertices), len(bent.faces),
             len(straight_truth.metadata["fins"]),
             "yes" if getattr(bent.visual, "uv", None) is not None else "no"))
    print("    body length %.4f, tube centerline length %.4f, bent into a %.0f deg C"
          % (float(straight_truth.extents[0]), total_length, TURN_DEG))

    bent_path = os.path.join(out_dir, "sevengill_synthetic_bent.glb")
    synth.export_glb(bent, bent_path)
    print("    input  -> %s (%.1f kB)" % (bent_path, os.path.getsize(bent_path) / 1024.0))

    # -- 2. the pipeline, exactly as the CLI runs it ------------------------
    print("\n[2] pipeline (rig_sevengill.run_pipeline, the same code path as the CLI)")
    rigged_path = os.path.join(out_dir, "sevengill_rigged.glb")
    result = rig_sevengill.run_pipeline(
        bent_path,
        out=rigged_path,
        motions=DEMO_MOTIONS,
        fps=fps,
        seconds=seconds,
        n_stations=n_stations,
        keep_bent=True,
        report=report_dir,
        validate=validate,
        seed=seed,
        verbose=True,
    )

    rest_path = os.path.join(out_dir, "sevengill_rest.glb")
    synth.export_glb(result.straight_mesh, rest_path)

    # -- 3. de-bend accuracy against ground truth ---------------------------
    pitch = float(result.centerline_info["pitch"])
    recovered = np.asarray(result.straight_mesh.vertices, dtype=float)
    reference = np.asarray(straight_truth.vertices, dtype=float)
    rms_all, max_all = _rms_after_translation(recovered, reference)
    body = truth_labels == "body"
    rms_body, max_body = _rms_after_translation(recovered[body], reference[body])

    # centerline vs the exact curve the bend used
    from scipy.spatial import cKDTree
    dense = mesh3d.resample_polyline(bend_info["centerline"], 4000)
    cl_dev, _ = cKDTree(dense).query(result.centerline)

    print("\n[3] de-bend round trip vs ground truth  (BL = %.4f world units, "
          "voxel pitch = %.5f = %.3f%% BL)" % (total_length, pitch,
                                               100.0 * pitch / total_length))
    print("    all vertices   RMS %.4f%% BL (%.2f px)   max %.4f%% BL (%.2f px)"
          % (100.0 * rms_all / total_length, rms_all / pitch,
             100.0 * max_all / total_length, max_all / pitch))
    print("    body only      RMS %.4f%% BL (%.2f px)   max %.4f%% BL (%.2f px)"
          % (100.0 * rms_body / total_length, rms_body / pitch,
             100.0 * max_body / total_length, max_body / pitch))
    print("    centerline vs the exact C-curve: mean %.5f BL (%.2f px), "
          "max %.5f BL (%.2f px)"
          % (cl_dev.mean() / total_length, cl_dev.mean() / pitch,
             cl_dev.max() / total_length, cl_dev.max() / pitch))

    # -- 4. fin labelling vs construction truth -----------------------------
    labels = np.asarray(result.detection.labels)
    print("\n[4] fin labels vs construction truth")
    print("    %-14s %6s %6s %8s %8s" % ("fin", "found", "truth", "purity", "recall"))
    purities = []
    for name in sorted(result.detection.fins):
        idx = np.asarray(result.detection.fins[name]["vertex_indices"], dtype=int)
        truth_mask = truth_labels == name
        hit = int(truth_mask[idx].sum())
        purity = hit / float(max(len(idx), 1))
        recall = hit / float(max(int(truth_mask.sum()), 1))
        purities.append(purity)
        print("    %-14s %6d %6d %7.1f%% %7.1f%%"
              % (name, len(idx), int(truth_mask.sum()), 100.0 * purity, 100.0 * recall))
    mislabelled_body = int(((labels != "body") & (truth_labels == "body")).sum())
    print("    body vertices given a fin label: %d" % mislabelled_body)

    # -- 5. UV / topology preservation --------------------------------------
    same_faces = np.array_equal(np.asarray(bent.faces), np.asarray(result.straight_mesh.faces))
    uv_in = getattr(bent.visual, "uv", None)
    uv_out = getattr(result.straight_mesh.visual, "uv", None)
    # ``uv_out`` has been through the GLB, where UVs are float32 and stored
    # v-flipped, so a v near 0 comes back with ~6e-8 of absolute error and a
    # *relative* error far above np.allclose's default rtol.  Compare on an
    # absolute tolerance a float32 can actually meet.
    uv_same = (uv_in is not None and uv_out is not None
               and np.allclose(uv_in, uv_out, rtol=0.0, atol=1e-6))
    reloaded = mesh3d.load_mesh(rigged_path, report=False)
    uv_rt = getattr(reloaded.visual, "uv", None)
    uv_err = (float(np.abs(np.asarray(uv_rt) - np.asarray(uv_out)).max())
              if uv_rt is not None and len(uv_rt) == len(uv_out) else float("nan"))
    print("\n[5] texture / topology through the de-bend and the GLB round trip")
    print("    faces identical: %s | UVs identical after de-bend (to float32): %s"
          " | UV error after GLB write+reload: %.2e"
          % (same_faces, uv_same, uv_err))

    # -- 6. the rig ---------------------------------------------------------
    sk = result.skeleton
    w = np.asarray(result.weights, dtype=float)
    nz = (w > 0).sum(axis=1)
    print("\n[6] rig: %d joints = %d schema spine + %d fin joints; "
          "weights %d x %d, <= %d influences (mean %.2f), rows sum to 1 (%.1e)"
          % (sk.num_joints, rig.NUM_SPINE_JOINTS, 2 * len(sk.fins),
             w.shape[0], w.shape[1], int(nz.max()), float(nz.mean()),
             float(np.abs(w.sum(axis=1) - 1.0).max())))
    for name in sorted(sk.fins):
        root, _tip = sk.fins[name]
        print("    %-14s root parented to %s" % (name, sk.names[int(sk.parents[root])]))

    # -- 7. as_scanned: does the rig reproduce the scan pose? ---------------
    as_clip = result.clips[rig_sevengill.AS_SCANNED_NAME]
    rot = rig.quat_to_rotmat(as_clip.quats[-1])
    posed = rig.lbs(recovered, w, sk, rot) + as_clip.meta["root_translation"]
    scan_rms, scan_max = _rms_after_translation(posed, np.asarray(bent.vertices, dtype=float))
    joint_err = np.asarray(as_clip.meta["joint_error"], dtype=float)
    print("\n[7] '%s' clip: LBS-posed rig vs the mesh as scanned"
          % rig_sevengill.AS_SCANNED_NAME)
    print("    spine joints land on the scanned centerline to max %.5f BL (%.2f px)"
          % (joint_err.max() / total_length, joint_err.max() / pitch))
    print("    skinned surface vs the scan: RMS %.3f%% BL, max %.3f%% BL "
          "(LBS blending, not chart error)"
          % (100.0 * scan_rms / total_length, 100.0 * scan_max / total_length))

    # -- 8. swimming kinematics --------------------------------------------
    import motion
    cruise = result.clips["cruise"]
    params = cruise.meta["params"]
    amp = motion.tail_tip_amplitude(params, s_j=motion.default_spine_fractions(),
                                    body_length=float(result.centerline_info["length"]))
    strain = motion.implied_skin_strain(params)
    dct = motion.dct_energy_fraction(cruise)
    print("\n[8] swimming kinematics (cruise)")
    print("    %.2f Hz, wavelength %.2f BL, prescribed tail amplitude %.3f BL, "
          "posed %.3f BL (%.2f x)"
          % (params.frequency_hz, params.wavelength_bl, amp["analytic_bl"],
             amp["fk_bl"], amp["fk_over_analytic"]))
    print("    implied longitudinal skin strain: anterior %.1f%%, mid %.1f%%, "
          "posterior %.1f%%  (literature bracket %.1f-%.1f%%)"
          % (100.0 * strain["anterior"], 100.0 * strain["mid_body"],
             100.0 * strain["posterior"],
             100.0 * motion.SKIN_STRAIN_BRACKET[0], 100.0 * motion.SKIN_STRAIN_BRACKET[1]))
    print("    DCT bending-mode energy: %.3f in 4 modes, %.3f in 6"
          % (dct[4], dct[6]))
    for name in sorted(result.clips):
        c = result.clips[name]
        print("    clip %-11s %3d frames  %5.2f s  loop=%-5s" % (name, c.num_frames,
                                                                 c.duration_s, c.loop))

    # -- 9. validators ------------------------------------------------------
    print("\n[9] glTF validation (Khronos gltf-validator)")
    validations = {}
    if validate:
        for label, path in (("input  sevengill_synthetic_bent.glb", bent_path),
                            ("rest   sevengill_rest.glb", rest_path),
                            ("output sevengill_rigged.glb", rigged_path)):
            issues = gltf_export.validate_glb(path)
            validations[path] = issues
            print("    %-38s %d errors, %d warnings"
                  % (label, issues["numErrors"], issues["numWarnings"]))
            for msg in issues.get("messages", []):
                if msg.get("severity", 0) <= 1:
                    print("        [%s] %s" % (msg.get("code"), msg.get("message")))
    else:
        print("    skipped (--no-validate)")

    elapsed = time.time() - t_start
    print("\n[10] outputs in %s  (total %.1f s)" % (out_dir, elapsed))
    for name in sorted(os.listdir(out_dir)):
        p = os.path.join(out_dir, name)
        if os.path.isfile(p):
            print("    %-34s %8.1f kB" % (name, os.path.getsize(p) / 1024.0))
    for name in sorted(os.listdir(report_dir)):
        print("    report/%-27s %8.1f kB"
              % (name, os.path.getsize(os.path.join(report_dir, name)) / 1024.0))
    print("=" * 78)

    return {
        "out_dir": out_dir,
        "report_dir": report_dir,
        "bent_glb": bent_path,
        "rest_glb": rest_path,
        "rigged_glb": rigged_path,
        "result": result,
        "total_length": total_length,
        "voxel_pitch": pitch,
        "rms_all": rms_all,
        "max_all": max_all,
        "rms_body": rms_body,
        "max_body": max_body,
        "centerline_max_dev": float(cl_dev.max()),
        "centerline_mean_dev": float(cl_dev.mean()),
        "fin_purity_min": float(min(purities)) if purities else float("nan"),
        "mislabelled_body": mislabelled_body,
        "uv_roundtrip_error": uv_err,
        "faces_identical": bool(same_faces),
        "as_scanned_joint_error_max": float(joint_err.max()),
        "as_scanned_surface_rms": scan_rms,
        "validations": validations,
        "elapsed_s": elapsed,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("-n", "--n-stations", type=int, default=64)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--no-validate", action="store_true")
    args = ap.parse_args(argv)
    return run(out_dir=args.out, seconds=args.seconds, fps=args.fps,
               n_stations=args.n_stations, seed=args.seed,
               validate=not args.no_validate)


if __name__ == "__main__":  # pragma: no cover
    main()
