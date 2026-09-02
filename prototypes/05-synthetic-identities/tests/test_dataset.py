"""End-to-end tests for the synthetic sevengill dataset generator.

The point of this module is NOT to re-test the five engine modules -- each has
its own suite.  It tests the two things only the integration can break:

1. THE DOWNSTREAM CONTRACT.  ``prototypes/01-melops-ablation`` must read the
   corpus with zero edits.  So this file does not assert against a
   reimplementation of the contract, it imports the real ``melops_data`` and
   runs the real ``run_ablation.py`` / ``diagnose.py`` as SUBPROCESSES, the
   way a user would.  A test that mimics the loader would pass while the
   loader failed.
2. THE EXCLUSION GUARANTEE.  "excluding the eye and mouth" has to hold in the
   delivered pixels, at whatever pose and view angle the frame happened to
   draw -- not merely in the chart the pattern was sampled in.  So the eye and
   mouth are located through the per-pixel chart GT and the delivered identity
   MASK is checked there.

One 10-individual corpus is generated once per session and shared; the
generation is the only slow step (~22 s at the test resolution).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_MELOPS = os.path.abspath(os.path.join(_ROOT, "..", "01-melops-ablation"))

import exclusions  # noqa: E402
import make_dataset  # noqa: E402
import pattern  # noqa: E402

pytestmark = pytest.mark.skipif(
    not os.path.isdir(_MELOPS),
    reason="prototypes/01-melops-ablation not present",
)


# Small enough to stay well inside the 90 s budget, large enough that the
# open-set split has a gallery, known queries and novel queries in every arm.
CORPUS_KW = dict(
    n_individuals=10,
    sightings_per_individual=6,
    years=4,
    resolution=(128, 256),
    tex_size=96,
    chart_resolution=(80, 160),
    n_spots=180,
    n_stations=48,
    n_around=32,
    shadow_map_size=256,
    seed=0,
)


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    """Generate the shared 10-individual corpus once. Returns (root, summary)."""
    root = str(tmp_path_factory.mktemp("corpus"))
    summary = make_dataset.generate(root, **CORPUS_KW)
    return root, summary


@pytest.fixture(scope="module")
def melops_data():
    """The real downstream loader, imported from prototype 01."""
    if _MELOPS not in sys.path:
        sys.path.insert(0, _MELOPS)
    import melops_data as md

    return md


def _run(script, args, cwd=_MELOPS, timeout=300):
    """Run one of prototype 01's CLIs as a subprocess, as a user would."""
    cmd = [sys.executable, os.path.join(_MELOPS, script)] + list(args)
    proc = subprocess.run(cmd, cwd=cwd, timeout=timeout,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = proc.stdout.decode("utf-8", "replace")
    assert proc.returncode == 0, "%s failed:\n%s" % (script, out[-4000:])
    return out


# ---------------------------------------------------------------------------
# 1. The generator produced a usable corpus at all
# ---------------------------------------------------------------------------

def test_corpus_generated_with_the_expected_shape(corpus):
    root, summary = corpus
    assert summary["n_images"] >= 30
    assert summary["n_individuals"] == 10
    for name in ("metadata.csv", "Melops_metadata.txt", "truth.jsonl",
                 "dataset.json"):
        assert os.path.exists(os.path.join(root, name)), name
    # Both flanks, or the cross-orientation arm has nothing to evaluate.
    assert summary["sides"]["L"] > 0 and summary["sides"]["R"] > 0
    # Singletons are drawn on purpose: the open-set split must survive them.
    assert summary["images_per_identity"]["min"] == 1
    assert summary["images_per_identity"]["max"] >= 4


def test_generation_is_deterministic_in_the_seed(tmp_path):
    """Same seed -> byte-identical metadata. Different seed -> different."""
    kw = dict(CORPUS_KW)
    kw.update(n_individuals=2, sightings_per_individual=2, resolution=(64, 128),
              tex_size=48, chart_resolution=(48, 96), n_spots=80,
              n_stations=32, n_around=20, shadow_map_size=128, save_gt=False)
    a = str(tmp_path / "a")
    b = str(tmp_path / "b")
    c = str(tmp_path / "c")
    make_dataset.generate(a, **kw)
    make_dataset.generate(b, **kw)
    kw["seed"] = 1
    make_dataset.generate(c, **kw)
    read = lambda d: open(os.path.join(d, "metadata.csv"), "rb").read()  # noqa: E731
    assert read(a) == read(b)
    assert read(a) != read(c)


# ---------------------------------------------------------------------------
# 2. THE DOWNSTREAM CONTRACT -- the real loader, all three bbox arms
# ---------------------------------------------------------------------------

def test_load_melops_accepts_all_three_bbox_arms(corpus, melops_data):
    root, summary = corpus
    for arm in melops_data.BBOX_PARTS:
        df = melops_data.load_melops(root, bbox=arm)
        assert len(df) == summary["n_images"]
        assert (df["orientation"] == df["side"]).all()
        assert set(df["side"]) <= {"L", "R"}
        assert not df["image_id"].duplicated().any()
        # Every bbox column survives, not just the requested one.
        for col in ("bbox", "bbox_body", "bbox_head", "bbox_headless"):
            assert len(df[col].iloc[0]) == 4


def test_every_crop_is_loadable_and_boxes_lie_inside_the_image(corpus, melops_data):
    """LTWH in body-crop pixels, and head/headless INSIDE the body crop.

    ``melops_data.load_crop`` applies bbox_head to the stored image, so a head
    box expressed in full-frame coordinates would crop the wrong thing while
    still loading without error.  This asserts the containment directly.
    """
    root, _ = corpus
    for arm in melops_data.BBOX_PARTS:
        df = melops_data.load_melops(root, bbox=arm)
        for _, row in df.iterrows():
            w, h = Image.open(os.path.join(root, row["path"])).size
            l, t, bw, bh = [float(v) for v in row["bbox"]]
            assert bw >= 2 and bh >= 2, (arm, row["image_id"], row["bbox"])
            assert -0.5 <= l and l + bw <= w + 0.5, (arm, row["image_id"])
            assert -0.5 <= t and t + bh <= h + 0.5, (arm, row["image_id"])
            crop = melops_data.load_crop(root, row)
            assert crop.size[0] >= 1 and crop.size[1] >= 1
        # The body box is the whole stored image, by construction.
        if arm == "body":
            for _, row in df.iterrows():
                w, h = Image.open(os.path.join(root, row["path"])).size
                assert list(row["bbox"]) == [0.0, 0.0, float(w), float(h)]


def test_head_box_follows_the_animal_not_the_image(corpus, melops_data):
    """The head/headless cut is ANATOMICAL, so it flips with the flank shown.

    The camera moves to the other side to shoot the other flank (the image is
    never mirrored), so the snout swaps image sides.  A head box that always
    sat on the same image side would mean the split was being guessed from the
    silhouette rather than cut in arc length through the chart GT.
    """
    root, _ = corpus
    df = melops_data.load_melops(root, bbox="head")
    centre_frac = []
    for _, row in df.iterrows():
        w, _h = Image.open(os.path.join(root, row["path"])).size
        l, _t, bw, _bh = [float(v) for v in row["bbox"]]
        centre_frac.append(((l + 0.5 * bw) / w, row["side"]))
    left = [c for c, s in centre_frac if s == "L"]
    right = [c for c, s in centre_frac if s == "R"]
    assert left and right
    # One flank puts the head near x=1, the other near x=0; which is which is
    # fixed by the camera placement, so assert only that the two groups
    # SEPARATE cleanly and that neither group straddles the middle.
    near, far = sorted([left, right], key=lambda g: float(np.mean(g)))
    assert max(near) < 0.4, max(near)
    assert min(far) > 0.6, min(far)


def test_length_sidecar_covers_every_image(corpus):
    """``readout_length_controlled.py`` reads Melops_metadata.txt this way."""
    import pandas as pd

    root, _ = corpus
    meta = pd.read_csv(os.path.join(root, "Melops_metadata.txt"),
                       sep=None, engine="python")
    assert list(meta.columns[:2]) == ["filename_year", "length"]
    lengths = meta.set_index(meta["filename_year"].astype(str))["length"]
    ids = pd.read_csv(os.path.join(root, "metadata.csv"))["image_id"].astype(str)
    mapped = ids.map(lengths)
    assert mapped.notna().all()
    assert (mapped > 0).all()


def test_the_recorded_length_carries_measurement_error(corpus):
    """The sidecar holds an ESTIMATE; truth.jsonl holds the animal."""
    root, summary = corpus
    truth = {}
    with open(os.path.join(root, "truth.jsonl")) as f:
        for line in f:
            rec = json.loads(line)
            truth[rec["image_id"]] = rec
    assert summary["args"]["length_noise"] == make_dataset.LENGTH_MEASUREMENT_RSD

    import pandas as pd
    meta = pd.read_csv(os.path.join(root, "Melops_metadata.txt"),
                       sep=None, engine="python")
    recorded = dict(zip(meta["filename_year"].astype(str),
                        meta["length"].astype(float)))
    diffs = []
    for image_id, rec in truth.items():
        assert rec["length_mm"] == pytest.approx(rec["length_cm"] * 10.0)
        # the sidecar carries the RECORDED value, not the true one
        assert recorded[image_id] == pytest.approx(rec["measured_length_mm"],
                                                   abs=0.05)
        diffs.append(rec["measured_length_mm"] / rec["length_mm"] - 1.0)
    assert any(abs(d) > 1e-9 for d in diffs), "no measurement error was applied"
    # realised relative sd within a factor of two of the requested one
    rsd = float(np.std(diffs))
    assert 0.5 * make_dataset.LENGTH_MEASUREMENT_RSD < rsd < 2.0 * \
        make_dataset.LENGTH_MEASUREMENT_RSD, rsd


def _one_nn_identity(values, identities):
    """Leave-one-out 1-NN identity accuracy from a single scalar per image."""
    values = np.asarray(values, dtype=np.float64)
    identities = np.asarray(identities)
    dist = np.abs(values[:, None] - values[None, :])
    np.fill_diagonal(dist, np.inf)
    return float((identities[dist.argmin(axis=1)] == identities).mean())


def test_length_alone_is_not_an_identity_oracle():
    """A near-unique length is a LABEL, not a body.

    Each animal draws one initial length and then only grows, so without
    measurement error the recorded length identifies it almost perfectly and
    every length-stratified readout downstream
    (``readout_length_controlled.py``) is measuring that label.  The bar is
    5x chance on a 40-animal set; the exact realised value is quoted in
    README.md.

    No rendering is needed -- the lengths come straight from the generator's
    own deterministic timeline -- so this stays cheap despite being a
    40-animal test.
    """
    n_individuals = 40
    context = make_dataset.build_pattern_context()
    identities, true_mm, recorded_mm = [], [], []
    for index in range(n_individuals):
        identity, _length, states = make_dataset.individual_timeline(
            context, 0, index)
        for sighting, (_date, _side, ind) in enumerate(states):
            identities.append(identity)
            true_mm.append(float(ind.length_cm) * 10.0)
            recorded_mm.append(make_dataset.measured_length_mm(
                ind.length_cm, 0, index, sighting))

    chance = 1.0 / n_individuals
    oracle = _one_nn_identity(true_mm, identities)
    measured = _one_nn_identity(recorded_mm, identities)
    # the TRUE length really is an oracle -- that is why the noise exists
    assert oracle > 10.0 * chance, oracle
    assert measured < 5.0 * chance, (measured, chance)
    # and turning the knob off must put the oracle back, or the test is
    # asserting something the knob does not control
    exact = _one_nn_identity(
        [make_dataset.measured_length_mm(v / 10.0, 0, 0, 0, rsd=0.0)
         for v in true_mm], identities)
    assert exact == pytest.approx(oracle)


# ---------------------------------------------------------------------------
# 3. The real CLIs, as subprocesses
# ---------------------------------------------------------------------------

def test_run_ablation_produces_results_and_a_verdict(corpus, tmp_path):
    root, _ = corpus
    out = str(tmp_path / "abl")
    log = _run("run_ablation.py",
               ["--data", "melops", "--root", root, "--backbone", "hist",
                "--out", out])
    results_path = os.path.join(out, "results.json")
    assert os.path.exists(results_path), log[-2000:]
    assert "VERDICT:" in log, log[-2000:]

    results = json.load(open(results_path))
    assert "verdict" in results
    for arm in ("head", "body", "headless", "cross_orientation"):
        assert arm in results["arms"], sorted(results["arms"])
    # A histogram backbone on turbid frames is weak, but it must beat chance:
    # the corpus must actually carry identity, or the whole engine is a
    # random-image generator with a schema attached.
    body = results["arms"]["body"]
    assert body["rank1"] > 1.0 / max(body["n_gallery"], 1)


def test_diagnose_populates_the_recapture_gap_buckets(corpus, tmp_path):
    """The gap curve is the pattern-STABILITY readout; it needs real spread.

    ``plan_sightings`` draws gaps log-uniformly precisely so that the buckets
    are populated by construction rather than by luck.
    """
    root, _ = corpus
    out = str(tmp_path / "diag")
    log = _run("diagnose.py",
               ["--data", "melops", "--root", root, "--backbone", "hist",
                "--arm", "body", "--out", out])
    diag_path = os.path.join(out, "diagnostics.json")
    assert os.path.exists(diag_path), log[-2000:]

    diag = json.load(open(diag_path))
    buckets = diag["recapture_gap"]["buckets"]
    populated = [b for b in buckets if b["n"] > 0]
    assert len(populated) >= 3, [(b["bucket_days"], b["n"]) for b in buckets]
    # Multi-year resights specifically: the drift model is only exercised if
    # the corpus reaches past a year.
    assert any(b["n"] > 0 for b in buckets
               if b["bucket_days"] in ("366-730", "731+"))


# ---------------------------------------------------------------------------
# 4. THE EXCLUSION GUARANTEE, checked in the delivered pixels
# ---------------------------------------------------------------------------

def _identity_mask(root, image_id):
    path = os.path.join(root, "masks", image_id + "_identity.png")
    assert os.path.exists(path), path
    return np.asarray(Image.open(path)) > 127


def _chart_gt(root, image_id):
    d = np.load(os.path.join(root, "gt", image_id + ".npz"))
    return d["chart_s"], d["chart_phi"]


@pytest.fixture(scope="module")
def eye_mouth_regions():
    schema = exclusions.load_schema(pattern.DEFAULT_SCHEMA_PATH)
    stations = exclusions.default_stations(schema)
    regions = exclusions.exclusion_regions(schema, stations=stations)
    by_name = {r.name: r for r in regions}
    assert {"eye_left", "eye_right", "mouth_jaw"} <= set(by_name)
    return by_name


def test_identity_masks_exist_for_every_image(corpus):
    root, _ = corpus
    ids = [json.loads(l)["image_id"] for l in
           open(os.path.join(root, "truth.jsonl"))]
    assert ids
    for image_id in ids:
        mask = _identity_mask(root, image_id)
        rgb = Image.open(os.path.join(root, "body", image_id + ".png"))
        assert mask.shape == (rgb.size[1], rgb.size[0])


def test_identity_mask_is_false_at_the_eye_and_the_mouth(corpus, eye_mouth_regions):
    """The binding requirement, checked where it matters: in the pixels.

    For every image, every pixel whose chart GT falls inside the eye or the
    mouth/jaw region must be excluded from the identity mask -- at whatever
    pose, flank and camera jitter that frame drew.  The test also asserts the
    check is NOT VACUOUS: the eye and the mouth must actually be visible and
    found in some frames, or a generator that rendered no head at all would
    pass.
    """
    root, _ = corpus
    ids = [json.loads(l)["image_id"] for l in
           open(os.path.join(root, "truth.jsonl"))]
    eye_l = eye_mouth_regions["eye_left"]
    eye_r = eye_mouth_regions["eye_right"]
    mouth = eye_mouth_regions["mouth_jaw"]

    n_eye_px = 0
    n_mouth_px = 0
    n_frames_with_eye = 0
    for image_id in ids:
        s, phi = _chart_gt(root, image_id)
        on_body = np.isfinite(s)
        if not on_body.any():
            continue
        mask = _identity_mask(root, image_id)
        ss = np.where(on_body, s, np.nan)
        pp = np.where(on_body, phi, np.nan)
        with np.errstate(invalid="ignore"):
            in_eye = (eye_l.contains(ss, pp) | eye_r.contains(ss, pp)) & on_body
            in_mouth = mouth.contains(ss, pp) & on_body
        # THE ASSERTION: no eye or mouth pixel is ever an identity pixel.
        assert not (mask & in_eye).any(), "eye leaked into identity: %s" % image_id
        assert not (mask & in_mouth).any(), "mouth leaked into identity: %s" % image_id
        n_eye_px += int(in_eye.sum())
        n_mouth_px += int(in_mouth.sum())
        n_frames_with_eye += int(in_eye.any())

    assert n_eye_px > 0, "no eye pixel was ever visible -- the test is vacuous"
    assert n_mouth_px > 0, "no mouth pixel was ever visible -- the test is vacuous"
    assert n_frames_with_eye >= 5, n_frames_with_eye


def test_no_exclusion_region_at_all_leaks_into_identity(corpus, eye_mouth_regions):
    """The same guarantee, widened to nares and the seven gill slits.

    The render mask is built from the whole region list, so if the eye holds
    and a naris does not, the failure is in one region's geometry rather than
    in the sampling -- worth separating.
    """
    root, _ = corpus
    ids = [json.loads(l)["image_id"] for l in
           open(os.path.join(root, "truth.jsonl"))]
    seen = {}
    for image_id in ids:
        s, phi = _chart_gt(root, image_id)
        on_body = np.isfinite(s)
        if not on_body.any():
            continue
        mask = _identity_mask(root, image_id)
        ss = np.where(on_body, s, np.nan)
        pp = np.where(on_body, phi, np.nan)
        for name, region in eye_mouth_regions.items():
            with np.errstate(invalid="ignore"):
                inside = region.contains(ss, pp) & on_body
            assert not (mask & inside).any(), "%s leaked in %s" % (name, image_id)
            seen[name] = seen.get(name, 0) + int(inside.sum())
    # Every region must have been observed somewhere, or it went untested.
    for name in eye_mouth_regions:
        assert seen.get(name, 0) > 0, "%s never visible -- untested" % name


def test_dilate_chart_mask_is_8_connected_wraps_phi_and_clamps_s():
    """Rows periodic, columns clamped, and the DIAGONAL must be covered.

    The diagonal is the whole point: nearest-neighbour lookup misplaces a
    point by half a cell in each axis independently, so a corner pixel lands
    on the diagonal neighbour. A 4-connected dilation misses it, and did.
    """
    m = np.zeros((8, 6), dtype=bool)
    m[3, 2] = True
    d = make_dataset.dilate_chart_mask(m, n_cells=1)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            assert d[3 + di, 2 + dj], "3x3 neighbourhood incomplete at %d,%d" % (di, dj)
    assert not d[3, 4] and not d[5, 2]

    m2 = np.zeros((8, 6), dtype=bool)
    m2[0, 0] = True                      # phi seam + snout corner
    d2 = make_dataset.dilate_chart_mask(m2, n_cells=1)
    assert d2[7, 0] and d2[7, 1], "phi must wrap around the +-pi seam"
    # s must NOT wrap: a mark at the snout must not appear at the caudal tip.
    assert not d2[0, 5] and not d2[7, 5], "s must clamp, not wrap"
    # A dilation is always a superset.
    assert (d >= m).all() and (d2 >= m2).all()


def test_identity_mask_is_the_exact_reduction_of_the_body_mask(corpus):
    """identity == visible skin, minus exclusion, shadow and occlusion.

    Restating render.py's contract at the dataset level catches a crop that
    slices the masks inconsistently -- the failure that would silently ship
    masks off by a pixel from their image.
    """
    root, _ = corpus
    ids = [json.loads(l)["image_id"] for l in
           open(os.path.join(root, "truth.jsonl"))]
    for image_id in ids[:12]:
        d = np.load(os.path.join(root, "gt", image_id + ".npz"))
        expect = (d["visible_skin"] & np.isfinite(d["chart_s"])
                  & ~d["exclusion"] & ~d["shadow"] & ~d["occlusion"])
        np.testing.assert_array_equal(_identity_mask(root, image_id), expect)


def test_shadow_and_occlusion_knobs_are_separable(corpus):
    """The canopy casts without occluding; in-frame kelp occludes.

    If a raised ``--shadow`` also raised occlusion, the two nuisance factors
    could never be ablated apart -- which is most of what a synthetic corpus
    is for.
    """
    root, _ = corpus
    truth = [json.loads(l) for l in open(os.path.join(root, "truth.jsonl"))]
    canopy_only = [t for t in truth
                   if t["scene"]["canopy_caster"] and not t["scene"]["kelp"]
                   and not t["scene"]["shark_occluder"]]
    assert canopy_only, "no canopy-only frame was drawn"
    assert all(t["px"]["occlusion"] == 0 for t in canopy_only)
    assert any(t["px"]["cast_shadow"] > 0 for t in canopy_only)
    # And something, somewhere, was actually occluded.
    assert any(t["px"]["occlusion"] > 0 for t in truth)


# ---------------------------------------------------------------------------
# 5. The flanks are rendered, never mirrored
# ---------------------------------------------------------------------------

def test_the_two_flanks_are_not_mirror_images():
    """L and R come from MOVING THE CAMERA, so they show different skin.

    A mirrored corpus teaches a model that a left flank and its reflection are
    the same animal, which is false (Schema S1: cross-flank Rank-1 fell to
    0.70% zero-shot).  Same individual, same pose, same light, same date --
    only the camera side differs -- and the two frames must not be flips of
    each other.
    """
    model = make_dataset.build_model(n_stations=40, n_around=28, tex_size=64)
    schema = exclusions.load_schema(pattern.DEFAULT_SCHEMA_PATH)
    stations = exclusions.default_stations(schema)
    regions = exclusions.exclusion_regions(schema, stations=stations)
    ind = pattern.Individual.generate(seed=7, identity="mirror_probe",
                                      date="2020-01-01", length_cm=220.0,
                                      regions=regions)
    texture, _chart, _spots = model_texture(model, ind)

    frames = {}
    for side in ("L", "R"):
        rng = np.random.default_rng([99, 0])
        scene = make_dataset.draw_scene(rng, side, occlusion=0.0, shadow=0.0,
                                        turbidity=0.0)
        out = make_dataset.render_sighting(model, texture, scene,
                                           resolution=(96, 192), seed=5,
                                           shadow_map_size=128)
        frames[side] = out

    # Both frames must actually show a body, or the comparison is empty.
    for side, out in frames.items():
        assert out["visible_skin"].sum() > 200, side

    # The chart GT proves it: the L frame's visible phi is on the +pi/2 side,
    # the R frame's on the -pi/2 side. A mirror would show the SAME phi.
    med = {}
    for side, out in frames.items():
        phi = out["chart_phi"][out["visible_skin"]]
        med[side] = float(np.nanmedian(phi))
    assert med["L"] > 0.6, med
    assert med["R"] < -0.6, med

    # And the images are not flips of one another.
    l_rgb = frames["L"]["rgb"]
    r_rgb = frames["R"]["rgb"]
    assert not np.allclose(l_rgb, r_rgb[:, ::-1], atol=0.02)


def model_texture(model, ind, date=None):
    return make_dataset.chart_to_texture(model, ind, date=date,
                                        chart_resolution=(64, 128))
