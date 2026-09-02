"""Tests for the OSEA contract wrapper (prototype 06 "spot-proxy").

Three groups:

1. ``TestPCAFrame`` -- the body-attached frame is equivariant under rigid
   motions and its *scalars* are invariant under the four-element sign group
   {u -> +-u} x {v -> +-v}, under mirroring, and under uniform scaling.
2. ``TestFeatures`` -- ``features()`` on hand-built detection dicts: values are
   checked against numbers computed by hand, not just "it ran".
3. ``TestDetect``  -- ``detect()`` on a blank image really loads the OSEA
   weights and really returns nothing.  Skipped with an explicit reason when
   the weights are absent.

Run: "MAIN/.venv/bin/python" -m pytest P06/tests -q
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import osea_contract as oc  # noqa: E402


# --------------------------------------------------------------------------- #
# fixtures / builders                                                          #
# --------------------------------------------------------------------------- #
def ellipse_polygon(a=200.0, b=80.0, n=128, cx=0.0, cy=0.0):
    """Closed polygon sampling an axis-aligned ellipse, semi-axes (a, b)."""
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.column_stack([cx + a * np.cos(t), cy + b * np.sin(t)])


def rot(theta_deg):
    t = math.radians(theta_deg)
    return np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])


MIRROR_X = np.array([[-1.0, 0.0], [0.0, 1.0]])


def make_det(poly, spot_xy, sizes=None, confs=None, obstr=None,
             image_size=(1000, 800), body_conf=0.9):
    """Hand-build a detection dict in the exact catalog.db column shapes."""
    poly = np.asarray(poly, dtype=float)
    spot_xy = np.asarray(spot_xy, dtype=float).reshape(-1, 2)
    n = len(spot_xy)
    if sizes is None:
        sizes = np.full(n, 10.0)
    if confs is None:
        confs = np.linspace(0.3, 0.9, n) if n else np.zeros(0)
    spots = []
    for i in range(n):
        s = float(sizes[i])
        cx, cy = float(spot_xy[i, 0]), float(spot_xy[i, 1])
        spots.append({
            "x": round(cx - s / 2, 1), "y": round(cy - s / 2, 1),
            "w": round(s, 1), "h": round(s, 1),
            "cx": round(cx, 1), "cy": round(cy, 1),
            "conf": round(float(confs[i]), 3),
        })
    x0, y0 = poly.min(axis=0)
    x1, y1 = poly.max(axis=0)
    obstr = [np.asarray(o, dtype=float).tolist() for o in (obstr or [])]
    return {
        "body_polygon": poly.tolist(),
        "body_bbox": {"x": int(x0), "y": int(y0), "w": int(x1 - x0), "h": int(y1 - y0)},
        "body_conf": body_conf,
        "obstruction_polygons": obstr,
        "obstruction_count": len(obstr),
        "head_polygon": None, "head_bbox": None, "head_conf": None,
        "spots": spots,
        "spot_count": n,
        "image_width": int(image_size[0]),
        "image_height": int(image_size[1]),
    }


def transform_det(det, M, offset=(0.0, 0.0), scale=1.0):
    """Apply x -> scale * (M @ x) + offset to the polygon, spots and spot sizes.

    Rebuilds the det through ``make_det`` so the result is a legal detection
    dict, and rescales the body bbox consistently.
    """
    off = np.asarray(offset, dtype=float)

    def T(pts):
        pts = np.asarray(pts, dtype=float).reshape(-1, 2)
        return scale * (pts @ np.asarray(M, dtype=float).T) + off

    poly = T(det["body_polygon"])
    spot_xy = T([[s["cx"], s["cy"]] for s in det["spots"]]) if det["spots"] else np.zeros((0, 2))
    sizes = [math.sqrt(s["w"] * s["h"]) * scale for s in det["spots"]]
    confs = [s["conf"] for s in det["spots"]]
    obstr = [T(o) for o in det["obstruction_polygons"]]
    out = make_det(poly, spot_xy, sizes, confs, obstr,
                   image_size=(det["image_width"] * scale, det["image_height"] * scale),
                   body_conf=det["body_conf"])
    return out


def flat(det, **kw):
    return oc.flat_scalars(oc.features(det, **kw))


# --------------------------------------------------------------------------- #
# 1. PCA frame                                                                 #
# --------------------------------------------------------------------------- #
class TestPCAFrame:

    def test_axes_of_axis_aligned_ellipse(self):
        # Tolerances are 1e-4, not 1e-12: the moments come from a 1024-px
        # raster fill (raster_moments), so the axis direction carries ~1e-5 rad
        # of quantisation.  That is the deliberate price of being immune to
        # self-intersecting YOLO contours.
        f = oc.pca_frame(ellipse_polygon(a=200, b=80))
        assert f is not None
        assert abs(abs(f["e_major"][0]) - 1.0) < 1e-4      # major axis is x
        assert abs(f["e_major"][1]) < 1e-4
        assert f["degenerate_contour"] is False
        assert f["L_major"] == pytest.approx(400.0, rel=2e-3)
        assert f["D_minor"] == pytest.approx(160.0, rel=2e-3)
        # +0.3%: cv2.fillPoly includes the boundary pixels, so a raster area
        # runs ~half a pixel of perimeter high.  The bias is identical for real
        # photos and synthetic renders (same function), so it cancels in the
        # domain-matching comparison this prototype exists for.
        assert f["area"] == pytest.approx(math.pi * 200 * 80, rel=5e-3)
        assert f["area"] > math.pi * 200 * 80
        assert f["aspect"] == pytest.approx(2.5, rel=3e-3)

    def test_origin_is_area_centroid(self):
        f = oc.pca_frame(ellipse_polygon(a=200, b=80, cx=137.0, cy=-42.0))
        assert f["origin"][0] == pytest.approx(137.0, abs=0.5)
        assert f["origin"][1] == pytest.approx(-42.0, abs=0.5)

    def test_frame_is_rotation_equivariant(self):
        """Rotating the polygon rotates the axes by the same angle."""
        poly = ellipse_polygon(a=200, b=80)
        base = oc.pca_frame(poly)
        for theta in (7.0, 33.0, 91.0, 174.0, 268.0):
            f = oc.pca_frame(poly @ rot(theta).T + np.array([500.0, -300.0]))
            assert f["L_major"] == pytest.approx(base["L_major"], rel=1e-3)
            assert f["D_minor"] == pytest.approx(base["D_minor"], rel=1e-3)
            # axes agree up to sign
            cos = abs(float(np.dot(f["e_major"], rot(theta) @ base["e_major"])))
            assert cos == pytest.approx(1.0, abs=1e-4), theta

    def test_degenerate_polygons_return_none(self):
        assert oc.pca_frame([[0, 0], [1, 1]]) is None            # < 3 vertices
        assert oc.pca_frame([[0, 0], [1, 1], [2, 2]]) is None    # zero area
        assert oc.pca_frame([]) is None

    def test_self_intersecting_contour_is_flagged_and_still_usable(self):
        """A bow-tie contour: shoelace area nearly cancels, raster area does not.

        This is the failure mode seen on real catalog image 799, whose
        1744-vertex body contour doubled back on itself and drove the analytic
        density to 23159 (corpus median 104).
        """
        bow = np.array([[0, 0], [400, 0], [0, 200], [400, 200]], dtype=float)
        f = oc.pca_frame(bow)
        assert f is not None
        assert f["degenerate_contour"] is True
        assert f["area_shoelace"] < 0.2 * f["area"]      # shoelace cancels
        assert f["area"] > 1000.0                        # raster does not
        # a plain rectangle must NOT be flagged
        rect = np.array([[0, 0], [400, 0], [400, 200], [0, 200]], dtype=float)
        g = oc.pca_frame(rect)
        assert g["degenerate_contour"] is False
        assert g["area"] == pytest.approx(400 * 200, rel=5e-3)
        assert g["area_shoelace"] == pytest.approx(400 * 200, rel=1e-9)

    def test_raster_moments_match_analytic_on_a_clean_polygon(self):
        poly = ellipse_polygon(a=200, b=80, cx=1000.0, cy=-500.0)
        area, origin, cov = oc.raster_moments(poly)
        assert area == pytest.approx(oc.polygon_area(poly), rel=5e-3)
        assert area > oc.polygon_area(poly)          # boundary-pixel bias
        assert origin[0] == pytest.approx(1000.0, abs=1.0)
        assert origin[1] == pytest.approx(-500.0, abs=1.0)
        # second moments of an ellipse: a^2/4 and b^2/4
        assert cov[0, 0] == pytest.approx(200.0 ** 2 / 4.0, rel=1e-2)
        assert cov[1, 1] == pytest.approx(80.0 ** 2 / 4.0, rel=1e-2)
        assert abs(cov[0, 1]) < 1e-2 * cov[0, 0]

    def test_vertex_density_does_not_move_the_frame(self):
        """Region moments, not vertex moments: resampling must not shift things."""
        sparse = ellipse_polygon(a=200, b=80, n=64)
        # duplicate-densify only the right half -> vertex mean would shift right
        dense = []
        for x, y in sparse:
            dense.append([x, y])
            if x > 0:
                dense.append([x, y])
        a = oc.pca_frame(sparse)
        b = oc.pca_frame(np.array(dense))
        assert b["origin"][0] == pytest.approx(a["origin"][0], abs=1.0)
        assert b["L_major"] == pytest.approx(a["L_major"], rel=1e-2)


# --------------------------------------------------------------------------- #
# 2. features()                                                                #
# --------------------------------------------------------------------------- #
class TestFeatures:

    def build(self):
        """Ellipse a=200,b=80 with a 3x3 lattice of 20 px spots, 25 px apart."""
        poly = ellipse_polygon(a=200.0, b=80.0)
        gx, gy = np.meshgrid([-25.0, 0.0, 25.0], [-25.0, 0.0, 25.0])
        spot_xy = np.column_stack([gx.ravel(), gy.ravel()])
        return make_det(poly, spot_xy, sizes=np.full(9, 20.0),
                        confs=np.full(9, 0.5), image_size=(1000, 800))

    def test_hand_computed_values(self):
        det = self.build()
        f = oc.features(det)
        assert f["ok"] is True
        s = f["scalars"]
        D = s["D_minor"]
        assert D == pytest.approx(160.0, rel=2e-3)
        assert s["L_major"] == pytest.approx(400.0, rel=2e-3)
        assert s["n_spots"] == 9
        # area_norm = pi*a*b / D^2
        assert s["area_norm"] == pytest.approx(math.pi * 200 * 80 / 160.0 ** 2, rel=3e-3)
        assert s["density"] == pytest.approx(9.0 / s["area_norm"], rel=1e-9)
        # every spot is 20x20 -> size = 20 / D
        assert s["size"]["q50"] == pytest.approx(20.0 / D, rel=2e-3)
        # lattice pitch 25 px -> NN distance 25/D for every point
        assert s["nn_median"] == pytest.approx(25.0 / D, rel=2e-3)
        assert s["conf"]["q50"] == pytest.approx(0.5)
        # bbox width 400 px on a 1000 px image
        assert s["bbox_width_frac"] == pytest.approx(0.4, rel=5e-3)
        assert s["body_conf"] == 0.9
        assert s["obstruction_count"] == 0
        assert s["obstruction_area_frac"] == 0.0
        # spots_uv: the lattice spans u in [-25/D, 25/D]
        uv = np.array(f["spots_uv"])
        assert uv.shape == (9, 4)
        assert np.abs(uv[:, 0]).max() == pytest.approx(25.0 / D, rel=2e-3)
        assert np.abs(uv[:, 1]).max() == pytest.approx(25.0 / D, rel=2e-3)

    @pytest.mark.parametrize("theta", [0.0, 17.0, 45.0, 90.0, 123.0, 250.0])
    def test_scalars_invariant_under_rotation_and_translation(self, theta):
        """Every body-frame scalar survives a rigid motion of the whole scene.

        ``bbox_width_frac`` is deliberately excluded: the body bbox is
        axis-aligned in *image* space, so a rotated animal genuinely has a
        different bbox width.  It is an image-frame nuisance descriptor (how
        much of the frame the animal fills), not a shape descriptor, and the
        real-photo pass uses it as exactly that.
        """
        det = self.build()
        base = flat(det)
        moved = transform_det(det, rot(theta), offset=(3000.0, -1200.0))
        got = flat(moved, image_size=(1000, 800))
        for k, v in base.items():
            if k == "bbox_width_frac":
                continue
            if v is None:
                assert got[k] is None, k
            else:
                assert got[k] == pytest.approx(v, rel=2e-2, abs=1e-6), k

    def test_scalars_invariant_under_mirroring(self):
        """A mirror flips handedness, i.e. it is exactly a v -> -v axis flip."""
        det = self.build()
        base = flat(det)
        got = flat(transform_det(det, MIRROR_X), image_size=(1000, 800))
        for k, v in base.items():
            if v is None:
                assert got[k] is None, k
            else:
                assert got[k] == pytest.approx(v, rel=2e-2, abs=1e-6), k

    def test_scalars_invariant_under_explicit_axis_sign_flips(self):
        """Force e_major/e_minor to flip and check every scalar is unchanged.

        This is the literal statement "features must be identical under u -> -u
        and v -> -v": we monkeypatch ``pca_frame`` to negate one or both axes
        after it has built the frame, and require bit-comparable scalars.
        """
        det = self.build()
        base = flat(det)
        real_pca = oc.pca_frame
        for su, sv in ((1, -1), (-1, 1), (-1, -1)):
            def flipped(poly, spots_uv=None, _su=su, _sv=sv):
                fr = real_pca(poly, spots_uv)
                if fr is None:
                    return None
                fr = dict(fr)
                fr["e_major"] = _su * fr["e_major"]
                fr["e_minor"] = _sv * fr["e_minor"]
                return fr
            oc.pca_frame = flipped
            try:
                got = flat(det)
            finally:
                oc.pca_frame = real_pca
            for k, v in base.items():
                assert got[k] == v or (v is None and got[k] is None), (su, sv, k)

    def test_uv_flips_sign_but_geometry_is_preserved(self):
        """The exported u/v are NOT sign-canonical; the constellation is."""
        det = self.build()
        f0 = np.array(oc.features(det)["spots_uv"])
        real_pca = oc.pca_frame

        def flipped(poly, spots_uv=None):
            fr = real_pca(poly, spots_uv)
            fr = dict(fr)
            fr["e_major"] = -fr["e_major"]
            return fr
        oc.pca_frame = flipped
        try:
            f1 = np.array(oc.features(det)["spots_uv"])
        finally:
            oc.pca_frame = real_pca
        assert np.allclose(f1[:, 0], -f0[:, 0])
        assert np.allclose(f1[:, 1], f0[:, 1])
        assert np.allclose(f1[:, 2:], f0[:, 2:])

    def test_scale_invariance_of_normalised_scalars(self):
        det = self.build()
        base = flat(det)
        got = flat(transform_det(det, np.eye(2), scale=3.0))
        for k in ("n_spots", "density", "area_norm", "aspect", "nn_median",
                  "size_q50", "conf_q50", "bbox_width_frac"):
            assert got[k] == pytest.approx(base[k], rel=2e-2), k
        # px-valued scalars scale as expected
        assert got["D_minor"] == pytest.approx(3.0 * base["D_minor"], rel=2e-2)
        assert got["area_px2"] == pytest.approx(9.0 * base["area_px2"], rel=2e-2)

    def test_no_body_polygon(self):
        det = self.build()
        det["body_polygon"] = None
        det["body_bbox"] = None
        f = oc.features(det)
        assert f["ok"] is False
        assert f["frame"] is None
        assert f["spots_uv"] == []
        assert f["scalars"]["n_spots"] == 9        # raw spots still reported
        assert f["scalars"]["density"] is None
        assert f["scalars"]["bbox_width_frac"] is None
        assert len(f["spots_raw"]) == 9

    def test_zero_spots(self):
        det = make_det(ellipse_polygon(), np.zeros((0, 2)))
        f = oc.features(det)
        assert f["ok"] is True
        assert f["scalars"]["n_spots"] == 0
        assert f["scalars"]["density"] == 0.0
        assert f["scalars"]["nn_median"] is None
        assert f["scalars"]["size"]["q50"] is None
        assert f["spots_uv"] == []

    def test_obstruction_area_fraction(self):
        """A half-plane obstruction covering the left half of the body."""
        poly = ellipse_polygon(a=200.0, b=80.0)
        # rectangle covering x in [-400, 0] -- extends well past the animal,
        # so the raw area ratio must exceed the clipped covered fraction.
        ob = [[-400, -300], [0, -300], [0, 300], [-400, 300]]
        det = make_det(poly, np.zeros((0, 2)), obstr=[ob])
        s = oc.features(det)["scalars"]
        assert s["obstruction_count"] == 1
        assert s["obstruction_area_frac"] == pytest.approx(0.5, abs=0.02)
        assert s["obstruction_area_ratio"] > 3.0

    def test_obstruction_ratio_uses_the_raster_area_not_the_shoelace_area(self):
        """The ratio's denominator must survive a self-intersecting contour.

        A bow tie traced as one contour has a shoelace area that cancels to
        almost nothing while its filled area is large -- exactly the failure
        ``raster_moments`` exists for (catalog image 799: shoelace 52 232 px2,
        raster 3 635 510 px2, a factor of 69.6).  Dividing the obstruction area
        by the shoelace area inflated ``obstruction_area_ratio`` by that same
        factor; it reported 74.9 for an obstruction covering 26% of the body.
        """
        # two mirrored lobes traced in one stroke -> the signed areas cancel
        body = [[0, 0], [400, 0], [0, 200], [400, 200]]
        shoelace = oc.polygon_area(body)
        raster = oc.raster_moments(body)[0]
        assert raster > 20.0 * shoelace, (shoelace, raster)

        ob = [[[100, 40], [300, 40], [300, 160], [100, 160]]]
        det = make_det(body, np.zeros((0, 2)), obstr=ob, image_size=(500, 300))
        s = oc.features(det)["scalars"]
        ob_area = oc.raster_moments(ob[0])[0]
        assert s["obstruction_area_ratio"] == pytest.approx(ob_area / raster, rel=0.02)
        # and the fix has to actually bite on this shape
        assert s["obstruction_area_ratio"] < 0.05 * (ob_area / shoelace)

    def test_flat_scalars_keys_are_stable(self):
        keys = set(flat(self.build()).keys())
        for k in ("n_spots", "density", "nn_median", "size_q50", "conf_q95",
                  "bbox_width_frac", "body_conf", "obstruction_area_frac"):
            assert k in keys

    def test_to_db_row_matches_column_set(self):
        row = oc.to_db_row(self.build())
        assert tuple(row.keys()) == oc.DB_COLUMNS
        assert row["head_polygon_json"] is None
        assert row["obstruction_polygon_json"] is None      # empty list -> NULL
        assert row["spot_count"] == 9


# --------------------------------------------------------------------------- #
# 2b. the silent max_det cap                                                   #
# --------------------------------------------------------------------------- #
class _FakeBoxes(object):
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n


class _FakeResult(object):
    def __init__(self, n):
        self.boxes = _FakeBoxes(n)


class _FakeSpotModel(object):
    """Just enough of an ultralytics model for the counting wrapper."""

    def __init__(self, n_boxes, max_det=None):
        self.n_boxes = n_boxes
        self.overrides = {} if max_det is None else {"max_det": max_det}
        self.calls = 0

    def predict(self, *a, **kw):
        self.calls += 1
        return [_FakeResult(self.n_boxes)]


class TestMaxDetReporting:
    """The spot detector is capped by ultralytics' max_det and says nothing.

    ``run_image`` filters spots (centre inside the body, centre not inside an
    obstruction) AFTER NMS has truncated them, so a capped image can report any
    ``spot_count`` at or below the cap -- on the real catalog 23 images are
    truncated and only 6 of them show ``spot_count == 300``.
    """

    def test_spot_max_det_is_the_ultralytics_default(self):
        assert oc.spot_max_det(None) == 300

    def test_spot_max_det_honours_a_model_override(self):
        assert oc.spot_max_det(_FakeSpotModel(0, max_det=77)) == 77

    def test_counting_predict_records_and_restores(self):
        model = _FakeSpotModel(300)
        before = model.predict
        with oc._CountingPredict(model) as counter:
            assert counter.count is None
            model.predict("crop")
            assert counter.count == 300
        assert model.calls == 1
        assert "predict" not in vars(model), "wrapper left behind on the model"
        assert model.predict.__func__ is before.__func__

    def test_counting_predict_is_a_no_op_without_a_spot_model(self):
        with oc._CountingPredict(None) as counter:
            assert counter.count is None

    def _detect_with_fake_pipeline(self, monkeypatch, n_raw, n_kept):
        """detect() over a stub run_image that truncates then filters."""
        class _Stub(object):
            @staticmethod
            def run_image(img, body_obstr, body_only, head, spot_model,
                          bc, hc, sc):
                res = spot_model.predict(img)                 # what NMS returned
                n = len(res[0].boxes)
                assert n == n_raw
                spots = [{"x": 0.0, "y": 0.0, "w": 4.0, "h": 4.0,
                          "cx": float(i), "cy": 1.0, "conf": 0.5}
                         for i in range(n_kept)]
                return {"body": [[0, 0], [10, 0], [10, 10]], "body_bbox": None,
                        "body_conf": 0.9, "obstructions": [], "head": None,
                        "head_bbox": None, "head_conf": None,
                        "spots": spots, "spot_count": len(spots)}

        monkeypatch.setattr(oc, "_infer_pipeline", lambda: _Stub)
        spot = _FakeSpotModel(n_raw)
        img = np.zeros((32, 32, 3), np.uint8)
        return oc.detect(img, models=(object(), spot))

    def test_truncation_is_flagged_even_when_spot_count_is_far_below_the_cap(
            self, monkeypatch):
        # the real shape of the bug: catalog image 619 stored n_spots=239 with
        # 300 raw boxes; the true kept count at max_det=5000 is 300.
        det = self._detect_with_fake_pipeline(monkeypatch, n_raw=300, n_kept=239)
        assert det["spot_count"] == 239
        assert det["spots_raw_count"] == 300
        assert det["spots_max_det"] == 300
        assert det["spots_truncated"] is True

    def test_an_uncapped_image_is_not_flagged(self, monkeypatch):
        det = self._detect_with_fake_pipeline(monkeypatch, n_raw=196, n_kept=189)
        assert det["spots_raw_count"] == 196
        assert det["spots_truncated"] is False

    def test_no_spot_model_reports_an_unknown_raw_count(self, monkeypatch):
        class _Stub(object):
            @staticmethod
            def run_image(img, body_obstr, body_only, head, spot_model,
                          bc, hc, sc):
                return {"body": None, "body_bbox": None, "body_conf": None,
                        "obstructions": [], "head": None, "head_bbox": None,
                        "head_conf": None, "spots": [], "spot_count": 0}

        monkeypatch.setattr(oc, "_infer_pipeline", lambda: _Stub)
        det = oc.detect(np.zeros((32, 32, 3), np.uint8), models=(object(), None))
        assert det["spots_raw_count"] is None
        assert det["spots_truncated"] is False

    def test_truncation_keys_are_not_db_columns(self):
        assert "spots_raw_count" not in oc.DB_COLUMNS
        assert "spots_truncated" not in oc.DB_COLUMNS


# --------------------------------------------------------------------------- #
# 3. detect() against the real weights                                         #
# --------------------------------------------------------------------------- #
def _weights_present():
    try:
        p = oc.weight_paths()
    except FileNotFoundError:
        return False, "OSEA main checkout not found (set SEVENGILL_MAIN_ROOT)"
    if p["body_obstr"] is None and p["body_only"] is None:
        return False, "no body weights at spot_detector/runs/body_obstr/v1 or runs/body/v2"
    if p["spots"] is None:
        return False, "no spot weights at spot_detector/runs/spots/{v2,v1}"
    return True, ""


_HAVE, _WHY = _weights_present()


@pytest.mark.skipif(not _HAVE, reason="OSEA weights unavailable: " + _WHY)
class TestDetect:

    def test_blank_image_has_no_body_and_no_spots(self):
        models = oc.load_models("cpu")
        det = oc.detect(np.zeros((640, 640, 3), np.uint8), models)
        assert det["body_polygon"] is None
        assert det["body_bbox"] is None
        assert det["body_conf"] is None
        assert det["spot_count"] == 0
        assert det["spots"] == []
        assert det["obstruction_count"] == 0
        assert det["head_polygon"] is None and det["head_conf"] is None
        assert det["image_width"] == 640 and det["image_height"] == 640
        # no body -> run_image never calls the spot model at all
        assert det["spots_raw_count"] is None
        assert det["spots_truncated"] is False
        assert det["spots_max_det"] == 300
        f = oc.features(det)
        assert f["ok"] is False and f["scalars"]["n_spots"] == 0

    def test_uniform_grey_image_also_empty(self):
        models = oc.load_models("cpu")
        det = oc.detect(np.full((480, 640, 3), 128, np.uint8), models)
        assert det["body_polygon"] is None
        assert det["spot_count"] == 0

    def test_detect_rejects_bad_input(self):
        models = oc.load_models("cpu")
        with pytest.raises(ValueError):
            oc.detect(np.zeros((64, 64), np.uint8), models)
        with pytest.raises(ValueError):
            oc.detect(np.zeros((64, 64, 3), np.float32), models)

    def test_spot_weights_fall_back_to_v1(self):
        p = oc.weight_paths()
        if p["spots"] is not None:
            assert p["spots"].is_file()
        assert p["head"] is None or p["head"].is_file()


def test_main_root_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("SEVENGILL_MAIN_ROOT", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        oc.main_root()
    (tmp_path / "spot_detector" / "scripts").mkdir(parents=True)
    (tmp_path / "spot_detector" / "scripts" / "infer_pipeline.py").write_text("")
    assert oc.main_root() == tmp_path.resolve()


def test_main_root_walk_up_finds_checkout(monkeypatch):
    monkeypatch.delenv("SEVENGILL_MAIN_ROOT", raising=False)
    root = oc.main_root()
    assert (root / "spot_detector" / "scripts" / "infer_pipeline.py").is_file()
    assert (root / "tagger").is_dir()
