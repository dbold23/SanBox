"""Behavioural tests for the rig: placement, weights, FK/LBS.

Ground truth comes from ``fixtures_rig.straight_capsule`` -- a straight elliptical
tube along X with eight box fins -- so every expected value is analytic.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fixtures_rig as fx  # noqa: E402
import rig  # noqa: E402

SCHEMA = rig.SCHEMA


@pytest.fixture(scope="module")
def capsule():
    return fx.straight_capsule()


@pytest.fixture(scope="module")
def skeleton(capsule):
    return rig.build_skeleton(capsule["centerline"], capsule["fin_info"])


@pytest.fixture(scope="module")
def weights(capsule, skeleton):
    return rig.compute_weights(capsule["vertices"], capsule["labels"], skeleton)


def identity_rotations(skeleton):
    return np.tile(np.eye(3), (skeleton.num_joints, 1, 1))


# ---------------------------------------------------------------------------
# skeleton placement
# ---------------------------------------------------------------------------
def test_spine_is_the_schema_skeleton(skeleton):
    """The rig's spine IS skeleton_sevengill's serial chain: names, order, parents."""
    assert skeleton.names[: rig.NUM_SPINE_JOINTS] == SCHEMA.SPINE_JOINTS
    assert rig.NUM_SPINE_JOINTS == 13
    expected_parents = [-1] + list(range(12))
    assert list(skeleton.parents[:13]) == expected_parents
    for name in SCHEMA.SPINE_JOINTS:
        parent = SCHEMA.KINEMATIC_TREE[name]
        j = skeleton.index(name)
        if parent is None:
            assert skeleton.parents[j] == -1
        else:
            assert skeleton.names[skeleton.parents[j]] == parent


def test_spine_joints_sit_at_the_declared_arclength_fractions(capsule, skeleton):
    """Each spine joint lands at its arc-length fraction of the straight centerline."""
    frac = rig.spine_arclength_fractions()
    half = fx.BODY_LENGTH / 2.0
    expected_x = half - frac * fx.BODY_LENGTH
    got = skeleton.joints[skeleton.spine_indices]
    np.testing.assert_allclose(got[:, 0], expected_x, atol=1e-9)
    np.testing.assert_allclose(got[:, 1:], 0.0, atol=1e-12)


def test_midline_fractions_come_from_the_schema():
    """The seven trunk stations use the schema's own MIDLINE_AXIS_FRACTIONS, scaled."""
    pf = 0.8
    frac = rig.spine_arclength_fractions(pf)
    trunk = frac[3:10] / pf
    np.testing.assert_allclose(trunk, SCHEMA.MIDLINE_AXIS_FRACTIONS, atol=1e-12)
    assert frac[SCHEMA.SPINE_JOINTS.index("spine_10_precaudal")] == pytest.approx(pf)


def test_spine_fractions_are_strictly_increasing_across_the_bracket():
    for pf in (0.70, 0.78, 0.85, 0.95):
        frac = rig.spine_arclength_fractions(pf)
        assert np.all(np.diff(frac) > 0)
        assert 0.0 < frac[0] and frac[-1] < 1.0
    with pytest.raises(ValueError):
        rig.spine_arclength_fractions(1.5)


def test_parents_precede_children(skeleton):
    """Guaranteed ordering, so FK can walk range(J) (cf. shark-pose-3d topo sort)."""
    for j, p in enumerate(skeleton.parents):
        assert p < j
    assert rig.topological_order(skeleton.parents) == list(range(skeleton.num_joints))


def test_topological_order_rejects_a_cycle():
    with pytest.raises(ValueError):
        rig.topological_order([1, 0])


def test_every_fin_gets_a_root_and_a_tip(capsule, skeleton):
    for name in capsule["fin_info"]:
        root, tip = skeleton.fins[name]
        assert skeleton.names[root] == "%s_fin_root" % name
        assert skeleton.names[tip] == "%s_fin_tip" % name
        assert skeleton.parents[tip] == root
        assert skeleton.kinds[root] == "fin_root"
        np.testing.assert_allclose(skeleton.joints[root], capsule["fin_info"][name]["insertion"])
        np.testing.assert_allclose(skeleton.joints[tip], capsule["fin_info"][name]["tip"])
    # the heterocercal caudal has its OWN upper-lobe root+tip, separate from the lower
    assert "caudal_upper" in skeleton.fins and "caudal_lower" in skeleton.fins
    assert skeleton.fins["caudal_upper"] != skeleton.fins["caudal_lower"]


def test_fin_roots_parent_to_the_nearest_spine_joint(capsule, skeleton):
    spine = skeleton.spine_indices
    spine_pos = skeleton.joints[spine]
    for name, (root, _tip) in skeleton.fins.items():
        insertion = np.asarray(capsule["fin_info"][name]["insertion"])
        nearest = spine[int(np.argmin(np.linalg.norm(spine_pos - insertion, axis=1)))]
        assert skeleton.parents[root] == nearest
        assert skeleton.kinds[skeleton.parents[root]] == "spine"


def test_fin_parent_can_be_pinned_explicitly(capsule):
    fin_info = {k: dict(v) for k, v in capsule["fin_info"].items()}
    fin_info["dorsal"]["parent"] = "spine_07_trunk_05"
    sk = rig.build_skeleton(capsule["centerline"], fin_info)
    root, _ = sk.fins["dorsal"]
    assert sk.names[sk.parents[root]] == "spine_07_trunk_05"
    with pytest.raises(ValueError):
        bad = {k: dict(v) for k, v in fin_info.items()}
        bad["dorsal"]["parent"] = "not_a_spine_joint"
        rig.build_skeleton(capsule["centerline"], bad)


def test_module_a_fin_detection_dict_is_accepted(capsule):
    """Module A reports insertion_centroid + vertex_indices; the adapter finds the tip."""
    labels = capsule["labels"]
    detection = {}
    for name in capsule["fin_info"]:
        idx = np.nonzero(np.asarray([str(x) == name for x in labels]))[0]
        detection[name] = {
            "insertion_centroid": capsule["fin_info"][name]["insertion"],
            "vertex_indices": idx,
        }
    fin_info = rig.fin_info_from_detection(detection, capsule["vertices"])
    sk = rig.build_skeleton(capsule["centerline"], fin_info)
    assert set(sk.fins) == set(capsule["fin_info"])
    for name in fin_info:
        root, tip = sk.fins[name]
        np.testing.assert_allclose(
            sk.joints[root], capsule["fin_info"][name]["insertion"], atol=1e-12
        )
        # the derived tip is the island's apex (farthest radial protrusion on the
        # insertion's side; caudal: farthest from the insertion), so it sits at
        # least as far out as the analytic tip's distance minus the box half-chord
        reach = np.linalg.norm(sk.joints[tip] - sk.joints[root])
        analytic = np.linalg.norm(
            capsule["fin_info"][name]["tip"] - capsule["fin_info"][name]["insertion"]
        )
        assert reach >= analytic * 0.9
    w = rig.compute_weights(capsule["vertices"], labels, sk)
    np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-12)


def test_insertion_centroid_is_accepted_as_an_alias(capsule):
    fin_info = {
        "dorsal": {
            "insertion_centroid": capsule["fin_info"]["dorsal"]["insertion"],
            "tip": capsule["fin_info"]["dorsal"]["tip"],
        }
    }
    sk = rig.build_skeleton(capsule["centerline"], fin_info)
    root, _ = sk.fins["dorsal"]
    np.testing.assert_allclose(
        sk.joints[root], capsule["fin_info"]["dorsal"]["insertion"], atol=1e-12
    )


def test_build_skeleton_rejects_a_degenerate_centerline(capsule):
    with pytest.raises(ValueError):
        rig.build_skeleton(np.zeros((1, 3)), capsule["fin_info"])


# ---------------------------------------------------------------------------
# weights
# ---------------------------------------------------------------------------
def test_weights_are_a_partition_of_unity_with_at_most_four_influences(weights):
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-12)
    assert weights.min() >= 0.0
    assert (weights > 0).sum(axis=1).max() <= 4


def test_bracketing_binding_uses_exactly_two_spine_joints(capsule, skeleton, weights):
    body = np.asarray([str(x) == rig.BODY_LABEL for x in capsule["labels"]])
    counts = (weights[body] > 0).sum(axis=1)
    assert counts.max() <= 2
    spine = set(skeleton.spine_indices.tolist())
    used = set(np.nonzero(weights[body].sum(axis=0))[0].tolist())
    assert used <= spine


def test_body_vertex_at_a_spine_joint_binds_to_that_joint(skeleton):
    j = skeleton.index("spine_06_trunk_04")
    v = skeleton.joints[j][None, :] + np.array([[0.0, 0.05, 0.0]])
    w = rig.compute_weights(v, np.array(["body"], dtype=object), skeleton)
    assert w[0, j] == pytest.approx(1.0, abs=1e-9)


def test_fin_vertices_bind_only_to_their_own_fin(capsule, skeleton, weights):
    for name, (root, tip) in skeleton.fins.items():
        mask = np.asarray([str(x) == name for x in capsule["labels"]])
        assert mask.sum() == 8
        rows = weights[mask]
        assert np.all(rows[:, [root, tip]].sum(axis=1) > 1.0 - 1e-12)
        other = np.ones(weights.shape[1], dtype=bool)
        other[[root, tip]] = False
        assert rows[:, other].max() == 0.0
    # base of a fin leans on the root, tip of a fin on the tip joint
    name = "dorsal"
    root, tip = skeleton.fins[name]
    mask = np.asarray([str(x) == name for x in capsule["labels"]])
    verts = capsule["vertices"][mask]
    order = np.argsort(verts[:, 2])
    assert weights[mask][order[0], root] > weights[mask][order[-1], root]


# ---------------------------------------------------------------------------
# the fin-base seam (fix M2)
# ---------------------------------------------------------------------------
def _unique_edges(faces):
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=0)
    return np.unique(np.sort(e, axis=1), axis=0)


def _seam_edges(capsule):
    """Mesh edges with exactly one endpoint inside a fin island."""
    e = _unique_edges(capsule["faces"])
    is_fin = np.asarray([str(x) != rig.BODY_LABEL for x in capsule["labels"]])
    return e[is_fin[e[:, 0]] != is_fin[e[:, 1]]]


# ``fixtures_rig``'s box fins are DISJOINT components floating at the body wall, which
# is enough for every weight test that came before but has no mesh edge crossing a fin
# base -- and the fin base is precisely what fix M2 is about.  A real detected fin is
# a welded patch of the same surface (``mesh3d.detect_fins`` labels the protruding
# blade and leaves its root ring labelled ``body``), so this builds that: a tube with
# two fins EXTRUDED from rows of its own vertices, sharing the base ring with the body.
_WELD_STATIONS, _WELD_AROUND, _WELD_LAYERS = 28, 16, 3
_WELD_FINS = {
    # name -> (phi index into the ring, first station, last station, outward step)
    "dorsal": (_WELD_AROUND // 4, 8, 13, (0.0, 0.0, 0.030)),
    "anal": (3 * _WELD_AROUND // 4, 17, 21, (0.0, 0.0, -0.026)),
}


def welded_fin_capsule():
    """Straight tube along X with two fins welded into its own surface.

    Same keys as ``fixtures_rig.straight_capsule`` (minus UVs), so it drops into
    ``build_skeleton`` / ``compute_weights`` unchanged.
    """
    verts, _uv, faces = fx._tube(_WELD_STATIONS, _WELD_AROUND)
    verts = list(verts)
    faces = [list(f) for f in np.asarray(faces)]
    labels = [rig.BODY_LABEL] * len(verts)
    fin_info = {}
    for name, (k, i0, i1, step) in _WELD_FINS.items():
        step = np.asarray(step, dtype=float)
        base = [i * _WELD_AROUND + k for i in range(i0, i1 + 1)]
        rows = [base]
        for layer in range(1, _WELD_LAYERS + 1):
            row = []
            for b in base:
                row.append(len(verts))
                verts.append(np.asarray(verts[b], dtype=float) + layer * step)
                labels.append(name)
            rows.append(row)
        for lo, hi in zip(rows[:-1], rows[1:]):
            for a, b, c, d in zip(lo[:-1], lo[1:], hi[1:], hi[:-1]):
                faces.append([a, b, c])
                faces.append([a, c, d])
        fin_info[name] = {
            "insertion": np.mean(np.asarray([verts[b] for b in base]), axis=0),
            "tip": np.mean(np.asarray([verts[t] for t in rows[-1]]), axis=0),
        }
    half = fx.BODY_LENGTH / 2.0
    return {
        "vertices": np.asarray(verts, dtype=float),
        "faces": np.asarray(faces, dtype=np.int64),
        "labels": np.asarray(labels, dtype=object),
        "centerline": np.column_stack(
            [np.linspace(half, -half, 64), np.zeros(64), np.zeros(64)]
        ),
        "fin_info": fin_info,
    }


@pytest.fixture(scope="module")
def welded():
    return welded_fin_capsule()


@pytest.fixture(scope="module")
def welded_skeleton(welded):
    return rig.build_skeleton(welded["centerline"], welded["fin_info"])


def _bend_rotations(skeleton, deg=25.0):
    """A hard, uniform yaw on every spine joint -- the pose that opens the seam."""
    rots = identity_rotations(skeleton)
    for j in skeleton.spine_indices:
        rots[int(j)] = rig.axis_angle_rotmat((0.0, 0.0, 1.0), np.deg2rad(deg))
    return rots


def _max_seam_edge_change(capsule, skeleton, w, rots):
    v = np.asarray(capsule["vertices"], dtype=float)
    seam = _seam_edges(capsule)
    rest = np.linalg.norm(v[seam[:, 0]] - v[seam[:, 1]], axis=1)
    posed = rig.lbs(v, w, skeleton, rots)
    now = np.linalg.norm(posed[seam[:, 0]] - posed[seam[:, 1]], axis=1)
    return float(np.abs(now - rest).max())


def test_fin_base_blend_ramps_the_fin_root_weight_out_into_the_body(welded, welded_skeleton):
    """REGRESSION (fix M2): the seam was a step, it is now a linear ramp.

    Ring 0 (the island itself) is 100% fin, ring R is 100% spine, and rings in
    between carry exactly ``(R - d) / R`` of fin-root weight.
    """
    rings = 3
    w = rig.compute_weights(
        welded["vertices"], welded["labels"], welded_skeleton,
        faces=welded["faces"], fin_blend_rings=rings,
    )
    ring, owner = rig.fin_seam_rings(
        welded["faces"], np.asarray(welded["labels"]).astype(str),
        len(welded["vertices"]), rings,
    )
    assert (ring == 1).sum() > 0, "the fixture must have a ring of body vertices at each fin base"
    for d in range(1, rings):
        rows = np.nonzero(ring == d)[0]
        assert len(rows), "ring %d is empty" % d
        for v in rows:
            root = int(welded_skeleton.fins[str(owner[v])][0])
            assert w[v, root] == pytest.approx((rings - d) / float(rings), abs=1e-12)
    # ring R itself is the zero end of the ramp: still pure spine
    spine = np.asarray(welded_skeleton.spine_indices, dtype=int)
    outer = np.nonzero(ring == rings)[0]
    if len(outer):
        assert w[outer][:, spine].sum(axis=1) == pytest.approx(1.0, abs=1e-12)


def test_fin_base_blend_keeps_a_partition_of_unity_and_four_influences(welded, welded_skeleton):
    for rings in (1, 2, 3, 5):
        w = rig.compute_weights(
            welded["vertices"], welded["labels"], welded_skeleton,
            faces=welded["faces"], fin_blend_rings=rings,
        )
        np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-12)
        assert w.min() >= 0.0
        assert (w > 0).sum(axis=1).max() <= 4
    w = rig.compute_weights(
        welded["vertices"], welded["labels"], welded_skeleton, sigma=0.25,
        faces=welded["faces"], fin_blend_rings=3,
    )
    np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-12)
    assert (w > 0).sum(axis=1).max() <= 4


def test_fin_base_blend_closes_the_seam_discontinuity_under_a_bend(welded, welded_skeleton):
    """The measurement the fix exists for: the hard seam tears, the ramp does not."""
    rots = _bend_rotations(welded_skeleton)
    hard = _max_seam_edge_change(
        welded, welded_skeleton,
        rig.compute_weights(welded["vertices"], welded["labels"], welded_skeleton),
        rots,
    )
    blended = _max_seam_edge_change(
        welded, welded_skeleton,
        rig.compute_weights(
            welded["vertices"], welded["labels"], welded_skeleton,
            faces=welded["faces"], fin_blend_rings=3,
        ),
        rots,
    )
    assert blended < 0.5 * hard, "hard %.5f, blended %.5f" % (hard, blended)
    assert blended / fx.BODY_LENGTH < 0.01, "%.4f %%BL" % (100.0 * blended / fx.BODY_LENGTH)


def test_fin_blend_rings_one_reproduces_the_hard_seam(welded, welded_skeleton):
    """R = 1 puts the ramp's zero at ring 1, i.e. changes nothing -- the escape hatch."""
    hard = rig.compute_weights(welded["vertices"], welded["labels"], welded_skeleton)
    same = rig.compute_weights(
        welded["vertices"], welded["labels"], welded_skeleton,
        faces=welded["faces"], fin_blend_rings=1,
    )
    np.testing.assert_allclose(hard, same, atol=1e-15)
    with pytest.raises(ValueError):
        rig.compute_weights(
            welded["vertices"], welded["labels"], welded_skeleton,
            faces=welded["faces"], fin_blend_rings=0,
        )


def test_identity_pose_is_untouched_by_the_fin_base_blend(welded, welded_skeleton):
    """Any partition of unity reproduces the rest mesh at identity; assert it anyway,
    because a blend that leaked weight would show up here first."""
    w = rig.compute_weights(
        welded["vertices"], welded["labels"], welded_skeleton, faces=welded["faces"]
    )
    posed = rig.lbs(welded["vertices"], w, welded_skeleton, identity_rotations(welded_skeleton))
    np.testing.assert_allclose(posed, welded["vertices"], atol=1e-12)


def test_fin_seam_rings_walks_only_through_body_vertices(welded):
    labels = np.asarray(welded["labels"]).astype(str)
    ring, owner = rig.fin_seam_rings(
        welded["faces"], labels, len(welded["vertices"]), 3
    )
    assert np.all(ring[labels != rig.BODY_LABEL] == -1)
    reached = ring > 0
    assert np.all(labels[reached] == rig.BODY_LABEL)
    assert np.all(np.asarray([str(o) for o in owner[reached]]) != "")
    assert np.all(ring <= 3)


def test_vertex_adjacency_is_symmetric_and_deduplicated(capsule):
    n = len(capsule["vertices"])
    neigh, starts = rig.vertex_adjacency(capsule["faces"], n)
    assert starts.shape == (n + 1,)
    assert int(starts[-1]) == len(neigh)
    pairs = set()
    for v in range(n):
        for u in neigh[starts[v]:starts[v + 1]]:
            assert u != v, "no self loops"
            pairs.add((v, int(u)))
    assert all((b, a) in pairs for a, b in pairs)
    assert len(pairs) == 2 * len(_unique_edges(capsule["faces"]))


def test_gaussian_widening_spreads_influence_and_still_partitions_unity(capsule, skeleton):
    w = rig.compute_weights(
        capsule["vertices"], capsule["labels"], skeleton, sigma=0.25
    )
    np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-12)
    assert (w > 0).sum(axis=1).max() <= 4
    body = np.asarray([str(x) == rig.BODY_LABEL for x in capsule["labels"]])
    assert (w[body] > 0).sum(axis=1).mean() > 2.0


def test_unknown_label_is_an_error(capsule, skeleton):
    labels = np.array(capsule["labels"], dtype=object)
    labels[0] = "second_dorsal"  # a sevengill has no second dorsal
    with pytest.raises(ValueError):
        rig.compute_weights(capsule["vertices"], labels, skeleton)


def test_integer_labels_need_names(capsule, skeleton):
    codes = np.zeros(len(capsule["vertices"]), dtype=int)
    with pytest.raises(ValueError):
        rig.compute_weights(capsule["vertices"], codes, skeleton)
    w = rig.compute_weights(
        capsule["vertices"], codes, skeleton, label_names=[rig.BODY_LABEL]
    )
    np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-12)


def test_weights_to_indexed_is_gltf_shaped(weights):
    idx, w32 = rig.weights_to_indexed(weights)
    assert idx.dtype == np.uint16 and idx.shape == (len(weights), 4)
    assert w32.dtype == np.float32 and w32.shape == (len(weights), 4)
    # exact float32 partition of unity: the glTF validator does not round
    assert np.all(w32.sum(axis=1) == np.float32(1.0))
    # unused slots carry joint 0 with weight 0
    assert np.all(idx[w32 == 0] == 0)


def test_prune_keeps_the_largest_influences():
    w = np.array([[0.4, 0.3, 0.2, 0.05, 0.05]])
    pruned = rig.prune_weights(w, 2)
    assert pruned[0, 0] == pytest.approx(0.4 / 0.7)
    assert pruned[0, 1] == pytest.approx(0.3 / 0.7)
    assert pruned[0, 2:].max() == 0.0


# ---------------------------------------------------------------------------
# forward kinematics / LBS
# ---------------------------------------------------------------------------
def test_identity_pose_is_the_rest_mesh(capsule, skeleton, weights):
    world = rig.forward_kinematics(skeleton, identity_rotations(skeleton))
    np.testing.assert_allclose(world, np.tile(np.eye(4), (skeleton.num_joints, 1, 1)), atol=1e-12)
    posed = rig.lbs(capsule["vertices"], weights, skeleton, identity_rotations(skeleton))
    np.testing.assert_array_equal(posed, capsule["vertices"])
    np.testing.assert_allclose(
        rig.posed_joints(skeleton, identity_rotations(skeleton)), skeleton.joints, atol=1e-12
    )


def test_thirty_degree_bend_moves_downstream_vertices_and_not_upstream_ones(
    capsule, skeleton, weights
):
    j = skeleton.index("spine_06_trunk_04")
    rot = identity_rotations(skeleton)
    rot[j] = rig.axis_angle_rotmat([0.0, 0.0, 1.0], np.deg2rad(30.0))
    posed = rig.lbs(capsule["vertices"], weights, skeleton, rot)
    moved = np.linalg.norm(posed - capsule["vertices"], axis=1)

    subtree = np.zeros(skeleton.num_joints, dtype=bool)
    subtree[skeleton.descendants(j)] = True
    downstream = weights[:, subtree].sum(axis=1)

    # anything with no weight on the rotated joint's subtree is untouched, exactly
    assert moved[downstream <= 0].max() == 0.0
    # everything fully inside the subtree moves, and further out means further moved
    fully = downstream >= 1.0 - 1e-12
    assert fully.sum() > 100
    assert moved[fully].min() > 1e-6
    lever = np.linalg.norm(capsule["vertices"][fully] - skeleton.joints[j], axis=1)
    assert np.corrcoef(lever, moved[fully])[0, 1] > 0.99

    # the pectoral fins are upstream of this joint; the caudal lobes are downstream
    for upstream_fin in ("pectoral_left", "pectoral_right"):
        mask = np.asarray([str(x) == upstream_fin for x in capsule["labels"]])
        assert moved[mask].max() == 0.0
    for downstream_fin in ("dorsal", "anal", "caudal_upper", "caudal_lower"):
        mask = np.asarray([str(x) == downstream_fin for x in capsule["labels"]])
        assert moved[mask].min() > 1e-3


def test_bend_angle_is_exactly_the_requested_rotation(skeleton):
    j = skeleton.index("spine_06_trunk_04")
    angle = np.deg2rad(30.0)
    rot = identity_rotations(skeleton)
    rot[j] = rig.axis_angle_rotmat([0.0, 0.0, 1.0], angle)
    joints = rig.posed_joints(skeleton, rot)
    pivot = skeleton.joints[j]
    tail = skeleton.index("spine_12_caudal_axis_2")
    before = skeleton.joints[tail] - pivot
    after = joints[tail] - pivot
    np.testing.assert_allclose(np.linalg.norm(after), np.linalg.norm(before), atol=1e-12)
    cos = np.dot(before, after) / (np.linalg.norm(before) * np.linalg.norm(after))
    assert np.arccos(np.clip(cos, -1, 1)) == pytest.approx(angle, abs=1e-9)
    # joints upstream of the pivot do not move at all
    np.testing.assert_array_equal(joints[:j], skeleton.joints[:j])


def test_fin_tip_rotation_flexes_only_the_distal_fin(capsule, skeleton, weights):
    root, tip = skeleton.fins["pectoral_left"]
    rot = identity_rotations(skeleton)
    rot[tip] = rig.axis_angle_rotmat([1.0, 0.0, 0.0], np.deg2rad(25.0))
    posed = rig.lbs(capsule["vertices"], weights, skeleton, rot)
    moved = np.linalg.norm(posed - capsule["vertices"], axis=1)
    mask = np.asarray([str(x) == "pectoral_left" for x in capsule["labels"]])
    assert moved[~mask].max() == 0.0
    assert moved[mask].max() > 1e-3


def test_rotations_compose_along_the_parent_chain(skeleton):
    a, b = skeleton.index("spine_04_trunk_02"), skeleton.index("spine_08_trunk_06")
    rot = identity_rotations(skeleton)
    rot[a] = rig.axis_angle_rotmat([0.0, 0.0, 1.0], 0.2)
    rot[b] = rig.axis_angle_rotmat([0.0, 0.0, 1.0], 0.3)
    world = rig.forward_kinematics(skeleton, rot)
    # both are rotations about +Z, so the child's world rotation is the sum of angles
    r = world[b][:3, :3]
    assert np.arctan2(r[1, 0], r[0, 0]) == pytest.approx(0.5, abs=1e-12)


def test_forward_kinematics_rejects_the_wrong_shape(skeleton):
    with pytest.raises(ValueError):
        rig.forward_kinematics(skeleton, np.tile(np.eye(3), (skeleton.num_joints - 1, 1, 1)))


def test_lbs_rejects_mismatched_weights(capsule, skeleton):
    with pytest.raises(ValueError):
        rig.lbs(
            capsule["vertices"],
            np.ones((len(capsule["vertices"]), skeleton.num_joints + 1)),
            skeleton,
            identity_rotations(skeleton),
        )


# ---------------------------------------------------------------------------
# quaternion helpers (module C builds animation curves with these)
# ---------------------------------------------------------------------------
def test_quaternion_roundtrip():
    rng = np.random.default_rng(0)
    axes = rng.normal(size=(64, 3))
    angles = rng.uniform(-np.pi + 1e-3, np.pi - 1e-3, size=64)
    mats = np.stack([rig.axis_angle_rotmat(a, t) for a, t in zip(axes, angles)])
    quats = rig.rotmat_to_quat(mats)
    np.testing.assert_allclose(np.linalg.norm(quats, axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(rig.quat_to_rotmat(quats), mats, atol=1e-10)


def test_identity_rotation_is_the_identity_quaternion():
    np.testing.assert_allclose(rig.rotmat_to_quat(np.eye(3)), [0.0, 0.0, 0.0, 1.0], atol=1e-12)


def test_tip_of_a_long_low_fin_is_its_apex():
    """A fin longer along the body than it is tall: the old rule (farthest from
    the insertion) picked a base corner, so the fin drive hinged it fore-aft."""
    n_along, n_up = 40, 8
    xs = np.linspace(-0.05, 0.05, n_along)          # 100 mm along the body
    zs = np.linspace(0.02, 0.04, n_up)              # 20 mm tall, root at body radius 0.02
    X, Z = np.meshgrid(xs, zs)
    verts = np.column_stack([X.ravel(), np.zeros(X.size), Z.ravel()])
    insertion = np.array([0.0, 0.0, 0.02])
    fin_info = rig.fin_info_from_detection(
        {"dorsal": {"insertion_centroid": insertion, "vertex_indices": np.arange(len(verts))}}, verts)
    tip = fin_info["dorsal"]["tip"]
    assert tip[2] == pytest.approx(0.04)             # on the apex row, not the z = 0.02 base row
    # a caudal lobe's axis is axial: its tip is the most posterior vertex, even
    # when the detected insertion sits mid-lobe (a tapering lobe's innermost
    # radii are at its far end)
    caudal = np.column_stack([np.linspace(-0.30, -0.20, 50), np.zeros(50), np.linspace(0.01, 0.03, 50)])
    for ins in (caudal[-1], caudal[25], caudal[0]):
        fin_info = rig.fin_info_from_detection(
            {"caudal_upper": {"insertion_centroid": ins, "vertex_indices": np.arange(50)}}, caudal)
        assert fin_info["caudal_upper"]["tip"][0] == pytest.approx(-0.30)
