"""Behavioral tests for the sevengill serial-spine skeleton and Schema S1.

Run from the deliverable directory:

    python -m pytest tests/ -q
"""

from __future__ import annotations

import os
import sys

import pytest
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import skeleton_sevengill as sk  # noqa: E402

YAML_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "keypoints_sevengill_v1.yaml",
)


@pytest.fixture(scope="module")
def schema():
    with open(YAML_PATH, "r") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# The tree is a valid tree
# ---------------------------------------------------------------------------


def test_exactly_one_root():
    roots = [j for j, p in sk.KINEMATIC_TREE.items() if p is None]
    assert roots == [sk.ROOT_JOINT]
    assert sk.PARENT_INDICES.count(-1) == 1
    assert sk.PARENT_INDICES[sk.ROOT_IDX] == -1


def test_tree_covers_joint_sequence_exactly():
    assert set(sk.KINEMATIC_TREE) == set(sk.JOINT_SEQUENCE)
    assert len(sk.JOINT_SEQUENCE) == len(set(sk.JOINT_SEQUENCE)) == sk.NUM_JOINTS


def test_every_parent_is_a_known_joint():
    for child, parent in sk.KINEMATIC_TREE.items():
        if parent is not None:
            assert parent in sk.JOINT_NAME_TO_IDX, child


def test_no_cycles_and_every_joint_reaches_the_root():
    for joint in sk.JOINT_SEQUENCE:
        chain = sk.get_kinematic_chain(joint)
        assert chain[0] == sk.ROOT_JOINT
        assert chain[-1] == joint
        assert len(chain) == len(set(chain)), "cycle through {0}".format(joint)


def test_parents_precede_children_in_the_joint_sequence():
    for idx, parent_idx in enumerate(sk.PARENT_INDICES):
        if parent_idx >= 0:
            assert parent_idx < idx, sk.JOINT_SEQUENCE[idx]


# ---------------------------------------------------------------------------
# The spine is SERIAL, not a star — the whole point of this module
# ---------------------------------------------------------------------------


def test_spine_is_a_serial_chain_derived_from_the_tree():
    assert sk.get_spine_chain() == sk.SPINE_JOINTS


def test_every_spine_joint_has_exactly_one_spine_child_except_the_last():
    spine = set(sk.SPINE_JOINTS)
    for name in sk.SPINE_JOINTS[:-1]:
        assert len([c for c in sk.get_children(name) if c in spine]) == 1, name
    assert [c for c in sk.get_children(sk.SPINE_JOINTS[-1]) if c in spine] == []


def test_spine_length_is_in_the_designed_10_to_20_range():
    assert 10 <= sk.NUM_SPINE_JOINTS <= 20
    assert sk.NUM_SPINE_SEGMENTS == sk.NUM_SPINE_JOINTS - 1


def test_axial_bending_dof_far_exceeds_the_white_shark_star():
    # The white-shark tree parents nearly everything to one mid-body root, leaving two
    # body-axis bending joints for the whole animal. This is the regression that matters.
    spine = set(sk.SPINE_JOINTS)
    axial_joints = [
        name
        for name in sk.SPINE_JOINTS[1:]
        if sk.KINEMATIC_TREE[name] in spine
    ]
    assert len(axial_joints) >= 10


def test_no_joint_is_a_fin_tip_parent_of_a_chain():
    for tip in ("pectoral_fin_tip", "dorsal_fin_apex", "caudal_upper_lobe_tip",
                "caudal_lower_lobe_tip"):
        assert sk.get_children(tip) == [], tip


def test_caudal_axis_continues_into_the_upper_lobe_not_a_symmetric_branch():
    # Heterocercal tail: the axis turns up into the long upper lobe.
    assert sk.KINEMATIC_TREE["caudal_upper_lobe_tip"] == "spine_12_caudal_axis_2"
    assert sk.KINEMATIC_TREE["caudal_lower_lobe_tip"] == "spine_11_caudal_axis_1"


# ---------------------------------------------------------------------------
# Adjacency is consistent with the tree
# ---------------------------------------------------------------------------


def test_adjacency_is_symmetric_hollow_and_binary():
    adj = sk.ADJACENCY_MATRIX
    assert adj.shape == (sk.NUM_JOINTS, sk.NUM_JOINTS)
    assert torch.equal(adj, adj.transpose(0, 1))
    assert torch.all(torch.diagonal(adj) == 0)
    assert torch.all((adj == 0) | (adj == 1))


def test_adjacency_agrees_with_bone_pairs():
    adj = sk.ADJACENCY_MATRIX
    assert sk.NUM_BONES == sk.NUM_JOINTS - 1
    assert len(sk.BONE_PAIRS) == sk.NUM_BONES
    assert float(adj.sum()) == 2.0 * sk.NUM_BONES
    for parent_idx, child_idx in sk.BONE_PAIRS:
        assert adj[parent_idx, child_idx] == 1.0
        assert sk.PARENT_INDICES[child_idx] == parent_idx


def test_adjacency_has_no_edge_that_is_not_a_bone():
    bones = set()
    for parent_idx, child_idx in sk.BONE_PAIRS:
        bones.add((parent_idx, child_idx))
        bones.add((child_idx, parent_idx))
    adj = sk.ADJACENCY_MATRIX
    for i in range(sk.NUM_JOINTS):
        for j in range(sk.NUM_JOINTS):
            if adj[i, j] == 1.0:
                assert (i, j) in bones


# ---------------------------------------------------------------------------
# Keypoint <-> joint mapping is TOTAL against the yaml
# ---------------------------------------------------------------------------


def test_yaml_keypoint_order_matches_the_module(schema):
    names = [kp["name"] for kp in schema["keypoints"]]
    ids = [kp["id"] for kp in schema["keypoints"]]
    assert ids == list(range(len(ids)))
    assert names == sk.SEVENGILL_KEYPOINT_SEQUENCE
    assert schema["schema"]["num_keypoints"] == sk.NUM_KEYPOINTS == 30


def test_every_yaml_keypoint_maps_to_an_existing_joint(schema):
    for kp in schema["keypoints"]:
        name = kp["name"]
        assert name in sk.KEYPOINT_TO_JOINT, name
        joint = sk.KEYPOINT_TO_JOINT[name]
        assert joint in sk.JOINT_NAME_TO_IDX, (name, joint)


def test_keypoint_to_joint_is_injective_and_indices_are_in_range():
    idxs = sk.keypoints_to_joint_indices()
    assert len(idxs) == sk.NUM_KEYPOINTS
    assert len(set(idxs)) == sk.NUM_KEYPOINTS  # no two keypoints share a joint
    assert all(0 <= i < sk.NUM_JOINTS for i in idxs)


def test_the_unmapped_joints_are_exactly_the_unlabelled_spine_stations():
    assert set(sk.UNLABELLED_JOINTS) == {
        "spine_00_cranium",
        "spine_01_branchial_1",
        "spine_02_branchial_7",
        "spine_11_caudal_axis_1",
        "spine_12_caudal_axis_2",
    }
    assert len(sk.UNLABELLED_JOINTS) + sk.NUM_KEYPOINTS == sk.NUM_JOINTS


def test_midline_keypoints_map_onto_consecutive_spine_stations(schema):
    fractions = tuple(schema["midline_definition"]["fractions"])
    assert fractions == sk.MIDLINE_AXIS_FRACTIONS
    mapped = [sk.KEYPOINT_TO_JOINT["midline_{0:02d}".format(k + 1)]
              for k in range(len(fractions))]
    start = sk.SPINE_JOINTS.index(mapped[0])
    assert mapped == sk.SPINE_JOINTS[start:start + len(fractions)]


def test_midline_fractions_are_deterministic_not_by_eye(schema):
    md = schema["midline_definition"]
    n = md["n_points"]
    assert md["fractions"] == [(k + 1) / float(n + 1) for k in range(n)]
    assert 4 <= n <= 8
    assert md["axis_origin"] == "snout_tip"
    assert md["axis_terminus"] == "precaudal_pit"


# ---------------------------------------------------------------------------
# Schema S1 internal consistency and the sevengill anatomy contract
# ---------------------------------------------------------------------------


def test_no_second_dorsal_anywhere_in_the_schema(schema):
    text = yaml.safe_dump(schema)
    names = [kp["name"] for kp in schema["keypoints"]]
    assert not any("second_dorsal" in n for n in names)
    assert "interdorsal_space" in schema["undefined_for_this_species"]
    assert "second_dorsal_height" in schema["undefined_for_this_species"]
    assert "interdorsal_space" not in schema["morphometric_pairs"]
    assert "second_dorsal" in text  # only as an explicit statement of absence


def test_seven_gill_slits_are_named_as_such(schema):
    names = [kp["name"] for kp in schema["keypoints"]]
    assert "gill_slit_1_dorsal_origin" in names
    assert "gill_slit_7_dorsal_origin" in names
    assert len(schema["ordered_ap_sequence"]["sequence"]) > 7
    slits = [s for s in schema["ordered_ap_sequence"]["sequence"] if s.startswith("gill_slit_")]
    assert slits == ["gill_slit_{0}".format(i) for i in range(1, 8)]


def test_no_ap_order_is_asserted_among_pelvic_dorsal_and_cloaca(schema):
    # The scan's prose and an earlier draft of this schema disagreed on the order of these
    # three, and the yaml's own cloaca definition contradicted both. No order may be asserted.
    block = schema["ordered_ap_sequence"]
    trio = {"pelvic_origin", "dorsal_fin_origin", "cloaca"}
    assert set(block["unordered_posterior_trio"]["members"]) == trio
    assert block["unordered_posterior_trio"]["constraint"] == "none"
    assert "[UNVERIFIED]" in block["unordered_posterior_trio"]["grade"]
    assert not trio & set(block["sequence"]), "the trio must not appear in the ordered sequence"
    cloaca = next(kp for kp in schema["keypoints"] if kp["name"] == "cloaca")
    assert "[UNVERIFIED]" in cloaca["sevengill_note"]
    assert "unordered_posterior_trio" in cloaca["sevengill_note"]


def test_tier_and_type_classifications_partition_the_schema(schema):
    ids = {kp["id"] for kp in schema["keypoints"]}
    tiers = schema["tier_classification"]
    tier_ids = tiers["tier_1"] + tiers["tier_2"] + tiers["tier_3"]
    assert sorted(tier_ids) == sorted(ids)
    types = schema["type_classification"]
    type_ids = types["type_I"] + types["type_II"] + types["type_III"]
    assert sorted(type_ids) == sorted(ids)


def test_per_keypoint_tier_and_type_agree_with_the_index_blocks(schema):
    tiers = schema["tier_classification"]
    types = schema["type_classification"]
    for kp in schema["keypoints"]:
        assert kp["id"] in tiers["tier_{0}".format(kp["tier"])], kp["name"]
        assert kp["id"] in types["type_{0}".format(kp["type"])], kp["name"]


def test_every_keypoint_declares_definition_type_and_lateral_visibility(schema):
    allowed = {"high", "moderate", "low", "never"}
    for kp in schema["keypoints"]:
        assert kp["description"].strip()
        assert kp["placement_guide"].strip()
        assert kp["type"] in {"I", "II", "III"}
        assert kp["lateral_visibility"] in allowed
        assert kp["tier"] in {1, 2, 3}


def test_skeleton_edges_and_morphometrics_reference_valid_ids(schema):
    ids = {kp["id"] for kp in schema["keypoints"]}
    for a, b in schema["skeleton_edges"]:
        assert a in ids and b in ids and a != b
    for pair in schema["morphometric_pairs"].values():
        assert len(pair) == 2
        assert all(i in ids for i in pair)


def _contract_kinematic_tree_onto_keypoints():
    """Recompute the yaml's skeleton_edges from KINEMATIC_TREE.

    For each keypoint k with joint j(k), walk j(k)'s ancestors to the nearest ancestor that is
    itself the image of some keypoint k', and emit (k', k). A keypoint whose joint has no
    keypoint ancestor emits no edge.
    """
    joint_to_kp = {joint: kp for kp, joint in sk.KEYPOINT_TO_JOINT.items()}
    kp_id = {name: i for i, name in enumerate(sk.SEVENGILL_KEYPOINT_SEQUENCE)}
    edges, roots = set(), set()
    for kp in sk.SEVENGILL_KEYPOINT_SEQUENCE:
        ancestor = sk.KINEMATIC_TREE[sk.KEYPOINT_TO_JOINT[kp]]
        while ancestor is not None and ancestor not in joint_to_kp:
            ancestor = sk.KINEMATIC_TREE[ancestor]
        if ancestor is None:
            roots.add(kp_id[kp])
        else:
            edges.add((kp_id[joint_to_kp[ancestor]], kp_id[kp]))
    return edges, roots


def test_yaml_skeleton_edges_are_the_true_contraction_of_the_kinematic_tree(schema):
    # The defect this pins: a hand-drawn edge list can assert parent/child relations the
    # skeleton does not contain (siblings and cousins drawn as bones).
    edges, roots = _contract_kinematic_tree_onto_keypoints()
    assert {tuple(e) for e in schema["skeleton_edges"]} == edges
    assert set(schema["skeleton_contraction_roots"]) == roots


def test_yaml_skeleton_edges_form_a_forest_partitioning_the_keypoints(schema):
    # A forest, NOT a spanning tree: the joint-tree root spine_00_cranium is unlabelled, so the
    # cranial and branchial keypoints have no keypoint ancestor to hang from.
    ids = {kp["id"] for kp in schema["keypoints"]}
    edges = [tuple(e) for e in schema["skeleton_edges"]]
    roots = set(schema["skeleton_contraction_roots"])
    assert len(edges) + len(roots) == len(ids)
    parents = {}
    for a, b in edges:
        assert a in ids and b in ids
        assert b not in roots, "keypoint {0} is both a root and a child".format(b)
        assert b not in parents, "keypoint {0} has two parents".format(b)
        parents[b] = a
    assert set(parents) | roots == ids
    for kp in ids:  # every keypoint reaches a root without cycling
        seen, cur = set(), kp
        while cur in parents:
            assert cur not in seen, "cycle through {0}".format(kp)
            seen.add(cur)
            cur = parents[cur]
        assert cur in roots


def test_morphometric_pairs_resolve_to_real_joints():
    for name, (a, b) in sk.MORPHOMETRIC_PAIRS.items():
        assert a in sk.JOINT_NAME_TO_IDX, name
        assert b in sk.JOINT_NAME_TO_IDX, name


# ---------------------------------------------------------------------------
# No mirror augmentation
# ---------------------------------------------------------------------------


def test_mirror_augmentation_is_disabled_in_both_schema_and_module(schema):
    assert schema["no_mirror_augmentation"]["allow_horizontal_flip"] is False
    assert schema["yolo_config"]["augment_overrides"]["fliplr"] == 0.0
    assert sk.ALLOW_HORIZONTAL_FLIP is False


def test_flip_idx_is_identity_because_the_schema_is_single_sided(schema):
    cfg = schema["yolo_config"]
    assert cfg["flip_idx"] == list(range(cfg["num_keypoints"]))
    assert cfg["kpt_shape"] == [sk.NUM_KEYPOINTS, 3]


def test_side_convention_is_a_record_field_not_duplicated_keypoints(schema):
    conv = schema["side_convention"]
    assert conv["field"] == "side"
    assert set(conv["values"]) == {"L", "R", "unknown"}
    names = [kp["name"] for kp in schema["keypoints"]]
    assert not any(n.endswith("_left") or n.endswith("_right") for n in names)


# ---------------------------------------------------------------------------
# Bending modes
# ---------------------------------------------------------------------------


def test_bending_basis_is_orthonormal():
    basis = sk.build_bending_basis()
    assert basis.shape == (sk.NUM_BENDING_MODES, sk.NUM_SPINE_SEGMENTS)
    gram = basis @ basis.transpose(0, 1)
    assert torch.allclose(gram, torch.eye(sk.NUM_BENDING_MODES), atol=1e-5)


def test_bending_modes_are_in_the_designed_4_to_6_bracket():
    assert 4 <= sk.NUM_BENDING_MODES <= 6


def test_projection_round_trips_a_profile_inside_the_span():
    torch.manual_seed(0)
    coeffs = torch.randn(sk.NUM_BENDING_MODES)
    angles = sk.reconstruct_from_bending_modes(coeffs)
    recovered = sk.project_to_bending_modes(angles)
    assert torch.allclose(recovered, coeffs, atol=1e-5)


def test_projection_discards_global_heading():
    torch.manual_seed(0)
    angles = torch.randn(sk.NUM_SPINE_SEGMENTS)
    before = sk.project_to_bending_modes(angles)
    after = sk.project_to_bending_modes(angles + 0.7)
    assert torch.allclose(before, after, atol=1e-6)


def test_a_straight_spine_has_zero_bending_coefficients():
    xs = torch.arange(sk.NUM_SPINE_JOINTS, dtype=torch.float32)
    positions = torch.stack([xs, torch.zeros_like(xs)], dim=-1)
    angles = sk.spine_tangent_angles(positions)
    assert torch.allclose(angles, torch.zeros_like(angles), atol=1e-6)
    coeffs = sk.project_to_bending_modes(angles)
    assert torch.allclose(coeffs, torch.zeros_like(coeffs), atol=1e-6)


def test_a_bent_spine_has_nonzero_bending_coefficients():
    thetas = torch.linspace(0.0, 1.2, sk.NUM_SPINE_JOINTS)
    positions = torch.stack([torch.sin(thetas), 1.0 - torch.cos(thetas)], dim=-1)
    angles = sk.spine_tangent_angles(positions)
    coeffs = sk.project_to_bending_modes(angles)
    assert float(coeffs.abs().max()) > 1e-3


def test_spine_tangent_angles_rejects_the_wrong_shape():
    with pytest.raises(ValueError):
        sk.spine_tangent_angles(torch.zeros(sk.NUM_SPINE_JOINTS, 3))
    with pytest.raises(ValueError):
        sk.spine_tangent_angles(torch.zeros(sk.NUM_SPINE_JOINTS - 1, 2))


def test_spine_tangent_angles_supports_a_batch_dimension():
    xs = torch.arange(sk.NUM_SPINE_JOINTS, dtype=torch.float32)
    positions = torch.stack([xs, torch.zeros_like(xs)], dim=-1)
    batch = positions.unsqueeze(0).expand(4, -1, -1)
    assert sk.spine_tangent_angles(batch).shape == (4, sk.NUM_SPINE_SEGMENTS)


def test_build_bending_basis_rejects_impossible_mode_counts():
    with pytest.raises(ValueError):
        sk.build_bending_basis(n_modes=0)
    with pytest.raises(ValueError):
        sk.build_bending_basis(n_modes=sk.NUM_SPINE_SEGMENTS + 1)


def test_the_dct_null_mode_is_rejected_not_normalised_into_garbage():
    # DCT-II mode k = n_segments is identically zero at these sample points; normalising it
    # divides by float noise and produced max|BB^T - I| = 0.607 before this guard.
    with pytest.raises(ValueError):
        sk.build_bending_basis(n_modes=sk.NUM_SPINE_SEGMENTS)


def test_every_valid_mode_count_gives_an_orthonormal_basis():
    for n_modes in range(1, sk.NUM_SPINE_SEGMENTS):
        basis = sk.build_bending_basis(n_modes=n_modes, dtype=torch.float64)
        assert basis.shape == (n_modes, sk.NUM_SPINE_SEGMENTS)
        gram = basis @ basis.transpose(0, 1)
        err = float((gram - torch.eye(n_modes, dtype=torch.float64)).abs().max())
        assert err < 1e-9, (n_modes, err)
