"""Tests for the skinned/animated GLB writer.

Every GLB written here is run through the Khronos glTF validator (Node subprocess)
and must report ZERO errors; warnings are surfaced in the assertion message.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pygltflib
import pytest
import trimesh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fixtures_rig as fx  # noqa: E402
import gltf_export  # noqa: E402
import rig  # noqa: E402

_COMPONENT_DTYPE = {5120: np.int8, 5121: np.uint8, 5122: np.int16,
                    5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


@pytest.fixture(scope="module")
def capsule():
    return fx.straight_capsule()


@pytest.fixture(scope="module")
def rigged(capsule):
    skeleton = rig.build_skeleton(capsule["centerline"], capsule["fin_info"])
    weights = rig.compute_weights(capsule["vertices"], capsule["labels"], skeleton)
    return skeleton, weights


@pytest.fixture(scope="module")
def swim_animation(rigged):
    """A short anguilliform wave: amplitude growing head-to-tail, phase travelling."""
    skeleton, _ = rigged
    n_joints = skeleton.num_joints
    times = np.linspace(0.0, 1.0, 17)
    rotations = np.zeros((len(times), n_joints, 4))
    rotations[..., 3] = 1.0
    spine = skeleton.spine_indices
    for t, time in enumerate(times):
        for n, j in enumerate(spine):
            gain = (n / float(len(spine) - 1)) ** 2
            angle = np.deg2rad(9.0) * gain * np.sin(2.0 * np.pi * (time - 0.12 * n))
            rotations[t, j] = rig.rotmat_to_quat(rig.axis_angle_rotmat([0.0, 0.0, 1.0], angle))
    return {"name": "cruise", "times": times, "rotations": rotations}


def read_accessor(gltf, index, blob):
    acc = gltf.accessors[index]
    view = gltf.bufferViews[acc.bufferView]
    dtype = _COMPONENT_DTYPE[acc.componentType]
    width = _WIDTH[acc.type]
    start = (view.byteOffset or 0)
    count = acc.count * width
    raw = blob[start:start + count * np.dtype(dtype).itemsize]
    return np.frombuffer(raw, dtype=dtype).reshape(acc.count, width)


def write(tmp_path, capsule, rigged, name="rig.glb", animations=None, mesh=None):
    skeleton, weights = rigged
    if mesh is None:
        mesh = fx.as_trimesh(capsule)
    path = str(tmp_path / name)
    gltf_export.write_skinned_glb(mesh, skeleton, weights, path, animations=animations)
    return path


# ---------------------------------------------------------------------------
# validator
# ---------------------------------------------------------------------------
def test_skinned_glb_passes_the_khronos_validator(tmp_path, capsule, rigged):
    path = write(tmp_path, capsule, rigged)
    issues = gltf_export.validate_glb(path, raise_on_error=False)
    assert issues["numErrors"] == 0, issues["messages"]
    assert issues["numWarnings"] == 0, issues["messages"]


def test_animated_glb_passes_the_khronos_validator(tmp_path, capsule, rigged, swim_animation):
    path = write(tmp_path, capsule, rigged, "anim.glb", animations=[swim_animation])
    issues = gltf_export.validate_glb(path, raise_on_error=False)
    assert issues["numErrors"] == 0, issues["messages"]
    assert issues["numWarnings"] == 0, issues["messages"]


def test_untextured_mesh_without_uvs_still_validates(tmp_path, capsule, rigged):
    mesh = fx.as_trimesh(capsule, textured=False)
    path = write(tmp_path, capsule, rigged, "plain.glb", mesh=mesh)
    issues = gltf_export.validate_glb(path, raise_on_error=False)
    assert issues["numErrors"] == 0, issues["messages"]
    gltf = pygltflib.GLTF2().load(path)
    assert gltf.meshes[0].primitives[0].attributes.TEXCOORD_0 is None
    assert not gltf.textures


# ---------------------------------------------------------------------------
# reload
# ---------------------------------------------------------------------------
def test_glb_reloads_in_trimesh_with_the_same_geometry(tmp_path, capsule, rigged):
    path = write(tmp_path, capsule, rigged)
    scene = trimesh.load(path, process=False)
    geom = list(scene.geometry.values())[0] if hasattr(scene, "geometry") else scene
    assert len(geom.vertices) == len(capsule["vertices"])
    assert len(geom.faces) == len(capsule["faces"])
    np.testing.assert_allclose(geom.vertices, capsule["vertices"], atol=1e-6)
    np.testing.assert_array_equal(geom.faces, capsule["faces"])


def test_uvs_and_texture_survive_the_export(tmp_path, capsule, rigged):
    """Positions are the only thing the pipeline moves; UVs and the texture ride along."""
    path = write(tmp_path, capsule, rigged)
    scene = trimesh.load(path, process=False)
    geom = list(scene.geometry.values())[0] if hasattr(scene, "geometry") else scene
    np.testing.assert_allclose(geom.visual.uv, capsule["uv"], atol=1e-6)
    image = geom.visual.material.baseColorTexture
    assert image is not None and image.size == (32, 32)
    original = fx.as_trimesh(capsule).visual.material.baseColorTexture
    np.testing.assert_array_equal(np.asarray(image), np.asarray(original))
    # on disk V is flipped (glTF measures V from the top of the image, trimesh from
    # the bottom); the round trip above is what has to be the identity
    gltf = pygltflib.GLTF2().load(path)
    stored = read_accessor(
        gltf, gltf.meshes[0].primitives[0].attributes.TEXCOORD_0, gltf.binary_blob()
    )
    np.testing.assert_allclose(stored[:, 1], 1.0 - capsule["uv"][:, 1], atol=1e-6)
    np.testing.assert_allclose(stored[:, 0], capsule["uv"][:, 0], atol=1e-6)


# ---------------------------------------------------------------------------
# skin structure
# ---------------------------------------------------------------------------
def test_one_node_per_joint_parented_like_the_skeleton(tmp_path, capsule, rigged):
    skeleton, _ = rigged
    path = write(tmp_path, capsule, rigged)
    gltf = pygltflib.GLTF2().load(path)
    assert len(gltf.nodes) == skeleton.num_joints + 1  # + the skinned mesh node
    assert gltf.skins[0].joints == list(range(skeleton.num_joints))
    for j in range(skeleton.num_joints):
        node = gltf.nodes[j]
        assert node.name == skeleton.names[j]
        parent = int(skeleton.parents[j])
        origin = np.zeros(3) if parent == -1 else skeleton.joints[parent]
        np.testing.assert_allclose(node.translation, skeleton.joints[j] - origin, atol=1e-6)
        np.testing.assert_allclose(node.rotation, [0.0, 0.0, 0.0, 1.0])
        children = sorted(node.children or [])
        assert children == sorted(skeleton.children(j))
    mesh_node = gltf.nodes[skeleton.num_joints]
    assert mesh_node.mesh == 0 and mesh_node.skin == 0
    assert sorted(gltf.scenes[0].nodes) == sorted([0, skeleton.num_joints])


def test_inverse_bind_matrices_undo_the_rest_pose(tmp_path, capsule, rigged):
    skeleton, _ = rigged
    path = write(tmp_path, capsule, rigged)
    gltf = pygltflib.GLTF2().load(path)
    blob = gltf.binary_blob()
    ibm = read_accessor(gltf, gltf.skins[0].inverseBindMatrices, blob)
    ibm = ibm.reshape(-1, 4, 4).transpose(0, 2, 1)  # glTF is column-major
    expected = np.tile(np.eye(4), (skeleton.num_joints, 1, 1))
    expected[:, :3, 3] = -skeleton.joints
    np.testing.assert_allclose(ibm, expected, atol=1e-6)


def test_joints_and_weights_use_the_gltf_types(tmp_path, capsule, rigged):
    skeleton, weights = rigged
    path = write(tmp_path, capsule, rigged)
    gltf = pygltflib.GLTF2().load(path)
    blob = gltf.binary_blob()
    attrs = gltf.meshes[0].primitives[0].attributes
    j_acc = gltf.accessors[attrs.JOINTS_0]
    w_acc = gltf.accessors[attrs.WEIGHTS_0]
    assert j_acc.componentType == pygltflib.UNSIGNED_SHORT and j_acc.type == "VEC4"
    assert w_acc.componentType == pygltflib.FLOAT and w_acc.type == "VEC4"
    joints = read_accessor(gltf, attrs.JOINTS_0, blob)
    w = read_accessor(gltf, attrs.WEIGHTS_0, blob)
    assert joints.max() < skeleton.num_joints
    np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-7)
    # the indexed pairs reproduce the dense weight matrix
    dense = np.zeros_like(weights)
    rows = np.repeat(np.arange(len(joints)), 4)
    np.add.at(dense, (rows, joints.reshape(-1)), w.reshape(-1))
    np.testing.assert_allclose(dense, weights, atol=1e-6)


def test_gltf_skinning_reproduces_rig_lbs(tmp_path, capsule, rigged, swim_animation):
    """The exported bind pose and the rig's FK are the same convention, not two.

    Evaluates glTF's own skinning equation (globalJointTransform @ inverseBindMatrix,
    blended by JOINTS_0/WEIGHTS_0) on a mid-animation frame straight out of the file
    and compares against ``rig.lbs`` on the same pose.
    """
    skeleton, weights = rigged
    path = write(tmp_path, capsule, rigged, "anim.glb", animations=[swim_animation])
    gltf = pygltflib.GLTF2().load(path)
    blob = gltf.binary_blob()
    frame = 5

    quats = swim_animation["rotations"][frame]
    globals_ = np.zeros((skeleton.num_joints, 4, 4))
    for j in range(skeleton.num_joints):
        node = gltf.nodes[j]
        local = np.eye(4)
        local[:3, :3] = rig.quat_to_rotmat(np.asarray(quats[j]))
        local[:3, 3] = node.translation
        parent = int(skeleton.parents[j])
        globals_[j] = local if parent == -1 else globals_[parent] @ local
    ibm = read_accessor(gltf, gltf.skins[0].inverseBindMatrices, blob)
    ibm = ibm.reshape(-1, 4, 4).transpose(0, 2, 1)
    skin_mats = globals_ @ ibm

    attrs = gltf.meshes[0].primitives[0].attributes
    joints = read_accessor(gltf, attrs.JOINTS_0, blob).astype(int)
    jw = read_accessor(gltf, attrs.WEIGHTS_0, blob).astype(float)
    verts = np.asarray(capsule["vertices"])
    homo = np.concatenate([verts, np.ones((len(verts), 1))], axis=1)
    blended = (skin_mats[joints] * jw[:, :, None, None]).sum(axis=1)
    gltf_posed = np.einsum("nab,nb->na", blended, homo)[:, :3]

    rotmats = rig.quat_to_rotmat(quats)
    expected = rig.lbs(verts, weights, skeleton, rotmats)
    np.testing.assert_allclose(gltf_posed, expected, atol=1e-5)
    assert np.abs(expected - verts).max() > 1e-3  # the frame really is posed


# ---------------------------------------------------------------------------
# animation
# ---------------------------------------------------------------------------
def test_animation_reloads_with_one_rotation_channel_per_joint(
    tmp_path, capsule, rigged, swim_animation
):
    skeleton, _ = rigged
    path = write(tmp_path, capsule, rigged, "anim.glb", animations=[swim_animation])
    gltf = pygltflib.GLTF2().load(path)
    assert len(gltf.animations) == 1
    anim = gltf.animations[0]
    assert anim.name == "cruise"
    assert len(anim.channels) == skeleton.num_joints
    assert len(anim.samplers) == skeleton.num_joints
    assert all(c.target.path == "rotation" for c in anim.channels)
    assert sorted(c.target.node for c in anim.channels) == list(range(skeleton.num_joints))
    assert all(s.interpolation == "LINEAR" for s in anim.samplers)

    blob = gltf.binary_blob()
    times = read_accessor(gltf, anim.samplers[0].input, blob).reshape(-1)
    np.testing.assert_allclose(times, swim_animation["times"], atol=1e-6)
    tail = skeleton.index("spine_12_caudal_axis_2")
    channel = [c for c in anim.channels if c.target.node == tail][0]
    out = read_accessor(gltf, anim.samplers[channel.sampler].output, blob)
    np.testing.assert_allclose(out, swim_animation["rotations"][:, tail, :], atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-6)


def _rest_translations(skeleton, n_times):
    n_joints = skeleton.num_joints
    rest = np.zeros((n_joints, 3))
    for j in range(n_joints):
        parent = int(skeleton.parents[j])
        rest[j] = skeleton.joints[j] - (0.0 if parent == -1 else skeleton.joints[parent])
    return np.tile(rest, (n_times, 1, 1))


def test_translations_add_a_channel_only_for_joints_that_move(
    tmp_path, capsule, rigged, swim_animation
):
    """REGRESSION (fix M4): a constant translation curve is not a channel.

    ``as_scanned`` hands the exporter a (T, J, 3) translation array in which only the
    ROOT moves -- every other joint restates its own rest offset once per frame.
    Writing all J of them cost 28 dead samplers, 28 dead accessors and 28 dead buffer
    views on the demo rig, and made every bone look animated in a viewer's channel
    list.  Only joints whose translation actually varies get a channel now.
    """
    skeleton, _ = rigged
    n_joints = skeleton.num_joints
    times = swim_animation["times"]
    translations = _rest_translations(skeleton, len(times))
    translations[:, 0, 0] += 0.05 * np.sin(2 * np.pi * times)
    anim = dict(swim_animation)
    anim["translations"] = translations
    path = write(tmp_path, capsule, rigged, "anim_t.glb", animations=[anim])
    issues = gltf_export.validate_glb(path, raise_on_error=False)
    assert issues["numErrors"] == 0, issues["messages"]
    gltf = pygltflib.GLTF2().load(path)
    channels = gltf.animations[0].channels
    moving = [c for c in channels if c.target.path == "translation"]
    assert len(moving) == 1, "only joint 0 moves"
    assert moving[0].target.node == 0
    assert len(channels) == n_joints + 1
    assert len(gltf.animations[0].samplers) == n_joints + 1


def test_every_moving_joint_still_gets_its_translation_channel(
    tmp_path, capsule, rigged, swim_animation
):
    """The pruning is on CONSTANT channels, not on translation channels at all."""
    skeleton, _ = rigged
    n_joints = skeleton.num_joints
    times = swim_animation["times"]
    translations = _rest_translations(skeleton, len(times))
    translations[:, :, 0] += 0.01 * np.sin(2 * np.pi * times)[:, None]
    anim = dict(swim_animation)
    anim["translations"] = translations
    path = write(tmp_path, capsule, rigged, "anim_t_all.glb", animations=[anim])
    gltf = pygltflib.GLTF2().load(path)
    channels = gltf.animations[0].channels
    assert sum(1 for c in channels if c.target.path == "translation") == n_joints
    assert len(channels) == 2 * n_joints


def test_a_wholly_constant_translation_array_writes_no_translation_channel(
    tmp_path, capsule, rigged, swim_animation
):
    skeleton, _ = rigged
    times = swim_animation["times"]
    anim = dict(swim_animation)
    anim["translations"] = _rest_translations(skeleton, len(times))
    path = write(tmp_path, capsule, rigged, "anim_t_const.glb", animations=[anim])
    issues = gltf_export.validate_glb(path, raise_on_error=False)
    assert issues["numErrors"] == 0, issues["messages"]
    gltf = pygltflib.GLTF2().load(path)
    channels = gltf.animations[0].channels
    assert all(c.target.path == "rotation" for c in channels)
    assert len(channels) == skeleton.num_joints


def test_two_animations_are_written_side_by_side(tmp_path, capsule, rigged, swim_animation):
    second = dict(swim_animation)
    second["name"] = "escape"
    path = write(tmp_path, capsule, rigged, "two.glb", animations=[swim_animation, second])
    gltf = pygltflib.GLTF2().load(path)
    assert [a.name for a in gltf.animations] == ["cruise", "escape"]
    issues = gltf_export.validate_glb(path, raise_on_error=False)
    assert issues["numErrors"] == 0, issues["messages"]


def test_malformed_animations_are_rejected(tmp_path, capsule, rigged, swim_animation):
    skeleton, weights = rigged
    mesh = fx.as_trimesh(capsule)
    path = str(tmp_path / "bad.glb")
    bad_times = dict(swim_animation)
    bad_times["times"] = swim_animation["times"][::-1]
    with pytest.raises(ValueError):
        gltf_export.write_skinned_glb(mesh, skeleton, weights, path, animations=[bad_times])
    bad_shape = dict(swim_animation)
    bad_shape["rotations"] = swim_animation["rotations"][:, :-1, :]
    with pytest.raises(ValueError):
        gltf_export.write_skinned_glb(mesh, skeleton, weights, path, animations=[bad_shape])


def test_fewer_influences_still_export_as_vec4(tmp_path, capsule, rigged):
    skeleton, weights = rigged
    mesh = fx.as_trimesh(capsule)
    path = str(tmp_path / "one.glb")
    gltf_export.write_skinned_glb(mesh, skeleton, weights, path, max_influences=1)
    issues = gltf_export.validate_glb(path, raise_on_error=False)
    assert issues["numErrors"] == 0, issues["messages"]
    gltf = pygltflib.GLTF2().load(path)
    assert gltf.accessors[gltf.meshes[0].primitives[0].attributes.JOINTS_0].type == "VEC4"
    w = read_accessor(
        gltf, gltf.meshes[0].primitives[0].attributes.WEIGHTS_0, gltf.binary_blob()
    )
    np.testing.assert_allclose(w[:, 0], 1.0, atol=1e-7)
    with pytest.raises(ValueError):
        gltf_export.write_skinned_glb(
            mesh, skeleton, weights, str(tmp_path / "five.glb"), max_influences=5
        )


def test_weight_shape_is_checked(tmp_path, capsule, rigged):
    skeleton, _ = rigged
    mesh = fx.as_trimesh(capsule)
    with pytest.raises(ValueError):
        gltf_export.write_skinned_glb(
            mesh, skeleton, np.ones((3, skeleton.num_joints)), str(tmp_path / "x.glb")
        )
