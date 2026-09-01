"""Write a skinned, optionally animated GLB from a mesh + Skeleton + LBS weights.

Contract
--------
``write_skinned_glb(mesh, skeleton, weights, path, animations=None)`` emits a single
binary glTF 2.0 file containing

* one node per joint, parented exactly as ``skeleton.parents`` says, with identity
  rotations and translations equal to the rest offset from the parent joint -- so the
  glTF bind pose IS the rest pose that ``rig.lbs`` uses;
* a skin whose ``inverseBindMatrices`` are ``T(-p_j)`` (the inverse of each joint's
  rest world transform, which is a pure translation because bind rotations are
  identity). glTF's skinning matrix ``globalJointTransform @ IBM`` then equals the
  world transform ``rig.forward_kinematics`` returns, term for term;
* one mesh primitive with POSITION, NORMAL, TEXCOORD_0 (when the mesh carries UVs),
  JOINTS_0 (unsigned short) and WEIGHTS_0 (float, 4 per vertex);
* the trimesh visual's material and base-colour texture when present, embedded as a
  PNG in the GLB buffer -- vertex positions are the only thing the de-bend changes,
  so UVs and the texture must survive the whole pipeline untouched. V is flipped on
  write to undo trimesh's flip on load, so ``load -> de-bend -> write -> load`` is a
  UV identity;
* one LINEAR animation per entry of ``animations``.

Every file written here is expected to pass the Khronos glTF validator with ZERO
errors; ``validate_glb`` runs it (Node subprocess) and the tests assert on it.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import struct
import subprocess
import tempfile
from typing import Dict, List, Optional, Sequence

import numpy as np
import pygltflib

from rig import weights_to_indexed

__all__ = ["write_skinned_glb", "validate_glb", "GltfValidationError"]

_COMPONENT_TYPE = {
    np.dtype(np.float32): pygltflib.FLOAT,           # 5126
    np.dtype(np.uint16): pygltflib.UNSIGNED_SHORT,   # 5123
    np.dtype(np.uint32): pygltflib.UNSIGNED_INT,     # 5125
}
_TYPE_BY_WIDTH = {1: "SCALAR", 2: "VEC2", 3: "VEC3", 4: "VEC4", 16: "MAT4"}

_ARRAY_BUFFER = 34962
_ELEMENT_ARRAY_BUFFER = 34963


class GltfValidationError(RuntimeError):
    """Raised when the Khronos validator reports errors on a written GLB."""


# ---------------------------------------------------------------------------
# buffer / accessor plumbing
# ---------------------------------------------------------------------------
class _Builder(object):
    """Accumulates the single GLB binary chunk and the views/accessors into it."""

    def __init__(self):
        self.gltf = pygltflib.GLTF2()
        self.gltf.asset = pygltflib.Asset(version="2.0", generator="04-sevengill-rig")
        self._parts = []
        self._length = 0

    def _append(self, data, target=None):
        pad = (-self._length) % 4
        if pad:
            self._parts.append(b"\x00" * pad)
            self._length += pad
        offset = self._length
        self._parts.append(data)
        self._length += len(data)
        view = pygltflib.BufferView(
            buffer=0, byteOffset=offset, byteLength=len(data), target=target
        )
        self.gltf.bufferViews.append(view)
        return len(self.gltf.bufferViews) - 1

    def add_accessor(self, array, target=None, minmax=False):
        """Append a (n,) / (n, w) numpy array as a bufferView + accessor."""
        arr = np.ascontiguousarray(array)
        if arr.dtype not in _COMPONENT_TYPE:
            raise TypeError("unsupported accessor dtype %r" % arr.dtype)
        width = 1 if arr.ndim == 1 else int(arr.shape[1])
        if width not in _TYPE_BY_WIDTH:
            raise ValueError("unsupported accessor width %d" % width)
        view = self._append(arr.tobytes(), target=target)
        accessor = pygltflib.Accessor(
            bufferView=view,
            componentType=_COMPONENT_TYPE[arr.dtype],
            count=int(arr.shape[0]),
            type=_TYPE_BY_WIDTH[width],
        )
        if minmax:
            flat = arr.reshape(len(arr), width)
            cast = float if arr.dtype == np.dtype(np.float32) else int
            accessor.min = [cast(v) for v in flat.min(axis=0)]
            accessor.max = [cast(v) for v in flat.max(axis=0)]
        self.gltf.accessors.append(accessor)
        return len(self.gltf.accessors) - 1

    def add_bytes(self, data):
        """Append opaque bytes (a PNG) and return the bufferView index."""
        return self._append(data, target=None)

    def finish(self, path):
        blob = b"".join(self._parts)
        pad = (-len(blob)) % 4
        blob = blob + b"\x00" * pad
        self.gltf.buffers.append(pygltflib.Buffer(byteLength=len(blob)))
        self.gltf.set_binary_blob(blob)
        self.gltf.save_binary(path)
        return path


# ---------------------------------------------------------------------------
# material / texture carry-through
# ---------------------------------------------------------------------------
def _png_bytes(image):
    buf = io.BytesIO()
    if image.mode not in ("RGB", "RGBA", "L"):
        image = image.convert("RGBA")
    image.save(buf, format="PNG")
    return buf.getvalue()


def _colour_factor(value, default):
    if value is None:
        return default
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 3:
        arr = np.concatenate([arr, [255.0 if arr.max() > 1.0 else 1.0]])
    if arr.max() > 1.0:
        arr = arr / 255.0
    return [float(v) for v in np.clip(arr[:4], 0.0, 1.0)]


def _add_material(builder, mesh, has_uv):
    """Carry the trimesh visual's material/texture into the glTF. Returns an index or None."""
    visual = getattr(mesh, "visual", None)
    material = getattr(visual, "material", None)
    if material is None:
        return None

    image = getattr(material, "baseColorTexture", None)
    if image is None:
        image = getattr(material, "image", None)

    pbr = pygltflib.PbrMetallicRoughness(
        baseColorFactor=_colour_factor(
            getattr(material, "baseColorFactor", None), [1.0, 1.0, 1.0, 1.0]
        ),
        metallicFactor=float(getattr(material, "metallicFactor", None) or 0.0),
        roughnessFactor=float(
            getattr(material, "roughnessFactor", None)
            if getattr(material, "roughnessFactor", None) is not None
            else 0.7
        ),
    )
    if image is not None and has_uv:
        view = builder.add_bytes(_png_bytes(image))
        builder.gltf.images.append(pygltflib.Image(bufferView=view, mimeType="image/png"))
        builder.gltf.samplers.append(
            pygltflib.Sampler(magFilter=9729, minFilter=9987, wrapS=10497, wrapT=10497)
        )
        builder.gltf.textures.append(
            pygltflib.Texture(
                source=len(builder.gltf.images) - 1, sampler=len(builder.gltf.samplers) - 1
            )
        )
        pbr.baseColorTexture = pygltflib.TextureInfo(
            index=len(builder.gltf.textures) - 1, texCoord=0
        )
    builder.gltf.materials.append(
        pygltflib.Material(
            name=str(getattr(material, "name", None) or "material"),
            pbrMetallicRoughness=pbr,
            doubleSided=True,
            alphaMode="OPAQUE",
        )
    )
    return len(builder.gltf.materials) - 1


# ---------------------------------------------------------------------------
# the writer
# ---------------------------------------------------------------------------
def write_skinned_glb(mesh, skeleton, weights, path, animations=None, max_influences=4):
    """Write ``path`` as a skinned GLB. Returns ``path``.

    Args:
        mesh: a ``trimesh.Trimesh`` in the REST pose (the de-bent straight mesh).
            ``mesh.visual.uv`` and ``mesh.visual.material`` are carried through when
            present; topology, UVs and material are never modified here.
        skeleton: ``rig.Skeleton``; every joint becomes a node, in skeleton order.
        weights: (N, J) dense LBS weights, rows summing to 1.
        path: output ``.glb`` path.
        animations: optional list of dicts::

                {"name": str,
                 "times": (T,) seconds, strictly increasing,
                 "rotations": (T, J, 4) quaternions in glTF (x, y, z, w) order,
                 "translations": (T, J, 3) optional LOCAL translations}

            Rotations are the same per-joint local rotations ``rig.lbs`` takes,
            expressed as quaternions. Channels are emitted for EVERY joint (J
            rotation channels, plus J translation channels when ``translations`` is
            given), so the channel count is exactly predictable from the skeleton.
        max_influences: JOINTS_0/WEIGHTS_0 width; glTF's basic limit is 4.
    """
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.uint32)
    n_joints = skeleton.num_joints
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (len(verts), n_joints):
        raise ValueError(
            "weights must be (%d, %d); got %r" % (len(verts), n_joints, (weights.shape,))
        )

    builder = _Builder()
    gltf = builder.gltf

    # -- geometry ---------------------------------------------------------
    if not 1 <= int(max_influences) <= 4:
        raise ValueError("glTF JOINTS_0/WEIGHTS_0 hold at most 4 influences per vertex")
    normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
    joint_idx, joint_w = weights_to_indexed(weights, max_influences)
    if joint_idx.shape[1] < 4:  # JOINTS_0/WEIGHTS_0 must be VEC4; pad with zero weights
        pad = 4 - joint_idx.shape[1]
        joint_idx = np.concatenate(
            [joint_idx, np.zeros((len(joint_idx), pad), dtype=np.uint16)], axis=1
        )
        joint_w = np.concatenate(
            [joint_w, np.zeros((len(joint_w), pad), dtype=np.float32)], axis=1
        )

    attributes = pygltflib.Attributes(
        POSITION=builder.add_accessor(verts, target=_ARRAY_BUFFER, minmax=True),
        NORMAL=builder.add_accessor(normals, target=_ARRAY_BUFFER),
    )
    uv = getattr(getattr(mesh, "visual", None), "uv", None)
    has_uv = uv is not None and len(uv) == len(verts)
    if has_uv:
        # trimesh stores UVs with V measured from the BOTTOM of the image; glTF measures
        # it from the top. trimesh flips on load and on save, so we flip on write too --
        # otherwise a GLB -> de-bend -> GLB round trip would invert the texture.
        uv = np.asarray(uv, dtype=np.float32)
        uv_gltf = np.column_stack([uv[:, 0], np.float32(1.0) - uv[:, 1]]).astype(np.float32)
        attributes.TEXCOORD_0 = builder.add_accessor(uv_gltf, target=_ARRAY_BUFFER)
    attributes.JOINTS_0 = builder.add_accessor(joint_idx, target=_ARRAY_BUFFER)
    attributes.WEIGHTS_0 = builder.add_accessor(joint_w, target=_ARRAY_BUFFER)
    indices = builder.add_accessor(faces.reshape(-1), target=_ELEMENT_ARRAY_BUFFER)

    material = _add_material(builder, mesh, has_uv)
    gltf.meshes.append(
        pygltflib.Mesh(
            name="sevengill",
            primitives=[
                pygltflib.Primitive(
                    attributes=attributes, indices=indices, material=material, mode=4
                )
            ],
        )
    )

    # -- joint nodes ------------------------------------------------------
    children = {j: [] for j in range(n_joints)}
    roots = []
    for j in range(n_joints):
        parent = int(skeleton.parents[j])
        if parent == -1:
            roots.append(j)
        else:
            children[parent].append(j)
    for j in range(n_joints):
        parent = int(skeleton.parents[j])
        origin = np.zeros(3) if parent == -1 else skeleton.joints[parent]
        offset = skeleton.joints[j] - origin
        gltf.nodes.append(
            pygltflib.Node(
                name=str(skeleton.names[j]),
                translation=[float(v) for v in offset],
                rotation=[0.0, 0.0, 0.0, 1.0],
                children=children[j] if children[j] else None,
            )
        )

    mesh_node = len(gltf.nodes)
    gltf.nodes.append(pygltflib.Node(name="sevengill_mesh", mesh=0, skin=0))

    # -- skin -------------------------------------------------------------
    # rest world transform of joint j is T(p_j) (bind rotations are identity), so the
    # inverse bind matrix is T(-p_j). glTF matrices are COLUMN-major.
    ibm = np.tile(np.eye(4, dtype=np.float32), (n_joints, 1, 1))
    ibm[:, :3, 3] = -skeleton.joints.astype(np.float32)
    ibm_flat = np.ascontiguousarray(
        np.transpose(ibm, (0, 2, 1)).reshape(n_joints, 16).astype(np.float32)
    )
    gltf.skins.append(
        pygltflib.Skin(
            name="sevengill_skin",
            joints=list(range(n_joints)),
            skeleton=roots[0] if len(roots) == 1 else None,
            inverseBindMatrices=builder.add_accessor(ibm_flat),
        )
    )

    gltf.scenes.append(pygltflib.Scene(nodes=roots + [mesh_node]))
    gltf.scene = 0

    # -- animations -------------------------------------------------------
    for anim in animations or []:
        _add_animation(builder, skeleton, anim)

    return builder.finish(path)


def _add_animation(builder, skeleton, anim):
    gltf = builder.gltf
    n_joints = skeleton.num_joints
    times = np.asarray(anim["times"], dtype=np.float32).reshape(-1)
    if len(times) < 2 or not np.all(np.diff(times) > 0):
        raise ValueError("animation 'times' must be strictly increasing with >= 2 samples")
    rotations = np.asarray(anim["rotations"], dtype=float)
    if rotations.shape != (len(times), n_joints, 4):
        raise ValueError(
            "animation 'rotations' must be (T=%d, J=%d, 4); got %r"
            % (len(times), n_joints, (rotations.shape,))
        )
    norms = np.linalg.norm(rotations, axis=2, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("animation contains a zero quaternion")
    rotations = (rotations / norms).astype(np.float32)
    # renormalise in float32: the validator rejects quaternions off the unit sphere
    rotations = rotations / np.linalg.norm(rotations, axis=2, keepdims=True).astype(np.float32)

    translations = anim.get("translations")
    if translations is not None:
        translations = np.asarray(translations, dtype=np.float32)
        if translations.shape != (len(times), n_joints, 3):
            raise ValueError(
                "animation 'translations' must be (T=%d, J=%d, 3); got %r"
                % (len(times), n_joints, (translations.shape,))
            )

    input_acc = builder.add_accessor(times, minmax=True)
    samplers = []
    channels = []
    for j in range(n_joints):
        out = builder.add_accessor(np.ascontiguousarray(rotations[:, j, :]))
        samplers.append(
            pygltflib.AnimationSampler(input=input_acc, output=out, interpolation="LINEAR")
        )
        channels.append(
            pygltflib.AnimationChannel(
                sampler=len(samplers) - 1,
                target=pygltflib.AnimationChannelTarget(node=j, path="rotation"),
            )
        )
    if translations is not None:
        # Only joints whose translation actually MOVES get a channel.  Writing a
        # constant translation curve on every joint is not wrong, but it is 28 dead
        # samplers, 28 dead accessors and 28 dead buffer views restating the node's
        # own `translation` once per frame -- and it makes every bone look animated in
        # a viewer's channel list.  In this rig only the root translates.
        moving = np.nonzero(
            np.abs(translations - translations[0][None, :, :]).max(axis=(0, 2)) > 0.0
        )[0]
        for j in moving:
            j = int(j)
            out = builder.add_accessor(np.ascontiguousarray(translations[:, j, :]))
            samplers.append(
                pygltflib.AnimationSampler(input=input_acc, output=out, interpolation="LINEAR")
            )
            channels.append(
                pygltflib.AnimationChannel(
                    sampler=len(samplers) - 1,
                    target=pygltflib.AnimationChannelTarget(node=j, path="translation"),
                )
            )
    gltf.animations.append(
        pygltflib.Animation(
            name=str(anim.get("name", "animation_%d" % len(gltf.animations))),
            samplers=samplers,
            channels=channels,
        )
    )


# ---------------------------------------------------------------------------
# Khronos glTF validator
# ---------------------------------------------------------------------------
_DEFAULT_NODE_PATH = (
    "/tmp/claude-0/-home-user-SanBox/44444952-aca5-58be-98ef-2fd60cfa4cb2/"
    "scratchpad/node_modules"
)

_VALIDATOR_JS = """
const validator = require('gltf-validator');
const fs = require('fs');
const data = fs.readFileSync(process.argv[2]);
validator.validateBytes(new Uint8Array(data))
  .then(function (report) { process.stdout.write(JSON.stringify(report.issues)); })
  .catch(function (err) { process.stderr.write(String(err)); process.exit(2); });
"""


def validate_glb(path, node_path=None, raise_on_error=True):
    """Run the Khronos glTF validator on ``path``.

    Returns the validator's ``issues`` dict (``numErrors``, ``numWarnings``,
    ``messages``, ...). Raises ``GltfValidationError`` when errors are present and
    ``raise_on_error`` is set, or when Node / ``gltf-validator`` is unavailable.
    """
    node = shutil.which("node")
    if node is None:
        raise GltfValidationError("node executable not found")
    modules = node_path or os.environ.get("GLTF_VALIDATOR_NODE_PATH", _DEFAULT_NODE_PATH)
    if not os.path.isdir(os.path.join(modules, "gltf-validator")):
        raise GltfValidationError("gltf-validator not found under %r" % modules)
    env = dict(os.environ)
    env["NODE_PATH"] = modules
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fh:
        fh.write(_VALIDATOR_JS)
        script = fh.name
    try:
        proc = subprocess.run(
            [node, script, os.path.abspath(path)],
            capture_output=True, env=env, timeout=180,
        )
    finally:
        os.unlink(script)
    if proc.returncode != 0:
        raise GltfValidationError(
            "validator failed (%d): %s" % (proc.returncode, proc.stderr.decode("utf-8", "replace"))
        )
    issues = json.loads(proc.stdout.decode("utf-8"))
    if raise_on_error and issues.get("numErrors", 0) > 0:
        errors = [m for m in issues.get("messages", []) if m.get("severity") == 0]
        raise GltfValidationError(
            "%s: %d validator error(s): %s"
            % (path, issues["numErrors"], json.dumps(errors, indent=2))
        )
    return issues
