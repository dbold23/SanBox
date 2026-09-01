"""Module R -- a pure-numpy z-buffer renderer for synthetic sevengill frames.

WHAT THIS IS.  A software rasteriser with no OpenGL, no GPU and no Blender.
It takes posed triangle meshes that carry per-vertex ``(s, phi)`` CANONICAL
CHART coordinates (0 snout -> 1 caudal; ``phi`` 0 = dorsal midline,
``+pi/2`` = the animal's LEFT flank, ``-pi/2`` = right, ``+-pi`` = ventral
midline -- prototype 04's ``TubeCoords`` convention, the same one
``bake.py`` and ``exclusions.py`` use) and produces a bundle of PIXEL-ALIGNED
images: the photograph, its depth, and the ground-truth masks a re-ID
experiment needs in order to know exactly which pixels are allowed to carry
identity evidence.

WHY THE CHART MAPS ARE THE POINT.  Because the pattern lives in chart space
(the binding design decision of prototype 05), every rendered pixel can be
labelled with the ``(s, phi)`` it came from.  That is what makes the identity
mask exact rather than heuristic: the chart-space exclusion regions from
``exclusions.py`` (eye, nares, mouth/jaw, gill slits) are pulled THROUGH the
per-pixel chart ground truth, so no pixel of an eye is ever scored as
identity evidence, at any pose, with no image-space eye detector.

THE IDENTITY MASK, in full::

    identity = visible_skin
               AND NOT exclusion    (chart-space, sampled through chart GT)
               AND NOT shadow       (attached OR cast)
               AND NOT occluded     (kelp, a second shark, anything in front)

CONTRACT DOWNSTREAM (``prototypes/01-melops-ablation/melops_data.load_melops``).
This module renders frames and masks; it does not write the dataset.  The two
things a dataset writer needs from it are here so they cannot drift:
:func:`mask_bbox_ltwh` emits the ``[left, top, width, height]`` FLOAT boxes
that loader parses (and can express one box inside another, which is how
``bbox_head`` relates to the body crop), and :func:`chart_span_mask` cuts the
head/headless split in ARC LENGTH through the chart ground truth rather than
guessing it from the silhouette.

CONTRACT TO PROTOTYPE 04 (not imported here -- another build owns that tree).
Real meshes get their per-vertex chart coordinates from
``mesh3d.tube_coords(mesh, centerline)``::

    tc = mesh3d.tube_coords(mesh, centerline)
    inst = render.Instance(vertices=mesh.vertices, faces=mesh.faces,
                           uv=mesh.visual.uv, texture=tex,
                           vertex_s=tc.s / tc.total_length, vertex_phi=tc.phi)

Nothing in this module knows about prototype 04; it is tested against
``fixtures.make_uv_tube``, which supplies the same two arrays exactly.

COORDINATE AND SIGN CONVENTIONS (all of them, once):
  * World is right-handed.  The fixture tube swims along +X, dorsal +Z,
    animal's left +Y.
  * Camera space: ``x`` right, ``y`` up, ``z`` FORWARD and positive in front
    of the eye.  ``depth`` in the output is that ``z`` (an along-axis depth,
    not a radial distance), ``+inf`` where nothing was hit.
  * Pixel space: ``x`` right, ``y`` DOWN, pixel centres at integer
    coordinates, i.e. pixel ``(row, col)`` has centre ``(col, row)``.
  * ``DirectionalLight.direction`` is the direction the light TRAVELS.  The
    unit vector from a surface point toward the light is ``L = -direction``.
    ``ndotl = dot(N, L)``; ``ndotl <= 0`` is ATTACHED shadow (self-shadow by
    orientation), a shadow-map hit on a pixel with ``ndotl > 0`` is CAST
    shadow.  ``shadow = attached OR cast``.

EVIDENCE.  Nothing in this module is a measurement of a real sevengill.  The
renderer is a geometric instrument; the only physical claims live in
``nuisance.py`` (water attenuation) and ``exclusions.py`` (anatomy), and are
graded there.  [EVIDENCE GRADE: n/a -- deterministic geometry.]
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "Camera",
    "DirectionalLight",
    "Instance",
    "Fragments",
    "rasterize",
    "render",
    "shadow_map",
    "sample_texture",
    "sample_chart_mask",
    "vertex_normals",
    "look_at_basis",
    "transform_instance",
    "resolve_exclusion_chart",
    "chart_span_mask",
    "mask_bbox_ltwh",
    "OUTPUT_KEYS",
    "DEFAULT_SHADOW_MAP_SIZE",
    "SHADOW_NORMAL_BIAS_TEXELS",
    "SHADOW_DEPTH_BIAS_TEXELS",
]

# ---------------------------------------------------------------------------
# Shadow-map bias constants.
#
# Both are in SHADOW-MAP TEXELS of world size (the light camera's ortho height
# divided by the map resolution), so they scale automatically with scene size
# and map resolution -- the two things that actually set the quantisation
# error of a shadow map.
#
# They exist to kill self-shadow acne: a convex body lit from any direction
# must come out with an EMPTY cast-shadow set (that is a test in
# tests/test_render.py), and a naive depth comparison fails that at grazing
# angles because the receiver and the caster are the same surface.  The
# normal offset moves the query point off the surface before projection,
# which is the standard fix and is robust where a pure depth bias is not.
# Measured on the fixture tube: with (2.0, 3.0) the false cast-shadow count is
# 0 px at 256^2 and 1024^2 map sizes; at (0.0, 0.0) it is thousands.
# [EVIDENCE GRADE: derived -- a numerical property of this rasteriser, not of
# any shark.]
# ---------------------------------------------------------------------------
DEFAULT_SHADOW_MAP_SIZE = 1024
SHADOW_NORMAL_BIAS_TEXELS = 2.0
SHADOW_DEPTH_BIAS_TEXELS = 3.0

#: Every key :func:`render` puts in its output dict (plus ``"meta"``).
OUTPUT_KEYS = (
    "rgb", "depth", "instance", "face", "normal", "ndotl",
    "chart_s", "chart_phi",
    "coverage", "visible_skin", "occlusion",
    "attached_shadow", "cast_shadow", "shadow",
    "exclusion", "identity",
)

_EPS = 1e-12


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, _EPS)


def look_at_basis(eye, target, up=(0.0, 0.0, 1.0), roll_deg=0.0):
    """Orthonormal camera basis ``(right, up, forward)``.

    ``forward`` points from ``eye`` to ``target``; ``right`` is
    ``forward x up_hint`` normalised, so with the default dorsal-up hint a
    camera on the animal's left sees the head to the viewer's... whichever way
    the animal is facing -- this function makes no anatomical claim, it is
    plain Gram-Schmidt.  ``roll_deg`` rotates ``right``/``up`` about
    ``forward``; positive roll turns the IMAGE CONTENT clockwise (measured in
    tests/test_render.py, so the sign is not a guess).
    """
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    fwd = _unit(target - eye)
    up_hint = _unit(np.asarray(up, dtype=np.float64))
    if abs(float(np.dot(fwd, up_hint))) > 1.0 - 1e-6:
        # degenerate hint (looking straight along it): pick another axis
        up_hint = np.array([1.0, 0.0, 0.0]) if abs(fwd[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    right = _unit(np.cross(fwd, up_hint))
    upv = np.cross(right, fwd)
    if roll_deg:
        a = math.radians(float(roll_deg))
        ca, sa = math.cos(a), math.sin(a)
        right, upv = ca * right + sa * upv, -sa * right + ca * upv
    return right, upv, fwd


class Camera(object):
    """Orthographic or pinhole camera with an integer pixel raster.

    Args:
        eye: ``(3,)`` camera centre in world coordinates.
        target: ``(3,)`` look-at point.
        up: dorsal hint; the roll is resolved from it (default ``+Z``).
        resolution: ``(H, W)`` in pixels.
        kind: ``"ortho"`` or ``"pinhole"``.
        ortho_height: world height of the view volume (``kind="ortho"``).
        fov_y_deg: vertical field of view (``kind="pinhole"``).
        roll_deg: camera roll about the view axis, degrees.
        near: pinhole near plane; triangles with any vertex at ``z <= near``
            are DROPPED, not clipped.  Keep the subject well in front of the
            eye; this is a documented limitation, not a bug to work around by
            moving the near plane to 0 (that divides by zero).

    ``project`` returns pixel coordinates and camera-space depth for arbitrary
    points, which is what the chart ground-truth test uses to find the pixel a
    known vertex lands in.
    """

    def __init__(self, eye, target, up=(0.0, 0.0, 1.0), resolution=(256, 256),
                 kind="ortho", ortho_height=1.0, fov_y_deg=40.0, roll_deg=0.0,
                 near=1e-3):
        if kind not in ("ortho", "pinhole"):
            raise ValueError("kind must be 'ortho' or 'pinhole', got %r" % (kind,))
        h, w = int(resolution[0]), int(resolution[1])
        if h < 1 or w < 1:
            raise ValueError("resolution must be positive, got %r" % (resolution,))
        self.eye = np.asarray(eye, dtype=np.float64).reshape(3)
        self.target = np.asarray(target, dtype=np.float64).reshape(3)
        self.up_hint = np.asarray(up, dtype=np.float64).reshape(3)
        self.resolution = (h, w)
        self.kind = kind
        self.ortho_height = float(ortho_height)
        self.fov_y_deg = float(fov_y_deg)
        self.roll_deg = float(roll_deg)
        self.near = float(near)
        self.right, self.up, self.forward = look_at_basis(
            self.eye, self.target, self.up_hint, self.roll_deg)

    # -- introspection ------------------------------------------------------
    @property
    def height(self):
        return self.resolution[0]

    @property
    def width(self):
        return self.resolution[1]

    @property
    def aspect(self):
        return self.resolution[1] / float(self.resolution[0])

    def replace(self, **kw):
        """A copy with fields overridden (seeded jitter uses this)."""
        base = dict(eye=self.eye, target=self.target, up=self.up_hint,
                    resolution=self.resolution, kind=self.kind,
                    ortho_height=self.ortho_height, fov_y_deg=self.fov_y_deg,
                    roll_deg=self.roll_deg, near=self.near)
        base.update(kw)
        return Camera(**base)

    # -- transforms ---------------------------------------------------------
    def world_to_camera(self, points):
        """``(..., 3)`` world -> camera coordinates (x right, y up, z forward)."""
        p = np.asarray(points, dtype=np.float64)
        d = p - self.eye
        return np.stack([d @ self.right, d @ self.up, d @ self.forward], axis=-1)

    def project(self, points):
        """World points -> ``(px, py, depth)``.

        ``px``/``py`` are pixel coordinates (centres at integers, ``y`` down)
        and ``depth`` is camera-space ``z``.  For a pinhole camera points with
        ``z <= near`` come back as NaN pixels; their depth is still returned.
        """
        cam = self.world_to_camera(points)
        x, y, z = cam[..., 0], cam[..., 1], cam[..., 2]
        h, w = self.resolution
        if self.kind == "ortho":
            half_h = 0.5 * self.ortho_height
            half_w = half_h * self.aspect
            ndc_x = x / half_w
            ndc_y = y / half_h
        else:
            tan_y = math.tan(math.radians(0.5 * self.fov_y_deg))
            tan_x = tan_y * self.aspect
            zz = np.where(z > self.near, z, np.nan)
            ndc_x = x / (zz * tan_x)
            ndc_y = y / (zz * tan_y)
        px = (ndc_x * 0.5 + 0.5) * w - 0.5
        py = (0.5 - ndc_y * 0.5) * h - 0.5
        return px, py, z

    @classmethod
    def fit_ortho(cls, points, direction=(0.0, -1.0, 0.0), up=(0.0, 0.0, 1.0),
                  resolution=(256, 256), margin=1.15, distance=None, **kw):
        """An orthographic camera framing ``points`` from ``direction``.

        ``direction`` is the direction the camera LOOKS (so ``(0,-1,0)`` puts
        the eye on the animal's left, ``+Y``, looking toward ``-Y``).  The
        ortho height is the fitted extent times ``margin``.

        ``distance`` (default ``2 * bounding-sphere radius``) does NOT change
        an orthographic projection -- but it DOES set the depth values, and
        ``nuisance.apply_turbidity`` turns depth into fog.  So for anything
        involving water, pass the real camera-to-subject range in world units;
        leaving the default and then complaining that the animal is invisible
        at 3-6 m visibility is a scene-setup error, not a renderer bug.
        """
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        centre = 0.5 * (pts.min(axis=0) + pts.max(axis=0))
        radius = float(np.linalg.norm(pts - centre, axis=1).max())
        d = _unit(np.asarray(direction, dtype=np.float64))
        dist = (2.0 * radius) if distance is None else float(distance)
        cam = cls(eye=centre - d * dist, target=centre, up=up,
                  resolution=resolution, kind="ortho", ortho_height=1.0, **kw)
        # Fit the height to the projected extent, honouring roll.
        loc = cam.world_to_camera(pts)
        half_h = float(np.abs(loc[:, 1]).max())
        half_w = float(np.abs(loc[:, 0]).max())
        cam.ortho_height = 2.0 * margin * max(half_h, half_w / cam.aspect, 1e-6)
        return cam

    def __repr__(self):
        return ("Camera(kind=%r, res=%dx%d, eye=%s, target=%s)"
                % (self.kind, self.resolution[1], self.resolution[0],
                   np.round(self.eye, 4).tolist(), np.round(self.target, 4).tolist()))


class DirectionalLight(object):
    """A distant light plus the ambient term.

    ``direction`` is where the light GOES (a sun vector).  ``L = -direction``
    is what a shading equation wants.  ``ambient`` is a flat fill so that
    attached-shadow pixels are dark but not black -- underwater there is
    always veiling light, and a zero-ambient render would make the identity
    mask look better than the physics allows.
    """

    def __init__(self, direction=(0.3, 0.2, -1.0), color=(1.0, 1.0, 1.0),
                 intensity=1.0, ambient=0.25):
        self.direction = _unit(np.asarray(direction, dtype=np.float64).reshape(3))
        self.color = np.asarray(color, dtype=np.float64).reshape(3)
        self.intensity = float(intensity)
        self.ambient = float(ambient)

    @property
    def L(self):
        """Unit vector from a surface point toward the light."""
        return -self.direction

    def __repr__(self):
        return ("DirectionalLight(direction=%s, intensity=%.3f, ambient=%.3f)"
                % (np.round(self.direction, 4).tolist(), self.intensity, self.ambient))


def vertex_normals(vertices, faces):
    """Area-weighted per-vertex normals, ``(V, 3)`` unit (zero-area safe)."""
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])  # |fn| = 2*area
    out = np.zeros_like(v)
    for k in range(3):
        np.add.at(out, f[:, k], fn)
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return np.where(n > _EPS, out / np.maximum(n, _EPS), np.array([0.0, 0.0, 1.0]))


class Instance(object):
    """One renderable mesh: geometry, albedo, chart coordinates, role.

    Args:
        vertices: ``(V, 3)``.
        faces: ``(F, 3)`` triangle indices.
        uv: ``(V, 2)`` in ``[0, 1]^2``; required if ``texture`` is given.
        texture: ``(Ht, Wt, 3)`` or ``(Ht, Wt, 4)`` float albedo in [0, 1].
            Alpha, if present, is IGNORED for coverage (this rasteriser is
            opaque); it is dropped so a bake's coverage-alpha cannot be
            mistaken for transparency.
        color: ``(3,)`` flat albedo used where there is no texture.
        vertex_s, vertex_phi: ``(V,)`` canonical chart coordinates.  Optional
            (an occluder has none); a SUBJECT without them can still render
            but its chart maps are NaN and its identity mask is empty.
        role: ``"subject"`` -- its skin is the identity surface -- or
            ``"occluder"``: anything in front of the subject, whose pixels
            become the occlusion mask.  A second shark used as a foreground
            occluder is an ``"occluder"``, even though it is a shark.
        casts_shadow: contributes to the shadow map.
        receives_shadow: is tested against the shadow map.
        double_sided: shade with the normal flipped toward the camera and skip
            back-face culling.  Kelp ribbons are open, one-quad-thick strips
            and MUST be double sided or half of every blade shades black.
        name: free-form label carried into ``meta``.

    Nothing here is transformed: vertices are already in world space.  Use
    :func:`transform_instance` to place a copy.
    """

    def __init__(self, vertices, faces, uv=None, texture=None,
                 color=(0.55, 0.55, 0.55), vertex_s=None, vertex_phi=None,
                 role="subject", casts_shadow=True, receives_shadow=True,
                 double_sided=False, name=None):
        self.vertices = np.ascontiguousarray(np.asarray(vertices, dtype=np.float64).reshape(-1, 3))
        self.faces = np.ascontiguousarray(np.asarray(faces, dtype=np.int64).reshape(-1, 3))
        nv = len(self.vertices)
        if self.faces.size and (self.faces.min() < 0 or self.faces.max() >= nv):
            raise ValueError("face index out of range for %d vertices" % nv)
        if role not in ("subject", "occluder"):
            raise ValueError("role must be 'subject' or 'occluder', got %r" % (role,))
        self.uv = None if uv is None else np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        if self.uv is not None and len(self.uv) != nv:
            raise ValueError("uv has %d rows, need %d" % (len(self.uv), nv))
        if texture is None:
            self.texture = None
        else:
            tex = np.asarray(texture, dtype=np.float64)
            if tex.ndim != 3 or tex.shape[2] not in (3, 4):
                raise ValueError("texture must be (H, W, 3|4), got %r" % (tex.shape,))
            self.texture = np.ascontiguousarray(tex[:, :, :3])
            if self.uv is None:
                raise ValueError("a textured instance needs uv")
        self.color = np.asarray(color, dtype=np.float64).reshape(3)
        self.vertex_s = None if vertex_s is None else np.asarray(vertex_s, dtype=np.float64).reshape(-1)
        self.vertex_phi = None if vertex_phi is None else np.asarray(vertex_phi, dtype=np.float64).reshape(-1)
        for nm, arr in (("vertex_s", self.vertex_s), ("vertex_phi", self.vertex_phi)):
            if arr is not None and len(arr) != nv:
                raise ValueError("%s has %d entries, need %d" % (nm, len(arr), nv))
        self.role = role
        self.casts_shadow = bool(casts_shadow)
        self.receives_shadow = bool(receives_shadow)
        self.double_sided = bool(double_sided)
        self.name = name
        self._normals = None

    @property
    def normals(self):
        """Cached area-weighted vertex normals."""
        if self._normals is None:
            self._normals = vertex_normals(self.vertices, self.faces)
        return self._normals

    @property
    def has_chart(self):
        return self.vertex_s is not None and self.vertex_phi is not None

    @classmethod
    def from_uv_tube(cls, tube, texture=None, **kw):
        """Build a subject from ``fixtures.make_uv_tube``'s :class:`UVTube`.

        This is the same call shape prototype 04's meshes will use; the tube
        fixture exists so the contract can be tested without it.
        """
        mesh = tube.mesh
        uv = kw.pop("uv", None)
        if uv is None:
            uv = np.asarray(mesh.visual.uv) if texture is not None else None
        return cls(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.faces),
                   uv=uv, texture=texture, vertex_s=tube.vertex_s,
                   vertex_phi=tube.vertex_phi, **kw)

    def __repr__(self):
        return ("Instance(name=%r, role=%r, V=%d, F=%d, chart=%s)"
                % (self.name, self.role, len(self.vertices), len(self.faces),
                   self.has_chart))


def transform_instance(inst, rotation=None, translation=None, scale=1.0,
                       **overrides):
    """A rigid(+uniform-scale) copy of ``inst``.

    Chart coordinates, UV and texture are carried over unchanged -- they are
    intrinsic to the surface, which is exactly why a second shark can be
    placed as a foreground occluder with one call.  ``rotation`` is a ``(3,3)``
    matrix applied about the instance's own centroid before ``translation``.
    """
    v = inst.vertices
    centre = v.mean(axis=0)
    out = (v - centre) * float(scale)
    if rotation is not None:
        out = out @ np.asarray(rotation, dtype=np.float64).T
    out = out + centre
    if translation is not None:
        out = out + np.asarray(translation, dtype=np.float64).reshape(3)
    kw = dict(uv=inst.uv, texture=inst.texture, color=inst.color,
              vertex_s=inst.vertex_s, vertex_phi=inst.vertex_phi,
              role=inst.role, casts_shadow=inst.casts_shadow,
              receives_shadow=inst.receives_shadow,
              double_sided=inst.double_sided, name=inst.name)
    kw.update(overrides)
    return Instance(vertices=out, faces=inst.faces, **kw)


# ---------------------------------------------------------------------------
# Rasteriser
# ---------------------------------------------------------------------------

class Fragments(object):
    """Per-pixel winning-fragment buffers from :func:`rasterize`.

    Attributes:
        instance: ``(H, W)`` int32 index into the instance list, ``-1`` = none.
        face: ``(H, W)`` int32 face index within that instance, ``-1`` = none.
        bary: ``(H, W, 3)`` PERSPECTIVE-CORRECT barycentric weights.
        depth: ``(H, W)`` camera-space z, ``+inf`` where nothing was hit.
    """

    __slots__ = ("instance", "face", "bary", "depth", "resolution")

    def __init__(self, instance, face, bary, depth):
        self.instance = instance
        self.face = face
        self.bary = bary
        self.depth = depth
        self.resolution = instance.shape

    @property
    def coverage(self):
        return self.instance >= 0

    def interpolate(self, instances, attribute):
        """Interpolate a per-vertex attribute over the covered pixels.

        ``attribute`` is a callable ``instance -> (V,)`` or ``(V, C)`` array,
        or ``None`` to skip that instance.  Returns ``(H, W)`` / ``(H, W, C)``
        filled with NaN where the attribute is unavailable or uncovered.
        """
        out = None
        for idx, inst in enumerate(instances):
            sel = self.instance == idx
            if not sel.any():
                continue
            attr = attribute(inst)
            if attr is None:
                continue
            attr = np.asarray(attr, dtype=np.float64)
            flat = attr.reshape(len(inst.vertices), -1)
            if out is None:
                out = np.full(self.resolution + (flat.shape[1],), np.nan)
            tri = inst.faces[self.face[sel]]                    # (n, 3)
            vals = flat[tri]                                    # (n, 3, C)
            out[sel] = np.einsum("nk,nkc->nc", self.bary[sel], vals)
        if out is None:
            return None
        return out[..., 0] if out.shape[2] == 1 else out


def _project_all(instances, camera):
    return [camera.project(inst.vertices) for inst in instances]


def rasterize(instances, camera, backface_cull=False):
    """Z-buffer rasterise ``instances`` through ``camera``.

    One pass, nearest fragment wins, ties broken by instance order (a later
    instance overwrites only if strictly nearer).  Barycentrics are
    perspective-correct for a pinhole camera and plain screen-space for an
    orthographic one (where they coincide).

    ``backface_cull`` drops triangles whose projected winding is clockwise.
    It is OFF by default because the fixture tube's winding is not guaranteed
    and a culled subject renders as an empty image, which is a confusing
    failure; double-sided instances are never culled.
    """
    h, w = camera.resolution
    inst_buf = np.full((h, w), -1, dtype=np.int32)
    face_buf = np.full((h, w), -1, dtype=np.int32)
    bary_buf = np.zeros((h, w, 3), dtype=np.float64)
    depth_buf = np.full((h, w), np.inf, dtype=np.float64)

    for i_idx, inst in enumerate(instances):
        px, py, pz = camera.project(inst.vertices)
        faces = inst.faces
        if not len(faces):
            continue
        fx, fy, fz = px[faces], py[faces], pz[faces]
        finite = np.isfinite(fx).all(axis=1) & np.isfinite(fy).all(axis=1)
        if camera.kind == "pinhole":
            finite &= (fz > camera.near).all(axis=1)
        # cheap whole-triangle frustum reject
        lo_x = np.floor(fx.min(axis=1)).astype(np.int64)
        hi_x = np.ceil(fx.max(axis=1)).astype(np.int64)
        lo_y = np.floor(fy.min(axis=1)).astype(np.int64)
        hi_y = np.ceil(fy.max(axis=1)).astype(np.int64)
        onscreen = (hi_x >= 0) & (lo_x < w) & (hi_y >= 0) & (lo_y < h)
        area2 = ((fx[:, 1] - fx[:, 0]) * (fy[:, 2] - fy[:, 0])
                 - (fx[:, 2] - fx[:, 0]) * (fy[:, 1] - fy[:, 0]))
        keep = finite & onscreen & (np.abs(area2) > 1e-12)
        if backface_cull and not inst.double_sided:
            keep &= area2 < 0.0     # y is down, so CCW in world = negative area
        idx = np.nonzero(keep)[0]

        perspective = camera.kind == "pinhole"
        for f in idx:
            x0, x1, x2 = fx[f]
            y0, y1, y2 = fy[f]
            z0, z1, z2 = fz[f]
            a2 = area2[f]
            xs0 = max(int(lo_x[f]), 0)
            xs1 = min(int(hi_x[f]) + 1, w)
            ys0 = max(int(lo_y[f]), 0)
            ys1 = min(int(hi_y[f]) + 1, h)
            if xs0 >= xs1 or ys0 >= ys1:
                continue
            X = np.arange(xs0, xs1, dtype=np.float64)[None, :]
            Y = np.arange(ys0, ys1, dtype=np.float64)[:, None]
            w0 = ((x1 - X) * (y2 - Y) - (x2 - X) * (y1 - Y)) / a2
            w1 = ((x2 - X) * (y0 - Y) - (x0 - X) * (y2 - Y)) / a2
            w2 = 1.0 - w0 - w1
            inside = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
            if not inside.any():
                continue
            if perspective:
                inv = w0 / z0 + w1 / z1 + w2 / z2
                with np.errstate(divide="ignore", invalid="ignore"):
                    z = 1.0 / inv
                b0, b1, b2 = w0 / z0 * z, w1 / z1 * z, w2 / z2 * z
            else:
                z = w0 * z0 + w1 * z1 + w2 * z2
                b0, b1, b2 = w0, w1, w2
            sub = depth_buf[ys0:ys1, xs0:xs1]
            win = inside & np.isfinite(z) & (z < sub)
            if not win.any():
                continue
            sub[win] = z[win]
            face_buf[ys0:ys1, xs0:xs1][win] = f
            inst_buf[ys0:ys1, xs0:xs1][win] = i_idx
            bb = bary_buf[ys0:ys1, xs0:xs1]
            bb[..., 0][win] = b0[win]
            bb[..., 1][win] = b1[win]
            bb[..., 2][win] = b2[win]

    return Fragments(inst_buf, face_buf, bary_buf, depth_buf)


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def sample_texture(texture, uv, fill=(0.5, 0.5, 0.5)):
    """Bilinear texture lookup with CLAMPED edges.

    ``uv`` is ``(..., 2)`` in ``[0, 1]^2`` with the same texel-centre
    convention as ``bake.rasterize_uv``: texel ``(y, x)`` has centre
    ``((x + 0.5) / W, (y + 0.5) / H)``.  NaN uv gives ``fill``.
    """
    tex = np.asarray(texture, dtype=np.float64)
    ht, wt = tex.shape[0], tex.shape[1]
    uv = np.asarray(uv, dtype=np.float64)
    bad = ~np.isfinite(uv).all(axis=-1)
    u = np.nan_to_num(uv[..., 0], nan=0.0)
    v = np.nan_to_num(uv[..., 1], nan=0.0)
    x = np.clip(u * wt - 0.5, 0.0, wt - 1.0)
    y = np.clip(v * ht - 0.5, 0.0, ht - 1.0)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, wt - 1)
    y1 = np.minimum(y0 + 1, ht - 1)
    fx = (x - x0)[..., None]
    fy = (y - y0)[..., None]
    out = (tex[y0, x0] * (1 - fx) * (1 - fy) + tex[y0, x1] * fx * (1 - fy)
           + tex[y1, x0] * (1 - fx) * fy + tex[y1, x1] * fx * fy)
    if bad.any():
        out[bad] = np.asarray(fill, dtype=np.float64)
    return out


def sample_chart_mask(mask, s, phi, axis_order="phi_major"):
    """NEAREST-neighbour lookup of a boolean chart mask at ``(s, phi)``.

    Nearest, not bilinear: a boolean exclusion mask has no meaningful
    interpolant, and blurring it would either leak identity pixels into an eye
    or eat skin around it.  ``s`` clamps to the body ends, ``phi`` wraps.
    NaN inputs (off-body pixels) return ``False``.

    ``axis_order`` names the caller's layout:
      * ``"phi_major"`` -- ``(H_phi, W_s)``, the layout ``exclusions.py`` and
        ``pattern.render_chart`` produce (their ``resolution=(H_phi, W_s)``);
      * ``"s_major"`` -- ``(n_s, n_phi)``, the layout ``bake.sample_chart``
        wants.
    Getting this wrong silently transposes the animal, so it is explicit and
    has no default that works for both.
    """
    m = np.asarray(mask)
    if m.ndim != 2:
        raise ValueError("chart mask must be 2-D, got %r" % (m.shape,))
    if axis_order == "phi_major":
        n_phi, n_s = m.shape
    elif axis_order == "s_major":
        n_s, n_phi = m.shape
    else:
        raise ValueError("axis_order must be 'phi_major' or 's_major'")
    s = np.asarray(s, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    ok = np.isfinite(s) & np.isfinite(phi)
    si = np.clip(np.floor(np.nan_to_num(s, nan=0.0) * n_s).astype(np.int64), 0, n_s - 1)
    ph = (np.nan_to_num(phi, nan=0.0) + np.pi) % (2.0 * np.pi) - np.pi
    pj = np.floor((ph + np.pi) / (2.0 * np.pi) * n_phi).astype(np.int64) % n_phi
    vals = m[pj, si] if axis_order == "phi_major" else m[si, pj]
    return np.asarray(vals, dtype=bool) & ok


def resolve_exclusion_chart(exclusion, resolution=(128, 256)):
    """Normalise the ``exclusion`` argument of :func:`render`.

    Accepts:
      * ``None``  -- no exclusion (identity mask keeps eyes; only sensible for
        a geometry test);
      * ``"auto"`` -- build it LAZILY from ``exclusions.build_exclusion_mask``
        against the Schema S1 yaml.  If that import fails (module not present,
        yaml missing) this returns ``None`` and the caller carries on with no
        exclusion rather than failing the render -- module R must not be
        blocked on module P.  The reason is returned for ``meta``;
      * a 2-D boolean array in ``(H_phi, W_s)`` layout;
      * ``(array, axis_order)``.

    Returns ``(mask_or_None, axis_order, note)``.
    """
    if exclusion is None:
        return None, "phi_major", "none"
    if isinstance(exclusion, tuple) and len(exclusion) == 2 and not np.isscalar(exclusion[0]):
        arr, order = exclusion
        return np.asarray(arr, dtype=bool), str(order), "provided"
    if isinstance(exclusion, str):
        if exclusion != "auto":
            raise ValueError("exclusion string must be 'auto', got %r" % (exclusion,))
        try:
            import exclusions as _exc  # lazy: module P is a sibling, not a dependency
            mask = _exc.build_exclusion_mask(_exc.DEFAULT_SCHEMA_PATH
                                             if hasattr(_exc, "DEFAULT_SCHEMA_PATH")
                                             else _default_schema_path(),
                                             resolution=resolution)
            return np.asarray(mask, dtype=bool), "phi_major", "exclusions.build_exclusion_mask"
        except Exception as exc:                      # pragma: no cover - env dependent
            return None, "phi_major", "unavailable: %s: %s" % (type(exc).__name__, exc)
    return np.asarray(exclusion, dtype=bool), "phi_major", "provided"


def chart_span_mask(chart_s, s_lo, s_hi, within=None):
    """Body pixels whose chart ``s`` falls in ``[s_lo, s_hi)``.

    The bridge from chart ground truth to a CROP: the head/headless split that
    ``melops_data`` wants as ``bbox_head`` / ``bbox_headless`` is a split in
    ARC LENGTH, not in pixels, so it must be cut here -- through the chart --
    and not guessed from the silhouette.  Pass the station from Schema S1
    (e.g. the last gill slit) as the cut; this function invents no fraction.

    ``within`` (typically ``out["visible_skin"]``) restricts the result.
    """
    s = np.asarray(chart_s, dtype=np.float64)
    out = np.isfinite(s) & (s >= float(s_lo)) & (s < float(s_hi))
    if within is not None:
        out &= np.asarray(within, dtype=bool)
    return out


def mask_bbox_ltwh(mask, relative_to=None, pad=0.0):
    """Tight ``[left, top, width, height]`` float bbox of a boolean mask.

    This is the ``melops_data`` bbox convention exactly: LTWH floats, pixel
    edges (a single lit pixel at column 3 gives ``left = 3.0, width = 1.0``),
    and ``relative_to`` subtracts another LTWH's origin -- which is what
    ``load_melops`` needs, because it applies ``bbox_head`` INSIDE the body
    crop, not in the full frame.

    Returns ``None`` for an empty mask; the caller decides whether an empty
    crop is a dropped frame or an error.
    """
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return None
    ys, xs = np.nonzero(m)
    left = float(xs.min()) - float(pad)
    top = float(ys.min()) - float(pad)
    w = float(xs.max() + 1) - float(xs.min()) + 2.0 * float(pad)
    h = float(ys.max() + 1) - float(ys.min()) + 2.0 * float(pad)
    if relative_to is not None:
        left -= float(relative_to[0])
        top -= float(relative_to[1])
    return [left, top, w, h]


def _default_schema_path():
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(
        here, "..", "..", "phase1b", "p0-sevengill-schema",
        "keypoints_sevengill_v1.yaml"))


# ---------------------------------------------------------------------------
# Shadow map
# ---------------------------------------------------------------------------

def shadow_map(instances, light, size=DEFAULT_SHADOW_MAP_SIZE, margin=1.05):
    """Depth buffer of the shadow CASTERS, seen from the light.

    Returns ``(camera, depth)``.  The camera is orthographic (the light is
    directional) and fitted to the casters' bounding sphere, so its texel
    world size -- the quantity the bias constants are expressed in -- is
    ``camera.ortho_height / size``.  Returns ``(None, None)`` when nothing
    casts.
    """
    casters = [i for i in instances if i.casts_shadow and len(i.faces)]
    if not casters:
        return None, None
    pts = np.concatenate([i.vertices for i in casters], axis=0)
    centre = 0.5 * (pts.min(axis=0) + pts.max(axis=0))
    radius = max(float(np.linalg.norm(pts - centre, axis=1).max()), 1e-6)
    d = light.direction
    up = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    cam = Camera(eye=centre - d * (2.0 * radius + 1.0), target=centre, up=up,
                 resolution=(int(size), int(size)), kind="ortho",
                 ortho_height=2.0 * radius * margin)
    frags = rasterize(casters, cam)
    return cam, frags.depth


def _shadow_lookup(cam, depth, world, normal, ndotl,
                   normal_bias=SHADOW_NORMAL_BIAS_TEXELS,
                   depth_bias=SHADOW_DEPTH_BIAS_TEXELS):
    """True where ``world`` is occluded from the light.  See the bias note."""
    texel = cam.ortho_height / float(cam.resolution[0])
    # Normal offset grows as the surface turns away from the light, which is
    # exactly where the depth quantisation error is worst.
    slope = np.sqrt(np.clip(1.0 - ndotl ** 2, 0.0, 1.0)) / np.maximum(np.abs(ndotl), 0.15)
    offset = normal * (texel * normal_bias * (1.0 + slope))[..., None]
    px, py, pz = cam.project(world + offset)
    h, w = cam.resolution
    xi = np.rint(px).astype(np.int64)
    yi = np.rint(py).astype(np.int64)
    inside = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h) & np.isfinite(pz)
    xi = np.clip(xi, 0, w - 1)
    yi = np.clip(yi, 0, h - 1)
    ref = depth[yi, xi]
    bias = texel * depth_bias * (1.0 + slope)
    return inside & np.isfinite(ref) & (ref < pz - bias)


# ---------------------------------------------------------------------------
# The renderer
# ---------------------------------------------------------------------------

#: Default water-coloured background.  A black background would make the
#: silhouette trivially separable, which is not the sevengill problem: at
#: 3-6 m visibility in kelp water the animal is a low-contrast shape against
#: veiling light (docs/sevengill-canonical-reid/01-evidence-and-answers.md,
#: La Jolla encounter conditions row) [SECONDARY -- dive-operator copy].
BACKGROUND_RGB = (0.075, 0.155, 0.150)


def render(instances, camera, light=None, exclusion="auto",
           exclusion_resolution=(128, 256), background=BACKGROUND_RGB,
           shadows=True, shadow_map_size=DEFAULT_SHADOW_MAP_SIZE,
           shadow_normal_bias=SHADOW_NORMAL_BIAS_TEXELS,
           shadow_depth_bias=SHADOW_DEPTH_BIAS_TEXELS,
           backface_cull=False, ambient=None):
    """Render one frame plus every aligned ground-truth map.

    Args:
        instances: list of :class:`Instance`, already posed in world space.
        camera: :class:`Camera`.
        light: :class:`DirectionalLight`; ``None`` gives a default overhead-ish
            key light.
        exclusion: chart-space exclusion mask -- see
            :func:`resolve_exclusion_chart`.  ``"auto"`` (default) pulls it
            from ``exclusions.py`` LAZILY and degrades to no-exclusion if that
            module is unavailable, so this module never hard-depends on it.
        exclusion_resolution: ``(H_phi, W_s)`` used only when building
            ``"auto"``.
        background: RGB behind everything.
        shadows: compute the cast-shadow pass at all.  With ``False`` the
            shadow mask is the attached set only.
        ambient: override ``light.ambient``.

    Returns:
        dict with the keys in :data:`OUTPUT_KEYS` plus ``"meta"``:

        ==================  ===========================================
        ``rgb``             ``(H, W, 3)`` float in [0, 1]
        ``depth``           ``(H, W)`` camera-space z, ``+inf`` = background
        ``instance``        ``(H, W)`` int32 index, ``-1`` = background
        ``face``            ``(H, W)`` int32 face index, ``-1`` = background
        ``normal``          ``(H, W, 3)`` unit world normal, NaN off-surface
        ``ndotl``           ``(H, W)`` ``dot(N, L)``, NaN off-surface
        ``chart_s``         ``(H, W)`` arc-length fraction, NaN off-body
        ``chart_phi``       ``(H, W)`` circumferential angle, NaN off-body
        ``coverage``        any geometry
        ``visible_skin``    a SUBJECT is the front-most surface here
        ``occlusion``       subject surface exists but something is in front
        ``attached_shadow`` ``ndotl <= 0`` on covered pixels
        ``cast_shadow``     lit-facing but blocked from the light
        ``shadow``          attached OR cast
        ``exclusion``       chart-space exclusion, sampled through chart GT
        ``identity``        the identity mask (see the module docstring)
        ==================  ===========================================

    All masks are ``bool`` and pixel-aligned with ``rgb``.  ``identity`` is by
    construction a subset of ``visible_skin`` and disjoint from ``shadow``,
    ``occlusion`` and ``exclusion``.
    """
    instances = list(instances)
    if light is None:
        light = DirectionalLight()
    amb = light.ambient if ambient is None else float(ambient)
    h, w = camera.resolution

    frags = rasterize(instances, camera, backface_cull=backface_cull)
    coverage = frags.coverage

    subject_idx = [k for k, i in enumerate(instances) if i.role == "subject"]
    subjects = [instances[k] for k in subject_idx]
    if not subjects:
        subject_any = np.zeros((h, w), dtype=bool)
    elif len(subjects) == len(instances):
        # Nothing can occlude a subject except another subject, and that case
        # is not occlusion (see below), so the second pass is redundant.
        subject_any = coverage
    else:
        subject_any = rasterize(subjects, camera,
                                backface_cull=backface_cull).coverage

    # One entry per instance PLUS a padding slot: indexing this by
    # frags.instance (-1 for background) then lands on the padding slot, which
    # is False, so background is never mistaken for a subject.
    is_subject = np.zeros(len(instances) + 1, dtype=bool)
    for k in subject_idx:
        is_subject[k] = True
    visible_skin = coverage & is_subject[frags.instance]
    # A subject surface is here, but it is not what the camera sees.
    occlusion = subject_any & ~visible_skin

    # -- interpolated per-pixel geometry ------------------------------------
    world = frags.interpolate(instances, lambda i: i.vertices)
    if world is None:
        world = np.full((h, w, 3), np.nan)
    normal = frags.interpolate(instances, lambda i: i.normals)
    if normal is None:
        normal = np.full((h, w, 3), np.nan)
    nlen = np.linalg.norm(normal, axis=-1, keepdims=True)
    normal = np.where(nlen > _EPS, normal / np.maximum(nlen, _EPS), np.nan)

    # Double-sided geometry (kelp ribbons) is shaded with the normal turned
    # toward the camera; otherwise half of every blade is lit from behind.
    two_sided = np.zeros(len(instances) + 1, dtype=bool)   # padded, see above
    for k, inst in enumerate(instances):
        two_sided[k] = inst.double_sided
    if two_sided[frags.instance].any():
        if camera.kind == "ortho":
            view = np.broadcast_to(-camera.forward, normal.shape)
        else:
            view = _unit(camera.eye - world)
        flip = two_sided[frags.instance] & (np.nansum(normal * view, axis=-1) < 0.0)
        normal = np.where(flip[..., None], -normal, normal)

    # -- chart ground truth (subjects only; phi interpolated on the circle) --
    chart_s = frags.interpolate(
        instances, lambda i: i.vertex_s if (i.role == "subject" and i.has_chart) else None)
    if chart_s is None:
        chart_s = np.full((h, w), np.nan)
    cs = frags.interpolate(
        instances,
        lambda i: (np.column_stack([np.cos(i.vertex_phi), np.sin(i.vertex_phi)])
                   if (i.role == "subject" and i.has_chart) else None))
    if cs is None:
        chart_phi = np.full((h, w), np.nan)
    else:
        chart_phi = np.arctan2(cs[..., 1], cs[..., 0])
    has_chart = np.isfinite(chart_s) & np.isfinite(chart_phi)

    # -- albedo -------------------------------------------------------------
    albedo = np.zeros((h, w, 3), dtype=np.float64)
    albedo[...] = np.asarray(background, dtype=np.float64)
    for idx, inst in enumerate(instances):
        sel = frags.instance == idx
        if not sel.any():
            continue
        if inst.texture is not None:
            tri = inst.faces[frags.face[sel]]
            uv = np.einsum("nk,nkc->nc", frags.bary[sel], inst.uv[tri])
            albedo[sel] = sample_texture(inst.texture, uv)
        else:
            albedo[sel] = inst.color

    # -- shading + shadows --------------------------------------------------
    L = light.L
    ndotl = np.where(coverage, np.nansum(normal * L, axis=-1), np.nan)
    attached = coverage & (ndotl <= 0.0)

    cast = np.zeros((h, w), dtype=bool)
    shadow_note = "off"
    if shadows:
        scam, sdepth = shadow_map(instances, light, size=shadow_map_size)
        if scam is None:
            shadow_note = "no casters"
        else:
            receives = np.zeros(len(instances) + 1, dtype=bool)   # padded
            for k, inst in enumerate(instances):
                receives[k] = inst.receives_shadow
            lit = coverage & (ndotl > 0.0) & receives[frags.instance]
            if lit.any():
                hit = _shadow_lookup(
                    scam, sdepth, world[lit], normal[lit], ndotl[lit],
                    normal_bias=shadow_normal_bias, depth_bias=shadow_depth_bias)
                cast[lit] = hit
            shadow_note = "map %dx%d, texel %.5g" % (
                scam.resolution[0], scam.resolution[1],
                scam.ortho_height / scam.resolution[0])
    shadow = attached | cast

    lam = np.clip(np.nan_to_num(ndotl, nan=0.0), 0.0, 1.0) * (~shadow)
    shade = amb + light.intensity * lam
    rgb = np.clip(albedo * shade[..., None] * light.color, 0.0, 1.0)
    rgb[~coverage] = np.asarray(background, dtype=np.float64)

    # -- exclusion + identity ----------------------------------------------
    exc_mask, exc_order, exc_note = resolve_exclusion_chart(
        exclusion, resolution=exclusion_resolution)
    if exc_mask is None:
        exclusion_px = np.zeros((h, w), dtype=bool)
    else:
        exclusion_px = np.zeros((h, w), dtype=bool)
        sel = visible_skin & has_chart
        if sel.any():
            exclusion_px[sel] = sample_chart_mask(
                exc_mask, chart_s[sel], chart_phi[sel], axis_order=exc_order)

    identity = visible_skin & has_chart & ~exclusion_px & ~shadow & ~occlusion

    return {
        "rgb": rgb,
        "depth": frags.depth,
        "instance": frags.instance,
        "face": frags.face,
        "normal": normal,
        "ndotl": ndotl,
        "chart_s": chart_s,
        "chart_phi": chart_phi,
        "coverage": coverage,
        "visible_skin": visible_skin,
        "occlusion": occlusion,
        "attached_shadow": attached,
        "cast_shadow": cast,
        "shadow": shadow,
        "exclusion": exclusion_px,
        "identity": identity,
        "meta": {
            "camera": repr(camera),
            "light": repr(light),
            "resolution": (h, w),
            "n_instances": len(instances),
            "subject_indices": subject_idx,
            "exclusion_source": exc_note,
            "exclusion_axis_order": exc_order,
            "shadow": shadow_note,
            "background": tuple(float(c) for c in np.asarray(background).ravel()),
            "ambient": amb,
        },
    }
