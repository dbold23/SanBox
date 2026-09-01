"""Scene fixtures for module R (render.py / nuisance.py).

Self-contained: a textured UV tube with per-vertex ``(s, phi)``, a procedural
texture, a light and a camera.  Nothing here is a measurement; it exists so
the renderer's contracts can be tested against exact answers.

The geometry comes from ``fixtures.make_uv_tube``, which is the module that
stands in for prototype 04's ``mesh3d.tube_coords``.
"""

from __future__ import annotations

import numpy as np

import fixtures
import render

#: World-unit length of the fixture animal.  A broadnose sevengill is roughly
#: 1.5-3 m, so "1 world unit = 1 m" keeps the turbidity numbers honest.
BODY_LENGTH = 2.0

#: Camera-to-subject range for the standard scene, in metres.  Inside the
#: 3-6 m typical La Jolla visibility bracket, so the fog is severe but the
#: animal is still there.
SUBJECT_RANGE_M = 2.5


def procedural_texture(size=(96, 192), seed=0, base=(0.46, 0.43, 0.37),
                       n_spots=140, spot_value=0.12, spot_px=4.0):
    """A grey-brown skin texture with dark round speckles, in UV space.

    Deliberately NOT the identity pattern generator (that is ``pattern.py``,
    in chart space): this is a texture with high-frequency content so that
    contrast, blur and fog tests have something to measure.
    """
    h, w = int(size[0]), int(size[1])
    rng = np.random.default_rng(int(seed))
    tex = np.empty((h, w, 3), dtype=np.float64)
    tex[...] = np.asarray(base, dtype=np.float64)
    ys = rng.uniform(0, h, size=n_spots)
    xs = rng.uniform(0, w, size=n_spots)
    Y, X = np.mgrid[0:h, 0:w].astype(np.float64)
    for cy, cx in zip(ys, xs):
        d2 = (Y - cy) ** 2 + np.minimum(np.abs(X - cx), w - np.abs(X - cx)) ** 2
        tex[d2 <= spot_px ** 2] = spot_value
    return tex


def subject(n_stations=48, n_around=32, seed=0, textured=True, bend=0.0,
            length=BODY_LENGTH, r_max=0.16, **kw):
    """The fixture shark: a textured UV tube carrying exact ``(s, phi)``."""
    tube = fixtures.make_uv_tube(n_stations=n_stations, n_around=n_around,
                                 length=length, r_max=r_max, bend=bend)
    tex = procedural_texture(seed=seed) if textured else None
    inst = render.Instance.from_uv_tube(tube, texture=tex, name="subject", **kw)
    return tube, inst


def side_camera(inst, resolution=(256, 256), side="left", kind="ortho",
                distance=SUBJECT_RANGE_M, **kw):
    """A camera looking at the animal's LEFT (+Y) or RIGHT (-Y) flank.

    Schema S1's side convention is about which flank is visible, and the flank
    is the identity surface, so every fixture camera is a flank camera.
    """
    direction = (0.0, -1.0, 0.0) if side == "left" else (0.0, 1.0, 0.0)
    cam = render.Camera.fit_ortho(inst.vertices, direction=direction,
                                  resolution=resolution, distance=distance, **kw)
    if kind == "ortho":
        return cam
    # Same framing, pinhole: choose the fov that subtends the ortho height.
    fov = 2.0 * np.degrees(np.arctan(0.5 * cam.ortho_height / distance))
    return render.Camera(eye=cam.eye, target=cam.target, resolution=resolution,
                         kind="pinhole", fov_y_deg=float(fov))


def key_light(direction=(0.25, 0.35, -1.0), ambient=0.22, intensity=1.0):
    """Sun from above and slightly behind the camera's shoulder."""
    return render.DirectionalLight(direction=direction, ambient=ambient,
                                   intensity=intensity)


def light_blocker(inst, light, offset=0.55, half=0.35, name="blocker"):
    """A flat quad placed between ``light`` and the animal.

    It is a shadow CASTER but sits out of the camera's way, so it produces a
    cast shadow without producing any occlusion -- which is what separates the
    two masks in the tests.
    """
    centre = inst.vertices.mean(axis=0)
    d = light.direction
    pos = centre - d * float(offset)
    a = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(a, d))) > 0.9:
        a = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(d, a)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(d, e1)
    e2 /= np.linalg.norm(e2)
    v = np.stack([pos - half * e1 - half * e2, pos + half * e1 - half * e2,
                  pos + half * e1 + half * e2, pos - half * e1 + half * e2])
    f = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    return render.Instance(vertices=v, faces=f, color=(0.2, 0.2, 0.2),
                           role="occluder", double_sided=True, name=name)


def front_facing_vertices(inst, camera, min_ndotv=0.5, s_range=(0.1, 0.9)):
    """Indices of vertices that are a FAIR chart-GT comparison set.

    Two filters, both geometric, neither hiding a renderer error:

    * ``min_ndotv`` -- a back-facing vertex projects into a pixel owned by the
      FRONT surface, so its ``phi`` legitimately disagrees with the chart map
      there.  Near the silhouette the same thing happens to within a pixel,
      because ``dx/dphi = R cos(theta)`` collapses at the limb.
    * ``s_range`` -- at the body ends the fixture tube's radius drops to its
      6% floor, so the whole ``2 pi`` of ``phi`` projects into about a pixel
      and a half.  No rasteriser can resolve ``phi`` there; that is a property
      of the fixture geometry, not of the renderer.
    """
    if camera.kind == "ortho":
        view = np.broadcast_to(-camera.forward, inst.vertices.shape)
    else:
        view = camera.eye - inst.vertices
        view = view / np.linalg.norm(view, axis=1, keepdims=True)
    ndotv = (inst.normals * view).sum(axis=1)
    keep = ndotv > float(min_ndotv)
    if s_range is not None and inst.vertex_s is not None:
        keep &= (inst.vertex_s > s_range[0]) & (inst.vertex_s < s_range[1])
    return np.nonzero(keep)[0]
