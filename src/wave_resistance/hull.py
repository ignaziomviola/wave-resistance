"""Hull geometry: tessellation, mirroring, prescribed attitude, exact plane clipping.

The hull is carried as a triangulated surface with consistently outward normals.  No
downstream code assumes port/starboard symmetry: a half model is mirrored once, here,
and the fact that it was mirrored is recorded as provenance.

Clipping against a plane is exact on the piecewise-linear surface, so the waterline is
resolved to the same order as the tessellation itself and there is no staircase error.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from . import nurbs
from .iges import IgesFile, NurbsSurface, read_iges

__all__ = ["Attitude", "TriMesh", "Hull", "PlaneSection"]

_TOL = 1e-12


@dataclass(frozen=True)
class Attitude:
    """Prescribed rigid-body attitude, as offsets from the reference condition.

    ``sinkage`` is positive downward (deeper immersion), in metres.  ``trim`` is positive
    bow-down and ``heel`` positive starboard-down, both in degrees.  Rotations are applied
    about ``pivot`` (trim about the y-axis, then heel about the x-axis), after which the
    hull is lowered by ``sinkage``.  The free surface is always the plane z = 0.
    """

    sinkage: float = 0.0
    trim: float = 0.0
    heel: float = 0.0
    pivot: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def matrix(self) -> tuple[np.ndarray, np.ndarray]:
        """Rotation and translation such that x' = R (x - pivot) + pivot - sinkage*z_hat."""
        t = np.radians(self.trim)
        p = np.radians(self.heel)
        # Trim about +y: bow-down means the bow (large x) moves to smaller z.
        ct, st = np.cos(t), np.sin(t)
        r_trim = np.array([[ct, 0.0, st], [0.0, 1.0, 0.0], [-st, 0.0, ct]])
        # Heel about +x: starboard-down means +y moves to smaller z.
        cp, sp = np.cos(p), np.sin(p)
        r_heel = np.array([[1.0, 0.0, 0.0], [0.0, cp, sp], [0.0, -sp, cp]])
        rot = r_heel @ r_trim
        piv = np.asarray(self.pivot, dtype=float)
        trans = piv - rot @ piv - np.array([0.0, 0.0, self.sinkage])
        return rot, trans


@dataclass(frozen=True)
class PlaneSection:
    """Result of intersecting a body with a plane.

    ``area`` is computed by the shoelace sum over ``segments``, which is a sum of
    independent edge terms and so needs no ordering into loops.  The plane basis is
    chosen so that the sum equals the Green's-theorem form (1/2) * closed integral of
    (y dz - z dy) for a plane of constant x, and (1/2) * closed integral of
    (x dy - y dx) for a plane of constant z.  A section that is left open along z = 0
    is therefore still correct, because an edge lying in z = 0 contributes exactly
    nothing to either form: that is what lets an immersed section area be taken from
    the hull cut alone, with no waterplane lid.

    Extents are deliberately not reported here: "beam" and "length" depend on the
    body axes, not on the plane basis, so callers derive them from ``segments``.
    """

    area: float
    centroid: np.ndarray            # 3-vector; NaN when area is zero
    segments: np.ndarray            # (n, 2, 3) oriented cut segments
    perimeter: float


class TriMesh:
    """Triangulated surface with outward-oriented faces."""

    __slots__ = ("vertices", "faces")

    def __init__(self, vertices: np.ndarray, faces: np.ndarray):
        self.vertices = np.ascontiguousarray(vertices, dtype=float)
        self.faces = np.ascontiguousarray(faces, dtype=np.int64)
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError("vertices must be (n, 3)")
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError("faces must be (m, 3)")

    # -- basic quantities -------------------------------------------------------
    def triangles(self) -> np.ndarray:
        return self.vertices[self.faces]                       # (m, 3, 3)

    def area_normals(self) -> np.ndarray:
        """Area-weighted normals: |result| is twice the triangle area."""
        t = self.triangles()
        return np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])

    def areas(self) -> np.ndarray:
        return 0.5 * np.linalg.norm(self.area_normals(), axis=1)

    def unit_normals(self) -> np.ndarray:
        an = self.area_normals()
        mag = np.linalg.norm(an, axis=1, keepdims=True)
        return np.divide(an, mag, out=np.zeros_like(an), where=mag > 0.0)

    def centroids(self) -> np.ndarray:
        return self.triangles().mean(axis=1)

    @property
    def n_faces(self) -> int:
        return self.faces.shape[0]

    # -- transforms ------------------------------------------------------------
    def transformed(self, rot: np.ndarray, trans: np.ndarray) -> "TriMesh":
        return TriMesh(self.vertices @ rot.T + trans, self.faces)

    def mirrored_y(self) -> "TriMesh":
        """Reflect in y and reverse the winding, so normals stay outward."""
        v = self.vertices.copy()
        v[:, 1] *= -1.0
        return TriMesh(v, self.faces[:, ::-1])

    def joined(self, other: "TriMesh") -> "TriMesh":
        return TriMesh(
            np.vstack([self.vertices, other.vertices]),
            np.vstack([self.faces, other.faces + self.vertices.shape[0]]),
        )

    def dropped_degenerate(self, rel_tol: float = 1e-12) -> "TriMesh":
        a = self.areas()
        keep = a > rel_tol * max(a.max(initial=0.0), 1e-300)
        return TriMesh(self.vertices, self.faces[keep])

    def flipped(self) -> "TriMesh":
        return TriMesh(self.vertices, self.faces[:, ::-1])

    def without_panels_in_plane(self, axis: int = 1, level: float = 0.0,
                                tol: float = 1e-12) -> "TriMesh":
        """Drop panels lying entirely in a plane, before mirroring across that plane.

        Such a panel has zero enclosed thickness, and mirroring turns it into two
        geometrically coincident panels with opposite normals.  Their rows in a
        source-influence matrix are then identical up to sign, so the matrix is exactly
        singular -- and a linear solver will return an answer for it without complaint.
        One such pair, at the stern-keel corner of a Wigley hull where the pointed end and
        the keel line meet, took the condition number from about 2 to 1e16.
        """
        scale = float(np.ptp(self.vertices, axis=0).max()) or 1.0
        coord = self.triangles()[:, :, axis] - level
        keep = np.abs(coord).max(axis=1) > tol * scale
        return TriMesh(self.vertices, self.faces[keep]).compacted()

    def compacted(self) -> "TriMesh":
        """Drop vertices no face references, so vertex extents describe the real surface."""
        if self.faces.size == 0:
            return TriMesh(np.zeros((0, 3)), self.faces)
        used, inverse = np.unique(self.faces.ravel(), return_inverse=True)
        return TriMesh(self.vertices[used], inverse.reshape(self.faces.shape))

    # -- clipping --------------------------------------------------------------
    def clipped_below(self, level: float = 0.0, axis: int = 2) -> "TriMesh":
        """Retain the part with coordinate ``axis`` <= ``level``, cutting triangles exactly."""
        v = self.vertices
        s = v[:, axis] - level
        tri = self.faces
        sign = s[tri] <= 0.0
        count = sign.sum(axis=1)
        out_v = [v]
        out_f = [tri[count == 3]]
        next_index = v.shape[0]
        new_verts: list[np.ndarray] = []
        new_faces: list[np.ndarray] = []

        def cut(ia: np.ndarray, ib: np.ndarray) -> np.ndarray:
            """Interpolated crossing points on edges ia->ib; returns new vertex indices."""
            nonlocal next_index
            pa, pb = v[ia], v[ib]
            sa, sb = s[ia], s[ib]
            w = sa / (sa - sb)
            new_verts.append(pa + w[:, None] * (pb - pa))
            idx = np.arange(next_index, next_index + ia.size)
            next_index += ia.size
            return idx

        # One vertex inside -> one triangle.
        m1 = count == 1
        if np.any(m1):
            f = tri[m1]
            inside = np.argmax(sign[m1], axis=1)
            i0 = f[np.arange(f.shape[0]), inside]
            i1 = f[np.arange(f.shape[0]), (inside + 1) % 3]
            i2 = f[np.arange(f.shape[0]), (inside + 2) % 3]
            a = cut(i0, i1)
            b = cut(i0, i2)
            new_faces.append(np.column_stack([i0, a, b]))
        # Two vertices inside -> quad, split into two triangles.
        m2 = count == 2
        if np.any(m2):
            f = tri[m2]
            outside = np.argmin(sign[m2], axis=1)
            j0 = f[np.arange(f.shape[0]), outside]              # the outside vertex
            j1 = f[np.arange(f.shape[0]), (outside + 1) % 3]
            j2 = f[np.arange(f.shape[0]), (outside + 2) % 3]
            a = cut(j0, j1)
            b = cut(j0, j2)
            new_faces.append(np.column_stack([j1, j2, b]))
            new_faces.append(np.column_stack([j1, b, a]))

        if new_verts:
            out_v.append(np.vstack(new_verts))
        if new_faces:
            out_f.append(np.vstack(new_faces))
        faces = np.vstack([f for f in out_f if f.size]) if any(f.size for f in out_f) else np.zeros((0, 3), np.int64)
        return TriMesh(np.vstack(out_v), faces).dropped_degenerate().compacted()

    # -- topology --------------------------------------------------------------
    def boundary_vertices(self, merge_tol: float = 1e-9) -> np.ndarray:
        """Vertices on edges used by exactly one triangle, after welding near-duplicates."""
        key = np.round(self.vertices / merge_tol).astype(np.int64)
        _, inverse = np.unique(key, axis=0, return_inverse=True)
        welded = inverse[self.faces]
        e = np.vstack([welded[:, [0, 1]], welded[:, [1, 2]], welded[:, [2, 0]]])
        e_sorted = np.sort(e, axis=1)
        _, idx, counts = np.unique(e_sorted, axis=0, return_index=True, return_counts=True)
        lone = e_sorted[idx[counts == 1]]
        if lone.size == 0:
            return np.zeros((0, 3))
        keep = np.unique(lone.ravel())
        # Map welded indices back to representative coordinates.
        rep = np.zeros((welded.max() + 1, 3))
        rep[welded.ravel()] = self.vertices[self.faces.ravel()]
        return rep[keep]

    def signed_volume(self) -> float:
        """(1/3) integral of r.n over the surface; positive for outward normals.

        This equals the enclosed volume only when the surface is closed, or when it is
        open solely along a plane through the origin whose normal is radial there --
        in practice, a hull clipped at the free surface z = 0.  Clip at z = 0 (use the
        reference datum shift), never at an arbitrary level, or the missing lid
        contributes (1/3) * level * area.
        """
        t = self.triangles()
        return float(np.einsum("ij,ij->", t.mean(axis=1), self.area_normals()) / 6.0)

    # -- plane sections --------------------------------------------------------
    def section(self, normal: np.ndarray, level: float, tol_rel: float = 1e-12) -> PlaneSection:
        """Intersect with the plane n.x = level.

        Vertices within a relative tolerance of the plane are classified as lying *on*
        it and each triangle configuration is then handled explicitly.  That matters:
        with a naive strict inequality, a plane containing mesh vertices silently loses
        boundary segments -- on a tessellated sphere, a cut through the pole meridians
        came out 3.4e-3 low in area, against 6.4e-5 for the same cut nudged off the
        vertices.  Waterplanes and station cuts land on mesh vertices routinely, so the
        degenerate cases are the normal case, not an exotic one.

        Cut segments are oriented so that a shoelace sum in the plane's own basis gives
        the positive enclosed area.  Because the shoelace is a sum of independent edge
        terms, the segments need no ordering into loops.
        """
        p_hat = np.asarray(normal, dtype=float)
        p_hat = p_hat / np.linalg.norm(p_hat)
        v = self.vertices
        s = v @ p_hat - level
        scale = float(np.ptp(v, axis=0).max()) or 1.0
        eps = tol_rel * scale
        tri = self.faces
        st = s[tri]
        below = st < -eps
        above = st > eps
        on = ~below & ~above
        n_below, n_above, n_on = below.sum(1), above.sum(1), on.sum(1)

        starts: list[np.ndarray] = []
        ends: list[np.ndarray] = []
        face_ids: list[np.ndarray] = []

        def crossing(ia: np.ndarray, ib: np.ndarray) -> np.ndarray:
            sa, sb = s[ia], s[ib]
            w = sa / (sa - sb)
            return v[ia] + w[:, None] * (v[ib] - v[ia])

        def pick(mask_faces: np.ndarray, sel: np.ndarray, k: int) -> list[np.ndarray]:
            """Vertex ids of the k positions where ``sel`` is True, per selected face."""
            pos = np.argsort(~sel, axis=1, kind="stable")[:, :k]
            rows = np.arange(mask_faces.shape[0])[:, None]
            return [mask_faces[rows[:, 0], pos[:, j]] for j in range(k)]

        # Case 1: an edge lies in the plane and the third vertex is below, so that edge
        # bounds the immersed region.  Emitting only from the below side counts it once.
        m = (n_on == 2) & (n_below == 1)
        if np.any(m):
            f = tri[m]
            a, b = pick(f, on[m], 2)
            starts.append(v[a]); ends.append(v[b]); face_ids.append(np.flatnonzero(m))

        # Case 2: one vertex on the plane, one below, one above.
        m = (n_on == 1) & (n_below == 1) & (n_above == 1)
        if np.any(m):
            f = tri[m]
            (o,) = pick(f, on[m], 1)
            (lo,) = pick(f, below[m], 1)
            (hi,) = pick(f, above[m], 1)
            starts.append(v[o]); ends.append(crossing(lo, hi)); face_ids.append(np.flatnonzero(m))

        # Case 3: a clean cut with no vertex on the plane.
        m = (n_on == 0) & (n_below >= 1) & (n_above >= 1)
        if np.any(m):
            f = tri[m]
            lone_below = n_below[m] == 1
            sel = np.where(lone_below[:, None], below[m], above[m])
            other = np.where(lone_below[:, None], above[m], below[m])
            (i0,) = pick(f, sel, 1)
            i1, i2 = pick(f, other, 2)
            starts.append(crossing(i0, i1)); ends.append(crossing(i0, i2))
            face_ids.append(np.flatnonzero(m))

        if not starts:
            return PlaneSection(0.0, np.full(3, np.nan), np.zeros((0, 2, 3)), 0.0)
        start = np.vstack(starts)
        end = np.vstack(ends)
        fid = np.concatenate(face_ids)

        # Orient by the in-plane projection of the outward face normal: with m the
        # outward in-plane normal, the boundary runs along p_hat x m.
        n_face = self.unit_normals()[fid]
        proj = n_face - (n_face @ p_hat)[:, None] * p_hat
        mag = np.linalg.norm(proj, axis=1)
        keep = mag > 1e-10
        proj = proj[keep] / mag[keep, None]
        start, end = start[keep], end[keep]
        d_want = np.cross(np.broadcast_to(p_hat, proj.shape), proj)
        flip = np.einsum("ij,ij->i", end - start, d_want) < 0.0
        start, end = (np.where(flip[:, None], end, start), np.where(flip[:, None], start, end))

        # Drop segments of zero length: they carry no shoelace contribution.
        length = np.linalg.norm(end - start, axis=1)
        alive = length > 1e-14 * scale
        start, end, length = start[alive], end[alive], length[alive]
        if start.shape[0] == 0:
            return PlaneSection(0.0, np.full(3, np.nan), np.zeros((0, 2, 3)), 0.0)

        helper = np.array([1.0, 0.0, 0.0]) if abs(p_hat[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        e1 = np.cross(helper, p_hat)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(p_hat, e1)
        a1, a2 = start @ e1, start @ e2
        b1, b2 = end @ e1, end @ e2
        twice = a1 * b2 - b1 * a2
        area2 = twice.sum()
        area = 0.5 * abs(area2)
        if area <= 0.0:
            centroid = np.full(3, np.nan)
        else:
            c1 = float(((a1 + b1) * twice).sum() / (3.0 * area2))
            c2 = float(((a2 + b2) * twice).sum() / (3.0 * area2))
            centroid = level * p_hat + c1 * e1 + c2 * e2
        return PlaneSection(area, centroid, np.stack([start, end], axis=1), float(length.sum()))


def tessellate(surface: NurbsSurface, n_u: int, n_v: int) -> TriMesh:
    """Uniform parametric tessellation of one patch into a TriMesh."""
    if n_u < 1 or n_v < 1:
        raise ValueError("n_u and n_v must be at least 1")
    u0, u1 = surface.u_range
    v0, v1 = surface.v_range
    u = np.linspace(u0, u1, n_u + 1)
    v = np.linspace(v0, v1, n_v + 1)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    pts = nurbs.evaluate(surface, uu, vv).reshape(-1, 3)
    idx = np.arange((n_u + 1) * (n_v + 1)).reshape(n_u + 1, n_v + 1)
    a = idx[:-1, :-1].ravel()
    b = idx[1:, :-1].ravel()
    c = idx[1:, 1:].ravel()
    d = idx[:-1, 1:].ravel()
    faces = np.vstack([np.column_stack([a, b, c]), np.column_stack([a, c, d])])
    return TriMesh(pts, faces).dropped_degenerate()


@dataclass
class Hull:
    """A hull as one or more NURBS patches plus the meshes derived from them."""

    patches: list[NurbsSurface]
    wetted_patch_indices: tuple[int, ...]
    mirrored: bool
    source: Path | None = None
    _datum_shift: float = 0.0

    @classmethod
    def from_iges(
        cls,
        path: str | Path,
        wetted_patches: tuple[int, ...] | None = None,
        mirror: bool | None = None,
    ) -> "Hull":
        """Load patches from an IGES file.

        ``mirror=None`` autodetects a half model: a patch whose transverse extent lies
        wholly on one side of y = 0 is taken to be half a hull.  ``wetted_patches``
        selects which patches take part in hydrostatics; the rest are retained and
        checked for immersion, never silently ignored.
        """
        f: IgesFile = read_iges(path)
        patches = list(f.surfaces)
        if wetted_patches is None:
            # The patch with the largest control-net bounding box is the canoe body.
            spans = [np.prod(p.bounding_box()[1] - p.bounding_box()[0] + 1e-12) for p in patches]
            wetted_patches = (int(np.argmax(spans)),)
        if mirror is None:
            lo, hi = patches[wetted_patches[0]].bounding_box()
            mirror = bool(lo[1] >= -1e-9 or hi[1] <= 1e-9)
        return cls(patches=patches, wetted_patch_indices=tuple(wetted_patches),
                   mirrored=bool(mirror), source=Path(path))

    # -- meshing ---------------------------------------------------------------
    def mesh(self, n_u: int, n_v: int, attitude: Attitude | None = None) -> TriMesh:
        """Full-hull mesh of the wetted patches at the given attitude, outward-oriented."""
        parts = [tessellate(self.patches[i], n_u, n_v) for i in self.wetted_patch_indices]
        mesh = parts[0]
        for extra in parts[1:]:
            mesh = mesh.joined(extra)
        if self.mirrored:
            mesh = mesh.without_panels_in_plane(axis=1, level=0.0)
            mesh = mesh.joined(mesh.mirrored_y())
        if self._datum_shift:
            mesh = TriMesh(mesh.vertices - np.array([0.0, 0.0, self._datum_shift]), mesh.faces)
        if attitude is not None:
            mesh = mesh.transformed(*attitude.matrix())
        return self._orient(mesh)

    def waterline_fitted_mesh(self, n_girth: int, n_long: int,
                              attitude: Attitude | None = None,
                              first_depth: float = 0.0) -> TriMesh:
        """Body-fitted mesh running from the waterline down to the keel, no clipping.

        Clipping a parametric mesh at z = 0 is exact but produces slivers along the
        waterline whose centroids sit arbitrarily close to z = 0.  Here the girth direction
        is parameterised from the waterline instead, so the depth of the first row is
        controlled rather than inherited from the parameterisation.

        ``first_depth`` is that control, **in metres**, and it is the lower edge of the
        topmost row.  Zero means uniform girth spacing, which is not the same as no
        control: it is the reference against which a nonzero value is judged.  A station
        whose own immersed depth is less than ``first_depth`` -- there is always a
        neighbourhood of the bow and stern where that holds -- gets a single row spanning
        all of it, so the guarantee is on the row's lower edge and not on the depth of
        every centroid.  Callers that need the latter must measure it, and
        ``docs/formulation.md`` section 15 does.

        Why this parameter exists at all: the Kelvin Green function diverges
        logarithmically as k0|z + zeta| tends to zero, and the oscillation count of a panel
        pair grows as |y_i - y_j| / |z_i + z_j|, so a mesh reaching the waterline is the
        expensive and delicate corner for the wave kernel.  Exposing the depth as a
        parameter is what makes the dependence measurable instead of accidental, and
        ``docs/formulation.md`` section 15 reports it.

        A caution on reading that dependence.  The first time it was measured the
        resistance varied by a factor of 2.5 across this parameter and it looked like the
        known waterline difficulty of the Neumann-Kelvin problem; it was a quadrature error
        in the Green function's gradient, and section 15 records it.  A strong dependence
        here is a reason to check the kernel before it is a reason to believe the physics.

        Only the wetted patches take part.  Requires the free surface at z = 0, so apply
        the reference datum shift first.
        """
        if n_girth < 1 or n_long < 1:
            raise ValueError("n_girth and n_long must be at least 1")
        if first_depth < 0.0:
            raise ValueError("first_depth must not be negative")
        rot, trans = attitude.matrix() if attitude is not None else (np.eye(3), np.zeros(3))
        shift = np.array([0.0, 0.0, self._datum_shift])

        def place(u, v, patch):
            pt = nurbs.evaluate(patch, u, v) - shift
            return pt @ rot.T + trans

        parts: list[TriMesh] = []
        for index in self.wetted_patch_indices:
            patch = self.patches[index]
            u0, u1 = patch.u_range
            v0, v1 = patch.v_range
            uniform = np.arange(n_girth + 1) / n_girth
            rows = []
            v_kept = []
            for v in np.linspace(v0, v1, n_long + 1):
                if place(u0, v, patch)[2] <= 0.0:
                    continue                     # station wholly submerged: no waterline
                if place(u1, v, patch)[2] > 0.0:
                    continue                     # station wholly dry
                u_wl = self._bisect_depth(place, patch, v, u0, u1, 0.0)
                if first_depth <= 0.0 or n_girth < 2:
                    rows.append(u_wl + (u1 - u_wl) * uniform)
                    v_kept.append(v)
                    continue
                keel_depth = -place(u1, v, patch)[2]
                if keel_depth <= first_depth:
                    # Station shallower than the requested first row: one row takes it all.
                    rows.append(u_wl + (u1 - u_wl) * uniform)
                    v_kept.append(v)
                    continue
                u_first = self._bisect_depth(place, patch, v, u_wl, u1, -first_depth)
                below = (u1 - u_first) * np.arange(1, n_girth) / (n_girth - 1)
                rows.append(np.concatenate([[u_wl, u_first], u_first + below]))
                v_kept.append(v)
            if len(rows) < 2:
                raise ValueError("fewer than two stations intersect the free surface")
            u_grid = np.array(rows)                              # (n_v, n_girth+1)
            v_grid = np.array(v_kept)[:, None] * np.ones((1, u_grid.shape[1]))
            pts = place(u_grid.ravel(), v_grid.ravel(), patch).reshape(-1, 3)
            nv, ng = u_grid.shape
            idx = np.arange(nv * ng).reshape(nv, ng)
            a = idx[:-1, :-1].ravel(); b = idx[1:, :-1].ravel()
            c = idx[1:, 1:].ravel(); d = idx[:-1, 1:].ravel()
            faces = np.vstack([np.column_stack([a, b, c]), np.column_stack([a, c, d])])
            parts.append(TriMesh(pts, faces).dropped_degenerate())
        mesh = parts[0]
        for extra in parts[1:]:
            mesh = mesh.joined(extra)
        if self.mirrored:
            mesh = mesh.without_panels_in_plane(axis=1, level=0.0)
            mesh = mesh.joined(mesh.mirrored_y())
        return self._orient(mesh)

    def other_patch_mesh(self, n_u: int, n_v: int, attitude: Attitude | None = None) -> TriMesh | None:
        """Mesh of the patches excluded from hydrostatics, for immersion checking."""
        others = [i for i in range(len(self.patches)) if i not in self.wetted_patch_indices]
        if not others:
            return None
        parts = [tessellate(self.patches[i], n_u, n_v) for i in others]
        mesh = parts[0]
        for extra in parts[1:]:
            mesh = mesh.joined(extra)
        if self.mirrored:
            mesh = mesh.without_panels_in_plane(axis=1, level=0.0)
            mesh = mesh.joined(mesh.mirrored_y())
        if self._datum_shift:
            mesh = TriMesh(mesh.vertices - np.array([0.0, 0.0, self._datum_shift]), mesh.faces)
        if attitude is not None:
            mesh = mesh.transformed(*attitude.matrix())
        return mesh

    @staticmethod
    def _bisect_depth(place, patch, v: float, lo: float, hi: float, level: float,
                      iterations: int = 60) -> float:
        """Parameter u in [lo, hi] at which the station crosses z = ``level``.

        Assumes z decreases with u over the bracket, which the caller has checked.
        """
        for _ in range(iterations):
            mid = 0.5 * (lo + hi)
            if place(mid, v, patch)[2] > level:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    @staticmethod
    def _orient(mesh: TriMesh) -> TriMesh:
        """Force outward normals using the sign of the enclosed volume below the datum."""
        probe = mesh.clipped_below(mesh.vertices[:, 2].max() - _TOL)
        return mesh if probe.signed_volume() >= 0.0 else mesh.flipped()

    def with_datum_shift(self, shift: float) -> "Hull":
        """Return a copy whose geometry is lowered by ``shift`` in z (the reference heave)."""
        return replace(self, _datum_shift=shift)

    @property
    def datum_shift(self) -> float:
        return self._datum_shift
