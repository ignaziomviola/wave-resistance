"""Hydrostatics of a clipped hull mesh, by the divergence theorem.

Every integral is taken over the wetted hull triangles alone.  With the free surface at
z = 0, the waterplane lid contributes nothing to any of the integrands used here, so no
lid has to be constructed:

    volume        div(0, y, 0) = 1   ->  V = closed integral of  y n_y dS
                  div(x, 0, 0) = 1   ->  V = closed integral of  x n_x dS
                  div(0, 0, z) = 1   ->  V = closed integral of  z n_z dS
                  div(r)/3 = 1       ->  V = (1/3) closed integral of r.n dS
    x moment      div(x^2/2, 0, 0) = x
    y moment      div(0, y^2/2, 0) = y
    z moment      div(0, 0, z^2/2) = z

On the lid n_x = n_y = 0 and z = 0, which kills the lid term in all seven.  The four
volume routes are algebraically independent given a triangulation, so their spread is a
genuine consistency check rather than a restatement.

Triangle integrals use the three-edge-midpoint rule, which is exact for the quadratic
integrands above on a flat triangle.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from .hull import Attitude, Hull, TriMesh

__all__ = ["Hydrostatics", "hydrostatics", "solve_reference_heave", "RHO_FRESH", "GRAVITY"]

RHO_FRESH = 1000.0      # kg/m^3, Delft towing tank
GRAVITY = 9.80665       # m/s^2


def _surface_integral(mesh: TriMesh, values: np.ndarray, component: int) -> float:
    """Integral of a scalar field against n_component over the mesh.

    ``values`` has shape (n_faces, 3) holding the field at the three edge midpoints.
    """
    area_normals = mesh.area_normals()          # = 2 * A * n_hat
    return float(np.einsum("i,i->", 0.5 * area_normals[:, component], values.mean(axis=1)))


def _midpoints(mesh: TriMesh) -> np.ndarray:
    t = mesh.triangles()
    return np.stack([0.5 * (t[:, 0] + t[:, 1]),
                     0.5 * (t[:, 1] + t[:, 2]),
                     0.5 * (t[:, 2] + t[:, 0])], axis=1)      # (n, 3, 3)


@dataclass(frozen=True)
class Hydrostatics:
    volume: float
    wetted_area: float
    waterplane_area: float
    midship_area: float
    lwl: float
    bwl: float
    draught: float
    lcb: float
    tcb: float
    vcb: float
    lcf: float
    tcf: float
    cp: float
    cm: float
    cb: float
    cwp: float
    lcb_from_midship: float
    lcf_from_midship: float
    x_fwd: float
    x_aft: float
    volume_routes: tuple[float, float, float, float]
    volume_spread: float
    n_faces: int

    def as_dict(self) -> dict:
        return asdict(self)

    def displacement_mass(self, rho: float = RHO_FRESH) -> float:
        return rho * self.volume


def hydrostatics(mesh: TriMesh, n_sections: int = 240) -> Hydrostatics:
    """Hydrostatics of a hull mesh whose free surface is the plane z = 0.

    ``mesh`` must be the *full* hull (both sides), unclipped; it is clipped here so that
    the waterline section can be taken from the uncut mesh, where z = 0 is a genuine
    transversal cut rather than a boundary.
    """
    below = mesh.clipped_below(0.0, axis=2)
    if below.n_faces == 0:
        raise ValueError("hull is entirely above the free surface z = 0")

    mid = _midpoints(below)
    x, y, z = mid[..., 0], mid[..., 1], mid[..., 2]
    v_x = _surface_integral(below, x, 0)
    v_y = _surface_integral(below, y, 1)
    v_z = _surface_integral(below, z, 2)
    v_rn = below.signed_volume()
    routes = (v_rn, v_x, v_y, v_z)
    volume = float(np.mean(routes))
    spread = float(max(routes) - min(routes))

    mom_x = _surface_integral(below, 0.5 * x * x, 0)
    mom_y = _surface_integral(below, 0.5 * y * y, 1)
    mom_z = _surface_integral(below, 0.5 * z * z, 2)
    lcb, tcb, vcb = mom_x / volume, mom_y / volume, mom_z / volume

    wetted = float(below.areas().sum())

    # Waterline from the uncut mesh: z = 0 is a transversal cut there.
    plane = mesh.section(np.array([0.0, 0.0, 1.0]), 0.0)
    if plane.area <= 0.0:
        raise ValueError("no waterplane section found at z = 0")
    pts = plane.segments.reshape(-1, 3)
    x_fwd, x_aft = float(pts[:, 0].max()), float(pts[:, 0].min())
    lwl = x_fwd - x_aft
    bwl = float(pts[:, 1].max() - pts[:, 1].min())
    draught = float(-below.vertices[:, 2].min())

    # Sectional areas from the clipped mesh: an edge lying in z = 0 contributes nothing
    # to the (1/2)(y dz - z dy) form, so no lid is needed.
    xs = np.linspace(x_aft, x_fwd, n_sections + 1)[1:-1]
    x_hat = np.array([1.0, 0.0, 0.0])
    areas = np.array([below.section(x_hat, xv).area for xv in xs])
    midship_area = float(areas.max()) if areas.size else float("nan")

    x_mid = 0.5 * (x_fwd + x_aft)
    return Hydrostatics(
        volume=volume, wetted_area=wetted, waterplane_area=plane.area,
        midship_area=midship_area, lwl=lwl, bwl=bwl, draught=draught,
        lcb=lcb, tcb=tcb, vcb=vcb,
        lcf=float(plane.centroid[0]), tcf=float(plane.centroid[1]),
        cp=volume / (midship_area * lwl), cm=midship_area / (bwl * draught),
        cb=volume / (lwl * bwl * draught), cwp=plane.area / (lwl * bwl),
        lcb_from_midship=lcb - x_mid, lcf_from_midship=float(plane.centroid[0]) - x_mid,
        x_fwd=x_fwd, x_aft=x_aft,
        volume_routes=routes, volume_spread=spread, n_faces=below.n_faces,
    )


def solve_reference_heave(
    hull: Hull,
    target_volume: float,
    n_u: int = 48,
    n_v: int = 240,
    tol: float = 1e-9,
    max_iter: int = 60,
) -> float:
    """Datum shift (metres) that floats the hull at ``target_volume`` with zero trim/heel.

    The shift lowers the geometry so that the flotation plane becomes z = 0.  Returns the
    shift; apply it with :meth:`Hull.with_datum_shift`.
    """
    if target_volume <= 0.0:
        raise ValueError("target_volume must be positive")
    base = hull.with_datum_shift(0.0).mesh(n_u, n_v)
    z_lo = float(base.vertices[:, 2].min())
    z_hi = float(base.vertices[:, 2].max())

    def volume_at(shift: float) -> float:
        moved = TriMesh(base.vertices - np.array([0.0, 0.0, shift]), base.faces)
        clipped = moved.clipped_below(0.0, axis=2)
        return clipped.signed_volume() if clipped.n_faces else 0.0

    lo, hi = z_lo + tol, z_hi
    if volume_at(hi) < target_volume:
        raise ValueError(
            f"target volume {target_volume:g} m^3 exceeds the hull's capacity "
            f"{volume_at(hi):g} m^3 up to the top of the surface"
        )
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if volume_at(mid) < target_volume:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)
