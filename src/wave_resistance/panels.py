"""Constant-source flat panels: exact influence coefficients, and the Rankine solver.

Every influence integrates the kernel over the whole panel.  Centroid value times area is
not used anywhere, because it is wrong exactly where it matters: for a panel on itself, for
adjacent panels, and near the waterline.

With the kernel G = -1/(4 pi r) the potential of a panel carrying constant source density
sigma is phi = -(sigma / 4 pi) I0, where I0 is the integral of 1/r over the panel.  In a
frame with the panel in its own plane zeta = 0,

    dI0/dzeta = -Omega,
    dI0/dxi   = -sum over edges of (eta_{i+1} - eta_i)/d_i * L_i,
    dI0/deta  = +sum over edges of (xi_{i+1}  - xi_i )/d_i * L_i,
    L_i       = log( (r_i + r_{i+1} + d_i) / (r_i + r_{i+1} - d_i) ),

with Omega the signed solid angle the panel subtends at the field point.  The normal
velocity is therefore exactly sigma * Omega / (4 pi), which is worth noticing: as the field
point approaches the panel Omega tends to +/- 2 pi, so the jump term of the integral
equation comes out of the same expression as everything else rather than being bolted on,
and a field point in the panel's own plane but outside it gets exactly zero, as symmetry
demands.

Signs and the branch of L are fixed by :func:`velocity_by_quadrature`, not asserted.
"""

from __future__ import annotations

import functools

import numpy as np

from .hull import TriMesh

__all__ = [
    "solid_angle", "source_velocity", "velocity_by_quadrature",
    "rankine_normal_velocity_matrix", "solve_rankine",
]


def solid_angle(triangles: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Signed solid angle subtended by each triangle at each point.

    Van Oosterom and Strackee's formula, which is stable for near-degenerate
    configurations and needs no case analysis:

        tan(Omega / 2) = a . (b x c) / (|a||b||c| + (a.b)|c| + (a.c)|b| + (b.c)|a|)

    with a, b, c the vectors from the field point to the three vertices.  The sign follows
    the triangle's winding, so it agrees with the outward normal used elsewhere.

    Sign convention, established by test rather than assumed: the result is
    +4 pi for a point enclosed by an outward-oriented closed surface, and 0 for a point
    outside it.  Equivalently it is the integral of n.(q - p)/|q - p|^3, so in a panel-local
    frame it equals minus the integral of zeta/r^3, which is why :func:`source_velocity`
    carries a minus sign on the normal component.

    ``triangles`` is (m, 3, 3) and ``points`` is (n, 3); the result is (n, m).
    """
    tri = np.asarray(triangles, dtype=float)
    pts = np.asarray(points, dtype=float)
    a = tri[None, :, 0, :] - pts[:, None, :]
    b = tri[None, :, 1, :] - pts[:, None, :]
    c = tri[None, :, 2, :] - pts[:, None, :]
    na = np.linalg.norm(a, axis=-1)
    nb = np.linalg.norm(b, axis=-1)
    nc = np.linalg.norm(c, axis=-1)
    numer = np.einsum("...i,...i->...", a, np.cross(b, c))
    denom = (na * nb * nc
             + np.einsum("...i,...i->...", a, b) * nc
             + np.einsum("...i,...i->...", a, c) * nb
             + np.einsum("...i,...i->...", b, c) * na)
    return 2.0 * np.arctan2(numer, denom)


def source_velocity(triangles: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Velocity induced at each point by each triangle carrying unit source density.

    Returns (n, m, 3) with the kernel G = -1/(4 pi r), so a unit-density panel of area A
    far away induces a velocity of magnitude A/(4 pi r^2) directed away from it.
    """
    tri = np.asarray(triangles, dtype=float)
    pts = np.asarray(points, dtype=float)
    m = tri.shape[0]

    # Panel-local orthonormal frame from the winding, so the normal matches solid_angle.
    e0 = tri[:, 1] - tri[:, 0]
    e1 = tri[:, 2] - tri[:, 0]
    nrm = np.cross(e0, e1)
    area2 = np.linalg.norm(nrm, axis=1)
    nhat = nrm / area2[:, None]
    xhat = e0 / np.linalg.norm(e0, axis=1)[:, None]
    yhat = np.cross(nhat, xhat)

    rel = pts[:, None, :] - tri[None, :, 0, :]                     # (n, m, 3)
    xi_p = np.einsum("nmi,mi->nm", rel, xhat)
    eta_p = np.einsum("nmi,mi->nm", rel, yhat)
    z_p = np.einsum("nmi,mi->nm", rel, nhat)

    # Vertices in the local frame, relative to vertex 0.
    vrel = tri - tri[:, 0:1, :]                                    # (m, 3, 3)
    vx = np.einsum("mvi,mi->mv", vrel, xhat)                       # (m, 3)
    vy = np.einsum("mvi,mi->mv", vrel, yhat)

    d_xi = np.zeros((pts.shape[0], m))
    d_eta = np.zeros((pts.shape[0], m))
    for k in range(3):
        k2 = (k + 1) % 3
        ax = vx[None, :, k] - xi_p
        ay = vy[None, :, k] - eta_p
        bx = vx[None, :, k2] - xi_p
        by = vy[None, :, k2] - eta_p
        dx = vx[None, :, k2] - vx[None, :, k]
        dy = vy[None, :, k2] - vy[None, :, k]
        d = np.hypot(dx, dy)
        ra = np.sqrt(ax * ax + ay * ay + z_p * z_p)
        rb = np.sqrt(bx * bx + by * by + z_p * z_p)
        num = ra + rb + d
        den = ra + rb - d
        # den vanishes only when the field point sits on the edge segment itself.
        safe = den > 1e-300
        log_term = np.where(safe, np.log(np.where(safe, num, 1.0) / np.where(safe, den, 1.0)), 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(d > 0.0, log_term / np.where(d > 0.0, d, 1.0), 0.0)
        d_xi += -dy * scale
        d_eta += dx * scale

    omega = solid_angle(tri, pts)
    # phi = -(1/4 pi) I0, so the velocity is -(1/4 pi) grad I0.  The normal component is
    # (1/4 pi) times the integral of zeta/r^3, which is -omega by the convention above.
    # Both in-plane signs and this one were fixed against velocity_by_quadrature.
    v_local = np.stack([-d_xi, -d_eta, -omega], axis=-1) / (4.0 * np.pi)
    return (v_local[..., 0, None] * xhat[None, :, :]
            + v_local[..., 1, None] * yhat[None, :, :]
            + v_local[..., 2, None] * nhat[None, :, :])


@functools.lru_cache(maxsize=8)
def _triangle_rule(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Tensor Gauss-Legendre rule on the reference triangle, via the Duffy transform.

    The weights integrate over the reference triangle and so sum to its area, 1/2.
    Callers scale by 2 * (actual triangle area) to map onto a physical panel.
    """
    n, w = np.polynomial.legendre.leggauss(order)
    a = 0.5 * (n + 1.0)
    wa = 0.5 * w
    u, v = np.meshgrid(a, a, indexing="ij")
    wu, wv = np.meshgrid(wa, wa, indexing="ij")
    # (u, v) -> (u, v(1-u)) with Jacobian (1-u).
    s = u.ravel()
    t = (v * (1.0 - u)).ravel()
    weight = (wu * wv * (1.0 - u)).ravel()
    return np.stack([s, t], axis=1), weight


def velocity_by_quadrature(triangles: np.ndarray, points: np.ndarray, order: int = 24
                           ) -> np.ndarray:
    """Same quantity as :func:`source_velocity`, by direct quadrature over the panel.

    Exists to fix the signs and the branch of the logarithm in the analytic form, and to
    bound its error; it is not used by the solver, being far slower and useless on the
    diagonal.
    """
    tri = np.asarray(triangles, dtype=float)
    pts = np.asarray(points, dtype=float)
    bary, weight = _triangle_rule(order)
    e0 = tri[:, 1] - tri[:, 0]
    e1 = tri[:, 2] - tri[:, 0]
    area = 0.5 * np.linalg.norm(np.cross(e0, e1), axis=1)
    # Quadrature points: (m, q, 3)
    q = (tri[:, 0:1, :] + bary[None, :, 0, None] * e0[:, None, :]
         + bary[None, :, 1, None] * e1[:, None, :])
    diff = pts[:, None, None, :] - q[None, :, :, :]                # (n, m, q, 3)
    r = np.linalg.norm(diff, axis=-1)
    kernel = diff / r[..., None] ** 3 / (4.0 * np.pi)
    return np.einsum("q,nmqi->nmi", weight * 2.0, kernel) * area[None, :, None]


def rankine_normal_velocity_matrix(mesh: TriMesh) -> np.ndarray:
    """Matrix A with A[i, j] the normal velocity at panel i's centroid from unit density on
    panel j, in an unbounded fluid.

    The diagonal is set explicitly to 1/2.  That is the exact limit -- a panel subtends
    2 pi at its own centroid, giving a normal velocity of sigma/2 on the side the normal
    points to -- but it must be imposed rather than evaluated: at a point in the panel's
    own plane the solid-angle formula has a vanishing numerator, so which of +/- 2 pi comes
    out depends on the sign of a floating-point zero.  Since the field point approaches
    from the fluid side, the limit wanted is +1/2.
    """
    tri = mesh.triangles()
    centroids = mesh.centroids()
    normals = mesh.unit_normals()
    vel = source_velocity(tri, centroids)
    a = np.einsum("nmi,ni->nm", vel, normals)
    np.fill_diagonal(a, 0.5)
    return a


def solve_rankine(mesh: TriMesh, speed: float = 1.0) -> np.ndarray:
    """Source density satisfying d(phi)/dn = u n_x on the mesh, in an unbounded fluid.

    This is the Neumann-Kelvin integral equation with the free-surface part switched off,
    which is what makes the analytic sphere a test of the jump term and of the panel
    integration rather than of the wave kernel.
    """
    a = rankine_normal_velocity_matrix(mesh)
    rhs = speed * mesh.unit_normals()[:, 0]
    return np.linalg.solve(a, rhs)
