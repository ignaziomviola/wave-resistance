"""The Neumann-Kelvin solve: influence matrix, source density, wave resistance.

The integral equation, derived in ``docs/formulation.md``, is

    (1/2) sigma(P) + PV integral over S of sigma(Q) dG(P;Q)/dn(P) dS = u n_x(P),

with G the Kelvin Green function, n directed into the fluid, and no waterline term: the
Green function already satisfies the free-surface condition at every point of z = 0, so a
source-only representation never forms one.

The matrix splits into a Rankine part, whose influences are analytic and carry the jump
term through the solid angle, and a wave part, which is smooth and is integrated over each
source panel by quadrature.  Resistance is taken from the far-field amplitudes rather than
by integrating hull pressure, which is far less sensitive to panel-level error.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .greens import MAX_CYCLES, oscillation_count, wave_part_gradient
from .hull import TriMesh
from .panels import source_velocity
from .spectrum import (panel_quadrature, resistance_constant, wave_resistance_integral)

__all__ = ["NKResult", "rankine_with_image_matrix", "wave_influence_matrix",
           "influence_matrix", "check_envelope", "velocity_matrix", "x_velocity_matrix",
           "pressure_resistance", "solve_nk"]


@dataclass
class NKResult:
    sigma: np.ndarray
    resistance: float
    speed: float
    froude: float
    k0: float
    n_panels: int
    condition_number: float
    body_residual_rms: float
    body_residual_max: float
    spectrum_blocks: int
    lambda_reached: float
    spectrum_converged: bool
    lambda_cap: float = float("inf")
    net_source_flux: float = 0.0
    pressure_resistance: float | None = None
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Fn = {self.froude:.4f}   u = {self.speed:.4f} m/s   k0 = {self.k0:.4f} 1/m",
            f"  panels                {self.n_panels}",
            f"  R_w                   {self.resistance:.6e} N",
            f"  cond(A)               {self.condition_number:.3e}",
            f"  body residual, rms    {self.body_residual_rms:.3e}  (fraction of u)",
            f"  body residual, max    {self.body_residual_max:.3e}",
            f"  spectrum blocks       {self.spectrum_blocks}, marched to lambda "
            f"{self.lambda_reached:.1f}, converged={self.spectrum_converged}",
            f"  spectrum counted to   lambda {self.lambda_cap:.2f}",
            f"  net source flux       {self.net_source_flux:.3e}  (fraction of u S)",
        ]
        if self.pressure_resistance is not None:
            lines.append(f"  R_w by pressure       {self.pressure_resistance:.6e} N  "
                         f"(far-field / pressure = "
                         f"{self.resistance / self.pressure_resistance:.3f})")
        return "\n".join(lines + [f"  note: {n}" for n in self.notes])


def _mirrored_in_z(triangles: np.ndarray) -> np.ndarray:
    """Reflect panels in z = 0 and reverse their winding, so the normal convention holds."""
    image = triangles.copy()
    image[:, :, 2] *= -1.0
    return image[:, ::-1, :]


def rankine_with_image_matrix(mesh: TriMesh, field_points: np.ndarray | None = None,
                              field_normals: np.ndarray | None = None,
                              collocated: bool = False) -> np.ndarray:
    """Normal velocity from the non-wave part of the Kelvin Green function.

    That part is -1/(4 pi R) + 1/(4 pi R1), i.e. the source together with a *negative*
    image at (xi, eta, -zeta), which is the combination that vanishes on z = 0.  The
    influence is therefore the source panel's field minus its image's, and omitting the
    image is not a small error near the free surface: it is what the image is there to
    cancel.

    ``panels.rankine_normal_velocity_matrix`` deliberately has no image, because it serves
    the unbounded-fluid problem used to verify the jump term; this is the Neumann-Kelvin
    counterpart.
    """
    if field_points is None:
        field_points = mesh.centroids()
        collocated = True
    if field_normals is None:
        field_normals = mesh.unit_normals()
    tri = mesh.triangles()
    direct = np.einsum("nmi,ni->nm", source_velocity(tri, field_points), field_normals)
    image = np.einsum("nmi,ni->nm", source_velocity(_mirrored_in_z(tri), field_points),
                      field_normals)
    if collocated:
        # The direct term's diagonal is the exact jump; the image's is regular, since the
        # image panel sits 2|z| below the field point and never coincides with it.
        np.fill_diagonal(direct, 0.5)
    return direct - image


def wave_influence_matrix(mesh: TriMesh, k0: float, order: int = 1,
                          field_points: np.ndarray | None = None,
                          field_normals: np.ndarray | None = None,
                          chunk: int = 4096, kernel_refine: float = 1.0) -> np.ndarray:
    """Normal velocity at each field point from unit source density on each panel, wave part.

    ``order`` sets the quadrature over the *source* panel.  Order 1 is the centroid rule;
    unlike the Rankine part the wave kernel is smooth, so that is a defensible choice, but
    it is a choice and :func:`influence_matrix` records it so a convergence check can be
    run against a higher order.

    ``kernel_refine`` scales the wave-angle quadrature inside the Green function itself.
    The two refinements are independent and V10 exercises them separately: ``order``
    resolves the variation of the kernel across a source panel, ``kernel_refine`` resolves
    the oscillatory theta integral at a fixed pair of points.  Raising it is the way to
    test whether the near-waterline panels of a mesh are being evaluated well enough,
    which is the constraint of ``docs/formulation.md`` section 9.
    """
    if field_points is None:
        field_points = mesh.centroids()
    if field_normals is None:
        field_normals = mesh.unit_normals()
    if order <= 1:
        q_pts = mesh.centroids()[:, None, :]
        q_w = mesh.areas()[:, None]
    else:
        q_pts, q_w = panel_quadrature(mesh, order)

    n_field = field_points.shape[0]
    n_src, n_q = q_pts.shape[:2]
    out = np.zeros((n_field, n_src))
    flat_q = q_pts.reshape(-1, 3)
    src_index = np.repeat(np.arange(n_src), n_q)
    flat_w = q_w.reshape(-1)

    # Rows per pass, so that the (rows x quadrature points) work array stays bounded.
    stride = max(1, chunk // max(flat_q.shape[0], 1))
    for start in range(0, n_field, stride):
        rows = slice(start, start + stride)
        fp = field_points[rows]
        fn = field_normals[rows]
        X = k0 * (fp[:, None, 0] - flat_q[None, :, 0])
        Y = k0 * (fp[:, None, 1] - flat_q[None, :, 1])
        Z = np.minimum(k0 * (fp[:, None, 2] + flat_q[None, :, 2]), -1e-9)
        grad = wave_part_gradient(X, Y, Z, refine=kernel_refine)
        normal_grad = np.einsum("ijk,ik->ij", grad, fn) * k0 ** 2 * flat_w[None, :]
        if n_q == 1:
            out[rows] = normal_grad
        else:
            acc = np.zeros((fp.shape[0], n_src))
            np.add.at(acc.T, src_index, normal_grad.T)
            out[rows] = acc
    return out


def check_envelope(mesh: TriMesh, k0: float) -> tuple[float, int]:
    """Worst panel-pair oscillation count on this mesh, and how many pairs exceed the cap.

    The Kelvin quadrature holds its per-panel cycle budget only while the count stays under
    ``greens.MAX_CYCLES``; beyond that the grid runs out of panels and the result degrades
    with nothing to signal it.  The count grows as |Y|/|Z|, so the binding pairs are those
    straddling the hull just below the free surface -- exactly the pairs a waterline-fitted
    mesh creates.  Reported by every solve and enforced by :func:`influence_matrix`.
    """
    c = mesh.centroids()
    X = k0 * (c[:, None, 0] - c[None, :, 0])
    Y = k0 * (c[:, None, 1] - c[None, :, 1])
    Z = np.minimum(k0 * (c[:, None, 2] + c[None, :, 2]), -1e-9)
    n = oscillation_count(X, Y, Z)
    return float(n.max()), int((n > MAX_CYCLES).sum())


def influence_matrix(mesh: TriMesh, k0: float, wave: bool = True, order: int = 1,
                     kernel_refine: float = 1.0) -> np.ndarray:
    """Full influence matrix for the Neumann-Kelvin problem.

    The diagonal of the direct Rankine term is forced to 1/2, the exact jump; the image and
    wave terms have no singular diagonal because the image sits 2|z| away and the wave part
    is bounded for z < 0.

    Refuses a mesh that leaves the Kelvin quadrature's validity envelope rather than
    returning a plausible number from a silently coarsened grid.
    """
    if wave:
        worst, over = check_envelope(mesh, k0)
        if over:
            raise ValueError(
                f"{over} panel pairs carry more than {MAX_CYCLES:.0f} wave cycles "
                f"(worst {worst:.0f}), outside the Kelvin quadrature's validity envelope. "
                "The count grows as |y_i - y_j| / |z_i + z_j|, so the cause is panels too "
                "close to the free surface on opposite sides of the hull. Move the topmost "
                "row down -- Hull.waterline_fitted_mesh(..., first_depth=...) -- or lower "
                "the speed. See docs/formulation.md section 15."
            )
    a = rankine_with_image_matrix(mesh, collocated=True)
    if wave:
        a = a + wave_influence_matrix(mesh, k0, order=order,
                                      kernel_refine=kernel_refine)
    return a


def velocity_matrix(mesh: TriMesh, k0: float, order: int = 1,
                    kernel_refine: float = 1.0, chunk: int = 4096) -> np.ndarray:
    """grad(phi) at each panel centroid from unit source density on each panel, (n, m, 3).

    The Rankine part is the analytic panel velocity minus its image's.  On the diagonal the
    in-plane components are finite and the formula gives them, but the normal component has
    the same ambiguity as in :func:`rankine_with_image_matrix`, and the fluid-side limit
    +1/2 is imposed.  The wave part has no singular diagonal.
    """
    tri = mesh.triangles()
    centroids = mesh.centroids()
    normals = mesh.unit_normals()

    v = source_velocity(tri, centroids)                                  # (n, m, 3)
    idx = np.arange(mesh.n_faces)
    normal_component = np.einsum("ni,ni->n", v[idx, idx, :], normals)
    v[idx, idx, :] += (0.5 - normal_component)[:, None] * normals
    v = v - source_velocity(_mirrored_in_z(tri), centroids)

    if order <= 1:
        q_pts = centroids[:, None, :]
        q_w = mesh.areas()[:, None]
    else:
        q_pts, q_w = panel_quadrature(mesh, order)
    flat_q = q_pts.reshape(-1, 3)
    flat_w = q_w.reshape(-1)
    n_src, n_q = q_pts.shape[:2]
    src_index = np.repeat(np.arange(n_src), n_q)

    stride = max(1, chunk // max(flat_q.shape[0], 1))
    for start in range(0, mesh.n_faces, stride):
        rows = slice(start, start + stride)
        fp = centroids[rows]
        X = k0 * (fp[:, None, 0] - flat_q[None, :, 0])
        Y = k0 * (fp[:, None, 1] - flat_q[None, :, 1])
        Z = np.minimum(k0 * (fp[:, None, 2] + flat_q[None, :, 2]), -1e-9)
        # grad of k0 g_w is k0^2 times the gradient with respect to (X, Y, Z).
        g = wave_part_gradient(X, Y, Z, refine=kernel_refine) * k0 ** 2
        g = g * flat_w[None, :, None]
        if n_q == 1:
            v[rows] += g
        else:
            acc = np.zeros((fp.shape[0], n_src, 3))
            np.add.at(acc.transpose(1, 0, 2), src_index, g.transpose(1, 0, 2))
            v[rows] += acc
    return v


def x_velocity_matrix(mesh: TriMesh, k0: float, order: int = 1,
                      kernel_refine: float = 1.0, chunk: int = 4096) -> np.ndarray:
    """The x-component of :func:`velocity_matrix`, kept for the linear pressure route."""
    return velocity_matrix(mesh, k0, order=order, kernel_refine=kernel_refine,
                           chunk=chunk)[:, :, 0]


def pressure_resistance(mesh: TriMesh, sigma: np.ndarray, speed: float, k0: float,
                        rho: float = 1000.0, order: int = 1,
                        kernel_refine: float = 1.0, linear_only: bool = False,
                        gradient: np.ndarray | None = None) -> float:
    """Wave resistance by integrating the Bernoulli pressure over the wetted hull.

    Steady Bernoulli with Phi = -u x + phi gives, dropping the constant,

        p = rho u phi_x - rho g z - (1/2) rho |grad phi|^2 .

    The hydrostatic term carries no x-force: over the wetted surface closed by the z = 0
    waterplane the divergence theorem gives the integral of z n_x as zero, and the lid sits
    at z = 0.  Measured on a Sysser 01 mesh it comes to 6e-5 of its own scale, so the mesh
    honours the identity.

    Pressure acts against the outward normal, so the force on the body is minus the integral
    of p n dS with n the into-the-fluid normal used throughout.  The body advances in +x
    while the onset flow runs in -x, so the resistance is minus that x-force and the two
    signs cancel:

        R_w = integral over S of [ rho u phi_x - (1/2) rho |grad phi|^2 ] n_x dS .

    **The quadratic term is not optional.**  It is smaller than u phi_x by one order in
    slenderness, so it is negligible for a thin hull and *not* for anything else: on a thin
    Wigley hull at B/L = 0.02 keeping only the linear term reproduced the Michell oracle to
    4 per cent, but on a submerged sphere, where |grad phi| is O(u) on the body, it left the
    pressure route 68 per cent below the far-field value, and on Sysser 01 a factor of three
    below.  Dropping it was the first version's error, and the thin-hull oracle could not
    see it.  ``linear_only=True`` reproduces that version for comparison.

    Two sign conventions had to be got right here and only the second was obvious: the
    pressure acts against the outward normal, and the resistance opposes the motion.  The
    first version had the product of the two inverted, returning the correct magnitude with
    the wrong sign, which the Wigley oracle caught at once.

    ``gradient`` accepts a matrix from :func:`velocity_matrix` so that several densities, or
    the linear and full forms, share one assembly; it is the expensive part.
    """
    grad = (velocity_matrix(mesh, k0, order=order, kernel_refine=kernel_refine)
            if gradient is None else gradient)
    velocity = np.einsum("nmi,m->ni", grad, sigma)
    normal_x = mesh.unit_normals()[:, 0]
    area = mesh.areas()
    integrand = rho * speed * velocity[:, 0]
    if not linear_only:
        integrand -= 0.5 * rho * np.einsum("ni,ni->n", velocity, velocity)
    return float(np.sum(integrand * normal_x * area))


def _body_condition_residual(mesh: TriMesh, sigma: np.ndarray, k0: float, speed: float,
                             n_check: int = 48, seed: int = 0,
                             kernel_refine: float = 1.0) -> tuple[float, float]:
    """Residual of d(phi)/dn = u n_x at points that are NOT collocation points.

    Checking only at the collocation points is circular: the solve makes the residual
    vanish there by construction.  The check points are edge midpoints of randomly chosen
    panels, pushed a little into the fluid so the Rankine self-influence stays finite.
    """
    rng = np.random.default_rng(seed)
    picks = rng.choice(mesh.n_faces, size=min(n_check, mesh.n_faces), replace=False)
    tri = mesh.triangles()[picks]
    normals = mesh.unit_normals()[picks]
    radius = np.sqrt(mesh.areas()[picks])[:, None]
    # Edge midpoint, nudged along the outward normal by a fraction of the panel size.
    pts = 0.5 * (tri[:, 0] + tri[:, 1]) + 0.05 * radius * normals

    rankine = rankine_with_image_matrix(mesh, field_points=pts, field_normals=normals)
    wave = wave_influence_matrix(mesh, k0, order=1, field_points=pts, field_normals=normals,
                                 kernel_refine=kernel_refine)
    predicted = (rankine + wave) @ sigma
    target = speed * normals[:, 0]
    residual = (predicted - target) / speed
    return float(np.sqrt(np.mean(residual ** 2))), float(np.abs(residual).max())


def solve_nk(mesh: TriMesh, speed: float, length: float, rho: float = 1000.0,
             gravity: float = 9.80665, order: int = 1, spectrum_order: int = 4,
             rtol: float = 1e-4, check_points: int = 48,
             max_condition: float = 1e8, kernel_refine: float = 1.0,
             lambda_cap: float | None = None) -> NKResult:
    """Solve the Neumann-Kelvin problem on a mesh and report the wave resistance."""
    k0 = gravity / speed ** 2
    a = influence_matrix(mesh, k0, wave=True, order=order, kernel_refine=kernel_refine)
    condition = float(np.linalg.cond(a))
    if condition > max_condition:
        raise ValueError(
            f"influence matrix is numerically singular: cond = {condition:.3e} exceeds "
            f"{max_condition:.1e}. The usual cause is coincident panels -- most often a "
            "panel lying in the mirror plane, duplicated by mirroring -- or a body so thin "
            "that opposite faces are far closer than a panel is wide. A linear solver will "
            "return an answer for such a system without complaining, so this is raised "
            "rather than reported."
        )
    rhs = speed * mesh.unit_normals()[:, 0]
    sigma = np.linalg.solve(a, rhs)

    pts, w = panel_quadrature(mesh, spectrum_order)
    integral, diag = wave_resistance_integral(
        pts.reshape(-1, 3), (w * sigma[:, None]).reshape(-1), k0, length,
        rtol=rtol, lambda_cap=lambda_cap, return_diagnostics=True)
    resistance = resistance_constant(rho, gravity, speed) * integral

    rms, worst = _body_condition_residual(mesh, sigma, k0, speed, n_check=check_points,
                                          kernel_refine=kernel_refine)
    # Net source strength.  A closed body in a stream emits none, and the free surface can
    # carry only a little, so this is a direct measure of how far the discrete solution is
    # from the continuous one -- and the far-field amplitude near lambda = 1, which is where
    # the resistance integrand is largest, is exactly what a spurious net source corrupts.
    area = mesh.areas()
    flux = float(abs(np.sum(sigma * area)) / (speed * area.sum()))
    notes = []
    if flux > 0.01:
        notes.append(f"net source flux is {flux:.1%} of u S; the far-field resistance is "
                     "not converged at this panel count -- compare the pressure route")
    if order <= 1:
        notes.append("wave influences use the centroid rule over the source panel; "
                     "raise `order` and compare to bound that choice")
    if not diag["converged"]:
        notes.append("the wave-resistance integral did not certify its tail")
    if diag.get("capped") and diag.get("beyond_cap", 0.0) > 0.02 * max(integral, 1e-30):
        share = diag["beyond_cap"] / (integral + diag["beyond_cap"])
        notes.append(f"{share:.0%} of the spectrum lies beyond lambda = "
                     f"{diag['lambda_cap']:.2f}, which this mesh cannot resolve, and is "
                     "excluded from the reported resistance")
    return NKResult(
        sigma=sigma, resistance=resistance, speed=speed,
        froude=speed / np.sqrt(gravity * length), k0=k0, n_panels=mesh.n_faces,
        condition_number=condition,
        body_residual_rms=rms, body_residual_max=worst,
        spectrum_blocks=int(diag["blocks"]), lambda_reached=float(diag["lambda_reached"]),
        spectrum_converged=bool(diag["converged"]),
        lambda_cap=float(diag.get("lambda_cap", float("inf"))),
        net_source_flux=flux, notes=notes,
    )
