"""Free-wave amplitudes and the wave-resistance integral.

From ``docs/formulation.md``, the amplitude of the wave travelling at angle theta, with
lambda = sec(theta), is

    A_pm(lambda) = integral over S of sigma exp(k0 lambda^2 zeta)
                     exp(-i k0 lambda (xi +/- sqrt(lambda^2 - 1) eta)) dS,

and both branches must be kept, because the transverse phase carries sin(theta) and so
changes sign with theta.  For a hull symmetric about y = 0 at zero heel A_minus is the
conjugate of A_plus and the two coincide in magnitude, but nothing here assumes that.

The resistance is

    R_W = C integral over lambda from 1 to infinity of
              (1/2)(|A_plus|^2 + |A_minus|^2) lambda^2 (lambda^2 - 1)^(-1/2) d(lambda),
    C   = rho g^2 / (pi u^4).

C is derived rather than fitted, and the derivation turns on which density a thin hull
carries.  For y = +/- f the two faces coalesce onto y = 0, and the centreplane sheet that
reproduces phi_y = -u f_x has strength m = -2 u f_x.  Each face carries sigma and the two
add, so 2 sigma = m and

    sigma -> u n_x        (n_x = -f_x on the starboard face, n into the fluid).

That is *not* the zeroth iterate sigma = 2 u n_x of the integral equation: for a thin body
the two faces' mutual influence is O(1), so dropping the integral operator is not the
thin-ship limit.  With sigma = u n_x and n_x dS = -f_x dx dz per face, A tends to
-2 u A_M with A_M = L^2 a Michell's amplitude, hence |A|^2 -> 4 u^2 |A_M|^2.  Matching
Michell's R_w = (4 rho g^2 / pi u^2) integral of |A_M|^2 lambda^2 (lambda^2-1)^(-1/2)
d(lambda) gives 4 u^2 C = 4 rho g^2/(pi u^2), so C = rho g^2/(pi u^4).

Imposing sigma = u n_x on a thin Wigley hull and pushing it through this module reproduces
the analytic Michell resistance to 0.12 % at 1630 panels and 0.84 % at 558 (B/L = 0.02,
Fn = 0.30), converging under refinement at Fn = 0.30 and 0.40 and at B/L = 0.02 and 0.005.
That check isolates the constant and the kernel from the linear solve.

The substitution lambda = sqrt(1 + t^2) removes the endpoint singularity, and the t grid is
graded uniform in t^2 because the phase k0 lambda^2 L grows quadratically in t.
"""

from __future__ import annotations

import functools

import numpy as np

from .hull import TriMesh

__all__ = ["panel_quadrature", "free_wave_amplitudes", "wave_resistance_integral",
           "wave_resistance", "resistance_constant", "resolved_lambda",
           "mesh_resolved_lambda"]



def resistance_constant(rho: float, gravity: float, speed: float) -> float:
    """C = rho g^2 / (pi u^4)."""
    return rho * gravity ** 2 / (np.pi * speed ** 4)


@functools.lru_cache(maxsize=8)
def _duffy(order: int) -> tuple[np.ndarray, np.ndarray]:
    n, w = np.polynomial.legendre.leggauss(order)
    a = 0.5 * (n + 1.0)
    wa = 0.5 * w
    u, v = np.meshgrid(a, a, indexing="ij")
    wu, wv = np.meshgrid(wa, wa, indexing="ij")
    return (np.stack([u.ravel(), (v * (1.0 - u)).ravel()], axis=1),
            (wu * wv * (1.0 - u)).ravel())


def panel_quadrature(mesh: TriMesh, order: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """Quadrature points and weights over every panel.

    Returns points (m, q, 3) and weights (m, q) that sum, per panel, to the panel area.
    The amplitude integrand oscillates across a panel once k0 lambda times the panel size
    exceeds one, which happens well inside the useful lambda range, so a genuine panel
    integral is needed here and not just the centroid.
    """
    tri = mesh.triangles()
    bary, w = _duffy(order)
    e0, e1 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    area = 0.5 * np.linalg.norm(np.cross(e0, e1), axis=1)
    pts = (tri[:, 0:1, :] + bary[None, :, 0, None] * e0[:, None, :]
           + bary[None, :, 1, None] * e1[:, None, :])
    return pts, (w[None, :] * 2.0) * area[:, None]


def free_wave_amplitudes(lam: np.ndarray, points: np.ndarray, weights: np.ndarray,
                         k0: float) -> tuple[np.ndarray, np.ndarray]:
    """A_plus and A_minus at each lambda.

    ``points`` is (n_q, 3) and ``weights`` is (n_q,), the latter already carrying the source
    density and the panel area.  Both outputs have shape ``lam.shape``.
    """
    lam = np.asarray(lam, dtype=float)
    xi, eta, zeta = points[:, 0], points[:, 1], points[:, 2]
    root = np.sqrt(np.maximum(lam * lam - 1.0, 0.0))
    decay = np.exp(k0 * lam[:, None] ** 2 * zeta[None, :])
    base = k0 * lam[:, None] * xi[None, :]
    cross = k0 * lam[:, None] * root[:, None] * eta[None, :]
    common = weights[None, :] * decay
    a_plus = (common * np.exp(-1j * (base + cross))).sum(axis=1)
    a_minus = (common * np.exp(-1j * (base - cross))).sum(axis=1)
    return a_plus, a_minus


def _phase_rate(k0: float, length: float, beam: float, lam: float) -> float:
    """Rate of change of the amplitude's phase with respect to lambda.

    A carries k0 lambda xi from the longitudinal offset and k0 lambda sqrt(lambda^2 - 1)
    eta from the transverse one, so the rate is about k0 (length + lambda * beam).
    """
    return k0 * (length + lam * beam)


def resolved_lambda(k0: float, vertical_size: float,
                    horizontal_size: float | None = None) -> float:
    """Largest lambda whose free wave the mesh can carry.

    The amplitude integrand is sigma exp(k0 lambda^2 zeta) exp(-i k0 lambda xi), so a panel
    has to resolve two different things and they scale differently in lambda.

    *Vertically*, the factor exp(k0 lambda^2 zeta) decays over a depth 1/(k0 lambda^2), and
    a panel of vertical extent dz resolves that only while k0 lambda^2 dz is below about pi:

        lambda <= sqrt(pi / (k0 dz)) .

    *Horizontally*, the phase advances at k0 lambda per unit length, so a panel of extent dx
    resolves it while k0 lambda dx is below about pi:

        lambda <= pi / (k0 dx) .

    The vertical limit is the binding one at every mesh size of interest, because it grows
    only as the square root.  Passing ``horizontal_size`` applies both and returns the
    smaller.

    Beyond the limit the amplitude integral is still *evaluated* correctly -- the phase is
    integrated over each panel exactly -- but what it integrates is a source density the mesh
    cannot represent at that scale, and near the waterline that density is the least
    accurate part of the solution.  The contribution is panel noise rather than waves.  On a
    280-panel Sysser 01 mesh at Fn = 0.30 the marching integral ran to lambda = 34, a wave
    0.8 mm long, on panels 48 mm across.

    That tail is nevertheless **not** where a coarse mesh loses its accuracy, and it is worth
    saying so because it is the obvious suspect.  Measured on that mesh, everything beyond
    lambda = 3.1 is 5 per cent of the integral, while the far-field and pressure routes
    disagree by a factor of three.  The integrand carries lambda^2/sqrt(lambda^2-1) against
    an amplitude that falls off algebraically, so the low-lambda end dominates -- and that is
    the end a spurious net source strength contaminates.  Cap the tail because integrating
    unrepresentable wavelengths is not defensible, not because it buys accuracy.
    """
    limit = np.sqrt(np.pi / max(k0 * vertical_size, 1e-30))
    if horizontal_size is not None:
        limit = min(limit, np.pi / max(k0 * horizontal_size, 1e-30))
    return float(limit)


def mesh_resolved_lambda(mesh: TriMesh, k0: float) -> float:
    """:func:`resolved_lambda` with both extents taken from the mesh's coarsest panel.

    The largest vertical and horizontal panel extents are used, not the mean, so the limit
    is the one every panel satisfies rather than the one an average panel does.
    """
    tri = mesh.triangles()
    dz = float(np.ptp(tri[:, :, 2], axis=1).max())
    dx = float(np.ptp(tri[:, :, 0], axis=1).max())
    return resolved_lambda(k0, dz, dx)


def wave_resistance_integral(points: np.ndarray, weights: np.ndarray, k0: float,
                             length: float, refine: float = 1.0, rtol: float = 1e-4,
                             quiet_blocks: int = 4, max_blocks: int = 4000,
                             lambda_cap: float | None = None,
                             return_diagnostics: bool = False):
    """Integral of (1/2)(|A_+|^2 + |A_-|^2) lambda^2 (lambda^2-1)^(-1/2) d(lambda).

    Substituting lambda = sqrt(1 + t^2) removes the endpoint singularity and turns this
    into the integral over t from 0 to infinity of
    (1/2)(|A_+|^2 + |A_-|^2) sqrt(1 + t^2) dt.

    The upper limit is found by marching and certifying the tail, not by a damping bound.
    A damping bound is the obvious choice, since A carries exp(k0 lambda^2 zeta), but it is
    useless for a surface-piercing hull: quadrature points arbitrarily close to z = 0 are
    damped arbitrarily slowly, so an envelope bound on |A| pushed the cutoff to lambda =
    417 on a thin Wigley hull and made the integral cost 771 s.  The envelope ignores
    cancellation.  What actually happens is that A falls off algebraically -- for the
    Wigley hull as lambda^-3, so the integrand goes as lambda^-4 -- and the honest test is
    whether the accumulated tail has stopped mattering.  Marching stops after
    ``quiet_blocks`` consecutive blocks each contributing less than ``rtol`` of the running
    total.
    """
    beam = float(np.ptp(points[:, 1]))
    nodes, gw = np.polynomial.legendre.leggauss(16)
    total = 0.0
    beyond = 0.0
    t_lo = 0.0
    quiet = 0
    blocks = 0
    lam_reached = 1.0
    t_cap = np.inf if lambda_cap is None else float(np.sqrt(max(lambda_cap ** 2 - 1.0, 0.0)))
    capped = False
    while blocks < max_blocks:
        lam = np.sqrt(1.0 + t_lo * t_lo)
        rate = max(_phase_rate(k0, length, beam, lam), 1e-12)
        # One block spans a few cycles of interference phase; refine tightens it.
        width = 2.0 * np.pi * 3.0 / (rate * max(refine, 1e-6))
        width = min(max(width, 1e-6), 50.0)
        n_sub = max(2, int(np.ceil(3.0 * max(refine, 1.0))))
        sub = np.linspace(t_lo, t_lo + width, n_sub + 1)
        mid = 0.5 * (sub[:-1] + sub[1:])
        half = 0.5 * (sub[1:] - sub[:-1])
        t = (mid[:, None] + half[:, None] * nodes[None, :]).ravel()
        w = (half[:, None] * gw[None, :]).ravel()
        lam_nodes = np.sqrt(1.0 + t * t)
        a_p, a_m = free_wave_amplitudes(lam_nodes, points, weights, k0)
        block = float(np.dot(w, 0.5 * (np.abs(a_p) ** 2 + np.abs(a_m) ** 2) * lam_nodes))
        if capped:
            beyond += block
        else:
            total += block
        blocks += 1
        t_lo += width
        lam_reached = float(np.sqrt(1.0 + t_lo * t_lo))
        if t_lo >= t_cap:
            capped = True
        reference = total + beyond
        if reference > 0.0 and abs(block) < rtol * reference:
            quiet += 1
            if quiet >= quiet_blocks:
                break
        else:
            quiet = 0
    if return_diagnostics:
        return total, {"blocks": blocks, "lambda_reached": lam_reached,
                       "converged": quiet >= quiet_blocks, "capped": capped,
                       "beyond_cap": beyond,
                       "lambda_cap": np.inf if lambda_cap is None else float(lambda_cap)}
    return total


def wave_resistance(mesh: TriMesh, sigma: np.ndarray, speed: float, length: float,
                    rho: float = 1000.0, gravity: float = 9.80665, order: int = 6,
                    refine: float = 1.0) -> float:
    """Wave resistance from a source density on a mesh, via the far-field amplitudes."""
    k0 = gravity / speed ** 2
    pts, w = panel_quadrature(mesh, order)
    flat_pts = pts.reshape(-1, 3)
    flat_w = (w * sigma[:, None]).reshape(-1)
    integral = wave_resistance_integral(flat_pts, flat_w, k0, length, refine)
    return resistance_constant(rho, gravity, speed) * integral
