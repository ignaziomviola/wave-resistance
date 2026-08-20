"""The parabolic Wigley hull: analytic geometry, exact hydrostatics, closed-form Michell amplitude.

This module exists only as a verification oracle.  There is no general thin-ship solver
in this package; the Michell functional appears here solely because for this one hull the
double integral separates and can be written in closed form, which pins the normalisation
of the far-field machinery without any quadrature error of its own.

Hull, with z measured upward from the undisturbed waterline:

    y = +/- (B/2) (1 - (2x/L)^2) (1 - (z/T)^2),   x in [-L/2, L/2],  z in [-T, 0]

Nondimensionalising every coordinate by L (X = x/L, Z = -z/L downward, Y = y/L) the
Michell amplitude separates as

    a(lambda) = (b/2) * G1(beta) * G2(alpha),   beta = lambda/Fn^2,  alpha = lambda^2/Fn^2

    G1(beta) = integral over X in [-1/2, 1/2] of (-8X) exp(-i beta X) dX
             = 16i [ sin(beta/2) - (beta/2) cos(beta/2) ] / beta^2
    G2(alpha) = integral over Z in [0, tau] of (1 - (Z/tau)^2) exp(-alpha Z) dZ
             = tau [ q^2 - 2 + 2(q+1) exp(-q) ] / q^3,   q = alpha * tau

with b = B/L and tau = T/L.  Both closed forms lose relative precision by cancellation as
their arguments go to zero, so each carries the Taylor series that replaces it there.
"""

from __future__ import annotations

from math import factorial

import numpy as np

from .hull import TriMesh

__all__ = [
    "wigley_offsets", "wigley_mesh", "wigley_exact_hydrostatics",
    "wigley_michell_amplitude", "michell_resistance_from_integral",
]

def _series_g2(n_terms: int = 22) -> np.ndarray:
    """Coefficients of (q^2 - 2 + 2(q+1)e^{-q})/q^3 = sum_j c_j q^j.

    Derived rather than transcribed: the q^m coefficient of 2(q+1)e^{-q} is
    2(-1)^m (1-m)/m!, so with j = m - 3 the coefficient is 2(-1)^j (j+2)/(j+3)!.
    """
    j = np.arange(n_terms)
    return np.array([2.0 * (-1.0) ** k * (k + 2) / factorial(k + 3) for k in j], dtype=float)


def _series_g1(n_terms: int = 12) -> np.ndarray:
    """Coefficients of 16i[sin(b) - b cos(b)]/(2b)^2 = 4i sum_k d_k b^(2k-1).

    sin(b) - b cos(b) = sum over k >= 1 of (-1)^k (-2k)/(2k+1)! b^(2k+1), so
    d_k = (-1)^k (-2k)/(2k+1)! = (-1)^(k+1) 2k/(2k+1)!.
    """
    k = np.arange(1, n_terms + 1)
    return np.array([(-1.0) ** (n + 1) * 2.0 * n / factorial(2 * n + 1) for n in k], dtype=float)


_G2_SERIES = _series_g2()
_G1_SERIES = _series_g1()
# Crossovers sit where the closed form has not yet lost digits to cancellation and the
# series has already converged to machine precision.
_G1_SMALL = 0.5
_G2_SMALL = 0.5


def wigley_offsets(x: np.ndarray, z: np.ndarray, length: float, beam: float, draught: float) -> np.ndarray:
    """Starboard half-breadth y(x, z) >= 0, zero outside the hull."""
    xn = 2.0 * np.asarray(x, dtype=float) / length
    zn = np.asarray(z, dtype=float) / draught
    y = 0.5 * beam * (1.0 - xn ** 2) * (1.0 - zn ** 2)
    return np.where((np.abs(xn) <= 1.0) & (zn >= -1.0) & (zn <= 0.0), y, 0.0)


def wigley_mesh(length: float = 1.0, beam: float | None = None, draught: float | None = None,
                n_x: int = 80, n_z: int = 40) -> TriMesh:
    """Full (both sides) triangulated Wigley hull with outward normals, keel closed at z = -T."""
    beam = 0.1 * length if beam is None else beam
    draught = beam / 1.6 if draught is None else draught
    x = np.linspace(-0.5 * length, 0.5 * length, n_x + 1)
    z = np.linspace(-draught, 0.0, n_z + 1)
    xx, zz = np.meshgrid(x, z, indexing="ij")
    yy = wigley_offsets(xx, zz, length, beam, draught)
    pts = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
    idx = np.arange((n_x + 1) * (n_z + 1)).reshape(n_x + 1, n_z + 1)
    a, b = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel()
    c, d = idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()
    faces = np.vstack([np.column_stack([a, b, c]), np.column_stack([a, c, d])])
    star = TriMesh(pts, faces).dropped_degenerate()
    # Panels lying wholly in y = 0 would be duplicated by the mirror, giving coincident
    # panels with opposite normals and an exactly singular influence matrix.  At the
    # stern-keel corner the grid produces exactly one such triangle.
    star = star.without_panels_in_plane(axis=1, level=0.0)
    # Orient the starboard sheet outward (+y), then mirror.
    if float(np.einsum("ij,ij->", star.centroids(), star.area_normals())) < 0.0:
        star = star.flipped()
    return star.joined(star.mirrored_y())


def wigley_exact_hydrostatics(length: float = 1.0, beam: float | None = None,
                              draught: float | None = None) -> dict[str, float]:
    """Closed-form hydrostatics: Cb = 4/9 and Cwp = Cm = Cp = 2/3, independent of L, B, T."""
    beam = 0.1 * length if beam is None else beam
    draught = beam / 1.6 if draught is None else draught
    return {
        "volume": 4.0 * beam * length * draught / 9.0,
        "waterplane_area": 2.0 * beam * length / 3.0,
        "midship_area": 2.0 * beam * draught / 3.0,
        "cb": 4.0 / 9.0, "cwp": 2.0 / 3.0, "cm": 2.0 / 3.0, "cp": 2.0 / 3.0,
        "lcb": 0.0, "vcb": -3.0 * draught / 8.0,
        "lwl": length, "bwl": beam, "draught": draught,
    }


def _g1(beta: np.ndarray) -> np.ndarray:
    """16i [sin(b) - b cos(b)] / beta^2 with b = beta/2, series-continued near zero."""
    beta = np.asarray(beta, dtype=float)
    b = 0.5 * beta
    small = np.abs(b) < _G1_SMALL
    out = np.empty(beta.shape, dtype=complex)
    with np.errstate(invalid="ignore", divide="ignore"):
        big = 16.0j * (np.sin(b) - b * np.cos(b)) / np.where(beta != 0.0, beta ** 2, 1.0)
    # 16i/beta^2 * sum = 4i * (b/3 - b^3/30 + ...)
    powers = b[..., None] ** (2 * np.arange(1, _G1_SERIES.size + 1) - 1)
    ser = 4.0j * (powers * _G1_SERIES).sum(axis=-1)
    out = np.where(small, ser, big)
    return out


def _g2(alpha: np.ndarray, tau: float) -> np.ndarray:
    """tau [q^2 - 2 + 2(q+1)e^{-q}] / q^3 with q = alpha*tau, series-continued near zero."""
    q = np.asarray(alpha, dtype=float) * tau
    small = np.abs(q) < _G2_SMALL
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        qs = np.where(q != 0.0, q, 1.0)
        big = tau * (qs ** 2 - 2.0 + 2.0 * (qs + 1.0) * np.exp(-qs)) / qs ** 3
    ser = tau * (q[..., None] ** np.arange(_G2_SERIES.size) * _G2_SERIES).sum(axis=-1)
    return np.where(small, ser, big)


def wigley_michell_amplitude(lam, froude: float, beam_over_length: float = 0.1,
                             draught_over_length: float = 0.0625) -> np.ndarray:
    """Closed-form Michell amplitude a(lambda) for the parabolic Wigley hull."""
    lam = np.asarray(lam, dtype=float)
    if np.any(lam < 1.0):
        raise ValueError("lambda must be >= 1 (lambda = sec theta)")
    if froude <= 0.0:
        raise ValueError("froude must be positive")
    inv_fn2 = 1.0 / froude ** 2
    beta = lam * inv_fn2
    alpha = lam * lam * inv_fn2
    return 0.5 * beam_over_length * _g1(beta) * _g2(alpha, draught_over_length)


def michell_resistance_from_integral(integral: float, froude: float, length: float,
                                     rho: float = 1000.0, gravity: float = 9.80665) -> float:
    """R_W = 4 rho g L^3 I / (pi Fn^2), with I = integral over t of sqrt(1+t^2) |a|^2."""
    return 4.0 * rho * gravity * length ** 3 * integral / (np.pi * froude ** 2)
