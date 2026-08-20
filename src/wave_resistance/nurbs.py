"""Vectorised NURBS surface evaluation.

Basis functions follow the Cox-de Boor triangular recurrence (Piegl & Tiller,
*The NURBS Book*, algorithm A2.2), with first derivatives obtained from the
degree-(p-1) basis via

    dN_{i,p}/dt = p [ N_{i,p-1}/(t_{i+p} - t_i) - N_{i+1,p-1}/(t_{i+p+1} - t_{i+1}) ]

Surface derivatives use the quotient rule on the homogeneous (weighted) surface, which
keeps rational and polynomial patches on one code path.

All routines are vectorised over the query points; the only Python-level loops run over
the polynomial degree, which is small and fixed.
"""

from __future__ import annotations

import numpy as np

from .iges import NurbsSurface

__all__ = ["find_span", "basis_functions", "basis_and_derivative", "evaluate", "evaluate_derivatives", "normals"]


def find_span(knots: np.ndarray, degree: int, n_cp: int, t: np.ndarray) -> np.ndarray:
    """Index of the knot span containing each t, clamped to the valid range."""
    span = np.searchsorted(knots, t, side="right") - 1
    return np.clip(span, degree, n_cp - 1)


def basis_functions(knots: np.ndarray, degree: int, n_cp: int, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Non-zero basis values at each t.

    Returns ``(span, N)`` with ``N`` of shape ``t.shape + (degree + 1,)``; entry ``k``
    belongs to control point ``span - degree + k``.
    """
    t = np.asarray(t, dtype=float)
    span = find_span(knots, degree, n_cp, t)
    shape = t.shape + (degree + 1,)
    N = np.zeros(shape)
    N[..., 0] = 1.0
    left = np.zeros(shape)
    right = np.zeros(shape)
    for j in range(1, degree + 1):
        left[..., j] = t - knots[span + 1 - j]
        right[..., j] = knots[span + j] - t
        saved = np.zeros(t.shape)
        for r in range(j):
            denom = right[..., r + 1] + left[..., j - r]
            temp = np.where(denom != 0.0, N[..., r] / np.where(denom != 0.0, denom, 1.0), 0.0)
            N[..., r] = saved + right[..., r + 1] * temp
            saved = left[..., j - r] * temp
        N[..., j] = saved
    return span, N


def basis_and_derivative(
    knots: np.ndarray, degree: int, n_cp: int, t: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Basis values and first derivatives; ``(span, N, dN)`` shaped as in `basis_functions`."""
    t = np.asarray(t, dtype=float)
    span, N = basis_functions(knots, degree, n_cp, t)
    if degree == 0:
        return span, N, np.zeros_like(N)
    # Degree p-1 basis on the same span: p entries, for control points span-p+1 .. span.
    lower = np.zeros(t.shape + (degree,))
    lower[..., 0] = 1.0
    left = np.zeros(t.shape + (degree,))
    right = np.zeros(t.shape + (degree,))
    for j in range(1, degree):
        left[..., j] = t - knots[span + 1 - j]
        right[..., j] = knots[span + j] - t
        saved = np.zeros(t.shape)
        for r in range(j):
            denom = right[..., r + 1] + left[..., j - r]
            temp = np.where(denom != 0.0, lower[..., r] / np.where(denom != 0.0, denom, 1.0), 0.0)
            lower[..., r] = saved + right[..., r + 1] * temp
            saved = left[..., j - r] * temp
        lower[..., j] = saved

    dN = np.zeros_like(N)
    for k in range(degree + 1):
        i = span - degree + k
        term = np.zeros(t.shape)
        if k >= 1:
            den = knots[i + degree] - knots[i]
            term += np.where(den != 0.0, lower[..., k - 1] / np.where(den != 0.0, den, 1.0), 0.0)
        if k <= degree - 1:
            den = knots[i + degree + 1] - knots[i + 1]
            term -= np.where(den != 0.0, lower[..., k] / np.where(den != 0.0, den, 1.0), 0.0)
        dN[..., k] = degree * term
    return span, N, dN


def _gather(surface: NurbsSurface, span_u: np.ndarray, span_v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Weighted control points and weights for the active (degree_u+1) x (degree_v+1) block."""
    pu, pv = surface.degree_u, surface.degree_v
    iu = span_u[..., None] - pu + np.arange(pu + 1)          # (..., pu+1)
    iv = span_v[..., None] - pv + np.arange(pv + 1)          # (..., pv+1)
    w = surface.weights[iu[..., :, None], iv[..., None, :]]  # (..., pu+1, pv+1)
    p = surface.control_points[iu[..., :, None], iv[..., None, :], :]
    return p * w[..., None], w


def evaluate(surface: NurbsSurface, u, v) -> np.ndarray:
    """Surface points at matched parameter arrays; returns shape ``u.shape + (3,)``."""
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    u, v = np.broadcast_arrays(u, v)
    span_u, Nu = basis_functions(surface.knots_u, surface.degree_u, surface.n_u, u)
    span_v, Nv = basis_functions(surface.knots_v, surface.degree_v, surface.n_v, v)
    pw, w = _gather(surface, span_u, span_v)
    outer = Nu[..., :, None] * Nv[..., None, :]
    num = np.einsum("...ij,...ijk->...k", outer, pw)
    den = np.einsum("...ij,...ij->...", outer, w)
    return num / den[..., None]


def evaluate_derivatives(surface: NurbsSurface, u, v) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Point and first parametric derivatives ``(S, dS/du, dS/dv)``."""
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    u, v = np.broadcast_arrays(u, v)
    span_u, Nu, dNu = basis_and_derivative(surface.knots_u, surface.degree_u, surface.n_u, u)
    span_v, Nv, dNv = basis_and_derivative(surface.knots_v, surface.degree_v, surface.n_v, v)
    pw, w = _gather(surface, span_u, span_v)

    def homogeneous(bu, bv):
        outer = bu[..., :, None] * bv[..., None, :]
        return (
            np.einsum("...ij,...ijk->...k", outer, pw),
            np.einsum("...ij,...ij->...", outer, w),
        )

    a, aw = homogeneous(Nu, Nv)
    au, awu = homogeneous(dNu, Nv)
    av, awv = homogeneous(Nu, dNv)
    s = a / aw[..., None]
    ds_du = (au - s * awu[..., None]) / aw[..., None]
    ds_dv = (av - s * awv[..., None]) / aw[..., None]
    return s, ds_du, ds_dv


def normals(surface: NurbsSurface, u, v, normalise: bool = True) -> np.ndarray:
    """Parametric normal ``dS/du x dS/dv``; unit length unless ``normalise=False``."""
    _, du, dv = evaluate_derivatives(surface, u, v)
    n = np.cross(du, dv)
    if not normalise:
        return n
    mag = np.linalg.norm(n, axis=-1, keepdims=True)
    return np.divide(n, mag, out=np.zeros_like(n), where=mag > 0.0)
