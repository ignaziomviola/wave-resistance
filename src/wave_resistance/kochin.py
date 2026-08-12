"""Kochin far-field wave integral for a 3-D Rankine source distribution."""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

_trapezoid = getattr(np, "trapezoid", np.trapz)


def kochin_wave_coefficient(
    panel_centroids: np.ndarray,
    panel_areas: np.ndarray,
    source_strengths: np.ndarray,
    froude_number: float,
    wetted_area_over_l2: float,
    *,
    integration_points: int = 6001,
    t_max: float = 40.0,
    chunk_size: int = 512,
) -> Tuple[float, float]:
    """Return area-normalized ``C_W`` and a resolved-tail indicator.

    The half-hull source distribution is reflected analytically through the
    cosine transverse phase.  With ``lambda=sqrt(1+t**2)`` the amplitude is

    ``a = -sum(sigma*dS*exp(k*z-i*kx*x)*cos(ky*y))``.

    The normalization is chosen so that a slender Wigley hull converges to
    Michell's closed-form amplitude as the panel grid is refined.
    """

    centroids = np.asarray(panel_centroids, dtype=float)
    areas = np.asarray(panel_areas, dtype=float)
    strengths = np.asarray(source_strengths, dtype=float)
    if centroids.ndim != 2 or centroids.shape[1] != 3:
        raise ValueError("panel_centroids must have shape (n, 3)")
    if areas.shape != (centroids.shape[0],) or strengths.shape != areas.shape:
        raise ValueError("panel areas and source strengths must match centroids")
    if froude_number <= 0.0 or wetted_area_over_l2 <= 0.0:
        raise ValueError("Froude number and wetted area must be positive")
    if integration_points < 1001 or t_max <= 5.0:
        raise ValueError("Kochin quadrature is too small")

    coordinate = np.linspace(0.0, math.sqrt(t_max), integration_points)
    t = coordinate**2
    lam = np.sqrt(1.0 + t**2)
    inverse_fn2 = 1.0 / froude_number**2
    kx = lam * inverse_fn2
    total_k = lam**2 * inverse_fn2
    ky = lam * t * inverse_fn2
    weighted_strength = strengths * areas
    amplitude = np.empty(lam.size, dtype=complex)
    for start in range(0, lam.size, chunk_size):
        stop = min(start + chunk_size, lam.size)
        phase = np.exp(
            total_k[start:stop, None] * centroids[None, :, 2]
            - 1j * kx[start:stop, None] * centroids[None, :, 0]
        ) * np.cos(ky[start:stop, None] * centroids[None, :, 1])
        amplitude[start:stop] = -(phase @ weighted_strength)
    density = lam * np.abs(amplitude) ** 2
    integral = float(_trapezoid(density, t))
    tail_start = int(0.80 * t.size)
    tail = float(_trapezoid(density[tail_start:], t[tail_start:]))
    coefficient = 8.0 * integral / (
        math.pi * froude_number**4 * wetted_area_over_l2
    )
    tail_fraction = tail / max(integral, np.finfo(float).tiny)
    return float(max(0.0, coefficient)), float(tail_fraction)
