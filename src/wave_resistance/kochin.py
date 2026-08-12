"""Kochin far-field wave integral for a 3-D Rankine source distribution."""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

_trapezoid = getattr(np, "trapezoid", np.trapz)


def kochin_amplitude(
    panel_centroids: np.ndarray,
    panel_areas: np.ndarray,
    source_strengths: np.ndarray,
    froude_number: float,
    *,
    integration_points: int = 6001,
    t_max: float = 40.0,
    chunk_size: int = 512,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return transverse coordinate, ``lambda``, and complex Kochin amplitude.

    The input represents one half of a hull that is symmetric about ``y=0``.
    Reflection is included analytically through the cosine transverse phase.
    Coordinates and panel areas must be nondimensionalised by the reference
    length and its square, respectively.
    """

    centroids = np.asarray(panel_centroids, dtype=float)
    areas = np.asarray(panel_areas, dtype=float)
    strengths = np.asarray(source_strengths, dtype=float)
    if centroids.ndim != 2 or centroids.shape[1] != 3:
        raise ValueError("panel_centroids must have shape (n, 3)")
    if areas.shape != (centroids.shape[0],) or strengths.shape != areas.shape:
        raise ValueError("panel areas and source strengths must match centroids")
    if froude_number <= 0.0:
        raise ValueError("Froude number must be positive")
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
    return t, lam, amplitude


def kochin_wave_pattern(
    panel_centroids: np.ndarray,
    panel_areas: np.ndarray,
    source_strengths: np.ndarray,
    froude_number: float,
    x_over_l: np.ndarray,
    y_over_l: np.ndarray,
    *,
    integration_points: int = 1601,
    t_max: float = 12.0,
    taper_start_fraction: float = 0.80,
    normalize: bool = True,
    chunk_size: int = 512,
) -> np.ndarray:
    """Reconstruct a regular-grid far-field wave-pattern phase map.

    The returned field evaluates the radiating superposition

    ``Re int(lambda*a*exp(i*kx*x)*cos(ky*y)*w dt)``.

    Here ``w`` is a raised-cosine taper over the final part of the finite
    spectrum.  This preserves the Kelvin-wave dispersion, transverse phase,
    and bow/stern interference represented by the Kochin amplitude.  With
    ``normalize=True`` (the default), the result is divided by its maximum
    absolute value and is intended for wave-pattern visualisation rather than
    dimensional elevation or resistance evaluation.
    """

    x = np.asarray(x_over_l, dtype=float)
    y = np.asarray(y_over_l, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.size < 2 or y.size < 2:
        raise ValueError("x_over_l and y_over_l must be one-dimensional grids")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("wave-pattern coordinates must be finite")
    if not 0.0 < taper_start_fraction < 1.0:
        raise ValueError("taper_start_fraction must lie in (0, 1)")

    t, lam, amplitude = kochin_amplitude(
        panel_centroids,
        panel_areas,
        source_strengths,
        froude_number,
        integration_points=integration_points,
        t_max=t_max,
        chunk_size=chunk_size,
    )
    taper = np.ones_like(t)
    taper_start = taper_start_fraction * t_max
    tapered = t > taper_start
    taper[tapered] = 0.5 * (
        1.0
        + np.cos(math.pi * (t[tapered] - taper_start) / (t_max - taper_start))
    )

    weights = np.empty_like(t)
    weights[0] = 0.5 * (t[1] - t[0])
    weights[-1] = 0.5 * (t[-1] - t[-2])
    weights[1:-1] = 0.5 * (t[2:] - t[:-2])
    inverse_fn2 = 1.0 / froude_number**2
    kx = lam * inverse_fn2
    ky = lam * t * inverse_fn2
    longitudinal_phase = np.exp(1j * kx[:, None] * x[None, :])
    transverse_phase = np.cos(ky[:, None] * y[None, :])
    spectral_weight = weights * taper * lam * amplitude
    field = np.real(
        transverse_phase.T @ (spectral_weight[:, None] * longitudinal_phase)
    )
    if normalize:
        maximum = float(np.max(np.abs(field)))
        if maximum > 0.0:
            field = field / maximum
    return field


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

    if froude_number <= 0.0 or wetted_area_over_l2 <= 0.0:
        raise ValueError("Froude number and wetted area must be positive")
    t, lam, amplitude = kochin_amplitude(
        panel_centroids,
        panel_areas,
        source_strengths,
        froude_number,
        integration_points=integration_points,
        t_max=t_max,
        chunk_size=chunk_size,
    )
    density = lam * np.abs(amplitude) ** 2
    integral = float(_trapezoid(density, t))
    tail_start = int(0.80 * t.size)
    tail = float(_trapezoid(density[tail_start:], t[tail_start:]))
    coefficient = 8.0 * integral / (
        math.pi * froude_number**4 * wetted_area_over_l2
    )
    tail_fraction = tail / max(integral, np.finfo(float).tiny)
    return float(max(0.0, coefficient)), float(tail_fraction)
