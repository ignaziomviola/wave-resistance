"""Far-field wave-cut analysis following the ITTC Newman-Sharma variables."""

from __future__ import annotations

import math

import numpy as np

_trapezoid = getattr(np, "trapezoid", np.trapz)


def wave_pattern_coefficient(
    x_over_l: np.ndarray,
    elevation_over_l: np.ndarray,
    froude_number: float,
    *,
    omega_max: float = 8.0,
    omega_points: int = 320,
) -> float:
    """Estimate ``C_WP`` from one longitudinal far-field wave cut.

    Coordinates are converted to the ITTC wave-number scale ``K0 = g/U^2``.
    A Tukey-like cosine taper is applied because a numerical free-surface
    domain provides a finite cut.  The implementation uses
    ``u = omega*sqrt(omega**2-1)`` and integrates both signs of ``u``.
    """

    x = np.asarray(x_over_l, dtype=float)
    elevation = np.asarray(elevation_over_l, dtype=float)
    if x.ndim != 1 or elevation.shape != x.shape or x.size < 32:
        raise ValueError("wave cut arrays must be one-dimensional with at least 32 points")
    if not np.all(np.diff(x) > 0.0) or not np.all(np.isfinite(elevation)):
        raise ValueError("wave cut coordinates must be increasing and finite")
    if froude_number <= 0.0 or omega_max <= 1.0 or omega_points < 32:
        raise ValueError("invalid wave-pattern settings")

    k0_l = 1.0 / froude_number**2
    chi = k0_l * x
    xi = k0_l * elevation
    xi = xi - np.mean(xi)
    edge_count = max(2, int(0.12 * x.size))
    window = np.ones_like(xi)
    ramp = 0.5 * (1.0 - np.cos(np.linspace(0.0, math.pi, edge_count)))
    window[:edge_count] = ramp
    window[-edge_count:] = ramp[::-1]
    xi *= window

    omega = np.linspace(1.0 + 1.0e-6, omega_max, omega_points)
    phase = np.exp(1j * omega[:, None] * chi[None, :])
    transform = _trapezoid(xi[None, :] * phase, chi, axis=1)
    root = np.sqrt(omega**2 - 1.0)
    cosine_sine = root * transform
    denominator = 2.0 * omega**2 - 1.0
    fg_squared = 16.0 * np.abs(cosine_sine) ** 2 / denominator**2
    weight = denominator / (1.0 + denominator)
    du_domega = denominator / root
    coefficient = _trapezoid(fg_squared * weight * du_domega, omega) / (8.0 * math.pi)
    return float(max(0.0, coefficient))


def transverse_energy_flux_coefficient(
    y_over_l: np.ndarray,
    elevation_over_l: np.ndarray,
    froude_number: float,
    wetted_area_over_l2: float,
) -> float:
    """Estimate area-normalized wave resistance from a transverse wave cut.

    The spectral weight follows the steady deep-water dispersion relation.
    This gives an energy-flux cross-check that is independent of hull pressure
    integration, while retaining the finite-cut limitations of any numerical
    wave-pattern analysis.
    """

    y = np.asarray(y_over_l, dtype=float)
    elevation = np.asarray(elevation_over_l, dtype=float)
    if y.ndim != 1 or elevation.shape != y.shape or y.size < 32:
        raise ValueError("transverse cut arrays must contain at least 32 points")
    if not np.all(np.diff(y) > 0.0):
        raise ValueError("transverse cut coordinates must be increasing")
    if froude_number <= 0.0 or wetted_area_over_l2 <= 0.0:
        raise ValueError("Froude number and wetted area must be positive")
    spacing = float(np.mean(np.diff(y)))
    if not np.allclose(np.diff(y), spacing, rtol=1.0e-6, atol=1.0e-12):
        raise ValueError("transverse energy analysis requires a uniform cut")
    signal = elevation - np.mean(elevation)
    edge_count = max(2, int(0.10 * signal.size))
    window = np.ones_like(signal)
    ramp = 0.5 * (1.0 - np.cos(np.linspace(0.0, math.pi, edge_count)))
    window[:edge_count] = ramp
    window[-edge_count:] = ramp[::-1]
    transform = spacing * np.fft.rfft(signal * window)
    wave_number = 2.0 * math.pi * np.fft.rfftfreq(signal.size, d=spacing)
    transverse_parameter = wave_number * froude_number**2
    longitudinal_squared = 0.5 * (
        1.0 + np.sqrt(1.0 + 4.0 * transverse_parameter**2)
    )
    group_weight = 1.0 - 0.5 / longitudinal_squared
    spectral_integral = 2.0 * float(
        _trapezoid(np.abs(transform) ** 2 * group_weight, wave_number)
    )
    coefficient = spectral_integral / (
        2.0 * math.pi * froude_number**2 * wetted_area_over_l2
    )
    return float(max(0.0, coefficient))
