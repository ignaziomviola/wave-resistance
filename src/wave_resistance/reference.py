"""Independent thin-ship reference expressions for slender-hull verification."""

from __future__ import annotations

import numpy as np

_trapezoid = getattr(np, "trapezoid", np.trapz)


def _longitudinal_factor(beta: np.ndarray, beam_over_length: float) -> np.ndarray:
    beta = np.asarray(beta, dtype=float)
    half = 0.5 * beta
    safe = np.where(beta == 0.0, 1.0, beta)
    regular = 8.0j * beam_over_length * (
        np.sin(half) / safe**2 - 0.5 * np.cos(half) / safe
    )
    series = 1.0j * beam_over_length * (
        beta / 3.0
        - beta**3 / 120.0
        + beta**5 / 13440.0
        - beta**7 / 2903040.0
    )
    return np.where(np.abs(beta) < 1.0e-3, series, regular)


def _vertical_factor(q: np.ndarray, draft_over_length: float) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    safe = np.where(q == 0.0, 1.0, q)
    regular = (q**2 - 2.0 + 2.0 * (q + 1.0) * np.exp(-q)) / safe**3
    series = (
        2.0 / 3.0
        - q / 4.0
        + q**2 / 15.0
        - q**3 / 72.0
        + q**4 / 420.0
        - q**5 / 2880.0
    )
    return draft_over_length * np.where(np.abs(q) < 1.0e-3, series, regular)


def wigley_michell_amplitude(
    wave_number_parameter: np.ndarray,
    froude_number: float,
    beam_over_length: float = 0.10,
    draft_over_length: float = 0.0625,
) -> np.ndarray:
    """Closed-form Michell amplitude for a parabolic Wigley hull."""

    lam = np.asarray(wave_number_parameter, dtype=float)
    if froude_number <= 0.0 or beam_over_length <= 0.0 or draft_over_length <= 0.0:
        raise ValueError("Froude number and hull ratios must be positive")
    beta = lam / froude_number**2
    q = lam**2 * draft_over_length / froude_number**2
    return _longitudinal_factor(beta, beam_over_length) * _vertical_factor(
        q, draft_over_length
    )


def wigley_michell_coefficient(
    froude_number: float,
    wetted_area_over_l2: float,
    beam_over_length: float = 0.10,
    draft_over_length: float = 0.0625,
    integration_points: int = 20001,
    t_max: float = 60.0,
) -> float:
    """Return Michell ``C_W`` using a dense transformed outer integral."""

    if wetted_area_over_l2 <= 0.0 or integration_points < 1001 or t_max <= 1.0:
        raise ValueError("invalid Michell integration settings")
    # A quadratic grid resolves the endpoint and the oscillatory far tail.
    coordinate = np.linspace(0.0, np.sqrt(t_max), integration_points)
    t = coordinate**2
    lam = np.sqrt(1.0 + t**2)
    amplitude = wigley_michell_amplitude(
        lam,
        froude_number,
        beam_over_length=beam_over_length,
        draft_over_length=draft_over_length,
    )
    integral = float(_trapezoid(lam * np.abs(amplitude) ** 2, t))
    return 8.0 * integral / (
        np.pi * froude_number**4 * wetted_area_over_l2
    )
