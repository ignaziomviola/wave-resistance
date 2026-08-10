"""Analytic reference expressions used to verify the numerical solver."""

from __future__ import annotations

from typing import Union

import numpy as np


ArrayLike = Union[float, np.ndarray]


def _wigley_longitudinal_factor(beta: np.ndarray, beam_over_length: float) -> np.ndarray:
    """Integral of the Wigley longitudinal slope on ``X in [0, 1]``."""

    beta = np.asarray(beta, dtype=float)
    half = 0.5 * beta
    small = np.abs(beta) < 1.0e-3
    # 8 i b [sin(beta/2)/beta^2 - cos(beta/2)/(2 beta)].
    regular = 8.0j * beam_over_length * (
        np.sin(half) / np.where(beta == 0.0, 1.0, beta * beta)
        - 0.5 * np.cos(half) / np.where(beta == 0.0, 1.0, beta)
    )
    # Series of the bracketed expression through beta^7.
    series = 1.0j * beam_over_length * (
        beta / 3.0
        - beta**3 / 120.0
        + beta**5 / 13440.0
        - beta**7 / 2903040.0
    )
    centred = np.where(small, series, regular)
    return np.exp(-0.5j * beta) * centred


def _wigley_vertical_factor(alpha_tau: np.ndarray, draft_over_length: float) -> np.ndarray:
    """Integral of ``(1-(Z/T)^2) exp(-alpha Z)`` over the draft."""

    q = np.asarray(alpha_tau, dtype=float)
    small = np.abs(q) < 1.0e-3
    denominator = np.where(q == 0.0, 1.0, q**3)
    regular = (q * q - 2.0 + 2.0 * (q + 1.0) * np.exp(-q)) / denominator
    series = (
        2.0 / 3.0
        - q / 4.0
        + q**2 / 15.0
        - q**3 / 72.0
        + q**4 / 420.0
        - q**5 / 2880.0
        + q**6 / 22680.0
    )
    return draft_over_length * np.where(small, series, regular)


def wigley_amplitude(
    lambda_: ArrayLike,
    froude: float,
    beam_over_length: float = 0.1,
    draft_over_length: float = 0.0625,
) -> np.ndarray:
    """Closed-form Michell amplitude for the standard parabolic Wigley hull.

    The hull occupies ``0 <= X <= 1`` and ``0 <= Z <= T/L``.  A different
    longitudinal origin changes only the complex phase, not resistance.
    """

    lam = np.asarray(lambda_, dtype=float)
    if not np.isfinite(froude) or froude <= 0.0:
        raise ValueError("froude must be finite and positive")
    if beam_over_length <= 0.0 or draft_over_length <= 0.0:
        raise ValueError("Wigley dimensions must be positive")
    beta = lam / (froude * froude)
    alpha_tau = lam * lam * draft_over_length / (froude * froude)
    return _wigley_longitudinal_factor(beta, beam_over_length) * _wigley_vertical_factor(
        alpha_tau, draft_over_length
    )
