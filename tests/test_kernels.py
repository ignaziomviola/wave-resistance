from __future__ import annotations

import numpy as np
import pytest

from wave_resistance.kernels import (
    bilinear_cell_amplitudes_direct,
    bilinear_cell_amplitudes_ibp,
    cancellation_ratio,
    compensated_sum,
    decaying_linear_basis_moments,
    michell_amplitude_direct,
    michell_amplitude_ibp,
    normalized_sinc,
    oscillatory_linear_basis_moments,
    sum_with_diagnostics,
)


def _gauss_rule(order: int = 256) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.legendre.leggauss(order)
    return 0.5 * (nodes + 1.0), 0.5 * weights


def _sample_hull(pointed: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # All three coordinates are divided by the same reference length L.
    x_over_l = np.array([-0.50, -0.27, 0.03, 0.31, 0.50])
    z_over_l = np.array([0.0, 0.018, 0.052, 0.10])
    y_over_l = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.031, 0.029, 0.019, 0.0],
            [0.047, 0.043, 0.026, 0.0],
            [0.028, 0.025, 0.014, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    if not pointed:
        y_over_l[0] = np.array([0.012, 0.010, 0.006, 0.0])
        y_over_l[-1] = np.array([0.017, 0.013, 0.007, 0.0])
    return x_over_l, z_over_l, y_over_l


@pytest.mark.parametrize(
    "argument",
    [0.0, 1.0e-14, -1.0e-14, 1.0e-6, -1.0e-6, 0.2, -3.4, 100.0],
)
def test_normalized_sinc_matches_definition(argument: float) -> None:
    expected = 1.0 if argument == 0.0 else np.sin(argument) / argument
    assert normalized_sinc(argument) == pytest.approx(expected, rel=2e-15, abs=2e-15)


def test_decaying_basis_moments_match_quadrature_and_limits() -> None:
    u, weights = _gauss_rule()
    exponents = np.array([0.0, 1.0e-14, 1.0e-7, 1.0e-3, 0.2, 3.0, 50.0])
    b0, b1 = decaying_linear_basis_moments(exponents)
    reference0 = np.array(
        [np.sum(weights * (1.0 - u) * np.exp(-a * u)) for a in exponents]
    )
    reference1 = np.array(
        [np.sum(weights * u * np.exp(-a * u)) for a in exponents]
    )
    np.testing.assert_allclose(b0, reference0, rtol=2e-13, atol=2e-15)
    np.testing.assert_allclose(b1, reference1, rtol=2e-13, atol=2e-15)

    large0, large1 = decaying_linear_basis_moments(1.0e4)
    assert large0 == pytest.approx(1.0e-4 - 1.0e-8, rel=2e-15)
    assert large1 == pytest.approx(1.0e-8, rel=2e-15)


def test_oscillatory_basis_moments_match_quadrature() -> None:
    u, weights = _gauss_rule(512)
    phases = np.array([0.0, 1.0e-14, -1.0e-7, 1.0e-3, -0.3, 4.0, 40.0])
    c0, c1 = oscillatory_linear_basis_moments(phases)
    reference0 = np.array(
        [np.sum(weights * (1.0 - u) * np.exp(-1j * q * u)) for q in phases]
    )
    reference1 = np.array(
        [np.sum(weights * u * np.exp(-1j * q * u)) for q in phases]
    )
    np.testing.assert_allclose(c0, reference0, rtol=5e-13, atol=3e-14)
    np.testing.assert_allclose(c1, reference1, rtol=5e-13, atol=3e-14)


def test_moments_against_optional_high_precision_reference() -> None:
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 80

    a = mp.mpf("1e-30")
    expected_b0 = mp.quad(lambda u: (1 - u) * mp.exp(-a * u), [0, 1])
    expected_b1 = mp.quad(lambda u: u * mp.exp(-a * u), [0, 1])
    b0, b1 = decaying_linear_basis_moments(float(a))
    assert b0 == pytest.approx(float(expected_b0), rel=0.0, abs=2e-16)
    assert b1 == pytest.approx(float(expected_b1), rel=0.0, abs=2e-16)

    q = mp.mpf("1e-25")
    expected_c0 = mp.quad(lambda u: (1 - u) * mp.exp(-1j * q * u), [0, 1])
    expected_c1 = mp.quad(lambda u: u * mp.exp(-1j * q * u), [0, 1])
    c0, c1 = oscillatory_linear_basis_moments(float(q))
    assert c0 == pytest.approx(complex(expected_c0), rel=0.0, abs=2e-16)
    assert c1 == pytest.approx(complex(expected_c1), rel=0.0, abs=2e-16)


def test_single_bilinear_cell_direct_and_ibp_terms_are_exact() -> None:
    x = np.array([-0.23, 0.31])
    z = np.array([0.0, 0.087])
    y = np.array([[0.019, 0.008], [0.041, 0.014]])
    lambda_ = 1.73
    froude = 0.31

    direct = bilinear_cell_amplitudes_direct(lambda_, froude, x, z, y)[0, 0]
    volume, boundary = bilinear_cell_amplitudes_ibp(lambda_, froude, x, z, y)

    u, weights = _gauss_rule(160)
    xx = x[0] + np.diff(x)[0] * u
    zz = z[0] + np.diff(z)[0] * u
    wx = np.diff(x)[0] * weights
    wz = np.diff(z)[0] * weights
    beta = lambda_ / froude**2
    alpha = lambda_**2 / froude**2

    slope_at_z0 = (y[1, 0] - y[0, 0]) / np.diff(x)[0]
    slope_at_z1 = (y[1, 1] - y[0, 1]) / np.diff(x)[0]
    slope = slope_at_z0 * (1.0 - u) + slope_at_z1 * u
    direct_reference = np.sum(wx * np.exp(-1j * beta * xx)) * np.sum(
        wz * slope * np.exp(-alpha * zz)
    )
    assert direct == pytest.approx(direct_reference, rel=3e-13, abs=2e-16)

    # The sum of the exact IBP volume and boundary pieces must recover the
    # same direct derivative integral, including non-zero end sections.
    ibp = compensated_sum(
        np.concatenate((volume.reshape(-1), boundary.reshape(-1)))
    )
    assert ibp == pytest.approx(direct_reference, rel=3e-13, abs=2e-16)


@pytest.mark.parametrize("pointed", [True, False])
def test_direct_and_ibp_amplitudes_agree(pointed: bool) -> None:
    x, z, y = _sample_hull(pointed=pointed)
    lambda_ = np.array([1.0, 1.08, 1.7, 3.5, 8.0])
    direct, direct_ratio = michell_amplitude_direct(lambda_, 0.27, x, z, y)
    ibp, ibp_ratio = michell_amplitude_ibp(lambda_, 0.27, x, z, y)
    assert direct.shape == lambda_.shape
    assert direct_ratio.shape == lambda_.shape
    assert ibp_ratio.shape == lambda_.shape
    np.testing.assert_allclose(direct, ibp, rtol=2e-11, atol=5e-17)
    assert np.all(direct_ratio >= 1.0)
    assert np.all(ibp_ratio >= 1.0)


def test_lambda_and_froude_broadcasting() -> None:
    x, z, y = _sample_hull()
    lambda_ = np.array([[1.0], [1.8]])
    froude = np.array([0.22, 0.28, 0.34])
    direct, ratio = michell_amplitude_direct(lambda_, froude, x, z, y)
    ibp, _ = michell_amplitude_ibp(lambda_, froude, x, z, y)
    assert direct.shape == (2, 3)
    assert ratio.shape == (2, 3)
    np.testing.assert_allclose(direct, ibp, rtol=5e-12, atol=5e-17)


def test_zero_translation_and_reversal_invariances() -> None:
    x = np.linspace(-0.5, 0.5, 7)
    z = np.array([0.0, 0.025, 0.06, 0.10])
    y = (
        0.045
        * (1.0 - (2.0 * x) ** 2)[:, None]
        * (1.0 - (z / z[-1]) ** 2)[None, :]
    )
    # Introduce fore-aft asymmetry while preserving pointed closure.
    asymmetric = y * (1.0 + 0.25 * x[:, None])
    lambda_ = np.linspace(1.0, 7.0, 19)
    amplitude, _ = michell_amplitude_ibp(lambda_, 0.29, x, z, asymmetric)
    translated, _ = michell_amplitude_ibp(
        lambda_, 0.29, x + 0.37, z, asymmetric
    )
    reversed_amplitude, _ = michell_amplitude_ibp(
        lambda_, 0.29, x, z, asymmetric[::-1]
    )
    np.testing.assert_allclose(
        np.abs(amplitude), np.abs(translated), rtol=1e-10, atol=1e-18
    )
    np.testing.assert_allclose(
        np.abs(amplitude), np.abs(reversed_amplitude), rtol=1e-10, atol=1e-18
    )

    zero, ratio = michell_amplitude_ibp(lambda_, 0.29, x, z, np.zeros_like(y))
    np.testing.assert_array_equal(zero, 0.0)
    np.testing.assert_array_equal(ratio, 1.0)


def test_compensated_sum_and_cancellation_diagnostics() -> None:
    terms = np.array([1.0e16, 1.0, -1.0e16])
    assert compensated_sum(terms) == pytest.approx(1.0)

    nearly_cancelled = np.array([1.0 + 0.0j, -1.0 + 1.0e-12j])
    total, diagnostics = sum_with_diagnostics(nearly_cancelled)
    assert total == pytest.approx(1.0e-12j)
    assert diagnostics.cancellation_ratio > 1.0e12
    assert diagnostics.absolute_sum == pytest.approx(2.0)
    assert diagnostics.term_count == 2
    assert cancellation_ratio(np.zeros(4)) == pytest.approx(1.0)
    assert np.isinf(cancellation_ratio([1.0, -1.0]))


@pytest.mark.parametrize(
    ("lambda_", "froude"),
    [(0.999, 0.3), (1.0, 0.0), (np.inf, 0.3), (1.0, np.nan)],
)
def test_invalid_spectral_parameters_are_rejected(
    lambda_: float, froude: float
) -> None:
    x, z, y = _sample_hull()
    with pytest.raises(ValueError):
        michell_amplitude_direct(lambda_, froude, x, z, y)


def test_shared_length_grid_contract_is_enforced_structurally() -> None:
    x, z, y = _sample_hull()
    with pytest.raises(ValueError, match="waterline"):
        michell_amplitude_direct(1.0, 0.3, x, z + 0.01, y)
    with pytest.raises(ValueError, match="shape"):
        michell_amplitude_direct(1.0, 0.3, x, z, y[:-1])
    invalid = y.copy()
    invalid[2, 1] = -0.001
    with pytest.raises(ValueError, match="non-negative"):
        michell_amplitude_direct(1.0, 0.3, x, z, invalid)
