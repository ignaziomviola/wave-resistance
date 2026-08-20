"""Verification of the Wigley oracle: closed forms against quadrature, hydrostatics exact."""

import numpy as np
import pytest
from scipy.integrate import quad

from wave_resistance.wigley import (
    _g1, _g2, wigley_exact_hydrostatics, wigley_michell_amplitude, wigley_offsets,
)


@pytest.mark.filterwarnings("ignore:The maximum number of subdivisions")
@pytest.mark.parametrize("beta", [1e-8, 1e-4, 1e-2, 0.5, 0.999, 1.0, 1.001, 3.0, 40.0, 400.0, 5000.0])
def test_g1_matches_direct_quadrature(beta):
    """G1(beta) = integral over [-1/2, 1/2] of (-8X) exp(-i beta X) dX."""
    re = quad(lambda X: -8 * X * np.cos(beta * X), -0.5, 0.5, limit=500)[0]
    im = quad(lambda X: 8 * X * np.sin(beta * X), -0.5, 0.5, limit=500)[0]
    ref = re + 1j * im
    got = complex(_g1(np.array(float(beta))))
    assert abs(got - ref) <= 1e-11 * max(abs(ref), 1e-12)


@pytest.mark.parametrize("alpha", [1e-8, 1e-2, 1.0, 7.9, 8.0, 8.1, 20.0, 500.0, 50000.0])
def test_g2_matches_direct_quadrature(alpha):
    """G2(alpha) = integral over [0, tau] of (1 - (Z/tau)^2) exp(-alpha Z) dZ."""
    tau = 0.0625
    ref = quad(lambda Z: (1 - (Z / tau) ** 2) * np.exp(-alpha * Z), 0.0, tau, limit=500)[0]
    got = float(_g2(np.array(float(alpha)), tau))
    assert got == pytest.approx(ref, rel=1e-11)


def test_g1_is_purely_imaginary():
    # The real part integrates an odd function over a symmetric interval.
    assert np.abs(_g1(np.array([0.3, 2.0, 17.0])).real).max() == 0.0


def test_amplitude_is_smooth_across_the_series_crossovers():
    fn = 0.30
    lam = np.linspace(1.0, 12.0, 40001)
    a = wigley_michell_amplitude(lam, fn)
    # No kink: third differences stay bounded relative to the local scale.
    d3 = np.abs(np.diff(a, 3))
    assert d3.max() / np.abs(a).max() < 1e-6


def test_amplitude_rejects_bad_arguments():
    with pytest.raises(ValueError):
        wigley_michell_amplitude(0.5, 0.3)
    with pytest.raises(ValueError):
        wigley_michell_amplitude(1.5, 0.0)


def test_offsets_vanish_at_the_ends_and_the_keel():
    length, beam, draught = 4.0, 0.4, 0.25
    assert wigley_offsets(np.array([-2.0, 2.0]), np.array([-0.1, -0.1]), length, beam, draught) \
        == pytest.approx([0.0, 0.0])
    assert wigley_offsets(np.array([0.0]), np.array([-draught]), length, beam, draught) \
        == pytest.approx([0.0])
    assert wigley_offsets(np.array([0.0]), np.array([0.0]), length, beam, draught) \
        == pytest.approx([0.5 * beam])


def test_exact_hydrostatic_coefficients_are_scale_free():
    for args in ((1.0, 0.1, 0.0625), (7.3, 1.9, 0.44)):
        e = wigley_exact_hydrostatics(*args)
        assert e["cb"] == pytest.approx(4.0 / 9.0)
        assert e["cwp"] == pytest.approx(2.0 / 3.0)
        assert e["cm"] == pytest.approx(2.0 / 3.0)
        assert e["cp"] == pytest.approx(2.0 / 3.0)
        assert e["vcb"] == pytest.approx(-3.0 * args[2] / 8.0)
