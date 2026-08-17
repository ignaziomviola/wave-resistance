import numpy as np

from wave_resistance.bem.free_surface import linear_residual, nonlinear_residual


def test_downward_positive_linear_signs_and_dispersion():
    speed, gravity = 2.0, 9.81
    k = gravity / speed**2
    x = np.linspace(0.0, 3.0, 20)
    eta = np.cos(k * x)
    phi_x = gravity * eta / speed
    phi_z = speed * (-k * np.sin(k * x))
    kin, dyn = linear_residual(
        np.c_[np.zeros_like(x), np.zeros_like(x), phi_z], eta,
        phi_x, -k * np.sin(k * x), speed, gravity,
    )
    assert np.max(abs(kin)) < 1.0e-12
    assert np.max(abs(dyn)) < 1.0e-12


def test_uniform_flow_exact_nonlinear_zero_residual():
    kin, dyn = nonlinear_residual(
        np.zeros((3, 3)), np.zeros(3), np.zeros(3), np.zeros(3), 2.0, 9.81
    )
    assert np.allclose(kin, 0.0)
    assert np.allclose(dyn, 0.0)
