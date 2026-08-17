import unittest

import numpy as np

from wave_resistance import MeshSettings, wigley_hull
from wave_resistance.free_surface import build_free_surface_mesh, minimum_points_per_wavelength
from wave_resistance.bem.free_surface import linear_residual, nonlinear_residual


class FreeSurfaceTests(unittest.TestCase):
    def test_waterplane_points_are_excluded(self):
        hull = wigley_hull(nx=13, nz=7)
        settings = MeshSettings(free_surface_x_points=17, free_surface_y_points=7)
        mesh = build_free_surface_mesh(hull, settings)
        breadth = hull.half_breadth_at_waterline(
            mesh.collocation_points[:, 0] * hull.length_ref_m
        ) / hull.length_ref_m
        self.assertTrue(np.all(mesh.collocation_points[:, 1] > breadth))

    def test_wavelength_resolution_metric(self):
        hull = wigley_hull(nx=13, nz=7)
        settings = MeshSettings(free_surface_x_points=17, free_surface_y_points=7)
        self.assertGreater(minimum_points_per_wavelength(build_free_surface_mesh(hull, settings), 0.3), 0.0)


def test_upward_positive_linear_signs_and_dispersion():
    speed, gravity = 2.0, 9.81
    k = gravity / speed**2
    x = np.linspace(0.0, 3.0, 20)
    eta = np.cos(k * x)
    phi_x = -gravity * eta / speed
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


if __name__ == "__main__":
    unittest.main()
