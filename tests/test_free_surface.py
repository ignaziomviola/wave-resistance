import unittest

import numpy as np

from wave_resistance import MeshSettings, image_inspired_yacht
from wave_resistance.free_surface import build_free_surface_mesh, minimum_points_per_wavelength


class FreeSurfaceTests(unittest.TestCase):
    def test_mesh_excludes_waterplane_and_has_positive_weights(self):
        hull = image_inspired_yacht()
        settings = MeshSettings(free_surface_x_points=17, free_surface_y_points=7)
        mesh = build_free_surface_mesh(hull, settings)
        breadth = hull.half_breadth_at_waterline(mesh.collocation_points[:, 0])
        self.assertTrue(np.all(mesh.collocation_points[:, 1] > breadth))
        self.assertTrue(np.all(mesh.source_weights > 0.0))
        self.assertTrue(np.all(mesh.source_points[:, 2] > 0.0))

    def test_wavelength_resolution_increases_with_froude_number(self):
        mesh = build_free_surface_mesh(image_inspired_yacht(), MeshSettings())
        self.assertGreater(
            minimum_points_per_wavelength(mesh, 0.4),
            minimum_points_per_wavelength(mesh, 0.2),
        )


if __name__ == "__main__":
    unittest.main()
