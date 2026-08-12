import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from wave_resistance import HullOffsets, image_inspired_yacht, wigley_hull


class HullGeometryTests(unittest.TestCase):
    def test_wigley_hydrostatics_match_analytic_volume(self):
        hull = wigley_hull(station_count=401, vertical_count=201)
        expected = 4.0 / 9.0 * 0.10 * 0.0625
        self.assertAlmostEqual(hull.hydrostatics.displacement_volume_m3, expected, places=7)
        self.assertAlmostEqual(hull.hydrostatics.longitudinal_center_of_buoyancy_m, 0.0, places=12)

    def test_image_inspired_ratios_and_aft_fullness(self):
        hull = image_inspired_yacht(station_count=101, vertical_count=51)
        hydro = hull.hydrostatics
        self.assertAlmostEqual(hydro.maximum_waterline_beam_m, 0.28, delta=2.0e-4)
        self.assertAlmostEqual(hydro.maximum_canoe_draft_m, 0.06, delta=2.0e-5)
        self.assertGreater(hydro.longitudinal_center_of_buoyancy_m, 0.0)
        self.assertGreater(hydro.displacement_volume_m3, 0.0)

    def test_scaling_preserves_shape_and_scales_volume(self):
        small = image_inspired_yacht(length_ref_m=1.0)
        large = image_inspired_yacht(length_ref_m=3.0)
        self.assertAlmostEqual(
            large.hydrostatics.displacement_volume_m3
            / small.hydrostatics.displacement_volume_m3,
            27.0,
            places=10,
        )
        np.testing.assert_allclose(small.nondimensional[2], large.nondimensional[2])

    def test_mesh_is_positive_and_points_into_starboard_fluid(self):
        mesh = image_inspired_yacht().resample(21, 11).to_mesh()
        self.assertTrue(np.all(mesh.areas > 0.0))
        self.assertTrue(np.all(mesh.normals[:, 1] >= -1.0e-12))
        np.testing.assert_allclose(np.linalg.norm(mesh.normals, axis=1), 1.0, atol=1.0e-14)

    def test_csv_round_trip_preserves_offsets_and_metadata(self):
        hull = image_inspired_yacht(station_count=11, vertical_count=7)
        with tempfile.TemporaryDirectory() as directory:
            offsets = Path(directory) / "hull.csv"
            metadata = Path(directory) / "hull.json"
            hull.to_csv(offsets, metadata)
            loaded = HullOffsets.from_csv(offsets, metadata)
            payload = json.loads(metadata.read_text(encoding="utf-8"))
        np.testing.assert_allclose(loaded.x_m, hull.x_m)
        np.testing.assert_allclose(loaded.z_m, hull.z_m)
        np.testing.assert_allclose(loaded.half_breadth_m, hull.half_breadth_m)
        self.assertEqual(payload["schema_version"], "2.0")

    def test_malformed_offsets_are_rejected(self):
        with self.assertRaises(ValueError):
            HullOffsets(
                np.array([0.0, 1.0, 0.5]),
                np.zeros((3, 3)),
                np.zeros((3, 3)),
            )


if __name__ == "__main__":
    unittest.main()
