import json
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np

from wave_resistance.geometry import (
    HullGeometryWarning,
    HullMetadata,
    OffsetHull,
    wigley_hull,
)


def metadata(length=1.0, beam=0.2, draft=1.0, wetted_area=2.0, name="test"):
    return HullMetadata(
        schema_version="1.0",
        name=name,
        length_ref_m=length,
        length_ref_kind="LWL",
        wetted_area_m2=wetted_area,
        beam_m=beam,
        draft_m=draft,
    )


class HullMetadataTests(unittest.TestCase):
    def test_principal_dimensions_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "length_ref_m"):
            metadata(length=0.0)
        with self.assertRaisesRegex(ValueError, "wetted_area_m2"):
            metadata(wetted_area=float("nan"))

    def test_required_json_contract(self):
        metadata = HullMetadata.from_mapping(
            {
                "schema_version": "1.0",
                "name": "example",
                "length_ref_m": 20.0,
                "length_ref_kind": "LWL",
                "wetted_area_m2": 40.0,
                "beam_m": 2.0,
                "draft_m": 1.0,
            }
        )
        self.assertEqual(metadata.length_ref_m, 20.0)
        self.assertEqual(metadata.beam_m, 2.0)
        self.assertAlmostEqual(metadata.beam_to_length, 0.1)
        with self.assertRaisesRegex(ValueError, "wetted_area_m2"):
            HullMetadata.from_mapping(
                {
                    "schema_version": "1.0",
                    "name": "incomplete",
                    "length_ref_m": 20.0,
                    "length_ref_kind": "LWL",
                }
            )


class OffsetHullTests(unittest.TestCase):
    def test_default_wigley_contract_and_dimensions(self):
        hull = wigley_hull()
        self.assertEqual(hull.metadata.schema_version, "1.0")
        self.assertEqual(hull.metadata.length_ref_kind, "LWL")
        self.assertAlmostEqual(hull.metadata.length_ref_m, 1.0)
        self.assertAlmostEqual(hull.beam_m, 0.1)
        self.assertAlmostEqual(hull.draft_m, 0.1 / 1.6)
        self.assertGreater(hull.metadata.wetted_area_m2, 0.0)
        self.assertEqual(hull.half_breadths.shape, (81, 33))

    def test_wigley_closure_scaling_and_hydrostatics(self):
        hull = wigley_hull(30.0, 3.0, 1.5, nx=201, nz=101)

        self.assertEqual(hull.half_breadths.shape, (201, 101))
        np.testing.assert_array_equal(hull.half_breadths[0, :], 0.0)
        np.testing.assert_array_equal(hull.half_breadths[-1, :], 0.0)
        np.testing.assert_array_equal(hull.half_breadths[:, -1], 0.0)
        self.assertAlmostEqual(float(np.max(hull.half_breadths)), 1.5)
        np.testing.assert_allclose(hull.x_over_length[[0, -1]], [0.0, 1.0])
        np.testing.assert_allclose(hull.z_over_draft[[0, -1]], [0.0, 1.0])
        x_l, z_l, y_l = hull.coordinates_over_length
        np.testing.assert_allclose(x_l, hull.x_m / hull.metadata.length_ref_m)
        np.testing.assert_allclose(z_l, hull.z_m / hull.metadata.length_ref_m)
        np.testing.assert_allclose(y_l, hull.half_breadth_m / hull.metadata.length_ref_m)

        self.assertAlmostEqual(hull.block_coefficient, 4.0 / 9.0, places=4)
        self.assertAlmostEqual(hull.diagnostics.waterplane_coefficient, 2.0 / 3.0, places=4)
        self.assertAlmostEqual(hull.diagnostics.midship_coefficient, 2.0 / 3.0, places=4)
        self.assertAlmostEqual(hull.diagnostics.prismatic_coefficient, 2.0 / 3.0, places=4)
        self.assertAlmostEqual(
            hull.diagnostics.longitudinal_center_of_buoyancy, 15.0, places=10
        )
        self.assertGreater(hull.diagnostics.maximum_longitudinal_slope, 0.0)
        self.assertGreater(hull.diagnostics.maximum_vertical_slope, 0.0)

    def test_from_tensor_and_arrays_are_immutable(self):
        hull_metadata = HullMetadata(
            schema_version="1.0",
            name="inferred dimensions",
            length_ref_m=1.0,
            length_ref_kind="LWL",
            wetted_area_m2=2.0,
        )
        hull = OffsetHull.from_tensor(
            hull_metadata,
            [0.0, 0.5, 1.0],
            [0.0, 1.0],
            [[0.0, 0.0], [0.1, 0.0], [0.0, 0.0]],
        )
        self.assertAlmostEqual(hull.displaced_volume, 0.05)
        self.assertAlmostEqual(hull.beam_m, 0.2)
        self.assertAlmostEqual(hull.draft_m, 1.0)
        with self.assertRaises(ValueError):
            hull.half_breadths[1, 0] = 0.4

    def test_irregular_profiles_are_piecewise_linearly_tensorized(self):
        hull_metadata = metadata()
        raw_x = [0, 0, 0.5, 0.5, 0.5, 1, 1]
        raw_z = [0, 1, 0, 0.25, 1, 0, 1]
        raw_y = [0, 0, 0.1, 0.075, 0, 0, 0]
        x_grid = np.linspace(0.0, 1.0, 5)
        z_grid = np.linspace(0.0, 1.0, 5)

        hull = OffsetHull.from_irregular(
            hull_metadata, raw_x, raw_z, raw_y, x_grid=x_grid, z_grid=z_grid
        )

        longitudinal = 1.0 - np.abs(2.0 * x_grid - 1.0)
        expected = 0.1 * longitudinal[:, None] * (1.0 - z_grid[None, :])
        np.testing.assert_allclose(hull.half_breadths, expected)

    def test_invalid_closures_duplicates_negatives_and_gaps_are_rejected(self):
        hull_metadata = metadata()
        valid = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.0]])

        pointed = valid.copy()
        pointed[0, 0] = 0.1
        with self.assertRaisesRegex(ValueError, "pointed"):
            OffsetHull(hull_metadata, [0, 0.5, 1], [0, 1], pointed)

        keel = valid.copy()
        keel[1, -1] = 0.1
        with self.assertRaisesRegex(ValueError, "centreplane"):
            OffsetHull(hull_metadata, [0, 0.5, 1], [0, 1], keel)

        negative = valid.copy()
        negative[1, 0] = -0.1
        with self.assertRaisesRegex(ValueError, "negative"):
            OffsetHull(hull_metadata, [0, 0.5, 1], [0, 1], negative)

        with self.assertRaisesRegex(ValueError, "duplicate"):
            OffsetHull(hull_metadata, [0, 0.5, 0.5], [0, 1], valid)

        gap = valid.copy()
        gap[1, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "gap"):
            OffsetHull(hull_metadata, [0, 0.5, 1], [0, 1], gap)

        reopened = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.1, 0.0, 0.05, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        with self.assertRaisesRegex(ValueError, "reopens"):
            OffsetHull(hull_metadata, [0, 0.5, 1], [0, 0.4, 0.8, 1], reopened)

        with self.assertRaisesRegex(ValueError, "coverage gap"):
            OffsetHull.from_irregular(
                hull_metadata,
                [0, 0, 0.5, 0.5, 1, 1],
                [0, 1, 0, 0.5, 0, 1],
                [0, 0, 0.1, 0.05, 0, 0],
            )

    def test_large_beam_to_length_emits_diagnostic_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            hull = wigley_hull(4.0, 1.0, 0.5, nx=11, nz=7)
        self.assertTrue(any(issubclass(item.category, HullGeometryWarning) for item in caught))
        self.assertIn("B/L", hull.diagnostics.validation_warnings[0])

    def test_csv_and_both_json_encodings_load(self):
        long_offsets = [
            {"x_m": 0.0, "z_m": 0.0, "half_breadth_m": 0.0},
            {"x_m": 0.0, "z_m": 1.0, "half_breadth_m": 0.0},
            {"x_m": 0.5, "z_m": 0.0, "half_breadth_m": 0.1},
            {"x_m": 0.5, "z_m": 1.0, "half_breadth_m": 0.0},
            {"x_m": 1.0, "z_m": 0.0, "half_breadth_m": 0.0},
            {"x_m": 1.0, "z_m": 1.0, "half_breadth_m": 0.0},
        ]
        metadata_document = {
            "schema_version": "1.0",
            "name": "tiny",
            "length_ref_m": 1.0,
            "length_ref_kind": "LWL",
            "wetted_area_m2": 2.0,
            "beam_m": 0.2,
            "draft_m": 1.0,
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            csv_path = root / "hull.csv"
            csv_path.write_text(
                "x_m,z_m,half_breadth_m\n"
                + "".join(
                    "{x_m},{z_m},{half_breadth_m}\n".format(**offset)
                    for offset in long_offsets
                ),
                encoding="utf-8",
            )
            metadata_path = root / "metadata.json"
            metadata_path.write_text(json.dumps(metadata_document), encoding="utf-8")
            csv_hull = OffsetHull.from_csv(csv_path, metadata_path)

            long_json_path = root / "long.json"
            long_json_path.write_text(
                json.dumps({"metadata": metadata_document, "offsets": long_offsets}),
                encoding="utf-8",
            )
            long_json_hull = OffsetHull.from_json(long_json_path)

            tensor_json_path = root / "tensor.json"
            tensor_json_path.write_text(
                json.dumps(
                    {
                        "metadata": metadata_document,
                        "x_m": [0.0, 0.5, 1.0],
                        "z_m": [0.0, 1.0],
                        "half_breadth_m": [[0.0, 0.0], [0.1, 0.0], [0.0, 0.0]],
                    }
                ),
                encoding="utf-8",
            )
            tensor_json_hull = OffsetHull.from_json(tensor_json_path)

        np.testing.assert_allclose(csv_hull.half_breadths, long_json_hull.half_breadths)
        np.testing.assert_allclose(csv_hull.half_breadths, tensor_json_hull.half_breadths)


if __name__ == "__main__":
    unittest.main()
