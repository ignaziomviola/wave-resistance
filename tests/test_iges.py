"""V1: IGES parsing against values decoded by hand from the actual Sysser 01 file."""

import numpy as np
import pytest

from wave_resistance.iges import _tokenise, read_iges


def test_units_and_scale(sysser01_iges):
    assert sysser01_iges.units_flag == 2                     # 2HMM
    assert sysser01_iges.units_name == "MM"
    assert sysser01_iges.metres_per_unit == pytest.approx(1e-3)
    assert sysser01_iges.model_space_scale == pytest.approx(1.0)


def test_two_surfaces_with_expected_structure(sysser01_iges):
    hull, transom = sysser01_iges.surfaces
    assert (hull.n_u, hull.n_v) == (40, 30)
    assert (hull.degree_u, hull.degree_v) == (3, 3)
    assert hull.form == 0 and hull.is_polynomial
    assert np.allclose(hull.weights, 1.0)
    assert (transom.n_u, transom.n_v) == (2, 40)
    assert (transom.degree_u, transom.degree_v) == (1, 3)
    assert transom.form == 8                                  # ruled surface


def test_knot_ranges_and_parameter_extents(sysser01_iges):
    hull = sysser01_iges.surfaces[0]
    assert hull.u_range == pytest.approx((0.0, 122.756555850919))
    assert hull.v_range == pytest.approx((0.0, 2229.526250067383))
    assert hull.knots_u.size == 40 + 3 + 1
    assert hull.knots_v.size == 30 + 3 + 1


def test_ignored_entities_are_reported_not_guessed(sysser01_iges):
    assert dict(sysser01_iges.ignored) == {314: 5, 406: 9}


def test_bounding_box_in_metres(sysser01_iges):
    lo, hi = sysser01_iges.surfaces[0].bounding_box()
    # Control-net box, which bounds the surface by the convex-hull property.
    assert lo == pytest.approx([-0.272, -0.29452560, -9.271047e-5], abs=1e-8)
    assert hi == pytest.approx([1.864, 0.0, 0.3181], abs=1e-8)


def test_hollerith_strings_may_contain_the_delimiter():
    # A Hollerith string is consumed by length, so an embedded comma is data.
    fields = _tokenise("3,5Ha,b,c;", ",", ";")
    assert fields == ["3", "a,b,c"]


def test_empty_fields_are_preserved():
    assert _tokenise("1,,3;", ",", ";") == ["1", "", "3"]


def test_d_exponent_form_is_accepted(sysser01_iges):
    # The file writes doubles as 1.0D0; a failure here would surface as a parse error.
    assert np.isfinite(sysser01_iges.surfaces[0].control_points).all()


def test_missing_surface_raises(tmp_path):
    p = tmp_path / "empty.igs"
    p.write_text("")
    with pytest.raises(ValueError):
        read_iges(p)
