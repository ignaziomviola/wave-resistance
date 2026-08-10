from __future__ import annotations

import numpy as np
import pytest

from wave_resistance.validation import (
    HullAttitude,
    IncompatibleValidationData,
    ResistanceQuantity,
    ValidationSeries,
    compare_series,
    score_values,
)


def series(
    fn=(0.1, 0.2, 0.3),
    coefficients=(1.0e-3, 2.0e-3, 1.0e-3),
    *,
    quantity="wave_pattern",
    attitude="fixed",
    label="observed",
) -> ValidationSeries:
    return ValidationSeries(
        fn,
        coefficients,
        quantity=quantity,
        attitude=attitude,
        label=label,
    )


def test_quantity_and_attitude_aliases_are_explicit() -> None:
    data = ValidationSeries(
        [0.2],
        [0.001],
        quantity="C_WP",
        attitude="FX",
    )

    assert data.quantity is ResistanceQuantity.WAVE_PATTERN
    assert data.quantity.symbol == "C_WP"
    assert data.attitude is HullAttitude.FIXED
    assert data.attitude.code == "FX"


def test_from_csv_loads_sorts_and_preserves_provenance(tmp_path) -> None:
    path = tmp_path / "wigley.csv"
    path.write_text(
        "# digitised source\n"
        "Fn,Cwp,u_standard\n"
        "0.30,0.0012,0.00003\n"
        "0.10,0.0002,0.00001\n"
        "0.20,0.0007,0.00002\n",
        encoding="utf-8",
    )

    data = ValidationSeries.from_csv(
        path,
        quantity="wave_pattern",
        attitude="fixed",
        coefficient_column="Cwp",
        standard_uncertainty_column="u_standard",
        metadata={"reference_length": "LWL"},
    )

    np.testing.assert_allclose(data.fn, [0.10, 0.20, 0.30])
    np.testing.assert_allclose(data.coefficients, [0.0002, 0.0007, 0.0012])
    np.testing.assert_allclose(data.standard_uncertainty, [1.0e-5, 2.0e-5, 3.0e-5])
    assert data.label == "wigley"
    assert data.metadata["source_path"] == str(path)
    assert data.metadata["reference_length"] == "LWL"
    with pytest.raises(ValueError):
        data.fn[0] = 0.5


def test_from_csv_rejects_missing_or_invalid_columns(tmp_path) -> None:
    missing = tmp_path / "missing.csv"
    missing.write_text("Fn,other\n0.2,0.001\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required column"):
        ValidationSeries.from_csv(
            missing,
            quantity="wave",
            attitude="fixed",
        )

    invalid = tmp_path / "invalid.csv"
    invalid.write_text("Fn,coefficient\nnot-a-number,0.001\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid numeric value"):
        ValidationSeries.from_csv(
            invalid,
            quantity="wave",
            attitude="fixed",
        )


def test_canonical_self_describing_csv_contract(tmp_path) -> None:
    path = tmp_path / "canonical.csv"
    path.write_text(
        "fn,value,quantity,uncertainty,attitude,source\n"
        "0.20,0.0007,wave_making,0.00002,fixed,Tank A\n"
        "0.30,0.0012,wave_making,0.00003,fixed,Tank A\n",
        encoding="utf-8",
    )
    data = ValidationSeries.from_csv(path)
    assert data.quantity is ResistanceQuantity.WAVE_MAKING
    assert data.attitude is HullAttitude.FIXED
    assert data.label == "Tank A"
    assert data.metadata["source"] == "Tank A"
    np.testing.assert_allclose(data.standard_uncertainty, [2e-5, 3e-5])


@pytest.mark.parametrize(
    ("fn", "coefficient", "uncertainty", "message"),
    [
        ([0.2, 0.2], [0.001, 0.002], None, "unique"),
        ([0.0], [0.001], None, "positive"),
        ([0.2], [0.001], [0.0], "strictly positive"),
        ([0.2, 0.3], [0.001], None, "same length"),
    ],
)
def test_series_rejects_invalid_arrays(fn, coefficient, uncertainty, message) -> None:
    with pytest.raises(ValueError, match=message):
        ValidationSeries(
            fn,
            coefficient,
            quantity="wave",
            attitude="fixed",
            standard_uncertainty=uncertainty,
        )


def test_compare_series_interpolates_and_computes_curve_metrics() -> None:
    observed = series()
    prediction = series(
        fn=(0.1, 0.15, 0.2, 0.25, 0.3),
        coefficients=(1.1e-3, 1.6e-3, 2.1e-3, 1.6e-3, 1.1e-3),
        label="model",
    )

    report = compare_series(observed, prediction)
    np.testing.assert_allclose(
        report.prediction_at_observations,
        np.asarray(observed.coefficients) + 1.0e-4,
    )
    np.testing.assert_allclose(report.errors, 1.0e-4)

    weights = np.array([0.05, 0.10, 0.05])
    expected_l2 = np.sqrt(
        np.sum(weights * (1.0e-4) ** 2)
        / np.sum(weights * np.asarray(observed.coefficients) ** 2)
    )
    assert report.metrics.normalized_l2 == pytest.approx(expected_l2)
    assert report.metrics.bias_drag_counts == pytest.approx(1.0)
    assert report.metrics.mae_drag_counts == pytest.approx(1.0)
    assert report.metrics.max_abs_error == pytest.approx(1.0e-4)
    assert report.metrics.max_abs_error_drag_counts == pytest.approx(1.0)
    assert report.metrics.hump.observed_fn == pytest.approx(0.2)
    assert report.metrics.hump.predicted_fn == pytest.approx(0.2)
    assert report.metrics.hump.absolute_error == pytest.approx(0.0)
    assert report.metrics.hollow.absolute_error is None


def test_dominant_hump_and_hollow_locations_use_interior_extrema() -> None:
    observed = series(
        fn=(0.1, 0.2, 0.3, 0.4, 0.5),
        coefficients=(0.4e-3, 1.3e-3, 0.8e-3, 0.2e-3, 0.7e-3),
    )
    prediction = series(
        fn=(0.1, 0.18, 0.28, 0.38, 0.46, 0.5),
        coefficients=(0.4e-3, 1.2e-3, 0.9e-3, 0.3e-3, 0.8e-3, 0.7e-3),
        label="model",
    )

    metrics = observed.compare(prediction).metrics
    assert metrics.hump.observed_fn == pytest.approx(0.2)
    assert metrics.hump.predicted_fn == pytest.approx(0.18)
    assert metrics.hump.absolute_error == pytest.approx(0.02)
    assert metrics.hollow.observed_fn == pytest.approx(0.4)
    assert metrics.hollow.predicted_fn == pytest.approx(0.38)
    assert metrics.hollow.absolute_error == pytest.approx(0.02)


def test_like_for_like_checks_quantity_attitude_and_range() -> None:
    observed = series()

    with pytest.raises(IncompatibleValidationData, match="quantities differ"):
        observed.compare(series(quantity="residuary", label="model"))

    with pytest.raises(IncompatibleValidationData, match="attitudes differ"):
        observed.compare(series(attitude="free", label="model"))

    with pytest.raises(IncompatibleValidationData, match="extrapolation is not allowed"):
        observed.compare(
            series(
                fn=(0.15, 0.2, 0.25),
                coefficients=(0.001, 0.002, 0.001),
                label="model",
            )
        )


def test_score_values_requires_prediction_semantics() -> None:
    observed = series()
    report = score_values(
        observed,
        [0.1, 0.2, 0.3],
        observed.coefficients,
        quantity="C_WP",
        attitude="FX",
    )
    assert report.metrics.normalized_l2 == pytest.approx(0.0)


def test_zero_reference_curve_has_defined_l2_behavior() -> None:
    zero = series(coefficients=(0.0, 0.0, 0.0))
    exact = series(coefficients=(0.0, 0.0, 0.0), label="exact")
    nonzero = series(coefficients=(0.0, 1.0e-4, 0.0), label="nonzero")

    assert zero.compare(exact).metrics.normalized_l2 == 0.0
    assert np.isinf(zero.compare(nonzero).metrics.normalized_l2)


def test_plain_text_report_contains_semantics_and_units() -> None:
    report = series().compare(series(label="Michell"))
    text = report.to_text()

    assert "Michell vs observed" in text
    assert "C_WP (wave_pattern)" in text
    assert "fixed (FX)" in text
    assert "drag counts" in text
    assert "Hump location error in Fn" in text
