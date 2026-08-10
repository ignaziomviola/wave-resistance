"""Validation utilities for wave-resistance predictions.

The central rule in this module is that comparisons must be like for like.
In particular, a wave-making coefficient is not silently compared with a
residuary or total-resistance coefficient, and a fixed-attitude calculation is
not silently compared with a model that was free to sink and trim.

All resistance values handled here are dimensionless coefficients based on
``0.5 * rho * speed**2 * wetted_area``.  The module deliberately uses explicit
quantity names because ``Cw`` is also commonly used for the waterplane-area
coefficient in naval architecture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import csv
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray


DRAG_COUNT = 1.0e-4


class IncompatibleValidationData(ValueError):
    """Raised when two series do not represent the same physical measurand."""


class ResistanceQuantity(str, Enum):
    """Resistance-coefficient semantics.

    ``WAVE_PATTERN`` is obtained from far-field wave analysis. ``WAVE_MAKING``
    is the force associated with wave generation. ``RESIDUARY`` is inferred
    by subtracting a friction/form-factor estimate from total resistance and
    is therefore not treated as interchangeable with either wave quantity.
    """

    TOTAL = "total"
    RESIDUARY = "residuary"
    WAVE_MAKING = "wave_making"
    WAVE_PATTERN = "wave_pattern"

    @property
    def symbol(self) -> str:
        return {
            ResistanceQuantity.TOTAL: "C_T",
            ResistanceQuantity.RESIDUARY: "C_R",
            ResistanceQuantity.WAVE_MAKING: "C_W",
            ResistanceQuantity.WAVE_PATTERN: "C_WP",
        }[self]

    @classmethod
    def parse(cls, value: Union[ResistanceQuantity, str]) -> ResistanceQuantity:
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "total": cls.TOTAL,
            "ct": cls.TOTAL,
            "c_t": cls.TOTAL,
            "residuary": cls.RESIDUARY,
            "residual": cls.RESIDUARY,
            "cr": cls.RESIDUARY,
            "c_r": cls.RESIDUARY,
            "wave": cls.WAVE_MAKING,
            "wave_making": cls.WAVE_MAKING,
            "wavemaking": cls.WAVE_MAKING,
            "cw": cls.WAVE_MAKING,
            "c_w": cls.WAVE_MAKING,
            "wave_pattern": cls.WAVE_PATTERN,
            "wavepattern": cls.WAVE_PATTERN,
            "cwp": cls.WAVE_PATTERN,
            "c_wp": cls.WAVE_PATTERN,
        }
        try:
            return aliases[normalized]
        except KeyError as exc:
            choices = ", ".join(item.value for item in cls)
            raise ValueError(
                f"unknown resistance quantity {value!r}; choose one of {choices}"
            ) from exc


class HullAttitude(str, Enum):
    """Model attitude used for a prediction or towing-tank measurement."""

    FIXED = "fixed"
    FREE_SINKAGE = "free_sinkage"
    FREE_SINKAGE_TRIM = "free_sinkage_trim"

    @property
    def code(self) -> str:
        """Return the conventional ITTC attitude code."""

        return {
            HullAttitude.FIXED: "FX",
            HullAttitude.FREE_SINKAGE: "FS",
            HullAttitude.FREE_SINKAGE_TRIM: "FR",
        }[self]

    @classmethod
    def parse(cls, value: Union[HullAttitude, str]) -> HullAttitude:
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "fixed": cls.FIXED,
            "fx": cls.FIXED,
            "free_sinkage": cls.FREE_SINKAGE,
            "sinkage_free": cls.FREE_SINKAGE,
            "fs": cls.FREE_SINKAGE,
            "free": cls.FREE_SINKAGE_TRIM,
            "free_sinkage_trim": cls.FREE_SINKAGE_TRIM,
            "free_to_sink_and_trim": cls.FREE_SINKAGE_TRIM,
            "fr": cls.FREE_SINKAGE_TRIM,
        }
        try:
            return aliases[normalized]
        except KeyError as exc:
            choices = ", ".join(item.value for item in cls)
            raise ValueError(f"unknown hull attitude {value!r}; choose one of {choices}") from exc


def _as_finite_vector(values: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one value")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.array(array, dtype=float, copy=True)


@dataclass(frozen=True)
class ValidationSeries:
    """A resistance-coefficient curve with explicit physical semantics.

    Parameters
    ----------
    froude_numbers:
        Froude numbers based on the reference length used by the source.
    coefficients:
        Dimensionless resistance coefficients. The normalization is
        ``R / (0.5 rho U^2 S)``.
    quantity, attitude:
        Mandatory semantics used to prevent invalid comparisons.
    standard_uncertainty:
        Optional one-standard-deviation uncertainty in the coefficient. Do
        not pass a 95-percent expanded uncertainty without first converting
        it to a standard uncertainty.

    Notes
    -----
    Points are sorted by Froude number on construction. Duplicate Froude
    numbers are rejected rather than averaged implicitly.
    """

    froude_numbers: ArrayLike
    coefficients: ArrayLike
    quantity: Union[ResistanceQuantity, str]
    attitude: Union[HullAttitude, str]
    label: str = ""
    standard_uncertainty: Optional[ArrayLike] = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        fn = _as_finite_vector(self.froude_numbers, "froude_numbers")
        coefficients = _as_finite_vector(self.coefficients, "coefficients")
        if fn.size != coefficients.size:
            raise ValueError("froude_numbers and coefficients must have the same length")
        if np.any(fn <= 0.0):
            raise ValueError("froude_numbers must be positive")

        uncertainty: Optional[NDArray[np.float64]] = None
        if self.standard_uncertainty is not None:
            uncertainty = _as_finite_vector(
                self.standard_uncertainty, "standard_uncertainty"
            )
            if uncertainty.size != fn.size:
                raise ValueError(
                    "standard_uncertainty and froude_numbers must have the same length"
                )
            if np.any(uncertainty <= 0.0):
                raise ValueError("standard_uncertainty must be strictly positive")

        order = np.argsort(fn, kind="stable")
        fn = fn[order]
        coefficients = coefficients[order]
        if uncertainty is not None:
            uncertainty = uncertainty[order]
        if np.any(np.diff(fn) <= 0.0):
            raise ValueError("froude_numbers must be unique")

        fn.setflags(write=False)
        coefficients.setflags(write=False)
        if uncertainty is not None:
            uncertainty.setflags(write=False)

        object.__setattr__(self, "froude_numbers", fn)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "quantity", ResistanceQuantity.parse(self.quantity))
        object.__setattr__(self, "attitude", HullAttitude.parse(self.attitude))
        object.__setattr__(self, "label", str(self.label))
        object.__setattr__(self, "standard_uncertainty", uncertainty)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @classmethod
    def from_csv(
        cls,
        path: Union[str, Path],
        *,
        quantity: Optional[Union[ResistanceQuantity, str]] = None,
        attitude: Optional[Union[HullAttitude, str]] = None,
        fn_column: str = "Fn",
        coefficient_column: str = "coefficient",
        standard_uncertainty_column: Optional[str] = None,
        label: Optional[str] = None,
        delimiter: str = ",",
        encoding: str = "utf-8-sig",
        metadata: Optional[Mapping[str, object]] = None,
    ) -> ValidationSeries:
        """Load a validation series from a header-based CSV file.

        The canonical self-describing schema is
        ``fn,value,quantity,uncertainty,attitude,source``.  Legacy files may
        instead supply ``quantity`` and ``attitude`` explicitly and select
        custom value columns. Blank lines and comment lines are ignored.
        """

        source = Path(path)
        fn: list[float] = []
        coefficients: list[float] = []
        uncertainty: Optional[list[float]] = (
            [] if standard_uncertainty_column is not None else None
        )
        row_quantities: list[str] = []
        row_attitudes: list[str] = []
        row_sources: list[str] = []

        with source.open("r", encoding=encoding, newline="") as stream:
            lines = (
                line
                for line in stream
                if line.strip() and not line.lstrip().startswith("#")
            )
            reader = csv.DictReader(lines, delimiter=delimiter)
            if reader.fieldnames is None:
                raise ValueError(f"{source} does not contain a CSV header")
            fields = set(reader.fieldnames)
            resolved_fn_column = fn_column
            if resolved_fn_column not in fields and fn_column == "Fn" and "fn" in fields:
                resolved_fn_column = "fn"
            resolved_coefficient_column = coefficient_column
            if (
                resolved_coefficient_column not in fields
                and coefficient_column == "coefficient"
                and "value" in fields
            ):
                resolved_coefficient_column = "value"
            resolved_uncertainty_column = standard_uncertainty_column
            if resolved_uncertainty_column is None and "uncertainty" in fields:
                resolved_uncertainty_column = "uncertainty"
                uncertainty = []

            required = {resolved_fn_column, resolved_coefficient_column}
            if resolved_uncertainty_column is not None:
                required.add(resolved_uncertainty_column)
            if quantity is None:
                required.add("quantity")
            if attitude is None:
                required.add("attitude")
            missing = sorted(required.difference(reader.fieldnames))
            if missing:
                raise ValueError(
                    f"{source} is missing required column(s): {', '.join(missing)}"
                )

            for row in reader:
                try:
                    fn.append(float(row[resolved_fn_column]))
                    coefficients.append(float(row[resolved_coefficient_column]))
                    if uncertainty is not None and resolved_uncertainty_column is not None:
                        uncertainty.append(float(row[resolved_uncertainty_column]))
                    if quantity is None:
                        row_quantities.append(str(row["quantity"]))
                    if attitude is None:
                        row_attitudes.append(str(row["attitude"]))
                    if "source" in row and row["source"]:
                        row_sources.append(str(row["source"]))
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"invalid numeric value in {source} near CSV row {reader.line_num}"
                    ) from exc

        resolved_quantity: Union[ResistanceQuantity, str]
        if quantity is None:
            parsed_quantities = {ResistanceQuantity.parse(item) for item in row_quantities}
            if len(parsed_quantities) != 1:
                raise ValueError("CSV quantity column must contain one consistent value")
            resolved_quantity = parsed_quantities.pop()
        else:
            resolved_quantity = quantity
        resolved_attitude: Union[HullAttitude, str]
        if attitude is None:
            parsed_attitudes = {HullAttitude.parse(item) for item in row_attitudes}
            if len(parsed_attitudes) != 1:
                raise ValueError("CSV attitude column must contain one consistent value")
            resolved_attitude = parsed_attitudes.pop()
        else:
            resolved_attitude = attitude

        combined_metadata = dict(metadata or {})
        combined_metadata.setdefault("source_path", str(source))
        if row_sources and len(set(row_sources)) == 1:
            combined_metadata.setdefault("source", row_sources[0])
        return cls(
            fn,
            coefficients,
            quantity=resolved_quantity,
            attitude=resolved_attitude,
            label=(row_sources[0] if label is None and row_sources else source.stem)
            if label is None
            else label,
            standard_uncertainty=uncertainty,
            metadata=combined_metadata,
        )

    @property
    def fn(self) -> NDArray[np.float64]:
        """Short alias for :attr:`froude_numbers`."""

        return self.froude_numbers

    def compare(self, prediction: ValidationSeries) -> ValidationReport:
        """Compare this observed series with a like-for-like prediction."""

        return compare_series(self, prediction)


@dataclass(frozen=True)
class ExtremumLocationMetrics:
    """Location comparison for a dominant interior hump or hollow."""

    observed_fn: Optional[float]
    predicted_fn: Optional[float]
    absolute_error: Optional[float]


@dataclass(frozen=True)
class ValidationMetrics:
    """Scalar metrics for one curve comparison."""

    n_points: int
    normalized_l2: float
    bias_drag_counts: float
    mae_drag_counts: float
    max_abs_error: float
    max_abs_error_drag_counts: float
    max_abs_error_fn: float
    hump: ExtremumLocationMetrics
    hollow: ExtremumLocationMetrics

    def as_dict(self) -> dict[str, Optional[Union[float, int]]]:
        return {
            "n_points": self.n_points,
            "normalized_l2": self.normalized_l2,
            "bias_drag_counts": self.bias_drag_counts,
            "mae_drag_counts": self.mae_drag_counts,
            "max_abs_error": self.max_abs_error,
            "max_abs_error_drag_counts": self.max_abs_error_drag_counts,
            "max_abs_error_fn": self.max_abs_error_fn,
            "hump_observed_fn": self.hump.observed_fn,
            "hump_predicted_fn": self.hump.predicted_fn,
            "hump_location_error": self.hump.absolute_error,
            "hollow_observed_fn": self.hollow.observed_fn,
            "hollow_predicted_fn": self.hollow.predicted_fn,
            "hollow_location_error": self.hollow.absolute_error,
        }


@dataclass(frozen=True)
class ValidationReport:
    """Aligned curves, errors and metrics for one validation comparison."""

    observation: ValidationSeries
    prediction: ValidationSeries
    prediction_at_observations: NDArray[np.float64]
    errors: NDArray[np.float64]
    metrics: ValidationMetrics

    def __post_init__(self) -> None:
        aligned = np.array(self.prediction_at_observations, dtype=float, copy=True)
        errors = np.array(self.errors, dtype=float, copy=True)
        aligned.setflags(write=False)
        errors.setflags(write=False)
        object.__setattr__(self, "prediction_at_observations", aligned)
        object.__setattr__(self, "errors", errors)

    def to_text(self) -> str:
        """Return a compact, reproducible plain-text report."""

        metric = self.metrics
        quantity = self.observation.quantity
        attitude = self.observation.attitude

        def location_line(name: str, item: ExtremumLocationMetrics) -> str:
            if item.absolute_error is None:
                return f"{name} location error in Fn: n/a"
            return (
                f"{name} location error in Fn: {item.absolute_error:.6g} "
                f"(observed {item.observed_fn:.6g}, predicted {item.predicted_fn:.6g})"
            )

        return "\n".join(
            [
                f"Validation: {self.prediction.label or 'prediction'} vs "
                f"{self.observation.label or 'observation'}",
                f"Quantity: {quantity.symbol} ({quantity.value})",
                f"Attitude: {attitude.value} ({attitude.code})",
                f"Points: {metric.n_points}",
                f"Normalized L2 error: {metric.normalized_l2:.6g}",
                f"Bias: {metric.bias_drag_counts:.6g} drag counts",
                f"MAE: {metric.mae_drag_counts:.6g} drag counts",
                f"Maximum absolute error: {metric.max_abs_error:.6g} "
                f"({metric.max_abs_error_drag_counts:.6g} drag counts) "
                f"at Fn={metric.max_abs_error_fn:.6g}",
                location_line("Hump", metric.hump),
                location_line("Hollow", metric.hollow),
            ]
        )


def _trapezoidal_point_weights(x: NDArray[np.float64]) -> NDArray[np.float64]:
    if x.size == 1:
        return np.ones(1, dtype=float)
    weights = np.empty_like(x)
    weights[0] = 0.5 * (x[1] - x[0])
    weights[-1] = 0.5 * (x[-1] - x[-2])
    if x.size > 2:
        weights[1:-1] = 0.5 * (x[2:] - x[:-2])
    return weights


def _local_extrema(
    x: NDArray[np.float64], y: NDArray[np.float64], kind: str
) -> list[tuple[float, float]]:
    if x.size < 3:
        return []
    if kind == "hump":
        indices = np.flatnonzero(
            (y[1:-1] > y[:-2]) & (y[1:-1] > y[2:])
        ) + 1
    elif kind == "hollow":
        indices = np.flatnonzero(
            (y[1:-1] < y[:-2]) & (y[1:-1] < y[2:])
        ) + 1
    else:  # pragma: no cover - internal programming error
        raise ValueError(f"unknown extremum kind {kind!r}")
    return [(float(x[index]), float(y[index])) for index in indices]


def _extremum_location_metrics(
    observation: ValidationSeries,
    prediction: ValidationSeries,
    kind: str,
) -> ExtremumLocationMetrics:
    observed = _local_extrema(observation.fn, observation.coefficients, kind)
    predicted = _local_extrema(prediction.fn, prediction.coefficients, kind)
    predicted = [
        item
        for item in predicted
        if observation.fn[0] <= item[0] <= observation.fn[-1]
    ]
    if not observed:
        return ExtremumLocationMetrics(None, None, None)

    if kind == "hump":
        observed_fn, _ = max(observed, key=lambda item: item[1])
    else:
        observed_fn, _ = min(observed, key=lambda item: item[1])

    if not predicted:
        return ExtremumLocationMetrics(observed_fn, None, None)
    predicted_fn, _ = min(predicted, key=lambda item: abs(item[0] - observed_fn))
    return ExtremumLocationMetrics(
        observed_fn, predicted_fn, abs(predicted_fn - observed_fn)
    )


def _check_compatible(
    observation: ValidationSeries, prediction: ValidationSeries
) -> None:
    if observation.quantity is not prediction.quantity:
        raise IncompatibleValidationData(
            "resistance quantities differ: "
            f"observation is {observation.quantity.symbol} "
            f"({observation.quantity.value}), prediction is "
            f"{prediction.quantity.symbol} ({prediction.quantity.value})"
        )
    if observation.attitude is not prediction.attitude:
        raise IncompatibleValidationData(
            "hull attitudes differ: "
            f"observation is {observation.attitude.value} "
            f"({observation.attitude.code}), prediction is "
            f"{prediction.attitude.value} ({prediction.attitude.code})"
        )


def compare_series(
    observation: ValidationSeries, prediction: ValidationSeries
) -> ValidationReport:
    """Score a prediction against observations after strict semantic checks.

    Prediction values are linearly interpolated to the experimental Froude
    numbers. Extrapolation is never performed. Curve-integrated metrics use
    trapezoidal point weights so that irregularly sampled regions do not gain
    disproportionate influence merely by containing more points.
    """

    _check_compatible(observation, prediction)
    if (
        prediction.fn[0] > observation.fn[0]
        or prediction.fn[-1] < observation.fn[-1]
    ):
        raise IncompatibleValidationData(
            "prediction does not cover the complete observed Froude-number range; "
            "extrapolation is not allowed"
        )

    aligned = np.interp(
        observation.fn, prediction.fn, prediction.coefficients
    )
    errors = aligned - observation.coefficients
    weights = _trapezoidal_point_weights(observation.fn)
    weight_sum = float(np.sum(weights))

    numerator = float(np.sum(weights * errors**2))
    denominator = float(np.sum(weights * observation.coefficients**2))
    if denominator == 0.0:
        normalized_l2 = 0.0 if numerator == 0.0 else float("inf")
    else:
        normalized_l2 = float(np.sqrt(numerator / denominator))

    bias_drag_counts = float(np.sum(weights * errors) / weight_sum / DRAG_COUNT)
    mae_drag_counts = float(
        np.sum(weights * np.abs(errors)) / weight_sum / DRAG_COUNT
    )
    maximum_index = int(np.argmax(np.abs(errors)))
    max_abs_error = float(abs(errors[maximum_index]))

    metrics = ValidationMetrics(
        n_points=int(observation.fn.size),
        normalized_l2=normalized_l2,
        bias_drag_counts=bias_drag_counts,
        mae_drag_counts=mae_drag_counts,
        max_abs_error=max_abs_error,
        max_abs_error_drag_counts=max_abs_error / DRAG_COUNT,
        max_abs_error_fn=float(observation.fn[maximum_index]),
        hump=_extremum_location_metrics(observation, prediction, "hump"),
        hollow=_extremum_location_metrics(observation, prediction, "hollow"),
    )
    return ValidationReport(observation, prediction, aligned, errors, metrics)


def score_values(
    observation: ValidationSeries,
    predicted_froude_numbers: ArrayLike,
    predicted_coefficients: ArrayLike,
    *,
    quantity: Union[ResistanceQuantity, str],
    attitude: Union[HullAttitude, str],
    label: str = "prediction",
) -> ValidationReport:
    """Convenience wrapper for scoring raw prediction arrays.

    Quantity and attitude remain mandatory to retain the same semantic safety
    as :func:`compare_series`.
    """

    prediction = ValidationSeries(
        predicted_froude_numbers,
        predicted_coefficients,
        quantity=quantity,
        attitude=attitude,
        label=label,
    )
    return compare_series(observation, prediction)


__all__ = [
    "DRAG_COUNT",
    "ExtremumLocationMetrics",
    "HullAttitude",
    "IncompatibleValidationData",
    "ResistanceQuantity",
    "ValidationMetrics",
    "ValidationReport",
    "ValidationSeries",
    "compare_series",
    "score_values",
]
