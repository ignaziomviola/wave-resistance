"""Hull-offset input and validation for low-order wave-resistance models.

The coordinate convention is deliberately explicit: ``x`` increases from the
aft end to the forward end, ``z`` is positive downwards from the undisturbed
waterline, and ``half_breadths`` is the non-negative transverse coordinate of
one side of a port/starboard-symmetric hull.  Consequently, the tensor of
offsets has shape ``(len(x), len(z))``.
"""

from __future__ import annotations

import csv
import json
import math
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


PathLike = Union[str, os.PathLike]


class HullGeometryWarning(UserWarning):
    """Warning for valid geometry that may be unsuitable for a slender model."""


@dataclass(frozen=True)
class HullMetadata:
    """Versioned SI metadata required by the public hull-input contract.

    ``beam_m`` and ``draft_m`` are optional because they can be recovered from
    a complete offset table.  ``wetted_area_m2`` is mandatory because the
    reported resistance coefficient uses it as its reference area.
    """

    schema_version: str
    name: str
    length_ref_m: float
    length_ref_kind: str
    wetted_area_m2: float
    beam_m: Optional[float] = None
    draft_m: Optional[float] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", str(self.schema_version))
        if not self.schema_version.strip():
            raise ValueError("schema_version must be non-empty")
        for key in ("length_ref_m", "wetted_area_m2"):
            value = float(getattr(self, key))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError("{} must be finite and strictly positive".format(key))
            object.__setattr__(self, key, value)
        for key in ("beam_m", "draft_m"):
            raw_value = getattr(self, key)
            if raw_value is not None:
                value = float(raw_value)
                if not math.isfinite(value) or value <= 0.0:
                    raise ValueError("{} must be finite and strictly positive".format(key))
                object.__setattr__(self, key, value)
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string")
        if not isinstance(self.length_ref_kind, str) or not self.length_ref_kind.strip():
            raise ValueError("length_ref_kind must be a non-empty string")

    @property
    def length(self) -> float:
        """Compatibility alias for ``length_ref_m``."""

        return self.length_ref_m

    @property
    def beam(self) -> float:
        if self.beam_m is None:
            raise AttributeError("beam_m is inferred by OffsetHull and is absent from metadata")
        return self.beam_m

    @property
    def draft(self) -> float:
        if self.draft_m is None:
            raise AttributeError("draft_m is inferred by OffsetHull and is absent from metadata")
        return self.draft_m

    @property
    def beam_to_length(self) -> float:
        return self.beam / self.length_ref_m

    @property
    def draft_to_length(self) -> float:
        return self.draft / self.length_ref_m

    def as_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "schema_version": self.schema_version,
            "name": self.name,
            "length_ref_m": self.length_ref_m,
            "length_ref_kind": self.length_ref_kind,
            "wetted_area_m2": self.wetted_area_m2,
        }
        if self.beam_m is not None:
            result["beam_m"] = self.beam_m
        if self.draft_m is not None:
            result["draft_m"] = self.draft_m
        return result

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "HullMetadata":
        """Build metadata from the required, versioned public JSON fields."""

        lowered = {str(key).strip().lower(): value for key, value in values.items()}

        def required(key: str) -> Any:
            if key not in lowered:
                raise ValueError("metadata is missing '{}'".format(key))
            return lowered[key]

        return cls(
            schema_version=str(required("schema_version")),
            name=str(required("name")),
            length_ref_m=float(required("length_ref_m")),
            length_ref_kind=str(required("length_ref_kind")),
            wetted_area_m2=float(required("wetted_area_m2")),
            beam_m=(float(lowered["beam_m"]) if lowered.get("beam_m") is not None else None),
            draft_m=(
                float(lowered["draft_m"]) if lowered.get("draft_m") is not None else None
            ),
        )

    @classmethod
    def from_json(cls, path: PathLike) -> "HullMetadata":
        with Path(path).open("r", encoding="utf-8") as stream:
            document = json.load(stream)
        if not isinstance(document, Mapping):
            raise ValueError("metadata JSON must contain an object")
        return cls.from_mapping(document)


@dataclass(frozen=True)
class GeometryDiagnostics:
    """Hydrostatic and offset-slope checks derived from a tensor hull."""

    displaced_volume: float
    waterplane_area: float
    maximum_sectional_area: float
    block_coefficient: float
    waterplane_coefficient: float
    midship_coefficient: float
    prismatic_coefficient: float
    longitudinal_center_of_buoyancy: float
    vertical_center_of_buoyancy: float
    maximum_longitudinal_slope: float
    maximum_vertical_slope: float
    validation_warnings: Tuple[str, ...]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "displaced_volume": self.displaced_volume,
            "waterplane_area": self.waterplane_area,
            "maximum_sectional_area": self.maximum_sectional_area,
            "block_coefficient": self.block_coefficient,
            "waterplane_coefficient": self.waterplane_coefficient,
            "midship_coefficient": self.midship_coefficient,
            "prismatic_coefficient": self.prismatic_coefficient,
            "longitudinal_center_of_buoyancy": self.longitudinal_center_of_buoyancy,
            "vertical_center_of_buoyancy": self.vertical_center_of_buoyancy,
            "maximum_longitudinal_slope": self.maximum_longitudinal_slope,
            "maximum_vertical_slope": self.maximum_vertical_slope,
            "validation_warnings": self.validation_warnings,
        }


def _as_float_vector(values: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError("{} must be one-dimensional".format(name))
    if array.size < 2:
        raise ValueError("{} must contain at least two values".format(name))
    if not np.all(np.isfinite(array)):
        raise ValueError("{} contains a non-finite value".format(name))
    return np.array(array, dtype=float, copy=True)


def _validate_axis(values: np.ndarray, name: str) -> None:
    increments = np.diff(values)
    if np.any(increments == 0.0):
        raise ValueError("{} contains duplicate coordinates".format(name))
    if np.any(increments < 0.0):
        raise ValueError("{} must be strictly increasing".format(name))


def _integrate(values: np.ndarray, coordinates: np.ndarray, axis: int) -> np.ndarray:
    """Trapezoidal integration compatible with old and new NumPy releases."""

    left = np.take(values, indices=range(values.shape[axis] - 1), axis=axis)
    right = np.take(values, indices=range(1, values.shape[axis]), axis=axis)
    shape = [1] * values.ndim
    shape[axis] = coordinates.size - 1
    widths = np.diff(coordinates).reshape(shape)
    return np.sum(0.5 * (left + right) * widths, axis=axis)


def _metadata_or_infer(
    metadata: Optional[HullMetadata],
    x: np.ndarray,
    z: np.ndarray,
    y: np.ndarray,
    name: str,
) -> HullMetadata:
    if metadata is not None:
        if not isinstance(metadata, HullMetadata):
            raise TypeError("metadata must be a HullMetadata instance")
        return metadata
    raise ValueError(
        "metadata is required; provide the versioned metadata JSON contract for '{}'".format(
            name
        )
    )


def _wetted_surface_area(
    x: np.ndarray, z: np.ndarray, half_breadths: np.ndarray
) -> float:
    """Estimate both sides of the wetted shell represented by an offset tensor."""

    edge_order = 2 if x.size >= 3 and z.size >= 3 else 1
    dy_dx, dy_dz = np.gradient(
        half_breadths, x, z, edge_order=edge_order
    )
    area_density = np.sqrt(1.0 + dy_dx * dy_dx + dy_dz * dy_dz)
    area_by_station = _integrate(area_density, z, axis=1)
    return float(2.0 * _integrate(area_by_station, x, axis=0))


@dataclass
class OffsetHull:
    """Validated tensor-product table of symmetric hull half-breadths."""

    metadata: HullMetadata
    x: np.ndarray
    z: np.ndarray
    half_breadths: np.ndarray
    _beam_m: float = field(init=False, repr=False)
    _draft_m: float = field(init=False, repr=False)
    _diagnostics: GeometryDiagnostics = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, HullMetadata):
            raise TypeError("metadata must be a HullMetadata instance")

        x = _as_float_vector(self.x, "x")
        z = _as_float_vector(self.z, "z")
        y = np.asarray(self.half_breadths, dtype=float)
        if y.ndim != 2 or y.shape != (x.size, z.size):
            raise ValueError(
                "half_breadths must have shape ({}, {}), not {}".format(
                    x.size, z.size, y.shape
                )
            )
        if not np.all(np.isfinite(y)):
            raise ValueError("half_breadths contains a gap or non-finite value")
        y = np.array(y, dtype=float, copy=True)

        _validate_axis(x, "x")
        _validate_axis(z, "z")

        length = self.metadata.length_ref_m
        beam = (
            self.metadata.beam_m
            if self.metadata.beam_m is not None
            else 2.0 * float(np.max(y))
        )
        draft = self.metadata.draft_m if self.metadata.draft_m is not None else float(z[-1])
        if beam <= 0.0 or draft <= 0.0:
            raise ValueError("beam and draft inferred from offsets must be positive")
        self._beam_m = beam
        self._draft_m = draft
        coordinate_tolerance = 1.0e-9 * max(length, beam, draft, 1.0)
        breadth_tolerance = 1.0e-10 * max(beam, 1.0)

        if np.any(x < 0.0) or np.any(z < 0.0):
            raise ValueError("x and z coordinates cannot be negative")
        if np.any(y < 0.0):
            raise ValueError("half_breadths cannot be negative")

        if not math.isclose(x[0], 0.0, abs_tol=coordinate_tolerance):
            raise ValueError("x must start at the aft perpendicular x=0")
        if not math.isclose(x[-1], length, rel_tol=1.0e-9, abs_tol=coordinate_tolerance):
            raise ValueError("x must end at the forward perpendicular x=L")
        if not math.isclose(z[0], 0.0, abs_tol=coordinate_tolerance):
            raise ValueError("z must include the undisturbed waterline z=0")
        if not math.isclose(z[-1], draft, rel_tol=1.0e-9, abs_tol=coordinate_tolerance):
            raise ValueError("z must reach the design draft z=T")
        x[0], x[-1], z[0], z[-1] = 0.0, length, 0.0, draft

        if not np.all(np.abs(y[0, :]) <= breadth_tolerance) or not np.all(
            np.abs(y[-1, :]) <= breadth_tolerance
        ):
            raise ValueError("a displacement hull must have pointed aft and forward ends")
        if not np.all(np.abs(y[:, -1]) <= breadth_tolerance):
            raise ValueError("the deepest offset row must close on the centreplane")
        y[0, :] = 0.0
        y[-1, :] = 0.0
        y[:, -1] = 0.0

        if 2.0 * float(np.max(y)) > beam + breadth_tolerance:
            raise ValueError("offsets exceed the metadata beam")
        if x.size > 2:
            if np.any(y[1:-1, 0] <= breadth_tolerance):
                raise ValueError("the interior waterline contains a gap or zero breadth")
            for station_profile in y[1:-1, :]:
                positive = np.flatnonzero(station_profile > breadth_tolerance)
                if positive.size and np.any(
                    station_profile[positive[0] : positive[-1] + 1] <= breadth_tolerance
                ):
                    raise ValueError(
                        "an interior station closes and reopens, creating a hull-surface gap"
                    )
            sectional_areas = 2.0 * _integrate(y, z, axis=1)
            if np.any(sectional_areas[1:-1] <= breadth_tolerance * draft):
                raise ValueError("an interior station contains a gap or zero area")

        x.setflags(write=False)
        z.setflags(write=False)
        y.setflags(write=False)
        self.x = x
        self.z = z
        self.half_breadths = y
        self._diagnostics = self._build_diagnostics()
        if self._diagnostics.displaced_volume <= 0.0:
            raise ValueError("offsets must enclose positive displaced volume")
        if self._diagnostics.waterplane_area <= 0.0:
            raise ValueError("offsets must enclose positive waterplane area")
        for message in self._diagnostics.validation_warnings:
            warnings.warn(message, HullGeometryWarning, stacklevel=2)

    def _build_diagnostics(self) -> GeometryDiagnostics:
        m = self.metadata
        length = m.length_ref_m
        beam = self._beam_m
        draft = self._draft_m
        sectional_areas = self.sectional_areas
        volume = float(_integrate(sectional_areas, self.x, axis=0))
        waterline_breadth = 2.0 * self.half_breadths[:, 0]
        waterplane_area = float(_integrate(waterline_breadth, self.x, axis=0))
        maximum_section = float(np.max(sectional_areas))

        if volume > 0.0:
            lcb = float(_integrate(self.x * sectional_areas, self.x, axis=0) / volume)
            first_moment_z_by_station = 2.0 * _integrate(
                self.half_breadths * self.z[np.newaxis, :], self.z, axis=1
            )
            vcb = float(_integrate(first_moment_z_by_station, self.x, axis=0) / volume)
        else:
            lcb = float("nan")
            vcb = float("nan")

        dx_slopes = np.diff(self.half_breadths, axis=0) / np.diff(self.x)[:, None]
        dz_slopes = np.diff(self.half_breadths, axis=1) / np.diff(self.z)[None, :]
        maximum_dx_slope = float(np.max(np.abs(dx_slopes)))
        maximum_dz_slope = float(np.max(np.abs(dz_slopes)))

        messages: List[str] = []
        beam_to_length = beam / length
        if beam_to_length > 0.20:
            messages.append(
                (
                    "B/L={:.3g} exceeds 0.20; slender-body wave-resistance "
                    "assumptions may be inaccurate"
                ).format(beam_to_length)
            )
        represented_beam = 2.0 * float(np.max(self.half_breadths))
        if represented_beam < 0.98 * beam:
            messages.append(
                "the offset grid represents only {:.1%} of the metadata beam".format(
                    represented_beam / beam
                )
            )

        return GeometryDiagnostics(
            displaced_volume=volume,
            waterplane_area=waterplane_area,
            maximum_sectional_area=maximum_section,
            block_coefficient=volume / (length * beam * draft),
            waterplane_coefficient=waterplane_area / (length * beam),
            midship_coefficient=maximum_section / (beam * draft),
            prismatic_coefficient=(
                volume / (length * maximum_section) if maximum_section > 0.0 else float("nan")
            ),
            longitudinal_center_of_buoyancy=lcb,
            vertical_center_of_buoyancy=vcb,
            maximum_longitudinal_slope=maximum_dx_slope,
            maximum_vertical_slope=maximum_dz_slope,
            validation_warnings=tuple(messages),
        )

    @property
    def x_over_length(self) -> np.ndarray:
        return self.x / self.metadata.length_ref_m

    @property
    def x_m(self) -> np.ndarray:
        """Dimensional longitudinal coordinates in metres."""

        return self.x

    @property
    def z_m(self) -> np.ndarray:
        """Dimensional downward coordinates in metres."""

        return self.z

    @property
    def half_breadth_m(self) -> np.ndarray:
        """Dimensional half-breadth tensor in metres."""

        return self.half_breadths

    @property
    def beam_m(self) -> float:
        """Metadata beam or, when absent, the maximum breadth of the offsets."""

        return self._beam_m

    @property
    def draft_m(self) -> float:
        """Metadata draft or, when absent, the deepest offset coordinate."""

        return self._draft_m

    @property
    def wetted_area_m2(self) -> float:
        return self.metadata.wetted_area_m2

    @property
    def z_over_draft(self) -> np.ndarray:
        return self.z / self._draft_m

    @property
    def half_breadths_over_beam(self) -> np.ndarray:
        return self.half_breadths / self._beam_m

    @property
    def coordinates_over_length(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(x/L, z/L, y/L)`` for a common-length wave kernel."""

        length = self.metadata.length_ref_m
        return self.x / length, self.z / length, self.half_breadths / length

    @property
    def sectional_areas(self) -> np.ndarray:
        return 2.0 * _integrate(self.half_breadths, self.z, axis=1)

    @property
    def diagnostics(self) -> GeometryDiagnostics:
        return self._diagnostics

    @property
    def displaced_volume(self) -> float:
        return self._diagnostics.displaced_volume

    @property
    def waterplane_area(self) -> float:
        return self._diagnostics.waterplane_area

    @property
    def block_coefficient(self) -> float:
        return self._diagnostics.block_coefficient

    @classmethod
    def from_tensor(
        cls,
        metadata: HullMetadata,
        x: Sequence[float],
        z: Sequence[float],
        half_breadths: Sequence[Sequence[float]],
    ) -> "OffsetHull":
        """Construct and validate an already tensor-product offset table."""

        return cls(
            metadata=metadata,
            x=np.asarray(x, dtype=float),
            z=np.asarray(z, dtype=float),
            half_breadths=np.asarray(half_breadths, dtype=float),
        )

    @classmethod
    def from_irregular(
        cls,
        metadata: HullMetadata,
        x: Sequence[float],
        z: Sequence[float],
        half_breadths: Sequence[float],
        *,
        x_grid: Optional[Sequence[float]] = None,
        z_grid: Optional[Sequence[float]] = None,
    ) -> "OffsetHull":
        """Interpolate long-form station offsets to a tensor product grid.

        Every sampled station must span ``z=0`` to ``z=T``.  Interpolation is
        piecewise linear first down each station and then along each requested
        depth.  Extrapolation is never permitted.
        """

        if not isinstance(metadata, HullMetadata):
            raise TypeError("metadata must be a HullMetadata instance")
        raw_x = np.asarray(x, dtype=float)
        raw_z = np.asarray(z, dtype=float)
        raw_y = np.asarray(half_breadths, dtype=float)
        if raw_x.ndim != 1 or raw_z.ndim != 1 or raw_y.ndim != 1:
            raise ValueError("irregular x, z and half_breadths must be one-dimensional")
        if not (raw_x.size == raw_z.size == raw_y.size) or raw_x.size == 0:
            raise ValueError("irregular x, z and half_breadths must have equal non-zero length")
        if not np.all(np.isfinite(raw_x)) or not np.all(np.isfinite(raw_z)) or not np.all(
            np.isfinite(raw_y)
        ):
            raise ValueError("irregular offsets contain a gap or non-finite value")
        if np.any(raw_x < 0.0) or np.any(raw_z < 0.0) or np.any(raw_y < 0.0):
            raise ValueError("irregular offsets cannot contain negative coordinates or breadths")

        pairs = np.column_stack((raw_x, raw_z))
        if np.unique(pairs, axis=0).shape[0] != pairs.shape[0]:
            raise ValueError("irregular offsets contain a duplicate (x, z) sample")

        station_x = np.unique(raw_x)
        _validate_axis(station_x, "station x")
        inferred_draft = metadata.draft_m if metadata.draft_m is not None else float(np.max(raw_z))
        tolerance = 1.0e-9 * max(metadata.length_ref_m, inferred_draft, 1.0)
        if not math.isclose(station_x[0], 0.0, abs_tol=tolerance) or not math.isclose(
            station_x[-1], metadata.length_ref_m, rel_tol=1.0e-9, abs_tol=tolerance
        ):
            raise ValueError("irregular stations must cover the full interval 0 <= x <= L")

        target_x = station_x if x_grid is None else _as_float_vector(x_grid, "x_grid")
        target_z = np.unique(raw_z) if z_grid is None else _as_float_vector(z_grid, "z_grid")
        _validate_axis(target_x, "x_grid")
        _validate_axis(target_z, "z_grid")
        if not math.isclose(target_x[0], 0.0, abs_tol=tolerance) or not math.isclose(
            target_x[-1], metadata.length_ref_m, rel_tol=1.0e-9, abs_tol=tolerance
        ):
            raise ValueError("x_grid must cover exactly 0 <= x <= L")
        if not math.isclose(target_z[0], 0.0, abs_tol=tolerance) or not math.isclose(
            target_z[-1], inferred_draft, rel_tol=1.0e-9, abs_tol=tolerance
        ):
            raise ValueError("z_grid must cover exactly 0 <= z <= T")

        station_tensor = np.empty((station_x.size, target_z.size), dtype=float)
        for station_index, coordinate in enumerate(station_x):
            selected = raw_x == coordinate
            profile_z = raw_z[selected]
            profile_y = raw_y[selected]
            order = np.argsort(profile_z)
            profile_z = profile_z[order]
            profile_y = profile_y[order]
            if profile_z.size < 2:
                raise ValueError("each station requires at least two depth samples")
            if not math.isclose(profile_z[0], 0.0, abs_tol=tolerance) or not math.isclose(
                profile_z[-1], inferred_draft, rel_tol=1.0e-9, abs_tol=tolerance
            ):
                raise ValueError(
                    "station x={} has a vertical coverage gap; it must span z=0 to z=T".format(
                        coordinate
                    )
                )
            station_tensor[station_index, :] = np.interp(target_z, profile_z, profile_y)

        tensor = np.empty((target_x.size, target_z.size), dtype=float)
        for depth_index in range(target_z.size):
            tensor[:, depth_index] = np.interp(
                target_x, station_x, station_tensor[:, depth_index]
            )
        return cls(metadata, target_x, target_z, tensor)

    @classmethod
    def from_csv(
        cls,
        offsets_path: PathLike,
        metadata_json_path: Optional[Union[PathLike, HullMetadata]] = None,
        *,
        x_grid: Optional[Sequence[float]] = None,
        z_grid: Optional[Sequence[float]] = None,
    ) -> "OffsetHull":
        """Load a long-form offsets CSV plus its versioned metadata JSON.

        Canonical columns are ``x_m,z_m,half_breadth_m``; concise aliases are
        accepted for convenience.  The second positional argument may also be
        an already parsed :class:`HullMetadata` instance.
        """

        csv_path = Path(offsets_path)
        with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
            lines = [line for line in stream if line.strip() and not line.lstrip().startswith("#")]
        reader = csv.DictReader(lines)
        if reader.fieldnames is None:
            raise ValueError("CSV file must contain a header row")
        normalized = {field.strip().lower(): field for field in reader.fieldnames}

        def column(aliases: Sequence[str]) -> str:
            for alias in aliases:
                if alias in normalized:
                    return normalized[alias]
            raise ValueError("CSV is missing '{}' column".format(aliases[0]))

        x_column = column(("x_m", "x", "longitudinal", "station"))
        z_column = column(("z_m", "z", "depth"))
        y_column = column(
            ("half_breadth_m", "half_breadth", "halfbreadth", "y", "offset")
        )
        raw_x: List[float] = []
        raw_z: List[float] = []
        raw_y: List[float] = []
        for row_number, row in enumerate(reader, start=2):
            try:
                raw_x.append(float(row[x_column]))
                raw_z.append(float(row[z_column]))
                raw_y.append(float(row[y_column]))
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "invalid numeric value on CSV row {}".format(row_number)
                ) from error
        x_array = np.asarray(raw_x, dtype=float)
        z_array = np.asarray(raw_z, dtype=float)
        y_array = np.asarray(raw_y, dtype=float)
        if isinstance(metadata_json_path, HullMetadata):
            metadata = metadata_json_path
        elif metadata_json_path is not None:
            metadata = HullMetadata.from_json(metadata_json_path)
        else:
            metadata = None
        resolved_metadata = _metadata_or_infer(
            metadata, x_array, z_array, y_array, csv_path.stem
        )
        return cls.from_irregular(
            resolved_metadata,
            x_array,
            z_array,
            y_array,
            x_grid=x_grid,
            z_grid=z_grid,
        )

    @classmethod
    def from_json(
        cls,
        path: PathLike,
        metadata: Optional[HullMetadata] = None,
        *,
        x_grid: Optional[Sequence[float]] = None,
        z_grid: Optional[Sequence[float]] = None,
    ) -> "OffsetHull":
        """Load either tensor-form or long-form offsets from JSON.

        Tensor form uses canonical top-level ``x_m``, ``z_m`` and
        ``half_breadth_m`` arrays (concise aliases are also accepted).
        Long form uses ``offsets`` as a list of objects with
        ``x``, ``z`` and ``half_breadth`` fields.  Either may include a
        top-level ``metadata`` object.
        """

        json_path = Path(path)
        with json_path.open("r", encoding="utf-8") as stream:
            document = json.load(stream)
        if not isinstance(document, Mapping):
            raise ValueError("JSON hull document must be an object")

        resolved_metadata = metadata
        if resolved_metadata is None and "metadata" in document:
            embedded = document["metadata"]
            if not isinstance(embedded, Mapping):
                raise ValueError("JSON metadata must be an object")
            resolved_metadata = HullMetadata.from_mapping(embedded)

        canonical_tensor = all(
            key in document for key in ("x_m", "z_m", "half_breadth_m")
        )
        concise_tensor = all(key in document for key in ("x", "z", "half_breadths"))
        if canonical_tensor or concise_tensor:
            if x_grid is not None or z_grid is not None:
                raise ValueError("x_grid and z_grid are only valid for long-form JSON offsets")
            tensor_x = np.asarray(document["x_m" if canonical_tensor else "x"], dtype=float)
            tensor_z = np.asarray(document["z_m" if canonical_tensor else "z"], dtype=float)
            tensor_y = np.asarray(
                document["half_breadth_m" if canonical_tensor else "half_breadths"],
                dtype=float,
            )
            resolved_metadata = _metadata_or_infer(
                resolved_metadata,
                tensor_x,
                tensor_z,
                tensor_y,
                json_path.stem,
            )
            return cls(resolved_metadata, tensor_x, tensor_z, tensor_y)

        offsets = document.get("offsets")
        if not isinstance(offsets, list) or not offsets:
            raise ValueError(
                "JSON must contain tensor arrays or a non-empty long-form 'offsets' list"
            )
        raw_x = []
        raw_z = []
        raw_y = []
        for index, offset in enumerate(offsets):
            if not isinstance(offset, Mapping):
                raise ValueError("JSON offset {} must be an object".format(index))
            try:
                breadth = offset.get(
                    "half_breadth_m", offset.get("half_breadth", offset.get("y"))
                )
                raw_x.append(float(offset.get("x_m", offset.get("x"))))
                raw_z.append(float(offset.get("z_m", offset.get("z"))))
                raw_y.append(float(breadth))
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("invalid JSON offset {}".format(index)) from error
        x_array = np.asarray(raw_x, dtype=float)
        z_array = np.asarray(raw_z, dtype=float)
        y_array = np.asarray(raw_y, dtype=float)
        resolved_metadata = _metadata_or_infer(
            resolved_metadata, x_array, z_array, y_array, json_path.stem
        )
        return cls.from_irregular(
            resolved_metadata,
            x_array,
            z_array,
            y_array,
            x_grid=x_grid,
            z_grid=z_grid,
        )

    @classmethod
    def wigley(
        cls,
        length_m: float = 1.0,
        beam_m: Optional[float] = None,
        draft_m: Optional[float] = None,
        nx: int = 81,
        nz: int = 33,
        name: str = "Wigley hull",
    ) -> "OffsetHull":
        return wigley_hull(
            length_m=length_m,
            beam_m=beam_m,
            draft_m=draft_m,
            nx=nx,
            nz=nz,
            name=name,
        )

    @classmethod
    def sailing_yacht(
        cls,
        length_waterline_m: float,
        waterline_beam_m: float,
        displaced_volume_m3: float,
        **kwargs: Any,
    ) -> "OffsetHull":
        """Construct a parametric modern sailing-yacht canoe body.

        This is a convenience alias for :func:`sailing_yacht_hull`.  Shape
        parameters such as ``widest_station_fraction`` may be supplied as
        keyword arguments.
        """

        return sailing_yacht_hull(
            length_waterline_m=length_waterline_m,
            waterline_beam_m=waterline_beam_m,
            displaced_volume_m3=displaced_volume_m3,
            **kwargs,
        )

    @classmethod
    def dufour_39_approx(cls, **kwargs: Any) -> "OffsetHull":
        """Construct the refined smooth 2026 Dufour 39 analytical surrogate."""

        return dufour_39_approx_hull(**kwargs)


def wigley_hull(
    length_m: float = 1.0,
    beam_m: Optional[float] = None,
    draft_m: Optional[float] = None,
    nx: int = 81,
    nz: int = 33,
    name: str = "Wigley hull",
) -> OffsetHull:
    """Construct the canonical parabolic Wigley hull on a regular SI grid.

    The default dimensions are ``B=L/10`` and ``T=B/1.6``.  Wetted area is
    integrated from the same offset tensor and stored in the public metadata.
    """

    if isinstance(nx, bool) or not isinstance(nx, (int, np.integer)) or nx < 3:
        raise ValueError("nx must be an integer of at least 3")
    if isinstance(nz, bool) or not isinstance(nz, (int, np.integer)) or nz < 2:
        raise ValueError("nz must be an integer of at least 2")
    length_value = float(length_m)
    if not math.isfinite(length_value) or length_value <= 0.0:
        raise ValueError("length_m must be finite and strictly positive")
    beam_value = length_value / 10.0 if beam_m is None else float(beam_m)
    draft_value = beam_value / 1.6 if draft_m is None else float(draft_m)
    if not math.isfinite(beam_value) or beam_value <= 0.0:
        raise ValueError("beam_m must be finite and strictly positive")
    if not math.isfinite(draft_value) or draft_value <= 0.0:
        raise ValueError("draft_m must be finite and strictly positive")

    x = np.linspace(0.0, length_value, int(nx))
    z = np.linspace(0.0, draft_value, int(nz))
    longitudinal = 1.0 - (2.0 * x / length_value - 1.0) ** 2
    vertical = 1.0 - (z / draft_value) ** 2
    half_breadths = 0.5 * beam_value * longitudinal[:, None] * vertical[None, :]
    metadata = HullMetadata(
        schema_version="1.0",
        name=name,
        length_ref_m=length_value,
        length_ref_kind="LWL",
        wetted_area_m2=_wetted_surface_area(x, z, half_breadths),
        beam_m=beam_value,
        draft_m=draft_value,
    )
    return OffsetHull(metadata, x, z, half_breadths)


def sailing_yacht_hull(
    length_waterline_m: float,
    waterline_beam_m: float,
    displaced_volume_m3: float,
    *,
    nx: int = 161,
    nz: int = 65,
    widest_station_fraction: float = 0.43,
    aft_waterline_exponent: float = 0.55,
    forward_waterline_exponent: float = 0.95,
    section_power: float = 2.0,
    midship_flare_power: float = 3.2,
    end_flare_increment: float = 2.0,
    bow_vee_increment: float = 1.0,
    bow_vee_start_fraction: float = 0.62,
    name: str = "Approximate modern sailing-yacht canoe body",
) -> OffsetHull:
    r"""Construct a transparent parametric sailing-yacht canoe body.

    No measured or proprietary lines are used.  With ``xi=x/L`` and
    ``eta=z/T``, the half-breadth is

    .. math::

       y = \frac{B_{WL}}{2}F(\xi)
           \left(1-\eta^p\right)^{q(\xi)},

    where ``F`` consists of two sine-power curves joined with zero slope at
    ``widest_station_fraction``.  The aft and forward exponents independently
    control the run and entrance.  The section exponent is

    .. math::

       q(\xi)=q_m+q_e[1-F(\xi)]
       +q_b\left[\max\left(\frac{\xi-\xi_b}{1-\xi_b},0\right)\right]^2.

    Thus sections become less full near both ends and acquire extra V-shape
    towards the bow.  The canoe-body draft ``T`` is derived so that trapezoidal
    integration of the returned offset tensor gives exactly
    ``displaced_volume_m3`` (apart from roundoff).  This makes the volume
    calibration reproducible at every requested grid resolution.

    The tensor closes on the centreplane at the deepest row and is pointed at
    both endpoints, as required by :class:`OffsetHull` and the present Michell
    implementation.  Real transoms, chines, keels, rudders and above-water
    topsides are not represented.  This smooth bare-canoe surrogate is therefore
    suitable for sensitivity studies, not construction, stability work or an
    assertion of vessel-identical resistance.
    """

    def positive_finite(value: float, parameter: str) -> float:
        resolved = float(value)
        if not math.isfinite(resolved) or resolved <= 0.0:
            raise ValueError("{} must be finite and strictly positive".format(parameter))
        return resolved

    if isinstance(nx, bool) or not isinstance(nx, (int, np.integer)) or nx < 3:
        raise ValueError("nx must be an integer of at least 3")
    if isinstance(nz, bool) or not isinstance(nz, (int, np.integer)) or nz < 2:
        raise ValueError("nz must be an integer of at least 2")

    length = positive_finite(length_waterline_m, "length_waterline_m")
    beam = positive_finite(waterline_beam_m, "waterline_beam_m")
    target_volume = positive_finite(displaced_volume_m3, "displaced_volume_m3")
    maximum_location = float(widest_station_fraction)
    if not math.isfinite(maximum_location) or not 0.0 < maximum_location < 1.0:
        raise ValueError("widest_station_fraction must lie strictly between 0 and 1")
    aft_exponent = positive_finite(aft_waterline_exponent, "aft_waterline_exponent")
    forward_exponent = positive_finite(
        forward_waterline_exponent, "forward_waterline_exponent"
    )
    vertical_power = positive_finite(section_power, "section_power")
    base_flare = positive_finite(midship_flare_power, "midship_flare_power")
    end_flare = float(end_flare_increment)
    bow_vee = float(bow_vee_increment)
    if not math.isfinite(end_flare) or end_flare < 0.0:
        raise ValueError("end_flare_increment must be finite and non-negative")
    if not math.isfinite(bow_vee) or bow_vee < 0.0:
        raise ValueError("bow_vee_increment must be finite and non-negative")
    bow_vee_start = float(bow_vee_start_fraction)
    if not math.isfinite(bow_vee_start) or not 0.0 <= bow_vee_start < 1.0:
        raise ValueError("bow_vee_start_fraction must lie in [0, 1)")

    xi = np.linspace(0.0, 1.0, int(nx))
    eta = np.linspace(0.0, 1.0, int(nz))
    longitudinal = np.empty_like(xi)
    aft = xi <= maximum_location
    aft_argument = np.clip(xi[aft] / maximum_location, 0.0, 1.0)
    forward_argument = np.clip(
        (1.0 - xi[~aft]) / (1.0 - maximum_location), 0.0, 1.0
    )
    longitudinal[aft] = np.sin(0.5 * np.pi * aft_argument) ** aft_exponent
    longitudinal[~aft] = (
        np.sin(0.5 * np.pi * forward_argument) ** forward_exponent
    )
    # Force exact closures independently of floating-point sine evaluation.
    longitudinal[0] = 0.0
    longitudinal[-1] = 0.0

    bow_progress = np.clip(
        (xi - bow_vee_start) / (1.0 - bow_vee_start), 0.0, 1.0
    )
    flare = (
        base_flare
        + end_flare * (1.0 - longitudinal)
        + bow_vee * bow_progress * bow_progress
    )
    vertical_base = np.maximum(1.0 - eta**vertical_power, 0.0)
    shape = longitudinal[:, None] * vertical_base[None, :] ** flare[:, None]
    shape[0, :] = 0.0
    shape[-1, :] = 0.0
    shape[:, -1] = 0.0

    # Since y=(B/2)*shape, x=L*xi and z=T*eta, volume is
    # L*B*T times this dimensionless double integral.  Evaluating it with the
    # same trapezoidal rule as OffsetHull makes the discrete target exact.
    section_coefficients = _integrate(shape, eta, axis=1)
    volume_coefficient = float(_integrate(section_coefficients, xi, axis=0))
    if not math.isfinite(volume_coefficient) or volume_coefficient <= 0.0:
        raise ValueError("shape parameters do not enclose a positive volume")
    draft = target_volume / (length * beam * volume_coefficient)
    if not math.isfinite(draft) or draft <= 0.0:
        raise ValueError("derived canoe-body draft is not finite and positive")

    x = length * xi
    z = draft * eta
    half_breadths = 0.5 * beam * shape
    metadata = HullMetadata(
        schema_version="1.0",
        name=name,
        length_ref_m=length,
        length_ref_kind="LWL",
        wetted_area_m2=_wetted_surface_area(x, z, half_breadths),
        beam_m=beam,
        draft_m=draft,
    )
    return OffsetHull(metadata, x, z, half_breadths)


def dufour_39_approx_hull(
    *,
    nx: int = 201,
    nz: int = 129,
    waterline_beam_m: float = 3.70,
    canoe_body_volume_m3: float = 8.00,
    maximum_hull_beam_m: float = 4.10,
    name: str = "Dufour 39 refined smooth bare canoe body (not builder lines)",
) -> OffsetHull:
    """Return a Michell-compatible approximation to the 2026 Dufour 39.

    Only public principal particulars inform this named preset: 12.00 m LOA,
    11.27 m hull length, 10.50 m LWL, 4.10 m maximum hull beam, 8600 kg light
    displacement and 1.95 m total draft.  These values are published by Dufour
    at https://www.dufour-yachts.com/en/sailboats/dufour-39/ (accessed August
    2026).  No builder offsets, drawings or reverse-engineered surface data are
    used.

    Michell's model needs the submerged bare canoe body, whose particulars are
    not published.  The defaults therefore make two explicit assumptions:
    ``BWL=3.70 m`` and bare-canoe volume ``8.00 m3``.  The latter is slightly
    below ``8600/1025 = 8.39 m3`` so that keel and rudder volume are excluded.
    The draft is derived to match that volume exactly on the requested grid.

    The refined default uses 129 cosine-spaced vertical levels and a smoothly
    varying superelliptic section family.  Midbody sections have rounded bilges,
    while the bow becomes progressively more V-shaped without introducing a
    chine or a slope discontinuity.  The full bow and broad run reproduce the
    published design description qualitatively; they are analytical assumptions,
    not traced geometry.  The real open transom is forced to a pointed aft
    closure; the keel, rudder, topsides and full 1.95 m draft are not modelled.

    The represented ``BWL/LWL`` is about 0.35, well outside the package's
    nominal slender-hull diagnostic threshold.  A converged Michell integral
    for this surrogate should not be interpreted as validated Dufour 39 drag.
    """

    if isinstance(nx, bool) or not isinstance(nx, (int, np.integer)) or nx < 3:
        raise ValueError("nx must be an integer of at least 3")
    if isinstance(nz, bool) or not isinstance(nz, (int, np.integer)) or nz < 9:
        raise ValueError("nz must be an integer of at least 9")
    waterline_beam = float(waterline_beam_m)
    target_volume = float(canoe_body_volume_m3)
    maximum_beam = float(maximum_hull_beam_m)
    for value, parameter in (
        (waterline_beam, "waterline_beam_m"),
        (target_volume, "canoe_body_volume_m3"),
        (maximum_beam, "maximum_hull_beam_m"),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("{} must be finite and strictly positive".format(parameter))
    if waterline_beam > maximum_beam:
        raise ValueError("waterline_beam_m cannot exceed maximum_hull_beam_m")

    length = 10.50
    xi = np.linspace(0.0, 1.0, int(nx))
    # Cosine spacing resolves the waterline and local centreline closures more
    # strongly than a uniform grid, without a privileged chine level.
    vertical_parameter = np.linspace(0.0, 1.0, int(nz))
    eta = 0.5 * (1.0 - np.cos(np.pi * vertical_parameter))

    widest_station = 0.43
    longitudinal = np.empty_like(xi)
    aft = xi <= widest_station
    longitudinal[aft] = np.sin(
        0.5 * np.pi * xi[aft] / widest_station
    ) ** 0.55
    longitudinal[~aft] = np.sin(
        0.5 * np.pi * (1.0 - xi[~aft]) / (1.0 - widest_station)
    ) ** 0.65
    longitudinal[0] = 0.0
    longitudinal[-1] = 0.0

    bow_progress = np.clip((xi - 0.62) / 0.38, 0.0, 1.0)
    # A simple rocker raises the canoe-body centreline towards both ends.
    # Multiplication by the waterline curve preserves continuous pointed ends.
    local_draft_fraction = longitudinal**0.45
    eta_grid = eta[np.newaxis, :]
    local_eta = eta_grid / np.maximum(local_draft_fraction[:, np.newaxis], 1.0e-15)
    # Smooth superelliptic sections: p controls upper-body fullness and q the
    # bilge/bottom transition.  Both vary continuously in x.  The bow is more
    # V-shaped, but there is no piecewise join and hence no artificial chine.
    section_power = (
        2.15
        - 0.30 * (1.0 - longitudinal)
        - 0.28 * bow_progress * bow_progress
    )
    bottom_exponent = (
        0.62
        + 0.16 * (1.0 - longitudinal)
        + 0.16 * bow_progress * bow_progress
    )
    vertical_base = np.maximum(
        1.0 - local_eta ** section_power[:, np.newaxis], 0.0
    )
    section = vertical_base ** bottom_exponent[:, np.newaxis]
    section = np.where(local_eta <= 1.0, section, 0.0)
    shape = longitudinal[:, np.newaxis] * section
    shape[0, :] = 0.0
    shape[-1, :] = 0.0
    shape[:, -1] = 0.0

    section_coefficients = _integrate(shape, eta, axis=1)
    volume_coefficient = float(_integrate(section_coefficients, xi, axis=0))
    draft = target_volume / (length * waterline_beam * volume_coefficient)
    x = length * xi
    z = draft * eta
    half_breadths = 0.5 * waterline_beam * shape
    metadata = HullMetadata(
        schema_version="1.0",
        name=name,
        length_ref_m=length,
        length_ref_kind="LWL",
        wetted_area_m2=_wetted_surface_area(x, z, half_breadths),
        beam_m=maximum_beam,
        draft_m=draft,
    )
    return OffsetHull(metadata, x, z, half_breadths)


__all__ = [
    "dufour_39_approx_hull",
    "GeometryDiagnostics",
    "HullGeometryWarning",
    "HullMetadata",
    "OffsetHull",
    "sailing_yacht_hull",
    "wigley_hull",
]
