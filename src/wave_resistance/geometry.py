"""Sectioned hull geometry, hydrostatics, and triangular panel generation.

The body-fixed coordinate system is right-handed.  ``x`` increases with the
oncoming flow (bow to stern), ``y`` is positive to starboard, and ``z`` is
positive upwards with the undisturbed free surface at ``z = 0``.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

_trapezoid = getattr(np, "trapezoid", np.trapz)


@dataclass(frozen=True)
class HullHydrostatics:
    displacement_volume_m3: float
    longitudinal_center_of_buoyancy_m: float
    waterplane_area_m2: float
    wetted_area_m2: float
    maximum_waterline_beam_m: float
    maximum_canoe_draft_m: float


@dataclass(frozen=True)
class TriMesh:
    """Flat triangular panels on the starboard half of a symmetric hull."""

    vertices: np.ndarray
    faces: np.ndarray
    centroids: np.ndarray
    normals: np.ndarray
    areas: np.ndarray

    @classmethod
    def from_vertices_faces(cls, vertices: np.ndarray, faces: np.ndarray) -> "TriMesh":
        vertices_array = np.asarray(vertices, dtype=float)
        faces_array = np.asarray(faces, dtype=int)
        if vertices_array.ndim != 2 or vertices_array.shape[1] != 3:
            raise ValueError("vertices must have shape (n, 3)")
        if faces_array.ndim != 2 or faces_array.shape[1] != 3:
            raise ValueError("faces must have shape (m, 3)")
        triangles = vertices_array[faces_array]
        cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        twice_area = np.linalg.norm(cross, axis=1)
        keep = twice_area > 1.0e-12
        faces_array = faces_array[keep]
        cross = cross[keep]
        twice_area = twice_area[keep]
        normals = cross / twice_area[:, None]
        centroids = vertices_array[faces_array].mean(axis=1)
        return cls(
            vertices=vertices_array,
            faces=faces_array,
            centroids=centroids,
            normals=normals,
            areas=0.5 * twice_area,
        )


@dataclass(frozen=True)
class HullOffsets:
    """Symmetric hull represented by ordered half-sections.

    ``half_breadth_m`` and ``z_m`` have shape ``(n_stations, n_vertical)``.
    Each station is ordered from the waterline to the keel/centreline.
    """

    x_m: np.ndarray
    z_m: np.ndarray
    half_breadth_m: np.ndarray
    name: str = "Unnamed hull"
    length_ref_m: Optional[float] = None
    provenance: str = ""

    def __post_init__(self) -> None:
        x = np.asarray(self.x_m, dtype=float)
        z = np.asarray(self.z_m, dtype=float)
        y = np.asarray(self.half_breadth_m, dtype=float)
        object.__setattr__(self, "x_m", x)
        object.__setattr__(self, "z_m", z)
        object.__setattr__(self, "half_breadth_m", y)
        if x.ndim != 1 or x.size < 3 or not np.all(np.diff(x) > 0.0):
            raise ValueError("x_m must contain at least three strictly increasing stations")
        if z.shape != y.shape or z.ndim != 2 or z.shape[0] != x.size or z.shape[1] < 3:
            raise ValueError("z_m and half_breadth_m must have shape (n_stations, n_vertical>=3)")
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(z)) or not np.all(np.isfinite(y)):
            raise ValueError("hull coordinates must be finite")
        if np.any(y < -1.0e-12):
            raise ValueError("half breadths cannot be negative")
        if not np.allclose(z[:, 0], 0.0, atol=1.0e-10):
            raise ValueError("the first point of every section must lie on z=0")
        if np.any(np.diff(z, axis=1) > 1.0e-12) or np.any(z > 1.0e-12):
            raise ValueError("section z coordinates must progress downwards from the waterline")
        if np.any(y[:, -1] > 1.0e-8 * max(1.0, float(np.max(y)))):
            raise ValueError("the final point of every section must close on the centreline")
        length = float(x[-1] - x[0]) if self.length_ref_m is None else float(self.length_ref_m)
        if not math.isfinite(length) or length <= 0.0:
            raise ValueError("length_ref_m must be finite and positive")
        object.__setattr__(self, "length_ref_m", length)

    @property
    def waterline_half_breadth_m(self) -> np.ndarray:
        return self.half_breadth_m[:, 0]

    @property
    def local_draft_m(self) -> np.ndarray:
        return -np.min(self.z_m, axis=1)

    @property
    def nondimensional(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        length = float(self.length_ref_m)
        return self.x_m / length, self.z_m / length, self.half_breadth_m / length

    def resample(self, station_count: int, vertical_count: int) -> "HullOffsets":
        """Resample the sectioned surface on smooth common normalized coordinates."""

        if station_count < 3 or vertical_count < 3:
            raise ValueError("resampling counts must be at least three")
        old_x = self.x_m
        new_x = np.linspace(old_x[0], old_x[-1], station_count)
        old_eta = np.linspace(0.0, 1.0, self.z_m.shape[1])
        new_eta = 0.5 * (1.0 - np.cos(np.linspace(0.0, math.pi, vertical_count)))
        z_eta = np.vstack([np.interp(new_eta, old_eta, row) for row in self.z_m])
        y_eta = np.vstack([np.interp(new_eta, old_eta, row) for row in self.half_breadth_m])
        new_z = np.vstack([np.interp(new_x, old_x, z_eta[:, j]) for j in range(vertical_count)]).T
        new_y = np.vstack([np.interp(new_x, old_x, y_eta[:, j]) for j in range(vertical_count)]).T
        new_y = np.maximum(new_y, 0.0)
        new_y[:, -1] = 0.0
        new_z[:, 0] = 0.0
        return HullOffsets(
            x_m=new_x,
            z_m=new_z,
            half_breadth_m=new_y,
            name=self.name,
            length_ref_m=self.length_ref_m,
            provenance=self.provenance,
        )

    def half_breadth_at_waterline(self, x_values: np.ndarray) -> np.ndarray:
        return np.interp(
            np.asarray(x_values, dtype=float),
            self.x_m,
            self.waterline_half_breadth_m,
            left=0.0,
            right=0.0,
        )

    def to_mesh(self) -> TriMesh:
        """Triangulate the starboard side with normals pointing into the fluid."""

        nx, nv = self.half_breadth_m.shape
        vertices = np.column_stack(
            (
                np.repeat(self.x_m, nv),
                self.half_breadth_m.reshape(-1),
                self.z_m.reshape(-1),
            )
        )
        faces = []
        for i in range(nx - 1):
            for j in range(nv - 1):
                p00 = i * nv + j
                p01 = i * nv + j + 1
                p10 = (i + 1) * nv + j
                p11 = (i + 1) * nv + j + 1
                faces.append((p00, p01, p10))
                faces.append((p10, p01, p11))
        mesh = TriMesh.from_vertices_faces(vertices, np.asarray(faces, dtype=int))
        external = mesh.centroids[:, 1] > 1.0e-12 * max(1.0, float(self.length_ref_m))
        mesh = TriMesh(
            mesh.vertices,
            mesh.faces[external],
            mesh.centroids[external],
            mesh.normals[external],
            mesh.areas[external],
        )
        normals = mesh.normals.copy()
        # The chosen winding is outward for regular starboard panels.  Use a
        # geometric check for panels near degenerate stems and the keel.
        wrong = normals[:, 1] < -1.0e-10
        normals[wrong] *= -1.0
        return TriMesh(mesh.vertices, mesh.faces, mesh.centroids, normals, mesh.areas)

    @property
    def hydrostatics(self) -> HullHydrostatics:
        """Integrate hydrostatic quantities directly from the offsets."""

        sectional_area = np.empty(self.x_m.size)
        for index, (z, y) in enumerate(zip(self.z_m, self.half_breadth_m)):
            order = np.argsort(z)
            sectional_area[index] = 2.0 * _trapezoid(y[order], z[order])
        volume = float(_trapezoid(sectional_area, self.x_m))
        if volume > 0.0:
            lcb = float(_trapezoid(sectional_area * self.x_m, self.x_m) / volume)
        else:
            lcb = float("nan")
        waterplane = float(2.0 * _trapezoid(self.waterline_half_breadth_m, self.x_m))
        wetted = float(2.0 * np.sum(self.to_mesh().areas))
        return HullHydrostatics(
            displacement_volume_m3=volume,
            longitudinal_center_of_buoyancy_m=lcb,
            waterplane_area_m2=waterplane,
            wetted_area_m2=wetted,
            maximum_waterline_beam_m=2.0 * float(np.max(self.waterline_half_breadth_m)),
            maximum_canoe_draft_m=float(np.max(self.local_draft_m)),
        )

    def to_csv(self, path: Path, metadata_path: Optional[Path] = None) -> None:
        """Write long-form station offsets and optional provenance metadata."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(["station_index", "point_index", "x_m", "z_m", "half_breadth_m"])
            for i, x_value in enumerate(self.x_m):
                for j in range(self.z_m.shape[1]):
                    writer.writerow([i, j, x_value, self.z_m[i, j], self.half_breadth_m[i, j]])
        if metadata_path is not None:
            metadata = {
                "schema_version": "2.0",
                "name": self.name,
                "length_ref_m": self.length_ref_m,
                "coordinate_system": "x bow-to-stern, y starboard, z upward; waterline z=0",
                "provenance": self.provenance,
                "hydrostatics": self.hydrostatics.__dict__,
            }
            Path(metadata_path).write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    @classmethod
    def from_csv(cls, path: Path, metadata_path: Optional[Path] = None) -> "HullOffsets":
        """Load the canonical long-form section schema."""

        rows = []
        with Path(path).open("r", newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            required = {"station_index", "point_index", "x_m", "z_m", "half_breadth_m"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError("offset CSV is missing canonical columns")
            for row in reader:
                rows.append(
                    (
                        int(row["station_index"]),
                        int(row["point_index"]),
                        float(row["x_m"]),
                        float(row["z_m"]),
                        float(row["half_breadth_m"]),
                    )
                )
        if not rows:
            raise ValueError("offset CSV is empty")
        station_ids = sorted({row[0] for row in rows})
        grouped = [[row for row in rows if row[0] == station] for station in station_ids]
        point_counts = {len(group) for group in grouped}
        if len(point_counts) != 1:
            raise ValueError("every station must contain the same number of points")
        grouped = [sorted(group, key=lambda row: row[1]) for group in grouped]
        x = np.asarray([group[0][2] for group in grouped])
        z = np.asarray([[row[3] for row in group] for group in grouped])
        y = np.asarray([[row[4] for row in group] for group in grouped])
        metadata: Dict[str, object] = {}
        if metadata_path is not None:
            metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        return cls(
            x,
            z,
            y,
            name=str(metadata.get("name", Path(path).stem)),
            length_ref_m=float(metadata.get("length_ref_m", x[-1] - x[0])),
            provenance=str(metadata.get("provenance", "")),
        )


def _piecewise_sine_profile(
    x: np.ndarray,
    peak_x: float,
    bow_exponent: float,
    stern_exponent: float,
) -> np.ndarray:
    values = np.empty_like(x)
    forward = x <= peak_x
    forward_coordinate = np.clip((x[forward] + 0.5) / (peak_x + 0.5), 0.0, 1.0)
    aft_coordinate = np.clip((0.5 - x[~forward]) / (0.5 - peak_x), 0.0, 1.0)
    values[forward] = np.sin(0.5 * math.pi * forward_coordinate) ** bow_exponent
    values[~forward] = np.sin(0.5 * math.pi * aft_coordinate) ** stern_exponent
    return values


def image_inspired_yacht(
    station_count: int = 41,
    vertical_count: int = 17,
    length_ref_m: float = 1.0,
    beam_to_length: float = 0.28,
    draft_to_length: float = 0.06,
) -> HullOffsets:
    """Return the generic yacht reconstructed from the supplied raster plan.

    The drawing contains no common dimensional scale.  Its projected contours
    determine longitudinal fullness, keel rocker, and sectional character;
    ``beam_to_length`` and ``draft_to_length`` establish the independent scale.
    """

    if station_count < 7 or vertical_count < 5:
        raise ValueError("image-inspired hull requires at least 7 x 5 points")
    if length_ref_m <= 0.0 or beam_to_length <= 0.0 or draft_to_length <= 0.0:
        raise ValueError("hull scale ratios must be positive")
    theta_x = np.linspace(0.0, math.pi, station_count)
    x = -0.5 * np.cos(theta_x)
    eta = 0.5 * (1.0 - np.cos(np.linspace(0.0, math.pi, vertical_count)))

    breadth_profile = _piecewise_sine_profile(
        x, peak_x=0.07, bow_exponent=1.22, stern_exponent=0.72
    )
    draft_profile = _piecewise_sine_profile(
        x, peak_x=0.04, bow_exponent=1.08, stern_exponent=0.88
    )
    half_beam = 0.5 * beam_to_length * breadth_profile
    draft = draft_to_length * draft_profile

    # Rounded U-sections amidships, becoming more V-shaped towards the bow
    # and moderately rounded in the run.  This reproduces the principal
    # underwater characteristics visible in the source body plan.
    longitudinal = (x + 0.5)
    bottom_exponent = 1.55 + 1.15 * np.sin(math.pi * longitudinal) ** 1.4
    side_exponent = 1.45 + 1.35 * np.sin(math.pi * longitudinal) ** 1.2
    side_exponent *= 0.88 + 0.12 * np.clip((x + 0.5) / 0.35, 0.0, 1.0)

    z = np.empty((station_count, vertical_count))
    y = np.empty_like(z)
    for index in range(station_count):
        z[index] = -draft[index] * eta
        inside = np.maximum(0.0, 1.0 - eta ** bottom_exponent[index])
        y[index] = half_beam[index] * inside ** (1.0 / side_exponent[index])
    y[:, -1] = 0.0
    x_dimensional = length_ref_m * x
    return HullOffsets(
        x_m=x_dimensional,
        z_m=length_ref_m * z,
        half_breadth_m=length_ref_m * y,
        name="Image-inspired generic displacement yacht",
        length_ref_m=length_ref_m,
        provenance=(
            "Parametric reconstruction of the immersed forms visible in yacht.pdf; "
            "nondimensional ratios selected independently because the raster has no common scale."
        ),
    )


def wigley_hull(
    station_count: int = 41,
    vertical_count: int = 17,
    length_ref_m: float = 1.0,
    beam_to_length: float = 0.10,
    draft_to_length: float = 0.0625,
    *,
    nx: Optional[int] = None,
    nz: Optional[int] = None,
    length_m: Optional[float] = None,
    beam_m: Optional[float] = None,
    draft_m: Optional[float] = None,
) -> HullOffsets:
    """Return the canonical parabolic Wigley hull.

    ``nx``/``nz`` and dimensional aliases are accepted for the exact-body
    examples while the original arguments remain unchanged.
    """

    if nx is not None:
        station_count = nx
    if nz is not None:
        vertical_count = nz
    if length_m is not None:
        length_ref_m = length_m
    if beam_m is not None:
        beam_to_length = beam_m / length_ref_m
    if draft_m is not None:
        draft_to_length = draft_m / length_ref_m

    x_over_l = np.linspace(-0.5, 0.5, station_count)
    eta = np.linspace(0.0, 1.0, vertical_count)
    x_factor = np.maximum(0.0, 1.0 - 4.0 * x_over_l**2)
    z = -draft_to_length * np.broadcast_to(eta, (station_count, vertical_count))
    y = (
        0.5
        * beam_to_length
        * x_factor[:, None]
        * np.maximum(0.0, 1.0 - eta[None, :] ** 2)
    )
    return HullOffsets(
        x_m=length_ref_m * x_over_l,
        z_m=length_ref_m * z,
        half_breadth_m=length_ref_m * y,
        name="Wigley hull",
        length_ref_m=length_ref_m,
        provenance="Analytic Wigley parabolic hull",
    )
