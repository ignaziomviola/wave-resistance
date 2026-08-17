"""Triangular surface meshes and the :class:`HullOffsets` adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np


def _unique_edges(faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    edges = np.sort(
        np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1
    )
    return np.unique(edges, axis=0, return_counts=True)


@dataclass
class SurfaceMesh:
    vertices: np.ndarray
    faces: np.ndarray
    tags: Optional[np.ndarray] = None
    physical_geometry: Any = None
    normals: np.ndarray = field(init=False)
    areas: np.ndarray = field(init=False)
    centroids: np.ndarray = field(init=False)
    boundary_edges: np.ndarray = field(init=False)
    face_adjacency: Tuple[np.ndarray, ...] = field(init=False)
    waterline_vertices: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.vertices = np.asarray(self.vertices, dtype=float).copy()
        self.faces = np.asarray(self.faces, dtype=int).copy()
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError("vertices must have shape (n, 3)")
        if self.faces.ndim != 2 or self.faces.shape[1] != 3 or len(self.faces) == 0:
            raise ValueError("faces must have non-empty shape (m, 3)")
        if not np.all(np.isfinite(self.vertices)):
            raise ValueError("mesh vertices must be finite")
        if np.any(self.faces < 0) or np.any(self.faces >= len(self.vertices)):
            raise ValueError("face index outside vertex array")
        if np.any(np.diff(np.sort(self.faces, axis=1), axis=1) == 0):
            raise ValueError("face contains a repeated vertex")
        canonical = np.sort(self.faces, axis=1)
        if len(np.unique(canonical, axis=0)) != len(self.faces):
            raise ValueError("duplicate face")

        triangles = self.vertices[self.faces]
        cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        norm = np.linalg.norm(cross, axis=1)
        scale = max(float(np.ptp(self.vertices, axis=0).max()), 1.0)
        if np.any(norm <= 1.0e-14 * scale**2):
            raise ValueError("degenerate triangle")
        self.areas = 0.5 * norm
        self.normals = cross / norm[:, None]
        self.centroids = triangles.mean(axis=1)

        edges, counts = _unique_edges(self.faces)
        if np.any(counts > 2):
            raise ValueError("non-manifold edge")
        self.boundary_edges = edges[counts == 1]
        incident = [[] for _ in range(len(self.faces))]
        owners = {}
        for face_index, face in enumerate(self.faces):
            for edge in (tuple(sorted(face[[0, 1]])), tuple(sorted(face[[1, 2]])), tuple(sorted(face[[2, 0]]))):
                if edge in owners:
                    other = owners[edge]
                    incident[face_index].append(other)
                    incident[other].append(face_index)
                else:
                    owners[edge] = face_index
        self.face_adjacency = tuple(np.asarray(row, dtype=int) for row in incident)
        tolerance = 1.0e-10 * scale
        self.waterline_vertices = np.flatnonzero(np.abs(self.vertices[:, 2]) <= tolerance)
        if self.tags is None:
            self.tags = np.full(len(self.faces), "surface", dtype=object)
        else:
            self.tags = np.asarray(self.tags, dtype=object)
            if len(self.tags) != len(self.faces):
                raise ValueError("one tag is required per face")

    def diagnostics(self) -> Dict[str, float]:
        p = self.vertices[self.faces]
        lengths = np.stack(
            [
                np.linalg.norm(p[:, 1] - p[:, 0], axis=1),
                np.linalg.norm(p[:, 2] - p[:, 1], axis=1),
                np.linalg.norm(p[:, 0] - p[:, 2], axis=1),
            ],
            axis=1,
        )
        a, b, c = lengths[:, 0], lengths[:, 1], lengths[:, 2]
        angles = np.column_stack(
            [
                np.arccos(np.clip((a*a + c*c - b*b) / (2*a*c), -1.0, 1.0)),
                np.arccos(np.clip((a*a + b*b - c*c) / (2*a*b), -1.0, 1.0)),
                np.arccos(np.clip((b*b + c*c - a*a) / (2*b*c), -1.0, 1.0)),
            ]
        )
        return {
            "panels": int(len(self.faces)),
            "vertices": int(len(self.vertices)),
            "min_area": float(self.areas.min()),
            "max_aspect_ratio": float((lengths.max(axis=1) / lengths.min(axis=1)).max()),
            "minimum_angle_deg": float(np.degrees(angles).min()),
            "boundary_edges": int(len(self.boundary_edges)),
            "waterline_vertices": int(len(self.waterline_vertices)),
        }


def _merge_vertices(vertices: np.ndarray, faces: np.ndarray, tolerance: float) -> Tuple[np.ndarray, np.ndarray]:
    keys = np.round(vertices / tolerance).astype(np.int64)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    merged = vertices[np.sort(first)]
    # np.unique orders keys rather than first occurrences; remap through the selected coordinates.
    coordinate_to_new = {tuple(np.round(v / tolerance).astype(np.int64)): i for i, v in enumerate(merged)}
    remap = np.asarray([coordinate_to_new[tuple(key)] for key in keys], dtype=int)
    return merged, remap[faces]


def offset_hull_mesh(hull) -> SurfaceMesh:
    """Mirror a public ``HullOffsets`` surface with outward body normals."""

    x = np.asarray(hull.x_m, dtype=float)
    z = np.asarray(hull.z_m, dtype=float)
    breadth = np.asarray(hull.half_breadth_m, dtype=float)
    nx, nz = breadth.shape
    vertices = []
    faces = []
    for side_index, side in enumerate((-1.0, 1.0)):
        base = side_index * nx * nz
        vertices.extend((x[i], side * breadth[i, j], z[i, j]) for i in range(nx) for j in range(nz))
        for i in range(nx - 1):
            for j in range(nz - 1):
                a = base + i * nz + j
                b, c, d = a + nz, a + nz + 1, a + 1
                faces.extend(((a, c, b), (a, d, c)) if side < 0.0 else ((a, b, c), (a, c, d)))
    vertices_array = np.asarray(vertices, dtype=float)
    faces_array = np.asarray(faces, dtype=int)
    tolerance = max(float(hull.length_ref_m), 1.0) * 1.0e-12
    vertices_array, faces_array = _merge_vertices(vertices_array, faces_array, tolerance)
    triangles = vertices_array[faces_array]
    keep = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1) > tolerance**2
    # Collapsed stem stations can otherwise create artificial centre-plane
    # caps; the centre plane is inside the displaced body, not wetted surface.
    keep &= ~np.all(np.abs(triangles[:, :, 1]) <= tolerance, axis=1)
    faces_array = faces_array[keep]
    _, unique_indices = np.unique(np.sort(faces_array, axis=1), axis=0, return_index=True)
    faces_array = faces_array[np.sort(unique_indices)]
    return SurfaceMesh(
        vertices_array,
        faces_array,
        tags=np.full(len(faces_array), "hull", dtype=object),
        physical_geometry=hull,
    )


def rectangular_free_surface(hull, nx=17, ny=13, upstream=1.0, downstream=2.0, lateral=1.0) -> SurfaceMesh:
    """Create a symmetric body-fitted graph mesh sharing the hull waterline."""

    if nx < 5 or ny < 5 or ny % 2 == 0:
        raise ValueError("free-surface nx>=5 and odd ny>=5 are required")
    length = float(hull.length_ref_m)
    x0, x1 = float(hull.x_m[0]), float(hull.x_m[-1])
    external = np.linspace(x0 - upstream * length, x1 + downstream * length, nx)
    xs = np.unique(np.r_[external, hull.x_m])
    half_count = (ny + 1) // 2
    vertices = []
    rows = []
    for xv in xs:
        half_breadth = float(hull.half_breadth_at_waterline(np.asarray([xv]))[0])
        positive = np.linspace(half_breadth, lateral * length, half_count)
        # Keep both inner waterline nodes; the two strips are not connected
        # across the hull waterplane cut-out.
        y_values = np.r_[-positive[::-1], positive]
        rows.append(np.arange(len(vertices), len(vertices) + len(y_values)))
        vertices.extend((xv, yv, 0.0) for yv in y_values)
    faces = []
    for i in range(len(rows) - 1):
        for j in range(2 * half_count - 1):
            if j == half_count - 1:
                continue
            a, b = rows[i][j], rows[i + 1][j]
            c, d = rows[i + 1][j + 1], rows[i][j + 1]
            faces.extend(((a, b, c), (a, c, d)))
    vertices_array = np.asarray(vertices, dtype=float)
    faces_array = np.asarray(faces, dtype=int)
    vertices_array, faces_array = _merge_vertices(vertices_array, faces_array, length * 1.0e-12)
    triangles = vertices_array[faces_array]
    keep = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1) > (length * 1.0e-12) ** 2
    return SurfaceMesh(
        vertices_array,
        faces_array[keep],
        tags=np.full(int(np.count_nonzero(keep)), "free_surface", dtype=object),
        physical_geometry=hull,
    )
