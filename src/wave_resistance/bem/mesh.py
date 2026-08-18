"""Triangular surface meshes and the :class:`OffsetHull` adapter."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..geometry import OffsetHull


@dataclass(frozen=True)
class MeshDiagnostics:
    minimum_area: float
    maximum_aspect_ratio: float
    minimum_angle_deg: float
    boundary_edge_count: int
    non_manifold_edge_count: int
    duplicate_face_count: int
    orientation_consistent: bool
    waterline_conformity_error: float = 0.0
    valid: bool = True
    messages: Tuple[str, ...] = ()


@dataclass(frozen=True)
class SurfaceMesh:
    """Immutable oriented triangular surface with per-face boundary tags."""

    vertices: np.ndarray
    faces: np.ndarray
    tags: np.ndarray
    name: str = "surface"
    waterline_vertices: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    physical_geometry: Optional[object] = None

    def __post_init__(self) -> None:
        vertices = np.array(self.vertices, dtype=float, copy=True)
        faces = np.array(self.faces, dtype=int, copy=True)
        tags = np.array(self.tags, dtype="U32", copy=True)
        waterline = np.array(self.waterline_vertices, dtype=int, copy=True)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.all(np.isfinite(vertices)):
            raise ValueError("vertices must be a finite (N,3) array")
        if faces.ndim != 2 or faces.shape[1] != 3 or faces.size == 0:
            raise ValueError("faces must be a non-empty (M,3) array")
        if np.any(faces < 0) or np.any(faces >= len(vertices)):
            raise ValueError("face contains an invalid vertex index")
        if tags.shape != (len(faces),):
            raise ValueError("tags must contain one entry per face")
        if len(np.unique(np.sort(faces, axis=1), axis=0)) != len(faces):
            raise ValueError("mesh contains duplicate faces")
        cross = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]],
                         vertices[faces[:, 2]] - vertices[faces[:, 0]])
        if np.any(np.linalg.norm(cross, axis=1) <= 1e-14):
            raise ValueError("mesh contains a degenerate triangle")
        for array in (vertices, faces, tags, waterline):
            array.setflags(write=False)
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "faces", faces)
        object.__setattr__(self, "tags", tags)
        object.__setattr__(self, "waterline_vertices", waterline)

    @classmethod
    def from_arrays(cls, vertices, faces, *, tags=None, name="surface",
                    waterline_vertices=(), physical_geometry=None) -> "SurfaceMesh":
        face_array = np.asarray(faces, dtype=int)
        if tags is None:
            tags = np.full(len(face_array), "surface")
        return cls(vertices, face_array, tags, name, waterline_vertices, physical_geometry)

    @property
    def triangles(self) -> np.ndarray:
        return self.vertices[self.faces]

    @property
    def centroids(self) -> np.ndarray:
        return np.mean(self.triangles, axis=1)

    @property
    def area_vectors(self) -> np.ndarray:
        t = self.triangles
        return 0.5 * np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])

    @property
    def areas(self) -> np.ndarray:
        return np.linalg.norm(self.area_vectors, axis=1)

    @property
    def normals(self) -> np.ndarray:
        return self.area_vectors / self.areas[:, None]

    @property
    def adjacency(self) -> Tuple[Tuple[int, ...], ...]:
        owners: Dict[Tuple[int, int], list] = {}
        for i, face in enumerate(self.faces):
            for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
                owners.setdefault(tuple(sorted((int(a), int(b)))), []).append(i)
        adjacent = [set() for _ in self.faces]
        for face_ids in owners.values():
            for i in face_ids:
                adjacent[i].update(j for j in face_ids if j != i)
        return tuple(tuple(sorted(values)) for values in adjacent)

    def diagnostics(self, *, waterline_reference: Optional[np.ndarray] = None) -> MeshDiagnostics:
        tri = self.triangles
        lengths = np.stack((np.linalg.norm(tri[:, 1]-tri[:, 0], axis=1),
                            np.linalg.norm(tri[:, 2]-tri[:, 1], axis=1),
                            np.linalg.norm(tri[:, 0]-tri[:, 2], axis=1)), axis=1)
        aspect = np.max(lengths, axis=1) / np.maximum(2*self.areas/np.max(lengths, axis=1), 1e-30)
        cosines = np.empty_like(lengths)
        cosines[:, 0] = (lengths[:, 0]**2 + lengths[:, 2]**2 - lengths[:, 1]**2)/(2*lengths[:, 0]*lengths[:, 2])
        cosines[:, 1] = (lengths[:, 0]**2 + lengths[:, 1]**2 - lengths[:, 2]**2)/(2*lengths[:, 0]*lengths[:, 1])
        cosines[:, 2] = (lengths[:, 1]**2 + lengths[:, 2]**2 - lengths[:, 0]**2)/(2*lengths[:, 1]*lengths[:, 2])
        minimum_angle = float(np.min(np.degrees(np.arccos(np.clip(cosines, -1, 1)))))
        owners: Dict[Tuple[int, int], int] = {}
        for face in self.faces:
            for edge in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
                key = tuple(sorted(map(int, edge)))
                owners[key] = owners.get(key, 0) + 1
        boundary = sum(v == 1 for v in owners.values())
        non_manifold = sum(v > 2 for v in owners.values())
        wl_error = 0.0
        if waterline_reference is not None and len(self.waterline_vertices):
            a = self.vertices[self.waterline_vertices]
            b = np.asarray(waterline_reference, dtype=float)
            wl_error = float(max(np.min(np.linalg.norm(b-p, axis=1)) for p in a))
        messages = []
        if non_manifold:
            messages.append("non-manifold edges")
        if minimum_angle < 5.0:
            messages.append("minimum panel angle below 5 degrees")
        return MeshDiagnostics(float(np.min(self.areas)), float(np.max(aspect)), minimum_angle,
                               boundary, non_manifold, 0, non_manifold == 0,
                               wl_error, non_manifold == 0, tuple(messages))

    def with_vertices(self, vertices: np.ndarray, *, name: Optional[str] = None) -> "SurfaceMesh":
        return SurfaceMesh(vertices, self.faces, self.tags, name or self.name,
                           self.waterline_vertices, self.physical_geometry)


def _deduplicated_mesh(points: Iterable[Sequence[float]], raw_faces: Iterable[Sequence[int]],
                       tags: Iterable[str], *, name: str, waterline_raw: Iterable[int],
                       physical_geometry=None) -> SurfaceMesh:
    points = list(points)
    scale = max(1.0, float(np.max(np.abs(points))))
    lookup: Dict[Tuple[int, int, int], int] = {}
    vertices = []
    remap = []
    for point in points:
        key = tuple(np.round(np.asarray(point) / (1e-12*scale)).astype(np.int64))
        if key not in lookup:
            lookup[key] = len(vertices)
            vertices.append(point)
        remap.append(lookup[key])
    faces, kept_tags = [], []
    seen = set()
    vertices_array = np.asarray(vertices, dtype=float)
    for face, tag in zip(raw_faces, tags):
        mapped = tuple(remap[int(i)] for i in face)
        if len(set(mapped)) < 3:
            continue
        if np.linalg.norm(np.cross(vertices_array[mapped[1]]-vertices_array[mapped[0]],
                                   vertices_array[mapped[2]]-vertices_array[mapped[0]])) < 1e-14*scale**2:
            continue
        key = tuple(sorted(mapped))
        if key in seen:
            continue
        seen.add(key); faces.append(mapped); kept_tags.append(tag)
    waterline = np.unique([remap[int(i)] for i in waterline_raw])
    return SurfaceMesh(vertices_array, np.asarray(faces), np.asarray(kept_tags), name,
                       waterline, physical_geometry)


def offset_hull_mesh(hull: OffsetHull, nx: Optional[int] = None,
                     nz: Optional[int] = None) -> SurfaceMesh:
    """Triangulate both sides of an offset hull with outward orientation."""
    if not isinstance(hull, OffsetHull):
        raise TypeError("hull must be an OffsetHull")
    nx = nx or len(hull.x); nz = nz or len(hull.z)
    if nx < 3 or nz < 2:
        raise ValueError("hull mesh requires nx>=3 and nz>=2")
    x = np.linspace(0, hull.metadata.length_ref_m, nx)
    z = np.linspace(0, hull.draft_m, nz)
    by_z = np.array([np.interp(x, hull.x, hull.half_breadths[:, j]) for j in range(len(hull.z))]).T
    breadth = np.array([np.interp(z, hull.z, row) for row in by_z])
    points=[]; faces=[]; tags=[]; waterline=[]
    for side in (-1.0, 1.0):
        base=len(points)
        for i, xi in enumerate(x):
            for k, zk in enumerate(z):
                points.append((xi, side*breadth[i,k], zk))
                if k == 0: waterline.append(base+i*nz+k)
        for i in range(nx-1):
            for k in range(nz-1):
                a=base+i*nz+k; b=a+nz; c=b+1; d=a+1
                # r_x cross r_z points to port; reverse it on starboard.
                # Opposite diagonals prevent coincident cross-seam edges at
                # the pointed ends and keel while retaining outward normals.
                cells=((a,b,c),(a,c,d)) if side < 0 else ((a,d,b),(b,d,c))
                faces.extend(cells); tags.extend(("hull","hull"))
    return _deduplicated_mesh(points, faces, tags, name="wetted_hull",
                              waterline_raw=waterline, physical_geometry=hull)


def free_surface_mesh(hull: OffsetHull, *, nx: int = 17, ny_half: int = 6,
                      upstream_lengths: float = 0.5, downstream_lengths: float = 1.5,
                      lateral_lengths: float = 0.75,
                      hull_nx: Optional[int] = None) -> SurfaceMesh:
    """Build a graph mesh whose inner edge exactly matches the hull waterline."""
    if nx < 5 or ny_half < 2:
        raise ValueError("free-surface mesh requires nx>=5 and ny_half>=2")
    length=hull.metadata.length_ref_m
    hull_x=np.linspace(0,length,hull_nx or min(len(hull.x),nx))
    x=np.unique(np.concatenate((np.linspace(-upstream_lengths*length,
                                             (1+downstream_lengths)*length, nx), hull_x)))
    inner=np.interp(np.clip(x,0,length), hull.x, hull.half_breadths[:,0])
    inner[(x<0)|(x>length)]=0.0
    outer=lateral_lengths*length
    points=[]; faces=[]; tags=[]; waterline=[]
    for side in (-1.0,1.0):
        base=len(points)
        for i,xi in enumerate(x):
            y=np.linspace(inner[i],outer,ny_half)*side
            for j,yj in enumerate(y):
                points.append((xi,yj,0.0))
                if j==0 and 0<=xi<=length: waterline.append(base+i*ny_half+j)
        for i in range(len(x)-1):
            for j in range(ny_half-1):
                a=base+i*ny_half+j; b=a+ny_half; c=b+1; d=a+1
                # Fluid is below (+z), hence the normal into air is -z.
                faces.extend(((a,c,b),(a,d,c))); tags.extend(("free_surface","free_surface"))
    return _deduplicated_mesh(points,faces,tags,name="free_surface",
                              waterline_raw=waterline,physical_geometry=hull)


def combine_meshes(*meshes: SurfaceMesh, name: str = "combined") -> SurfaceMesh:
    vertices=[]; faces=[]; tags=[]; waterline=[]; offset=0
    for mesh in meshes:
        vertices.extend(mesh.vertices)
        faces.extend(mesh.faces+offset)
        tags.extend(mesh.tags)
        waterline.extend(mesh.waterline_vertices+offset)
        offset += len(mesh.vertices)
    return SurfaceMesh(np.asarray(vertices),np.asarray(faces),np.asarray(tags),name,
                       np.asarray(waterline,dtype=int))


def mirrored_double_body(mesh: SurfaceMesh) -> Tuple[SurfaceMesh, int]:
    """Return a closed double body and the number of physical hull faces."""
    mirrored=np.array(mesh.vertices,copy=True); mirrored[:,2]*=-1
    mirror_faces=mesh.faces[:,[0,2,1]]+len(mesh.vertices)
    combined=SurfaceMesh(np.vstack((mesh.vertices,mirrored)),
                         np.vstack((mesh.faces,mirror_faces)),
                         np.concatenate((mesh.tags,np.full(len(mesh.faces),"image_hull"))),
                         "double_body")
    return combined,len(mesh.faces)
