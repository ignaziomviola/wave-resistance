"""Analytic test bodies with closed-form volume, area and section properties."""

import numpy as np

from wave_resistance.hull import TriMesh


def box_mesh(length: float, width: float, depth: float, freeboard: float,
             n: int = 6) -> TriMesh:
    """Closed rectangular box spanning z in [-depth, +freeboard], centred in x and y.

    The freeboard keeps z = 0 a transversal cut rather than a boundary, exactly as for a
    real hull.
    """
    xs = np.linspace(-0.5 * length, 0.5 * length, n + 1)
    ys = np.linspace(-0.5 * width, 0.5 * width, n + 1)
    zs = np.linspace(-depth, freeboard, n + 1)
    verts: list[np.ndarray] = []
    faces: list[list[int]] = []

    def add_quad_grid(corner, e1, e2, outward):
        base = len(verts)
        a = np.linspace(0.0, 1.0, n + 1)
        for i in a:
            for j in a:
                verts.append(corner + i * e1 + j * e2)
        idx = np.arange((n + 1) ** 2).reshape(n + 1, n + 1) + base
        for i in range(n):
            for j in range(n):
                q = [idx[i, j], idx[i + 1, j], idx[i + 1, j + 1], idx[i, j + 1]]
                t1, t2 = [q[0], q[1], q[2]], [q[0], q[2], q[3]]
                nrm = np.cross(verts[t1[1]] - verts[t1[0]], verts[t1[2]] - verts[t1[0]])
                if nrm @ outward < 0:
                    t1, t2 = t1[::-1], t2[::-1]
                faces.append(t1)
                faces.append(t2)

    lx, ly, lz = length, width, depth + freeboard
    o = np.array([-0.5 * length, -0.5 * width, -depth])
    ex, ey, ez = np.array([lx, 0, 0]), np.array([0, ly, 0]), np.array([0, 0, lz])
    add_quad_grid(o, ex, ey, np.array([0, 0, -1.0]))                 # bottom
    add_quad_grid(o + ez, ex, ey, np.array([0, 0, 1.0]))             # top
    add_quad_grid(o, ex, ez, np.array([0, -1.0, 0]))                 # y = -W/2
    add_quad_grid(o + ey, ex, ez, np.array([0, 1.0, 0]))             # y = +W/2
    add_quad_grid(o, ey, ez, np.array([-1.0, 0, 0]))                 # x = -L/2
    add_quad_grid(o + ex, ey, ez, np.array([1.0, 0, 0]))             # x = +L/2
    return TriMesh(np.array(verts), np.array(faces))


def box_exact(length: float, width: float, depth: float) -> dict[str, float]:
    return {
        "volume": length * width * depth,
        "waterplane_area": length * width,
        "midship_area": width * depth,
        "wetted_area": length * width + 2 * length * depth + 2 * width * depth,
        "lwl": length, "bwl": width, "draught": depth,
        "cb": 1.0, "cp": 1.0, "cm": 1.0, "cwp": 1.0,
        "lcb": 0.0, "vcb": -0.5 * depth,
    }


def uv_sphere(radius: float = 1.0, centre=(0.0, 0.0, 0.0), n_theta: int = 64,
              n_phi: int = 128) -> TriMesh:
    """Closed sphere with outward normals."""
    theta = np.linspace(0.0, np.pi, n_theta + 1)
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi + 1)[:-1]
    tt, pp = np.meshgrid(theta, phi, indexing="ij")
    pts = np.stack([radius * np.sin(tt) * np.cos(pp),
                    radius * np.sin(tt) * np.sin(pp),
                    radius * np.cos(tt)], axis=-1) + np.asarray(centre)
    idx = np.arange((n_theta + 1) * n_phi).reshape(n_theta + 1, n_phi)
    faces = []
    for i in range(n_theta):
        a, b = idx[i], idx[i + 1]
        a2, b2 = np.roll(a, -1), np.roll(b, -1)
        faces.append(np.column_stack([a, b, b2]))
        faces.append(np.column_stack([a, b2, a2]))
    mesh = TriMesh(pts.reshape(-1, 3), np.vstack(faces)).dropped_degenerate()
    return mesh if mesh.signed_volume() > 0 else mesh.flipped()


def icosphere(radius: float = 1.0, subdivisions: int = 3) -> TriMesh:
    """Closed sphere from a subdivided icosahedron: near-uniform, no polar slivers.

    A UV sphere is a poor panel-method test body because its polar triangles are slivers,
    which spoils the convergence rate for reasons that have nothing to do with the method
    being tested.
    """
    t = (1.0 + 5.0 ** 0.5) / 2.0
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ], dtype=float)
    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ])
    for _ in range(subdivisions):
        cache: dict[tuple[int, int], int] = {}
        vlist = list(verts)
        new_faces = []

        def midpoint(i: int, j: int) -> int:
            key = (min(i, j), max(i, j))
            if key not in cache:
                cache[key] = len(vlist)
                vlist.append(0.5 * (vlist[i] + vlist[j]))
            return cache[key]

        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            new_faces += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        verts = np.array(vlist)
        faces = np.array(new_faces)
    verts = radius * verts / np.linalg.norm(verts, axis=1)[:, None]
    mesh = TriMesh(verts, faces)
    return mesh if mesh.signed_volume() > 0 else mesh.flipped()
