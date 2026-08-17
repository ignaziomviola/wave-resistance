"""Transparent dense assembly for constant-source Rankine panels."""

import numpy as np

from .kernels import source_panel_potential, source_panel_velocity


def potential_influence(points, source_mesh, order=12):
    points = np.atleast_2d(np.asarray(points, dtype=float))
    triangles = source_mesh.vertices[source_mesh.faces]
    matrix = np.empty((len(points), len(triangles)))
    for i, point in enumerate(points):
        for j, triangle in enumerate(triangles):
            matrix[i, j] = source_panel_potential(point, triangle, order)
    return matrix


def normal_influence(points, normals, source_mesh, order=12):
    points = np.atleast_2d(np.asarray(points, dtype=float))
    normals = np.atleast_2d(np.asarray(normals, dtype=float))
    triangles = source_mesh.vertices[source_mesh.faces]
    matrix = np.empty((len(points), len(triangles)))
    for i, (point, normal) in enumerate(zip(points, normals)):
        for j, triangle in enumerate(triangles):
            matrix[i, j] = np.dot(source_panel_velocity(point, triangle, order), normal)
    return matrix


def assemble_source_operators(mesh, order=12):
    """Return single-layer and exterior normal-derivative operators."""

    single_layer = potential_influence(mesh.centroids, mesh, order)
    normal_derivative = normal_influence(mesh.centroids, mesh.normals, mesh, order)
    # Principal-value self integral is zero on a planar constant panel.  The
    # exterior trace supplies the source-sheet jump.
    np.fill_diagonal(normal_derivative, -0.5)
    return single_layer, normal_derivative


def evaluate_velocity(points, mesh, strengths, order=12):
    points = np.atleast_2d(np.asarray(points, dtype=float))
    strengths = np.asarray(strengths, dtype=float)
    output = np.zeros((len(points), 3))
    triangles = mesh.vertices[mesh.faces]
    for i, point in enumerate(points):
        for strength, triangle in zip(strengths, triangles):
            output[i] += strength * source_panel_velocity(point, triangle, order)
    return output


def reconstruct_surface_gradient(mesh, panel_potential, neighbour_depth=2):
    """Weighted least-squares tangential gradient of constant panel values.

    The local plane fit uses only centroid differences projected into each
    target panel. It exactly reproduces linear tangential fields whenever the
    neighbour stencil spans both tangent directions.
    """

    values = np.asarray(panel_potential, dtype=float)
    gradients = np.zeros((len(mesh.faces), 3))
    for index, (point, normal) in enumerate(zip(mesh.centroids, mesh.normals)):
        neighbours = {index}
        frontier = {index}
        for _ in range(neighbour_depth):
            frontier = {int(j) for i in frontier for j in mesh.face_adjacency[i]} - neighbours
            neighbours.update(frontier)
        ids = np.asarray(sorted(neighbours - {index}), dtype=int)
        if len(ids) < 2:
            continue
        delta = mesh.centroids[ids] - point
        delta -= (delta @ normal)[:, None] * normal
        rhs = values[ids] - values[index]
        distance = np.linalg.norm(delta, axis=1)
        weights = 1.0 / np.maximum(distance, np.finfo(float).eps)
        matrix = np.vstack((delta * weights[:, None], normal[None, :]))
        target = np.r_[rhs * weights, 0.0]
        gradients[index] = np.linalg.lstsq(matrix, target, rcond=None)[0]
    return gradients
