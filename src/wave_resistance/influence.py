"""Desingularized Rankine-source influence operators.

Each flat panel is represented by a source carrying its panel area and placed a
controlled distance outside the fluid domain.  This method of fundamental
solutions avoids uncontrolled diagonal singularities while retaining the
three-dimensional Rankine kernel and an explicit refinement parameter.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np


FOUR_PI = 4.0 * math.pi


def rankine_potential(field_points: np.ndarray, source_points: np.ndarray) -> np.ndarray:
    """Return the unit-source potential matrix ``1 / (4 pi r)``."""

    field = np.asarray(field_points, dtype=float)
    source = np.asarray(source_points, dtype=float)
    separation = field[:, None, :] - source[None, :, :]
    distance = np.linalg.norm(separation, axis=2)
    if np.any(distance <= np.finfo(float).eps):
        raise ValueError("field and source points must be desingularized")
    return 1.0 / (FOUR_PI * distance)


def rankine_velocity(field_points: np.ndarray, source_points: np.ndarray) -> np.ndarray:
    """Return gradients of the unit-source potential.

    The returned array has shape ``(n_field, n_source, 3)``.
    """

    field = np.asarray(field_points, dtype=float)
    source = np.asarray(source_points, dtype=float)
    separation = field[:, None, :] - source[None, :, :]
    distance_squared = np.sum(separation * separation, axis=2)
    if np.any(distance_squared <= np.finfo(float).eps):
        raise ValueError("field and source points must be desingularized")
    denominator = FOUR_PI * distance_squared ** 1.5
    return -separation / denominator[:, :, None]


def _source_images(
    source_points: np.ndarray,
    source_weights: np.ndarray,
    mirror_y: bool,
    mirror_z: bool,
) -> Tuple[Tuple[np.ndarray, np.ndarray], ...]:
    images = [(source_points, source_weights)]
    if mirror_y:
        reflected = source_points.copy()
        reflected[:, 1] *= -1.0
        images.append((reflected, source_weights))
    if mirror_z:
        existing = list(images)
        for points, weights in existing:
            reflected = points.copy()
            reflected[:, 2] *= -1.0
            images.append((reflected, weights))
    return tuple(images)


def influence_matrices(
    field_points: np.ndarray,
    source_points: np.ndarray,
    source_weights: np.ndarray,
    *,
    mirror_y: bool = True,
    mirror_z: bool = False,
    chunk_size: int = 256,
    potential: bool = True,
    velocity: bool = True,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Assemble weighted potential and velocity influence matrices."""

    field = np.asarray(field_points, dtype=float)
    source = np.asarray(source_points, dtype=float)
    weights = np.asarray(source_weights, dtype=float)
    if field.ndim != 2 or field.shape[1] != 3:
        raise ValueError("field_points must have shape (n, 3)")
    if source.ndim != 2 or source.shape[1] != 3 or weights.shape != (source.shape[0],):
        raise ValueError("invalid source points or weights")
    if np.any(weights <= 0.0):
        raise ValueError("source weights must be positive")
    potential_matrix = np.zeros((field.shape[0], source.shape[0])) if potential else None
    velocity_matrix = (
        np.zeros((field.shape[0], source.shape[0], 3)) if velocity else None
    )
    images = _source_images(source, weights, mirror_y, mirror_z)
    for start in range(0, field.shape[0], chunk_size):
        stop = min(start + chunk_size, field.shape[0])
        block = field[start:stop]
        for image_points, image_weights in images:
            if potential_matrix is not None:
                potential_matrix[start:stop] += rankine_potential(block, image_points) * image_weights
            if velocity_matrix is not None:
                velocity_matrix[start:stop] += (
                    rankine_velocity(block, image_points) * image_weights[None, :, None]
                )
    return potential_matrix, velocity_matrix


def normal_influence(
    field_points: np.ndarray,
    field_normals: np.ndarray,
    source_points: np.ndarray,
    source_weights: np.ndarray,
    *,
    mirror_y: bool = True,
    mirror_z: bool = False,
    chunk_size: int = 256,
) -> np.ndarray:
    """Return source-induced velocity projected onto field normals."""

    normals = np.asarray(field_normals, dtype=float)
    if normals.shape != np.asarray(field_points).shape:
        raise ValueError("field_normals must match field_points")
    _, velocity = influence_matrices(
        field_points,
        source_points,
        source_weights,
        mirror_y=mirror_y,
        mirror_z=mirror_z,
        chunk_size=chunk_size,
        potential=False,
        velocity=True,
    )
    assert velocity is not None
    return np.einsum("ijk,ik->ij", velocity, normals)


def panel_normal_influence(
    collocation_points: np.ndarray,
    panel_normals: np.ndarray,
    panel_centroids: np.ndarray,
    panel_areas: np.ndarray,
    *,
    mirror_y: bool = True,
    mirror_z: bool = False,
    jump: float = -0.5,
) -> np.ndarray:
    """Approximate the constant-source panel normal operator.

    Off-diagonal panels use area-weighted centroid influence.  The singular
    primary-panel limit is supplied analytically by the single-layer jump
    term; symmetry images remain regular and are evaluated explicitly.
    """

    fields = np.asarray(collocation_points, dtype=float)
    normals = np.asarray(panel_normals, dtype=float)
    sources = np.asarray(panel_centroids, dtype=float)
    areas = np.asarray(panel_areas, dtype=float)
    if fields.shape != normals.shape or fields.shape != sources.shape:
        raise ValueError("panel collocation, normals, and centroids must match")
    separation = fields[:, None, :] - sources[None, :, :]
    distance_squared = np.sum(separation * separation, axis=2)
    diagonal = np.arange(fields.shape[0])
    distance_squared[diagonal, diagonal] = np.inf
    primary_velocity = -separation / (
        FOUR_PI * distance_squared[:, :, None] ** 1.5
    )
    matrix = np.einsum("ijk,ik->ij", primary_velocity, normals) * areas[None, :]
    matrix[diagonal, diagonal] = jump
    image_points = []
    if mirror_y:
        reflected = sources.copy()
        reflected[:, 1] *= -1.0
        image_points.append(reflected)
    if mirror_z:
        reflected = sources.copy()
        reflected[:, 2] *= -1.0
        image_points.append(reflected)
        if mirror_y:
            reflected_both = reflected.copy()
            reflected_both[:, 1] *= -1.0
            image_points.append(reflected_both)
    for points in image_points:
        velocity = rankine_velocity(fields, points) * areas[None, :, None]
        matrix += np.einsum("ijk,ik->ij", velocity, normals)
    return matrix


def apply_sources(
    field_points: np.ndarray,
    source_points: np.ndarray,
    source_weights: np.ndarray,
    strengths: np.ndarray,
    *,
    mirror_y: bool = True,
    mirror_z: bool = False,
    chunk_size: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate potential and velocity generated by solved source strengths."""

    potential_matrix, velocity_matrix = influence_matrices(
        field_points,
        source_points,
        source_weights,
        mirror_y=mirror_y,
        mirror_z=mirror_z,
        chunk_size=chunk_size,
    )
    assert potential_matrix is not None and velocity_matrix is not None
    strengths_array = np.asarray(strengths, dtype=float)
    return potential_matrix @ strengths_array, np.einsum(
        "ijk,j->ik", velocity_matrix, strengths_array
    )
