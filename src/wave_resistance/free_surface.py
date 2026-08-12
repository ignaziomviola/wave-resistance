"""Body-fitted half-plane free-surface discretisation and radiation operator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .geometry import HullOffsets
from .influence import influence_matrices
from .models import MeshSettings, SolverSettings


@dataclass(frozen=True)
class FreeSurfaceMesh:
    collocation_points: np.ndarray
    source_points: np.ndarray
    source_weights: np.ndarray
    sponge: np.ndarray
    dx: float
    dy: float
    x_bounds: Tuple[float, float]
    y_max: float


def build_free_surface_mesh(
    hull: HullOffsets,
    settings: MeshSettings,
    sponge_start_fraction: float = 0.72,
) -> FreeSurfaceMesh:
    """Create a structured half-plane mesh with a waterplane cut-out.

    Three upstream stencil locations must remain outside the waterplane.  This
    deliberately leaves a narrow numerical gap at the waterline instead of
    imposing a free-surface condition through the body.
    """

    length = float(hull.length_ref_m)
    x_hull = hull.x_m / length
    x_min = float(x_hull[0] - settings.upstream_lengths)
    x_max = float(x_hull[-1] + settings.downstream_lengths)
    y_max = float(settings.lateral_lengths)
    x_values = np.linspace(x_min, x_max, settings.free_surface_x_points)
    dx = float(x_values[1] - x_values[0])
    y_edges = np.linspace(0.0, y_max, settings.free_surface_y_points + 1)
    y_values = 0.5 * (y_edges[:-1] + y_edges[1:])
    dy = float(y_edges[1] - y_edges[0])

    collocation = []
    sponge = []
    for x_value in x_values[3:]:
        stencil_x = x_value - dx * np.arange(4)
        physical_x = stencil_x * length
        breadth = hull.half_breadth_at_waterline(physical_x) / length
        required_y = float(np.max(breadth) + settings.waterline_gap_cells * dy)
        for y_value in y_values:
            if y_value <= required_y:
                continue
            collocation.append((x_value, y_value, 0.0))
            downstream_fraction = np.clip(
                (x_value - (x_hull[-1] + sponge_start_fraction * settings.downstream_lengths))
                / max((1.0 - sponge_start_fraction) * settings.downstream_lengths, 1.0e-12),
                0.0,
                1.0,
            )
            lateral_fraction = np.clip(
                (y_value - sponge_start_fraction * y_max)
                / max((1.0 - sponge_start_fraction) * y_max, 1.0e-12),
                0.0,
                1.0,
            )
            sponge.append(max(downstream_fraction, lateral_fraction) ** 2)
    points = np.asarray(collocation, dtype=float)
    if points.size == 0:
        raise ValueError("free-surface mesh contains no active points")
    source_points = points.copy()
    source_points[:, 2] = settings.free_surface_source_offset * min(dx, dy)
    return FreeSurfaceMesh(
        collocation_points=points,
        source_points=source_points,
        source_weights=np.full(points.shape[0], dx * dy),
        sponge=np.asarray(sponge, dtype=float),
        dx=dx,
        dy=dy,
        x_bounds=(x_min, x_max),
        y_max=y_max,
    )


def free_surface_operator(
    mesh: FreeSurfaceMesh,
    froude_number: float,
    source_points: np.ndarray,
    source_weights: np.ndarray,
    solver_settings: SolverSettings,
) -> np.ndarray:
    """Build the linearized free-surface operator for one source family.

    A second-order, four-point upstream stencil is used for ``phi_xx``.  In
    the adopted coordinates the undisturbed flow is in the positive x
    direction, so the stencil samples decreasing x.
    """

    potential_blocks = []
    vertical_velocity = None
    for offset in range(4):
        field = mesh.collocation_points.copy()
        field[:, 0] -= offset * mesh.dx
        potential, velocity = influence_matrices(
            field,
            source_points,
            source_weights,
            mirror_y=True,
            mirror_z=False,
            chunk_size=solver_settings.kernel_chunk_size,
            potential=True,
            velocity=(offset == 0),
        )
        assert potential is not None
        potential_blocks.append(potential)
        if offset == 0:
            assert velocity is not None
            vertical_velocity = velocity[:, :, 2]
    assert vertical_velocity is not None
    second_derivative = (
        2.0 * potential_blocks[0]
        - 5.0 * potential_blocks[1]
        + 4.0 * potential_blocks[2]
        - potential_blocks[3]
    ) / mesh.dx**2
    damping = (
        solver_settings.sponge_strength
        * mesh.sponge[:, None]
        * potential_blocks[0]
    )
    return froude_number**2 * second_derivative + vertical_velocity + damping


def minimum_points_per_wavelength(mesh: FreeSurfaceMesh, froude_number: float) -> float:
    """Return streamwise points per fundamental deep-water wavelength."""

    wavelength_over_length = 2.0 * np.pi * froude_number**2
    return float(wavelength_over_length / mesh.dx)
