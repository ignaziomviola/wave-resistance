"""Dense exact-body Rankine-panel reference solvers.

The classes in this module are intentionally separate from the established
Michell API. Coordinates follow :class:`OffsetHull`: ``x`` is aft-to-forward,
``z`` is positive downward, and the body-fixed incident stream is in ``-x``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
import math

import numpy as np

from .models import WaterProperties
from .bem.assembly import (
    assemble_source_operators,
    normal_influence,
    potential_influence,
    reconstruct_surface_gradient,
)
from .bem.diagnostics import ConvergenceStatus, PotentialFlowFailure, UnsupportedPotentialFlow
from .bem.mesh import SurfaceMesh, offset_hull_mesh, rectangular_free_surface
from .bem.resistance import downstream_energy_flux, mixed_force_balance, pressure_force


@dataclass(frozen=True)
class GeometrySettings:
    free_surface_nx: int = 17
    free_surface_ny: int = 13

    def __post_init__(self):
        if self.free_surface_nx < 5 or self.free_surface_ny < 5 or self.free_surface_ny % 2 == 0:
            raise ValueError("free-surface dimensions require nx>=5 and odd ny>=5")


@dataclass(frozen=True)
class PhysicsSettings:
    froude_number: float = 0.25
    fixed_sinkage_m: float = 0.0
    fixed_trim_rad: float = 0.0
    water: WaterProperties = field(default_factory=WaterProperties)

    def __post_init__(self):
        if not math.isfinite(self.froude_number) or self.froude_number <= 0.0:
            raise ValueError("froude_number must be finite and positive")
        if self.fixed_sinkage_m != 0.0 or self.fixed_trim_rad != 0.0:
            raise UnsupportedPotentialFlow("nonzero sinkage/trim geometry transforms are not released")


@dataclass(frozen=True)
class BEMSettings:
    quadrature_order: int = 10
    algebraic_tolerance: float = 1.0e-7
    condition_limit: float = 1.0e13
    derivative_step_fraction: float = 0.15

    def __post_init__(self):
        if self.quadrature_order < 3 or self.algebraic_tolerance <= 0.0:
            raise ValueError("invalid BEM settings")
        if not 0.01 <= self.derivative_step_fraction <= 0.5:
            raise ValueError("derivative_step_fraction must lie in [0.01, 0.5]")


@dataclass(frozen=True)
class FreeSurfaceSettings:
    upstream_lengths: float = 1.0
    downstream_lengths: float = 2.0
    lateral_lengths: float = 1.0
    sponge_fraction: float = 0.2
    sponge_strength: float = 0.15

    def __post_init__(self):
        if min(self.upstream_lengths, self.downstream_lengths, self.lateral_lengths) <= 0.0:
            raise ValueError("free-surface extents must be positive")
        if not 0.0 < self.sponge_fraction < 0.5 or self.sponge_strength < 0.0:
            raise ValueError("invalid sponge settings")


@dataclass(frozen=True)
class NonlinearSettings:
    mode: str = "fixed_waterline_nonlinear"
    homotopy_steps: int = 3
    max_iterations: int = 18
    relaxation: float = 0.5
    residual_tolerance: float = 5.0e-4
    update_tolerance: float = 5.0e-5
    max_slope: float = 0.5

    def __post_init__(self):
        if self.mode != "fixed_waterline_nonlinear":
            raise UnsupportedPotentialFlow(
                "only the explicitly approximate fixed_waterline_nonlinear mode is released"
            )
        if self.homotopy_steps < 1 or self.max_iterations < 1:
            raise ValueError("continuation and iteration counts must be positive")
        if not 0.0 < self.relaxation <= 1.0:
            raise ValueError("relaxation must lie in (0, 1]")


@dataclass(frozen=True)
class ConvergenceSettings:
    force_relative_tolerance: float = 0.02
    absolute_force_tolerance: float = 1.0e-6
    require_mesh_study: bool = False
    require_domain_study: bool = False


@dataclass
class PotentialFlowResult:
    method_name: str
    formulation_version: str
    froude_number: float
    speed_m_s: float
    resistance_pressure_N: float
    resistance_far_field_N: float
    coefficient_pressure: float
    coefficient_far_field: float
    force_balance_discrepancy_N: float
    potential: np.ndarray
    surface_velocity: np.ndarray
    hull_pressure_Pa: np.ndarray
    free_surface_vertices: np.ndarray
    free_surface_faces: np.ndarray
    elevation: np.ndarray
    wetted_vertices: np.ndarray
    wetted_faces: np.ndarray
    linear_residual_history: list
    nonlinear_residual_history: list
    metadata: dict
    status: ConvergenceStatus
    failure_reasons: tuple = ()
    wave_cuts: dict = field(default_factory=dict)

    @property
    def accepted(self):
        return self.status.accepted

    def to_json(self, path):
        scalar_names = (
            "method_name", "formulation_version", "froude_number", "speed_m_s",
            "resistance_pressure_N", "resistance_far_field_N", "coefficient_pressure",
            "coefficient_far_field", "force_balance_discrepancy_N", "failure_reasons",
        )
        data = {}
        for name in scalar_names:
            value = getattr(self, name)
            data[name] = None if isinstance(value, float) and not math.isfinite(value) else value
        data["status"] = asdict(self.status)
        data["metadata"] = self.metadata
        data["linear_residual_history"] = self.linear_residual_history
        data["nonlinear_residual_history"] = self.nonlinear_residual_history
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def to_npz(self, path):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            "potential": self.potential,
            "surface_velocity": self.surface_velocity,
            "hull_pressure_Pa": self.hull_pressure_Pa,
            "free_surface_vertices": self.free_surface_vertices,
            "free_surface_faces": self.free_surface_faces,
            "elevation": self.elevation,
            "wetted_vertices": self.wetted_vertices,
            "wetted_faces": self.wetted_faces,
        }
        arrays.update({f"wave_cut_{key}": value for key, value in self.wave_cuts.items()})
        np.savez_compressed(target, **arrays)


def _speed(hull, physics):
    return physics.froude_number * math.sqrt(
        physics.water.g_m_s2 * hull.metadata.length_ref_m
    )


def _dense_solve(matrix, rhs):
    row_scale = np.maximum(np.max(np.abs(matrix), axis=1), 1.0e-14)
    scaled = matrix / row_scale[:, None]
    scaled_rhs = rhs / row_scale
    condition = float(np.linalg.cond(scaled))
    try:
        solution = np.linalg.solve(scaled, scaled_rhs)
        backend = "numpy.solve"
    except np.linalg.LinAlgError:
        solution = np.linalg.lstsq(scaled, scaled_rhs, rcond=1.0e-11)[0]
        backend = "numpy.lstsq"
    residual = float(np.linalg.norm(matrix @ solution - rhs) / max(np.linalg.norm(rhs), 1.0))
    return solution, residual, condition, backend


def _sponge(points, hull, settings):
    x = points[:, 0]
    y = np.abs(points[:, 1])
    length = float(hull.metadata.length_ref_m)
    downstream_start = hull.x_m[0] - (1.0 - settings.sponge_fraction) * settings.downstream_lengths * length
    x_end = hull.x_m[0] - settings.downstream_lengths * length
    lateral_start = (1.0 - settings.sponge_fraction) * settings.lateral_lengths * length
    y_end = settings.lateral_lengths * length
    sx = np.clip((downstream_start - x) / max(downstream_start - x_end, 1.0e-12), 0.0, 1.0) ** 2
    sy = np.clip((y - lateral_start) / max(y_end - lateral_start, 1.0e-12), 0.0, 1.0) ** 2
    return np.maximum(sx, sy)


def _linear_fs_operator(points, source_mesh, speed, gravity, length, bem, free_surface):
    h = bem.derivative_step_fraction * math.sqrt(float(np.median(source_mesh.areas)))
    h = max(h, 1.0e-5 * length)
    blocks = []
    for offset in range(4):
        shifted = points.copy()
        shifted[:, 0] += offset * h
        blocks.append(potential_influence(shifted, source_mesh, bem.quadrature_order))
    dxx = (2.0 * blocks[0] - 5.0 * blocks[1] + 4.0 * blocks[2] - blocks[3]) / h**2
    normals = np.tile((0.0, 0.0, 1.0), (len(points), 1))
    dz = normal_influence(points, normals, source_mesh, bem.quadrature_order)
    return -speed**2 / gravity * dxx + dz, blocks[0]


def _vertex_elevation(fs_mesh, hull_mesh, hull_sigma, fs_sigma, speed, gravity, bem):
    length = float(fs_mesh.physical_geometry.metadata.length_ref_m)
    h = max(bem.derivative_step_fraction * math.sqrt(float(np.median(fs_mesh.areas))), 1.0e-5 * length)
    plus, minus = fs_mesh.vertices.copy(), fs_mesh.vertices.copy()
    plus[:, 0] += h
    minus[:, 0] -= h
    phi_plus = potential_influence(plus, hull_mesh, bem.quadrature_order) @ hull_sigma
    phi_plus += potential_influence(plus, fs_mesh, bem.quadrature_order) @ fs_sigma
    phi_minus = potential_influence(minus, hull_mesh, bem.quadrature_order) @ hull_sigma
    phi_minus += potential_influence(minus, fs_mesh, bem.quadrature_order) @ fs_sigma
    eta = -speed * (phi_plus - phi_minus) / (2.0 * h * gravity)
    boundary = np.unique(fs_mesh.boundary_edges)
    eta[boundary] = 0.0
    return eta


def _far_field_from_mesh(fs_mesh, eta, speed, water):
    xmax = np.max(fs_mesh.vertices[:, 0])
    x_values = np.unique(fs_mesh.vertices[:, 0])
    cut_x = x_values[1] if len(x_values) > 1 else xmax
    ids = np.flatnonzero(np.isclose(fs_mesh.vertices[:, 0], cut_x))
    order = np.argsort(fs_mesh.vertices[ids, 1])
    ids = ids[order]
    y, elevation = fs_mesh.vertices[ids, 1], eta[ids]
    power = downstream_energy_flux(y, elevation, speed, water.rho_kg_m3, water.g_m_s2)
    resistance = power / speed if speed > 0.0 else 0.0
    return resistance, {"x": np.full_like(y, cut_x), "y": y, "elevation": elevation}


class DoubleBodyPotentialFlowSolver:
    def __init__(self, hull, physics=None, bem=None, convergence=None):
        self.hull = hull
        self.physics = physics or PhysicsSettings()
        self.bem = bem or BEMSettings()
        self.convergence = convergence or ConvergenceSettings()

    def _validate(self):
        if not hasattr(self.hull, "half_breadth_m") or not hasattr(self.hull, "metadata"):
            raise UnsupportedPotentialFlow("only OffsetHull geometries are released")
        if np.any(self.hull.half_breadth_m < 0.0):
            raise UnsupportedPotentialFlow("asymmetric or invalid offsets are unsupported")

    def solve(self):
        self._validate()
        mesh = offset_hull_mesh(self.hull)
        speed = _speed(self.hull, self.physics)
        single_layer, normal_derivative = assemble_source_operators(mesh, self.bem.quadrature_order)
        rhs = speed * mesh.normals[:, 0]
        sigma, residual, condition, _ = _dense_solve(normal_derivative, rhs)
        potential = single_layer @ sigma
        disturbance_gradient = reconstruct_surface_gradient(mesh, potential)
        total_velocity = disturbance_gradient + np.array([-speed, 0.0, 0.0])
        total_velocity -= np.sum(total_velocity * mesh.normals, axis=1)[:, None] * mesh.normals
        water = self.physics.water
        pressure = 0.5 * water.rho_kg_m3 * (speed**2 - np.sum(total_velocity**2, axis=1))
        resistance = float(-pressure_force(mesh, pressure)[0])
        reference_area = self.hull.wetted_area_m2
        denominator = 0.5 * water.rho_kg_m3 * speed**2 * reference_area
        coefficient = resistance / denominator
        algebraic = residual <= self.bem.algebraic_tolerance and condition <= self.bem.condition_limit
        force_ok = abs(coefficient) <= self.convergence.force_relative_tolerance
        mesh_ok = not self.convergence.require_mesh_study
        domain_ok = not self.convergence.require_domain_study
        status = ConvergenceStatus(algebraic, True, force_ok, mesh_ok, domain_ok, algebraic and force_ok and mesh_ok and domain_ok)
        reasons = []
        if not algebraic:
            reasons.append("double-body algebraic solve failed")
        if not force_ok:
            reasons.append("double-body d'Alembert force check failed")
        return PotentialFlowResult(
            "double-body-rankine", "constant-source-rankine-v1", self.physics.froude_number,
            speed, resistance, 0.0, coefficient, 0.0, abs(resistance), potential,
            total_velocity, pressure, np.empty((0, 3)), np.empty((0, 3), dtype=int),
            np.empty(0), mesh.vertices, mesh.faces,
            [{"relative_residual": residual, "condition": condition, "source_flux": float(mesh.areas @ sigma)}],
            [], {"mesh": mesh.diagnostics(), "actual_wetted_area_m2": float(mesh.areas.sum()), "reference_wetted_area_m2": reference_area},
            status, tuple(reasons), {},
        )


class LinearPotentialFlowSolver(DoubleBodyPotentialFlowSolver):
    def __init__(self, hull, physics=None, geometry=None, bem=None, free_surface=None, convergence=None):
        super().__init__(hull, physics, bem, convergence)
        self.geometry = geometry or GeometrySettings()
        self.free_surface = free_surface or FreeSurfaceSettings()

    def _solve_coupled(self, fs_mesh):
        hull_mesh = offset_hull_mesh(self.hull)
        speed = _speed(self.hull, self.physics)
        _, hull_hull = assemble_source_operators(hull_mesh, self.bem.quadrature_order)
        hull_fs = normal_influence(hull_mesh.centroids, hull_mesh.normals, fs_mesh, self.bem.quadrature_order)
        fs_hull, fs_hull_potential = _linear_fs_operator(
            fs_mesh.centroids, hull_mesh, speed, self.physics.water.g_m_s2,
            self.hull.metadata.length_ref_m, self.bem, self.free_surface,
        )
        fs_fs, fs_fs_potential = _linear_fs_operator(
            fs_mesh.centroids, fs_mesh, speed, self.physics.water.g_m_s2,
            self.hull.metadata.length_ref_m, self.bem, self.free_surface,
        )
        np.fill_diagonal(fs_fs, np.diag(fs_fs) - 0.5)
        sponge = _sponge(fs_mesh.centroids, self.hull, self.free_surface)
        fs_fs += self.free_surface.sponge_strength * sponge[:, None] * fs_fs_potential / self.hull.metadata.length_ref_m
        matrix = np.block([[hull_hull, hull_fs], [fs_hull, fs_fs]])
        rhs = np.r_[speed * hull_mesh.normals[:, 0], np.zeros(len(fs_mesh.faces))]
        strengths, residual, condition, backend = _dense_solve(matrix, rhs)
        return hull_mesh, strengths[:len(hull_mesh.faces)], strengths[len(hull_mesh.faces):], residual, condition, backend

    def solve(self):
        self._validate()
        fs_mesh = rectangular_free_surface(
            self.hull, self.geometry.free_surface_nx, self.geometry.free_surface_ny,
            self.free_surface.upstream_lengths, self.free_surface.downstream_lengths,
            self.free_surface.lateral_lengths,
        )
        hull_mesh, hull_sigma, fs_sigma, residual, condition, backend = self._solve_coupled(fs_mesh)
        speed = _speed(self.hull, self.physics)
        potential = potential_influence(hull_mesh.centroids, hull_mesh, self.bem.quadrature_order) @ hull_sigma
        potential += potential_influence(hull_mesh.centroids, fs_mesh, self.bem.quadrature_order) @ fs_sigma
        velocity = reconstruct_surface_gradient(hull_mesh, potential) + np.array([-speed, 0.0, 0.0])
        velocity -= np.sum(velocity * hull_mesh.normals, axis=1)[:, None] * hull_mesh.normals
        water = self.physics.water
        pressure = 0.5 * water.rho_kg_m3 * (speed**2 - np.sum(velocity**2, axis=1))
        resistance = float(-pressure_force(hull_mesh, pressure)[0])
        eta = _vertex_elevation(fs_mesh, hull_mesh, hull_sigma, fs_sigma, speed, water.g_m_s2, self.bem)
        far, wave_cut = _far_field_from_mesh(fs_mesh, eta, speed, water)
        balance = mixed_force_balance(resistance, far, self.convergence.force_relative_tolerance, self.convergence.absolute_force_tolerance)
        reference_area = self.hull.wetted_area_m2
        denominator = 0.5 * water.rho_kg_m3 * speed**2 * reference_area
        algebraic = residual <= self.bem.algebraic_tolerance and condition <= self.bem.condition_limit
        mesh_ok = not self.convergence.require_mesh_study
        domain_ok = not self.convergence.require_domain_study
        status = ConvergenceStatus(algebraic, algebraic, balance, mesh_ok, domain_ok, algebraic and balance and mesh_ok and domain_ok)
        reasons = []
        if not algebraic:
            reasons.append("linear BEM system did not satisfy residual/condition limits")
        if not balance:
            reasons.append("pressure and downstream wave-flux forces disagree")
        return PotentialFlowResult(
            "linear-exact-body-rankine", "constant-source-rankine-v1", self.physics.froude_number,
            speed, resistance, far, resistance / denominator, far / denominator,
            abs(resistance - far), potential, velocity, pressure,
            fs_mesh.vertices.copy(), fs_mesh.faces.copy(), eta, hull_mesh.vertices,
            hull_mesh.faces, [{"relative_residual": residual, "condition": condition, "backend": backend}],
            [], {"hull_mesh": hull_mesh.diagnostics(), "free_surface_mesh": fs_mesh.diagnostics(),
                 "actual_wetted_area_m2": float(hull_mesh.areas.sum()), "reference_wetted_area_m2": reference_area,
                 "radiation": "four-point upstream derivative plus downstream/lateral quadratic sponge"},
            status, tuple(reasons), wave_cut,
        )
