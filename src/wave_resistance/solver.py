"""Three-dimensional linear free-surface Rankine-source solver."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .free_surface import (
    FreeSurfaceMesh,
    build_free_surface_mesh,
    free_surface_operator,
    minimum_points_per_wavelength,
)
from .geometry import HullOffsets, TriMesh
from .influence import apply_sources, normal_influence, panel_normal_influence
from .kochin import kochin_wave_coefficient
from .models import (
    CaseDiagnostics,
    MeshSettings,
    SolverSettings,
    WaterProperties,
    WaveResistanceResult,
)
from .wave_pattern import transverse_energy_flux_coefficient


@dataclass(frozen=True)
class _PreparedGeometry:
    hull: HullOffsets
    mesh: TriMesh
    collocation: np.ndarray
    normals: np.ndarray
    areas: np.ndarray
    source_points: np.ndarray
    evaluation_points: np.ndarray
    free_surface: FreeSurfaceMesh
    wetted_area_over_l2: float


def _solve_linear_system(
    matrix: np.ndarray,
    right_hand_side: np.ndarray,
    settings: SolverSettings,
) -> Tuple[np.ndarray, str, int, float, float]:
    row_scale = np.maximum(np.max(np.abs(matrix), axis=1), 1.0e-14)
    scaled_matrix = matrix / row_scale[:, None]
    scaled_rhs = right_hand_side / row_scale
    condition = float(np.linalg.cond(scaled_matrix))
    backend = "numpy.solve"
    iterations = 1
    try:
        if matrix.shape[0] > settings.direct_unknown_limit:
            try:
                from scipy.sparse.linalg import gmres  # type: ignore

                iteration_counter = [0]

                def count_iteration(_residual: object) -> None:
                    iteration_counter[0] += 1

                solution, info = gmres(
                    scaled_matrix,
                    scaled_rhs,
                    rtol=settings.gmres_relative_tolerance,
                    atol=0.0,
                    restart=80,
                    maxiter=settings.gmres_max_iterations,
                    callback=count_iteration,
                    callback_type="pr_norm",
                )
                backend = "scipy.gmres"
                iterations = iteration_counter[0]
                if info != 0:
                    raise np.linalg.LinAlgError(f"GMRES did not converge (info={info})")
            except ImportError:
                solution = np.linalg.solve(scaled_matrix, scaled_rhs)
                backend = "numpy.solve (SciPy unavailable)"
        else:
            solution = np.linalg.solve(scaled_matrix, scaled_rhs)
    except np.linalg.LinAlgError:
        solution, _, _, _ = np.linalg.lstsq(scaled_matrix, scaled_rhs, rcond=1.0e-11)
        backend = "numpy.lstsq"
        iterations = 1
    residual = float(
        np.linalg.norm(matrix @ solution - right_hand_side)
        / max(np.linalg.norm(right_hand_side), 1.0)
    )
    return solution, backend, iterations, condition, residual


class LinearFreeSurfaceSolver:
    """Solve steady wave-making by desingularized Rankine-source collocation.

    The method is deliberately a transparent research implementation.  It is
    intended for convergence studies and hull-form screening, not as a
    substitute for a validated nonlinear RANS/VOF calculation.
    """

    def __init__(
        self,
        mesh_settings: Optional[MeshSettings] = None,
        solver_settings: Optional[SolverSettings] = None,
        water: Optional[WaterProperties] = None,
    ) -> None:
        self.mesh_settings = mesh_settings or MeshSettings()
        self.solver_settings = solver_settings or SolverSettings()
        self.water = water or WaterProperties()

    def _prepare(self, hull: HullOffsets) -> _PreparedGeometry:
        resampled = hull.resample(
            self.mesh_settings.hull_stations,
            self.mesh_settings.hull_vertical_points,
        )
        physical_mesh = resampled.to_mesh()
        length = float(resampled.length_ref_m)
        collocation = physical_mesh.centroids / length
        areas = physical_mesh.areas / length**2
        characteristic_size = np.sqrt(areas)
        source_points = collocation.copy()
        evaluation_points = collocation + (
            self.mesh_settings.hull_source_offset
            * characteristic_size[:, None]
            * physical_mesh.normals
        )
        free_surface = build_free_surface_mesh(
            resampled,
            self.mesh_settings,
            sponge_start_fraction=self.solver_settings.sponge_start_fraction,
        )
        return _PreparedGeometry(
            hull=resampled,
            mesh=physical_mesh,
            collocation=collocation,
            normals=physical_mesh.normals,
            areas=areas,
            source_points=source_points,
            evaluation_points=evaluation_points,
            free_surface=free_surface,
            wetted_area_over_l2=2.0 * float(np.sum(areas)),
        )

    def _double_body_solution(
        self, geometry: _PreparedGeometry
    ) -> Tuple[np.ndarray, np.ndarray, float, str, int, float, float]:
        matrix = panel_normal_influence(
            geometry.collocation,
            geometry.normals,
            geometry.source_points,
            geometry.areas,
            mirror_y=True,
            mirror_z=True,
        )
        rhs = -geometry.normals[:, 0]
        strengths, backend, iterations, condition, residual = _solve_linear_system(
            matrix, rhs, self.solver_settings
        )
        _, disturbance_velocity = apply_sources(
            geometry.evaluation_points,
            geometry.source_points,
            geometry.areas,
            strengths,
            mirror_y=True,
            mirror_z=True,
            chunk_size=self.solver_settings.kernel_chunk_size,
        )
        total_velocity = disturbance_velocity.copy()
        total_velocity[:, 0] += 1.0
        normal_component = np.sum(total_velocity * geometry.normals, axis=1)
        total_velocity -= normal_component[:, None] * geometry.normals
        pressure_coefficient = 1.0 - np.sum(total_velocity**2, axis=1)
        drag_l2 = -2.0 * float(
            np.sum(pressure_coefficient * geometry.normals[:, 0] * geometry.areas)
        )
        drag_coefficient = drag_l2 / geometry.wetted_area_over_l2
        return (
            strengths,
            total_velocity,
            drag_coefficient,
            backend,
            iterations,
            condition,
            residual,
        )

    def _coupled_system(
        self, geometry: _PreparedGeometry, froude_number: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        fs = geometry.free_surface
        hull_hull = panel_normal_influence(
            geometry.collocation,
            geometry.normals,
            geometry.source_points,
            geometry.areas,
            mirror_y=True,
            mirror_z=False,
        )
        hull_free_surface = normal_influence(
            geometry.evaluation_points,
            geometry.normals,
            fs.source_points,
            fs.source_weights,
            mirror_y=True,
            mirror_z=False,
            chunk_size=self.solver_settings.kernel_chunk_size,
        )
        free_surface_hull = free_surface_operator(
            fs,
            froude_number,
            geometry.source_points,
            geometry.areas,
            self.solver_settings,
        )
        free_surface_free_surface = free_surface_operator(
            fs,
            froude_number,
            fs.source_points,
            fs.source_weights,
            self.solver_settings,
        )
        matrix = np.block(
            [
                [hull_hull, hull_free_surface],
                [free_surface_hull, free_surface_free_surface],
            ]
        )
        rhs = np.concatenate((-geometry.normals[:, 0], np.zeros(fs.source_points.shape[0])))
        return matrix, rhs

    def _velocity_on_hull(
        self,
        geometry: _PreparedGeometry,
        hull_strengths: np.ndarray,
        free_surface_strengths: np.ndarray,
    ) -> np.ndarray:
        _, hull_velocity = apply_sources(
            geometry.evaluation_points,
            geometry.source_points,
            geometry.areas,
            hull_strengths,
            mirror_y=True,
            chunk_size=self.solver_settings.kernel_chunk_size,
        )
        _, free_surface_velocity = apply_sources(
            geometry.collocation,
            geometry.free_surface.source_points,
            geometry.free_surface.source_weights,
            free_surface_strengths,
            mirror_y=True,
            chunk_size=self.solver_settings.kernel_chunk_size,
        )
        total = hull_velocity + free_surface_velocity
        total[:, 0] += 1.0
        normal_component = np.sum(total * geometry.normals, axis=1)
        total -= normal_component[:, None] * geometry.normals
        return total

    def _wave_field(
        self,
        geometry: _PreparedGeometry,
        froude_number: float,
        hull_strengths: np.ndarray,
        free_surface_strengths: np.ndarray,
    ) -> Tuple[Dict[str, np.ndarray], float]:
        fs = geometry.free_surface
        points = fs.collocation_points
        _, hull_velocity = apply_sources(
            points,
            geometry.source_points,
            geometry.areas,
            hull_strengths,
            mirror_y=True,
            chunk_size=self.solver_settings.kernel_chunk_size,
        )
        _, fs_velocity = apply_sources(
            points,
            fs.source_points,
            fs.source_weights,
            free_surface_strengths,
            mirror_y=True,
            chunk_size=self.solver_settings.kernel_chunk_size,
        )
        elevation = -froude_number**2 * (hull_velocity[:, 0] + fs_velocity[:, 0])

        stern = geometry.hull.x_m[-1] / geometry.hull.length_ref_m
        x_cut_value = min(
            stern + 0.55 * self.mesh_settings.downstream_lengths,
            fs.x_bounds[1] - 2.0 * fs.dx,
        )
        y_cut = np.linspace(
            -0.95 * fs.y_max,
            0.95 * fs.y_max,
            self.solver_settings.wave_cut_points,
        )
        cut_points = np.column_stack(
            (np.full_like(y_cut, x_cut_value), y_cut, np.zeros_like(y_cut))
        )
        _, cut_hull_velocity = apply_sources(
            cut_points,
            geometry.source_points,
            geometry.areas,
            hull_strengths,
            mirror_y=True,
            chunk_size=self.solver_settings.kernel_chunk_size,
        )
        _, cut_fs_velocity = apply_sources(
            cut_points,
            fs.source_points,
            fs.source_weights,
            free_surface_strengths,
            mirror_y=True,
            chunk_size=self.solver_settings.kernel_chunk_size,
        )
        cut_elevation = -froude_number**2 * (
            cut_hull_velocity[:, 0] + cut_fs_velocity[:, 0]
        )
        if self.solver_settings.include_wave_pattern_diagnostic:
            pattern_coefficient = transverse_energy_flux_coefficient(
                y_cut,
                cut_elevation,
                froude_number,
                geometry.wetted_area_over_l2,
            )
        else:
            pattern_coefficient = float("nan")
        fields = {
            "x": points[:, 0],
            "y": points[:, 1],
            "elevation_over_l": elevation,
            "transverse_cut_x": np.full_like(y_cut, x_cut_value),
            "transverse_cut_y": y_cut,
            "transverse_cut_elevation_over_l": cut_elevation,
        }
        return fields, pattern_coefficient

    def solve(
        self,
        hull: HullOffsets,
        froude_numbers: Sequence[float],
        *,
        retain_wave_fields: bool = False,
    ) -> WaveResistanceResult:
        """Solve one symmetric hull over a sequence of Froude numbers."""

        if not isinstance(hull, HullOffsets):
            raise TypeError("hull must be a HullOffsets instance")
        fn = np.atleast_1d(np.asarray(froude_numbers, dtype=float))
        if fn.ndim != 1 or fn.size == 0 or not np.all(np.isfinite(fn)) or np.any(fn <= 0.0):
            raise ValueError("froude_numbers must be a non-empty finite positive sequence")
        outside = (fn < self.solver_settings.fn_min) | (fn > self.solver_settings.fn_max)
        if np.any(outside) and self.solver_settings.reject_outside_envelope:
            raise ValueError(
                "Froude numbers must lie in [{:.3g}, {:.3g}]".format(
                    self.solver_settings.fn_min, self.solver_settings.fn_max
                )
            )

        geometry = self._prepare(hull)
        (
            _,
            double_body_velocity,
            double_body_drag,
            _,
            _,
            _,
            _,
        ) = self._double_body_solution(geometry)
        double_body_pressure = 1.0 - np.sum(double_body_velocity**2, axis=1)
        # The radiating Neumann-Kelvin problem uses the linearized Havelock
        # source distribution.  It is exactly -n_x on the mean hull surface;
        # retaining full 3-D panel coordinates preserves transverse phase and
        # finite-breadth interference.  The solved double-body distribution is
        # used above as a nonlinear near-field diagnostic, not substituted
        # into the linear far-field condition.
        radiation_strengths = -geometry.normals[:, 0]

        speeds = []
        resistances = []
        kochin_coefficients = []
        pressure_coefficients = []
        cut_energy_coefficients = []
        diagnostics = []
        wave_fields: Optional[Dict[float, Dict[str, np.ndarray]]] = (
            {} if retain_wave_fields else None
        )
        length = float(hull.length_ref_m)
        for fn_value_raw in fn:
            fn_value = float(fn_value_raw)
            matrix, rhs = self._coupled_system(geometry, fn_value)
            strengths, backend, iterations, condition, residual = _solve_linear_system(
                matrix, rhs, self.solver_settings
            )
            hull_count = geometry.source_points.shape[0]
            hull_strengths = strengths[:hull_count]
            free_surface_strengths = strengths[hull_count:]
            total_velocity = self._velocity_on_hull(
                geometry, hull_strengths, free_surface_strengths
            )
            full_pressure = 1.0 - np.sum(total_velocity**2, axis=1)
            wave_pressure = full_pressure - double_body_pressure
            force_coefficient_l2 = -2.0 * float(
                np.sum(wave_pressure * geometry.normals[:, 0] * geometry.areas)
            )
            pressure_coefficient = force_coefficient_l2 / geometry.wetted_area_over_l2
            kochin_coefficient, kochin_tail_fraction = kochin_wave_coefficient(
                geometry.collocation,
                geometry.areas,
                radiation_strengths,
                fn_value,
                geometry.wetted_area_over_l2,
                integration_points=self.solver_settings.kochin_integration_points,
                t_max=self.solver_settings.kochin_t_max,
                chunk_size=self.solver_settings.kernel_chunk_size,
            )
            fields, cut_energy_coefficient = self._wave_field(
                geometry, fn_value, hull_strengths, free_surface_strengths
            )
            if wave_fields is not None:
                wave_fields[fn_value] = fields
            speed = fn_value * math.sqrt(self.water.gravity_m_s2 * length)
            resistance = (
                0.5
                * self.water.density_kg_m3
                * speed**2
                * geometry.wetted_area_over_l2
                * length**2
                * kochin_coefficient
            )
            difference = abs(pressure_coefficient - kochin_coefficient) / max(
                abs(kochin_coefficient), 1.0e-12
            )
            points_per_wavelength = minimum_points_per_wavelength(
                geometry.free_surface, fn_value
            )
            flags = []
            if not (self.solver_settings.fn_min <= fn_value <= self.solver_settings.fn_max):
                flags.append("outside declared Froude-number envelope")
            if residual > self.solver_settings.linear_residual_tolerance:
                flags.append("linear-system residual above tolerance")
            if condition > self.solver_settings.condition_warning:
                flags.append("ill-conditioned influence matrix")
            if kochin_tail_fraction > self.solver_settings.kochin_tail_tolerance:
                flags.append("Kochin integral tail above tolerance")
            if pressure_coefficient < 0.0:
                flags.append("negative near-field pressure diagnostic")
            if abs(double_body_drag) > 1.0e-3:
                flags.append("double-body drag bias exceeds 1e-3")
            if points_per_wavelength < 8.0:
                flags.append("fewer than eight streamwise points per wavelength")
            if difference > 0.05:
                flags.append("near-field pressure and Kochin coefficients differ by more than 5%")
            diagnostics.append(
                CaseDiagnostics(
                    unknown_count=matrix.shape[0],
                    matrix_condition=condition,
                    relative_residual=residual,
                    solver_backend=backend,
                    solver_iterations=iterations,
                    double_body_drag_coefficient=double_body_drag,
                    kochin_wave_coefficient=kochin_coefficient,
                    kochin_tail_fraction=kochin_tail_fraction,
                    near_field_pressure_coefficient=pressure_coefficient,
                    wave_cut_energy_coefficient=cut_energy_coefficient,
                    pressure_kochin_difference=difference,
                    minimum_points_per_transverse_wavelength=points_per_wavelength,
                    flags=tuple(flags),
                )
            )
            speeds.append(speed)
            resistances.append(resistance)
            kochin_coefficients.append(kochin_coefficient)
            pressure_coefficients.append(pressure_coefficient)
            cut_energy_coefficients.append(cut_energy_coefficient)

        return WaveResistanceResult(
            froude_number=fn.copy(),
            speed_m_s=np.asarray(speeds),
            wave_resistance_N=np.asarray(resistances),
            wave_resistance_coefficient=np.asarray(kochin_coefficients),
            near_field_pressure_coefficient=np.asarray(pressure_coefficients),
            wave_cut_energy_coefficient=np.asarray(cut_energy_coefficients),
            diagnostics=tuple(diagnostics),
            length_ref_m=length,
            wetted_area_m2=geometry.wetted_area_over_l2 * length**2,
            water=self.water,
            wave_fields=wave_fields,
        )
