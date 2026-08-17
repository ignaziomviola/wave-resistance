"""Restricted, fixed-waterline nonlinear free-surface continuation."""

import numpy as np

from .bem.assembly import normal_influence, potential_influence, reconstruct_surface_gradient
from .bem.diagnostics import ConvergenceStatus
from .bem.free_surface import nonlinear_residual
from .bem.mesh import SurfaceMesh, offset_hull_mesh, rectangular_free_surface
from .bem.resistance import mixed_force_balance, pressure_force
from .potential_flow import (
    LinearPotentialFlowSolver,
    NonlinearSettings,
    _dense_solve,
    _far_field_from_mesh,
    _speed,
    _sponge,
)


def _panel_to_vertex(mesh, values):
    total = np.zeros(len(mesh.vertices))
    count = np.zeros(len(mesh.vertices))
    for face, value in zip(mesh.faces, values):
        total[face] += value
        count[face] += 1.0
    return total / np.maximum(count, 1.0)


def _vertex_slopes(mesh, elevation):
    panel_gradient = reconstruct_surface_gradient(mesh, np.mean(elevation[mesh.faces], axis=1))
    gx = _panel_to_vertex(mesh, panel_gradient[:, 0])
    gy = _panel_to_vertex(mesh, panel_gradient[:, 1])
    return gx, gy


class NonlinearPotentialFlowSolver(LinearPotentialFlowSolver):
    """Pseudo-time/homotopy solve of the exact steady graph conditions.

    The waterline and outer-domain boundary vertices remain fixed. Interior
    vertices move vertically. A converged step satisfies exact no-flux on the
    moved surface and exact atmospheric Bernoulli head.
    """

    def __init__(self, *args, nonlinear=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.nonlinear = nonlinear or NonlinearSettings()

    def _exact_streamline_solve(self, hull_mesh, fs_mesh):
        speed = _speed(self.hull, self.physics)
        hull_hull = normal_influence(
            hull_mesh.centroids, hull_mesh.normals, hull_mesh, self.bem.quadrature_order
        )
        hull_fs = normal_influence(
            hull_mesh.centroids, hull_mesh.normals, fs_mesh, self.bem.quadrature_order
        )
        fs_hull = normal_influence(
            fs_mesh.centroids, fs_mesh.normals, hull_mesh, self.bem.quadrature_order
        )
        fs_fs = normal_influence(
            fs_mesh.centroids, fs_mesh.normals, fs_mesh, self.bem.quadrature_order
        )
        np.fill_diagonal(hull_hull, -0.5)
        np.fill_diagonal(fs_fs, -0.5)
        # Weakly drive source strength to zero in the sponge while retaining
        # the exact kinematic equation in the interior.
        damping = self.free_surface.sponge_strength * _sponge(
            fs_mesh.centroids, self.hull, self.free_surface
        )
        fs_fs += np.diag(damping)
        matrix = np.block([[hull_hull, hull_fs], [fs_hull, fs_fs]])
        rhs = -speed * np.r_[hull_mesh.normals[:, 0], fs_mesh.normals[:, 0]]
        strengths, residual, condition, backend = _dense_solve(matrix, rhs)
        nh = len(hull_mesh.faces)
        return strengths[:nh], strengths[nh:], residual, condition, backend

    def solve(self):
        linear = super().solve()
        hull_mesh = offset_hull_mesh(self.hull)
        fs_mesh = rectangular_free_surface(
            self.hull, self.geometry.free_surface_nx, self.geometry.free_surface_ny,
            self.free_surface.upstream_lengths, self.free_surface.downstream_lengths,
            self.free_surface.lateral_lengths,
        )
        eta_linear = linear.elevation.copy()
        eta = eta_linear.copy()
        fixed = np.zeros(len(fs_mesh.vertices), dtype=bool)
        fixed[np.unique(fs_mesh.boundary_edges)] = True
        length = float(self.hull.length_ref_m)
        waterline_y = self.hull.half_breadth_at_waterline(fs_mesh.vertices[:, 0])
        fixed |= (
            (fs_mesh.vertices[:, 0] >= self.hull.x_m[0] - 1.0e-10 * length)
            & (fs_mesh.vertices[:, 0] <= self.hull.x_m[-1] + 1.0e-10 * length)
            & np.isclose(np.abs(fs_mesh.vertices[:, 1]), waterline_y, atol=1.0e-9 * length)
        )
        eta[fixed] = 0.0
        history = []
        converged = True
        hull_sigma = fs_sigma = None
        last_algebraic = np.inf
        last_condition = np.inf
        speed = _speed(self.hull, self.physics)
        gravity = self.physics.water.gravity_m_s2

        for lam in np.linspace(0.0, 1.0, self.nonlinear.homotopy_steps + 1)[1:]:
            step_converged = False
            for iteration in range(self.nonlinear.max_iterations):
                vertices = fs_mesh.vertices.copy()
                vertices[:, 2] = eta
                try:
                    moved = SurfaceMesh(vertices, fs_mesh.faces, fs_mesh.tags, self.hull)
                    hull_sigma, fs_sigma, last_algebraic, last_condition, backend = self._exact_streamline_solve(hull_mesh, moved)
                except (ValueError, np.linalg.LinAlgError):
                    converged = False
                    history.append({"lambda": float(lam), "iteration": iteration, "failure": "invalid or singular moved mesh"})
                    break

                hull_phi = potential_influence(hull_mesh.centroids, hull_mesh, self.bem.quadrature_order) @ hull_sigma
                hull_phi += potential_influence(hull_mesh.centroids, moved, self.bem.quadrature_order) @ fs_sigma
                fs_phi = potential_influence(moved.centroids, hull_mesh, self.bem.quadrature_order) @ hull_sigma
                fs_phi += potential_influence(moved.centroids, moved, self.bem.quadrature_order) @ fs_sigma
                hull_velocity = reconstruct_surface_gradient(hull_mesh, hull_phi) + np.array([speed, 0.0, 0.0])
                hull_velocity -= np.sum(hull_velocity * hull_mesh.normals, axis=1)[:, None] * hull_mesh.normals
                fs_velocity = reconstruct_surface_gradient(moved, fs_phi) + np.array([speed, 0.0, 0.0])
                fs_velocity -= np.sum(fs_velocity * moved.normals, axis=1)[:, None] * moved.normals

                panel_eta = moved.centroids[:, 2]
                if np.any(moved.normals[:, 2] <= 0.0):
                    converged = False
                    history.append({"lambda": float(lam), "iteration": iteration, "failure": "loss of single-valued graph"})
                    break
                # For a graph with upward normal n proportional to
                # (-eta_x,-eta_y,1), these are the exact planar-panel slopes.
                eta_x_panel = -moved.normals[:, 0] / moved.normals[:, 2]
                eta_y_panel = -moved.normals[:, 1] / moved.normals[:, 2]
                disturbance = fs_velocity - np.array([speed, 0.0, 0.0])
                kin, dyn = nonlinear_residual(
                    disturbance, panel_eta, eta_x_panel, eta_y_panel, speed, gravity
                )
                exact_target_panel = panel_eta - dyn / gravity
                exact_target = _panel_to_vertex(moved, exact_target_panel)
                target = (1.0 - lam) * eta_linear + lam * exact_target
                target[fixed] = 0.0
                target *= 1.0 - _sponge(vertices, self.hull, self.free_surface)
                homotopy_dynamic = gravity * (eta - target)
                update = self.nonlinear.relaxation * (target - eta)
                update[fixed] = 0.0
                eta += update
                eta[fixed] = 0.0
                eta_x, eta_y = _vertex_slopes(moved, eta)
                slope = float(np.max(np.hypot(eta_x, eta_y)))
                record = {
                    "lambda": float(lam), "iteration": int(iteration),
                    "bem": float(last_algebraic),
                    "kinematic": float(np.max(np.abs(kin))),
                    "dynamic": float(np.max(np.abs(homotopy_dynamic)) / max(gravity * length, 1.0e-30)),
                    "exact_dynamic": float(
                        np.max(np.abs(dyn[_sponge(moved.centroids, self.hull, self.free_surface) < 0.1]))
                        / max(gravity * length, 1.0e-30)
                    ),
                    "update": float(np.max(np.abs(update)) / length),
                    "waterline": float(np.max(np.abs(eta[fixed])) / length),
                    "geometry": 0.0, "max_slope": slope, "backend": backend,
                }
                history.append(record)
                if not np.all(np.isfinite(eta)) or slope > self.nonlinear.max_slope:
                    converged = False
                    break
                if (
                    record["kinematic"] / speed <= self.nonlinear.residual_tolerance
                    and record["dynamic"] <= self.nonlinear.residual_tolerance
                    and record["update"] <= self.nonlinear.update_tolerance
                ):
                    step_converged = True
                    break
            if not step_converged:
                converged = False
                break

        final_vertices = fs_mesh.vertices.copy()
        final_vertices[:, 2] = eta
        final_mesh = SurfaceMesh(final_vertices, fs_mesh.faces, fs_mesh.tags, self.hull)
        if hull_sigma is None or fs_sigma is None:
            linear.method_name = "nonlinear-exact-body-rankine-fixed-waterline"
            linear.status = ConvergenceStatus(linear.status.algebraic_converged, False, False, linear.status.mesh_converged, linear.status.domain_converged, False)
            linear.failure_reasons = tuple(dict.fromkeys(linear.failure_reasons + ("nonlinear continuation failed",)))
            linear.nonlinear_residual_history = history
            return linear

        hull_phi = potential_influence(hull_mesh.centroids, hull_mesh, self.bem.quadrature_order) @ hull_sigma
        hull_phi += potential_influence(hull_mesh.centroids, final_mesh, self.bem.quadrature_order) @ fs_sigma
        hull_velocity = reconstruct_surface_gradient(hull_mesh, hull_phi) + np.array([speed, 0.0, 0.0])
        hull_velocity -= np.sum(hull_velocity * hull_mesh.normals, axis=1)[:, None] * hull_mesh.normals
        water = self.physics.water
        pressure = 0.5 * water.density_kg_m3 * (speed**2 - np.sum(hull_velocity**2, axis=1))
        resistance = float(pressure_force(hull_mesh, pressure)[0])
        far, wave_cut = _far_field_from_mesh(final_mesh, eta, speed, water)
        balance = mixed_force_balance(
            resistance, far, self.convergence.force_relative_tolerance,
            self.convergence.absolute_force_tolerance,
        )
        algebraic = last_algebraic <= self.bem.algebraic_tolerance and last_condition <= self.bem.condition_limit
        mesh_ok = not self.convergence.require_mesh_study
        domain_ok = not self.convergence.require_domain_study
        accepted = algebraic and converged and balance and mesh_ok and domain_ok
        denominator = 0.5 * water.density_kg_m3 * speed**2 * self.hull.hydrostatics.wetted_area_m2
        reasons = []
        if not algebraic:
            reasons.append("nonlinear BEM system failed residual/condition limits")
        if not converged:
            reasons.append("nonlinear continuation failed")
        if not balance:
            reasons.append("pressure and downstream wave-flux forces disagree")
        linear.method_name = "nonlinear-exact-body-rankine-fixed-waterline"
        linear.resistance_pressure_N = resistance
        linear.resistance_far_field_N = far
        linear.coefficient_pressure = resistance / denominator
        linear.coefficient_far_field = far / denominator
        linear.force_balance_discrepancy_N = abs(resistance - far)
        linear.potential = hull_phi
        linear.surface_velocity = hull_velocity
        linear.hull_pressure_Pa = pressure
        linear.free_surface_vertices = final_vertices
        linear.elevation = eta
        linear.nonlinear_residual_history = history
        linear.wave_cuts = wave_cut
        linear.metadata["nonlinear_mode"] = self.nonlinear.mode
        linear.metadata["fixed_waterline_is_approximation"] = True
        linear.metadata["nonlinear_condition"] = float(last_condition)
        linear.status = ConvergenceStatus(algebraic, converged, balance, mesh_ok, domain_ok, accepted)
        linear.failure_reasons = tuple(reasons)
        return linear
