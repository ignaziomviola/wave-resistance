"""Restricted fixed-waterline nonlinear free-surface continuation."""
from __future__ import annotations
import math
import numpy as np
from .bem.assembly import (collocation_normal_matrix, evaluate_velocity,
                           influence_matrices)
from .bem.diagnostics import ResidualHistory, validate_graph
from .bem.free_surface import (boundary_vertices, least_squares_derivative,
                               project_panel_elevation, sponge_strength)
from .bem.linear_solvers import solve_dense
from .bem.mesh import combine_meshes
from .bem.resistance import pressure_resistance
from .potential_flow import LinearPotentialFlowSolver, _BaseSolver


class NonlinearPotentialFlowSolver(_BaseSolver):
    """Exact-condition graph solver with fixed design-waterline approximation."""
    method_name="nonlinear-exact-body-rankine-bem-fixed-waterline"

    def solve(self,hull,froude_number):
        self._validate(hull,froude_number)
        linear=LinearPotentialFlowSolver(self.geometry,self.physics,self.bem,self.free_surface,
                                         self.nonlinear,self.convergence).solve(hull,froude_number)
        if not linear.algebraic_converged:
            linear.method_name=self.method_name
            linear.failure_reasons=("linear initial condition failed",)
            linear.accepted=False
            linear.pressure_resistance_N=linear.far_field_resistance_N=float("nan")
            return linear
        hmesh,fbase=self._meshes(hull); fmesh=fbase
        eta=np.array(linear.free_surface_elevation_m,copy=True)
        if eta.shape!=(len(fmesh.faces),): eta=np.zeros(len(fmesh.faces))
        fixed_vertices=boundary_vertices(fbase)
        vertex_eta=project_panel_elevation(fbase,eta,fixed_vertices)
        speed=froude_number*math.sqrt(self.physics.water.g_m_s2*hull.metadata.length_ref_m)
        uniform=np.array((-speed,0.,0.)); g=self.physics.water.g_m_s2
        history=ResidualHistory(); failure=[]; sigma=np.zeros(len(hmesh.faces)+len(fmesh.faces))
        algebraic=True; converged=False; last_update=float("inf")
        sponge=sponge_strength(fbase.centroids,hull.metadata.length_ref_m,
            upstream=self.free_surface.upstream_lengths,
            downstream=self.free_surface.downstream_lengths,
            lateral=self.free_surface.lateral_lengths,
            fraction=self.free_surface.sponge_fraction,
            maximum=self.free_surface.sponge_strength)
        wave_scale=max(speed**2/g,1e-12)
        for lam in np.linspace(0,1,self.nonlinear.continuation_steps):
            step_converged=False
            for _ in range(self.nonlinear.max_iterations):
                vertices=np.array(fbase.vertices,copy=True); vertices[:,2]=vertex_eta
                fmesh=fbase.with_vertices(vertices,name="nonlinear_free_surface")
                valid,reasons,slope=validate_graph(fmesh,self.nonlinear.max_slope)
                if not valid:
                    failure.extend(reasons); break
                combined=combine_meshes(hmesh,fmesh); h=collocation_normal_matrix(combined,far_field_ratio=self.bem.far_field_ratio)
                rhs=-(combined.normals@uniform)
                solved=solve_dense(h,rhs,rtol=self.bem.linear_tolerance,
                                   condition_limit=self.bem.condition_limit)
                history.bem.extend(solved.residual_history)
                history.matrix_condition.append(solved.condition_number)
                if not solved.converged:
                    failure.append("singular or incompatible BEM system"); algebraic=False; break
                sigma=solved.solution
                fs_velocity=evaluate_velocity(combined,sigma,fmesh.centroids,far_field_ratio=self.bem.far_field_ratio)+uniform
                # ``evaluate_velocity`` returns the +normal boundary limit,
                # appropriate for body panels.  The fluid lies on the
                # -normal side of the upper free surface, whose source-sheet
                # jump differs by +sigma*n from that default limit.
                fs_velocity += sigma[len(hmesh.faces):,None]*fmesh.normals
                fs_phi_matrix,_=influence_matrices(combined,fmesh.centroids,fmesh.normals,far_field_ratio=self.bem.far_field_ratio)
                phi=fs_phi_matrix@sigma
                dx=least_squares_derivative(fmesh.centroids,axis=0,neighbours=self.free_surface.derivative_neighbours)
                linear_target=-(speed/g)*(dx@phi)
                exact_target=(np.sum(fs_velocity**2,axis=1)-speed**2)/(2*g)
                target=((1-lam)*linear_target+lam*exact_target)/(1+sponge)
                target_vertices=project_panel_elevation(fbase,target,fixed_vertices)
                projected_target=np.mean(target_vertices[fbase.faces],axis=1)
                relaxation=self.nonlinear.relaxation
                while True:
                    updated_vertices=vertex_eta+relaxation*(target_vertices-vertex_eta)
                    candidate_vertices=np.array(fbase.vertices,copy=True)
                    candidate_vertices[:,2]=updated_vertices
                    candidate=fbase.with_vertices(candidate_vertices,name="nonlinear_free_surface")
                    candidate_valid,candidate_reasons,candidate_slope=validate_graph(
                        candidate,self.nonlinear.max_slope)
                    if candidate_valid or relaxation<=self.nonlinear.relaxation/128:
                        break
                    relaxation*=.5
                if not candidate_valid:
                    failure.extend(candidate_reasons); break
                updated_eta=np.mean(updated_vertices[fbase.faces],axis=1)
                current_eta=fmesh.centroids[:,2]
                target_residual=float(np.sqrt(np.mean((projected_target-current_eta)**2))/wave_scale)
                last_update=float(np.sqrt(np.mean((updated_eta-current_eta)**2))/wave_scale)
                normal_velocity=np.sum(fs_velocity*fmesh.normals,axis=1)
                # Residual of the discretised Bernoulli condition, including
                # the explicit artificial damping in the sponge region.
                dynamic=g*(projected_target-current_eta) if lam==1 else g*(target-current_eta)
                history.hull_impermeability.append(float(np.max(np.abs(rhs[:len(hmesh.faces)]-h[:len(hmesh.faces)]@sigma))/speed))
                kinematic_residual=float(np.sqrt(np.mean(normal_velocity**2))/max(speed,1e-12))
                dynamic_residual=float(np.sqrt(np.mean(dynamic**2))/max(speed**2,1e-12))
                history.free_surface_kinematic.append(kinematic_residual)
                history.free_surface_dynamic.append(dynamic_residual)
                history.nonlinear_update.append(last_update); history.waterline.append(0.)
                history.geometry.append(candidate_slope)
                vertex_eta=updated_vertices; eta=updated_eta
                waterline_error=float(np.max(np.abs(vertex_eta[fbase.waterline_vertices]))) if len(fbase.waterline_vertices) else 0.
                history.waterline[-1]=waterline_error
                residual_ok=(target_residual<self.nonlinear.residual_tolerance and
                             kinematic_residual<self.nonlinear.residual_tolerance)
                if lam==1:
                    residual_ok=residual_ok and dynamic_residual<self.nonlinear.residual_tolerance
                if residual_ok:
                    step_converged=True; break
            if failure or not step_converged:
                if not failure: failure.append("divergent nonlinear continuation")
                break
        converged=not failure and step_converged
        vertices=np.array(fbase.vertices,copy=True); vertices[:,2]=vertex_eta
        fmesh=fbase.with_vertices(vertices,name="nonlinear_free_surface")
        eta=fmesh.centroids[:,2]; combined=combine_meshes(hmesh,fmesh)
        if algebraic and np.all(np.isfinite(sigma)):
            hull_velocity=evaluate_velocity(combined,sigma,hmesh.centroids,far_field_ratio=self.bem.far_field_ratio)+uniform
            smat,_=influence_matrices(combined,hmesh.centroids,hmesh.normals,far_field_ratio=self.bem.far_field_ratio)
            potential=-speed*hmesh.centroids[:,0]+smat@sigma
            _,pressure=pressure_resistance(hmesh,hull_velocity,self.physics.water.rho_kg_m3,speed)
        else:
            hull_velocity=np.full((len(hmesh.faces),3),np.nan); potential=np.full(len(hmesh.faces),np.nan); pressure=np.full(len(hmesh.faces),np.nan)
        return self._result(hull,froude_number,hmesh,hull_velocity,pressure,potential,sigma,history,
            fmesh=fmesh,eta=eta,algebraic=algebraic,fs_converged=converged,reasons=tuple(dict.fromkeys(failure)))
