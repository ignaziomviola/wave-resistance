"""Restricted fixed-waterline nonlinear free-surface continuation."""
from __future__ import annotations
import math
import numpy as np
from .bem.assembly import (collocation_normal_matrix, evaluate_velocity,
                           influence_matrices)
from .bem.diagnostics import ResidualHistory, validate_graph
from .bem.free_surface import least_squares_derivative, move_graph
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
        speed=froude_number*math.sqrt(self.physics.water.g_m_s2*hull.metadata.length_ref_m)
        uniform=np.array((-speed,0.,0.)); g=self.physics.water.g_m_s2
        history=ResidualHistory(); failure=[]; sigma=np.zeros(len(hmesh.faces)+len(fmesh.faces))
        algebraic=True; converged=False; last_update=float("inf")
        for lam in np.linspace(0,1,self.nonlinear.continuation_steps):
            step_converged=False
            for _ in range(self.nonlinear.max_iterations):
                fmesh=move_graph(fbase,eta,fbase.waterline_vertices)
                valid,reasons,slope=validate_graph(fmesh,self.nonlinear.max_slope)
                if not valid:
                    failure.extend(reasons); algebraic=False; break
                combined=combine_meshes(hmesh,fmesh); h=collocation_normal_matrix(combined,far_field_ratio=self.bem.far_field_ratio)
                rhs=-(combined.normals@uniform)
                solved=solve_dense(h,rhs,rtol=self.bem.linear_tolerance)
                history.bem.extend(solved.residual_history)
                if not solved.converged:
                    failure.append("singular or incompatible BEM system"); algebraic=False; break
                sigma=solved.solution
                fs_velocity=evaluate_velocity(combined,sigma,fmesh.centroids,far_field_ratio=self.bem.far_field_ratio)+uniform
                fs_phi_matrix,_=influence_matrices(combined,fmesh.centroids,fmesh.normals,far_field_ratio=self.bem.far_field_ratio)
                phi=fs_phi_matrix@sigma
                dx=least_squares_derivative(fmesh.centroids,axis=0,neighbours=self.free_surface.derivative_neighbours)
                linear_target=-(speed/g)*(dx@phi)
                exact_target=(np.sum(fs_velocity**2,axis=1)-speed**2)/(2*g)
                target=(1-lam)*linear_target+lam*exact_target
                # Outer and shared-waterline panels are relaxed toward zero by
                # their incident vertices, preserving the graph topology.
                updated=eta+self.nonlinear.relaxation*(target-eta)
                vertex_fixed=set(map(int,fbase.waterline_vertices))
                boundary=np.flatnonzero(np.any(np.isin(fbase.faces,list(vertex_fixed)),axis=1))
                updated[boundary]=0.0
                last_update=float(np.linalg.norm(updated-eta)/max(np.linalg.norm(updated),1e-10))
                normal_velocity=np.sum(fs_velocity*fmesh.normals,axis=1)
                dynamic=.5*(np.sum(fs_velocity**2,axis=1)-speed**2)-g*eta
                history.hull_impermeability.append(float(np.max(np.abs(rhs[:len(hmesh.faces)]-h[:len(hmesh.faces)]@sigma))/speed))
                history.free_surface_kinematic.append(float(np.linalg.norm(normal_velocity)/max(speed,1e-12)))
                history.free_surface_dynamic.append(float(np.linalg.norm(dynamic)/(g*max(np.linalg.norm(eta),1e-8))))
                history.nonlinear_update.append(last_update); history.waterline.append(float(np.max(np.abs(updated[boundary]))) if len(boundary) else 0.)
                history.geometry.append(slope)
                eta=updated
                if last_update<self.nonlinear.residual_tolerance:
                    step_converged=True; break
            if failure or not step_converged:
                if not failure: failure.append("divergent nonlinear continuation")
                break
        converged=not failure and step_converged
        fmesh=move_graph(fbase,eta,fbase.waterline_vertices); combined=combine_meshes(hmesh,fmesh)
        if algebraic and np.all(np.isfinite(sigma)):
            hull_velocity=evaluate_velocity(combined,sigma,hmesh.centroids,far_field_ratio=self.bem.far_field_ratio)+uniform
            smat,_=influence_matrices(combined,hmesh.centroids,hmesh.normals,far_field_ratio=self.bem.far_field_ratio)
            potential=-speed*hmesh.centroids[:,0]+smat@sigma
            _,pressure=pressure_resistance(hmesh,hull_velocity,self.physics.water.rho_kg_m3,speed)
        else:
            hull_velocity=np.full((len(hmesh.faces),3),np.nan); potential=np.full(len(hmesh.faces),np.nan); pressure=np.full(len(hmesh.faces),np.nan)
        return self._result(hull,froude_number,hmesh,hull_velocity,pressure,potential,sigma,history,
            fmesh=fmesh,eta=eta,algebraic=algebraic,fs_converged=converged,reasons=tuple(dict.fromkeys(failure)))
