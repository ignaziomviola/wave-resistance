"""Public exact-body Rankine-panel solvers and result models."""
from __future__ import annotations
import json, math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np

from .geometry import OffsetHull
from .models import WaterProperties
from .bem.assembly import (collocation_matrices, collocation_normal_matrix,
                           evaluate_velocity, influence_matrices)
from .bem.diagnostics import ResidualHistory, validate_graph
from .bem.free_surface import least_squares_derivative, move_graph, sponge_strength
from .bem.linear_solvers import solve_dense
from .bem.mesh import (SurfaceMesh, combine_meshes, free_surface_mesh,
                       mirrored_double_body, offset_hull_mesh)
from .bem.resistance import (force_balance, momentum_flux_resistance,
                             pressure_resistance)

FORMULATION_VERSION="rankine-source-v1"

@dataclass(frozen=True)
class GeometrySettings:
    hull_nx: int=9
    hull_nz: int=5
    free_surface_nx: int=13
    free_surface_ny_half: int=4
    bow_refinement: float=1.0
    stern_refinement: float=1.0
    waterline_refinement: float=1.0
    def __post_init__(self):
        if self.hull_nx<3 or self.hull_nz<2: raise ValueError("hull_nx>=3 and hull_nz>=2 required")
        if self.free_surface_nx<5 or self.free_surface_ny_half<2: raise ValueError("invalid free-surface resolution")

@dataclass(frozen=True)
class PhysicsSettings:
    water: WaterProperties=field(default_factory=WaterProperties)
    fixed_sinkage_m: float=0.0
    fixed_trim_rad: float=0.0
    deep_water: bool=True
    def __post_init__(self):
        if not self.deep_water: raise ValueError("finite depth is unsupported in release 1")
        if self.fixed_sinkage_m!=0 or self.fixed_trim_rad!=0: raise ValueError("nonzero sinkage/trim input is unsupported")

@dataclass(frozen=True)
class BEMSettings:
    linear_tolerance: float=1e-8
    far_field_ratio: float=4.0
    condition_limit: float=1e13

@dataclass(frozen=True)
class FreeSurfaceSettings:
    upstream_lengths: float=0.5
    downstream_lengths: float=1.5
    lateral_lengths: float=0.75
    sponge_fraction: float=0.3
    sponge_strength: float=2.0
    derivative_neighbours: int=8
    def __post_init__(self):
        if min(self.upstream_lengths,self.downstream_lengths,self.lateral_lengths)<=0: raise ValueError("domain dimensions must be positive")

@dataclass(frozen=True)
class NonlinearSettings:
    continuation_steps: int=5
    max_iterations: int=20
    relaxation: float=0.15
    residual_tolerance: float=2e-3
    max_slope: float=0.6
    fixed_waterline_nonlinear: bool=True
    fail_on_nonconvergence: bool=True
    def __post_init__(self):
        if self.continuation_steps<2 or self.max_iterations<1: raise ValueError("invalid nonlinear iteration counts")
        if not 0<self.relaxation<=1: raise ValueError("relaxation must lie in (0,1]")
        if not self.fixed_waterline_nonlinear: raise ValueError("moving-waterline nonlinear mode is not supported in release 1")

@dataclass(frozen=True)
class ConvergenceSettings:
    force_relative_tolerance: float=0.02
    absolute_force_tolerance_N: float=1e-5
    require_mesh_study: bool=False
    require_domain_study: bool=False

@dataclass
class PotentialFlowResult:
    method_name: str
    formulation_version: str
    froude_number: float
    speed_m_s: float
    pressure_resistance_N: float
    far_field_resistance_N: float
    c_pressure: float
    c_far_field: float
    force_balance_discrepancy_N: float
    potential: np.ndarray
    surface_velocity: np.ndarray
    hull_pressure_Pa: np.ndarray
    free_surface_vertices: np.ndarray
    free_surface_faces: np.ndarray
    free_surface_elevation_m: np.ndarray
    hull_vertices: np.ndarray
    hull_faces: np.ndarray
    actual_wetted_area_m2: float
    reference_wetted_area_m2: float
    residual_history: Dict[str,np.ndarray]
    mesh_metadata: Dict[str,object]
    domain_metadata: Dict[str,object]
    wave_cuts: Dict[str,np.ndarray]
    algebraic_converged: bool
    free_surface_converged: bool
    force_balance_converged: bool
    mesh_converged: bool
    domain_converged: bool
    accepted: bool
    failure_reasons: Tuple[str,...]=()
    source_strengths: np.ndarray=field(default_factory=lambda:np.empty(0),repr=False)

    def to_json(self,path):
        scalar={k:v for k,v in vars(self).items() if isinstance(v,(str,float,int,bool,tuple))}
        scalar["failure_reasons"]=list(self.failure_reasons)
        Path(path).write_text(json.dumps(scalar,indent=2,allow_nan=True),encoding="utf-8")
    def to_npz(self,path):
        fields={"potential":self.potential,"surface_velocity":self.surface_velocity,
                "hull_pressure_Pa":self.hull_pressure_Pa,"free_surface_vertices":self.free_surface_vertices,
                "free_surface_faces":self.free_surface_faces,"free_surface_elevation_m":self.free_surface_elevation_m,
                "hull_vertices":self.hull_vertices,"hull_faces":self.hull_faces,"source_strengths":self.source_strengths}
        fields.update({"residual_"+k:v for k,v in self.residual_history.items()})
        np.savez_compressed(path,**fields)

class _BaseSolver:
    method_name="potential-flow"
    def __init__(self,geometry=None,physics=None,bem=None,free_surface=None,nonlinear=None,convergence=None):
        self.geometry=geometry or GeometrySettings(); self.physics=physics or PhysicsSettings()
        self.bem=bem or BEMSettings(); self.free_surface=free_surface or FreeSurfaceSettings()
        self.nonlinear=nonlinear or NonlinearSettings(); self.convergence=convergence or ConvergenceSettings()
    def _validate(self,hull,fn):
        if not isinstance(hull,OffsetHull): raise TypeError("hull must be an OffsetHull")
        if not math.isfinite(fn) or fn<=0: raise ValueError("Froude number must be finite and positive")
        if np.any(hull.half_breadths[1:-1,0]<=0): raise ValueError("invalid or disconnected waterline")
    def _meshes(self,hull):
        h=offset_hull_mesh(hull,self.geometry.hull_nx,self.geometry.hull_nz)
        f=free_surface_mesh(hull,nx=self.geometry.free_surface_nx,ny_half=self.geometry.free_surface_ny_half,
            upstream_lengths=self.free_surface.upstream_lengths,downstream_lengths=self.free_surface.downstream_lengths,
            lateral_lengths=self.free_surface.lateral_lengths,hull_nx=self.geometry.hull_nx)
        return h,f
    def _empty_fs(self): return np.empty((0,3)),np.empty((0,3),int),np.empty(0)
    def _result(self,hull,fn,hmesh,velocity,pressure,potential,strengths,history,*,
                fmesh=None,eta=None,algebraic=True,fs_converged=True,far=None,reasons=()):
        speed=fn*math.sqrt(self.physics.water.g_m_s2*hull.metadata.length_ref_m)
        rp=float(np.sum(pressure*hmesh.normals[:,0]*hmesh.areas)) if len(pressure) else float("nan")
        combined=hmesh if fmesh is None else combine_meshes(hmesh,fmesh)
        all_strengths=strengths
        rf=momentum_flux_resistance(combined,all_strengths,self.physics.water.rho_kg_m3,speed,
                                    depth=hull.metadata.length_ref_m) if far is None and np.all(np.isfinite(strengths)) else far
        discrepancy,force_ok=force_balance(rp,rf,self.convergence.force_relative_tolerance,
                                            self.convergence.absolute_force_tolerance_N)
        reasons=tuple(reasons)
        if not force_ok and "independent force balance failed" not in reasons:
            reasons=reasons+("independent force balance failed",)
        mesh_diag=hmesh.diagnostics(); mesh_ok=mesh_diag.valid
        mesh_converged=mesh_ok and not self.convergence.require_mesh_study
        domain_converged=not self.convergence.require_domain_study
        fatal=any(reason!="independent force balance failed" for reason in reasons)
        accepted=bool(algebraic and fs_converged and force_ok and mesh_converged and domain_converged and not reasons)
        if fatal:
            rp=rf=float("nan")
        ref=hull.metadata.wetted_area_m2; denom=.5*self.physics.water.rho_kg_m3*speed**2*ref
        if fmesh is None: fv,ff,fe=self._empty_fs()
        else: fv,ff,fe=fmesh.vertices,fmesh.faces,(eta if eta is not None else fmesh.centroids[:,2])
        wave_cuts={}
        if fmesh is not None:
            ytol=max(hull.beam_m,1e-8); mask=np.abs(fmesh.centroids[:,1])<ytol
            wave_cuts={"centreline_x_m":fmesh.centroids[mask,0],"centreline_eta_m":np.asarray(fe)[mask]}
        return PotentialFlowResult(self.method_name,FORMULATION_VERSION,float(fn),speed,rp,rf,
            rp/denom if accepted and np.isfinite(rp) else float("nan"),rf/denom if accepted and np.isfinite(rf) else float("nan"),discrepancy,
            potential,velocity,pressure,fv,ff,np.asarray(fe),hmesh.vertices,hmesh.faces,float(np.sum(hmesh.areas)),ref,
            history.as_dict(),{"hull":asdict(mesh_diag)},asdict(self.free_surface),wave_cuts,
            algebraic,fs_converged,force_ok,mesh_converged,domain_converged,accepted,tuple(reasons),all_strengths)

class DoubleBodyPotentialFlowSolver(_BaseSolver):
    method_name="double-body-rankine-bem"
    def solve(self,hull:OffsetHull,froude_number:float)->PotentialFlowResult:
        self._validate(hull,froude_number); hmesh=offset_hull_mesh(hull,self.geometry.hull_nx,self.geometry.hull_nz)
        closed,nphysical=mirrored_double_body(hmesh); speed=froude_number*math.sqrt(self.physics.water.g_m_s2*hull.metadata.length_ref_m)
        uniform=np.array((-speed,0.,0.)); matrix=collocation_normal_matrix(closed,far_field_ratio=self.bem.far_field_ratio)
        rhs=-(closed.normals@uniform); flux=float(np.dot(rhs,closed.areas)); rhs-=flux/np.sum(closed.areas)
        solved=solve_dense(matrix,rhs,rtol=self.bem.linear_tolerance); sigma=solved.solution
        perturb=evaluate_velocity(closed,sigma,closed.centroids[:nphysical],far_field_ratio=self.bem.far_field_ratio)
        velocity=perturb+uniform; potential_matrix,_=influence_matrices(closed,closed.centroids[:nphysical],closed.normals[:nphysical],far_field_ratio=self.bem.far_field_ratio)
        potential=-speed*closed.centroids[:nphysical,0]+potential_matrix@sigma
        rp,pressure=pressure_resistance(hmesh,velocity,self.physics.water.rho_kg_m3,speed)
        hist=ResidualHistory(); hist.bem.extend(solved.residual_history); hist.hull_impermeability.append(float(np.max(np.abs(np.sum(velocity*hmesh.normals,axis=1)))/speed)); hist.waterline.append(0.)
        # D'Alembert provides the independent double-body force check.
        return self._result(hull,froude_number,hmesh,velocity,pressure,potential,sigma[:nphysical],hist,
                            far=0.,algebraic=solved.converged,reasons=(() if solved.converged else ("singular or incompatible BEM system",)))

class LinearPotentialFlowSolver(_BaseSolver):
    method_name="linear-exact-body-rankine-bem"
    def solve(self,hull:OffsetHull,froude_number:float)->PotentialFlowResult:
        self._validate(hull,froude_number); hmesh,fmesh=self._meshes(hull); combined=combine_meshes(hmesh,fmesh)
        nh=len(hmesh.faces); nf=len(fmesh.faces); speed=froude_number*math.sqrt(self.physics.water.g_m_s2*hull.metadata.length_ref_m); g=self.physics.water.g_m_s2
        s,h=collocation_matrices(combined,far_field_ratio=self.bem.far_field_ratio); dx=least_squares_derivative(fmesh.centroids,axis=0,neighbours=self.free_surface.derivative_neighbours)
        sponge=sponge_strength(fmesh.centroids,hull.metadata.length_ref_m,upstream=self.free_surface.upstream_lengths,
          downstream=self.free_surface.downstream_lengths,lateral=self.free_surface.lateral_lengths,
          fraction=self.free_surface.sponge_fraction,maximum=self.free_surface.sponge_strength)
        a=np.zeros((nh+2*nf,nh+nf+nf)); b=np.zeros(nh+2*nf)
        uniform=np.array((-speed,0.,0.)); a[:nh,:nh+nf]=h[:nh]; b[:nh]=-(hmesh.normals@uniform)
        a[nh:nh+nf,:nh+nf]=h[nh:]; a[nh:nh+nf,nh+nf:]=speed*dx
        a[nh+nf:,:nh+nf]=(speed/g)*dx@s[nh:]; a[nh+nf:,nh+nf:]=np.eye(nf)+np.diag(sponge)
        solved=solve_dense(a,b,rtol=self.bem.linear_tolerance); sigma=solved.solution[:nh+nf]; eta=solved.solution[nh+nf:]
        perturb=evaluate_velocity(combined,sigma,hmesh.centroids,far_field_ratio=self.bem.far_field_ratio); velocity=perturb+uniform
        potential=-speed*hmesh.centroids[:,0]+s[:nh]@sigma; rp,pressure=pressure_resistance(hmesh,velocity,self.physics.water.rho_kg_m3,speed)
        hist=ResidualHistory(); hist.bem.extend(solved.residual_history); hist.hull_impermeability.append(float(np.max(np.abs(np.sum(velocity*hmesh.normals,axis=1)))/speed)); hist.free_surface_kinematic.append(float(np.linalg.norm(h[nh:]@sigma+speed*dx@eta)/max(speed,1e-12))); hist.free_surface_dynamic.append(float(np.linalg.norm(eta+(speed/g)*dx@(s[nh:]@sigma))/max(np.linalg.norm(eta),1e-12))); hist.waterline.append(0.)
        moved=move_graph(fmesh,eta,fmesh.waterline_vertices)
        reasons=() if solved.converged else ("singular or incompatible BEM system",)
        return self._result(hull,froude_number,hmesh,velocity,pressure,potential,sigma,hist,fmesh=moved,eta=eta,algebraic=solved.converged,fs_converged=solved.converged,reasons=reasons)
