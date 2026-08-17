import numpy as np
from wave_resistance import *

def test_result_exports_and_failure_gates(tmp_path):
 h=wigley_hull(nx=4,nz=3)
 r=NonlinearPotentialFlowSolver(h,geometry=GeometrySettings(5,5),bem=BEMSettings(4)).solve()
 assert r.method_name.startswith("nonlinear") and len(r.free_surface_faces)>0
 assert r.status.accepted == (r.status.algebraic_converged and r.status.free_surface_converged and r.status.force_balance_converged and r.status.mesh_converged and r.status.domain_converged)
 r.to_json(tmp_path/"a.json"); r.to_npz(tmp_path/"a.npz"); assert (tmp_path/"a.json").exists()
