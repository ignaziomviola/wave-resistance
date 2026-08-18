import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from wave_resistance import (DoubleBodyPotentialFlowSolver, GeometrySettings,
    LinearPotentialFlowSolver, NonlinearPotentialFlowSolver, NonlinearSettings,
    wigley_hull)

GEOMETRY=GeometrySettings(hull_nx=3,hull_nz=2,free_surface_nx=5,free_surface_ny_half=2)

class PotentialFlowSolverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.hull=wigley_hull(nx=11,nz=5)

    def test_double_body_uniform_flow_has_zero_drag_and_flux(self):
        result=DoubleBodyPotentialFlowSolver(geometry=GEOMETRY).solve(self.hull,.3)
        self.assertTrue(result.algebraic_converged)
        self.assertTrue(result.accepted)
        self.assertLess(abs(result.pressure_resistance_N),1e-10)
        self.assertLess(result.residual_history["hull_impermeability"][-1],1e-10)
        self.assertEqual(result.formulation_version,"rankine-source-v1")

    def test_linear_solution_has_separate_physical_acceptance_gate_and_exports(self):
        result=LinearPotentialFlowSolver(geometry=GEOMETRY).solve(self.hull,.3)
        self.assertTrue(result.algebraic_converged)
        self.assertFalse(result.force_balance_converged)
        self.assertFalse(result.accepted)
        self.assertTrue(np.isnan(result.far_field_resistance_N))
        self.assertTrue(np.isnan(result.force_balance_discrepancy_N))
        self.assertTrue(np.isnan(result.c_pressure))
        self.assertIn("independent force balance failed",result.failure_reasons)
        self.assertEqual(result.free_surface_elevation_m.shape,(len(result.free_surface_faces),))
        self.assertGreater(len(result.wave_cuts["centreline_x_m"]),0)
        self.assertEqual(result.wave_cuts["centreline_x_m"].shape,
                         result.wave_cuts["centreline_eta_m"].shape)
        with tempfile.TemporaryDirectory() as directory:
            summary=Path(directory)/"summary.json"; fields=Path(directory)/"fields.npz"
            result.to_json(summary); result.to_npz(fields)
            self.assertEqual(json.loads(summary.read_text())["accepted"],False)
            self.assertIn("potential",np.load(fields))

    def test_linear_free_surface_residuals_follow_normal_convention(self):
        result=LinearPotentialFlowSolver(geometry=GEOMETRY).solve(self.hull,.3)
        self.assertLess(result.residual_history["free_surface_kinematic"][-1],1e-8)
        self.assertLess(result.residual_history["free_surface_dynamic"][-1],1e-8)

    def test_nonlinear_failure_never_returns_coefficient(self):
        settings=NonlinearSettings(continuation_steps=2,max_iterations=1,
                                   relaxation=.1,residual_tolerance=1e-12)
        result=NonlinearPotentialFlowSolver(geometry=GEOMETRY,nonlinear=settings).solve(self.hull,.3)
        self.assertFalse(result.free_surface_converged)
        self.assertFalse(result.accepted)
        self.assertTrue(np.isnan(result.pressure_resistance_N))
        self.assertTrue(np.isnan(result.c_pressure))
        self.assertTrue(any(reason in result.failure_reasons for reason in
                            ("divergent nonlinear continuation",
                             "singular or incompatible BEM system")))

    def test_free_surface_velocity_uses_fluid_side_jump(self):
        from wave_resistance.bem.assembly import (collocation_normal_matrix,
                                                   evaluate_velocity)
        from wave_resistance.bem.mesh import combine_meshes
        solver=NonlinearPotentialFlowSolver(geometry=GEOMETRY)
        hull_mesh,free_mesh=solver._meshes(self.hull)
        combined=combine_meshes(hull_mesh,free_mesh)
        strengths=np.linspace(-.2,.3,len(combined.faces))
        matrix=collocation_normal_matrix(combined)
        expected=matrix[len(hull_mesh.faces):]@strengths
        velocity=evaluate_velocity(combined,strengths,free_mesh.centroids)
        velocity+=strengths[len(hull_mesh.faces):,None]*free_mesh.normals
        np.testing.assert_allclose(np.sum(velocity*free_mesh.normals,axis=1),
                                   expected,rtol=2e-12,atol=2e-12)

if __name__=="__main__": unittest.main()
