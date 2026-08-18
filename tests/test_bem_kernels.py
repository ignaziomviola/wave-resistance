import math
import unittest
import numpy as np

from wave_resistance.bem.kernels import (source_panel_potential,
                                         source_panel_velocity, solid_angle)

class PanelKernelTests(unittest.TestCase):
    def setUp(self):
        self.triangle=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])

    def test_self_potential_matches_independent_tensor_quadrature(self):
        point=np.mean(self.triangle,axis=0)
        # Independent high-order Duffy integration split about the centroid.
        nodes,weights=np.polynomial.legendre.leggauss(80); v=(nodes+1)/2; w=weights/2
        reference=0.
        for b,c in ((self.triangle[0],self.triangle[1]),
                    (self.triangle[1],self.triangle[2]),
                    (self.triangle[2],self.triangle[0])):
            direction=(1-v[:,None])*(b-point)+v[:,None]*(c-point)
            reference+=np.linalg.norm(np.cross(b-point,c-point))*np.dot(w,1/np.linalg.norm(direction,axis=1))/(4*np.pi)
        self.assertAlmostEqual(source_panel_potential(point,self.triangle),reference,places=11)

    def test_velocity_is_potential_gradient_off_panel(self):
        point=np.array([.2,.25,.4]); step=2e-5; finite=np.empty(3)
        for i in range(3):
            delta=np.zeros(3); delta[i]=step
            finite[i]=(source_panel_potential(point+delta,self.triangle,rtol=1e-10)-
                       source_panel_potential(point-delta,self.triangle,rtol=1e-10))/(2*step)
        np.testing.assert_allclose(source_panel_velocity(point,self.triangle),finite,rtol=2e-5,atol=2e-7)

    def test_jump_and_solid_angle_limits(self):
        centroid=np.mean(self.triangle,axis=0)
        np.testing.assert_allclose(source_panel_velocity(centroid,self.triangle)[2],-.5,atol=1e-13)
        self.assertAlmostEqual(solid_angle(centroid,self.triangle),2*math.pi,places=12)
        above=solid_angle(centroid+np.array([0,0,1e-7]),self.triangle)
        self.assertAlmostEqual(above,2*math.pi,places=5)

    def test_far_panel_converges_to_point_source(self):
        point=np.array([30.,-20.,40.]); area=.5; centroid=np.mean(self.triangle,axis=0)
        expected=area/(4*np.pi*np.linalg.norm(point-centroid))
        self.assertLess(abs(source_panel_potential(point,self.triangle)-expected)/expected,2e-4)

if __name__=="__main__": unittest.main()
