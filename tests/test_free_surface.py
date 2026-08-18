import unittest
import numpy as np
from wave_resistance import wigley_hull
from wave_resistance.bem.diagnostics import validate_graph
from wave_resistance.bem.free_surface import (least_squares_derivative,
                                               move_graph, sponge_strength)
from wave_resistance.bem.mesh import free_surface_mesh

class FreeSurfaceDiscretisationTests(unittest.TestCase):
    def test_plane_wave_derivative_and_linear_dispersion(self):
        x=np.linspace(0,2*np.pi,80); points=np.column_stack((x,np.zeros_like(x),np.zeros_like(x)))
        derivative=least_squares_derivative(points,neighbours=5,upwind=False)
        k=.5; phi=np.sin(k*x)
        interior=slice(3,-3)
        np.testing.assert_allclose((derivative@phi)[interior],k*np.cos(k*x[interior]),rtol=.015,atol=2e-3)
        # g k = omega^2 and omega=kU for a steady deep-water wave.
        g=9.80665; speed=np.sqrt(g/k)
        self.assertAlmostEqual((k*speed)**2,g*k,places=12)

    def test_sponge_is_zero_in_interior_and_increases_outward(self):
        points=np.array([[.5,0,0],[2.4,0,0],[.5,.72,0]])
        value=sponge_strength(points,1.,upstream=.5,downstream=1.5,lateral=.75,
                              fraction=.3,maximum=2.)
        self.assertEqual(value[0],0.)
        self.assertGreater(value[1],0.); self.assertGreater(value[2],0.)

    def test_invalid_graph_slope_is_rejected(self):
        hull=wigley_hull(nx=11,nz=5); mesh=free_surface_mesh(hull,nx=5,ny_half=2,hull_nx=3)
        moved=move_graph(mesh,np.linspace(-1,1,len(mesh.faces)),mesh.waterline_vertices)
        valid,reasons,_=validate_graph(moved,.05)
        self.assertFalse(valid); self.assertIn("slope",reasons[0])

if __name__=="__main__": unittest.main()
