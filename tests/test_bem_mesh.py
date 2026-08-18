import unittest
import numpy as np
from wave_resistance import wigley_hull
from wave_resistance.bem.mesh import free_surface_mesh, offset_hull_mesh

class SurfaceMeshTests(unittest.TestCase):
    def test_wigley_orientation_topology_and_waterline_conformity(self):
        hull=wigley_hull(nx=21,nz=9)
        mesh=offset_hull_mesh(hull,5,3)
        self.assertTrue(mesh.diagnostics().valid)
        starboard=mesh.centroids[:,1]>1e-12; port=mesh.centroids[:,1]<-1e-12
        self.assertTrue(np.all(mesh.normals[starboard,1]>0))
        self.assertTrue(np.all(mesh.normals[port,1]<0))
        free=free_surface_mesh(hull,nx=5,ny_half=2,hull_nx=5)
        waterline=mesh.vertices[mesh.waterline_vertices]
        error=max(np.min(np.linalg.norm(free.vertices-waterline_point,axis=1)) for waterline_point in waterline)
        self.assertLess(error,1e-12)
        self.assertTrue(np.all(free.normals[:,2]<0))
        self.assertTrue(free.diagnostics().valid)

    def test_duplicate_and_degenerate_faces_are_rejected(self):
        from wave_resistance.bem.mesh import SurfaceMesh
        vertices=np.array([[0,0,0],[1,0,0],[0,1,0]],float)
        with self.assertRaisesRegex(ValueError,"duplicate"):
            SurfaceMesh.from_arrays(vertices,[[0,1,2],[2,1,0]])
        with self.assertRaisesRegex(ValueError,"degenerate"):
            SurfaceMesh.from_arrays(vertices,[[0,1,1]])

    def test_inconsistent_shared_edge_orientation_is_reported(self):
        from wave_resistance.bem.mesh import SurfaceMesh
        vertices=np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0]],float)
        mesh=SurfaceMesh.from_arrays(vertices,[[0,1,2],[1,2,3]])
        diagnostics=mesh.diagnostics()
        self.assertFalse(diagnostics.orientation_consistent)
        self.assertFalse(diagnostics.valid)

    def test_endpoint_refinement_is_active_and_waterline_remains_conforming(self):
        hull=wigley_hull(nx=21,nz=9)
        uniform=offset_hull_mesh(hull,7,5)
        refined=offset_hull_mesh(hull,7,5,bow_refinement=2.,
                                 stern_refinement=2.,waterline_refinement=2.)
        uniform_x=np.unique(uniform.vertices[:,0]); refined_x=np.unique(refined.vertices[:,0])
        uniform_z=np.unique(uniform.vertices[:,2]); refined_z=np.unique(refined.vertices[:,2])
        self.assertLess(refined_x[1]-refined_x[0],uniform_x[1]-uniform_x[0])
        self.assertLess(refined_x[-1]-refined_x[-2],uniform_x[-1]-uniform_x[-2])
        self.assertLess(refined_z[1]-refined_z[0],uniform_z[1]-uniform_z[0])
        free=free_surface_mesh(hull,nx=7,ny_half=3,hull_nx=7,
                              bow_refinement=2.,stern_refinement=2.)
        waterline=refined.vertices[refined.waterline_vertices]
        error=max(np.min(np.linalg.norm(free.vertices-point,axis=1)) for point in waterline)
        self.assertLess(error,1e-12)

if __name__=="__main__": unittest.main()
