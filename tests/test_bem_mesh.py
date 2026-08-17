import numpy as np
import pytest
from wave_resistance import wigley_hull
from wave_resistance.bem import SurfaceMesh,offset_hull_mesh,rectangular_free_surface
from wave_resistance.bem.assembly import reconstruct_surface_gradient

def test_triangle_geometry_and_validation():
 m=SurfaceMesh([[0,0,0],[1,0,0],[0,1,0]],[[0,1,2]])
 assert m.areas[0] == pytest.approx(.5); assert np.allclose(m.normals[0],[0,0,1])
 with pytest.raises(ValueError,match="degenerate"): SurfaceMesh([[0,0,0],[1,0,0],[2,0,0]],[[0,1,2]])

def test_offset_adapter_and_free_surface():
 h=wigley_hull(nx=5,nz=4); body=offset_hull_mesh(h); fs=rectangular_free_surface(h,5,5)
 assert len(body.faces)>0 and np.all(body.areas>0); assert np.allclose(fs.vertices[:,2],0)
 assert body.diagnostics()["minimum_angle_deg"]>0
 # Every analytic waterline station is represented on both sides.
 for x,b in zip(h.x_m,h.waterline_half_breadth_m):
  assert np.any(np.all(np.isclose(fs.vertices,[x,b,0.],atol=1e-12),axis=1))
  assert np.any(np.all(np.isclose(fs.vertices,[x,-b,0.],atol=1e-12),axis=1))

def test_surface_gradient_reproduces_linear_field():
 vertices=np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0]],float)
 mesh=SurfaceMesh(vertices,[[0,1,2],[1,3,2]])
 values=2*mesh.centroids[:,0]-3*mesh.centroids[:,1]
 # Two faces alone do not span a 2-D stencil; tile a structured 2x2 patch.
 vertices=np.array([(i,j,0.) for i in range(3) for j in range(3)])
 faces=[]
 for i in range(2):
  for j in range(2):
   a=3*i+j; faces.extend(((a,a+3,a+4),(a,a+4,a+1)))
 mesh=SurfaceMesh(vertices,faces)
 values=2*mesh.centroids[:,0]-3*mesh.centroids[:,1]
 gradient=reconstruct_surface_gradient(mesh,values,neighbour_depth=3)
 assert np.allclose(gradient,[2,-3,0],atol=1e-12)
