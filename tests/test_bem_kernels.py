import numpy as np
import pytest
from wave_resistance.bem.kernels import source_panel_potential,source_panel_velocity

TRI=np.array([[0.,0,0],[1,0,0],[0,1,0]])
def test_far_field_and_gradient():
 x=np.array([0.2,.2,10.]); p=source_panel_potential(x,TRI,24)
 assert p == pytest.approx(.5/(4*np.pi*np.linalg.norm(x-TRI.mean(0))),rel=.04)
 eps=1e-5; fd=np.array([(source_panel_potential(x+eps*np.eye(3)[i],TRI,24)-source_panel_potential(x-eps*np.eye(3)[i],TRI,24))/(2*eps) for i in range(3)])
 assert np.allclose(source_panel_velocity(x,TRI,24),fd,rtol=2e-4,atol=1e-8)
def test_self_integral_is_finite_and_refines():
 x=TRI.mean(0); a=source_panel_potential(x,TRI,12); b=source_panel_potential(x,TRI,24)
 assert np.isfinite(a) and a == pytest.approx(b,rel=2e-5)

def test_near_singular_limit_converges_to_self_value():
 x=TRI.mean(0)
 self_value=source_panel_potential(x,TRI,28)
 near_value=source_panel_potential(x+[0,0,1e-8],TRI,28)
 assert near_value == pytest.approx(self_value,rel=2e-4)
