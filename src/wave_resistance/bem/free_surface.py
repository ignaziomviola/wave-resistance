"""Free-surface residuals for the public upward-positive ``z`` convention."""
import numpy as np

def linear_residual(phi, eta, dphi_dx, deta_dx, speed, gravity):
    """Return kinematic ``U eta_x-phi_z`` and dynamic ``U phi_x+g eta``."""
    return speed*deta_dx-phi[:,2], speed*dphi_dx+gravity*eta

def nonlinear_residual(velocity, eta, eta_x, eta_y, speed, gravity):
    """Exact graph residuals for perturbation velocity and upward ``z``."""
    total=velocity+np.array([speed,0.,0.])
    kinematic=total[:,2]-total[:,0]*eta_x-total[:,1]*eta_y
    dynamic=0.5*(np.sum(total*total,axis=1)-speed**2)+gravity*eta
    return kinematic,dynamic

def sponge_strength(x,y,bounds,fraction=.2):
    xmin,xmax,ymax=bounds; sx=np.clip((x-(xmax-fraction*(xmax-xmin)))/(fraction*(xmax-xmin)),0,1)**2
    sy=np.clip((np.abs(y)-(1-fraction)*ymax)/(fraction*ymax),0,1)**2
    return np.maximum(sx,sy)
