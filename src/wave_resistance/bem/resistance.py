import numpy as np

def pressure_force(mesh, pressure):
    return -np.sum(pressure[:,None]*mesh.normals*mesh.areas[:,None],axis=0)

def downstream_energy_flux(x, eta, speed, rho, gravity):
    """Linear wave-energy flux through a transverse downstream cut (watts)."""
    if len(eta)<2: return 0.
    integrate = getattr(np, "trapezoid", None)
    if integrate is None:  # NumPy 1.x compatibility
        integrate = np.trapz
    return float(rho*gravity*speed*integrate(eta*eta,x=x)/2)

def mixed_force_balance(a,b,rtol,atol): return abs(a-b) <= max(rtol*max(abs(a),abs(b)),atol)
