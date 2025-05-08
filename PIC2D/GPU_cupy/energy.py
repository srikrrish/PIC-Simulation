from initialize import QM
import numpy as np
import cupy as cp

def kinetic(vp, Q, wp=1):
    return cp.sum(Q * wp * cp.sum(vp ** 2, axis=0) * 0.5 / QM)

def potential(rho, phi, dx):
    #print(np.sum(rho*phi))
    return cp.sum(rho * phi) * dx[0] * dx[1] * 0.5

def specPotential(rhoHat, phiHat, L):
    return cp.real(cp.sum(rhoHat * cp.conjugate(phiHat) / (2 * L[0] * L[1])))

def energypotx(Eg, dx):
    return cp.sum(Eg[0] ** 2) * dx[0] * dx[1]

def energypot(Eg, dx):
    return cp.sum((Eg[0] ** 2) + (Eg[1] ** 2)) * dx[0] * dx[1] * 0.5
