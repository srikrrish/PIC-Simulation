from initialize import QM
import numpy as np

def kinetic(vp, Q, wp=1):
    return np.sum(Q * wp * np.sum(vp ** 2, axis=0) * 0.5 / QM)

def potential(rho, phi, dx):
    #print(np.sum(rho*phi))
    return np.sum(rho * phi * dx[0] * dx[1] / 2)

def specPotential(rhoHat, phiHat, L):
    return np.real(np.sum(rhoHat * np.conjugate(phiHat) / (2 * L[0] * L[1])))

def energypotx(Eg, dx):
    return np.sum(Eg[0] ** 2) * dx[0] * dx[1]

def energypot(Eg, dx):
    return np.sum((Eg[0] ** 2) + (Eg[1] ** 2)) * dx[0] * dx[1] * 0.5
