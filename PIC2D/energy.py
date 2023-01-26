from initialize import Q, QM, dx, L
import numpy as np

def kinetic(vp, wp=1):
    return np.sum(Q * wp * np.sum(vp ** 2, axis=0) * 0.5 / QM)

def potential(rho, phi):
    print(np.sum(rho*phi))
    return np.sum(rho * phi * dx[0] * dx[1] / 2)

def specPotential(rhoHat, phiHat):
    return np.real(np.sum(rhoHat * np.conjugate(phiHat) / (2 * L[0] * L[1])))