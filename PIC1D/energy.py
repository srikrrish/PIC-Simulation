from initialize import Q, QM, dx, L
import numpy as np

def kinetic(vp, wp=1):
    return sum(Q * wp * vp ** 2 * 0.5 / QM)

def potential(rho, phi):
    return sum(rho * phi * dx / 2)

def specPotential(rhoHat, phiHat):
    return sum(rhoHat * np.conjugate(phiHat) / (2 * L))
