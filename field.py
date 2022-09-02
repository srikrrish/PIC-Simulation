import numpy as np
from initialize import L

def field(rho):
    rhoHat = np.fft.fft(rho)
    phiHat, EHat = fieldInFourier(rhoHat)
    phi = np.real(np.fft.ifft(phiHat))
    E = np.real(np.fft.ifft(EHat))
    return phi, E


def fieldInFourier(rhoHat):
    Ka = np.arange(1, rhoHat.size // 2)
    Kb = Ka[::-1]
    K = np.append(np.append(Ka, [rhoHat.size // 2]), - Kb)
    phiHat = np.append([0], rhoHat[1:] * (L / (2 * np.pi * K)) ** 2)
    EHat = np.append([0], rhoHat[1:] * L / (2j * np.pi * K))
    EHat[rhoHat.size // 2] = 0
    return phiHat, EHat