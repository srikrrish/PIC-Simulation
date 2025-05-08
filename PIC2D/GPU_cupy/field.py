import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from initialize import NG, DT
import figures



def field(rho, L):
    rhoHat = cp.fft.fft2(rho)
    phiHat, EHat = fieldInFourier(rhoHat, L)
    phi = cp.real(cp.fft.ifft2(phiHat))
    E = cp.real(cp.array([cp.fft.ifft2(EHat[0]), cp.fft.ifft2(EHat[1])]))
    return phi, E


def fieldInFourier(rhoHat, L):
    Ja = cp.arange(rhoHat.shape[0] // 2)
    Jb = Ja[:0:-1]
    J = cp.append(cp.append(Ja, [-rhoHat.shape[0] // 2]), - Jb)
    Ka = cp.arange(rhoHat.shape[1] // 2)
    Kb = Ka[:0:-1]
    K = cp.append(cp.append(Ka, [-rhoHat.shape[1] // 2]), - Kb)
    J = cp.transpose(cp.expand_dims(J, 0).repeat(rhoHat.shape[1], axis=0)) * 2 * cp.pi / L[0]
    K = cp.expand_dims(K, 0).repeat(rhoHat.shape[0], axis=0) * 2 * cp.pi / L[1]
    absolute = J ** 2 + K ** 2
    absolute[0,0] = 1
    phiHat = rhoHat / absolute
    phiHat[0,0] = 0
    E0 = phiHat * -1j * J
    E1 = phiHat * -1J * K
    EHat = cp.array([E0, E1])
    return phiHat, EHat


def fieldAmpere(Jb, rhon, L):
    rhonHat = cp.fft.fft2(rhon)
    JHat = cp.array([cp.fft.fft2(Jb[0]), cp.fft.fft2(Jb[1])])
    phiHat, EHat, rhonp1Hat = fieldInFourierCurrent(JHat, rhonHat, L)
    phi = cp.real(cp.fft.ifft2(phiHat))
    rhonp1 = cp.real(cp.fft.ifft2(rhonp1Hat))
    E = cp.real(cp.array([cp.fft.ifft2(EHat[0]), cp.fft.ifft2(EHat[1])]))
    return phi, E, rhonp1

def fieldInFourierCurrent(JHat, rhonHat, L):
    Ja = cp.arange(rhonHat.shape[0] // 2)
    Jb = Ja[:0:-1]
    J = cp.append(cp.append(Ja, [-rhonHat.shape[0] // 2]), - Jb)
    Ka = cp.arange(rhonHat.shape[1] // 2)
    Kb = Ka[:0:-1]
    K = cp.append(cp.append(Ka, [-rhonHat.shape[1] // 2]), - Kb)
    J = cp.transpose(cp.expand_dims(J, 0).repeat(rhonHat.shape[1], axis=0)) * 2 * cp.pi / L[0]
    K = cp.expand_dims(K, 0).repeat(rhonHat.shape[0], axis=0) * 2 * cp.pi / L[1]
    absolute = J ** 2 + K ** 2
    absolute[0,0] = 1
    rhonp1Hat = rhonHat + DT * (-1j * J * JHat[0,:,:] + -1j * K * JHat[1,:,:])
    phiHat = rhonp1Hat / absolute
    phiHat[0,0] = 0
    E0 = phiHat * -1j * J
    E1 = phiHat * -1j * K
    EHat = cp.array([E0, E1])
    return phiHat, EHat, rhonp1Hat
