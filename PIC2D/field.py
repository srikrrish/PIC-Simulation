import numpy as np
import matplotlib.pyplot as plt
from initialize import NG, DT
import figures



def field(rho, L):
    rhoHat = np.fft.fft2(rho)
    phiHat, EHat = fieldInFourier(rhoHat, L)
    phi = np.real(np.fft.ifft2(phiHat))
    E = np.real(np.array([np.fft.ifft2(EHat[0]), np.fft.ifft2(EHat[1])]))
    return phi, E


def fieldInFourier(rhoHat, L):
    Ja = np.arange(rhoHat.shape[0] // 2)
    Jb = Ja[:0:-1]
    J = np.append(np.append(Ja, [-rhoHat.shape[0] // 2]), - Jb)
    Ka = np.arange(rhoHat.shape[1] // 2)
    Kb = Ka[:0:-1]
    K = np.append(np.append(Ka, [-rhoHat.shape[1] // 2]), - Kb)
    J = np.transpose(np.expand_dims(J, 0).repeat(rhoHat.shape[1], axis=0)) * 2 * np.pi / L[0]
    K = np.expand_dims(K, 0).repeat(rhoHat.shape[0], axis=0) * 2 * np.pi / L[1]
    absolute = J ** 2 + K ** 2
    absolute[0,0] = 1
    phiHat = rhoHat / absolute
    phiHat[0,0] = 0
    E0 = phiHat * -1j * J
    E1 = phiHat * -1J * K
    EHat = np.array([E0, E1])
    return phiHat, EHat


def fieldAmpere(Jb, dphiNewton, L):
    #rhonHat = np.fft.fft2(rhon)
    dphiNewtonHat = np.fft.fft2(dphiNewton.reshape([NG,NG]))
    JHat = np.array([np.fft.fft2(Jb[0]), np.fft.fft2(Jb[1])])
    dphiHat, dEHat = fieldInFourierCurrent(JHat, dphiNewtonHat, L)
    dphi = np.real(np.fft.ifft2(dphiHat))
    resphi = dphiNewton - dphi.flatten()
    #rhonp1 = np.real(np.fft.ifft2(rhonp1Hat))
    dE = np.real(np.array([np.fft.ifft2(dEHat[0]), np.fft.ifft2(dEHat[1])]))
    return resphi, dE

def fieldInFourierCurrent(JHat, dphiNewtonHat, L):
    Ja = np.arange(dphiNewtonHat.shape[0] // 2)
    Jb = Ja[:0:-1]
    J = np.append(np.append(Ja, [-dphiNewtonHat.shape[0] // 2]), - Jb)
    Ka = np.arange(dphiNewtonHat.shape[1] // 2)
    Kb = Ka[:0:-1]
    K = np.append(np.append(Ka, [-dphiNewtonHat.shape[1] // 2]), - Kb)
    J = np.transpose(np.expand_dims(J, 0).repeat(dphiNewtonHat.shape[1], axis=0)) * 2 * np.pi / L[0]
    K = np.expand_dims(K, 0).repeat(dphiNewtonHat.shape[0], axis=0) * 2 * np.pi / L[1]
    absolute = J ** 2 + K ** 2
    absolute[0,0] = 1
    #rhonp1Hat = rhonHat + DT * (-1j * J * JHat[0,:,:] + -1j * K * JHat[1,:,:])
    drhoHat = DT * (-1j * J * JHat[0,:,:] + -1j * K * JHat[1,:,:])
    #phiHat = rhonp1Hat / absolute
    dphiHat = drhoHat / absolute
    dphiHat[0,0] = 0
    #E0 = phiHat * -1j * J
    #E1 = phiHat * -1j * K
    dE0 = dphiNewtonHat * -1j * J
    dE1 = dphiNewtonHat * -1j * K
    dEHat = np.array([dE0, dE1])
    return dphiHat, dEHat
