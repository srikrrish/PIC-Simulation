import initialize
from energy import kinetic
import numpy as np
import finufft

def accelerate(vp, it, M, E, wp):
    a = np.transpose(M * E) * initialize.QM / wp
    return push(vp, a, it)


def accelInFourier(vp, xp, it, EgHat, Shat, wp):
    coeff = EgHat * Shat
    coeff = np.append(coeff[0], coeff[:0:-1])
    a = np.real(finufft.nufft1d2(xp * 2 * np.pi / initialize.L, coeff, eps=1e-12, modeord=1) * initialize.QM / (initialize.L * wp))
    return push(vp, a, it)


def push(vp, a, it):
    if it == 0:
        return vp + a * initialize.DT / 2, kinetic(vp + a * initialize.DT / 2)
    else:
        return vp + a * initialize.DT, kinetic(vp + a * initialize.DT / 2)


def move(xp, vp, wp, it=None):
    if wp is 1:
        return xp + vp * initialize.DT, 1
    else:
        return xp + vp * initialize.DT, wp + initialize.DT * initialize.findsource(xp + vp * initialize.DT / 2, vp, initialize.L, it + 0.5, initialize.DT)


def toPeriodic(x, L, discrete=False):
    out = (x < 0)
    x[out] = x[out] + L
    if discrete:
        out = (x > L - 1)
    else:
        out = (x >= L)
    x[out] = x[out] - L
    return x
