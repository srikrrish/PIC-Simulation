from initialize import QM, DT, N, NT, NG, findsource, L
from energy import kinetic
import numpy as np
#import finufft

def accelerate(it, M, Eg, Eout, wp):
    Etemp = M * Eg
    a = np.transpose(Etemp) * QM / wp
    Eout[it,:] = Etemp.astype(np.float32)
    return a, Eout


def accelInFourier(vp, xp, it, EgHat, Shat, wp):
    coeff = EgHat * Shat
    coeff = np.append(coeff[0], coeff[:0:-1])
    a = np.real(finufft.nufft1d2(xp * 2 * np.pi / initialize.L, coeff, eps=1e-12, modeord=1) * initialize.QM / (initialize.L * wp))
    return push(vp, a, it)


def push(vp, a, it):
    if it == 0:
        return vp + a * DT / 2, kinetic(vp + a * DT / 2)
    else:
        return vp + a * DT, kinetic(vp + a * DT)


def move(xp, vp, wp, it=None):
    if wp == 1:
        return xp + vp * DT, 1
    else:
        return xp + vp * DT, wp + DT * findsource(xp + vp * DT / 2, vp, L, it + 0.5, DT)


def toPeriodic(x, L, discrete=False):
    out = (x < 0)
    x[out] = x[out] + L
    if discrete:
        out = (x > L - 1)
    else:
        out = (x >= L)
    x[out] = x[out] - L
    return x
