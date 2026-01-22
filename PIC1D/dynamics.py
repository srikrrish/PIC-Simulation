from initialize import QM, DT, N, NT, NG, findsource, L
from energy import kinetic
import numpy as np
import finufft

def accelerate(it, M, Eg, Eout, wp):
    Etemp = M * Eg
    a = np.transpose(Etemp) * QM / wp
    Eout[it,:] = Etemp.astype(np.float32)
    return a, Eout


def accelInFourier(vp, xp, it, EgHat, Eout, Shat, wp):
    coeff = EgHat * Shat
    coeff = np.append(coeff[0], coeff[:0:-1])
    Etemp = np.real(finufft.nufft1d2(xp * 2 * np.pi / L, coeff, eps=1e-12, modeord=1) / L)
    a = Etemp * QM / wp
    Eout[it,:] = Etemp.astype(np.float32)
    return a, Eout


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
