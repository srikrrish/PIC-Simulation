from initialize import dx, NG, rho_back, Q, N, L
import numpy as np
from dynamics import toPeriodic
from scipy import sparse
import finufft

p = np.linspace(0, N - 1, N).astype(int)

def interpMatrix(XP, wp):
    # projection p->g
    g1 = np.floor(XP / dx).astype(int)  # which grid point to project onto
    g = np.array([g1 - 1, g1, g1 + 1])  # used to determine bc
    delta = XP % dx
    fraz = np.array([(1 - delta) ** 2 / 2, 1 - ((1 - delta) ** 2 / 2 + delta ** 2 / 2), delta ** 2 / 2] * wp)

    # apply bc on the projection
    g = toPeriodic(g, NG, True)
    return sparse.csr_matrix((fraz[0], (p, g[0])),shape=(N,NG)) + sparse.csr_matrix((fraz[1], (p, g[1])),shape=(N,NG)) + sparse.csr_matrix(
        (fraz[2], (p, g[2])),shape=(N,NG))


def interpolate(M):
    return np.asarray((Q / dx) * M.sum(0) + rho_back * np.ones([1, NG]))[0]


def specInterpolate(XP, Shat, wp=1):
    rhoHat = Q * Shat * finufft.nufft1d1(XP * 2 * np.pi / L, 0j + np.zeros(N) + wp, NG, eps=1e-12, modeord=1)
    rhoHat = np.append(rhoHat[0], rhoHat[:0:-1])
    return rhoHat
