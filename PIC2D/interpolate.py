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
    return sparse.csr_matrix((fraz[0], (p, g[0]))) + sparse.csr_matrix((fraz[1], (p, g[1]))) + sparse.csr_matrix(
        (fraz[2], (p, g[2])))


def interpolate(M):
    return np.asarray((Q / dx) * M.sum(0) + rho_back * np.ones([1, NG]))[0]


def specInterpolate(XP, Shat, wp=1):
    rhoHat = Q * Shat * finufft.nufft2d1(XP[0] * 2 * np.pi / L[0], XP[1] * 2 * np.pi / L[1], 0j + np.zeros(N) + wp, NG, eps=1e-12, modeord=1)
    rhoHat[1:, :] = np.flip(rhoHat[1:, :], 0)
    rhoHat[:, 1:] = np.flip(rhoHat[:,1:], 1)
    return rhoHat
