import numpy as np
from initialize import NG, L, dx


def specKernel(order=2):
    PL = 1
    Ka = np.arange(1, NG // 2)
    Kb = Ka[::-1]
    K = np.append(np.append(Ka, [- NG // 2]), - Kb)
    K = ((2 * np.pi) / L) * K
    Shat = (np.sin(K * dx / 2) / (K * dx / 2)) ** order
    Shat = np.append([1], Shat)
    K = np.append([0], K)
    return Shat, K
