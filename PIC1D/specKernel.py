import numpy as np
from initialize import NG, L


def specKernel(order=2):
    PL = 1
    Ka = np.arange(1, NG // 2)
    Kb = Ka[::-1]
    K = np.append(np.append(Ka, [- NG // 2]), - Kb)
    Shat = (L * np.sin(np.pi * K * PL / L) / (np.pi * K * PL)) ** order
    Shat = np.append([1], Shat)
    K = np.append([0], K)
    return Shat, K
