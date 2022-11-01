import numpy as np
from scipy import special


def specKernel(NG, L, order=2):
    r = 0.5
    Ja = np.arange(0, NG[0] // 2)
    Jb = Ja[:0:-1]
    J = (np.append(np.append(Ja, [- NG[0] // 2]), - Jb) * 2 * np.pi / L[0]) ** 2 * np.ones([NG[1], 1])
    Ka = np.arange(0, NG[1] // 2)
    Kb = Ka[:0:-1]
    K = (np.append(np.append(Ka, [- NG[1] // 2]), - Kb) * 2 * np.pi / L[1]) ** 2 * np.ones([NG[0], 1])
    Kabsolute = np.transpose(np.sqrt(J + np.transpose(K)))
    Kabsolute[0,0] = 1  # avoid 0 on denominator
    Shat = (2 * special.j1(r * Kabsolute) / (r * Kabsolute)) ** order
    Shat[0, 0] = 1
    print(Shat.shape)
    return Shat
