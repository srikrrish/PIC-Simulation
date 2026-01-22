import cupy as cp
from scipy import special


def specKernel(NG, L, dx, order=2):
    #PL = L / NG
    Ka = cp.arange(1, NG // 2)
    Kb = Ka[::-1]
    K = cp.append(cp.append(Ka, [- NG // 2]), - Kb)
    K = ((2 * cp.pi) / L[0]) * K
    #Shat0 = (L[0] * cp.sin(cp.pi * K * PL[0] / L[0]) / (cp.pi * K * PL[0])) ** order
    Shat0 = (cp.sin(K * dx[0] / 2) / (K * dx[0] / 2)) ** order
    Shat0 = cp.append([1], Shat0).reshape(NG, 1)
    
    #Ka = cp.arange(1, NG // 2)
    #Kb = Ka[::-1]
    #K = cp.append(cp.append(Ka, [- NG // 2]), - Kb)
    #Shat1 = (L[1] * cp.sin(cp.pi * K * PL[1] / L[1]) / (cp.pi * K * PL[1])) ** order
    Shat1 = (cp.sin(K * dx[1] / 2) / (K * dx[1] / 2)) ** order
    Shat1 = cp.append([1], Shat1)

    return cp.kron(Shat0, Shat1)

    
def circleKernel(NG, L, order=2):
    r = cp.min(L / NG)
    Ja = cp.arange(0, NG[0] // 2)
    Jb = Ja[:0:-1]
    J = (cp.append(cp.append(Ja, [- NG[0] // 2]), - Jb) * 2 * cp.pi / L[0]) ** 2 * cp.ones([NG[1], 1])
    Ka = cp.arange(0, NG[1] // 2)
    Kb = Ka[:0:-1]
    K = (cp.append(cp.append(Ka, [- NG[1] // 2]), - Kb) * 2 * cp.pi / L[1]) ** 2 * cp.ones([NG[0], 1])
    Kabsolute = cp.transpose(cp.sqrt(J + cp.transpose(K)))
    Kabsolute[0,0] = 1  # avoid 0 on denominator
    Shat = (2 * special.j1(r * Kabsolute) / (r * Kabsolute)) ** order
    Shat[0, 0] = 1
    return Shat
