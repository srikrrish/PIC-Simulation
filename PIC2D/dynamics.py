from initialize import QM, DT, findsource, N, NT, NG
from energy import kinetic
import interpolate, field
import numpy as np
#import finufft

def accelerate(M, E, wp, Eout, it, itk):
    Extemp = M * E[0].flatten()
    Eytemp = M * E[1].flatten()
    a1 = np.transpose(Extemp) * QM / wp
    a2 = np.transpose(Eytemp) * QM / wp
    Eout[(itk*NT)+it,:,0] = Extemp.astype(np.float32)
    Eout[(itk*NT)+it,:,1] = Eytemp.astype(np.float32)
    #a1 = np.reshape(Eout[it,:,0], (1,N)) * QM / wp
    #a2 = np.reshape(Eout[it,:,1], (1,N)) * QM / wp
    return np.array([a1, a2]), Eout
    #return np.concatenate((a1, a2), axis=0), Eout


def Epart(M, Efield):
    En = np.zeros([2, N])
    En[0,:] = M * Efield[0].flatten()
    En[1,:] = M * Efield[1].flatten()
    return En

def accelerateML(E, wp):
    a1 =  E[0,:] * QM / wp
    a2 =  E[1,:] * QM / wp
    return np.array([a1, a2])

def acceleratePicard(Ehalf):
    a =  QM * Ehalf
    return a

def accelInFourier(xp, EgHat, Shat, wp, L):
    coeff1 = np.conjugate(EgHat[0] * Shat)
    a1 = np.real(finufft.nufft2d2(xp[0] * 2 * np.pi / L[0], xp[1] * 2 * np.pi / L[1], coeff1, eps=1e-12, modeord=1) * QM / (L[0] * L[1] * wp))
    coeff2 = np.conjugate(EgHat[1] * Shat)
    a2 = np.real(finufft.nufft2d2(xp[0] * 2 * np.pi / L[0], xp[1] * 2 * np.pi / L[1], coeff2, eps=1e-12, modeord=1) * QM / (L[1] * L[0] * wp))

    return np.array([a1, a2])

def nonlinResidue(vNewton,xn,vn,rhon,Egn,dx,L,Q):
    xhalf = xn + DT * 0.5 * vNewton
    xhalf = toPeriodicND(xhalf, L)
    Mhalf = interpolate.interpMatrix(xhalf, dx, L)
    #Mvx = interpolate.interpvelMatrix(xhalf, dx, L,vNewton[0,:])
    #Mvy = interpolate.interpvelMatrix(xhalf, dx, L,vNewton[1,:])
    #Jb = np.zeros([2,NG,NG])
    #Jb[0,:,:] = interpolate.interpolate(Mvx,dx,Q,L)
    #Jb[1,:,:] = interpolate.interpolate(Mvy,dx,Q,L)
    Jb = interpolate.interpolateCurrent(Mhalf, dx, Q, vNewton, L)
    #phi, Eg, rhonp1 = field.fieldAmpere(Jb, rhon, L)
    #Ehalf = (Eg + Egn) / 2
    #Ehalfp = Epart(Mhalf, Ehalf)
    #res = vNewton - vn - 0.5 * QM * DT * Ehalfp
    Egnp = Epart(Mhalf, Egn)
    dphi, dEg = field.fieldAmpere(Jb, rhon, L)
    dEgp = Epart(Mhalf, dEg)
    res = vNewton - vn - 0.25 * QM * DT * (2*Egnp + dEgp)
    return res

def fullnonlinResidue(vNewton,xn,vn,Egn,dx,L,Q):
    xhalf = np.zeros([2,N])
    xhalf[0,:] = xn[0,:] + DT * 0.5 * vNewton[0:N]
    xhalf[1,:] = xn[1,:] + DT * 0.5 * vNewton[N:2*N]
    xhalf = toPeriodicND(xhalf, L)
    Mhalf = interpolate.interpMatrix(xhalf, dx, L)
    #Mvx = interpolate.interpvelMatrix(xhalf, dx, L,vNewton[0,:])
    #Mvy = interpolate.interpvelMatrix(xhalf, dx, L,vNewton[1,:])
    #Jb = np.zeros([2,NG,NG])
    #Jb[0,:,:] = interpolate.interpolate(Mvx,dx,Q,L)
    #Jb[1,:,:] = interpolate.interpolate(Mvy,dx,Q,L)
    Jb = interpolate.interpolateCurrent(Mhalf, dx, Q, vNewton[0:2*N], L)
    #phi, Eg, rhonp1 = field.fieldAmpere(Jb, rhon, L)
    #Ehalf = (Eg + Egn) / 2
    #Ehalfp = Epart(Mhalf, Ehalf)
    #res = vNewton - vn - 0.5 * QM * DT * Ehalfp
    Egnp = Epart(Mhalf, Egn)
    res = np.zeros([2*N+NG**2]) 
    res[2*N:2*N+NG**2], dEg = field.fieldAmpere(Jb, vNewton[2*N:2*N+NG**2], L)
    dEgp = Epart(Mhalf, dEg)
    res[0:N] = vNewton[0:N] - vn[0,:] - 0.25 * QM * DT * (2*Egnp[0,:] + dEgp[0,:])
    res[N:2*N] = vNewton[N:2*N] - vn[1,:] - 0.25 * QM * DT * (2*Egnp[1,:] + dEgp[1,:])
    return res

def calcResidue(xk,xn,vk,vn,Ehalfp):
    resv = vk - vn - QM * DT * Ehalfp
    #resx = xk - xn - DT * 0.5 * (vk + vn)
    #resxnorm = np.sqrt(np.sum(resx[0,:]**2 + resx[1,:]**2))
    resvnorm = np.sqrt(np.sum(resv[0,:]**2 + resv[1,:]**2))
    #resnorm = cp.sqrt(cp.sum(resx[0,:]**2 + resx[1,:]**2 + resv[0,:]**2 + resv[1,:]**2 ))
    return resvnorm

def push(vp, a, Q, it):
    if it == 0:
        return vp + a * DT / 2, kinetic(vp + a * DT / 2, Q)
    else:
        return vp + a * DT, kinetic(vp + a * DT, Q)

def pushPicard(vp, ak, Q):
    return vp + ak * DT, kinetic(vp + ak * DT, Q)

def movePicard(xp, vp, vkp1):
    return xp + DT * 0.5 * (vp + vkp1)


def move(xp, vp, wp, L, it=None):
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

def toPeriodicND(x, L, dim=2, discrete=False):
    for i in range(dim):
        x[i] = toPeriodic(x[i], L[i], discrete)
    return x
