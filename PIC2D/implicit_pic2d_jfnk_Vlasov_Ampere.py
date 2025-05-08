import time
from initialize import *
import energy, interpolate, field, dynamics, figures
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
nk = 1
kval = np.array([0.5])


for itk in range(nk):
    k = np.array([kval[itk], kval[itk]])
    L = np.array([2*np.pi/k[0], 2*np.pi/k[1]])  # Length of the container
    alpha = 0.05  # Magnitude of perturbation in x
    Q = L[0] * L[1] / (QM * N)  # Charge of a particle
    rho_back = - Q * N / (L[0] * L[1])  # background rho
    dx = L / NG  # cell length
    np.random.seed(0)
    t = time.time()
    xp = InvTransSampling(alpha,k,L,N)
    vp = np.random.randn(2, N)
    particle_init_time = time.time()-t
    #Ek = []
    #Ep = []
    E = []
    momentum = []
    Exp = []


    t1 = time.time()
    xp = dynamics.toPeriodicND(xp, L)
    M = interpolate.interpMatrix(xp, dx, L)
    rhon = interpolate.interpolate(M, dx, Q, L)
    rhon = rhon + rho_back
    phi, Egn = field.field(rhon, L)
    En = dynamics.Epart(M, Egn)
    #Egn = np.zeros([2,NG,NG])
    vinNewton = np.zeros([2*N+NG**2])
    vinNewton[0:N] = vp[0,:]
    vinNewton[N:2*N] = vp[1,:]
    vinNewton[2*N:2*N+NG**2] = phi.flatten()
    xhalf = np.zeros([2,N])
    for it in range(NT):
        print(it)
        ##Newton iterations
        ##Initial guess from previous time step

        sol = optimize.newton_krylov(lambda vNewton:dynamics.fullnonlinResidue(vNewton,xp,vp,Egn,dx,L,Q), vinNewton,verbose=True,maxiter=100,f_rtol=1e-10)
        #sol = optimize.anderson(lambda vNewton:dynamics.nonlinResidue(vNewton,xp,vp,rhon,Egn,dx,L,Q), vinNewton,verbose=True,maxiter=100,f_tol=1e-4)
        #sol = optimize.broyden1(lambda vNewton:dynamics.nonlinResidue(vNewton,xp,vp,rhon,Egn,dx,L,Q), vinNewton,verbose=True)
        vinNewton = sol
        xhalf[0,:] = xp[0,:] + DT * 0.5 * sol[0:N]
        xhalf[1,:] = xp[1,:] + DT * 0.5 * sol[N:2*N]
        xhalf = dynamics.toPeriodicND(xhalf, L)
        Mn = interpolate.interpMatrix(xp, dx, L)
        rhon = interpolate.interpolate(Mn, dx, Q, L)
        rhon = rhon + rho_back

        xp[0,:] = xp[0,:] + DT * sol[0:N]
        xp[1,:] = xp[1,:] + DT * sol[N:2*N]
        vp[0,:] = 2*sol[0:N] - vp[0,:]
        vp[1,:] = 2*sol[N:2*N] - vp[1,:]
        xp = dynamics.toPeriodicND(xp, L)
        Mhalf = interpolate.interpMatrix(xhalf, dx, L)
        Jb = interpolate.interpolateCurrent(Mhalf, dx, Q, sol[0:2*N], L)
        JHat = np.array([np.fft.fft2(Jb[0]), np.fft.fft2(Jb[1])])
        Ja = np.arange(rhon.shape[0] // 2)
        Jbb = Ja[:0:-1]
        J = np.append(np.append(Ja, [-rhon.shape[0] // 2]), - Jbb)
        Ka = np.arange(rhon.shape[1] // 2)
        Kb = Ka[:0:-1]
        K = np.append(np.append(Ka, [-rhon.shape[1] // 2]), - Kb)
        J = np.transpose(np.expand_dims(J, 0).repeat(rhon.shape[1], axis=0)) * 2 * np.pi / L[0]
        K = np.expand_dims(K, 0).repeat(rhon.shape[0], axis=0) * 2 * np.pi / L[1]
        divJ  = np.real(np.fft.ifft2(1j * J * JHat[0,:,:] + 1j * K * JHat[1,:,:]))
        #phi, Egn, rhon = field.fieldAmpere(Jb, rhon, L)
        res, dEgn = field.fieldAmpere(Jb, sol[2*N:2*N+NG**2], L)
        Egn = dEgn + Egn
        M = interpolate.interpMatrix(xp, dx, L)
        rhonp1 = interpolate.interpolate(M, dx, Q, L)
        rhonp1 = rhonp1 + rho_back
        resCont = (rhonp1 - rhon)/DT + divJ
        resContnorm = np.max(np.abs(resCont))
        print('Time step: ',it,' Continuity residual: ',resContnorm)
        En = dynamics.Epart(M, Egn)
        Egp = np.sum(En[0,:]**2) * (L[0] * L[1]) / N
        #potential = energy.potential(rhon, phi, dx)
        potential = energy.energypot(Egn, dx)
        kinetic =  energy.kinetic(vp, Q)
        E.append(kinetic + potential)
        if(it > 0):
            energy_error = np.abs(E[it] - E[0]) / np.abs(E[0])
            print('Time step: ',it,' energy error: ',energy_error)
        momx = np.sum(Q * vp[0,:] / QM)
        momy = np.sum(Q * vp[1,:] / QM)
        momentum.append(np.sqrt(momx**2 + momy**2))
        Exp.append(Egp)


    figures.landauDecayFigIppl(Exp, k)
    figures.conservationErrors(E,momentum)
    #figures.energyFig(E,k,Ek,Ep)
    Int_time = time.time() - t1
    print('Particle initialization time:',particle_init_time)
    print('Integration time:',Int_time)
    print('Total time:',time.time()-t)

