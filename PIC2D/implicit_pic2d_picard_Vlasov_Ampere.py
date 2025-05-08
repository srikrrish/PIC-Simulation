import time
from initialize import *
import energy, interpolate, field, dynamics, figures
import matplotlib.pyplot as plt
import numpy as np
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
    for it in range(NT):
        print(it)
        ##Picard iterations
        tol = 1e-10
        err_pos = 10
        err_vel = 10
        ##Initial guess from previous time step
        xk = xp
        vk = vp
        Eg = Egn
        itpicard = 0
        maxit = 10
        relres = 1
        M = interpolate.interpMatrix(xk, dx, L)
        xhalf = (xk + xp) / 2
        xhalf = dynamics.toPeriodicND(xhalf, L)
        Mhalf = interpolate.interpMatrix(xhalf, dx, L)
        vhalf = (vk + vp) / 2
        Jb = interpolate.interpolateCurrent(Mhalf, dx, Q, vhalf, L)
        Ehalf = (Eg + Egn) / 2
        Ehalfp = dynamics.Epart(Mhalf, Ehalf)
        res0 = dynamics.calcResidue(xk,xp,vk,vp,Ehalfp)
        #while ((err_pos > tol) or (err_vel > tol)) and (itpicard < maxit):
        while (relres > tol):
        #while ((err_pos > tol) or (err_vel > tol)):
            ak = dynamics.acceleratePicard(Ehalfp)
            vkp1,kinetic = dynamics.pushPicard(vp, ak, Q)
            xkp1 = dynamics.movePicard(xp,vp,vkp1)
            xkp1 = dynamics.toPeriodicND(xkp1, L)
            phi, Eg, rhonp1 = field.fieldAmpere(Jb, rhon, L)
            #err_pos = np.sqrt(np.sum((xkp1[0,:] - xk[0,:])**2) + np.sum((xkp1[1,:] - xk[1,:])**2)) / np.sqrt(np.sum(xkp1[0,:]**2) + np.sum(xkp1[1,:]**2))
            #err_vel = np.sqrt(np.sum((vkp1[0,:] - vk[0,:])**2) + np.sum((vkp1[1,:] - vk[1,:])**2)) / np.sqrt(np.sum(vkp1[0,:]**2) + np.sum(vkp1[1,:]**2))
            #if(itpicard == 0):
            #    res0 = res
            #breakpoint()
            xk = xkp1
            vk = vkp1
            M = interpolate.interpMatrix(xk, dx, L)
            xhalf = (xk + xp) / 2
            xhalf = dynamics.toPeriodicND(xhalf, L)
            Mhalf = interpolate.interpMatrix(xhalf, dx, L)
            vhalf = (vk + vp) / 2
            Jb = interpolate.interpolateCurrent(Mhalf, dx, Q, vhalf, L)
            Ehalf = (Eg + Egn) / 2
            Ehalfp = dynamics.Epart(Mhalf, Ehalf)
            res = dynamics.calcResidue(xk,xp,vk,vp,Ehalfp)
            relres = res/res0
            itpicard = itpicard + 1
            print('Picard iteration: ',itpicard,'relative residual: ',relres)


        vhalf = (vkp1 + vp) / 2
        xhalf = (xkp1 + xp) / 2
        xhalf = dynamics.toPeriodicND(xhalf, L)
        xp = xkp1
        vp = vkp1
        xp = dynamics.toPeriodicND(xp, L)
        M = interpolate.interpMatrix(xp, dx, L)
        Mhalf = interpolate.interpMatrix(xhalf, dx, L)
        #rho = interpolate.interpolate(M, dx, Q, L)
        #phi, Eg = field.field(rho, L)
        #En = dynamics.Epart(M, Eg)
        Jb = interpolate.interpolateCurrent(Mhalf, dx, Q, vhalf, L)
        phi, Egn, rhonp1 = field.fieldAmpere(Jb, rhon, L)
        En = dynamics.Epart(M, Egn)
        rhon = rhonp1
        Egp = np.sum(En[0,:]**2) * (L[0] * L[1]) / N
        #potential = np.sum(En[0,:]**2 + En[1,:]**2) * 0.5 * (L[0] * L[1]) / N
        #vp, kinetic = dynamics.push(vp, a, Q, it)
        #xp, wp = dynamics.move(xp, vp, wp, L, it)
        potential = energy.potential(rhonp1, phi, dx)
        #potential = energy.energypot(Egn, dx)
        #Ek.append(kinetic.get())
        #Ep.append(potential.get())
        E.append((kinetic + potential))
        if(it > 0):
            energy_error = np.abs(E[it] - E[0]) / np.abs(E[0])
            print('Time step: ',it,' energy error: ',energy_error)
        momx = np.sum(Q * vp[0,:] / QM)
        momy = np.sum(Q * vp[1,:] / QM)
        momentum.append((np.sqrt(momx**2 + momy**2)))
        #momentum.append(sum(Q * vp / QM))
        #phiMax.append(np.max(phi))
        Exp.append(Egp)
    #figures.field2D(rho)

    #figures.landauDecayFig(Exp)
    figures.landauDecayFigIppl(Exp, k)
    figures.conservationErrors(E,momentum)
    #figures.energyFig(E,k,Ek,Ep)
    Int_time = time.time() - t1
    print('Particle initialization time:',particle_init_time)
    print('Integration time:',Int_time)
    print('Total time:',time.time()-t)


#np.save('data/pos',pos)
#np.save('data/Eout',Eout)
