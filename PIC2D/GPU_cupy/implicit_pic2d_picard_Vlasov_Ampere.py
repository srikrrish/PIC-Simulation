import time
from initialize import *
import energy, interpolate, field, dynamics, figures
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
import torch
nk = 1
#kval = np.array([0.25, 0.5, 1.0])
kval = cp.array([0.5])
#pos = cp.zeros([NT*nk, N, 2], dtype='f');
Eout = cp.zeros([NT*nk, N, 2], dtype='f');


for itk in range(nk):
    k = cp.array([kval[itk], kval[itk]])
    L = cp.array([2*cp.pi/k[0], 2*cp.pi/k[1]])  # Length of the container
    alpha = 0.05  # Magnitude of perturbation in x
    Q = L[0] * L[1] / (QM * N)  # Charge of a particle
    rho_back = - Q * N / (L[0] * L[1])  # background rho
    dx = L / NG  # cell length
    cp.random.seed(0)
    np.random.seed(0)
    t = time.time()
    xpc = InvTransSampling(alpha,k,L,N)
    vpc = np.random.randn(2, N)
    xp = cp.asarray(xpc)
    vp = cp.asarray(vpc)
    particle_init_time = time.time()-t
    #bins = np.linspace(0,L[0],1000)
    #plt.hist(xp[0],bins)
    #plt.savefig('X_dist.png')
    #plt.clf()
    #plt.hist(xp[1],bins)
    #plt.savefig('Y_dist.png')
    #plt.clf()
    #Ek = []
    #Ep = []
    E = []
    momentum = []
    Exp = []


    t1 = time.time()
    xp = dynamics.toPeriodicND(xp, L)
    M = interpolate.interpMatrix(xp, wp, dx, L)
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
        itpicard = 0
        maxit = 10
        Ehalfp = En
        relres = 1
        #while ((err_pos > tol) or (err_vel > tol)) and (itpicard < maxit):
        while (relres > tol):
        #while ((err_pos > tol) or (err_vel > tol)):
            #xk_wop = xk
            #xk = dynamics.toPeriodicND(xk, L)
            M = interpolate.interpMatrix(xk, wp, dx, L)
            xhalf = (xk + xp) / 2
            xhalf = dynamics.toPeriodicND(xhalf, L)
            Mhalf = interpolate.interpMatrix(xhalf, wp, dx, L)
            #rho = interpolate.interpolate(M, dx, Q, L)
            vhalf = (vk + vp) / 2
            Jb = interpolate.interpolateCurrent(Mhalf, dx, Q, vhalf, L)
            #phi, Eg = field.field(rho, L)
            phi, Eg, rhonp1 = field.fieldAmpere(Jb, rhon, L)
            Ehalf = (Eg + Egn) / 2
            Ehalfp = dynamics.Epart(Mhalf, Ehalf)
            ak = dynamics.acceleratePicard(Ehalfp)
            vkp1,kinetic = dynamics.pushPicard(vp, ak, Q)
            xkp1 = dynamics.movePicard(xp,vp,vkp1)
            xkp1 = dynamics.toPeriodicND(xkp1, L)
            omega = 0.0
            xkp1 = (1-omega)*xkp1 + omega*xk
            vkp1 = (1-omega)*vkp1 + omega*vk
            xkp1 = dynamics.toPeriodicND(xkp1, L)
            err_pos = cp.sqrt(cp.sum((xkp1[0,:] - xk[0,:])**2) + cp.sum((xkp1[1,:] - xk[1,:])**2)) / cp.sqrt(cp.sum(xkp1[0,:]**2) + cp.sum(xkp1[1,:]**2))
            err_vel = cp.sqrt(cp.sum((vkp1[0,:] - vk[0,:])**2) + cp.sum((vkp1[1,:] - vk[1,:])**2)) / cp.sqrt(cp.sum(vkp1[0,:]**2) + cp.sum(vkp1[1,:]**2))
            res = dynamics.calcResidue(xk,xp,vk,vp,Ehalfp)
            if(itpicard == 0):
                res0 = res

            relres = res/res0
            #breakpoint()
            xk = xkp1
            vk = vkp1
            itpicard = itpicard + 1
            print('Picard iteration: ',itpicard,'relative residual: ',relres,' error pos: ',err_pos,' error vel: ',err_vel)


        vhalf = (vkp1 + vp) / 2
        xhalf = (xkp1 + xp) / 2
        xp = xkp1
        vp = vkp1
        xp = dynamics.toPeriodicND(xp, L)
        M = interpolate.interpMatrix(xp, wp, dx, L)
        Mhalf = interpolate.interpMatrix(xhalf, wp, dx, L)
        #rho = interpolate.interpolate(M, dx, Q, L)
        #phi, Eg = field.field(rho, L)
        #En = dynamics.Epart(M, Eg)
        Jb = interpolate.interpolateCurrent(Mhalf, dx, Q, vhalf, L)
        phi, Egn, rhonp1 = field.fieldAmpere(Jb, rhon, L)
        En = dynamics.Epart(M, Egn)
        rhon = rhonp1
        Egp = cp.sum(En[0,:]**2) * (L[0] * L[1]) / N
        #potential = cp.sum(En[0,:]**2 + En[1,:]**2) * 0.5 * (L[0] * L[1]) / N
        #vp, kinetic = dynamics.push(vp, a, Q, it)
        #xp, wp = dynamics.move(xp, vp, wp, L, it)
        potential = energy.potential(rhonp1, phi, dx)
        #potential = energy.energypot(Egn, dx)
        #Ek.append(kinetic.get())
        #Ep.append(potential.get())
        E.append((kinetic + potential).get())
        if(it > 0):
            energy_error = np.abs(E[it] - E[0]) / np.abs(E[0])
            print('Time step: ',it,' energy error: ',energy_error)
        momx = cp.sum(Q * vp[0,:] / QM)
        momy = cp.sum(Q * vp[1,:] / QM)
        momentum.append((cp.sqrt(momx**2 + momy**2)).get())
        #momentum.append(sum(Q * vp / QM))
        #phiMax.append(np.max(phi))
        Exp.append(Egp.get())
    #figures.field2D(rho)

    #figures.landauDecayFig(Exp)
    figures.landauDecayFigIppl(Exp, k)
    figures.conservationErrors(E,momentum)
    #figures.energyFig(E,k,Ek,Ep)
    Int_time = time.time() - t1
    print('Particle initialization time:',particle_init_time)
    print('Integration time:',Int_time)
    print('Total time:',time.time()-t)


#cp.save('data/pos',pos)
#cp.save('data/Eout',Eout)
