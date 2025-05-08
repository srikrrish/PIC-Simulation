import time
from initialize import *
import energy, interpolate, field, dynamics, figures
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
import torch
#Eg = np.zeros([2,NG**2])
#model = torch.load('_Models/fno_dse.pt', map_location=torch.device('cpu'))
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
    #rho_back = - Q * N / (L[0] * L[1])  # background rho
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
    rho = interpolate.interpolate(M, dx, Q, L)
    phi, Eg = field.field(rho, L)
    En = dynamics.Epart(M, Eg)
    for it in range(NT):
        print(it)
        ##Picard iterations
        tol = 1e-5
        err_pos = 10
        err_vel = 10
        ##Initial guess from previous time step
        xk = xp
        vk = vp
        itpicard = 0
        maxit = 10
        while ((err_pos > tol) or (err_vel > tol)) and (itpicard < maxit):
        #while ((err_pos > tol) or (err_vel > tol)):
            xk_wop = xk
            xk = dynamics.toPeriodicND(xk, L)
            M = interpolate.interpMatrix(xk, wp, dx, L)
            rho = interpolate.interpolate(M, dx, Q, L)
            phi, Eg = field.field(rho, L)
            Ek = dynamics.Epart(M, Eg)
            ak = dynamics.acceleratePicard(Ek, En)
            vkp1,kinetic = dynamics.pushPicard(vp, ak, Q)
            xkp1 = dynamics.movePicard(xp,vp,ak)
            err_pos = cp.sqrt(cp.sum((xkp1[0,:] - xk[0,:])**2) + cp.sum((xkp1[1,:] - xk[1,:])**2)) / cp.sqrt(cp.sum(xkp1[0,:]**2) + cp.sum(xkp1[1,:]**2))
            err_vel = cp.sqrt(cp.sum((vkp1[0,:] - vk[0,:])**2) + cp.sum((vkp1[1,:] - vk[1,:])**2)) / cp.sqrt(cp.sum(vkp1[0,:]**2) + cp.sum(vkp1[1,:]**2))
            xk = xkp1
            vk = vkp1
            itpicard = itpicard + 1
            print('Picard iteration: ',itpicard,' error pos: ',err_pos,' error vel: ',err_vel)


        xp = xkp1
        vp = vkp1
        xp = dynamics.toPeriodicND(xp, L)
        M = interpolate.interpMatrix(xp, wp, dx, L)
        rho = interpolate.interpolate(M, dx, Q, L)
        phi, Eg = field.field(rho, L)
        En = dynamics.Epart(M, Eg)
        Egp = cp.sum(En[0,:]**2) * (L[0] * L[1]) / N
        potential = cp.sum(En[0,:]**2 + En[1,:]**2) * 0.5 * (L[0] * L[1]) / N
        #vp, kinetic = dynamics.push(vp, a, Q, it)
        #xp, wp = dynamics.move(xp, vp, wp, L, it)
        #potential = energy.potential(rho, phi, dx)
        #Ek.append(kinetic.get())
        #Ep.append(potential.get())
        E.append((kinetic + potential).get())
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
