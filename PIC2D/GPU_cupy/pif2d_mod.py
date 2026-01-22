import time
from initialize import *
import energy, interpolate, field, dynamics, figures
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
import torch
import specKernel
#Eg = np.zeros([2,NG**2])
#model = torch.load('_Models/fno_dse.pt', map_location=torch.device('cpu'))
nk = 1
#kval = np.array([0.25, 0.5, 1.0])
kval = cp.array([0.5])
pos = cp.zeros([NT*nk, N, 2], dtype='f');
Eout = cp.zeros([NT*nk, N, 2], dtype='f');


for itk in range(nk):
    k = cp.array([kval[itk], kval[itk]])
    L = cp.array([2*cp.pi/k[0], 2*cp.pi/k[1]])  # Length of the container
    Q = L[0] * L[1] / (QM * N)  # Charge of a particle
    #rho_back = - Q * N / (L[0] * L[1])  # background rho
    dx = L / NG  # cell length
    cp.random.seed(0)
    np.random.seed(0)
    t = time.time()
    xpc,vpc = InvTransSampling(alpha,k,L,N)
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


    Shat = specKernel.specKernel(NG, L, dx)
    t1 = time.time()
    for it in range(NT):
        print(it)
        xp = dynamics.toPeriodicND(xp, L)
        rhoHat = interpolate.specInterpolate(xp, Shat, Q, L)
        phiHat, EgHat = field.fieldInFourier(rhoHat,L)
        pos[(itk*NT)+it,:,:] = cp.transpose(xp.astype(cp.float32))
        a,Eout = dynamics.accelInFourier(xp, EgHat, Eout, Shat, wp, L, it, itk)
        Egp = cp.sum(Eout[(itk*NT)+it,:,1]**2) * (L[0] * L[1]) / N
        potential = cp.sum(Eout[(itk*NT)+it,:,0]**2 + Eout[(itk*NT)+it,:,1]**2) * 0.5 * (L[0] * L[1]) / N
        vp, kinetic = dynamics.push(vp, a, Q, it)
        xp, wp = dynamics.move(xp, vp, wp, L, it)
        E.append((kinetic + potential).get())
        momx = cp.sum(Q * vp[0,:] / QM)
        momy = cp.sum(Q * vp[1,:] / QM)
        momentum.append((cp.sqrt(momx**2 + momy**2)).get())
        Exp.append(Egp.get())

    figures.landauDecayFigIppl(Exp,'strong')
    figures.conservationErrors(E,momentum)
    Int_time = time.time() - t1
    print('Particle initialization time:',particle_init_time)
    print('Integration time:',Int_time)
    print('Total time:',time.time()-t)


cp.save('data/pos_strongLandau_pif_500k',pos)
cp.save('data/Eout_strongLandau_pif_500k',Eout)
