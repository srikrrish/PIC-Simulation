import time
from initialize import *
import energy, interpolate, field, dynamics, figures
import matplotlib.pyplot as plt
import numpy as np
import torch
#Eg = np.zeros([2,NG**2])
#model = torch.load('_Models/fno_dse.pt', map_location=torch.device('cpu'))
nk = 1
#kval = np.array([0.25, 0.5, 1.0])
kval = np.array([0.5])
#pos = np.zeros([NT*nk, N, 2], dtype='f');
Eout = np.zeros([NT*nk, N, 2], dtype='f');


for itk in range(nk):
    k = np.array([kval[itk], kval[itk]])
    L = np.array([2*np.pi/k[0], 2*np.pi/k[1]])  # Length of the container
    alpha = 0.05  # Magnitude of perturbation in x
    Q = L[0] * L[1] / (QM * N)  # Charge of a particle
    #rho_back = - Q * N / (L[0] * L[1])  # background rho
    dx = L / NG  # cell length
    np.random.seed(0)
    t = time.time()
    xp = InvTransSampling(alpha,k,L,N)
    vp = np.random.randn(2, N)
    particle_init_time = time.time()-t
    #bins = np.linspace(0,L[0],1000)
    #plt.hist(xp[0],bins)
    #plt.savefig('X_dist.png')
    #plt.clf()
    #plt.hist(xp[1],bins)
    #plt.savefig('Y_dist.png')
    #plt.clf()
    Ek = []
    Ep = []
    E = []
    momentum = []
    Exp = []


    t1 = time.time()
    for it in range(NT):
        print(it)
        xp = dynamics.toPeriodicND(xp, L)
        M = interpolate.interpMatrix(xp, wp, dx, L)
        rho = interpolate.interpolate(M, dx, Q, L)
        #print(np.abs((np.sum(rho*dx[0]*dx[1]) - (Q*N))/(Q*N)))
        #print(np.sum(rho*dx[0]*dx[1]))
        phi, Eg = field.field(rho, L)
        #vp, kinetic = dynamics.accelerate(M, Eg, wp)
        #ti = (it*DT) * np.ones([N, 1])
        #pos[(itk*NT)+it,:,0] = np.squeeze(ti.astype(np.float32))
        #pos[(itk*NT)+it,:,:] = np.transpose(xp.astype(np.float32))
        #inputs = torch.tensor(np.transpose(xp), dtype=torch.float)
        #predictions = model(inputs)
        #Efieldparticle = predictions.numpy() * Q
        #Efieldparticle = np.transpose(Efieldparticle)
        a, Eout = dynamics.accelerate(M, Eg, wp, Eout, it, itk)
        #a = dynamics.accelerateML(Efieldparticle, wp)
        #Eg[0] = np.transpose(M) * np.reshape(Efieldparticle[0,:], (N,1))
        #Eg[1] = np.transpose(M) * np.reshape(Efieldparticle[1,:], (N,1))
        #Egp = energy.energypotx(Eg, dx)
        Egp = np.sum(Eout[(itk*NT)+it,:,0]**2) * (L[0] * L[1]) / N
        #potential = np.sum(Eout[(itk*NT)+it,:,0]**2 + Eout[(itk*NT)+it,:,1]**2) * 0.5 * (L[0] * L[1]) / N
        vp, kinetic = dynamics.push(vp, a, Q, it)
        xp, wp = dynamics.move(xp, vp, wp, L, it)
        #potential = energy.potential(rho, phi, dx)
        #Ek.append(kinetic)
        #Ep.append(potential)
        #E.append(kinetic + potential)
        #momx = np.sum(Q * vp[0,:] / QM)
        #momy = np.sum(Q * vp[1,:] / QM)
        #momentum.append(np.sqrt(momx**2 + momy**2))
        #momentum.append(sum(Q * vp / QM))
        #phiMax.append(np.max(phi))
        Exp.append(Egp)
    #figures.field2D(rho)

    #figures.landauDecayFig(Exp)
    figures.landauDecayFigIppl(Exp, k)
    #figures.conservationErrors(E,momentum)
    #figures.energyFig(E,k,Ek,Ep)
    Int_time = time.time() - t1
    print('Particle initialization time:',particle_init_time)
    print('Integration time:',Int_time)
    print('Total time:',time.time()-t)


#np.save('data/pos',pos)
#np.save('data/Eout',Eout)
