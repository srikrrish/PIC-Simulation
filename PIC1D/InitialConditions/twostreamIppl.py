import numpy as np
import matplotlib.pyplot as plt
#from scipy import optimize

def findsource():
    return None

def f(x,alpha,kd,u):
    return (x  + (alpha * (np.sin(kd * x) / kd)) - u)

def fprime(x,alpha,kd,u):
    return(1 + (alpha * np.cos(kd * x)))

def Newton1d(xi, alpha, kd, u):
    tol = 1e-12
    max_iter = 20

    k=0
    x=0
    while (k <= max_iter) and (np.abs(f(xi,alpha,kd,u)) > tol):
        x = xi - (f(xi,alpha,kd,u)/fprime(xi,alpha,kd,u))
        xi = x
        k = k+1

    if(k == max_iter):
        print('Newton iterations did not converge')
        exit()
    return x,k

def InvTransSampling(alpha,k,L,N):
    xp = np.zeros(N)
    vp = np.zeros(N)
    Nhalf = int(N/2)
    vp[:Nhalf] = -np.pi/2.0 + 0.1 * np.random.randn(Nhalf)
    vp[Nhalf:] =  np.pi/2.0 + 0.1 * np.random.randn(Nhalf)
    u0 = np.random.rand(N)
    for i in range(N):
        print(i)
        u =  L * u0[i]
        x = u / (1+alpha)
        xp[i],niter = Newton1d(x,alpha,k,u)

    return xp,vp

k = 0.5
L = 2*np.pi/k  # Length of the container
DT = 0.05  # Length of a time step
T = 60
NT = int(T/DT)  # number of time steps
NG = 32 # Number of Grid points
N = 50000  # Number of simulation particles
QM = -1  # charge per mass
VT = 1  # Thermal Velocity
alpha = 0.01  # Magnitude of perturbation in x
Q = L / (QM * N)  # Charge of a particle
rho_back = - Q * N / L  # background rho
dx = L / NG  # cell length
np.random.seed(0)
xp,vp = InvTransSampling(alpha,k,L,N)
bins = np.linspace(0,L,1000)
plt.hist(xp,bins)
plt.savefig('X_dist.png')
plt.clf()
binsv = np.linspace(-10,10,1000)
plt.hist(vp,binsv)
plt.savefig('V_dist.png')
plt.clf()
wp = 1
Ek = []
Ep = []
E = []
momentum = []
#phiMax = []
Exp = []




