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
    xp = np.zeros([2, N])
    u0 = np.random.rand(2, N)
    for i in range(N):
        print(i)
        for d in range(2):
            u =  L[d] * u0[d, i]
            x = u / (1+alpha)
            #xp[d,i] = optimize.newton(f, x, fprime,args=(alpha,k[d],u))
            xp[d,i],niter = Newton1d(x,alpha,k[d],u)
            #print(niter)

    return xp

DT = 0.5  # Length of a time step
T = 20
NT = 200#int(T/DT)  # number of time steps
NG = 32#512 # Number of Grid points
N = 40000  # Number of simulation particles
QM = -1  # charge per mass
VT = 1  # Thermal Velocity
wp = 1

#k = np.array([0.5, 0.5])
#L = np.array([2*np.pi/k[0], 2*np.pi/k[1]])  # Length of the container
#DT = .002  # Length of a time step
#T = 2.5#20
#NT = int(T/DT)  # number of time steps
#NG = 32 # Number of Grid points
#N = 100000  # Number of simulation particles
#QM = -1  # charge per mass
#VT = 1  # Thermal Velocity
#alpha = 0.05  # Magnitude of perturbation in x
#Q = L[0] * L[1] / (QM * N)  # Charge of a particle
##rho_back = - Q * N / (L[0] * L[1])  # background rho
#dx = L / NG  # cell length
#np.random.seed(0)
#xp = InvTransSampling(alpha,k,L,N)
#vp = np.random.randn(2, N)
#bins = np.linspace(0,L[0],1000)
#plt.hist(xp[0],bins)
#plt.savefig('X_dist.png')
#plt.clf()
#plt.hist(xp[1],bins)
#plt.savefig('Y_dist.png')
#plt.clf()
#wp = 1
#Ek = []
#Ep = []
#E = []
#momentum = []
##phiMax = []
#Exp = []




