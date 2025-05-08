import numpy as np

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
    u0 = np.random.rand(N)
    for i in range(N):
        print(i)
        u =  L * u0[i]
        x = u / (1+alpha)
        xp[i],niter = Newton1d(x,alpha,k,u)

    return xp



k = 0.5
L = 2*np.pi/k  # Length of the container
DT = 0.002  # Length of a time step
T = 2.5
NT = int(T/DT)  # number of time steps
NG = 32  # Number of Grid points
N = 100000  # Number of simulation particles
WP = 1  # omega p
QM = -1  # charge per mass
VT = 1  # Thermal Velocity
#lambdaD = VT / WP
#XP1 = 0.2  # Magnitude of perturbation in x
#mode = 2  # Mode of the sin wave in perturbation
Q = L / (QM * N)  # Charge of a particle
rho_back = - Q * N / L  # background rho
dx = L / NG  # cell length
alpha = 0.05
np.random.seed(0)
xp = InvTransSampling(alpha,k,L,N)
vp = np.random.randn(N)
#i = 0
#xp0 = np.zeros(int(N / mode))
#vp0 = VT * np.random.randn(int(N / mode))
#while i < int(N / mode):
#    x = np.random.rand(1, 2) * np.array([L / mode, N / L * (1 + XP1)])
#    if x[0, 1] < N / L * (1 + XP1 * np.sin(2 * np.pi * x[0, 0] / L * mode)):
#        xp0[i] = x[0, 0]
#        i = i + 1
#xp = xp0
#vp = vp0
#for j in range(mode - 1):
#    xp = np.append(xp, xp0 + L * (j + 1) / mode)
#    vp = np.append(vp, vp0)
wp = 1
Ek = []
Ep = []
E = []
momentum = []
Exp = []
#phiMax = []
