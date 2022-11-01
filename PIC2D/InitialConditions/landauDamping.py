import numpy as np

def findsource():
    return None

L = np.array([32, 1])  # Length of the container
DT = .02  # Length of a time step
NT = 500  # number of time steps
NG = 16 # Number of Grid points
N = 20000  # Number of simulation particles
WP = 1  # omega p
QM = -1  # charge per mass
VT = 1  # Thermal Velocity
lambdaD = VT / WP
XP1 = 0.2  # Magnitude of perturbation in x
mode = 2  # Mode of the sin wave in perturbation
Q = WP ** 2 * L[0] * L[1] / (QM * N)  # Charge of a particle
rho_back = - Q * N / (L[0] * L[1])  # background rho
dx = L / NG  # cell length
k = lambdaD * mode * 2 * np.pi / L[0]
i = 0
xp = np.zeros([2, N])
xp[1] = np.random.rand(N) * L[1]
while i < N:
   x = np.random.rand(1, 2) * np.array([L[0], N / L[0] * (1 + XP1)])
   if x[0, 1] < N / L[0] * (1 + XP1 * np.sin(2 * np.pi * x[0, 0] / L[0] * mode)):
       xp[0, i] = x[0, 0]
       i = i + 1
vp = VT * np.random.randn(2, N)
wp = 1
Ek = []
Ep = []
E = []
momentum = []
phiMax = []
