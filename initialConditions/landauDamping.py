import numpy as np

L = 32  # Length of the container
DT = .02  # Length of a time step
NT = 1000  # number of time steps
NG = 32  # Number of Grid points
N = 40000  # Number of simulation particles
WP = 1  # omega p
QM = -1  # charge per mass
VT = 1  # Thermal Velocity
lambdaD = VT / WP
XP1 = 0.2  # Magnitude of perturbation in x
mode = 2  # Mode of the sin wave in perturbation
Q = WP ** 2 * L / (QM * N)  # Charge of a particle
rho_back = - Q * N / L  # background rho
dx = L / NG  # cell length
k = lambdaD * mode * 2 * np.pi / L
i = 0
xp0 = np.zeros(int(N / mode))
vp0 = VT * np.random.randn(int(N / mode))
while i < int(N / mode):
    x = np.random.rand(1, 2) * np.array([L / mode, N / L * (1 + XP1)])
    if x[0, 1] < N / L * (1 + XP1 * np.sin(2 * np.pi * x[0, 0] / L * mode)):
        xp0[i] = x[0, 0]
        i = i + 1
xp = xp0
vp = vp0
for j in range(mode - 1):
    xp = np.append(xp, xp0 + L * (j + 1) / mode)
    vp = np.append(vp, vp0)
wp = 1
Ek = []
Ep = []
E = []
momentum = []
phiMax = []