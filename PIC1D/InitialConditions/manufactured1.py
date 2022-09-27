import numpy as np

def findsource(xp, vp, L, it, DT):
    E = Q * N * np.sin(np.pi * it * DT / 4) * np.cos(2 * np.pi * xp / L) / (4 * np.pi)
    S1 = - 0.5 * (np.sqrt(np.pi / 2) / L) * np.exp(- vp ** 2 / 2) * np.cos(np.pi * it * DT / 4) * np.sin(2 * np.pi * xp / L) / 4
    S2 = - 0.5 * (np.sqrt(2 * np.pi) * vp / L ** 2) * np.exp(- vp ** 2 / 2) * np.sin(np.pi * it * DT / 4) * np.cos(
        2 * np.pi * xp / L)
    S3 = - (QM * E / L) * (1 - 0.5 * np.sin(np.pi * it * DT / 4) * np.sin(2 * np.pi * xp / L)) * (np.exp(- vp ** 2 / 2) * vp / np.sqrt(2 * np.pi))
    return (S1 + S2 + S3) * Q * N / f0


L = 2 * np.pi # Length of the container
DT = .01  # Length of a time step
NT = 800  # number of time steps
NG = 128  # Number of Grid points
N = 200000  # Number of simulation particles
WP = 1  # omega p
DT = DT * WP
QM = -1  # charge per mass
VT = 1  # Thermal Velocity
lambdaD = VT / WP
L = L / lambdaD
Q = WP ** 2 * L / (QM * N)  # Charge of a particle
rho_back = - Q * N / L  # background rho
dx = L / NG  # cell length
xp = np.linspace(0, L, N, endpoint=False)
vp = np.random.randn(N)
wp = np.ones(N)
f0 = N * Q * np.exp(- vp ** 2 / 2) * np.sqrt(1 / (2 * np.pi)) / L
# Energy
Ek = []  # Kinetic Energy
Ep = []  # Potential Energy
E = []  # Total Energy
momentum = []
PhiMax = []
