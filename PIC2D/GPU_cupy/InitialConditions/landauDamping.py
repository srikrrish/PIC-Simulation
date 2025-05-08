import numpy as np
import matplotlib.pyplot as plt

def findsource():
    return None


def InvTransSampling(XP1,k,L,N):
    i = 0
    xp = np.zeros([2, N])
    xp[1] = np.random.rand(N) * L[1]
    while i < N:
       x = np.random.rand(1, 2) * np.array([L[0], N / L[0] * (1 + XP1)])
       if x[0, 1] < N / L[0] * (1 + XP1 * np.sin(2 * np.pi * x[0, 0] / L[0] * mode)):
           xp[0, i] = x[0, 0]
           i = i + 1

    return xp


DT = .05  # Length of a time step
T = 20
NT = int(T/DT)  # number of time steps
NG = 32 # Number of Grid points
N = 100000  # Number of simulation particles
QM = -1  # charge per mass
VT = 1  # Thermal Velocity
wp = 1
