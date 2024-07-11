import numpy as np

def findsource(xp, vp, L, it, DT):
    S = 4*np.pi**2*(1/L[0]**2+1/L[1]**2)
    P = 1 / (S*L[0]*L[1])
    source1 = - 4 * np.pi ** 3 * f0 * P * np.cos(np.pi * it * DT) * (np.sin(2*np.pi*xp[0]/L[0])/L[0]**2+np.cos(2*np.pi*xp[1]/L[1])/L[1]**2)
    source2 = - 8 * np.pi ** 3 * f0 * P * np.sin(np.pi * it * DT) * (np.cos(2*np.pi*xp[0]/L[0])*vp[0]/L[0]**3-np.sin(2*np.pi*xp[1]/L[1])*vp[1]/L[1]**3)
    E1 = -N*Q*2*np.pi*P*np.sin(np.pi * it * DT)*np.cos(2*np.pi*xp[0]/L[0])/L[0]
    E2 = N*Q*2*np.pi*P*np.sin(np.pi * it * DT)*np.sin(2*np.pi*xp[1]/L[1])/L[1]
    source3 = QM * (8*np.exp(-np.sum(vp**2, axis=0))/(np.pi*L[0]*L[1]))*((vp[0]-vp[0]**3)*vp[1]**2*E1+(vp[1]-vp[1]**3)*vp[0]**2*E2)*(P*S-P*np.sin(np.pi * it * DT)*4*np.pi**2*(np.sin(2*np.pi*xp[0]/L[0])/L[0]**2+np.cos(2*np.pi*xp[1]/L[1])/L[1]**2))
    return (source1 + source2 + source3) / f0

L = np.array([10, 10]) # Length of the container
DT = .005  # Length of a time step
T = 2
NT = int(T/DT)  # number of time steps
print(NT)
NG = 64 # Number of Grid points
N = 32000  # Number of simulation particles
WP = 1  # omega p
DT = DT * WP
QM = -1  # charge per mass
VT = 1  # Thermal Velocity
lambdaD = VT / WP
L = L / lambdaD
Q = WP ** 2 * L[0] * L[1] / (QM * N)  # Charge of a particle
rho_back = - Q * N / (L[0] * L[1])  # background rho
dx = L / NG  # cell length
xp = (np.random.rand(2, N).T * L).T
print(xp)
vp = np.zeros([2, N])
for j in range(2):
    i = 0
    while i < N:
        v = (np.random.rand(2) + np.array([-0.5, 0])) * np.array([10, 1])
        if v[1] < 2 * v[0] ** 2 * np.exp(-v[0]**2) / np.sqrt(np.pi):
            vp[j, i] = v[0]
            i = i + 1
wp = np.ones(N)
f0 = 4 * vp[0] ** 2 * vp[1] ** 2 * np.exp(-np.sum(vp**2, axis=0))
# Energy
Ek = []  # Kinetic Energy
Ep = []  # Potential Energy
E = []  # Total Energy
momentum = []
PhiMax = []
