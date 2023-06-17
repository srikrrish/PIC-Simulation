import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
import finufft
import matplotlib.animation as animation
fig, ax = plt.subplots()
artist = []
L = 1
N = 40000
NG = 128
QM = -1
WP = 1  # omega p
VT = 1  # Thermal Velocity
lambdaD = VT / WP
Q = -0.5 / N  # Charge of a particle
# self.rho_back = - self.Q * self.N / self.L  # background rho
dx = L / NG
B0 = np.array([0,0,300])

sigmas = np.array([[1/10],[1/30]]) / np.sqrt(2)
# plt.scatter(XP[0,:], XP[1,:], alpha=0.1)
# plt.xlim([-0.5, 0.5])
# plt.ylim([-0.5, 0.5])
# plt.show()

# Prepare for convolution kernel:
extension = 8
wm = np.linspace(- NG * np.pi / L, NG * np.pi / L, extension*NG, endpoint=False) ## 8 times finer than regular Fourier step
wm1, wm2 = np.meshgrid(wm, wm)
s = np.sqrt(wm1**2 + wm2**2)

## Construct mollified Green's function
LT = 1.5 * L ## Truncation window size
green = (1-sp.jv(0, LT*s)) / (s**2) - (LT*np.log(LT)*sp.jv(1, LT*s)) / s ## Green function in spectral space
green[extension*NG//2, extension*NG//2] = (LT**2/4 - LT**2*np.log(LT)/2)

r = L / NG
J = np.fft.fftshift(wm) * np.ones([NG * extension, 1])
Kabsolute = np.transpose(np.sqrt(J**2 + np.transpose(J)**2))
Kabsolute[0,0] = 1  # avoid 0 on denominator
Shat = (2 * sp.j1(r * Kabsolute) / (r * Kabsolute)) ** 2 
Shat[0, 0] = 1
Shat = Shat * (L / NG) ** 2 / (r **2)

green1 = Shat * np.fft.fftshift(green)
green2 = Shat ** 2 * np.fft.fftshift(green)

## Precomputation
T1 = np.fft.ifftshift(np.fft.ifft2(green1)) # * deltahat
T1 = T1[extension*NG//4:extension*NG*3//4, extension*NG//4:extension*NG*3//4]
T1 = np.fft.fft2(T1) # This is the kernel for potential field

T2 = np.fft.ifftshift(np.fft.ifft2(green2)) # * deltahat
T2 = T2[extension*NG//4:extension*NG*3//4, extension*NG//4:extension*NG*3//4]
T2 = np.fft.fft2(T2) # This is the kernel for acceleration

dE = []

for DT in [0.001]: #[0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]:
    Eks = [] # Kinetic Energy
    Eps = [] # Potential Energy
    Etotals = [] # Total Energy
    XP = np.random.randn(2, N) * sigmas
    VP = np.random.rand(2, N)
    NT = int(0.5 // DT)  # number of time steps
    for clock in range(NT):
        '''
        1. Solve the electric field using Vico-Greengard and evaluate at particle positions using NUFFT
        
        2. Push the particles using leapfrog
        '''
        # NUFFT Type 1 to evaluate exp(-ikX)
        raw = finufft.nufft2d1(XP[0, :] * np.pi / (2*L), XP[1, :] * np.pi / (2*L), 0j + np.ones(N), (4*NG, 4*NG), eps=1E-14, modeord=1)
        
        # Compute Electric Field
        rho_Hat = np.conjugate(raw) * Shat[::2,::2] * Q
        phi_Hat = Q * T1 * np.conjugate(raw)
        E0 = Q * T2 * np.conjugate(raw) * -1j * np.transpose(J)[::2, ::2]
        E1 = Q * T2 * np.conjugate(raw) * -1j * J[::2, ::2] 

        # Compute Acceleration due to Electric Field
        coeff1 = np.conjugate(E0 * Shat[::2, ::2])
        a1 = np.array(np.real(finufft.nufft2d2(XP[0, :] * np.pi / (2*L) + np.pi, XP[1, :] * np.pi / (2*L) + np.pi, coeff1, eps=1e-14, modeord=1) * QM))
        coeff2 = np.conjugate(E1 * Shat[::2, ::2])
        a2 = np.array(np.real(finufft.nufft2d2(XP[0, :] * np.pi / (2*L) + np.pi, XP[1, :] * np.pi / (2*L) + np.pi, coeff2, eps=1e-14, modeord=1) * QM))
        if N==1:
            a = np.array([[a1], [a2]])
        else:
            a = np.array([a1, a2])

        # Compute Acceleration due to Magnetic Field using Boris Algorithm
        if not clock==0:
            Vm = VP + a * DT / 2
            Vprime = Vm + np.cross(Vm, B0, axisa=0)[:, 0:2].T * QM * DT / 2
            Vp = Vm + np.cross(Vprime, B0, axisa=0)[:, 0:2].T * QM * DT / (1 + (np.linalg.norm(B0)*QM*DT/2) ** 2)
            new_VP = Vp + a * DT / 2
            Ek = 0.5 * Q / QM * np.sum((VP ** 2 + new_VP ** 2) / 2)
            VP = new_VP
        else:
            Ek = 0.5 * Q / QM * np.sum(VP ** 2)
            VP = VP + DT * (a + QM * np.cross(VP, B0, axisa=0)[:, 0:2].T) / 2
        XP = XP + DT * VP

        ## Compute Energy
        Eks.append(Ek)
        rho = np.fft.fftshift(np.real(np.fft.ifft2(rho_Hat)))
        Ep = np.sum(np.fft.fft2(rho) * np.conjugate(phi_Hat) / (2 * L ** 2))
        Eps.append(Ep)
        Etotals.append(Ep + Ek)

        if clock%40==0:
            print(clock)
            container = ax.imshow(-rho[int(1.5*NG):int(2.5*NG), int(1.5*NG):int(2.5*NG)])
            # fig.colorbar(container)
            artist.append([container])
    dE.append(np.sum(np.abs(np.diff(Etotals / Etotals[0]))))
ani = animation.ArtistAnimation(fig=fig, artists=artist, interval=40)
ani.save(filename="test.gif", writer="Pillow")
tick = np.linspace(0, clock*DT, clock+1, endpoint=False)
plt.clf()
plt.plot(tick, Etotals, label='Total Energy')
#plt.plot(tick, Eks, label='Kinetic Energy')
#plt.plot(tick, Eps, label='Potential Energy')
plt.xlim(0, 0.5)
#plt.loglog([0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002], dE, label='Energy Total Variation')
#plt.legend()
plt.show()