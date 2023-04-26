import numpy as np
import matplotlib.pyplot as plt
import finufft
import scipy.special as sp

## Construct an on-grid analytic solution 
## Define source for Poisson's problem Delta f = -rho

sigma = 1/16 ## Width of Gaussian
L = 2
N = 64
h = L / N

def construct_solution(L, N, sigma):
    x = np.linspace(-L/2, L/2, N, endpoint=False)
    x, y = np.meshgrid(x, x)
    r2 = x**2+y**2
    rho = 1/(2*np.pi*sigma**2) * np.exp(-r2/(2*sigma**2)) ## Distance from the origin
    exact_pot = 1/(4*np.pi)*(sp.expi(-r2/(2*sigma**2))-np.log(r2))
    exact_pot[N//2,N//2] = (np.euler_gamma-np.log(2*sigma**2))/(4*np.pi)
    exact_Ex = -1/(2*np.pi)*(x/r2)*(np.exp(-r2/(2*sigma**2))-1) ## Exact x-component of the electric field
    exact_Ex[N//2,N//2]=0
    exact_Ey = -1/(2*np.pi)*(y/r2)*(np.exp(-r2/(2*sigma**2))-1) ## Exact y-component of the electic field
    exact_Ey[N//2,N//2]=0
    return [rho, exact_pot, exact_Ex, exact_Ey]

def off_grid_solution(X, L, N, sigma):
    x = np.linspace(-L/2, L/2, N, endpoint=False)
    h = L / N
    x, y = np.meshgrid(x, x)
    rho = np.zeros([N,N])
    exact_pot = np.zeros([N,N])
    for i in range(np.shape(X)[1]):
        r2 = (x-X[0,i]) ** 2 + (y-X[1,i]) ** 2
        rho += 1/(2*np.pi*sigma**2) * np.exp(-r2/(2*sigma**2))
        pot = 1/(4*np.pi)*(sp.expi(-r2/(2*sigma**2))-np.log(r2))
        if np.any((X[:, i] + np.array([L/2, L/2])) % h)==0:
            center = ((X[:, i] + np.array([L/2, L/2])) // h).astype(int)
            pot[center[1], center[0]] = (np.euler_gamma-np.log(2*sigma**2))/(4*np.pi)
        exact_pot += pot
    return [rho, exact_pot]

############################################################
## Construct Fourier domain
def free_space_solver(rho, L, sigma=0, nufft=False): # free space poisson solver
    """ 
        Based on the article "Fast convolution with free-space          
        Green's function" by F. Vico, L. Greengard, and M.
        Ferrando, Journal of Computational Physics 323 (2016)
        191-203
        Revised from the Matlab code by Junyi Zou and Antoine Cerfon
        Contact: cerfon@cims.nyu.edu
    """
    if sigma==0: 
        extension = 4
    else:
        extension = 8

    if nufft==False:
        N = np.shape(rho)[0]
    else:
        N = np.shape(rho)[0] // 4

    wm = np.linspace(- N * np.pi / L, N * np.pi / L, extension*N, endpoint=False) ## 4 times finer than regular Fourier step
    wm1, wm2 = np.meshgrid(wm, wm)
    s = np.sqrt(wm1**2 + wm2**2)

    ## Construct mollified Green's function
    LT = 1.5 * L ## Truncation window size
    green = (1-sp.jv(0, LT*s)) / (s**2) - (LT*np.log(LT)*sp.jv(1, LT*s)) / s ## Green function in spectral space
    green[extension*N//2, extension*N//2] = (LT**2/4 - LT**2*np.log(LT)/2)

    if sigma > 0: # Need to convolute with shape function
        l = LT
        S = np.exp(-sigma**2*s**2 / 2) * np.real(sp.erf(l/(np.sqrt(2)*sigma)+1j*s/np.sqrt(2)*sigma)) / (4*L)**2
        green = S * green

    ## Precomputation
    T1 = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(green))) # * deltahat
    T = T1[extension*N//4:extension*N*3//4, extension*N//4:extension*N*3//4]

    ## Free space solution
    if nufft == True:
        phiHat = np.fft.fft2(T) * rho 
    else:
        phiHat = np.fft.fft2(T)*np.fft.fft2(rho, s=[extension*N//2, extension*N//2])
    return phiHat

''' Test 1
phi, S = free_space_solver(rho1, L, Shape=True)
phi = np.real(np.fft.ifft2(phi))[N:2*N, N:2*N]
S = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(S)))
print(S)
plt.subplot(311)
plt.imshow(phi)
plt.colorbar()
plt.title('vico-greengard')
plt.subplot(312)
plt.imshow(exact_pot)
plt.colorbar()
plt.title('analytic')
plt.subplot(313)
#phi2 = phi2[0:N, 0:N]
plt.imshow(exact_pot - phi)
plt.colorbar()
plt.title('error')
plt.show()
'''

## Test 2: On-grid convergence
'''
sigma = 1/16
L = 2
error = []
Ns = np.array(np.arange(16, 96, step=4))
sol = construct_solution(L, 128, sigma)
x = np.linspace(0, 2*np.pi, 512, endpoint=False)
x, y = np.meshgrid(x, x)
x = x.flatten()
y = y.flatten()
print(x, y)
for N in Ns:
    rho1 = np.zeros([N,N])
    rho1[N//2, N//2] = 1
    rho = construct_solution(L, N, sigma)[0]
    phiHat = free_space_solver(rho1, L, sigma)
    phi2 = np.fft.ifft2(phiHat)
    phi = np.fft.ifft2(free_space_solver(rho, L))
    phi1 = np.real(finufft.nufft2d2(x, y, np.conjugate(phiHat), eps=10**-14, modeord=1)).reshape([512, 512]) / N**2
    plt.subplot(131)
    plt.imshow(np.real(phi1))
    plt.colorbar()
    plt.title('Vico-Greengard')
    plt.subplot(132)
    plt.imshow(np.real(phi2))
    plt.colorbar()
    plt.title('Fourier Interpolant of Vico-Greengard')
    plt.subplot(133)
    phi1 = phi1[256:384, 256:384] / 16
    plt.imshow(np.abs(sol[1]-phi1))
    plt.colorbar()
    plt.title('Analytic')
    plt.show()
    error.append(2/128 * np.sqrt(np.sum((phi1-sol[1]) **2)))
plt.loglog(np.array(np.arange(16, 96, step=4)), error, marker='o', label=r'$L_2$ norm error')
plt.yscale('log')
plt.loglog(np.linspace(16, 96, 4), np.linspace(16, 96, 4)**(-3), label=r'$3^{nd}$ order reference')
plt.legend()
plt.grid(True, which='both', ls='--', alpha=0.6)
plt.xlabel('Number of modes in each dimension')
plt.ylabel(r'$\varepsilon(\phi)$')
plt.savefig('free_space_convergence2.png', dpi=300)
'''



'''
Next Steps: 
1. Generate point charges off grid within the domain, 
see if the convolutions with shape function & Green's 
function are translation-invariant;
2. Implement the solver in Particle-In-Fourier method,
and test the convergence of the free-space solution 
of the Vlasov-Poisson system, as well as conservation
properties.
'''

def offgrid_fs_solve(Xs, L, N, sigma):
    ''' This is the function that returns the Fourier space of the potential, from the distribution of off-grid particles
        
        Some parameters:
            Xs: the positions of the particles
            L: length of a box side
            N: number of Fourier modes
            sigma: width of the Gaussian 
        
        Returns:
            phiHat: the Fourier space of the potential phi (if want to retrieve phi then let phi=np.fft.ifft(phiHat)[2*N:3*N, 2*N:3*N])
    '''
    
    raw = finufft.nufft2d1(Xs[1, :] * np.pi / (2*L) + np.pi / 4, Xs[0, :] * np.pi / (2*L) + np.pi / 4, 0j + np.ones(Np), (4*N, 4*N), eps=1E-14, modeord=1)
    ## Please note that finufft has a very strange bug, that its computed Fourier modes values are COMPLEX CONJUGATES of the true values
    phiHat = free_space_solver(np.conjugate(raw), L, sigma, nufft=True)
    return phiHat

error1 = []
error2 = []
errorinf = []
L = 2
Np = 10
Ns = np.array(np.arange(16, 96, step=4))
Xs = np.random.rand(2, Np) * L - L/2
#Xs = np.array([[2/64],[0]])
sigma = 1/16
x = np.linspace(0, 2*np.pi, 512, endpoint=False)
x, y = np.meshgrid(x, x)
x = x.flatten()
y = y.flatten()
rho, exact_pot = off_grid_solution(Xs, L, 128, sigma)
for N in Ns:
    phiHat = offgrid_fs_solve(Xs, L, N, sigma)
    phi = np.real(finufft.nufft2d2(y, x, np.conjugate(phiHat), eps=10**-14, modeord=1)).reshape([512, 512])
    plt.subplot(131)
    plt.imshow(phi)
    plt.subplot(132)
    plt.imshow(exact_pot)
    plt.subplot(133)
    plt.imshow(exact_pot - phi[256:384, 256:384])
    plt.show()
    error = np.abs(phi[256:384, 256:384]-exact_pot)
    error1.append((2/128) ** 2 * np.sum(error))
    error2.append(2/128 * np.sqrt(np.sum(error **2)))
    errorinf.append(np.max(error))

plt.plot(np.array(np.arange(16, 96, step=4)), error1, marker='o', label=r'$L_1$ norm error')
plt.plot(np.array(np.arange(16, 96, step=4)), error2, marker='+', label=r'$L_2$ norm error')
plt.plot(np.array(np.arange(16, 96, step=4)), errorinf, marker='*', label=r'$L_\infty$ norm error')
plt.yscale('log')
plt.legend()
plt.grid(True, which='both', ls='--', alpha=0.6)
plt.xlabel('Number of modes in each dimension')
plt.ylabel(r'$\varepsilon(\phi)$')
plt.show()
