from initialize import *
from sparseGrid import sparseGrid2D
import matplotlib.pyplot as plt
import energy, interpolate, field, dynamics, figures, specKernel
import time

picNum = 0
grid = sparseGrid2D(NG, L)
t = time.time()
for it in range(NT):
    print(it)
    xp = dynamics.toPeriodicND(xp, L)
    phiHat, rhoHat, a = 0,0, 0
    for i in range(grid.logN):
        [n1, n2] = grid.pgrid[i]
        srhoHat = interpolate.specInterpolate(xp, grid.pShats[i], wp, [n1,n2])
        sphiHat, sEgHat = field.fieldInFourier(srhoHat)
        kn = np.zeros([int(NG/n1), int(NG/n2)])
        kn[0, 0] = 1
        srhoHat, sphiHat = np.kron(kn,np.fft.fftshift(srhoHat)), np.kron(kn, np.fft.fftshift(sphiHat))
        rhoHat += np.roll(srhoHat, (int(-n1/2), int(-n2/2)), axis=(0,1))
        phiHat += np.roll(sphiHat, (int(-n1/2), int(-n2/2)), axis=(0,1))
        a += dynamics.accelInFourier(xp, sEgHat, grid.pShats[i], wp)
    for i in range(grid.logN-1):
        [n1,n2] = grid.ngrid[i]
        srhoHat = interpolate.specInterpolate(xp, grid.nShats[i], wp, [n1,n2])
        sphiHat, sEgHat = field.fieldInFourier(srhoHat)
        kn = np.zeros([int(NG/n1), int(NG/n2)])
        kn[0, 0] = 1
        srhoHat, sphiHat = np.kron(kn, np.fft.fftshift(srhoHat)), np.kron(kn, np.fft.fftshift(sphiHat))
        rhoHat -= np.roll(srhoHat, (int(-n1 / 2), int(-n2 / 2)), axis=(0,1))
        phiHat -= np.roll(sphiHat, (int(-n1 / 2), int(-n2 / 2)), axis=(0,1))
        a-= dynamics.accelInFourier(xp, sEgHat, grid.nShats[i], wp)
    if it > 0:
        vp = vp + a * DT / 2
    kinetic = energy.kinetic(vp)
    vp = vp + a * DT / 2
    xp, wp = dynamics.move(xp, vp, wp, it)
    potential = energy.specPotential(rhoHat, phiHat)
    Ek.append(kinetic)
    Ep.append(potential)
    E.append(kinetic + potential)
    momentum.append(np.sum(Q * vp / QM, axis=1))
    if it % 25 ==0:
        rho = np.fft.ifft2(rhoHat) * NG**2 / (L[0] * L[1])
        plt.subplot(4,4,int(it//25+1))
        plt.imshow(np.real(rho))
    #phiMax.append(np.max(np.fft.ifft2(phiHat) * NG**2 / (L[0] * L[1])))
phi = np.fft.ifft2(phiHat) * NG**2 / (L[0] * L[1])
rho = np.fft.ifft2(rhoHat) * NG**2 / (L[0] * L[1])
#figures.field2D(rho)
dphi = np.sqrt(np.sum((phi)**2) * L[0] * L[1] / NG**2)
print('error = ' + str(dphi))
print(time.time()-t)
figures.field2D(phi)
plt.show()

#figures.energyFig(momentum)
#figures.landauDecayFig(phiMax)
plt.show()