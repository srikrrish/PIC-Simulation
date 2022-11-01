from initialize import *
import initialize
import time
import matplotlib.pyplot as plt
import energy, interpolate, field, dynamics, figures, specKernel

picNum = 0
Shat = specKernel.specKernel(NG, L)
t = time.time()
for it in range(NT):
    print(it)
    xp = dynamics.toPeriodicND(xp, L)
   # if it % 25 == 1 and picNum < 16:
    #    picNum = picNum + 1
    #    plt.subplot(4, 4, picNum)
    #    figures.phaseSpace(xp[0], vp[0], wp, NG[0])
    #    plt.title('$t$=%s' % str(np.round(it * DT, 4)))
    rhoHat = interpolate.specInterpolate(xp, Shat, wp)
    phiHat, EgHat = field.fieldInFourier(rhoHat)
    a = dynamics.accelInFourier(xp, EgHat, Shat, wp)
    if it > 0:
        vp = vp + a * DT / 2
    kinetic = energy.kinetic(vp)
    vp = vp + a * DT / 2
    xp, wp = dynamics.move(xp, vp, wp)
    potential = energy.specPotential(rhoHat, phiHat)
    Ek.append(kinetic)
    Ep.append(potential)
    E.append(kinetic + potential)
    momentum.append(np.sum(Q * vp / QM, axis=1))
    phiMax.append(np.max(np.fft.ifft2(phiHat) * NG[0] * NG[1] / (L[0] * L[1])))
print(time.time()-t)
figures.energyFig(E)
plt.show()
figures.landauDecayFig(phiMax)
plt.show()
