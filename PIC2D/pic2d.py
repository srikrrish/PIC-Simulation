from initialize import *
import matplotlib.pyplot as plt
import energy, interpolate, field, dynamics, figures

picNum = 0
for it in range(NT):
    print(it)
    xp = dynamics.toPeriodicND(xp, L)
    if it % 25 == 1 and picNum < 16:
        picNum = picNum + 1
        plt.subplot(4, 4, picNum)
        figures.phaseSpace(xp, vp, wp)
        plt.title('$t$=%s' % str(np.round(it * DT, 4)))
    M = interpolate.interpMatrix(xp, wp)
    rho = interpolate.interpolate(M)
    phi, Eg = field.field(rho)
    vp, kinetic = dynamics.accelerate(vp, it, M, Eg, wp)
    xp, wp = dynamics.move(xp, vp, wp, it)
    potential = energy.potential(rho, phi)
    Ek.append(kinetic)
    Ep.append(potential)
    E.append(kinetic + potential)
    momentum.append(sum(Q * vp / QM))
    #phiMax.append(np.max(phi))
plt.show()
#figures.landauDecayFig(phiMax)
#plt.show()