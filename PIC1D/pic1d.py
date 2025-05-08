from initialize import *
import matplotlib.pyplot as plt
import energy, interpolate, field, dynamics, figures
import numpy as np

#picNum = 0
pos = np.zeros([NT, N], dtype='f');
Eout = np.zeros([NT, N], dtype='f');
for it in range(NT):
    print(it)
    xp = dynamics.toPeriodic(xp, L)
    #if it % 25 == 1 and picNum < 16:
    #    picNum = picNum + 1
    #    plt.subplot(4, 4, picNum)
    #    figures.phaseSpace(xp, vp, wp)
    #    plt.title('$t$=%s' % str(np.round(it * DT, 4)))
    M = interpolate.interpMatrix(xp, wp)
    rho = interpolate.interpolate(M)
    phi, Eg = field.field(rho)
    pos[it,:] = xp.astype(np.float32)
    a, Eout = dynamics.accelerate(it, M, Eg, Eout, wp)
    vp, kinetic = dynamics.push(vp, a, it)
    xp, wp = dynamics.move(xp, vp, wp, it)
    potential = energy.potential(rho, phi)
    Egp = np.sum(Eout[it,:]**2) * L / N
    Ek.append(kinetic)
    Ep.append(potential)
    E.append(kinetic + potential)
    Exp.append(Egp)
    momentum.append(np.abs(np.sum(Q * vp / QM)))
    #phiMax.append(np.max(phi))
#plt.show()
#figures.landauDecayFig(phiMax)
#plt.show()
figures.landauDecayFigIppl(Exp)
figures.conservationErrors(E,momentum)
np.save('data/pos',pos)
np.save('data/Eout',Eout)

