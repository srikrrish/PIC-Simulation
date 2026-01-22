from initialize import *
import matplotlib.pyplot as plt
import energy, interpolate, field, dynamics, figures, specKernel
import numpy as np

#picNum = 0
Shat, K = specKernel.specKernel()
pos = np.zeros([NT, N], dtype='f');
#vel = np.zeros([NT, N], dtype='f');
Eout = np.zeros([NT, N], dtype='f');
for it in range(NT):
    print(it)
    xp = dynamics.toPeriodic(xp, L)
    #if it % 25 == 1 and picNum < 16:
    #    picNum = picNum + 1
    #    plt.subplot(4, 4, picNum)
    #    figures.phaseSpace(xp, vp)
    #    plt.title('$t$=%s' % str(np.round(it * DT, 4)))
    rhoHat = interpolate.specInterpolate(xp, Shat, wp)
    phiHat, EgHat = field.fieldInFourier(rhoHat)
    pos[it,:] = xp.astype(np.float32)
    #vel[it,:] = vp.astype(np.float32)
    a, Eout = dynamics.accelInFourier(vp, xp, it, EgHat, Eout, Shat, wp)
    vp, kinetic = dynamics.push(vp, a, it)
    xp, wp = dynamics.move(xp, vp, wp)
    #potential = energy.potential(rhoHat, phiHat)
    #potential = energy.specPotential(rhoHat, phiHat)
    Egp = np.sum(Eout[it,:]**2) * L / N
    potential = Egp * 0.5
    #Egp = 2 * potential
    Ek.append(kinetic)
    Ep.append(potential)
    E.append(kinetic + potential)
    Exp.append(Egp)
    momentum.append(np.abs(np.sum(Q * vp / QM)))
    #phiMax.append(np.max(np.fft.ifft(phiHat) * NG / L))
#plt.show()
#figures.landauDecayFig(phiMax)
#plt.show()
figures.landauDecayFigIppl(Exp,'weak')
#figures.twostreamFigIppl(Exp,'tsi')
figures.conservationErrors(E,momentum)
np.save('data/pos_weakLandau_50k',pos)
#np.save('data/vel_weakLandau',vel)
np.save('data/Eout_weakLandau_50k',Eout)
