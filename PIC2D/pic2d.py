from initialize import *
import time
import matplotlib.pyplot as plt
import energy, interpolate, field, dynamics, figures
import specKernel
t = time.time()
picNum = 0
for it in range(NT):
    print(it)
    xp = dynamics.toPeriodicND(xp, L)
    M = interpolate.interpMatrix(xp, wp)
    rho = interpolate.interpolate(M)
    phi, Eg = field.field(rho)
    #vp, kinetic = dynamics.accelerate(M, Eg, wp)
    a = dynamics.accelerate(M, Eg, wp)
    Egp = energy.energypotx(Eg)
    vp, kinetic = dynamics.push(vp, a, it)
    xp, wp = dynamics.move(xp, vp, wp, it)
    potential = energy.potential(rho, phi)
    Ek.append(kinetic)
    Ep.append(potential)
    E.append(kinetic + potential)
    momentum.append(sum(Q * vp / QM))
    #phiMax.append(np.max(phi))
    Exp.append(Egp)
#figures.field2D(rho)

#dphi = np.sqrt(np.sum((phi)**2) * L[0] * L[1] / (NG[0] * NG[1]))
#dphi = np.sqrt(np.sum((phi)**2) * L[0] * L[1] / (NG * NG))
#print('error = ' + str(dphi))
figures.landauDecayFig(Exp)
figures.energyFig(E,Ek,Ep)
print(time.time()-t)
#print(rho)
#figures.field2D(rho)
#plt.savefig('rho_field2.png')
#plt.show()
