from initialize import *
import time
from sparseGrid import *
import matplotlib.pyplot as plt
import energy, interpolate, field, dynamics, figures
t = time.time()
grid = sparseGrid2D(NG, L)
print(grid.pgrid)
picNum = 0
for it in range(NT):
    print(it)
    phi, rho, a = 0,0, 0
    xp = dynamics.toPeriodicND(xp, L)
    for i in range(grid.logN):
        M = interpolate.interpMatrix(xp, wp, grid.ph[i])
        srho = interpolate.interpolate(M, grid.ph[i])
        sphi, sEg = field.field(srho)
        rho += sparseGrid2D.bilinear(srho, NG)
        phi += sparseGrid2D.bilinear(sphi, NG)
        a += dynamics.accelerate(M, sEg, wp)
    for i in range(grid.logN-1):
        M = interpolate.interpMatrix(xp, wp, grid.nh[i])
        srho = interpolate.interpolate(M, grid.nh[i])
        sphi, sEg = field.field(srho)
        rho -= sparseGrid2D.bilinear(srho, NG)
        phi -= sparseGrid2D.bilinear(sphi, NG)
        a -= dynamics.accelerate(M, sEg, wp)
    kinetic = energy.kinetic(vp)
    vp = vp + a * DT / 2
    xp, wp = dynamics.move(xp, vp, wp, it)
    #potential = energy.potential(rho, phi)
    #Ek.append(kinetic)
    #Ep.append(potential)
    #E.append(kinetic + potential)
    momentum.append(np.sum(Q * vp / QM, axis=1))
    #phiMax.append(np.max(phi))
#figures.field2D(rho)

dphi = np.sqrt(np.sum((phi)**2) * L[0] * L[1] / NG**2)
print('error = ' + str(dphi))
print(time.time()-t)
#print(E)
figures.field2D(phi)
#plt.show()
#figures.landauDecayFig(phiMax)
plt.show()

 