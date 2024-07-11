import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
from initialize import dx, NT, DT, Q, VT, k
import landauDecay, dynamics
import mpl_toolkits.mplot3d

def phaseSpace(xp, vp, wp, NG):
    g1 = np.floor(xp / dx[0]).astype(int)  # which grid point to project onto
    g = np.array([g1 - 1, g1, g1 + 1])
    g = g[:, np.abs(vp) < 10 * VT]
    g = dynamics.toPeriodic(g, NG, True)
    delta = xp % dx[0]
    fraz = np.array([(1 - delta) ** 2 / 2, 1 - ((1 - delta) ** 2 / 2 + delta ** 2 / 2), delta ** 2 / 2] * wp)
    fraz = fraz[:, np.abs(vp) < 10 * VT]
    vp = vp[np.abs(vp) < 10 * VT]
    col = (vp + 10 * VT) // (20 * VT / 128)
    col = col.astype(int)
    mat = sparse.csr_matrix((- fraz[0] * Q, (col, g[0]))) + sparse.csr_matrix((- fraz[1] * Q, (col, g[1]))) +sparse.csr_matrix((- fraz[2] * Q, (col, g[2])))
    mat = mat.todense()
    mat = np.append(mat, np.zeros([128 - mat.shape[0], mat.shape[1]]), axis=0)
    plt.imshow(mat, vmin=0, vmax=np.max(mat), cmap='plasma', interpolation="nearest")
    #plt.colorbar()
    plt.axis('off')

def energyFig(E, Ek=None, Ep=None):
    plt.plot(np.linspace(0, NT * DT, NT), E / E[0], label='Total Energy')
    if not Ek is None:
        plt.plot(np.linspace(0, NT * DT, NT), Ek / E[0], label='Kinetic Energy')
    if not Ep is None:
        plt.plot(np.linspace(0, NT * DT, NT), Ep / E[0], label='Potential Energy')
    plt.yscale('log')
    plt.legend()
    plt.ylabel('Normalized Energy', fontsize='14')
    plt.xlabel('$\omega_p$t', fontsize='14')
    plt.grid(color='gray')
    #plt.show()
    plt.savefig('landau_energy_1e6_32.png')
    plt.clf()

def landauDecayFig(phiMax):
    a = np.linspace(0, (NT - 1) * DT, NT)
    #plt.plot(a, phiMax, label='$\phi_{max}$')
    plt.plot(a, phiMax, label='$\int E_x^2 dV$')
    pp = landauDecay.period(k)
    b = phiMax[int(pp // (2 * DT))] * np.exp((a[0:2000] - pp / 2) * landauDecay.decayRate(k))
    plt.plot(a[0:2000], b, label='predicted decay rate', color='seagreen')
    #plt.title('Landau Damping Decay Rate, k=$\pi$/8', fontsize='14')
    plt.title('Landau Damping Decay Rate, k=0.5', fontsize='14')
    plt.yscale('log')
    #plt.ylabel('$L^{\infty}(\Phi)$', fontsize='14')
    plt.ylabel('$\int E_x^2 dV$', fontsize='14')
    plt.xlabel('normalized time unit: $\omega_p$t', fontsize='14')
    plt.legend()
    plt.grid(color='gray')
    #plt.show()
    plt.savefig('landau_decay_rate_1e6_32.png')
    plt.clf()

def field2D(field):
    from initialize import NG
    if isinstance(NG, int):
        NG=[NG,NG]
    x = np.linspace(0, NG[0]-1, NG[0]).astype(int).tolist()
    y = np.linspace(0, NG[0]-1, NG[0]).astype(int).tolist()
    values = []
    fig = plt.figure()
    ax = mpl_toolkits.mplot3d.Axes3D(fig)
    x, y = np.meshgrid(x, y)
    ax.plot_surface(x, y, field, cmap='cool')
    #plt.show()
    fig.savefig('rho_field.png')
