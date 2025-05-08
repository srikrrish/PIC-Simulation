import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
from initialize import dx, NT, DT, Q, VT, NG, k
import landauDecay, dynamics

def phaseSpace(xp, vp, wp):
    g1 = np.floor(xp / dx).astype(int)  # which grid point to project onto
    g = np.array([g1 - 1, g1, g1 + 1])
    g = g[:, np.abs(vp) < 10 * VT]
    g = dynamics.toPeriodic(g, NG, True)
    delta = xp % dx
    fraz = np.array([(1 - delta) ** 2 / 2, 1 - ((1 - delta) ** 2 / 2 + delta ** 2 / 2), delta ** 2 / 2] * wp)
    fraz = fraz[:, np.abs(vp) < 10 * VT]
    vp = vp[np.abs(vp) < 10 * VT]
    col = (vp + 10 * VT) // (20 * VT / 128)
    col = col.astype(int)
    mat = sparse.csr_matrix((- fraz[0] * Q, (col, g[0]))) + sparse.csr_matrix((- fraz[1] * Q, (col, g[1]))) +sparse.csr_matrix((- fraz[2] * Q, (col, g[2])))
    mat = mat.todense()
    print(mat.ndim, np.zeros([128 - mat.shape[0], mat.shape[1]]).ndim)
    mat = np.append(mat, np.zeros([128 - mat.shape[0], mat.shape[1]]), axis=0)
    plt.imshow(mat, vmin=0, vmax=np.max(mat), cmap='plasma', interpolation="nearest")
    plt.colorbar()
    plt.axis('off')


def energyFig(E, Ek=None, Ep=None):
    plt.plot(np.linspace(0, NT * DT, NT), E / E[0], label='Total Energy')
    if not Ek is None:
        plt.plot(np.linspace(0, NT * DT, NT), Ek / E[0], label='Kinetic Energy')
    if not Ep is None:
        plt.plot(np.linspace(0, NT * DT, NT), Ep / E[0], label='Potential Energy')
    plt.legend()
    plt.ylabel('Normalized Energy', fontsize='14')
    plt.xlabel('$\omega_p$t', fontsize='14')

def landauDecayFig(phiMax):
    a = np.linspace(0, (NT - 1) * DT, NT)
    plt.plot(a, phiMax, label='$\phi_{max}$')
    pp = landauDecay.period(k)
    b = phiMax[int(pp // (2 * DT))] * np.exp((a[0:2000] - pp / 2) * landauDecay.decayRate(k))
    plt.plot(a[0:2000], b, label='predicted decay rate', color='seagreen')
    plt.title('Landau Damping Decay Rate, k=$\pi$/8', fontsize='14')
    plt.yscale('log')
    plt.ylabel('$L^{\infty}(\Phi)$', fontsize='14')
    plt.xlabel('normalized time unit: $\omega_p$t', fontsize='14')
    plt.legend()
    plt.grid(color='gray')


def landauDecayFigIppl(Ex):
    a = np.linspace(0, (NT - 1) * DT, NT)
    plt.plot(a, Ex, label='$\int E_x^2 dV$')
    gamma = -0.3066
    ind = np.argmin(np.abs(a - 2.5))
    theo_ref = np.exp(gamma * a)
    theo_ref = (Ex[ind]/theo_ref[ind])*theo_ref
    plt.plot(a, theo_ref, label='predicted decay rate', color='seagreen')
    #plt.title('Landau Damping Decay Rate, k=0.5', fontsize='14')
    plt.yscale('log')
    plt.ylabel('$\int E_x^2 dV$', fontsize='14')
    plt.xlabel('normalized time unit: $\omega_p$t', fontsize='14')
    plt.legend()
    plt.grid(color='gray')
    plt.savefig('landau_decay_rate.png')
    plt.clf()


def conservationErrors(E,M):
    plt.plot(np.linspace(0, NT * DT, NT), np.abs(E - E[0]) / np.abs(E[0]), label='Energy')
    plt.plot(np.linspace(0, NT * DT, NT), np.abs(M - M[0]) / np.abs(M[0]), label='Momentum')
    plt.yscale('log')
    plt.legend()
    plt.ylabel('Rel. error', fontsize='14')
    plt.xlabel('$\omega_p$t', fontsize='14')
    plt.grid(color='gray')
    #plt.show()
    plt.savefig('landau_conservation_errors_ref.png')
    plt.clf()
