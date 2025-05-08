import numpy as np
import cupy as cp
from scipy import sparse
import matplotlib.pyplot as plt
from initialize import NT, DT, VT
import landauDecay, dynamics
import mpl_toolkits.mplot3d




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

def landauDecayFigIppl(Ex, k):
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
    plt.savefig('landau_decay_rate_k_'+str(k[0])+'.png')
    plt.clf()
