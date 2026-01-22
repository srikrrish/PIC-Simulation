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
    plt.savefig('conservation_errors.png')
    plt.clf()

def landauDecayFigIppl(Ex,label='weak'):
    a = np.linspace(0, (NT - 1) * DT, NT)
    plt.plot(a, Ex, label='$\int E_x^2 dV$')
    if(label == 'weak'):
        gamma1 = -0.3066
    else:
        gamma1 = -0.562
        gamma2 = 0.168
        ind2 = np.argmin(np.abs(a - 20.592))
        theo_ref2 = np.exp(gamma2 * a)
        theo_ref2 = (Ex[ind2]/theo_ref2[ind2])*theo_ref2


    ind1 = np.argmin(np.abs(a - 2.5))
    theo_ref1 = np.exp(gamma1 * a)
    theo_ref1 = (Ex[ind1]/theo_ref1[ind1])*theo_ref1
    plt.plot(a, theo_ref1, label='predicted decay rate', color='seagreen')
    if(label == 'strong'):
        plt.plot(a, theo_ref2, label='predicted growth rate', color='red')
    #plt.title('Landau Damping Decay Rate, k=0.5', fontsize='14')
    plt.yscale('log')
    if(label == 'strong'):
        ax = plt.gca()
        ax.set_ylim([1e-3,1e3])
        #ax.set_ylim([1e-5,1e3])
    else:
        ax = plt.gca()
        #ax.set_ylim([1e-3,1e3])
        ax.set_ylim([1e-5,1])

    plt.ylabel('$\int E_x^2 dV$', fontsize='14')
    plt.xlabel('normalized time unit: $\omega_p$t', fontsize='14')
    plt.legend()
    plt.grid(color='gray')
    plt.savefig('landau_decay_rate.png')
    plt.clf()


def twostreamFigIppl(Ex,label='tsi'):
    a = np.linspace(0, (NT - 1) * DT, NT)
    plt.plot(a, Ex, label='$\int E_x^2 dV$')
    if(label == 'tsi'):
        gamma = 0.4952
    else:
        gamma = 0.356

    ind = np.argmin(np.abs(a - 8.0))
    theo_ref = np.exp(gamma * a)
    theo_ref = (Ex[ind]/theo_ref[ind])*theo_ref
    plt.plot(a, theo_ref, label='predicted growth rate', color='seagreen')
    #plt.title('Landau Damping Decay Rate, k=0.5', fontsize='14')
    plt.yscale('log')
    ax = plt.gca()
    if(label == 'tsi'):
        ax.set_ylim([1e-4,1e3])
    else:
        ax.set_ylim([1e-2,1e3])
    plt.ylabel('$\int E_y^2 dV$', fontsize='14')
    plt.xlabel('normalized time unit: $\omega_p$t', fontsize='14')
    plt.legend()
    plt.grid(color='gray')
    plt.savefig('growth_rate.png')
    plt.clf()
