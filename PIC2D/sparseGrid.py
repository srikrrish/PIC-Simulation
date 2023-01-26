import numpy as np
import specKernel
import matplotlib.pyplot as plt
class sparseGrid2D:
    pgrid = []
    ngrid = []
    ph = []
    nh = []
    pShats = []
    nShats = []
    logN = 0
    N = 0

    def __init__(self, N, L):
        self.N = N
        self.logN = np.log2(N).astype(int)
        self.pgrid = 2**np.array([np.arange(self.logN) + 1, np.arange(self.logN, 0, -1)]).T
        self.ngrid = 2**np.array([np.arange(self.logN - 1) + 1, np.arange(self.logN - 1, 0, -1)]).T
        self.ph = L / self.pgrid
        self.nh = L / self.ngrid
        Shat = np.fft.fftshift(specKernel.specKernel(np.array([N, N]).T, L))
        for i in range(self.logN):
            pShat = Shat[int((N-self.pgrid[i,0])/2):int((N+self.pgrid[i,0])/2),:][:,int((N-self.pgrid[i,1])/2):int((N+self.pgrid[i,1])/2)]
            pShat = np.fft.ifftshift(pShat)
            self.pShats.append(pShat)
        for i in range(self.logN - 1):
            nShat = Shat[int((N-self.ngrid[i,0])/2):int((N+self.ngrid[i,0])/2),:][:,int((N-self.ngrid[i,1])/2):int((N+self.ngrid[i,1])/2)]
            nShat = np.fft.ifftshift(nShat)
            self.nShats.append(nShat)

    def bilinear(M, size):
        M = np.array(M)
        size = int(size)
        shape = np.array(M.shape).astype(int)
        MM = np.zeros([size, shape[1]])
        for i in range(shape[1]):
            MM[:,i] = np.interp(np.linspace(0, 1, size, endpoint=False)+1/(2*size), np.linspace(0, 1, shape[0], endpoint=False)+1/(2*shape[0]), M[:,i], period=1).T
        MMM = np.zeros([size, size])
        for i in range(size):
            MMM[i,:] = np.interp(np.linspace(0, 1, size, endpoint=False)+1/(2*size), np.linspace(0, 1, shape[1], endpoint=False)+1/(2*shape[0]), MM[i,:], period=1)
        return MMM

