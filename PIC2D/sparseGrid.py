import numpy as np
import specKernel
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